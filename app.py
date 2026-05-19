from __future__ import annotations

import io
import warnings
from datetime import datetime
from pathlib import Path

import urllib3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─── API endpoints ─────────────────────────────────────────────────────────────
_BASE = "https://data.wa.aemo.com.au/public/public-data/datafiles"
LEGACY_URL   = _BASE + "/operational-demand/operational-demand-{year}.csv"
MODERN_URL   = _BASE + "/tt30gen/total-sent-out-generation-{year}.csv"
DPV_URL      = _BASE + "/distributed-pv/distributed-pv-{year}.csv"

FIRST_YEAR        = 2006
LEGACY_LAST_YEAR  = 2023   # operational-demand ends Sep 30 2023 (WEM reform)
MODERN_FIRST_YEAR = 2024   # tt30gen continues Jan 1 2024 onwards
MAX_YEAR          = 2026   # latest year with tt30gen data
DPV_FIRST_YEAR    = 2020   # distributed PV data starts here

DATA_DIR = Path("data")

PERTH_LAT = -31.9505
PERTH_LON = 115.8605

# ─── Path helpers ──────────────────────────────────────────────────────────────

def _demand_path(year: int) -> Path:
    if year <= LEGACY_LAST_YEAR:
        return DATA_DIR / f"operational-demand-{year}.csv"
    return DATA_DIR / f"total-sent-out-generation-{year}.csv"

def _dpv_path(year: int) -> Path:
    return DATA_DIR / f"distributed-pv-{year}.csv"

def _demand_url(year: int) -> str:
    return LEGACY_URL.format(year=year) if year <= LEGACY_LAST_YEAR else MODERN_URL.format(year=year)

# ─── Download helpers ──────────────────────────────────────────────────────────

def _fetch(url: str, path: Path) -> None:
    # verify=False: workaround for Python 3.14 SSL issue on macOS with this public endpoint
    r = requests.get(url, timeout=120, verify=False)
    r.raise_for_status()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)


def ensure_demand_cached(year: int, *, force: bool = False) -> Path:
    path = _demand_path(year)
    if path.exists() and not force:
        return path
    _fetch(_demand_url(year), path)
    return path


def ensure_dpv_cached(year: int, *, force: bool = False) -> Path:
    path = _dpv_path(year)
    if path.exists() and not force:
        return path
    _fetch(DPV_URL.format(year=year), path)
    return path

# ─── CSV parsers ───────────────────────────────────────────────────────────────

def _localise(ts: pd.Series) -> pd.Series:
    # WA trialled DST 2006-2009. Two edge cases arise:
    #   nonexistent="shift_forward" — clocks spring forward (Oct): missing interval → shift to next
    #   ambiguous="NaT"            — clocks fall back (Mar): duplicate timestamp → drop (≤1 row/year)
    return ts.dt.tz_localize("Australia/Perth", nonexistent="shift_forward", ambiguous="NaT")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_daily(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily weather from Open-Meteo (historical + forecast) for Perth.

    Returns DataFrame with columns: date, temp_max, temp_min, radiation.
    Returns empty DataFrame on any failure.
    """
    try:
        daily_vars = "temperature_2m_max,temperature_2m_min,shortwave_radiation_sum"
        frames = []

        # Historical archive
        try:
            r = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": PERTH_LAT,
                    "longitude": PERTH_LON,
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": daily_vars,
                    "timezone": "Australia/Perth",
                },
                timeout=30,
                verify=False,
            )
            r.raise_for_status()
            data = r.json()
            if "daily" in data:
                d = data["daily"]
                frames.append(pd.DataFrame({
                    "date": pd.to_datetime(d["time"]),
                    "temp_max": d["temperature_2m_max"],
                    "temp_min": d["temperature_2m_min"],
                    "radiation": d["shortwave_radiation_sum"],
                }))
        except Exception:
            pass

        # Forecast (up to 16 days ahead)
        try:
            r2 = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": PERTH_LAT,
                    "longitude": PERTH_LON,
                    "daily": daily_vars,
                    "timezone": "Australia/Perth",
                    "forecast_days": 16,
                },
                timeout=30,
                verify=False,
            )
            r2.raise_for_status()
            data2 = r2.json()
            if "daily" in data2:
                d2 = data2["daily"]
                frames.append(pd.DataFrame({
                    "date": pd.to_datetime(d2["time"]),
                    "temp_max": d2["temperature_2m_max"],
                    "temp_min": d2["temperature_2m_min"],
                    "radiation": d2["shortwave_radiation_sum"],
                }))
        except Exception:
            pass

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined["date"] = combined["date"].dt.tz_localize(None)
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Fill NaN beyond forecast coverage using 7-period seasonal repeat, then ffill/bfill
        for col in ["temp_max", "temp_min", "radiation"]:
            mask = combined[col].isna()
            if mask.any():
                filled = combined[col].copy()
                for i in combined.index[mask]:
                    lookback = i - 7
                    if lookback >= 0 and not pd.isna(filled.iloc[lookback]):
                        filled.iloc[i] = filled.iloc[lookback]
                combined[col] = filled
        combined = combined.ffill().bfill()

        return combined
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_halfhourly(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo (historical + forecast) for Perth,
    resampled to 30-minute intervals via linear interpolation.

    Returns DataFrame with columns: ts, temp, radiation.
    Returns empty DataFrame on any failure.
    """
    try:
        hourly_vars = "temperature_2m,shortwave_radiation"
        frames = []

        # Historical archive
        try:
            r = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": PERTH_LAT,
                    "longitude": PERTH_LON,
                    "start_date": start_date,
                    "end_date": end_date,
                    "hourly": hourly_vars,
                    "timezone": "Australia/Perth",
                },
                timeout=30,
                verify=False,
            )
            r.raise_for_status()
            data = r.json()
            if "hourly" in data:
                h = data["hourly"]
                frames.append(pd.DataFrame({
                    "ts": pd.to_datetime(h["time"]),
                    "temp": h["temperature_2m"],
                    "radiation": h["shortwave_radiation"],
                }))
        except Exception:
            pass

        # Forecast (up to 16 days ahead)
        try:
            r2 = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": PERTH_LAT,
                    "longitude": PERTH_LON,
                    "hourly": hourly_vars,
                    "timezone": "Australia/Perth",
                    "forecast_days": 16,
                },
                timeout=30,
                verify=False,
            )
            r2.raise_for_status()
            data2 = r2.json()
            if "hourly" in data2:
                h2 = data2["hourly"]
                frames.append(pd.DataFrame({
                    "ts": pd.to_datetime(h2["time"]),
                    "temp": h2["temperature_2m"],
                    "radiation": h2["shortwave_radiation"],
                }))
        except Exception:
            pass

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined["ts"] = combined["ts"].dt.tz_localize(None)
        combined = combined.drop_duplicates(subset=["ts"]).sort_values("ts").set_index("ts")

        # Resample to 30-min with linear interpolation
        combined_30 = combined.resample("30min").interpolate(method="linear").reset_index()

        # Fill NaN beyond forecast coverage using 336-period seasonal repeat (7 days * 48 slots)
        for col in ["temp", "radiation"]:
            mask = combined_30[col].isna()
            if mask.any():
                filled = combined_30[col].copy()
                for i in combined_30.index[mask]:
                    lookback = i - 336
                    if lookback >= 0 and not pd.isna(filled.iloc[lookback]):
                        filled.iloc[i] = filled.iloc[lookback]
                combined_30[col] = filled
        combined_30 = combined_30.ffill().bfill()

        return combined_30
    except Exception:
        return pd.DataFrame()


def _parse_demand(raw: bytes, year: int) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    ts = pd.to_datetime(df["Trading Interval"], errors="coerce")
    if year <= LEGACY_LAST_YEAR:
        mw = pd.to_numeric(df["Operational Demand (MW)"], errors="coerce")
        source = "Operational Demand"
    else:
        # 30-min MWh × 2 = average MW (validated against legacy overlap)
        mw = pd.to_numeric(df["Total Sent Out Generation (MWh)"], errors="coerce") * 2
        source = "Sent-Out Generation"
    out = pd.DataFrame({"ts": ts, "operational_demand_mw": mw})
    out = out.dropna(subset=["ts", "operational_demand_mw"]).sort_values("ts")
    out["ts"] = _localise(out["ts"])
    out["year"] = year
    out["source"] = source
    return out


def _parse_dpv(raw: bytes, year: int) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    ts = pd.to_datetime(df["Trading Interval"], errors="coerce")
    mw = pd.to_numeric(df["Estimated DPV Generation (MW)"], errors="coerce")
    out = pd.DataFrame({"ts": ts, "dpv_mw": mw})
    out = out.dropna(subset=["ts", "dpv_mw"]).sort_values("ts")
    out["ts"] = _localise(out["ts"])
    return out

# ─── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_demand_year(year: int) -> pd.DataFrame:
    return _parse_demand(_demand_path(year).read_bytes(), year)


@st.cache_data(show_spinner=False)
def load_dpv_year(year: int) -> pd.DataFrame:
    return _parse_dpv(_dpv_path(year).read_bytes(), year)


def load_uploaded(uploaded_file) -> pd.DataFrame:
    df = _parse_demand(uploaded_file.getvalue(), LEGACY_LAST_YEAR)
    df["year"] = df["ts"].dt.year.astype(int)
    return df

# ─── Resample helpers ──────────────────────────────────────────────────────────

def _resample_label(freq: str) -> str:
    return {
        "30min": "30 min (raw)", "1h": "Hourly",
        "1D": "Daily", "1W": "Weekly", "1ME": "Monthly",
    }.get(freq, freq)


def _to_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = df.set_index("ts").resample("1D", offset="0h").mean(numeric_only=True)
    return d.reset_index().assign(ts=lambda x: x["ts"].dt.tz_convert("Australia/Perth"))

# ─── Forecasting ───────────────────────────────────────────────────────────────

# Intervals per day for each supported forecast frequency
_STEPS_PER_DAY = {"1D": 1, "30min": 48}


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if not _SKLEARN_OK:
        return {}
    mae  = mean_absolute_error(actual, pred)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    nz   = actual != 0
    mape = float(np.mean(np.abs((actual[nz] - pred[nz]) / actual[nz])) * 100)
    return {"MAE (MW)": round(mae, 1), "RMSE (MW)": round(rmse, 1), "MAPE (%)": round(mape, 2)}


@st.cache_data(show_spinner=False)
def forecast_prophet(parquet_bytes: bytes, test_days: int, horizon_days: int,
                     freq: str = "1D", *, weather_parquet: bytes | None = None):
    from prophet import Prophet

    df = pd.read_parquet(io.BytesIO(parquet_bytes))
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    prop = df.rename(columns={"ts": "ds", "operational_demand_mw": "y"})[["ds", "y"]].dropna()

    cutoff = prop["ds"].max() - pd.Timedelta(days=test_days)
    train, test = prop[prop["ds"] <= cutoff].copy(), prop[prop["ds"] > cutoff].copy()

    # Determine weather regressors to use
    weather_cols: list[str] = []
    weather_df: pd.DataFrame | None = None
    if weather_parquet is not None:
        try:
            wdf = pd.read_parquet(io.BytesIO(weather_parquet))
            if freq == "1D":
                weather_cols = ["temp_max", "temp_min", "radiation"]
                if all(c in wdf.columns for c in weather_cols):
                    wdf["date"] = pd.to_datetime(wdf["date"]).dt.normalize()
                    weather_df = wdf[["date"] + weather_cols].copy()
                    # Join weather onto train via date
                    train["_date"] = train["ds"].dt.normalize()
                    train = train.merge(weather_df.rename(columns={"date": "_date"}),
                                       on="_date", how="left").drop(columns=["_date"])
                    train[weather_cols] = train[weather_cols].ffill().bfill()
                    # Check any col all-NaN → skip weather
                    if any(train[c].isna().all() for c in weather_cols):
                        weather_cols = []
                        weather_df = None
                        for c in ["temp_max", "temp_min", "radiation"]:
                            if c in train.columns:
                                train = train.drop(columns=[c])
            else:
                weather_cols = ["temp", "radiation"]
                if all(c in wdf.columns for c in weather_cols):
                    wdf["ts"] = pd.to_datetime(wdf["ts"]).dt.tz_localize(None)
                    weather_df = wdf[["ts"] + weather_cols].copy()
                    train = train.merge(weather_df.rename(columns={"ts": "ds"}),
                                       on="ds", how="left")
                    train[weather_cols] = train[weather_cols].ffill().bfill()
                    if any(train[c].isna().all() for c in weather_cols):
                        weather_cols = []
                        weather_df = None
                        for c in ["temp", "radiation"]:
                            if c in train.columns:
                                train = train.drop(columns=[c])
        except Exception:
            weather_cols = []
            weather_df = None

    is_subday = freq != "1D"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(
            yearly_seasonality=not is_subday,
            weekly_seasonality=True,
            daily_seasonality=is_subday,
            interval_width=0.95,
            changepoint_prior_scale=0.05,
        )
        for col in weather_cols:
            m.add_regressor(col)
        m.fit(train)

    steps = (test_days + horizon_days) * _STEPS_PER_DAY[freq]
    future = m.make_future_dataframe(periods=steps, freq=freq)

    # Merge weather onto future dataframe
    if weather_cols and weather_df is not None:
        try:
            if freq == "1D":
                future["_date"] = future["ds"].dt.normalize()
                future = future.merge(weather_df.rename(columns={"date": "_date"}),
                                      on="_date", how="left").drop(columns=["_date"])
            else:
                future = future.merge(weather_df.rename(columns={"ts": "ds"}),
                                      on="ds", how="left")
            future[weather_cols] = future[weather_cols].ffill().bfill()
            # If still NaN after fill (e.g. beyond coverage), drop weather cols
            if any(future[c].isna().any() for c in weather_cols):
                future[weather_cols] = future[weather_cols].ffill().bfill()
        except Exception:
            # Weather merge failed — drop cols so Prophet doesn't crash
            for col in weather_cols:
                if col in future.columns:
                    future = future.drop(columns=[col])
            weather_cols = []

    fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    merged = test.set_index("ds").join(
        fc[fc["ds"].isin(test["ds"])].set_index("ds"), how="inner"
    )
    metrics = _metrics(merged["y"].values, merged["yhat"].values) if not merged.empty else {}
    return train, test, fc, metrics


@st.cache_data(show_spinner=False)
def forecast_ets(parquet_bytes: bytes, test_days: int, horizon_days: int, freq: str = "1D"):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    df = pd.read_parquet(io.BytesIO(parquet_bytes))
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df = df.dropna(subset=["operational_demand_mw"])

    # At 30-min: S=48 (daily cycle). Cap training to last 365d to keep fit fast.
    if freq != "1D":
        season = 48
        cap = df["ts"].max() - pd.Timedelta(days=test_days + 365)
        df = df[df["ts"] >= cap].copy()
    else:
        season = 7

    cutoff = df["ts"].max() - pd.Timedelta(days=test_days)
    train = df[df["ts"] <= cutoff].copy()
    test  = df[df["ts"] > cutoff].copy()

    series = train.set_index("ts")["operational_demand_mw"].asfreq(freq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            series, trend="add", seasonal="add", seasonal_periods=season,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=False)

    steps     = (test_days + horizon_days) * _STEPS_PER_DAY[freq]
    step_td   = pd.Timedelta(days=1) if freq == "1D" else pd.Timedelta(minutes=30)
    pred      = model.forecast(steps)
    sim       = model.simulate(steps, repetitions=100, error="add", random_errors="bootstrap")
    fc        = pd.DataFrame({
        "ds": pd.date_range(train["ts"].max() + step_td, periods=steps, freq=freq),
        "yhat": pred.values,
        "yhat_lower": sim.quantile(0.025, axis=1).values,
        "yhat_upper": sim.quantile(0.975, axis=1).values,
    })

    merged = test.set_index("ts").rename_axis("ds").join(
        fc[fc["ds"].isin(test["ts"])].set_index("ds"), how="inner"
    )
    merged.columns = [c.replace("operational_demand_mw", "y") for c in merged.columns]
    metrics   = _metrics(merged["y"].values, merged["yhat"].values) if not merged.empty else {}
    train_out = train.rename(columns={"ts": "ds", "operational_demand_mw": "y"})
    test_out  = test.rename(columns={"ts": "ds", "operational_demand_mw": "y"})
    return train_out, test_out, fc, metrics


@st.cache_data(show_spinner=False)
def forecast_naive(parquet_bytes: bytes, test_days: int, horizon_days: int,
                   season: int = 7, freq: str = "1D"):
    """Seasonal naive: ŷ(t+h) = y(t + h - k·S).
    At daily freq season is in days (7=weekly, 365=yearly).
    At 30-min freq season is in intervals (336=weekly, 48=daily).
    """
    df = pd.read_parquet(io.BytesIO(parquet_bytes))
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df = df.dropna(subset=["operational_demand_mw"]).sort_values("ts")

    cutoff = df["ts"].max() - pd.Timedelta(days=test_days)
    train = df[df["ts"] <= cutoff].copy()
    test  = df[df["ts"] > cutoff].copy()

    series     = train.set_index("ts")["operational_demand_mw"]
    sigma      = float((series - series.shift(season)).dropna().std())
    full       = df.set_index("ts")["operational_demand_mw"]
    step_td    = pd.Timedelta(days=1) if freq == "1D" else pd.Timedelta(minutes=30)
    season_td  = step_td * season
    steps      = (test_days + horizon_days) * _STEPS_PER_DAY[freq]
    fc_dates   = pd.date_range(train["ts"].max() + step_td, periods=steps, freq=freq)

    yhat = []
    for d in fc_dates:
        lookup = d - season_td
        while lookup not in full.index and lookup >= full.index.min():
            lookup -= season_td
        yhat.append(float(full.get(lookup, np.nan)))

    yhat   = np.array(yhat)
    h      = np.arange(1, steps + 1)
    margin = 1.96 * sigma * np.sqrt(np.ceil(h / season).astype(float))

    fc = pd.DataFrame({
        "ds": fc_dates,
        "yhat": yhat,
        "yhat_lower": yhat - margin,
        "yhat_upper": yhat + margin,
    })

    merged = test.set_index("ts").rename_axis("ds").join(
        fc[fc["ds"].isin(test["ts"])].set_index("ds"), how="inner"
    )
    merged.columns = [c.replace("operational_demand_mw", "y") for c in merged.columns]
    metrics   = _metrics(merged["y"].values, merged["yhat"].values) if not merged.empty else {}
    train_out = train.rename(columns={"ts": "ds", "operational_demand_mw": "y"})
    test_out  = test.rename(columns={"ts": "ds", "operational_demand_mw": "y"})
    return train_out, test_out, fc, metrics


@st.cache_data(show_spinner=False)
def forecast_sarima(
    parquet_bytes: bytes,
    test_days: int,
    horizon_days: int,
    p: int = 1, d: int = 1, q: int = 1,
    P: int = 1, D: int = 1, Q: int = 1,
    s: int = 7,
    freq: str = "1D",
    *,
    weather_parquet: bytes | None = None,
):
    """SARIMA(p,d,q)(P,D,Q)[s] via statsmodels SARIMAX.
    At 30-min freq s defaults to 48 (daily); training capped to last 180d.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    df = pd.read_parquet(io.BytesIO(parquet_bytes))
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df = df.dropna(subset=["operational_demand_mw"])

    # At 30-min: cap training to last 180d to keep MLE tractable
    if freq != "1D":
        cap = df["ts"].max() - pd.Timedelta(days=test_days + 180)
        df = df[df["ts"] >= cap].copy()

    cutoff = df["ts"].max() - pd.Timedelta(days=test_days)
    train = df[df["ts"] <= cutoff].copy()
    test  = df[df["ts"] > cutoff].copy()

    series = train.set_index("ts")["operational_demand_mw"].asfreq(freq)
    steps   = (test_days + horizon_days) * _STEPS_PER_DAY[freq]
    step_td = pd.Timedelta(days=1) if freq == "1D" else pd.Timedelta(minutes=30)

    # Build exogenous arrays from weather if provided
    train_exog: np.ndarray | None = None
    forecast_exog: np.ndarray | None = None
    if weather_parquet is not None:
        try:
            wdf = pd.read_parquet(io.BytesIO(weather_parquet))
            if freq == "1D":
                weather_cols = ["temp_max", "temp_min", "radiation"]
                if all(c in wdf.columns for c in weather_cols):
                    wdf["date"] = pd.to_datetime(wdf["date"]).dt.normalize()
                    wdf = wdf.set_index("date")
                    # Align train
                    train_idx = series.index.normalize()
                    exog_train = wdf.reindex(train_idx)[weather_cols].ffill().bfill()
                    # Forecast index
                    fc_idx = pd.date_range(train["ts"].max() + step_td, periods=steps, freq=freq)
                    exog_fc = wdf.reindex(fc_idx.normalize())[weather_cols].ffill().bfill()
                    if not exog_train.isna().all(axis=None) and not exog_fc.isna().all(axis=None):
                        train_exog = exog_train.values
                        forecast_exog = exog_fc.values
            else:
                weather_cols = ["temp", "radiation"]
                if all(c in wdf.columns for c in weather_cols):
                    wdf["ts"] = pd.to_datetime(wdf["ts"]).dt.tz_localize(None)
                    wdf = wdf.set_index("ts")
                    exog_train = wdf.reindex(series.index)[weather_cols].ffill().bfill()
                    fc_idx = pd.date_range(train["ts"].max() + step_td, periods=steps, freq=freq)
                    exog_fc = wdf.reindex(fc_idx)[weather_cols].ffill().bfill()
                    if not exog_train.isna().all(axis=None) and not exog_fc.isna().all(axis=None):
                        train_exog = exog_train.values
                        forecast_exog = exog_fc.values
        except Exception:
            train_exog = None
            forecast_exog = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = SARIMAX(
                series,
                exog=train_exog,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            fcast = model.get_forecast(steps=steps, exog=forecast_exog)
        except Exception:
            # Fall back to no exog if weather causes any error
            model = SARIMAX(
                series,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            fcast = model.get_forecast(steps=steps)

    ci = fcast.conf_int(alpha=0.05)

    fc = pd.DataFrame({
        "ds": pd.date_range(train["ts"].max() + step_td, periods=steps, freq=freq),
        "yhat": fcast.predicted_mean.values,
        "yhat_lower": ci.iloc[:, 0].values,
        "yhat_upper": ci.iloc[:, 1].values,
    })

    merged = test.set_index("ts").rename_axis("ds").join(
        fc[fc["ds"].isin(test["ts"])].set_index("ds"), how="inner"
    )
    merged.columns = [c.replace("operational_demand_mw", "y") for c in merged.columns]
    metrics   = _metrics(merged["y"].values, merged["yhat"].values) if not merged.empty else {}
    train_out = train.rename(columns={"ts": "ds", "operational_demand_mw": "y"})
    test_out  = test.rename(columns={"ts": "ds", "operational_demand_mw": "y"})
    return train_out, test_out, fc, metrics

# ─── Chart helpers ─────────────────────────────────────────────────────────────

def _forecast_chart(train, test, fc, test_days, horizon_days, model_name,
                    hist_rows: int = 365) -> go.Figure:
    fig = go.Figure()
    train_plot = train.tail(hist_rows)
    fig.add_trace(go.Scatter(
        x=train_plot["ds"], y=train_plot["y"],
        name="Historical", line=dict(color="#4C78A8", width=1.5),
    ))
    if not test.empty:
        fig.add_trace(go.Scatter(
            x=test["ds"], y=test["y"],
            name="Actual (test)", line=dict(color="#2ca02c", width=2),
        ))
    cutoff = test["ds"].max() if not test.empty else train["ds"].max()
    fc_test   = fc[fc["ds"] <= cutoff]
    fc_future = fc[fc["ds"] > cutoff]
    if not fc_test.empty:
        fig.add_trace(go.Scatter(
            x=fc_test["ds"], y=fc_test["yhat"],
            name="Fitted (test)", line=dict(color="#ff7f0e", width=2, dash="dot"),
        ))
    if not fc_future.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_future["ds"], fc_future["ds"].iloc[::-1]]),
            y=pd.concat([fc_future["yhat_upper"], fc_future["yhat_lower"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(214,39,40,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI",
        ))
        fig.add_trace(go.Scatter(
            x=fc_future["ds"], y=fc_future["yhat"],
            name=f"Forecast (+{horizon_days}d)", line=dict(color="#d62728", width=2.5),
        ))
    fig.update_layout(
        title=f"{model_name} — daily demand forecast",
        xaxis_title="Date", yaxis_title="Operational Demand (MW)",
        hovermode="x unified", height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def _heatmap_fig(df: pd.DataFrame) -> go.Figure:
    loc = df.copy()
    loc["ts_l"] = loc["ts"].dt.tz_convert("Australia/Perth")
    loc["hour"] = loc["ts_l"].dt.hour
    loc["dow"]  = loc["ts_l"].dt.day_name()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = (loc.groupby(["dow","hour"])["operational_demand_mw"].mean()
               .unstack(fill_value=np.nan)
               .reindex([d for d in order if d in loc["dow"].unique()]))
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="RdYlGn_r", colorbar=dict(title="MW"),
    ))
    fig.update_layout(
        title="Average demand by hour and day of week",
        xaxis_title="Hour (Perth local)", height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _monthly_box_fig(df: pd.DataFrame) -> go.Figure:
    loc = df.copy()
    names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    loc["month_name"] = loc["ts"].dt.tz_convert("Australia/Perth").dt.month.map(names)
    fig = px.box(
        loc, x="month_name", y="operational_demand_mw",
        category_orders={"month_name": list(names.values())},
        labels={"month_name":"Month","operational_demand_mw":"Demand (MW)"},
        title="Demand distribution by month",
    )
    fig.update_layout(height=400, margin=dict(l=10,r=10,t=50,b=10))
    return fig

# ─── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar_and_load() -> pd.DataFrame | None:
    with st.sidebar:
        st.header("Data source")
        mode = st.radio("Source", ["WEM API (auto)", "Upload CSV"], index=0)

        if mode == "WEM API (auto)":
            all_years = list(range(FIRST_YEAR, MAX_YEAR + 1))
            # Default: last 4 years that have local cache or recent years
            cached_years = [y for y in all_years if _demand_path(y).exists()]
            default_years = cached_years if cached_years else list(range(MAX_YEAR - 3, MAX_YEAR + 1))

            years = st.multiselect(
                "Years to load",
                options=all_years,
                default=default_years,
            )
            force = st.toggle("Force re-download", value=False)

            col1, col2 = st.columns(2)
            with col1:
                fetch_btn = st.button("Fetch data", type="primary", use_container_width=True)
            with col2:
                auto_btn  = st.button("Auto (recent 3y)", use_container_width=True)

            if auto_btn:
                auto_years = list(range(MAX_YEAR - 2, MAX_YEAR + 1))
                with st.spinner("Downloading recent years…"):
                    for y in auto_years:
                        try:
                            ensure_demand_cached(y, force=True)
                            # also grab DPV for recent years
                            if y >= DPV_FIRST_YEAR:
                                try:
                                    ensure_dpv_cached(y, force=True)
                                except Exception:
                                    pass
                        except Exception as exc:
                            st.error(f"{y}: {exc}")
                st.success(f"Downloaded {auto_years[0]}–{auto_years[-1]}.")
                st.rerun()

            if fetch_btn:
                if not years:
                    st.warning("Select at least one year.")
                else:
                    with st.spinner("Downloading…"):
                        for y in sorted(years):
                            try:
                                ensure_demand_cached(y, force=force)
                                if y >= DPV_FIRST_YEAR:
                                    try:
                                        ensure_dpv_cached(y, force=force)
                                    except Exception:
                                        pass
                            except Exception as exc:
                                st.error(f"{y}: {exc}")
                    st.success("Done.")
                    st.rerun()

            # Auto-download current year on first load if not cached
            if MAX_YEAR not in [y for y in all_years if _demand_path(y).exists()]:
                with st.spinner(f"Auto-fetching {MAX_YEAR} data…"):
                    try:
                        ensure_demand_cached(MAX_YEAR)
                        ensure_dpv_cached(MAX_YEAR)
                    except Exception:
                        pass
                st.rerun()

            ready = [y for y in (years or default_years) if _demand_path(y).exists()]
            if not ready:
                st.info("Click **Fetch data** or **Auto (recent 3y)** to download.")
                _sidebar_info()
                return None

            with st.spinner("Loading data…"):
                dfs = [load_demand_year(y) for y in sorted(ready)]
            df = pd.concat(dfs, ignore_index=True)

            # Data freshness
            last_ts = df["ts"].dt.tz_convert("Australia/Perth").max()
            st.caption(f"Latest data: **{last_ts.strftime('%d %b %Y')}**")
            st.caption(f"Loaded {len(ready)} year(s): {ready[0]}–{ready[-1]}")
            _sidebar_source_note()
            return df

        else:
            uploaded = st.file_uploader("Upload operational-demand CSV", type=["csv"])
            if uploaded is None:
                st.info("Upload a CSV to begin.")
                return None
            return load_uploaded(uploaded)


def _sidebar_info():
    st.divider()
    st.caption("**Data sources:**")
    st.caption("• 2006–2023: AEMO WEM Operational Demand (MW)")
    st.caption("• 2024–2026: Total Sent-Out Generation × 2 (MW)")
    st.caption("• Distributed PV loaded alongside demand")


def _sidebar_source_note():
    st.divider()
    with st.expander("ℹ️ About the data"):
        st.caption(
            "**2006–Sep 2023**: Official *Operational Demand (MW)* — "
            "pre-reform WEM.\n\n"
            "**2024–2026**: *Total Sent-Out Generation (MWh × 2 = MW)* — "
            "post-reform WEM. Values are ~99.4% comparable to the legacy metric.\n\n"
            "**Oct–Dec 2023 gap**: Q4 2023 is unavailable due to the WEM market "
            "reform transition (Oct 1 2023)."
        )

# ─── Tab: Historical ──────────────────────────────────────────────────────────

def _tab_historical(df: pd.DataFrame) -> None:
    local_dates = df["ts"].dt.tz_convert("Australia/Perth").dt.date
    min_d, max_d = local_dates.min(), local_dates.max()

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        start = st.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d, key="h_start")
    with c2:
        end   = st.date_input("End date",   value=max_d, min_value=min_d, max_value=max_d, key="h_end")
    with c3:
        freq  = st.selectbox("Resample", ["30min","1h","1D","1W","1ME"],
                             index=2, format_func=_resample_label, key="h_freq")

    if start > end:
        st.error("Start must be ≤ end date.")
        return

    mask = (local_dates >= start) & (local_dates <= end)
    dff  = (df.loc[mask, ["ts","operational_demand_mw"]].copy()
              .sort_values("ts").set_index("ts"))
    if freq != "30min":
        dff = dff.resample(freq).mean(numeric_only=True)
    dff = dff.reset_index()

    fig = px.line(dff, x="ts", y="operational_demand_mw",
                  title="Operational Demand (MW)",
                  labels={"ts":"Time (Australia/Perth)","operational_demand_mw":"MW"})
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=520, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data preview & download"):
        st.dataframe(dff.tail(500), use_container_width=True)
        st.download_button("Download filtered CSV",
                           dff.to_csv(index=False).encode(),
                           "aemo_filtered.csv", "text/csv")

# ─── Tab: Recent Data ─────────────────────────────────────────────────────────

def _tab_recent(df: pd.DataFrame) -> None:
    st.subheader("Recent Demand & Distributed PV")
    st.caption(
        "Shows the most recent intervals from the WEM API. "
        "Distributed PV (rooftop solar) is updated daily — "
        "**Underlying demand** = Grid demand + DPV generation."
    )

    # Slider: how many days back to show
    days_back = st.slider("Show last N days", 14, 180, 60, key="rec_days")

    local_ts = df["ts"].dt.tz_convert("Australia/Perth")
    cutoff   = local_ts.max() - pd.Timedelta(days=days_back)
    mask     = local_ts >= cutoff
    recent   = df.loc[mask].copy()

    if recent.empty:
        st.warning("No data in selected window.")
        return

    # Try to load DPV for the relevant years
    dpv_years = sorted({y for y in recent["year"].unique() if y >= DPV_FIRST_YEAR})
    dpv_frames = []
    for y in dpv_years:
        if _dpv_path(y).exists():
            dpv_frames.append(load_dpv_year(y))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["ts"], y=recent["operational_demand_mw"],
        name="Grid Demand (MW)",
        line=dict(color="#4C78A8", width=1.5),
    ))

    if dpv_frames:
        dpv = pd.concat(dpv_frames, ignore_index=True)
        dpv_local = dpv["ts"].dt.tz_convert("Australia/Perth")
        dpv_recent = dpv.loc[dpv_local >= cutoff].copy()

        if not dpv_recent.empty:
            fig.add_trace(go.Scatter(
                x=dpv_recent["ts"], y=dpv_recent["dpv_mw"],
                name="Distributed PV (MW)",
                line=dict(color="#f4a261", width=1.5),
                fill="tozeroy", fillcolor="rgba(244,162,97,0.10)",
            ))

            # Underlying demand = grid + DPV (merge on nearest 30-min timestamp)
            merged_dpv = (recent.set_index("ts")[["operational_demand_mw"]]
                          .join(dpv_recent.set_index("ts")[["dpv_mw"]], how="inner"))
            if not merged_dpv.empty:
                merged_dpv["underlying_mw"] = (merged_dpv["operational_demand_mw"]
                                               + merged_dpv["dpv_mw"])
                fig.add_trace(go.Scatter(
                    x=merged_dpv.index, y=merged_dpv["underlying_mw"],
                    name="Underlying Demand (MW)",
                    line=dict(color="#e63946", width=2, dash="dot"),
                ))

    last_demand = recent["ts"].max().tz_convert("Australia/Perth")
    last_dpv    = (dpv_recent["ts"].max().tz_convert("Australia/Perth")
                   if dpv_frames and not dpv_recent.empty else None)

    fig.update_layout(
        title="Recent WEM Demand",
        xaxis_title="Time (Australia/Perth)",
        yaxis_title="MW",
        hovermode="x unified",
        height=480,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Freshness info
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Latest grid demand", last_demand.strftime("%d %b %Y"))
    fc2.metric("Latest DPV data", last_dpv.strftime("%d %b %Y") if last_dpv else "N/A")
    fc3.metric("Intervals in view", f"{len(recent):,}")

    if not dpv_frames:
        st.info(
            "Distributed PV data not yet downloaded for these years. "
            "Click **Fetch data** in the sidebar to include it."
        )

# ─── Tab: Forecast ────────────────────────────────────────────────────────────

_MODEL_OPTIONS = ["Seasonal Naive", "Holt-Winters (ETS)", "SARIMA", "Prophet"]


def _tab_forecast(df: pd.DataFrame) -> None:
    st.subheader("Demand Forecast")

    # ── Resolution ──────────────────────────────────────────────────────────
    res_col, _, _ = st.columns([2, 2, 2])
    with res_col:
        freq = st.radio(
            "Resolution",
            ["1D", "30min"],
            format_func=lambda f: "Daily (averaged)" if f == "1D" else "30-minute (raw intervals)",
            horizontal=True,
            key="fc_freq",
        )

    spd = _STEPS_PER_DAY[freq]  # intervals per day

    if freq == "1D":
        st.caption("Trains on **daily-averaged** demand.")
    else:
        st.caption(
            "Trains on **raw 30-minute intervals**. "
            "ETS uses S=48 (daily cycle, last 365d of data). "
            "SARIMA uses s=48 (last 180d). "
            "Naive uses S=336 (same slot last week). "
            "Prophet captures both daily and weekly seasonality."
        )

    # ── Model + sliders ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("Model", _MODEL_OPTIONS, key="fc_model")
    with c2:
        test_days = st.slider("Test window (days)", 7, 90, 30, key="fc_test")
    with c3:
        horizon = st.slider("Forecast horizon (days)", 7, 90 if freq == "30min" else 180, 14 if freq == "30min" else 60, key="fc_horizon")

    # ── Model-specific controls ──────────────────────────────────────────────
    sarima_params: dict = {}
    if model_name == "SARIMA":
        default_s = 48 if freq == "30min" else 7
        label = f"SARIMA order  (p,d,q)(P,D,Q)[s]  —  s default {default_s} ({'daily' if freq == '30min' else 'weekly'})"
        with st.expander(label, expanded=True):
            st.caption(
                "Non-seasonal **(p,d,q)** and seasonal **(P,D,Q)** orders. "
                f"Seasonal period **s** defaults to {default_s} for {freq} data — override if needed."
            )
            a1, a2, a3, a4, a5, a6, a7 = st.columns(7)
            sarima_params = {
                "p": a1.number_input("p", 0, 3, 1, key="fc_p"),
                "d": a2.number_input("d", 0, 2, 1, key="fc_d"),
                "q": a3.number_input("q", 0, 3, 1, key="fc_q"),
                "P": a4.number_input("P", 0, 2, 1, key="fc_P"),
                "D": a5.number_input("D", 0, 2, 1, key="fc_D"),
                "Q": a6.number_input("Q", 0, 2, 1, key="fc_Q"),
                "s": a7.number_input("s", 1, 336, default_s, key="fc_s"),
            }

    naive_season = 336 if freq == "30min" else 7
    if model_name == "Seasonal Naive":
        if freq == "30min":
            naive_season = st.radio(
                "Seasonal period",
                [336, 48],
                format_func=lambda s: "Weekly (S=336, same slot last week)" if s == 336 else "Daily (S=48, same slot yesterday)",
                horizontal=True,
                key="fc_naive_season",
            )
        else:
            naive_season = st.radio(
                "Seasonal period",
                [7, 365],
                format_func=lambda s: f"Weekly (S={s})" if s == 7 else f"Yearly (S={s}, needs ≥2y data)",
                horizontal=True,
                key="fc_naive_season",
            )

    # ── Weather regressors toggle (Prophet / SARIMA only) ────────────────────
    use_weather = False
    if model_name in ("Prophet", "SARIMA"):
        use_weather = st.checkbox("🌡️ Weather regressors (Open-Meteo)", value=False,
                                  key="fc_weather")

    run_btn = st.button("Run forecast", type="primary", key="fc_run")

    if not run_btn and "fc_result" not in st.session_state:
        st.info("Configure settings above and click **Run forecast**.")
        return

    if run_btn:
        # Prepare data at the chosen resolution
        if freq == "1D":
            data = _to_daily(df)
        else:
            data = (df[["ts", "operational_demand_mw"]].copy()
                    .sort_values("ts")
                    .assign(ts=lambda x: x["ts"].dt.tz_localize(None)))

        buf = io.BytesIO()
        data[["ts", "operational_demand_mw"]].to_parquet(buf, index=False)
        parquet_bytes = buf.getvalue()

        # Fetch weather if requested
        weather_parquet: bytes | None = None
        if use_weather and model_name in ("Prophet", "SARIMA"):
            try:
                _data_ts = data["ts"].dt.tz_localize(None) if data["ts"].dt.tz is not None else data["ts"]
                w_start = _data_ts.min().strftime("%Y-%m-%d")
                w_end   = (_data_ts.max() + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d")
                with st.spinner("Fetching weather data from Open-Meteo…"):
                    if freq == "1D":
                        wdf = fetch_weather_daily(w_start, w_end)
                    else:
                        wdf = fetch_weather_halfhourly(w_start, w_end)
                if wdf.empty:
                    st.warning("Weather fetch returned no data — running without weather regressors.")
                else:
                    date_col = "date" if freq == "1D" else "ts"
                    w_min = pd.to_datetime(wdf[date_col]).min().strftime("%Y-%m-%d")
                    w_max = pd.to_datetime(wdf[date_col]).max().strftime("%Y-%m-%d")
                    st.caption(f"Weather: {w_min} → {w_max}, {len(wdf):,} rows, last fetched live")
                    w_buf = io.BytesIO()
                    wdf.to_parquet(w_buf, index=False)
                    weather_parquet = w_buf.getvalue()
            except Exception as exc:
                st.warning(f"Weather fetch failed ({exc}) — running without weather regressors.")
                weather_parquet = None

        with st.spinner(f"Running {model_name} at {freq} resolution…"):
            try:
                if model_name == "Prophet":
                    train, test, fc, metrics = forecast_prophet(
                        parquet_bytes, test_days, horizon, freq=freq,
                        weather_parquet=weather_parquet,
                    )
                elif model_name == "Holt-Winters (ETS)":
                    train, test, fc, metrics = forecast_ets(parquet_bytes, test_days, horizon, freq=freq)
                elif model_name == "Seasonal Naive":
                    train, test, fc, metrics = forecast_naive(parquet_bytes, test_days, horizon,
                                                              season=naive_season, freq=freq)
                else:  # SARIMA
                    train, test, fc, metrics = forecast_sarima(
                        parquet_bytes, test_days, horizon,
                        **{k: int(v) for k, v in sarima_params.items()},
                        freq=freq,
                        weather_parquet=weather_parquet,
                    )
                st.session_state["fc_result"] = (train, test, fc, metrics, model_name, horizon, test_days, freq, weather_parquet)
            except ImportError as exc:
                lib = "prophet" if model_name == "Prophet" else "statsmodels"
                st.error(f"{model_name} not installed. Run: `pip install {lib}`\n\n`{exc}`")
                return
            except Exception as exc:
                st.error(f"Forecast failed: {exc}")
                return

    result = st.session_state["fc_result"]
    train, test, fc, metrics, model_name, horizon, test_days = result[:7]
    saved_freq          = result[7] if len(result) > 7 else "1D"
    saved_weather_parquet = result[8] if len(result) > 8 else None
    saved_spd           = _STEPS_PER_DAY[saved_freq]

    if metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"{metrics.get('MAE (MW)', 0):,.1f} MW")
        m2.metric("RMSE", f"{metrics.get('RMSE (MW)', 0):,.1f} MW")
        m3.metric("MAPE", f"{metrics.get('MAPE (%)', 0):.2f} %")

    # Daily: show full year of history (365 rows).
    # 30-min: cap at 90 days = 4 320 rows to keep Plotly responsive.
    hist_rows = 365 if saved_freq == "1D" else 90 * 48
    st.plotly_chart(
        _forecast_chart(train, test, fc, test_days, horizon, model_name, hist_rows=hist_rows),
        use_container_width=True,
    )

    future_only = fc[fc["ds"] > (test["ds"].max() if not test.empty else train["ds"].max())].copy()
    future_only.columns = ["datetime", "forecast_mw", "lower_95_mw", "upper_95_mw"]
    st.download_button(f"Download {horizon}-day forecast CSV",
                       future_only.to_csv(index=False).encode(),
                       "aemo_forecast.csv", "text/csv")

    with st.expander("Full forecast table"):
        st.dataframe(fc.rename(columns={"ds": "datetime", "yhat": "forecast_mw",
                                        "yhat_lower": "lower_95_mw", "yhat_upper": "upper_95_mw"}),
                     use_container_width=True)

# ─── Tab: Analysis ────────────────────────────────────────────────────────────

def _tab_analysis(df: pd.DataFrame) -> None:
    st.subheader("Demand Analysis")

    local_ts = df["ts"].dt.tz_convert("Australia/Perth")
    demand   = df["operational_demand_mw"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Peak demand",   f"{demand.max():,.0f} MW")
    k2.metric("Peak at",       local_ts.iloc[demand.idxmax()].strftime("%Y-%m-%d %H:%M"))
    k3.metric("Average demand", f"{demand.mean():,.0f} MW")
    k4.metric("Min demand",    f"{demand.min():,.0f} MW")

    st.divider()
    st.plotly_chart(_heatmap_fig(df), use_container_width=True)
    st.plotly_chart(_monthly_box_fig(df), use_container_width=True)

    years_present = sorted(df["year"].unique()) if "year" in df.columns else []
    if len(years_present) >= 2:
        st.subheader("Year-over-year daily average")
        yoy = (df.assign(
            date_local=local_ts.dt.date,
            yr=local_ts.dt.year,
        ).groupby(["yr","date_local"])["operational_demand_mw"]
         .mean().reset_index())
        yoy["day_of_year"] = pd.to_datetime(yoy["date_local"]).dt.day_of_year
        fig = px.line(
            yoy, x="day_of_year", y="operational_demand_mw", color="yr",
            labels={"day_of_year":"Day of year","operational_demand_mw":"Avg demand (MW)","yr":"Year"},
            title="Daily average demand by year",
        )
        fig.update_layout(height=400, margin=dict(l=10,r=10,t=50,b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # DPV vs Grid demand overview (if available)
    dpv_years = [y for y in years_present if y >= DPV_FIRST_YEAR and _dpv_path(y).exists()]
    if dpv_years:
        st.subheader("Grid Demand vs Distributed PV")
        dpv_frames = [load_dpv_year(y) for y in dpv_years]
        dpv_all    = pd.concat(dpv_frames, ignore_index=True)

        demand_daily = _to_daily(df)
        dpv_daily    = (dpv_all.set_index("ts").resample("1D").mean(numeric_only=True)
                        .reset_index()
                        .assign(ts=lambda x: x["ts"].dt.tz_convert("Australia/Perth")))

        merged = demand_daily.merge(dpv_daily, on="ts", how="inner")
        if not merged.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=merged["ts"], y=merged["operational_demand_mw"],
                                      name="Grid Demand (MW)", line=dict(color="#4C78A8")))
            fig2.add_trace(go.Scatter(x=merged["ts"], y=merged["dpv_mw"],
                                      name="Distributed PV (MW)", line=dict(color="#f4a261"),
                                      fill="tozeroy", fillcolor="rgba(244,162,97,0.12)"))
            fig2.update_layout(
                title="Daily average: Grid demand vs rooftop solar",
                xaxis_title="Date", yaxis_title="MW",
                height=400, hovermode="x unified",
                margin=dict(l=10,r=10,t=50,b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="AEMO WEM Demand Forecasting", layout="wide",
                       page_icon="⚡")
    st.title("⚡ AEMO WEM — Operational Demand Dashboard")
    st.caption(
        "Real data from the [WEM public data portal](https://data.wa.aemo.com.au). "
        "2006–2023 operational demand + 2024–2026 total sent-out generation."
    )

    df = _sidebar_and_load()
    if df is None:
        st.stop()

    tab_hist, tab_recent, tab_fc, tab_analysis = st.tabs(
        ["Historical Data", "Recent Data", "Forecasting", "Analysis"]
    )
    with tab_hist:
        _tab_historical(df)
    with tab_recent:
        _tab_recent(df)
    with tab_fc:
        _tab_forecast(df)
    with tab_analysis:
        _tab_analysis(df)


if __name__ == "__main__":
    main()
