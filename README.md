# AEMO WEM Operational Demand Dashboard

A Streamlit dashboard for exploring and forecasting operational demand in Western Australia's Wholesale Electricity Market (WEM), pulling real data directly from AEMO's public portal.

## Features

| Tab | What it shows |
|-----|--------------|
| **Historical Data** | Full time series with date-range filter and resample controls (30 min → monthly) |
| **Recent Data** | Last N days: Grid Demand + Distributed PV (rooftop solar, updated daily) + Underlying Demand overlay |
| **Forecasting** | Prophet or Holt-Winters ETS with train/test split, 95% CI bands, and MAE/RMSE/MAPE metrics |
| **Analysis** | Peak KPIs, hour-of-day × day-of-week heatmap, monthly box plots, year-over-year comparison, DPV vs grid demand |

## Data sources

All data is fetched from [AEMO's WA public data portal](https://data.wa.aemo.com.au):

| Period | Endpoint | Unit |
|--------|----------|------|
| 2006 – Sep 2023 | `operational-demand/operational-demand-{year}.csv` | MW (direct) |
| 2024 – present | `tt30gen/total-sent-out-generation-{year}.csv` | MWh × 2 ≈ MW |
| 2020 – yesterday | `distributed-pv/distributed-pv-{year}.csv` | MW (updated daily) |

> **Note:** Oct–Dec 2023 data is unavailable — this quarter coincides with the WEM market reform (1 Oct 2023).

## Run locally

```bash
git clone <this-repo>
cd <this-repo>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py \
  --server.fileWatcherType none \
  --server.headless true
```

On first launch click **Auto (recent 3y)** in the sidebar to download the last three years of demand and DPV data, or use **Fetch data** to pick specific years.

## Requirements

```
pandas >= 2.2       plotly >= 5.20      requests >= 2.31
streamlit >= 1.35   numpy >= 1.26       scikit-learn >= 1.4
statsmodels >= 0.14 prophet >= 1.1      pyarrow >= 14.0
```

## Project structure

```
app.py           # Streamlit dashboard (data loading, charts, forecasting)
pipeline.py      # CLI script for batch-downloading annual CSVs
requirements.txt
data/            # Downloaded CSVs — git-ignored, created on first run
```
