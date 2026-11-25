# Credit Risk Model Drift Monitoring

> Train a probability-of-default model on the earliest years of real lending history, replay it through decades of data drift, and compare retraining strategies — never, scheduled, and drift-triggered.

## What This Project Does

A GBM baseline trained on SBA 7(a) loans from FY1991–1995 is scored against every subsequent year through FY2018. At each step, Population Stability Index (PSI), Kolmogorov–Smirnov, and chi-square tests measure how far the new data has drifted from the reference. Alerts classify each period as green, amber, or red.

Three strategies are then compared:

**SBA 7(a)**

| Strategy | Rule | Retrains | Mean Monitoring AUC | Final Eval AUC |
|----------|------|----------|---------------------|----------------|
| Never retrain | Keep the original model | 0 | 0.7513 | 0.7441 |
| Scheduled | Retrain every year | 13 | 0.8246 | 0.9251 |
| Drift-triggered | Retrain only on red alert | 13 | 0.8246 | 0.9251 |

Scheduled and drift-triggered produce identical results here because all 23 monitoring periods have red drift alerts — interest rate data availability changed structurally from FY2000, and the 2007–2009 financial crisis reshaped default patterns. The honest finding: when drift is pervasive, a fixed schedule is simpler and equally effective.

**Freddie Mac SFLLD**

| Strategy | Rule | Retrains | Mean Monitoring AUC | Final Eval AUC |
|----------|------|----------|---------------------|----------------|
| Never retrain | Keep the original model | 0 | 0.7866 | 0.7532 |
| Scheduled | Retrain every year | 9 | 0.7970 | 0.7836 |
| Drift-triggered | Retrain only on red alert | 9 | 0.7970 | 0.7836 |

Freddie Mac shows a more nuanced drift story: 1 green, 3 amber, 13 red alerts across 17 monitoring years. Retraining provides a steady 3pp AUC improvement. Scheduled and drift-triggered again produce identical results because the majority of periods are red.

## Key Results

### SBA 7(a)
- **Baseline AUC degrades from 0.944 to 0.60** without retraining (FY1996 → FY2010)
- **Retraining recovers AUC to 0.95+** once crisis-era data enters the training window
- **Label delay matters**: with SBA's 5-year outcome window, the first retrain is possible at FY2000 — the model must endure 4 years of drift before labels allow a refresh
- **13 of 19 challengers were promoted**; 6 were rejected because they didn't beat the champion by the 0.005 AUC threshold
- **Final evaluation (FY2019–2021)**: retrained model achieves 0.92 mean AUC vs 0.74 for the stale baseline

### Freddie Mac SFLLD
- **1.25M sample mortgages** across 25 vintage years (1999–2023)
- **Nuanced drift trajectory**: 1 green, 3 amber, 13 red alerts (unlike SBA's all-red)
- **Financial crisis visible**: default rates spike to 6.2% in 2007, 4.6% in 2019
- **Shorter label delay** (24 months vs 5 years) enables faster retraining response
- **9 of 16 challengers promoted**; 7 rejected
- **Final evaluation (2021–2023)**: retrained model achieves 0.784 AUC vs 0.753 for stale baseline

## Project Structure

```
credit-risk-drift-monitoring/
├── drift/                    # Core library
│   ├── config.py             # All settings, thresholds, paths
│   ├── features.py           # Feature engineering (17 features)
│   ├── pipeline.py           # LR and GBM training pipelines
│   ├── drift.py              # PSI, KS, chi-square detection
│   ├── replay.py             # Drift replay orchestrator
│   ├── retrain.py            # Champion/challenger retraining
│   ├── label_delay.py        # No-look-ahead enforcement
│   ├── splits.py             # Time-ordered train/monitor/eval splits
│   ├── metrics.py            # AUC, calibration, cost-sensitive thresholds
│   ├── charts.py             # Plotly chart style and helpers
│   ├── sba/                  # SBA 7(a) loader and labels
│   ├── freddie/              # Freddie Mac loader and labels
│   └── uci/                  # UCI Credit Card loader
├── streamlit_app.py          # Interactive dashboard
├── reports/                  # Saved results (JSON, <25MB each)
├── tests/                    # pytest suite
├── docs/decisions.md         # Every design decision with rationale
├── WALKTHROUGH.md            # Plain-English explanations
└── requirements.txt          # Pinned dependencies
```

## Quick Start

```bash
# 1. Clone and set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# 2. Place SBA data in data/sba/
# Download from https://data.sba.gov/dataset/7a-504-foia
# Files: foia-7afy1991-fy2024-asof-260930.csv (or similar)
```

```bash
# 3. Run the pipeline (Stages 0–4)
# See WALKTHROUGH.md for step-by-step instructions
```

```bash
# 4. Launch the dashboard
streamlit run streamlit_app.py
```

## Data Setup

### SBA 7(a) FOIA (primary dataset)

1. Download from [data.sba.gov](https://data.sba.gov/dataset/7a-504-foia)
2. Place CSV files in `data/sba/`
3. The loader auto-detects files matching `foia-7a*.csv`

### Freddie Mac (optional — requires registration)

1. Register at [freddiemac.com](https://freddiemac.com/research/datasets/sf-loanlevel-dataset)
2. Download sample files (`sample_YYYY.zip`) into `data/freddie/`
3. Terms: internal use only, no redistribution

### UCI Credit Card (auto-downloaded)

Downloaded automatically on first run. Used only for the drift detector benchmark.

## Design Principles

- **No look-ahead**: only data available at each point in time is used
- **Final evaluation scored once**: FY2019–2021 are never used to tune anything
- **Honest results**: seeds, thresholds, and splits are fixed; nothing is adjusted to look better
- **Raw data never committed**: only code and aggregated results are in the repository
- **Champion/challenger with bootstrap CIs**: retrained models must beat the incumbent by ≥0.005 AUC, confirmed with 200-iteration bootstrap confidence intervals

## Technical Details

| Setting | Value |
|---------|-------|
| Python | 3.14 |
| Seed | 42 |
| Models | Logistic Regression + HistGradientBoosting |
| Features | 8 numeric + 9 categorical = 17 |
| Reference period | FY1991–1995 (5 years) |
| Monitoring | FY1996–2018 (23 years) |
| Final evaluation | FY2019–2021 (3 years) |
| Rolling window | 3 years |
| Promotion threshold | 0.005 AUC |
| PSI bins | 10 (quantile-based for numerics, equal-width for scores) |
| Alert thresholds | Green <0.10, Amber 0.10–0.25, Red >0.25 |
| Cost ratio | 5:1 (missed default costs 5x a false alarm) |
| Bootstrap iterations | 200 (fast) / 1000 (full) |

## Dashboard

The Streamlit dashboard has up to seven tabs (dynamic based on available results):

1. **Overview** — KPI cards, alert timeline, default rate chart (SBA)
2. **Drift Analysis** — Score PSI over time with alert bands, feature PSI heatmap (SBA)
3. **Model Performance** — AUC over time by strategy, retrain event table (SBA)
4. **Strategy Comparison** — Summary metrics, final evaluation, AUC lift chart (SBA)
5. **Freddie Mac** — Full mortgage drift replay, strategy comparison, retrain events
6. **UCI Benchmark** — Synthetic drift benchmark on UCI credit card data
7. **About** — Methodology, design decisions, alert thresholds, all three datasets

## Tests

```bash
pytest tests/ -v
```

30+ tests cover PSI computation, KS statistics, chi-square tests, alert classification, label derivation, time-ordered splits, and label delay enforcement.

## See Also

- [WALKTHROUGH.md](WALKTHROUGH.md) — plain-English explanation for non-experts
- [docs/decisions.md](docs/decisions.md) — every design decision with rationale
