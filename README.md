# HKJC Race Outcome Modeling: Chronological ML, Probability Calibration, and Market-Aware Strategy Research

## Overview

Play responsibly. Gamble at your own risk.

I explored whether Hong Kong Jockey Club race outcomes can be modeled from structured historical race-result data, and whether those model outputs can be translated into a useful market-facing decision framework.

The workflow begins with a custom-built scraper and a private raw-data pipeline that collect and structure race-result data into SQLite databases. From there, the public research pipeline builds a one-horse-per-race modeling table, engineers strictly historical runner, jockey, trainer, and interaction features, trains multiple machine-learning baselines, calibrates predicted probabilities, and studies how model probabilities compare with tote-market prices.

The main conclusion is not simply that machine learning can rank horses reasonably well, but also that **market information is extremely valuable**. The problem has upgraded from a model fitting problem to a problem about **decision-time data availability and execution**.

## Research Question

Can a chronological horse-racing prediction pipeline:

1. produce useful runner-level win probabilities,
2. outperform a market-free benchmark when current market information is included,
3. and identify potentially favorable betting candidates in a pari-mutuel environment?

## The Repository

This public repository focuses on the **research pipeline** rather than the full private data-collection stack.

It includes:

- data cleaning and modeling-table construction,
- leakage-aware historical feature engineering,
- chronological train / validation / test evaluation,
- baseline model comparison,
- probability calibration,
- model-vs-market comparison,
- and offline strategy research.

It does **not** include the private scraper, raw HTML archive, or full database-refresh pipeline. The private scraper can collect HKJC result pages that remain publicly available online, with historical coverage extending back to **1979-12-01**.

## Pipeline

```text
Private scraper / raw HKJC result pages
    -> cleaned SQLite database
    -> one-horse-per-race modeling table
    -> historical feature generation
    -> baseline model training (Logistic Regression, LightGBM, CatBoost)
    -> probability calibration (raw, sigmoid, isotonic, race-normalized)
    -> model-vs-market comparison
    -> offline strategy backtesting
```

## Data Coverage

At the current stage, the public research pipeline is built on a historical dataset covering races from **2020-01-01 to 2026-03-29**.

The cleaned modeling universe used for feature engineering and baseline training contains:

- **64,091** finished-runner rows,
- **5,282** modeled races,
- **4,089** distinct horses,
- **96** distinct jockeys,
- and **105** distinct trainers.

This gives the project enough scale to support meaningful chronological evaluation, feature engineering, and model comparison, while still leaving clear room for improvement in live market-data coverage and execution realism.

## Methodology

### 1) Modeling table construction

Runner-level records are built so that each row represents one horse in one race. Race-level information is joined onto runner rows, and only valid finished runners are retained for the main supervised-learning table.

Special cases such as withdrawals, non-standard result statuses, dead heats, and cancelled / abnormal pages are handled explicitly during the database-cleaning pipeline so that the main modeling table remains suitable for supervised learning while excluded rows remain traceable separately.

### 2) Historical feature engineering

Features are designed to use information available **before** each race, including:

- horse historical starts, wins, top-3 rates,
- same-course, same-distance, and same-surface history,
- jockey historical performance,
- trainer historical performance,
- horse-jockey interaction history,
- horse-trainer interaction history,
- and race context such as distance, draw, field size, and carried weight.

Two feature sets are maintained:

- **Market-free**: excludes current-race odds
- **Market-aware**: includes current-race `win_odds` and `log_win_odds`

This separation is important because it isolates the incremental predictive value of the public market signal.

### 3) Chronological evaluation

All model evaluation is performed using time-based train / validation / test splits rather than random shuffles. This makes the setup closer to a real forecasting workflow.

### 4) Models compared

The project compares several baseline model families:

- Logistic Regression
- LightGBM
- CatBoost

### 5) Probability calibration

Raw model probabilities are further calibrated using:

- raw output,
- sigmoid calibration,
- isotonic calibration,
- and race-normalized variants.

Model-method combinations are ranked using proper scoring rules such as **log loss** and **Brier score**, along with race-level ranking metrics.

## Main Results

### Raw baseline model comparison (test split)

Among the baseline model families, the strongest **uncalibrated** market-aware models were:

| Model | Log Loss | Brier Score | Top-Pick Win Rate | Winner-in-Top-3 Rate |
|---|---:|---:|---:|---:|
| CatBoost market-aware | 0.235398 | 0.065685 | 32.70% | 62.14% |
| LightGBM market-aware | 0.235666 | 0.065809 | 32.34% | 61.41% |
| Logistic Regression market-aware | 0.235710 | 0.065843 | 31.25% | 62.77% |

The first logistic baseline also showed a clear gap between market-free and market-aware variants on the test split:

| Variant | Log Loss | Brier Score | Top-Pick Win Rate | Winner-in-Top-3 Rate |
|---|---:|---:|---:|---:|
| Market-free | 0.258271 | 0.070539 | 24.91% | 51.27% |
| Market-aware | 0.235710 | 0.065843 | 31.25% | 62.77% |

**Interpretation:** current market odds contain substantial predictive value. Historical form alone retains signal, but the public market remains a very strong input.

### Best calibrated model-method combination

The strongest calibrated test result in the current workflow was:

**CatBoost market-aware + sigmoid calibration + race normalization**

- Log loss: **0.234958**
- Brier score: **0.065478**
- Top-pick win rate: **32.70%**
- Winner-in-top-3 rate: **62.14%**

Calibration improved proper-scoring performance modestly and produced cleaner race-level probability vectors, rather than dramatically changing rank accuracy.

## Strategy Research Findings

It is obvious that **predictive accuracy does not automatically imply profitability** due to varying dividends.

When betting every model top pick from the shortlisted market-aware pipelines, results remained negative on the test set:

| Strategy | ROI |
|---|---:|
| CatBoost market-aware + sigmoid_race_norm + top_pick_all | -8.41% |
| LightGBM market-aware + raw_race_norm + top_pick_all | -9.00% |
| Logistic Regression market-aware + raw_race_norm + top_pick_all | -15.06% |

However, more selective offline filters produced attractive-looking backtest pockets. Examples include:

| Strategy | Selections | Hit Rate | ROI |
|---|---:|---:|---:|
| CatBoost + sigmoid_race_norm + top_pick_ev_ge_0_05 | 20 | 40.0% | 51.5% |
| CatBoost + sigmoid_race_norm + top_pick_ev_ge_0 | 62 | 38.7% | 29.7% |
| Logistic Regression + raw_race_norm + top_pick_ev_ge_0_10 | 17 | 29.4% | 16.5% |
| LightGBM + raw_race_norm + top_pick_ev_ge_0_10 | 70 | 32.9% | 7.9% |

These results are promising as **research signals**, but...

## Core Bottleneck

HKJC uses a pari-mutuel betting system, that is, the final odds are not fully known at the exact moment the bet must be placed. A large proportion of bets can flow in during the last minute of betting, leading to fluctuation live odds. We have seen that odds carry strong signals. The strongest model and the strongest selective strategy variants both depend on odds-derived information.

Even if we can predict the final odds, the next question is if we are able to execute the strategy before the pool closes. 

## Why This Matters

The work began as a scraping + ML + backtesting problem, but the most important insight is now about:

- data timing,
- market microstructure,
- pre-off odds observability,
- and real-time execution.

I came to the realization that a quantitative research pipeline can discover that **the real constraint is data and implementation, not model complexity**.

## Current Limitations

- The strongest workflows are market-aware and therefore sensitive to decision-time odds availability.
- Offline ROI pockets rely on odds-based selection filters that may not be stable live.
- The current research pipeline is batch-oriented rather than a production real-time inference system.
- Some historical-feature timing choices are acceptable for research, but would need to be tightened further for a stricter live-trading style setup.

## Possible Next Steps

There are several realistic continuation paths:

1. **Pre-off live-odds collection**  
   Build or source a fresh stream of timestamped live odds so that decisions can be evaluated using information actually available before race start. However, HKJC does not have an archive of historical live odds, which means the data will have to be collected in real time. Someone has collected historical live odds for 2 seasons starting in 2016, but I think the data is too dated to be used in current times. Also, there are local data brokers selling such data, but I am undecided on making such an investment. 

2. **Final-odds forecasting layer**  
   Train a time series model that maps earlier market snapshots to estimated final odds, then feed those estimated odds into the market-aware research framework. Similarly, the problem boils down to sourcing the data. 

3. **Stricter ex-ante pipeline**  
   Refine feature timing so the entire workflow is closer to a true pre-race deployment setup.

4. **Real-time execution prototype**  
   Add live data ingestion, real-time feature generation, inference, candidate selection, and cutoff-aware execution logic.

## Public Repository Structure

```text
hkjc-ml-research/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── build_modeling_table.py
│   ├── build_historical_features.py
│   ├── sanity_check_features.py
│   ├── train_logistic_baseline.py
│   ├── train_lightgbm_baseline.py
│   ├── train_catboost_baseline.py
│   ├── compare_baseline_models.py
│   ├── calibrate_probabilities.py
│   ├── compare_model_vs_market.py
│   └── backtest_top_runner_strategy.py
└── results/
    ├── baseline_model_test_ranking.csv
    ├── calibrated_probability_test_ranking.csv
    ├── top_runner_strategy_summary.csv
    └── model_market_threshold_summary.csv
```

## How to Read This Repository

A simple way to navigate the project is:

1. **Modeling-table scripts**  
   Build the one-horse-per-race supervised-learning base table.

2. **Historical feature scripts**  
   Add horse, jockey, trainer, and interaction history while preserving chronological logic.

3. **Training scripts**  
   Fit Logistic Regression, LightGBM, and CatBoost baselines.

4. **Calibration and comparison scripts**  
   Improve probability quality and compare model families fairly.

5. **Model-vs-market and backtest scripts**  
   Study how calibrated model probabilities relate to tote-market prices and simple offline decision rules.

## Notes

- The private scraper and raw data pipeline are intentionally excluded.
- The public repository is meant to showcase the **research system** and the **main findings**.
- This project is presented as a quantitative research workflow, not as financial or betting advice.
