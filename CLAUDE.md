# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

NBA game prediction dashboard. A daily Python pipeline collects data, trains an XGBoost model, generates predictions, and writes static JSON files consumed by a plain-HTML/CSS/JS frontend deployed on Vercel.

## Running the pipeline

Activate the virtual environment first (Windows):
```
.venv\Scripts\activate
```

Run the full daily pipeline (all 12 steps in order):
```
python run_all.py
```

Run a single pipeline step:
```
python src/predict.py
python src/feature_engineering.py
# etc.
```

The pipeline requires the `ODDS_API_KEY` environment variable for `fetch_odds.py`. `run_all.py` sets a default key if the env var is absent.

Run tests:
```
python -m pytest tests/
```

Run a single test file:
```
python -m pytest tests/test_fetch_bracket.py
```

## Pipeline order and data flow

`run_all.py` executes these scripts in sequence — each step depends on output from the previous:

1. `src/nba_api_test.py` → `data/nba_api_games.csv` (raw game results from nba_api)
2. `src/scrape_upcoming_games.py` → `data/schedule_round_1.csv` (upcoming games)
3. `src/injury_reports.py` → `data/current_injuries.csv` (scrapes covers.com)
4. `src/fetch_player_stats.py` → `data/player_boxscores.csv`
5. `src/tabelle.py` → `data/base_games.csv` (cleaned game table)
6. `src/feature_engineering.py` → `data/model_data.csv` (rolling stats, ELO, player features)
7. `src/predict.py` → `output/predictions.xlsx` (today's predictions using saved model)
8. `src/clean_excel.py` → cleans predictions.xlsx
9. `src/create_excel.py` → `output/all_predictions.xlsx` (cumulative history with actuals)
10. `src/fetch_odds.py` → `web/odds.json` (the-odds-api.com, needs `ODDS_API_KEY`)
11. `src/export_json.py` → `web/predictions.json` (today/yesterday/all as JSON)
12. `src/fetch_bracket.py` → `web/bracket.json` (playoff bracket with series predictions)

The `data/` and `output/` directories are gitignored.

## ML model architecture

- **Model**: XGBoost binary classifier (`home_win` target), saved as `models/best_xgb_model.pkl`
- **Feature list**: `models/feature_cols.csv` (read at prediction time — do not hardcode feature lists)
- **Features**: rolling last-5-game stats (win rate, points, rebounds, assists), ELO ratings, rest days, back-to-back flags, home/away win rates, player availability metrics derived from injury reports
- **Training**: `src/feature_engineering.py` uses `TimeSeriesSplit` for cross-validation and saves the best model; `src/predict.py` loads the saved model and applies it to upcoming games
- `src/train.py` is a standalone analysis/comparison script (LR vs XGB), not part of the daily pipeline

## Frontend

Static files in `web/` — no build step, no framework:
- `web/index.html` + `web/style.css` + `web/app.js`
- Data is fetched from `web/predictions.json`, `web/odds.json`, `web/bracket.json`
- Four tabs: Today, Yesterday, All Predictions, Bracket
- Deployed to Vercel; `vercel.json` sets `web/` as the output directory

## GitHub Actions

`.github/workflows/fetch-odds.yml` runs `src/fetch_odds.py` every 5 hours and commits `web/odds.json` directly to main. The `ODDS_API_KEY` secret must be set in the repo.

## Monitoring agents (`src/agents/`)

Four standalone scripts that can be run after the pipeline or on demand. All write a JSON report to `output/` and exit with code 1 on failure.

| Script | What it does | Output |
|---|---|---|
| `pipeline_runner.py` | Runs all pipeline steps with per-step timing; stops on first error | `output/pipeline_report.json` |
| `data_quality_checker.py` | Null rates per feature, duplicates, staleness, value ranges | `output/data_quality_report.json` |
| `feature_drift_detector.py` | Features in `feature_cols.csv` missing from `model_data.csv`; model feature-count mismatch | `output/feature_drift_report.json` |
| `odds_feature_injector.py` | Blends model probability with no-vig market probability from `odds.json` (α=0.5) | updates `predictions.xlsx` in place |
| `model_evaluator.py` | Overall/7-day/30-day accuracy vs. baseline, confidence calibration, daily trend | `output/model_evaluation.json` |
| `auto_retrainer.py` | Retrains XGBoost + re-calibrates when accuracy drops below baseline or `--force` | `output/retraining_report.json` |
| `calibration_wrapper.py` | Fits IsotonicRegression on last 90 days, saves `models/calibration_model.pkl` | `output/calibration_report.json` |
| `prediction_auditor.py` | Sanity-checks today's predictions: high confidence, model vs. market divergence, ELO upsets, form upsets | `output/prediction_audit_report.json` |
| `dashboard_exporter.py` | Aggregates all agent reports into `web/dashboard.json` for the frontend Dashboard tab | `web/dashboard.json` |

`data_quality_checker`, `feature_drift_detector`, `odds_feature_injector`, `prediction_auditor`, `model_evaluator`, `auto_retrainer`, and `dashboard_exporter` are wired into `run_all.py`. `calibration_wrapper` is triggered automatically by `auto_retrainer` after retraining.

## Dashboard tab

The webapp has a fifth tab "Dashboard" that shows a maintenance overview:
- **Status banner**: Overall OK / Warning / Critical based on all agent reports
- **Model Performance**: Overall accuracy, 7-day, 30-day, vs. baseline, calibration table
- **System Health**: Per-component status cards (Data Quality, Feature Drift, Prediction Audit, Retraining)
- **Observations**: All flags and issues from all agents, sorted by severity

Data source: `web/dashboard.json` (written by `dashboard_exporter.py` at end of pipeline).

Run any agent directly:
```
python src/agents/model_evaluator.py
python src/agents/auto_retrainer.py --force
python src/agents/calibration_wrapper.py
python src/agents/pipeline_runner.py   # alternative to run_all.py
```

## Brand / design constraints

Defined in `PRODUCT.md` (gitignored but present locally):
- Target feel: sharp, energized, trustworthy — live scoreboard energy, not a sportsbook or sterile dashboard
- WCAG AAA accessibility required; `prefers-reduced-motion` must be respected
- All four tabs (Today, Yesterday, All Predictions, Bracket) are treated as equally important
- Fonts: Oswald + JetBrains Mono (loaded from Google Fonts)
