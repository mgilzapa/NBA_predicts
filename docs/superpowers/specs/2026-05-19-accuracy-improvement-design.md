# NBA Prediction Accuracy Improvement — Design Spec

**Date:** 2026-05-19  
**Goal:** Regular season model accuracy >76% (current: 72%). Separate playoff model (no accuracy target yet).  
**Approach:** Option A — Incremental Feature Engineering + Model Tuning + Playoff Model

---

## Context

- Model: XGBoost binary classifier (`home_win` target), 63 active features after leakage cleanup
- Test accuracy: 72.0% vs 59.6% home-team baseline (+12.4%)
- Training data: 12,787 rows (Oct 2016 – May 2026), 11,969 regular season + 818 playoff
- Top signal: `elo_diff` (8.3% importance), rolling win rates, player stats
- Dead features (0% importance): `h2h_winrate_diff`, `home_series_wins`, `is_playoff`

---

## Phase 1: Data Quality Audit

Expand `src/agents/data_quality_checker.py` with the following checks:

| Check | Description |
|---|---|
| Null rates by feature + season-year | Catch features that silently degrade in older/newer data |
| Duplicate game detection | Same `gameId` appearing twice in model_data.csv |
| Label distribution shift | Home win rate per season; drift from ~56% signals pipeline issue |
| Outlier detection | IQR method on: `home_elo`, `rest_days`, `last5_avg_points`, `elo_diff` |
| Date ordering sanity | No row uses a feature derived from its own game or future games |
| Feature coverage by season | Which seasons lack box score data (player stats, pace, etc.) |

Output: richer `output/data_quality_report.json` with `by_season` breakdown and `outlier_flags` section.

---

## Phase 2: Feature Engineering (`src/feature_engineering.py`)

### A. Rolling Margin of Victory (MOV)

In `team_history`, after computing `win`:
```python
team_history["mov"] = team_history["points"] - team_history["points_allowed"]
team_history["last5_mov"]  = team_history.groupby("team")["mov"].transform(
    lambda x: shifted_rolling(x, 5, 3).mean()
)
team_history["last10_mov"] = team_history.groupby("team")["mov"].transform(
    lambda x: shifted_rolling(x, 10, 5).mean()
)
```

Merge into `df` via `merge_history()` for home and away.  
Add derived: `mov_diff = home_last5_mov - away_last5_mov`.

Add to `feature_cols`:
`home_last5_mov`, `away_last5_mov`, `mov_diff`, `home_last10_mov`, `away_last10_mov`

### B. ELO with Home-Court Advantage (HCA) Feature

After ELO ratings are computed, add one derived feature:

```python
HCA = 100  # ~3-4 point NBA home advantage expressed as ELO points
df["elo_expected_home_win"] = 1 / (
    1 + 10 ** ((df["away_elo"] - df["home_elo"] - HCA) / 400)
)
```

Add to `feature_cols`: `elo_expected_home_win`

This gives XGBoost a calibrated pre-game win probability rather than making it reconstruct the sigmoid from `elo_diff`.

### C. Turnovers / Steals / Blocks Rolling Stats

In `build_team_stats()`, extend the stats passed to `team_stats_all` to include `steals`, `blocks`, `turnovers` from `team_game_stats`.

Add to rolling `metrics` list:
```python
metrics = ["PTS", "rebounds", "AST", "MIN", "player_count", "STL", "BLK", "TOV"]
```

Produces: `last5_stl`, `last5_blk`, `last5_tov` (home and away).  
Add diffs: `stl_diff_last5`, `blk_diff_last5`, `tov_diff_last5`.

Note: these features have the same null coverage as `last5_pts` — only games with box score data (~10% of total). The same min_periods logic applies.

### D. Pace-Adjusted Efficiency (approximation)

Using existing rolling data (no box scores needed):

```python
# Pace proxy: total points scored per game (fast-paced games score more)
df["home_pace_proxy"] = df["home_last5_avg_points"] + df["home_last5_avg_points_allowed"]
df["away_pace_proxy"] = df["away_last5_avg_points"] + df["away_last5_avg_points_allowed"]

# Offensive efficiency relative to game pace
df["home_pts_per_pace"] = df["home_last5_avg_points"] / df["home_pace_proxy"].replace(0, np.nan)
df["away_pts_per_pace"] = df["away_last5_avg_points"] / df["away_pace_proxy"].replace(0, np.nan)
df["pts_per_pace_diff"] = df["home_pts_per_pace"] - df["away_pts_per_pace"]
```

Add to `feature_cols`: `home_pts_per_pace`, `away_pts_per_pace`, `pts_per_pace_diff`

Limitation: `pace_proxy` conflates offensive and defensive pace. True pace (per possession) would require FGA/FTA data not currently collected. This is the best approximation from available data.

### E. Feature Pruning

Add to `EXCLUDE_COLS` in all four files (`predict.py`, `auto_retrainer.py`, `calibration_wrapper.py`, `fetch_bracket.py`):
- `h2h_winrate_diff` — 0% feature importance
- `home_series_wins` — 0% feature importance  
- `is_playoff` — 0% feature importance (note: separate playoff model handles this context)

---

## Phase 3: Regular Season Model (`src/agents/auto_retrainer.py`)

**Training data filter:** Exclude `is_playoff==1` rows.  
Rationale: regular season model should not be pulled by the small (818 row) playoff distribution.

**Test set filter:** Also filter test set to `is_playoff==0` rows. The current 60-day window spans late regular season + current playoffs; the regular season accuracy target applies only to regular season games.

**Hyperparameter tuning:** Run `auto_retrainer.py --optimize --trials 100` after feature engineering.  
Evaluation: regular season games in the last 60 days (approximately March–April 2026).

**Evaluation target:** Test accuracy > 76% on regular-season-only test set (expected ~180-200 games).

---

## Phase 4: Playoff Model (new script)

**File:** `src/agents/playoff_model_trainer.py`

**Training data:** `is_playoff==1` rows from `model_data.csv` (~818 rows).  
**Train/test split:** Playoff games from 2024-05-01 onward as test (≈130 games covering 2024-25 and late 2023-24 playoffs), games before that as train.

**Conservative hyperparameters** (regularized for small dataset):
```python
XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=1.0,
    objective="binary:logistic",
    random_state=42,
)
```

**Feature set:** Same `feature_cols` minus `EXCLUDE_COLS` — no separate playoff feature list.  
**Output:** `models/best_xgb_model_playoff.pkl`  
**Report:** `output/playoff_model_report.json` with train/test accuracy and vs. home-team baseline.

### Integration Points

**`src/predict.py`:** When processing upcoming games tagged `is_playoff=1`, load and use `best_xgb_model_playoff.pkl` if it exists, falling back to the regular model.

**`src/fetch_bracket.py`:** Always use `best_xgb_model_playoff.pkl` for series win probability calculations (all bracket games are playoff games).

---

## Phase 5: Calibration Update

After both models are retrained:
- Re-run `calibration_wrapper.py` for the regular season model (existing behavior)
- Add equivalent playoff calibration in `playoff_model_trainer.py` using an `IsotonicRegression` on playoff data

---

## Files Changed

| File | Change |
|---|---|
| `src/agents/data_quality_checker.py` | Add 5 new checks: duplicates, label shift, outliers, date ordering, coverage by season |
| `src/feature_engineering.py` | Add MOV, ELO+HCA, steals/blocks/turnovers rolling, pace features |
| `src/agents/auto_retrainer.py` | Add `is_playoff==0` filter for training; add new features to EXCLUDE_COLS prune list |
| `src/agents/calibration_wrapper.py` | Add new features to EXCLUDE_COLS prune list |
| `src/predict.py` | Add new features to EXCLUDE_COLS prune list; use playoff model when `is_playoff=1` |
| `src/fetch_bracket.py` | Add new features to EXCLUDE_COLS prune list; use playoff model |
| `src/agents/playoff_model_trainer.py` | New file: trains and saves playoff-specific XGBoost |

---

## Success Criteria

- [ ] Data quality audit passes with no critical issues
- [ ] feature_engineering.py generates all new features without error
- [ ] Regular season model test accuracy ≥ 76%
- [ ] Playoff model trains and saves without error
- [ ] `predict.py` uses playoff model for playoff games
- [ ] `fetch_bracket.py` uses playoff model with no shape mismatch warnings
- [ ] All existing pipeline steps run without regression (`python run_all.py` clean)
