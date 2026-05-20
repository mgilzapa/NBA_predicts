"""playoff_model_trainer.py

Trains the playoff XGBoost model using Leave-One-Season-Out cross-validation.

Training data: playoff rows from data/model_data.csv (seasons 2016–2024).
Current season (2025) is excluded — used as live evaluation only.

New features beyond the base model:
  base_prob_home_win   — output of the regular-season model (transfer feature)
  series_game_number   — game 1–7 computed from series win counts
  home/away_playoff_last3_pts     — rolling scoring average in these playoffs
  home/away_playoff_margin_last3  — rolling margin in these playoffs
  home_playoff_home_winrate_hist  — historical playoff home-court advantage per team

Validation: LOSO — each season held out once as the test set.

Saves:
  models/best_xgb_model_playoff.pkl
  models/feature_cols_playoff.csv
  models/calibration_model_playoff.pkl
  output/playoff_model_report.json

Usage:
    python src/agents/playoff_model_trainer.py
    python src/agents/playoff_model_trainer.py --force
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

BASE_DIR             = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DATA_CSV       = os.path.join(BASE_DIR, "data",   "model_data.csv")
NBA_GAMES_CSV        = os.path.join(BASE_DIR, "data",   "nba_api_games.csv")
BASE_MODEL_PKL       = os.path.join(BASE_DIR, "models", "best_xgb_model.pkl")
BASE_FEATURE_CSV     = os.path.join(BASE_DIR, "models", "feature_cols.csv")
MODEL_PKL            = os.path.join(BASE_DIR, "models", "best_xgb_model_playoff.pkl")
FEATURE_COLS_CSV     = os.path.join(BASE_DIR, "models", "feature_cols_playoff.csv")
CALIB_PKL            = os.path.join(BASE_DIR, "models", "calibration_model_playoff.pkl")
REPORT_PATH          = os.path.join(BASE_DIR, "output", "playoff_model_report.json")

CURRENT_SEASON_INT = 2025   # excluded from training; current playoffs are live evaluation

BASE_EXCLUDE_COLS = {
    "home_off_rating", "away_off_rating", "off_rating_diff",
    "home_def_rating", "away_def_rating", "def_rating_diff",
    "home_net_rating", "away_net_rating", "net_rating_diff",
    "same_division", "away_opponent_strength",
    "home_injury_impact", "away_injury_impact", "injury_impact_diff",
    "home_playoff_exp", "away_playoff_exp", "playoff_exp_diff",
    "h2h_winrate_diff",
    "is_playoff",
}

# home_series_wins was excluded when the regular-season model was trained (zero importance).
# We must use the same set when running the base model to produce base_prob_home_win,
# but we KEEP home_series_wins for the playoff model features (it matters in playoffs).
BASE_MODEL_RUN_EXCLUDE = BASE_EXCLUDE_COLS | {"home_series_wins"}

XGBOOST_PARAMS = dict(
    max_depth=3,
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)


def _season_label(season_int):
    y = int(season_int)
    return f"{y}-{str(y + 1)[-2:]}"


def _round_from_game_id(game_id):
    return int(str(int(game_id))[5])


def build_playoff_rolling_features(model_df, games_df):
    """
    For each row in model_df (playoff games), compute rolling playoff stats
    for both home and away teams using games from the same season that occurred
    BEFORE the game date (no leakage).

    Returns a DataFrame indexed the same as model_df with columns:
      home_playoff_last3_pts, away_playoff_last3_pts,
      home_playoff_margin_last3, away_playoff_margin_last3
    """
    playoff_games = games_df[games_df["season_type"] == "Playoffs"].copy()
    playoff_games["GAME_DATE"] = pd.to_datetime(playoff_games["GAME_DATE"])
    playoff_games = playoff_games.sort_values("GAME_DATE")

    # Build per-(team, season) ordered game log
    from collections import defaultdict
    team_season_log = defaultdict(list)  # (team, season_label) -> [{date, pts, margin}]

    for _, row in playoff_games.iterrows():
        h_pts = float(row["homeScore"])
        a_pts = float(row["awayScore"])
        for team, pts, margin in [
            (row["hometeamName"], h_pts, h_pts - a_pts),
            (row["awayteamName"], a_pts, a_pts - h_pts),
        ]:
            team_season_log[(team, row["season"])].append({
                "date": row["GAME_DATE"], "pts": pts, "margin": margin
            })

    model_df = model_df.copy()
    model_df["gameDateTimeEst"] = pd.to_datetime(model_df["gameDateTimeEst"])

    result = []
    for idx, row in model_df.iterrows():
        game_date = row["gameDateTimeEst"]
        season_label = _season_label(row["season"])

        def rolling(team, n=3):
            log = team_season_log.get((team, season_label), [])
            prev = [g for g in log if g["date"] < game_date]
            if not prev:
                return np.nan, np.nan
            tail = prev[-n:]
            return (
                round(np.mean([g["pts"] for g in tail]), 2),
                round(np.mean([g["margin"] for g in tail]), 2),
            )

        h_pts, h_margin = rolling(row["hometeamName"])
        a_pts, a_margin = rolling(row["awayteamName"])
        result.append({
            "home_playoff_last3_pts": h_pts,
            "away_playoff_last3_pts": a_pts,
            "home_playoff_margin_last3": h_margin,
            "away_playoff_margin_last3": a_margin,
        })

    return pd.DataFrame(result, index=model_df.index)


def build_historical_winrate(games_df):
    """Per-team historical playoff home win rate (all seasons)."""
    hist = games_df[games_df["season_type"] == "Playoffs"].copy()
    agg = hist.groupby("hometeamName").agg(
        total=("home_win", "count"),
        wins=("home_win", "sum"),
    ).reset_index()
    agg["home_playoff_home_winrate_hist"] = (agg["wins"] / agg["total"]).round(4)
    return agg.set_index("hometeamName")["home_playoff_home_winrate_hist"].to_dict()


def train():
    for path, label in [
        (MODEL_DATA_CSV, "model_data.csv"),
        (NBA_GAMES_CSV, "nba_api_games.csv"),
        (BASE_MODEL_PKL, "best_xgb_model.pkl"),
        (BASE_FEATURE_CSV, "feature_cols.csv"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at {path}")
            return False

    print("\n" + "=" * 60)
    print("  PLAYOFF MODEL TRAINER — LOSO")
    print("=" * 60)

    # ── Load base model ────────────────────────────────────────────
    base_model    = joblib.load(BASE_MODEL_PKL)
    base_feat_csv = pd.read_csv(BASE_FEATURE_CSV).squeeze().tolist()
    # base_feat_run: exact feature set used to run the base model (matches training-time exclusions)
    base_feat_run = [c for c in base_feat_csv if c not in BASE_MODEL_RUN_EXCLUDE]
    # base_feat: playoff model's base-layer features (home_series_wins kept as informative for playoffs)
    base_feat     = [c for c in base_feat_csv if c not in BASE_EXCLUDE_COLS]

    # ── Load data ─────────────────────────────────────────────────
    df = pd.read_csv(MODEL_DATA_CSV)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")

    games_df = pd.read_csv(NBA_GAMES_CSV)
    hist_wr  = build_historical_winrate(games_df)

    # Filter to playoff rows, exclude current season
    df_playoff = df[(df["is_playoff"] == 1) & (df["season"] < CURRENT_SEASON_INT)].copy()

    # Drop rows with null home_win or critical base features
    _dropna = [c for c in base_feat if c in df_playoff.columns and c != "market_prob_home_win"]
    df_playoff = df_playoff.dropna(subset=_dropna + ["home_win"])

    print(f"  Historical playoff rows: {len(df_playoff)}")
    print(f"  Seasons in training:     {sorted(df_playoff['season'].unique().tolist())}")

    # ── Add base_prob_home_win ────────────────────────────────────
    valid_base_run_feat = [c for c in base_feat_run if c in df_playoff.columns]
    X_base = df_playoff[valid_base_run_feat].fillna(0).values
    df_playoff["base_prob_home_win"] = base_model.predict_proba(X_base)[:, 1]

    # ── Add series_game_number ────────────────────────────────────
    df_playoff["series_game_number"] = (
        df_playoff["home_series_wins"] + df_playoff["away_series_wins"] + 1
    ).clip(1, 7)

    # ── Add playoff rolling stats ─────────────────────────────────
    rolling_feats = build_playoff_rolling_features(df_playoff, games_df)
    df_playoff = df_playoff.join(rolling_feats)

    # ── Add historical home win rate ──────────────────────────────
    df_playoff["home_playoff_home_winrate_hist"] = (
        df_playoff["hometeamName"].map(hist_wr).fillna(0.5)
    )

    # ── Define playoff feature set ────────────────────────────────
    playoff_extra = [
        "base_prob_home_win",
        "series_game_number",
        "home_playoff_last3_pts",
        "away_playoff_last3_pts",
        "home_playoff_margin_last3",
        "away_playoff_margin_last3",
        "home_playoff_home_winrate_hist",
    ]
    # Keep base features that are present, strip excluded ones
    base_keep = [c for c in base_feat if c in df_playoff.columns]
    playoff_feat = base_keep + [c for c in playoff_extra if c not in base_keep]

    print(f"  Total features:          {len(playoff_feat)}")

    # ── LOSO cross-validation ─────────────────────────────────────
    seasons = sorted(df_playoff["season"].unique())
    oof_probs = np.zeros(len(df_playoff))
    oof_preds = np.zeros(len(df_playoff), dtype=int)
    fold_accs = []

    print(f"\n  LOSO Folds ({len(seasons)} seasons):")
    for s in seasons:
        mask_test  = df_playoff["season"] == s
        mask_train = ~mask_test

        X_train = df_playoff.loc[mask_train, playoff_feat].fillna(0).values
        y_train = df_playoff.loc[mask_train, "home_win"].values
        X_test  = df_playoff.loc[mask_test,  playoff_feat].fillna(0).values
        y_test  = df_playoff.loc[mask_test,  "home_win"].values

        if len(X_train) < 30 or len(X_test) == 0:
            print(f"    Season {s}: skipped (train={len(X_train)}, test={len(X_test)})")
            continue

        fold_model = XGBClassifier(**XGBOOST_PARAMS)
        fold_model.fit(X_train, y_train)

        probs = fold_model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        oof_probs[mask_test.values] = probs
        oof_preds[mask_test.values] = preds

        acc  = float(np.mean(preds == y_test))
        base = float(np.mean(y_test))
        fold_accs.append(acc)
        print(f"    Season {s}: acc={acc:.3f}  baseline={base:.3f}  n={len(y_test)}")

    mean_acc  = float(np.mean(fold_accs)) if fold_accs else 0.0
    base_mean = float(df_playoff["home_win"].mean())
    print(f"\n  LOSO mean accuracy:      {mean_acc:.3f}")
    print(f"  Baseline (home always):  {base_mean:.3f}")
    print(f"  vs. baseline:            {mean_acc - base_mean:+.3f}")

    # ── Final model — train on all seasons ───────────────────────
    X_all = df_playoff[playoff_feat].fillna(0).values
    y_all = df_playoff["home_win"].values

    final_model = XGBClassifier(**XGBOOST_PARAMS)
    final_model.fit(X_all, y_all)

    train_acc = float(final_model.score(X_all, y_all))
    print(f"  Final train accuracy:    {train_acc:.3f}")

    # ── Calibration on OOF predictions ───────────────────────────
    valid_oof = oof_probs > 0
    calib = IsotonicRegression(out_of_bounds="clip")
    calib.fit(oof_probs[valid_oof].reshape(-1, 1), y_all[valid_oof])

    # ── Save everything ───────────────────────────────────────────
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)

    joblib.dump(final_model, MODEL_PKL)
    joblib.dump(calib, CALIB_PKL)
    pd.Series(playoff_feat).to_csv(FEATURE_COLS_CSV, index=False)

    print(f"\n  Model saved:             {MODEL_PKL}")
    print(f"  Feature list saved:      {FEATURE_COLS_CSV}")
    print(f"  Calibration saved:       {CALIB_PKL}")

    report = {
        "generated_at":     pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "playoff_rows":     int(len(df_playoff)),
        "seasons_trained":  [int(s) for s in seasons],
        "feature_count":    len(playoff_feat),
        "loso_mean_acc":    round(mean_acc, 4),
        "baseline_acc":     round(base_mean, 4),
        "vs_baseline":      round(mean_acc - base_mean, 4),
        "fold_accuracies":  {int(s): round(a, 4) for s, a in zip(seasons, fold_accs)},
        "train_accuracy":   round(train_acc, 4),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved:            {REPORT_PATH}")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Retrain even if model exists")
    args = parser.parse_args()

    if not args.force and os.path.exists(MODEL_PKL):
        print(f"Playoff model already exists at {MODEL_PKL}. Use --force to retrain.")
        sys.exit(0)

    ok = train()
    sys.exit(0 if ok else 1)
