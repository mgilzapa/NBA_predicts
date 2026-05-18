"""auto_retrainer.py

Checks model_evaluation.json for performance degradation.
If triggered, retrains the XGBoost model headlessly (no input() prompts),
saves the new model, then re-runs calibration.

Trigger conditions (any one is sufficient):
  - alert_model_below_baseline is True  (model < home-team baseline)
  - 7-day accuracy exists and is below ACCURACY_FLOOR
  - --force flag passed on command line

Usage:
    python src/agents/auto_retrainer.py           # auto-check
    python src/agents/auto_retrainer.py --force   # always retrain
"""
import argparse
import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_JSON = os.path.join(BASE_DIR, "output", "model_evaluation.json")
MODEL_DATA_CSV = os.path.join(BASE_DIR, "data", "model_data.csv")
FEATURE_COLS_CSV = os.path.join(BASE_DIR, "models", "feature_cols.csv")
MODEL_PKL = os.path.join(BASE_DIR, "models", "best_xgb_model.pkl")
CALIB_SCRIPT = os.path.join(BASE_DIR, "src", "agents", "calibration_wrapper.py")
REPORT_PATH = os.path.join(BASE_DIR, "output", "retraining_report.json")

ACCURACY_FLOOR = 0.50   # retrain if 7-day accuracy drops below this
TRAIN_FROM = pd.Timestamp("2018-10-01")
TEST_DAYS = 60

EXCLUDE_COLS = [
    "home_off_rating", "away_off_rating", "off_rating_diff",
    "home_def_rating", "away_def_rating", "def_rating_diff",
    "home_net_rating", "away_net_rating", "net_rating_diff",
    "home_h2h_winrate", "away_h2h_winrate", "h2h_winrate_diff",
    "same_division", "is_playoff", "away_opponent_strength",
]


def should_retrain(force: bool) -> tuple[bool, str]:
    if force:
        return True, "--force flag gesetzt"

    if not os.path.exists(EVAL_JSON):
        return False, "model_evaluation.json nicht gefunden — kein Retraining"

    with open(EVAL_JSON) as f:
        ev = json.load(f)

    if ev.get("alert_model_below_baseline"):
        return True, "Modell schlechter als Baseline"

    last7 = ev.get("last_7_days", {})
    acc7 = last7.get("accuracy")
    games7 = last7.get("games", 0)
    if acc7 is not None and games7 >= 10 and acc7 < ACCURACY_FLOOR:
        return True, f"7-Tage-Accuracy {acc7:.2%} unter Schwelle {ACCURACY_FLOOR:.2%}"

    return False, f"Kein Trigger — Accuracy OK (7d: {acc7})"


def retrain() -> dict:
    for path, label in [(MODEL_DATA_CSV, "model_data.csv"), (FEATURE_COLS_CSV, "feature_cols.csv")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} nicht gefunden")

    df = pd.read_csv(MODEL_DATA_CSV)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")

    feature_cols = pd.read_csv(FEATURE_COLS_CSV).squeeze().tolist()
    feature_cols = [c for c in feature_cols if c in df.columns and c not in EXCLUDE_COLS]

    now = pd.Timestamp.now(tz="US/Eastern").tz_localize(None)
    test_start = now - pd.Timedelta(days=TEST_DAYS)

    df_model = df.dropna(subset=feature_cols + ["home_win"]).copy()
    df_model = df_model[
        (df_model.get("home_elo_games_played", pd.Series(999, index=df_model.index)) > 20) &
        (df_model.get("away_elo_games_played", pd.Series(999, index=df_model.index)) > 20)
    ] if "home_elo_games_played" in df_model.columns else df_model

    train = df_model[df_model["gameDateTimeEst"] < test_start].copy()
    test = df_model[
        (df_model["gameDateTimeEst"] >= test_start) &
        (df_model["gameDateTimeEst"] < now)
    ].copy()

    if len(train) < 100:
        raise ValueError(f"Zu wenige Trainingsdaten: {len(train)} Spiele")

    # Time-decay sample weights: ältere Spiele bekommen weniger Gewicht
    days_old = (test_start - train["gameDateTimeEst"]).dt.days.clip(lower=0)
    sample_weights = (1 / (1 + days_old / 365)).values

    X_train = train[feature_cols].values
    y_train = train["home_win"].values
    X_test = test[feature_cols].values
    y_test = test["home_win"].values

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test) if len(test) > 0 else None
    baseline_acc = float(test["home_win"].mean()) if len(test) > 0 else None

    joblib.dump(model, MODEL_PKL)

    return {
        "train_games": len(train),
        "test_games": len(test),
        "feature_count": len(feature_cols),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4) if test_acc is not None else None,
        "baseline_accuracy": round(baseline_acc, 4) if baseline_acc is not None else None,
        "vs_baseline": round(test_acc - baseline_acc, 4) if (test_acc and baseline_acc) else None,
    }


def run(force: bool = False):
    triggered, reason = should_retrain(force)

    print(f"\n{'='*55}")
    print(f"  AUTO RETRAINER")
    print(f"{'='*55}")
    print(f"  Trigger:  {reason}")

    if not triggered:
        print(f"  Status:   Kein Retraining notwendig.")
        print(f"{'='*55}\n")
        report = {"retrained": False, "reason": reason}
    else:
        print(f"  Status:   Starte Retraining...")
        try:
            stats = retrain()
            print(f"  Training: {stats['train_games']} Spiele, {stats['feature_count']} Features")
            if stats["test_accuracy"] is not None:
                print(f"  Test-Acc: {stats['test_accuracy']:.2%}  (Baseline: {stats['baseline_accuracy']:.2%})")
            print(f"  Modell gespeichert: {MODEL_PKL}")

            # Kalibrierung sofort neu berechnen
            if os.path.exists(CALIB_SCRIPT):
                print(f"  Starte Kalibrierung...")
                subprocess.run([sys.executable, CALIB_SCRIPT], check=False, cwd=BASE_DIR)

            report = {"retrained": True, "reason": reason, **stats}
        except Exception as exc:
            print(f"  FEHLER beim Retraining: {exc}")
            report = {"retrained": False, "reason": reason, "error": str(exc)}

    report["generated_at"] = pd.Timestamp.now(tz="US/Eastern").tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Report: {REPORT_PATH}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Immer neu trainieren")
    args = parser.parse_args()
    run(force=args.force)
