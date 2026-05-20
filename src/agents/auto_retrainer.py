"""auto_retrainer.py

Checks model_evaluation.json for performance degradation.
If triggered, retrains the XGBoost model headlessly (no input() prompts),
saves the new model, then re-runs calibration.

Trigger conditions (any one is sufficient):
  - alert_model_below_baseline is True  (model < home-team baseline)
  - 7-day accuracy exists and is below ACCURACY_FLOOR
  - --force flag passed on command line

Usage:
    python src/agents/auto_retrainer.py                      # auto-check
    python src/agents/auto_retrainer.py --force              # always retrain
    python src/agents/auto_retrainer.py --optimize           # Optuna search (50 trials)
    python src/agents/auto_retrainer.py --optimize --trials 100
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

ACCURACY_FLOOR  = 0.50   # retrain if 7-day accuracy drops below this
TRAIN_FROM      = pd.Timestamp("2018-10-01")
TEST_DAYS       = 60
MIN_IMPROVEMENT = 0.005  # --optimize only saves if new acc > old acc + 0.5%

EXCLUDE_COLS = [
    "home_off_rating", "away_off_rating", "off_rating_diff",
    "home_def_rating", "away_def_rating", "def_rating_diff",
    "home_net_rating", "away_net_rating", "net_rating_diff",
    "same_division", "away_opponent_strength",
    # Temporal leakage: injury data is today's snapshot applied to all historical rows
    "home_injury_impact", "away_injury_impact", "injury_impact_diff",
    # Temporal leakage: total playoff exp across all time merged statically to every row
    "home_playoff_exp", "away_playoff_exp", "playoff_exp_diff",
    # Zero importance in trained model — dead weight
    "h2h_winrate_diff", "home_series_wins", "is_playoff",
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
    feature_cols, train, test, sample_weights = _load_data()

    if len(train) < 100:
        raise ValueError(f"Zu wenige Trainingsdaten: {len(train)} Spiele")

    X_train = train[feature_cols].values
    y_train = train["home_win"].values
    X_test  = test[feature_cols].values
    y_test  = test["home_win"].values

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


def _load_data():
    """Gemeinsame Datenlade-Logik für retrain() und optimize()."""
    for path, label in [(MODEL_DATA_CSV, "model_data.csv"), (FEATURE_COLS_CSV, "feature_cols.csv")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} nicht gefunden")

    df = pd.read_csv(MODEL_DATA_CSV)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")

    feature_cols = pd.read_csv(FEATURE_COLS_CSV).squeeze().tolist()
    feature_cols = [c for c in feature_cols if c in df.columns and c not in EXCLUDE_COLS]

    now        = pd.Timestamp.now(tz="US/Eastern").tz_localize(None)
    test_start = now - pd.Timedelta(days=TEST_DAYS)

    _DROPNA_EXCLUDE = {"market_prob_home_win"}
    _dropna_cols = [c for c in feature_cols if c not in _DROPNA_EXCLUDE]
    df_model = df.dropna(subset=_dropna_cols + ["home_win"]).copy()
    if "home_elo_games_played" in df_model.columns:
        df_model = df_model[
            (df_model["home_elo_games_played"] > 20) &
            (df_model["away_elo_games_played"] > 20)
        ]
    # Regular-season model only — playoff rows go to the separate playoff model
    if "is_playoff" in df_model.columns:
        df_model = df_model[df_model["is_playoff"] == 0]

    train = df_model[df_model["gameDateTimeEst"] < test_start].copy()
    test  = df_model[
        (df_model["gameDateTimeEst"] >= test_start) &
        (df_model["gameDateTimeEst"] < now)
    ].copy()

    days_old       = (test_start - train["gameDateTimeEst"]).dt.days.clip(lower=0)
    sample_weights = (1 / (1 + days_old / 365)).values

    return feature_cols, train, test, sample_weights


def optimize(n_trials: int = 50):
    """Optuna hyperparameter search — speichert nur wenn Acc > aktuelles Modell + MIN_IMPROVEMENT."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.model_selection import TimeSeriesSplit

    feature_cols, train, test, sample_weights = _load_data()

    X_train = train[feature_cols].values
    y_train = train["home_win"].values
    X_test  = test[feature_cols].values
    y_test  = test["home_win"].values

    # Aktuelle Modell-Accuracy als Vergleichswert (None wenn Feature-Shape geändert)
    current_acc = None
    if os.path.exists(MODEL_PKL):
        current_model = joblib.load(MODEL_PKL)
        if len(test) > 0:
            try:
                current_acc = current_model.score(X_test, y_test)
            except ValueError:
                pass  # feature count changed — treat as no prior baseline

    print(f"\n{'='*55}")
    print(f"  OPTUNA HYPERPARAMETER SEARCH")
    print(f"{'='*55}")
    print(f"  Trials:          {n_trials}")
    print(f"  Trainingsdaten:  {len(train)} Spiele, {len(feature_cols)} Features")
    print(f"  Testdaten:       {len(test)} Spiele")
    if current_acc is not None:
        print(f"  Aktuelle Acc:    {current_acc:.4%}")
    print(f"  Speichern wenn:  neue Acc > {(current_acc or 0) + MIN_IMPROVEMENT:.4%}")
    print(f"{'='*55}")

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 7),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 2.0),
            "objective":         "binary:logistic",
            "eval_metric":       "logloss",
            "random_state":      42,
            "verbosity":         0,
        }
        cv_scores = []
        for fold_train_idx, fold_val_idx in tscv.split(X_train):
            Xf_train, Xf_val = X_train[fold_train_idx], X_train[fold_val_idx]
            yf_train, yf_val = y_train[fold_train_idx], y_train[fold_val_idx]
            sw_fold = sample_weights[fold_train_idx]
            m = XGBClassifier(**params)
            m.fit(Xf_train, yf_train, sample_weight=sw_fold)
            cv_scores.append(m.score(Xf_val, yf_val))
        return float(np.mean(cv_scores))

    study = optuna.create_study(direction="maximize")

    completed = [0]
    def _cb(study, trial):
        completed[0] += 1
        if completed[0] % 10 == 0:
            print(f"  Trial {completed[0]:>3}/{n_trials}  |  beste CV-Acc: {study.best_value:.4%}")

    study.optimize(objective, n_trials=n_trials, callbacks=[_cb], show_progress_bar=False)

    best_params = study.best_params
    best_cv     = study.best_value
    print(f"\n  Beste CV-Accuracy: {best_cv:.4%}")
    print(f"  Beste Parameter:   {best_params}")

    # Endmodell mit besten Params auf vollem Trainingsset trainieren
    final_model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X_train, y_train, sample_weight=sample_weights)
    new_acc      = final_model.score(X_test, y_test) if len(test) > 0 else None
    baseline_acc = float(test["home_win"].mean())     if len(test) > 0 else None

    print(f"\n{'='*55}")
    if new_acc is not None:
        print(f"  Neue Test-Acc:    {new_acc:.4%}")
        print(f"  Alte Test-Acc:    {current_acc:.4%}" if current_acc else "  Altes Modell:    nicht vorhanden")
        print(f"  Baseline (Heim):  {baseline_acc:.4%}")

    improved = (
        new_acc is not None and
        (current_acc is None or new_acc >= current_acc + MIN_IMPROVEMENT)
    )

    report = {
        "mode":              "optimize",
        "trials":            n_trials,
        "best_cv_accuracy":  round(best_cv, 4),
        "best_params":       best_params,
        "new_test_accuracy": round(new_acc, 4)      if new_acc      is not None else None,
        "old_test_accuracy": round(current_acc, 4)  if current_acc  is not None else None,
        "baseline_accuracy": round(baseline_acc, 4) if baseline_acc is not None else None,
        "improvement":       round(new_acc - current_acc, 4) if (new_acc and current_acc) else None,
        "saved":             improved,
        "train_games":       len(train),
        "feature_count":     len(feature_cols),
    }

    if improved:
        joblib.dump(final_model, MODEL_PKL)
        print(f"  Verbesserung: +{new_acc - (current_acc or 0):.4%} — Modell gespeichert.")
        if os.path.exists(CALIB_SCRIPT):
            subprocess.run([sys.executable, CALIB_SCRIPT], check=False, cwd=BASE_DIR)
    else:
        delta = (new_acc - current_acc) if (new_acc and current_acc) else 0
        print(f"  Keine ausreichende Verbesserung ({delta:+.4%}) — altes Modell behalten.")

    print(f"{'='*55}\n")

    report["generated_at"] = pd.Timestamp.now(tz="US/Eastern").tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {REPORT_PATH}\n")

    return report


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
    parser.add_argument("--force",    action="store_true", help="Immer neu trainieren")
    parser.add_argument("--optimize", action="store_true", help="Optuna Hyperparameter-Suche")
    parser.add_argument("--trials",   type=int, default=50, help="Anzahl Optuna Trials (default: 50)")
    args = parser.parse_args()

    if args.optimize:
        optimize(n_trials=args.trials)
    else:
        run(force=args.force)
