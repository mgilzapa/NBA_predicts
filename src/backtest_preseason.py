"""backtest_preseason.py

Vergleicht drei Varianten im Umgang mit Preseason-Spielen per Walk-Forward-Backtest:

  keep          – Preseason zaehlt ueberall mit (aktueller Stand)
  exclude_form  – Preseason nur im ELO (K=8), nicht in Form-Features / Training
  drop          – Preseason komplett ignoriert

Fuer jede Variante wird feature_engineering.py mit PRESEASON_MODE neu ausgefuehrt
(Ausgabe nach output/backtest_preseason/, das produktive model_data.csv bleibt unberuehrt).
Dann wird pro Testsaison ein Modell auf allen frueheren Spielen trainiert (gleiche Parameter,
Features und Gewichtung wie auto_retrainer.py) und auf Regular Season + Playoffs der Saison
getestet. Alle Varianten werden auf exakt denselben Testspielen bewertet.

Usage:
    python src/backtest_preseason.py            # Features neu berechnen + Backtest
    python src/backtest_preseason.py --reuse    # vorhandene Feature-Dateien wiederverwenden
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "agents"))
from auto_retrainer import EXCLUDE_COLS, PRESEASON_WEIGHT  # noqa: E402

OUT_DIR = os.path.join(BASE_DIR, "output", "backtest_preseason")
REPORT_PATH = os.path.join(BASE_DIR, "output", "backtest_preseason.json")
FEATURE_COLS_CSV = os.path.join(BASE_DIR, "models", "feature_cols.csv")

MODES = ["keep", "exclude_form", "drop"]
TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
EARLY_DAYS = 21  # "Saisonstart" = erste 3 Wochen der Regular Season


def season_start_year(date):
    return date.year if date.month >= 8 else date.year - 1


def build_features(mode):
    os.makedirs(os.path.join(OUT_DIR, mode), exist_ok=True)
    path = os.path.join(OUT_DIR, mode, "model_data.csv")
    env = {**os.environ, "PRESEASON_MODE": mode, "MODEL_DATA_PATH": path}
    print(f"\n>>> feature_engineering.py mit PRESEASON_MODE={mode}")
    subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "src", "feature_engineering.py")],
        cwd=BASE_DIR, env=env, check=True, stdout=subprocess.DEVNULL,
    )
    return path


def load(path, feature_cols):
    df = pd.read_csv(path)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    df = df.dropna(subset=feature_cols + ["home_win"])
    df = df[(df["home_elo_games_played"] > 20) & (df["away_elo_games_played"] > 20)]
    df["season"] = df["gameDateTimeEst"].apply(season_start_year)
    return df


def make_model():
    # identisch zu auto_retrainer.retrain()
    return XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10, gamma=1.0, reg_alpha=0.5,
        reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss",
        random_state=42, verbosity=0,
    )


def predict_season(df, season, feature_cols):
    test_start = pd.Timestamp(f"{season}-08-01")
    train = df[df["gameDateTimeEst"] < test_start]
    days_old = (test_start - train["gameDateTimeEst"]).dt.days.clip(lower=0)
    weights = 1 / (1 + days_old / 365)
    weights = weights.where(train["is_preseason"] != 1, weights * PRESEASON_WEIGHT)

    test = df[(df["season"] == season) & (df["is_preseason"] == 0)]
    model = make_model()
    model.fit(train[feature_cols].values, train["home_win"].values, sample_weight=weights.values)
    probs = model.predict_proba(test[feature_cols].values)[:, 1]
    return pd.DataFrame({
        "gameId": test["gameId"].values,
        "date": test["gameDateTimeEst"].values,
        "season": season,
        "is_playoff": test["is_playoff"].values,
        "home_win": test["home_win"].values,
        "prob": probs,
    })


def metrics(frame):
    if frame.empty:
        return {"games": 0}
    return {
        "games": int(len(frame)),
        "accuracy": round(float(((frame["prob"] >= 0.5) == frame["home_win"]).mean()), 4),
        "log_loss": round(float(log_loss(frame["home_win"], frame["prob"], labels=[0, 1])), 4),
    }


def main(reuse):
    feature_cols = pd.read_csv(FEATURE_COLS_CSV).squeeze().tolist()
    feature_cols = [c for c in feature_cols if c not in EXCLUDE_COLS]

    preds = {}
    for mode in MODES:
        path = os.path.join(OUT_DIR, mode, "model_data.csv")
        if not (reuse and os.path.exists(path)):
            path = build_features(mode)
        df = load(path, [c for c in feature_cols if c != "market_prob_home_win"])
        cols = [c for c in feature_cols if c in df.columns]
        print(f">>> Backtest {mode}: {len(df)} Spiele, {len(cols)} Features")
        preds[mode] = pd.concat(
            [predict_season(df, s, cols) for s in TEST_SEASONS], ignore_index=True
        )

    # Nur Spiele bewerten, die in allen Varianten vorhergesagt wurden
    common = set.intersection(*(set(p["gameId"]) for p in preds.values()))
    for mode in MODES:
        p = preds[mode][preds[mode]["gameId"].isin(common)].copy()
        rs_start = p[p["is_playoff"] == 0].groupby("season")["date"].min()
        p["early"] = p["date"] < p["season"].map(rs_start) + pd.Timedelta(days=EARLY_DAYS)
        preds[mode] = p

    report = {"test_seasons": TEST_SEASONS, "early_days": EARLY_DAYS, "results": {}}
    for mode in MODES:
        p = preds[mode]
        report["results"][mode] = {
            "all": metrics(p),
            "season_start": metrics(p[p["early"]]),
            "rest_of_season": metrics(p[~p["early"] & (p["is_playoff"] == 0)]),
            "playoffs": metrics(p[p["is_playoff"] == 1]),
            "per_season_start": {
                int(s): metrics(g[g["early"]]) for s, g in p.groupby("season")
            },
        }

    def row(label, key):
        cells = []
        for mode in MODES:
            m = report["results"][mode][key]
            cells.append(f"{m['accuracy']:.2%} / {m['log_loss']:.4f}")
        n = report["results"][MODES[0]][key]["games"]
        print(f"  {label:<18} {n:>6}   " + "   ".join(f"{c:<18}" for c in cells))

    print(f"\n{'=' * 90}")
    print(f"  BACKTEST PRESEASON  (Saisons {TEST_SEASONS[0]}–{TEST_SEASONS[-1]}, Accuracy / Log-Loss)")
    print(f"{'=' * 90}")
    print(f"  {'':<18} {'Spiele':>6}   " + "   ".join(f"{m:<18}" for m in MODES))
    row("Gesamt", "all")
    row(f"Saisonstart ({EARLY_DAYS}T)", "season_start")
    row("Rest Reg. Season", "rest_of_season")
    row("Playoffs", "playoffs")
    print("\n  Saisonstart pro Saison (Accuracy):")
    for s in TEST_SEASONS:
        cells = [f"{report['results'][m]['per_season_start'][s]['accuracy']:.2%}" for m in MODES]
        print(f"    {s}-{(s + 1) % 100:02d}   " + "   ".join(f"{c:<18}" for c in cells))
    print(f"{'=' * 90}\n")

    report["generated_at"] = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="Feature-Dateien nicht neu berechnen")
    main(parser.parse_args().reuse)
