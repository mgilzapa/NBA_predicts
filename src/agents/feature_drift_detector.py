"""feature_drift_detector.py — checks that feature_cols.csv stays in sync with model_data.csv and the saved model."""
import json
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURE_COLS_CSV = os.path.join(BASE_DIR, "models", "feature_cols.csv")
MODEL_DATA_CSV = os.path.join(BASE_DIR, "data", "model_data.csv")
MODEL_PKL = os.path.join(BASE_DIR, "models", "best_xgb_model.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "output", "feature_drift_report.json")

# Columns excluded from the model at training/prediction time (kept in feature_cols.csv for reference)
EXCLUDE_COLS = [
    "home_off_rating", "away_off_rating", "off_rating_diff",
    "home_def_rating", "away_def_rating", "def_rating_diff",
    "home_net_rating", "away_net_rating", "net_rating_diff",
    "same_division", "away_opponent_strength",
    "injury_impact_diff",
]

# Column prefixes that indicate a feature-like column in model_data.csv
FEATURE_PREFIXES = (
    "home_", "away_", "elo", "winrate", "pts_", "reb_", "ast_",
    "average_", "rest_", "playoff_", "missing_", "top",
)
# Suffixes to exclude from "potential new feature" suggestions
EXCLUDE_SUFFIXES = ("Name", "Id", "id", "Score", "_win", "teamName", "gameId")


def detect():
    issues = []
    warnings = []

    # ── 1. feature_cols.csv laden ─────────────────────────────────────────
    if not os.path.exists(FEATURE_COLS_CSV):
        print("FEHLER: models/feature_cols.csv nicht gefunden.")
        return False

    expected = pd.read_csv(FEATURE_COLS_CSV).squeeze().tolist()

    # ── 2. Spalten aus model_data.csv lesen ───────────────────────────────
    available = []
    if os.path.exists(MODEL_DATA_CSV):
        available = list(pd.read_csv(MODEL_DATA_CSV, nrows=0).columns)
    else:
        issues.append("data/model_data.csv nicht gefunden — Spaltenprüfung nicht möglich")

    # ── 3. Fehlende Features (model braucht sie, aber Daten haben sie nicht) ──
    missing = [f for f in expected if f not in available]
    if missing:
        issues.append(
            f"{len(missing)} Feature(s) in feature_cols.csv aber nicht in model_data.csv: {missing}"
        )

    # ── 4. Potenzielle neue Features (in Daten aber nicht genutzt) ────────
    new_candidates = [
        c for c in available
        if any(c.startswith(p) for p in FEATURE_PREFIXES)
        and not any(c.endswith(s) for s in EXCLUDE_SUFFIXES)
        and c not in expected
    ]
    if new_candidates:
        shown = new_candidates[:10]
        suffix = f"  (+{len(new_candidates) - 10} weitere)" if len(new_candidates) > 10 else ""
        warnings.append(
            f"{len(new_candidates)} mögliche neue Features in model_data.csv (nicht in feature_cols.csv): "
            f"{shown}{suffix}"
        )

    # ── 5. Modell-Datei vorhanden? ────────────────────────────────────────
    if not os.path.exists(MODEL_PKL):
        issues.append("models/best_xgb_model.pkl nicht gefunden — Modell muss neu trainiert werden")

    # ── 6. Modell erwartet gleiche Anzahl Features wie feature_cols.csv (minus EXCLUDE_COLS) ───
    effective_features = [c for c in expected if c not in EXCLUDE_COLS and c in available]
    if os.path.exists(MODEL_PKL) and not missing:
        try:
            import joblib
            model = joblib.load(MODEL_PKL)
            if hasattr(model, "n_features_in_"):
                if model.n_features_in_ != len(effective_features):
                    issues.append(
                        f"Modell erwartet {model.n_features_in_} Features, "
                        f"feature_cols.csv hat {len(effective_features)} effektive Features "
                        f"({len(expected)} gesamt - {len(EXCLUDE_COLS)} EXCLUDE_COLS)"
                    )
        except Exception as exc:
            warnings.append(f"Modell konnte nicht geladen werden für Feature-Count-Check: {exc}")

    # ── Report ────────────────────────────────────────────────────────────
    report = {
        "generated_at": pd.Timestamp.now(tz="US/Eastern").tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S"),
        "expected_features": len(expected),
        "effective_features": len(effective_features),
        "available_columns": len(available),
        "missing_from_data": missing,
        "potential_new_features": new_candidates,
        "issues": issues,
        "warnings": warnings,
        "passed": len(issues) == 0,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    status = "OK" if not issues else "DRIFT ERKANNT"
    print(f"\n{'='*55}")
    print(f"  FEATURE DRIFT DETECTION  —  {status}")
    print(f"{'='*55}")
    print(f"  feature_cols.csv:   {len(expected)} Features")
    print(f"  model_data.csv:     {len(available)} Spalten")

    if warnings:
        print(f"\n  Warnungen ({len(warnings)}):")
        for w in warnings:
            print(f"    !  {w}")
    if issues:
        print(f"\n  Probleme ({len(issues)}):")
        for i in issues:
            print(f"    x  {i}")
    if not issues and not warnings:
        print("\n  Kein Drift festgestellt.")

    print(f"\n  Report: {REPORT_PATH}")
    print(f"{'='*55}\n")

    return len(issues) == 0


if __name__ == "__main__":
    detect()
    sys.exit(0)  # always exit 0 — drift is a warning, not a pipeline blocker
