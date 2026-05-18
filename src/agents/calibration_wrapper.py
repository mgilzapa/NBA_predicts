"""calibration_wrapper.py

Trains an IsotonicRegression calibrator on top of the saved XGBoost model,
using the most recent 90 days as a held-out calibration set.

Saves models/calibration_model.pkl — predict.py loads it automatically if present.

Run manually after retraining or via auto_retrainer.py.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DATA_CSV = os.path.join(BASE_DIR, "data", "model_data.csv")
FEATURE_COLS_CSV = os.path.join(BASE_DIR, "models", "feature_cols.csv")
MODEL_PKL = os.path.join(BASE_DIR, "models", "best_xgb_model.pkl")
CALIB_PKL = os.path.join(BASE_DIR, "models", "calibration_model.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "output", "calibration_report.json")

CALIB_DAYS = 90   # days of data to use for calibration (held out from training)
EXCLUDE_COLS = [
    "home_off_rating", "away_off_rating", "off_rating_diff",
    "home_def_rating", "away_def_rating", "def_rating_diff",
    "home_net_rating", "away_net_rating", "net_rating_diff",
    "home_h2h_winrate", "away_h2h_winrate", "h2h_winrate_diff",
    "same_division", "is_playoff", "away_opponent_strength",
]


def calibrate():
    for path, label in [(MODEL_DATA_CSV, "model_data.csv"), (FEATURE_COLS_CSV, "feature_cols.csv"), (MODEL_PKL, "best_xgb_model.pkl")]:
        if not os.path.exists(path):
            print(f"FEHLER: {label} nicht gefunden.")
            return False

    df = pd.read_csv(MODEL_DATA_CSV)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")

    feature_cols = pd.read_csv(FEATURE_COLS_CSV).squeeze().tolist()
    feature_cols = [c for c in feature_cols if c in df.columns and c not in EXCLUDE_COLS]

    model = joblib.load(MODEL_PKL)

    now = pd.Timestamp.now(tz="US/Eastern").tz_localize(None)
    calib_start = now - pd.Timedelta(days=CALIB_DAYS)

    df_calib = df[
        (df["gameDateTimeEst"] >= calib_start) &
        (df["gameDateTimeEst"] < now)
    ].dropna(subset=feature_cols + ["home_win"]).copy()

    if len(df_calib) < 30:
        print(f"WARNUNG: Nur {len(df_calib)} Kalibrierungsbeispiele — mindestens 30 benötigt.")
        return False

    X_calib = df_calib[feature_cols].values
    y_calib = df_calib["home_win"].values
    raw_probs = model.predict_proba(X_calib)[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs.reshape(-1, 1), y_calib)

    # Evaluate: compare calibration error before and after
    def mean_calibration_error(probs, labels, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        errors = []
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() > 0:
                errors.append(abs(probs[mask].mean() - labels[mask].mean()))
        return np.mean(errors) if errors else 0.0

    cal_probs = calibrator.predict(raw_probs.reshape(-1, 1))
    mce_before = mean_calibration_error(raw_probs, y_calib)
    mce_after = mean_calibration_error(cal_probs, y_calib)

    joblib.dump(calibrator, CALIB_PKL)

    report = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "calib_samples": int(len(df_calib)),
        "calib_days": CALIB_DAYS,
        "mean_cal_error_before": round(float(mce_before), 4),
        "mean_cal_error_after": round(float(mce_after), 4),
        "improvement": round(float(mce_before - mce_after), 4),
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  CALIBRATION")
    print(f"{'='*50}")
    print(f"  Kalibrierungs-Samples:  {len(df_calib)}")
    print(f"  Mean Cal. Error vorher: {mce_before:.4f}")
    print(f"  Mean Cal. Error nachher:{mce_after:.4f}")
    print(f"  Verbesserung:           {mce_before - mce_after:+.4f}")
    print(f"  Modell gespeichert:     {CALIB_PKL}")
    print(f"{'='*50}\n")

    return True


if __name__ == "__main__":
    ok = calibrate()
    sys.exit(0 if ok else 1)
