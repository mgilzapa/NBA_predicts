"""model_evaluator.py — computes accuracy metrics from all_predictions.xlsx and flags regressions."""
import json
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALL_PREDICTIONS = os.path.join(BASE_DIR, "output", "all_predictions.xlsx")
REPORT_PATH = os.path.join(BASE_DIR, "output", "model_evaluation.json")

MIN_GAMES_FOR_ALERT = 20  # don't raise alert on tiny sample sizes


def evaluate():
    if not os.path.exists(ALL_PREDICTIONS):
        print("WARNUNG: output/all_predictions.xlsx nicht gefunden.")
        return None

    df = pd.read_excel(ALL_PREDICTIONS)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Only completed games (Actual Winner known)
    df_eval = df[df["Actual Winner"].notna()].copy()
    if df_eval.empty:
        print("Keine abgeschlossenen Spiele für Auswertung vorhanden.")
        return None

    df_eval["correct"] = df_eval["Predicted Winner"] == df_eval["Actual Winner"]
    df_eval["home_win_actual"] = df_eval["Actual Winner"] == df_eval["Home Team"]

    total = len(df_eval)
    overall_acc = df_eval["correct"].mean()
    baseline_acc = df_eval["home_win_actual"].mean()

    # Rolling windows
    now = pd.Timestamp.now(tz="US/Eastern").tz_localize(None)
    df_7 = df_eval[df_eval["Date"] >= now - pd.Timedelta(days=7)]
    df_30 = df_eval[df_eval["Date"] >= now - pd.Timedelta(days=30)]

    def window_stats(subset):
        if subset.empty:
            return {"games": 0, "accuracy": None}
        return {"games": len(subset), "accuracy": round(subset["correct"].mean(), 4)}

    # Confidence calibration buckets
    calibration = []
    if "probability_home_win" in df_eval.columns:
        df_eval["confidence"] = df_eval["probability_home_win"].apply(lambda p: max(p, 1 - p))
        for lo, hi in [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]:
            mask = (df_eval["confidence"] >= lo) & (df_eval["confidence"] < hi)
            sub = df_eval[mask]
            if not sub.empty:
                calibration.append({
                    "range": f"{int(lo*100)}-{int(min(hi, 1.0)*100)}%",
                    "games": len(sub),
                    "accuracy": round(sub["correct"].mean(), 4),
                })

    # Daily accuracy for trend (last 14 days)
    daily_trend = []
    df_14 = df_eval[df_eval["Date"] >= now - pd.Timedelta(days=14)]
    for date, grp in df_14.groupby(df_14["Date"].dt.date):
        daily_trend.append({
            "date": str(date),
            "games": len(grp),
            "accuracy": round(grp["correct"].mean(), 4),
        })

    alert = total >= MIN_GAMES_FOR_ALERT and overall_acc < baseline_acc

    report = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "date_range": {
            "from": str(df_eval["Date"].min().date()),
            "to": str(df_eval["Date"].max().date()),
        },
        "total_games": total,
        "overall_accuracy": round(overall_acc, 4),
        "baseline_accuracy": round(baseline_acc, 4),
        "vs_baseline": round(overall_acc - baseline_acc, 4),
        "last_7_days": window_stats(df_7),
        "last_30_days": window_stats(df_30),
        "calibration": calibration,
        "daily_trend_14d": daily_trend,
        "alert_model_below_baseline": alert,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Ausgabe ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  MODEL EVALUATION REPORT")
    print(f"{'='*55}")
    print(f"  Zeitraum:   {report['date_range']['from']}  →  {report['date_range']['to']}")
    print(f"  Spiele:     {total}")
    print()
    print(f"  Accuracy gesamt:   {overall_acc:.2%}")
    print(f"  Baseline (Heim):   {baseline_acc:.2%}")
    print(f"  vs. Baseline:      {overall_acc - baseline_acc:+.2%}")

    stats_7 = window_stats(df_7)
    stats_30 = window_stats(df_30)
    if stats_7["games"]:
        print(f"\n  Letzte  7 Tage:    {stats_7['accuracy']:.2%}  ({stats_7['games']} Spiele)")
    if stats_30["games"]:
        print(f"  Letzte 30 Tage:    {stats_30['accuracy']:.2%}  ({stats_30['games']} Spiele)")

    if calibration:
        print(f"\n  Kalibrierung (Konfidenz → Trefferquote):")
        for b in calibration:
            print(f"    {b['range']:<10}  {b['accuracy']:.2%}  ({b['games']} Spiele)")

    if alert:
        print(f"\n  WARNUNG: Modell ist schlechter als die Baseline!")
    else:
        print(f"\n  Kein Regressions-Alert.")

    print(f"\n  Report: {REPORT_PATH}")
    print(f"{'='*55}\n")

    return report


if __name__ == "__main__":
    evaluate()
    sys.exit(0)  # non-blocking — missing data is a warning, not a pipeline error
