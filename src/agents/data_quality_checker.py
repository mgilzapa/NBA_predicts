"""data_quality_checker.py — validates model_data.csv after each pipeline run."""
import json
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DATA_CSV = os.path.join(BASE_DIR, "data", "model_data.csv")
BASE_GAMES_CSV = os.path.join(BASE_DIR, "data", "base_games.csv")
FEATURE_COLS_CSV = os.path.join(BASE_DIR, "models", "feature_cols.csv")
REPORT_PATH = os.path.join(BASE_DIR, "output", "data_quality_report.json")

NULL_WARN = 0.05   # 5% nulls → warning
NULL_FAIL = 0.20   # 20% nulls → error
NULL_FAIL_H2H = 0.95
NULL_FAIL_RATING = 0.95
MAX_STALE_DAYS = 3

# Features to check for IQR outliers (must exist in df)
OUTLIER_COLS = ["home_elo", "away_elo", "elo_diff", "home_rest_days", "away_rest_days",
                "home_last5_avg_points", "away_last5_avg_points"]
IQR_MULTIPLIER = 5.0  # flag only extreme outliers (5× IQR)

# Box-score-derived features — used for coverage check by season
BOX_SCORE_FEATURES = ["home_last5_pts", "away_last5_pts", "home_last5_rebounds",
                      "home_last5_ast", "home_last5_player_count"]


def _season_label(date):
    """Oct–Dec → current year; Jan–Jun → previous year (e.g. 2025 = 2024-25 season)."""
    if pd.isna(date):
        return None
    return date.year if date.month >= 10 else date.year - 1


def check():
    issues = []
    warnings = []

    if not os.path.exists(MODEL_DATA_CSV):
        print("FEHLER: data/model_data.csv nicht gefunden.")
        return False

    df = pd.read_csv(MODEL_DATA_CSV)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")
    total_rows = len(df)

    df["_season"] = df["gameDateTimeEst"].apply(_season_label)

    # ── 1. Feature null-Prüfung ───────────────────────────────────────────
    feature_cols = []
    if os.path.exists(FEATURE_COLS_CSV):
        feature_cols = pd.read_csv(FEATURE_COLS_CSV).squeeze().tolist()

    for col in feature_cols:
        if col not in df.columns:
            issues.append(f"Feature '{col}' komplett fehlend in model_data.csv")
            continue
        null_rate = df[col].isna().mean()
        if col == "market_prob_home_win":
            # Always NaN in historical data by design — filled live at prediction time
            continue
        if "h2h" in col:
            threshold = NULL_FAIL_H2H
        elif "rating" in col:
            threshold = NULL_FAIL_RATING
        else:
            threshold = NULL_FAIL
        if null_rate >= threshold:
            issues.append(f"'{col}': {null_rate:.1%} null (Schwelle: {threshold:.0%})")
        elif null_rate >= NULL_WARN:
            warnings.append(f"'{col}': {null_rate:.1%} null")

    # ── 2. Duplikate (gameId + Datum + Teams) ─────────────────────────────
    if "gameId" in df.columns:
        n_dups_id = df.duplicated(subset=["gameId"]).sum()
        if n_dups_id > 0:
            issues.append(f"{n_dups_id} doppelte gameId-Einträge")

    key_cols = [c for c in ["gameDateTimeEst", "hometeamName", "awayteamName"] if c in df.columns]
    if key_cols:
        n_dups = df.duplicated(subset=key_cols).sum()
        if n_dups > 0:
            issues.append(f"{n_dups} doppelte Zeilen (Datum + Heim + Auswärts)")

    # ── 3. Label-Distribution-Shift (home_win rate per season) ───────────
    season_stats = {}
    if "home_win" in df.columns:
        for season, grp in df.groupby("_season"):
            if season is None:
                continue
            hw_rate = grp["home_win"].mean()
            season_stats[str(season)] = {
                "games": int(len(grp)),
                "home_win_rate": round(float(hw_rate), 4),
                "alert": bool(abs(hw_rate - 0.563) > 0.08),
            }
            if abs(hw_rate - 0.563) > 0.08:
                warnings.append(
                    f"Saison {season}: home_win_rate={hw_rate:.3f} weicht stark vom Durchschnitt ab"
                )

    # ── 4. Outlier-Erkennung (IQR-Methode) ───────────────────────────────
    outlier_flags = {}
    for col in OUTLIER_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = max(q3 - q1, 1.0)  # guard against IQR=0 on heavily-clustered features
        lo, hi = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        n_out = int(((series < lo) | (series > hi)).sum())
        outlier_flags[col] = {"n_outliers": n_out, "low_bound": round(lo, 2), "high_bound": round(hi, 2)}
        if n_out > 0:
            warnings.append(f"'{col}': {n_out} Ausreißer außerhalb [{lo:.1f}, {hi:.1f}]")

    # ── 5. Temporal-Ordering-Sanity ───────────────────────────────────────
    if "gameDateTimeEst" in df.columns:
        future_mask = df["gameDateTimeEst"] > pd.Timestamp.now(tz="US/Eastern").tz_localize(None)
        n_future = future_mask.sum()
        if n_future > 0:
            issues.append(f"{n_future} Zeilen mit gameDateTimeEst in der Zukunft (Trainingsdaten-Leakage?)")

        if "home_win" in df.columns:
            # Rows with a valid home_win but a future date are suspicious
            labeled_future = (future_mask & df["home_win"].notna()).sum()
            if labeled_future > 0:
                issues.append(f"{labeled_future} Zeilen: home_win gesetzt aber Datum in der Zukunft")

    # ── 6. Feature-Coverage by Season (box-score features) ───────────────
    coverage_by_season = {}
    for col in BOX_SCORE_FEATURES:
        if col not in df.columns:
            continue
        for season, grp in df.groupby("_season"):
            if season is None:
                continue
            coverage = 1 - grp[col].isna().mean()
            key = str(season)
            if key not in coverage_by_season:
                coverage_by_season[key] = {}
            coverage_by_season[key][col] = round(float(coverage), 3)

    # ── 7. Aktualität ─────────────────────────────────────────────────────
    latest = df["gameDateTimeEst"].max()
    if pd.notna(latest):
        now = pd.Timestamp.now(tz="US/Eastern").tz_localize(None)
        days_old = (now - latest).days
        if days_old > MAX_STALE_DAYS:
            warnings.append(
                f"Neueste Daten sind {days_old} Tage alt (letztes Spiel: {latest.date()})"
            )

    # ── 8. home_win binär ─────────────────────────────────────────────────
    if "home_win" in df.columns:
        bad = ~df["home_win"].isin([0, 1, 0.0, 1.0])
        if bad.sum() > 0:
            issues.append(f"home_win: {bad.sum()} Werte die nicht 0/1 sind")

    # ── 9. Win-Rate-Wertebereich ──────────────────────────────────────────
    rate_cols = [
        c for c in df.columns
        if ("winrate" in c.lower() or "win_rate" in c.lower())
        and not any(c.endswith(s) for s in ("_diff", "_trend", "trend_diff"))
    ]
    for col in rate_cols:
        out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
        if out_of_range > 0:
            issues.append(f"'{col}': {out_of_range} Werte außerhalb [0, 1]")

    # ── 10. base_games.csv ───────────────────────────────────────────────
    if not os.path.exists(BASE_GAMES_CSV):
        warnings.append("data/base_games.csv nicht gefunden")

    # ── Report ────────────────────────────────────────────────────────────
    report = {
        "generated_at": pd.Timestamp.now(tz="US/Eastern").tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S"),
        "total_rows": total_rows,
        "features_checked": len(feature_cols),
        "latest_game_date": str(latest.date()) if pd.notna(latest) else None,
        "issues": issues,
        "warnings": warnings,
        "passed": len(issues) == 0,
        "by_season": season_stats,
        "outlier_flags": outlier_flags,
        "coverage_by_season": coverage_by_season,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    status = "OK" if not issues else "FEHLER"
    print(f"\n{'='*55}")
    print(f"  DATA QUALITY CHECK  —  {status}")
    print(f"{'='*55}")
    print(f"  Zeilen gesamt:      {total_rows}")
    print(f"  Features geprüft:   {len(feature_cols)}")
    if pd.notna(latest):
        print(f"  Letztes Spiel:      {latest.date()}")
    print(f"  Saisons im Dataset: {len(season_stats)}")

    if warnings:
        print(f"\n  Warnungen ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"    !  {w}")
        if len(warnings) > 10:
            print(f"    ... und {len(warnings)-10} weitere")
    if issues:
        print(f"\n  Fehler ({len(issues)}):")
        for i in issues:
            print(f"    x  {i}")
    if not issues and not warnings:
        print("\n  Alle Prüfungen bestanden.")

    print(f"\n  Report: {REPORT_PATH}")
    print(f"{'='*55}\n")

    return len(issues) == 0


if __name__ == "__main__":
    ok = check()
    sys.exit(0 if ok else 1)
