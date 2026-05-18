"""data_quality_checker.py — validates model_data.csv after each pipeline run."""
import json
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DATA_CSV = os.path.join(BASE_DIR, "data", "model_data.csv")
BASE_GAMES_CSV = os.path.join(BASE_DIR, "data", "base_games.csv")
FEATURE_COLS_CSV = os.path.join(BASE_DIR, "models", "feature_cols.csv")
REPORT_PATH = os.path.join(BASE_DIR, "output", "data_quality_report.json")

NULL_WARN = 0.05   # 5% nulls → warning
NULL_FAIL = 0.20   # 20% nulls → error
# h2h data is structurally sparse (teams only meet a few times per season)
NULL_FAIL_H2H = 0.95
MAX_STALE_DAYS = 3


def check():
    issues = []
    warnings = []

    # ── 1. model_data.csv exists ───────────────────────────────────────────
    if not os.path.exists(MODEL_DATA_CSV):
        print("FEHLER: data/model_data.csv nicht gefunden.")
        return False

    df = pd.read_csv(MODEL_DATA_CSV)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")
    total_rows = len(df)

    # ── 2. Feature-Spalten null-Prüfung ───────────────────────────────────
    feature_cols = []
    if os.path.exists(FEATURE_COLS_CSV):
        feature_cols = pd.read_csv(FEATURE_COLS_CSV).squeeze().tolist()

    for col in feature_cols:
        if col not in df.columns:
            issues.append(f"Feature '{col}' komplett fehlend in model_data.csv")
            continue
        null_rate = df[col].isna().mean()
        # h2h columns are structurally sparse — use relaxed threshold
        threshold = NULL_FAIL_H2H if "h2h" in col else NULL_FAIL
        if null_rate >= threshold:
            issues.append(f"'{col}': {null_rate:.1%} null (Schwelle: {threshold:.0%})")
        elif null_rate >= NULL_WARN:
            warnings.append(f"'{col}': {null_rate:.1%} null")

    # ── 3. Duplikate ──────────────────────────────────────────────────────
    key_cols = [c for c in ["gameDateTimeEst", "hometeamName", "awayteamName"] if c in df.columns]
    if key_cols:
        n_dups = df.duplicated(subset=key_cols).sum()
        if n_dups > 0:
            issues.append(f"{n_dups} doppelte Zeilen (Datum + Heim + Auswärts)")

    # ── 4. Aktualität ─────────────────────────────────────────────────────
    latest = df["gameDateTimeEst"].max()
    if pd.notna(latest):
        now = pd.Timestamp.now(tz="US/Eastern").tz_localize(None)
        days_old = (now - latest).days
        if days_old > MAX_STALE_DAYS:
            warnings.append(
                f"Neueste Daten sind {days_old} Tage alt (letztes Spiel: {latest.date()})"
            )

    # ── 5. home_win ist binär ─────────────────────────────────────────────
    if "home_win" in df.columns:
        bad = ~df["home_win"].isin([0, 1, 0.0, 1.0])
        if bad.sum() > 0:
            issues.append(f"home_win: {bad.sum()} Werte die nicht 0/1 sind")

    # ── 6. Win-Rate-Wertebereich (nur echte Raten, keine Differenzen/Trends) ─
    rate_cols = [
        c for c in df.columns
        if ("winrate" in c.lower() or "win_rate" in c.lower())
        and not any(c.endswith(s) for s in ("_diff", "_trend", "trend_diff"))
    ]
    for col in rate_cols:
        out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
        if out_of_range > 0:
            issues.append(f"'{col}': {out_of_range} Werte außerhalb [0, 1]")

    # ── 7. base_games.csv existiert ───────────────────────────────────────
    if not os.path.exists(BASE_GAMES_CSV):
        warnings.append("data/base_games.csv nicht gefunden")

    # ── Report schreiben ──────────────────────────────────────────────────
    report = {
        "generated_at": pd.Timestamp.now(tz="US/Eastern").tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S"),
        "total_rows": total_rows,
        "features_checked": len(feature_cols),
        "latest_game_date": str(latest.date()) if pd.notna(latest) else None,
        "issues": issues,
        "warnings": warnings,
        "passed": len(issues) == 0,
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

    if warnings:
        print(f"\n  Warnungen ({len(warnings)}):")
        for w in warnings:
            print(f"    !  {w}")
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
