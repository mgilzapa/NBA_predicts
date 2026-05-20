"""prediction_auditor.py — sanity-checks today's predictions before they go live.

Checks:
  1. Unrealistically high confidence (>85%)
  2. Model vs. market divergence (>25pp difference in win probability)
  3. ELO upset: model picks team with significantly lower ELO (diff > 100)
  4. Form upset: model picks team with much worse last-5 winrate (gap > 0.4)
  5. Missing features: games that ran without complete feature data

Runs after odds_feature_injector — reads predictions.xlsx + odds.json + model_data.csv.
Writes output/prediction_audit_report.json. Exits 0 always (warnings only, not a blocker).
"""
import json
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREDICTIONS_XLSX = os.path.join(BASE_DIR, "output", "predictions.xlsx")
ODDS_JSON        = os.path.join(BASE_DIR, "web", "odds.json")
MODEL_DATA_CSV   = os.path.join(BASE_DIR, "data", "model_data.csv")
REPORT_PATH      = os.path.join(BASE_DIR, "output", "prediction_audit_report.json")

HIGH_CONF_THRESHOLD  = 0.85   # flag if model confidence > 85%
ODDS_DIVERGE_THRESH  = 0.25   # flag if model vs. market differ by > 25pp
ELO_UPSET_THRESH     = 100    # flag if model picks team with ELO > 100 lower
FORM_UPSET_THRESH    = 0.40   # flag if model picks team whose last-5 winrate is 0.4 worse


def _no_vig_prob(home_odds: float, away_odds: float) -> float:
    """Convert decimal odds to no-vig home win probability."""
    p_home_raw = 1 / home_odds
    p_away_raw = 1 / away_odds
    total = p_home_raw + p_away_raw
    return p_home_raw / total if total > 0 else 0.5


def _load_odds() -> dict:
    """Returns dict keyed by (date, home_lower, away_lower) -> no_vig_home_prob."""
    if not os.path.exists(ODDS_JSON):
        return {}
    with open(ODDS_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for key, game in raw.get("games", {}).items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        date_str, home_lower, away_lower = parts
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue
        probs = [_no_vig_prob(b["home"], b["away"]) for b in bookmakers
                 if b.get("home") and b.get("away")]
        if probs:
            result[(date_str, home_lower, away_lower)] = sum(probs) / len(probs)
    return result


def _latest_team_stats() -> dict:
    """Returns dict: team_name -> {elo, last5_winrate} from most recent row in model_data.csv."""
    if not os.path.exists(MODEL_DATA_CSV):
        return {}
    df = pd.read_csv(MODEL_DATA_CSV, usecols=lambda c: c in {
        "gameDateTimeEst", "hometeamName", "awayteamName",
        "home_elo", "away_elo",
        "home_last5_winrate", "away_last5_winrate",
    })
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")
    df = df.sort_values("gameDateTimeEst")

    stats = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("hometeamName")):
            stats[row["hometeamName"]] = {
                "elo": row.get("home_elo"),
                "last5_winrate": row.get("home_last5_winrate"),
            }
        if pd.notna(row.get("awayteamName")):
            stats[row["awayteamName"]] = {
                "elo": row.get("away_elo"),
                "last5_winrate": row.get("away_last5_winrate"),
            }
    return stats


def audit() -> dict:
    warnings = []
    flags = []

    # ── Load predictions ─────────────────────────────────────────────────────
    if not os.path.exists(PREDICTIONS_XLSX):
        print("WARNUNG: output/predictions.xlsx nicht gefunden — kein Audit moeglich.")
        return {"predictions": 0, "flags": [], "warnings": ["predictions.xlsx fehlt"], "passed": False}

    try:
        df = pd.read_excel(PREDICTIONS_XLSX, sheet_name="predictions_today")
    except Exception as exc:
        warnings.append(f"predictions.xlsx konnte nicht gelesen werden: {exc}")
        df = pd.DataFrame()

    if df.empty:
        report = {
            "generated_at": pd.Timestamp.now(tz="US/Eastern").tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S"),
            "predictions": 0,
            "flags": [],
            "warnings": warnings + ["Keine Predictions heute"],
            "passed": True,
        }
        _write_and_print(report)
        return report

    odds_map   = _load_odds()
    team_stats = _latest_team_stats()

    for _, row in df.iterrows():
        home  = str(row.get("Home Team", "")).strip()
        away  = str(row.get("Away Team", "")).strip()
        pred  = str(row.get("Predicted Winner", "")).strip()
        prob  = row.get("probability_home_win")
        date  = str(row.get("Date", ""))[:10]  # YYYY-MM-DD

        if pd.isna(prob):
            flags.append({
                "game": f"{away} @ {home}",
                "type": "missing_probability",
                "detail": "probability_home_win fehlt — Feature-Daten unvollstaendig?",
                "severity": "high",
            })
            continue

        home_prob = float(prob)
        confidence = max(home_prob, 1 - home_prob)

        # ── Check 1: Unrealistically high confidence ──────────────────────
        if confidence > HIGH_CONF_THRESHOLD:
            flags.append({
                "game": f"{away} @ {home}",
                "type": "high_confidence",
                "detail": f"Confidence {confidence:.1%} > {HIGH_CONF_THRESHOLD:.0%} — ggf. Feature-Anomalie",
                "severity": "medium",
            })

        # ── Check 2: Model vs. market divergence ─────────────────────────
        odds_key = (date, home.lower(), away.lower())
        market_prob = odds_map.get(odds_key)
        if market_prob is not None:
            divergence = abs(home_prob - market_prob)
            if divergence > ODDS_DIVERGE_THRESH:
                model_winner  = home if home_prob > 0.5 else away
                market_winner = home if market_prob > 0.5 else away
                direction = "GLEICH" if model_winner == market_winner else "ENTGEGENGESETZT"
                flags.append({
                    "game": f"{away} @ {home}",
                    "type": "odds_divergence",
                    "detail": (
                        f"Modell: {home_prob:.1%} fuer {home} | "
                        f"Markt: {market_prob:.1%} fuer {home} | "
                        f"Diff: {divergence:.1%} ({direction})"
                    ),
                    "severity": "high" if direction == "ENTGEGENGESETZT" else "low",
                })

        # ── Check 3: ELO upset ───────────────────────────────────────────
        home_elo = team_stats.get(home, {}).get("elo")
        away_elo = team_stats.get(away, {}).get("elo")
        if home_elo and away_elo:
            elo_diff = home_elo - away_elo
            if pred == home and elo_diff < -ELO_UPSET_THRESH:
                flags.append({
                    "game": f"{away} @ {home}",
                    "type": "elo_upset",
                    "detail": (
                        f"Modell tippt {home} (ELO {home_elo:.0f}) gegen "
                        f"{away} (ELO {away_elo:.0f}) — ELO-Diff: {elo_diff:.0f}"
                    ),
                    "severity": "low",
                })
            elif pred == away and elo_diff > ELO_UPSET_THRESH:
                flags.append({
                    "game": f"{away} @ {home}",
                    "type": "elo_upset",
                    "detail": (
                        f"Modell tippt {away} (ELO {away_elo:.0f}) gegen "
                        f"{home} (ELO {home_elo:.0f}) — ELO-Diff: {elo_diff:.0f}"
                    ),
                    "severity": "low",
                })

        # ── Check 4: Form upset ──────────────────────────────────────────
        home_wr = team_stats.get(home, {}).get("last5_winrate")
        away_wr = team_stats.get(away, {}).get("last5_winrate")
        if home_wr is not None and away_wr is not None:
            if pred == home and (home_wr - away_wr) < -FORM_UPSET_THRESH:
                flags.append({
                    "game": f"{away} @ {home}",
                    "type": "form_upset",
                    "detail": (
                        f"Modell tippt {home} (L5-WR {home_wr:.0%}) gegen "
                        f"{away} (L5-WR {away_wr:.0%}) — Form-Diff: {home_wr - away_wr:.0%}"
                    ),
                    "severity": "low",
                })
            elif pred == away and (away_wr - home_wr) < -FORM_UPSET_THRESH:
                flags.append({
                    "game": f"{away} @ {home}",
                    "type": "form_upset",
                    "detail": (
                        f"Modell tippt {away} (L5-WR {away_wr:.0%}) gegen "
                        f"{home} (L5-WR {home_wr:.0%}) — Form-Diff: {away_wr - home_wr:.0%}"
                    ),
                    "severity": "low",
                })

    high_severity = [f for f in flags if f.get("severity") == "high"]

    report = {
        "generated_at": pd.Timestamp.now(tz="US/Eastern").tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S"),
        "predictions": len(df),
        "flags": flags,
        "warnings": warnings,
        "passed": len(high_severity) == 0,
    }
    _write_and_print(report)
    return report


def _write_and_print(report: dict):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    flags    = report.get("flags", [])
    warnings = report.get("warnings", [])
    n_pred   = report.get("predictions", 0)
    status   = "OK" if report["passed"] else "WARNUNG"

    print(f"\n{'='*55}")
    print(f"  PREDICTION AUDIT  --  {status}")
    print(f"{'='*55}")
    print(f"  Predictions heute:  {n_pred}")

    if not flags and not warnings:
        print("\n  Alle Predictions sehen plausibel aus.")
    else:
        high   = [f for f in flags if f.get("severity") == "high"]
        medium = [f for f in flags if f.get("severity") == "medium"]
        low    = [f for f in flags if f.get("severity") == "low"]

        if high:
            print(f"\n  Kritische Flags ({len(high)}):")
            for fl in high:
                print(f"    [!] {fl['game']}")
                print(f"        {fl['type']}: {fl['detail']}")
        if medium:
            print(f"\n  Warnungen ({len(medium)}):")
            for fl in medium:
                print(f"    [~] {fl['game']}")
                print(f"        {fl['type']}: {fl['detail']}")
        if low:
            print(f"\n  Hinweise ({len(low)}):")
            for fl in low:
                print(f"    [i] {fl['game']}")
                print(f"        {fl['type']}: {fl['detail']}")
        if warnings:
            print(f"\n  System-Warnungen:")
            for w in warnings:
                print(f"    !  {w}")

    print(f"\n  Report: {REPORT_PATH}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    audit()
    sys.exit(0)
