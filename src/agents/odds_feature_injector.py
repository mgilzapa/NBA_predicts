"""odds_feature_injector.py

Reads web/odds.json and output/predictions.xlsx, computes no-vig implied
home-win probability per bookmaker, and writes it as an informational column
(market_prob_home_win) to predictions_today — for display purposes only.

The blend has been removed: probability_home_win is the model's direct output,
which already incorporates market_prob_home_win as a trained feature via predict.py.

Run order in pipeline: after fetch_odds.py, before export_json.py.
"""
import json
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ODDS_JSON = os.path.join(BASE_DIR, "web", "odds.json")
PREDICTIONS_XLSX = os.path.join(BASE_DIR, "output", "predictions.xlsx")

MIN_BOOKMAKERS = 1


def normalize(name: str) -> str:
    return name.strip().lower()


def no_vig_home_prob(bookmakers: list) -> float | None:
    """Average no-vig implied home probability across all bookmakers."""
    probs = []
    for bm in bookmakers:
        h, a = bm.get("home"), bm.get("away")
        if h and a and h > 1.0 and a > 1.0:
            raw_h = 1.0 / h
            raw_a = 1.0 / a
            probs.append(raw_h / (raw_h + raw_a))
    if len(probs) < MIN_BOOKMAKERS:
        return None
    return sum(probs) / len(probs)


def build_odds_map(odds_data: dict) -> dict:
    """Build {(norm_home, norm_away): implied_prob} for today's games.
    Includes games stored under tomorrow's UTC date (late ET games shift by one day in UTC)."""
    from datetime import date as date_cls, timedelta
    eastern_now = pd.Timestamp.now(tz="US/Eastern")
    today_str = eastern_now.strftime("%Y-%m-%d")
    tomorrow_str = (date_cls.fromisoformat(today_str) + timedelta(days=1)).isoformat()

    result = {}
    for entry in odds_data.get("games", {}).values():
        if entry.get("date", "") not in (today_str, tomorrow_str):
            continue
        prob = no_vig_home_prob(entry.get("bookmakers", []))
        if prob is not None:
            key = (normalize(entry["home_team"]), normalize(entry["away_team"]))
            result[key] = prob
    return result


def inject():
    if not os.path.exists(ODDS_JSON):
        print("WARNUNG: web/odds.json nicht gefunden — Odds-Blend übersprungen.")
        return

    if not os.path.exists(PREDICTIONS_XLSX):
        print("WARNUNG: output/predictions.xlsx nicht gefunden — Odds-Blend übersprungen.")
        return

    with open(ODDS_JSON, encoding="utf-8") as f:
        odds_data = json.load(f)

    odds_map = build_odds_map(odds_data)
    if not odds_map:
        print("INFO: Keine heutigen Odds in odds.json gefunden — Blend übersprungen.")
        return

    with pd.ExcelFile(PREDICTIONS_XLSX) as xls:
        if "predictions_today" not in xls.sheet_names:
            print("WARNUNG: Sheet 'predictions_today' fehlt — Blend übersprungen.")
            return
        df = pd.read_excel(xls, "predictions_today")

    if df.empty or "probability_home_win" not in df.columns:
        print("INFO: Keine Vorhersagen heute — Blend übersprungen.")
        return

    matched = 0
    for idx, row in df.iterrows():
        home = normalize(str(row.get("Home Team", "")))
        away = normalize(str(row.get("Away Team", "")))
        market_prob = odds_map.get((home, away))
        if market_prob is None:
            continue
        # Write market prob as info column only — do NOT touch probability_home_win
        df.at[idx, "market_prob_home_win"] = round(market_prob, 4)
        matched += 1

    if matched == 0:
        print("INFO: Keine Übereinstimmung zwischen Predictions und Odds-Teams.")
        return

    # Write back — preserve other sheets
    with pd.ExcelFile(PREDICTIONS_XLSX) as xls:
        other_sheets = {s: pd.read_excel(xls, s) for s in xls.sheet_names if s != "predictions_today"}

    with pd.ExcelWriter(PREDICTIONS_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="predictions_today", index=False)
        for sheet_name, sheet_df in other_sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(
        f"OK: market_prob_home_win gesetzt für {matched}/{len(df)} Spiele "
        f"({len(odds_map)} Spiele in odds.json)"
    )


if __name__ == "__main__":
    inject()
