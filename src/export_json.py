import json
import os
import pandas as pd
from datetime import date as date_cls, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PREDICTIONS_XLSX = os.path.join(BASE_DIR, "output", "predictions.xlsx")
ALL_PREDICTIONS_XLSX = os.path.join(BASE_DIR, "output", "all_predictions.xlsx")
MODEL_DATA_CSV = os.path.join(BASE_DIR, "data", "model_data.csv")
ODDS_JSON = os.path.join(BASE_DIR, "web", "odds.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "web", "predictions.json")


def _next_date_str(date_str: str) -> str:
    try:
        return (date_cls.fromisoformat(date_str) + timedelta(days=1)).isoformat()
    except Exception:
        return ""


def load_odds_map() -> dict:
    """Return odds.json games dict, keyed by 'YYYY-MM-DD|home_lower|away_lower'."""
    if not os.path.exists(ODDS_JSON):
        return {}
    try:
        with open(ODDS_JSON, encoding="utf-8") as f:
            return json.load(f).get("games", {})
    except Exception:
        return {}


def calc_bet_value(game: dict, odds_map: dict) -> tuple:
    """
    Returns (ev, best_decimal_odds, bookmaker_name) for the predicted winner,
    or (None, None, None) if no odds are available.

    EV = model_probability × best_decimal_odds − 1
    Positive = value bet; negative = no value.
    """
    date = game.get("date", "")
    home = game.get("home_team", "").lower().strip()
    away = game.get("away_team", "").lower().strip()
    predicted = game.get("predicted_winner", "")
    predicted_is_home = predicted == game.get("home_team", "")
    odds_field = "home" if predicted_is_home else "away"

    # Try game date first, then next day (odds API stores in UTC — late ET games shift)
    entry = odds_map.get(f"{date}|{home}|{away}") or \
            odds_map.get(f"{_next_date_str(date)}|{home}|{away}")
    if entry is None:
        return None, None, None

    bookmakers = [b for b in entry.get("bookmakers", []) if b.get(odds_field, 0) > 1.0]
    if not bookmakers:
        return None, None, None

    best_bm = max(bookmakers, key=lambda b: b[odds_field])
    best_odds = best_bm[odds_field]

    prob_home = game.get("probability_home_win", 0.5)
    model_prob = prob_home if predicted_is_home else (1.0 - prob_home)

    ev = model_prob * best_odds - 1.0
    return round(ev, 4), round(best_odds, 2), best_bm.get("name", "")


def get_latest_elo():
    if not os.path.exists(MODEL_DATA_CSV):
        return {}
    df = pd.read_csv(MODEL_DATA_CSV, usecols=["gameDateTimeEst", "hometeamName", "awayteamName", "home_elo", "away_elo"])
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")
    df = df.dropna(subset=["gameDateTimeEst"])

    rows = []
    for _, r in df.iterrows():
        rows.append({"team": r["hometeamName"], "elo": r["home_elo"], "dt": r["gameDateTimeEst"]})
        rows.append({"team": r["awayteamName"], "elo": r["away_elo"], "dt": r["gameDateTimeEst"]})

    elo_df = pd.DataFrame(rows).dropna()
    latest = elo_df.sort_values("dt").groupby("team")["elo"].last()
    return latest.to_dict()


def row_to_game(row, elo_map, include_result=False):
    date_val = row.get("Date") or row.get("date", "")
    if pd.isnull(date_val):
        date_str = ""
        time_str = ""
    else:
        dt = pd.to_datetime(date_val)
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M")

    home = str(row.get("Home Team", ""))
    away = str(row.get("Away Team", ""))
    predicted = str(row.get("Predicted Winner", ""))
    prob_home = float(row.get("probability_home_win", 0.5))

    game_id_raw = row.get("gameId", "")
    if pd.isnull(game_id_raw):
        game_id = ""
    else:
        game_id = str(int(float(game_id_raw))) if game_id_raw != "" else ""

    entry = {
        "gameId": game_id,
        "date": date_str,
        "time": time_str,
        "home_team": home,
        "away_team": away,
        "predicted_winner": predicted,
        "probability_home_win": round(prob_home, 4),
        "home_elo": round(elo_map.get(home, 0), 1),
        "away_elo": round(elo_map.get(away, 0), 1),
    }

    if include_result:
        actual_raw = row.get("Actual Winner", None)
        if pd.isnull(actual_raw) if actual_raw is not None else True:
            actual = None
            correct = None
        else:
            actual = str(actual_raw)
            correct = predicted == actual
        entry["actual_winner"] = actual
        entry["correct"] = correct

    return entry


def _game_key(g):
    """Stable dedup key: gameId preferred, else date|home|away."""
    gid = g.get("gameId") or ""
    if gid:
        return gid
    return f"{g.get('date','')}|{g.get('home_team','')}|{g.get('away_team','')}"


def build_json():
    elo_map = get_latest_elo()

    eastern_now = pd.Timestamp.now(tz="US/Eastern")
    today_naive = eastern_now.tz_localize(None).normalize()
    yesterday_naive = today_naive - pd.Timedelta(days=1)

    # ── 1. Load existing predictions.json as the base (append-only) ──────────
    # predictions.json is the source of truth for "all" — we never delete entries.
    existing_all: dict[str, dict] = {}  # key → game dict
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, encoding="utf-8") as f:
                old = json.load(f)
            for g in old.get("all", []):
                existing_all[_game_key(g)] = g
        except Exception:
            pass  # corrupt file — start fresh, existing entries lost (unavoidable)

    # ── 2. TODAY from predictions.xlsx ───────────────────────────────────────
    odds_map = load_odds_map()
    today_games = []
    if os.path.exists(PREDICTIONS_XLSX):
        with pd.ExcelFile(PREDICTIONS_XLSX) as xls:
            if "predictions_today" in xls.sheet_names:
                df_today = pd.read_excel(xls, "predictions_today")
                for _, row in df_today.iterrows():
                    g = row_to_game(row, elo_map)
                    ev, best_odds, best_bm = calc_bet_value(g, odds_map)
                    g["bet_value"] = ev
                    g["best_odds"] = best_odds
                    g["best_bookmaker"] = best_bm
                    today_games.append(g)

    # ── 3. Updates from all_predictions.xlsx (actual_winner / correct) ───────
    # This file holds the ground truth once games are played. We use it ONLY to
    # update actual_winner/correct on existing entries — never to delete them.
    updates: dict[str, dict] = {}  # key → {actual_winner, correct}
    if os.path.exists(ALL_PREDICTIONS_XLSX):
        df_all = pd.read_excel(ALL_PREDICTIONS_XLSX)
        df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
        df_all = df_all.dropna(subset=["Date"])
        for _, row in df_all.iterrows():
            g = row_to_game(row, elo_map, include_result=True)
            k = _game_key(g)
            updates[k] = {"actual_winner": g["actual_winner"], "correct": g["correct"]}

    # ── 4. Merge: existing + today + updates ─────────────────────────────────
    merged: dict[str, dict] = dict(existing_all)  # start from persisted state

    # Add today's games (new entries only; never overwrite existing prediction data)
    for g in today_games:
        k = _game_key(g)
        if k not in merged:
            merged[k] = {**g, "actual_winner": None, "correct": None}

    # Apply actual_winner / correct updates from all_predictions.xlsx
    for k, upd in updates.items():
        if k in merged:
            # Only update if we now have a real result (don't overwrite with None)
            if upd["actual_winner"] is not None:
                merged[k]["actual_winner"] = upd["actual_winner"]
                merged[k]["correct"] = upd["correct"]
        else:
            # Entry exists in all_predictions.xlsx but not yet in predictions.json
            # (e.g., first run after a gap) — add it
            g = updates[k]
            # Re-build full entry from all_predictions.xlsx
            pass  # handled below via df_all second pass

    # Second pass: add any all_predictions.xlsx entries not yet in merged
    if os.path.exists(ALL_PREDICTIONS_XLSX):
        df_all2 = pd.read_excel(ALL_PREDICTIONS_XLSX)
        df_all2["Date"] = pd.to_datetime(df_all2["Date"], errors="coerce")
        df_all2 = df_all2.dropna(subset=["Date"])
        for _, row in df_all2.iterrows():
            g = row_to_game(row, elo_map, include_result=True)
            k = _game_key(g)
            if k not in merged:
                merged[k] = g

    # ── 5. Sort newest-first ─────────────────────────────────────────────────
    all_games = sorted(merged.values(), key=lambda g: g.get("date", ""), reverse=True)

    # ── 6. Yesterday slice ───────────────────────────────────────────────────
    yesterday_str = yesterday_naive.strftime("%Y-%m-%d")
    yesterday_games = [g for g in all_games if g.get("date") == yesterday_str]

    payload = {
        "generated_at": eastern_now.strftime("%Y-%m-%dT%H:%M:%S"),
        "today": today_games,
        "yesterday": yesterday_games,
        "all": all_games,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    new_count = len(all_games) - len(existing_all)
    print(f"OK: predictions.json geschrieben ({len(today_games)} heute, {len(yesterday_games)} gestern, {len(all_games)} gesamt, +{max(new_count,0)} neu)")


if __name__ == "__main__":
    build_json()
