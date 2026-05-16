import json
import os
import pandas as pd
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PREDICTIONS_XLSX = os.path.join(BASE_DIR, "output", "predictions.xlsx")
ALL_PREDICTIONS_XLSX = os.path.join(BASE_DIR, "output", "all_predictions.xlsx")
MODEL_DATA_CSV = os.path.join(BASE_DIR, "data", "model_data.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "web", "predictions.json")


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


def build_json():
    elo_map = get_latest_elo()

    eastern_now = pd.Timestamp.now(tz="US/Eastern")
    today_naive = eastern_now.tz_localize(None).normalize()
    yesterday_naive = today_naive - pd.Timedelta(days=1)

    # --- TODAY ---
    today_games = []
    if os.path.exists(PREDICTIONS_XLSX):
        with pd.ExcelFile(PREDICTIONS_XLSX) as xls:
            if "predictions_today" in xls.sheet_names:
                df_today = pd.read_excel(xls, "predictions_today")
                for _, row in df_today.iterrows():
                    today_games.append(row_to_game(row, elo_map))

    # --- ALL + YESTERDAY ---
    all_games = []
    yesterday_games = []
    if os.path.exists(ALL_PREDICTIONS_XLSX):
        df_all = pd.read_excel(ALL_PREDICTIONS_XLSX)
        df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
        df_all = df_all.dropna(subset=["Date"])
        df_all = df_all.sort_values("Date", ascending=False)

        for _, row in df_all.iterrows():
            game = row_to_game(row, elo_map, include_result=True)
            all_games.append(game)
            if pd.Timestamp(row["Date"]).normalize() == yesterday_naive:
                yesterday_games.append(game)

    payload = {
        "generated_at": eastern_now.strftime("%Y-%m-%dT%H:%M:%S"),
        "today": today_games,
        "yesterday": yesterday_games,
        "all": all_games,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"OK: predictions.json geschrieben ({len(today_games)} heute, {len(yesterday_games)} gestern, {len(all_games)} gesamt)")


if __name__ == "__main__":
    build_json()
