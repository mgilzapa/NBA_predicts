import json
import os
import requests
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "web", "odds.json")
API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


def normalize_team(name: str) -> str:
    return name.strip().lower()


def make_key(date: str, home_team: str, away_team: str) -> str:
    return f"{date}|{normalize_team(home_team)}|{normalize_team(away_team)}"


def parse_bookmakers(api_game: dict) -> list:
    home = api_game["home_team"]
    result = []
    for bm in api_game.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market["key"] != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
            home_price = outcomes.get(home)
            away_prices = [v for k, v in outcomes.items() if k != home]
            away_price = away_prices[0] if away_prices else None
            if home_price and away_price:
                result.append({
                    "name": bm["title"],
                    "home": round(home_price, 3),
                    "away": round(away_price, 3),
                })
    return result


def fetch_and_save(api_key: str, output_path: str) -> dict:
    resp = requests.get(API_URL, params={
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }, timeout=15)
    resp.raise_for_status()

    games = {}
    for g in resp.json():
        commence = g.get("commence_time", "")
        date = commence[:10] if commence else ""
        key = make_key(date, g["home_team"], g["away_team"])
        games[key] = {
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "date": date,
            "bookmakers": parse_bookmakers(g),
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "games": games,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


if __name__ == "__main__":
    api_key = os.environ["ODDS_API_KEY"]
    result = fetch_and_save(api_key, OUTPUT_PATH)
    print(f"OK: odds.json geschrieben ({len(result['games'])} Spiele)")
