import json
import os
import requests
from datetime import datetime, timedelta, timezone

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
            if home_price is not None and away_price is not None:
                result.append({
                    "name": bm["title"],
                    "home": round(home_price, 3),
                    "away": round(away_price, 3),
                })
                break  # only take the first h2h market per bookmaker
    return result


def fetch_and_save(api_key: str, output_path: str) -> dict:
    # Load existing accumulated odds
    existing_games = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                existing_games = json.load(f).get("games", {})
        except (json.JSONDecodeError, KeyError):
            pass

    resp = requests.get(API_URL, params={
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }, timeout=15)
    resp.raise_for_status()

    new_games = {}
    for g in resp.json():
        commence = g.get("commence_time", "")
        date = commence[:10] if commence else ""
        key = make_key(date, g["home_team"], g["away_team"])
        new_games[key] = {
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "date": date,
            "bookmakers": parse_bookmakers(g),
        }

    # Merge: new data overwrites same-day entries, old entries are kept
    merged = {**existing_games, **new_games}

    # Prune entries older than 14 days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
    merged = {k: v for k, v in merged.items() if v.get("date", "") >= cutoff}

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "games": merged,
    }

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


if __name__ == "__main__":
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("WARNUNG: ODDS_API_KEY nicht gesetzt, Quoten werden nicht aktualisiert.")
        exit(0)
    result = fetch_and_save(api_key, OUTPUT_PATH)
    print(f"OK: odds.json geschrieben ({len(result['games'])} Spiele)")
