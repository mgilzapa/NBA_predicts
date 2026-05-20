"""playoff_feature_engineering.py

Computes Category A+B playoff features for today's scheduled playoff games.
Writes data/playoff_game_features.csv (one row per gameId).

Category A — series context:
  series_game_number, series_score_diff, is_closeout_game

Category B — playoff rolling stats (home_ / away_ prefixed):
  playoff_last3_pts, playoff_margin_last3, playoff_games_played,
  prev_series_games, playoff_home_winrate_hist

Category C (base_prob_home_win, market_prob_home_win) is added by predict.py
at prediction time.

Run as step 6c in run_all.py (playoffs only).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR              = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEDULE_CSV          = os.path.join(BASE_DIR, "data", "schedule_round_1.csv")
PLAYOFF_STATS_CSV     = os.path.join(BASE_DIR, "data", "playoff_stats.csv")
BRACKET_JSON          = os.path.join(BASE_DIR, "web", "bracket.json")
OUTPUT_CSV            = os.path.join(BASE_DIR, "data", "playoff_game_features.csv")

TEAM_NAME_MAPPING = {
    "76ers": "Philadelphia 76ers", "Bucks": "Milwaukee Bucks",
    "Bulls": "Chicago Bulls", "Cavaliers": "Cleveland Cavaliers",
    "Celtics": "Boston Celtics", "Clippers": "LA Clippers",
    "Grizzlies": "Memphis Grizzlies", "Hawks": "Atlanta Hawks",
    "Heat": "Miami Heat", "Hornets": "Charlotte Hornets",
    "Jazz": "Utah Jazz", "Kings": "Sacramento Kings",
    "Knicks": "New York Knicks", "Lakers": "Los Angeles Lakers",
    "Magic": "Orlando Magic", "Mavericks": "Dallas Mavericks",
    "Nets": "Brooklyn Nets", "Nuggets": "Denver Nuggets",
    "Pacers": "Indiana Pacers", "Pelicans": "New Orleans Pelicans",
    "Pistons": "Detroit Pistons", "Raptors": "Toronto Raptors",
    "Rockets": "Houston Rockets", "Spurs": "San Antonio Spurs",
    "Suns": "Phoenix Suns", "Thunder": "Oklahoma City Thunder",
    "Timberwolves": "Minnesota Timberwolves",
    "Trail Blazers": "Portland Trail Blazers",
    "Warriors": "Golden State Warriors", "Wizards": "Washington Wizards",
    "Phoenix": "Phoenix Suns",
}


def _load_bracket_series(bracket):
    """Return a mapping frozenset({home, away}) -> series dict from bracket.json."""
    series_map = {}
    for conf in ("east", "west"):
        for rnd_val in bracket.get(conf, {}).values():
            if isinstance(rnd_val, list):
                for sr in rnd_val:
                    if sr.get("status") != "complete":
                        key = frozenset([sr["home_team"], sr["away_team"]])
                        series_map[key] = sr
    finals = bracket.get("finals")
    if isinstance(finals, dict) and finals.get("status") != "complete":
        key = frozenset([finals["home_team"], finals["away_team"]])
        series_map[key] = finals
    return series_map


def main():
    for path, label in [(SCHEDULE_CSV, "schedule_round_1.csv"),
                        (PLAYOFF_STATS_CSV, "playoff_stats.csv")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at {path}")
            return 1

    schedule = pd.read_csv(SCHEDULE_CSV)
    playoff_stats = pd.read_csv(PLAYOFF_STATS_CSV)

    # Map short team names → full names
    for col in ("homeTeamName", "awayTeamName"):
        if col in schedule.columns:
            schedule[col] = schedule[col].map(TEAM_NAME_MAPPING).fillna(schedule[col])

    schedule["gameDateTimeEst"] = pd.to_datetime(
        schedule["gameDateTimeEst"], errors="coerce"
    )
    schedule = schedule.dropna(subset=["gameDateTimeEst"])

    # Build series map from bracket.json
    series_map = {}
    if os.path.exists(BRACKET_JSON):
        with open(BRACKET_JSON) as f:
            bracket = json.load(f)
        series_map = _load_bracket_series(bracket)

    # Build playoff stats lookup keyed by team_name
    stats_idx = playoff_stats.set_index("team_name")

    def _get_stat(team, col):
        try:
            return float(stats_idx.at[team, col])
        except (KeyError, ValueError):
            return np.nan

    rows = []
    home_col = "homeTeamName" if "homeTeamName" in schedule.columns else "hometeamName"
    away_col = "awayTeamName" if "awayTeamName" in schedule.columns else "awayteamName"

    for _, row in schedule.iterrows():
        home = str(row.get(home_col, ""))
        away = str(row.get(away_col, ""))
        game_id = row.get("gameId", np.nan)

        # Series context from bracket.json
        key = frozenset([home, away])
        sr = series_map.get(key, {})
        hw = int(sr.get("home_wins", 0))
        aw = int(sr.get("away_wins", 0))
        if sr.get("home_team") == home:
            home_series_wins, away_series_wins = hw, aw
        else:
            home_series_wins, away_series_wins = aw, hw

        # Parse game number from schedule (e.g. "Game 2" → 2)
        sgn_raw = str(row.get("seriesGameNumber", ""))
        try:
            series_game_number = int(sgn_raw.strip().split()[-1])
        except (ValueError, IndexError):
            series_game_number = home_series_wins + away_series_wins + 1

        series_score_diff = home_series_wins - away_series_wins
        is_closeout_game = int(home_series_wins == 3 or away_series_wins == 3)

        rows.append({
            "gameId": game_id,
            "series_game_number": series_game_number,
            "series_score_diff": series_score_diff,
            "is_closeout_game": is_closeout_game,
            "home_playoff_last3_pts": _get_stat(home, "playoff_pts_last3"),
            "away_playoff_last3_pts": _get_stat(away, "playoff_pts_last3"),
            "home_playoff_margin_last3": _get_stat(home, "playoff_margin_last3"),
            "away_playoff_margin_last3": _get_stat(away, "playoff_margin_last3"),
            "home_playoff_games_played": _get_stat(home, "playoff_games_played"),
            "away_playoff_games_played": _get_stat(away, "playoff_games_played"),
            "home_prev_series_games": _get_stat(home, "prev_series_games"),
            "away_prev_series_games": _get_stat(away, "prev_series_games"),
            "home_playoff_home_winrate_hist": _get_stat(home, "playoff_home_winrate_hist"),
        })

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"OK: playoff_game_features.csv written ({len(result)} games)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
