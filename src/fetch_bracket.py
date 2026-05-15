"""fetch_bracket.py — builds web/bracket.json from existing playoff game data."""
import json
import os

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_CSV       = os.path.join(BASE_DIR, "data", "nba_api_games.csv")
MODEL_DATA_CSV  = os.path.join(BASE_DIR, "data", "model_data.csv")
MODEL_PKL       = os.path.join(BASE_DIR, "models", "best_xgb_model.pkl")
FEATURE_CSV     = os.path.join(BASE_DIR, "models", "feature_cols.csv")
OUTPUT_JSON     = os.path.join(BASE_DIR, "web", "bracket.json")

SEASON_CODE = "42500"   # prefix of 2025-26 playoff game IDs
SEASON      = "2025-26"

# Games 1,2,5,7 are hosted by the series home team (higher seed); 3,4,6 by away
SERIES_HOME_GAMES = {1, 2, 5, 7}

# Upper bound of matchup index for East conference per round
EAST_BOUNDARY = {1: 4, 2: 2, 3: 1}

# R2 matchup index → which two R1 matchup indices produced it
R1_TO_R2 = {
    0: (0, 3),   # East R2 matchup 0: winner of R1 matchup 0 vs winner of R1 matchup 3
    1: (1, 2),   # East R2 matchup 1: winner of R1 matchup 1 vs winner of R1 matchup 2
    2: (4, 7),   # West R2 matchup 2: winner of R1 matchup 4 vs winner of R1 matchup 7
    3: (5, 6),   # West R2 matchup 3: winner of R1 matchup 5 vs winner of R1 matchup 6
}

# R3 matchup index → which two R2 matchup indices produced it
R2_TO_R3 = {
    0: (0, 1),   # East Conf Finals: winner of R2 matchup 0 vs 1
    1: (2, 3),   # West Conf Finals: winner of R2 matchup 2 vs 3
}


def parse_game_id(game_id):
    """Return (round, matchup, game_num) from game_id like 42500131."""
    s = str(game_id)
    return int(s[5]), int(s[6]), int(s[7])


def get_conference(round_num, matchup):
    """Return 'east', 'west', or 'finals' for a (round, matchup) pair."""
    if round_num == 4:
        return 'finals'
    return 'east' if matchup < EAST_BOUNDARY[round_num] else 'west'


def simulate_series(home_wins, away_wins, p1, p2):
    """
    Enumerate all remaining game paths in a best-of-7 series.

    p1: P(series home team wins) when series home team is hosting (games 1,2,5,7)
    p2: P(series home team wins) when series away team is hosting (games 3,4,6)

    Returns (p_home_wins_series, expected_total_games_rounded).
    If the series is already decided, returns immediately with 0 remaining games.
    """
    if home_wins == 4:
        return 1.0, home_wins + away_wins
    if away_wins == 4:
        return 0.0, home_wins + away_wins

    total_p_home = 0.0
    total_exp_games = 0.0

    def dfs(h, a, prob):
        nonlocal total_p_home, total_exp_games
        if h == 4 or a == 4:
            total_p_home += prob * (1.0 if h == 4 else 0.0)
            total_exp_games += prob * (h + a)
            return
        game_num = h + a + 1
        p = p1 if game_num in SERIES_HOME_GAMES else p2
        dfs(h + 1, a, prob * p)
        dfs(h, a + 1, prob * (1.0 - p))

    dfs(home_wins, away_wins, 1.0)
    return total_p_home, round(total_exp_games)
