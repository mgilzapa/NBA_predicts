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


def build_series_dict(df_playoff):
    """
    Group playoff game rows into series objects keyed by (round, matchup).

    Each value is:
      {'home_team', 'away_team', 'home_wins', 'away_wins', 'status', 'winner'}

    home_team / away_team are determined from game 1 of each series (game_num == 1).
    """
    raw = {}   # (round, matchup) -> {series_home, series_away, win_counts}
    for _, row in df_playoff.iterrows():
        gid = int(row['GAME_ID'])
        rnd, matchup, game_num = parse_game_id(gid)
        key = (rnd, matchup)

        if key not in raw:
            raw[key] = {'series_home': None, 'series_away': None, 'wins': {}}

        entry = raw[key]
        game_host  = row['hometeamName']
        game_visit = row['awayteamName']
        winner     = game_host if row['home_win'] else game_visit

        # Game 1 is always hosted by the series home team (higher seed)
        if game_num == 1:
            entry['series_home'] = game_host
            entry['series_away'] = game_visit

        entry['wins'][winner] = entry['wins'].get(winner, 0) + 1

    result = {}
    for key, entry in raw.items():
        home = entry['series_home']
        away = entry['series_away']
        # Fall back to most-frequent home team if game 1 is missing
        if home is None:
            teams = list(entry['wins'].keys())
            home = teams[0] if teams else 'TBD'
            away = teams[1] if len(teams) > 1 else 'TBD'

        hw = entry['wins'].get(home, 0)
        aw = entry['wins'].get(away, 0)

        if hw == 4:
            status, winner = 'complete', home
        elif aw == 4:
            status, winner = 'complete', away
        elif hw + aw == 0:
            status, winner = 'upcoming', None
        else:
            status, winner = 'active', None

        result[key] = {
            'home_team':  home,
            'away_team':  away,
            'home_wins':  hw,
            'away_wins':  aw,
            'status':     status,
            'winner':     winner,
        }
    return result


EXCLUDE_COLS = {
    "home_off_rating", "away_off_rating", "off_rating_diff",
    "home_def_rating", "away_def_rating", "def_rating_diff",
    "home_net_rating", "away_net_rating", "net_rating_diff",
    "home_h2h_winrate", "away_h2h_winrate", "h2h_winrate_diff",
    "same_division", "is_playoff", "away_opponent_strength",
}

DERIVED_PAIRS = [
    ("winrate_diff",                "home_last5_winrate",       "away_last5_winrate"),
    ("winrate_trend_diff",          "home_winrate_trend",       "away_winrate_trend"),
    ("average_points_diff",         "home_last5_avg_points",    "away_last5_avg_points"),
    ("average_points_allowed_diff", "home_last5_avg_points_allowed", "away_last5_avg_points_allowed"),
    ("rest_days_diff",              "home_rest_days",           "away_rest_days"),
    ("pts_diff_last5",              "home_last5_pts",           "away_last5_pts"),
    ("reb_diff_last5",              "home_last5_rebounds",      "away_last5_rebounds"),
    ("ast_diff_last5",              "home_last5_ast",           "away_last5_ast"),
    ("min_diff_last5",              "home_last5_min",           "away_last5_min"),
    ("player_count_diff_last5",     "home_last5_player_count",  "away_last5_player_count"),
    ("elo_diff",                    "home_elo",                 "away_elo"),
]


def get_team_snapshots(df_model):
    """
    Return (home_snap, away_snap): dicts mapping team_name → latest feature values.

    home_snap[team] = {'home_last5_winrate': ..., 'home_elo': ..., ...}
    away_snap[team] = {'away_last5_winrate': ..., 'away_elo': ..., ...}
    """
    df = df_model.copy()
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])

    home_cols = [c for c in df.columns if c.startswith("home_")]
    away_cols = [c for c in df.columns if c.startswith("away_")]

    def latest(df_grp, team_col, feat_cols):
        snap = {}
        for team, grp in df_grp.groupby(team_col):
            grp = grp.sort_values("gameDateTimeEst")
            row = {}
            for col in feat_cols:
                non_null = grp[col].dropna()
                if non_null.empty:
                    row[col] = np.nan
                else:
                    try:
                        row[col] = float(non_null.iloc[-1])
                    except (ValueError, TypeError):
                        row[col] = np.nan
            snap[team] = row
        return snap

    # Build home snapshot from home-role games + away-role games (columns renamed)
    away_to_home = {c: c.replace("away_", "home_", 1) for c in away_cols}
    df_h1 = df[["gameDateTimeEst", "hometeamName"] + home_cols].rename(columns={"hometeamName": "_team"})
    df_h2 = df[["gameDateTimeEst", "awayteamName"] + away_cols].rename(
        columns={"awayteamName": "_team", **away_to_home})
    df_home_all = pd.concat([df_h1, df_h2], ignore_index=True)
    home_snap = latest(df_home_all, "_team", home_cols)

    # Build away snapshot from away-role games + home-role games (columns renamed)
    home_to_away = {c: c.replace("home_", "away_", 1) for c in home_cols}
    df_a1 = df[["gameDateTimeEst", "awayteamName"] + away_cols].rename(columns={"awayteamName": "_team"})
    df_a2 = df[["gameDateTimeEst", "hometeamName"] + home_cols].rename(
        columns={"hometeamName": "_team", **home_to_away})
    df_away_all = pd.concat([df_a1, df_a2], ignore_index=True)
    away_snap = latest(df_away_all, "_team", away_cols)

    return home_snap, away_snap


def build_feature_row(home_team, away_team, home_snap, away_snap, feature_cols):
    """
    Build a single feature vector (numpy array, length = len(feature_cols))
    for a game where `home_team` is hosting `away_team`.
    """
    h = home_snap.get(home_team, {})
    a = away_snap.get(away_team, {})
    merged = {**h, **a}
    merged["is_playoff"] = 1.0

    # Compute all derived difference features
    for diff_col, home_col, away_col in DERIVED_PAIRS:
        if home_col in merged and away_col in merged:
            merged[diff_col] = merged[home_col] - merged[away_col]

    return np.array([merged.get(c, np.nan) for c in feature_cols], dtype=float)


def predict_series(home_team, away_team, home_snap, away_snap, model, feature_cols):
    """
    Predict win probability and expected length for a full series.

    Returns (p_home_wins_series, expected_games) starting from 0-0.
    Uses simulate_series with home_wins=0, away_wins=0.
    """
    # p1: P(series home team wins) when they host (games 1,2,5,7)
    row_home_hosts = build_feature_row(home_team, away_team, home_snap, away_snap, feature_cols)
    p1 = float(model.predict_proba(row_home_hosts.reshape(1, -1))[0, 1])

    # p2: P(series home team wins) when away team hosts (games 3,4,6)
    row_away_hosts = build_feature_row(away_team, home_team, home_snap, away_snap, feature_cols)
    p_away_wins_away_game = float(model.predict_proba(row_away_hosts.reshape(1, -1))[0, 1])
    p2 = 1.0 - p_away_wins_away_game

    return simulate_series(0, 0, p1, p2)


def _series_obj(home_team, away_team, home_wins, away_wins, status, winner,
                predict_fn):
    """
    Build the series dict for bracket.json.
    predict_fn(home, away) → (prob, exp_games) for 0-0 start.
    """
    if status == 'tbd':
        return {
            'home_team': home_team or 'TBD',
            'away_team': away_team or 'TBD',
            'home_wins': 0, 'away_wins': 0,
            'status': 'tbd', 'winner': None,
            'prediction': None,
        }

    prob, exp_games = predict_fn(home_team, away_team) if predict_fn else (0.5, 6)
    pred_winner = home_team if prob >= 0.5 else away_team
    pred_prob   = prob if prob >= 0.5 else 1.0 - prob

    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_wins': home_wins,
        'away_wins': away_wins,
        'status':    status,
        'winner':    winner,
        'prediction': {
            'winner':           pred_winner,
            'win_probability':  round(pred_prob, 4),
            'predicted_length': exp_games,
        },
    }


def _effective_winner(series_map, key, predict_fn):
    """
    Return the team expected to advance from a series.
    If the series hasn't started yet (not in series_map), derive teams
    recursively from the previous round and predict from there.
    """
    s = series_map.get(key)
    if s is not None:
        if s['status'] == 'complete':
            return s['winner']
        prob, _ = predict_fn(s['home_team'], s['away_team']) if predict_fn else (0.5, 6)
        return s['home_team'] if prob >= 0.5 else s['away_team']

    # Series not started yet — derive teams from previous round recursively
    rnd, matchup = key
    if rnd == 2:
        r1a, r1b = R1_TO_R2[matchup]
        home = _effective_winner(series_map, (1, r1a), predict_fn)
        away = _effective_winner(series_map, (1, r1b), predict_fn)
    elif rnd == 3:
        r2a, r2b = R2_TO_R3[matchup]
        home = _effective_winner(series_map, (2, r2a), predict_fn)
        away = _effective_winner(series_map, (2, r2b), predict_fn)
    elif rnd == 4:
        home = _effective_winner(series_map, (3, 0), predict_fn)
        away = _effective_winner(series_map, (3, 1), predict_fn)
    else:
        return None

    if home and away and predict_fn:
        prob, _ = predict_fn(home, away)
        return home if prob >= 0.5 else away
    return home


def build_bracket_json(series_map, predict_fn=None):
    """
    Build the full bracket dict from a series_map keyed by (round, matchup).
    Fills TBD entries for future rounds using predicted winners.
    """
    def get_series(rnd, matchup):
        key = (rnd, matchup)
        if key in series_map:
            s = series_map[key]
            return _series_obj(
                s['home_team'], s['away_team'],
                s['home_wins'], s['away_wins'],
                s['status'], s['winner'],
                predict_fn,
            )
        # Series not started: determine teams from previous round if possible
        if rnd == 2:
            r1a, r1b = R1_TO_R2[matchup]
            home = _effective_winner(series_map, (1, r1a), predict_fn)
            away = _effective_winner(series_map, (1, r1b), predict_fn)
        elif rnd == 3:
            r2a, r2b = R2_TO_R3[matchup]
            home = _effective_winner(series_map, (2, r2a), predict_fn)
            away = _effective_winner(series_map, (2, r2b), predict_fn)
        elif rnd == 4:
            home = _effective_winner(series_map, (3, 0), predict_fn)
            away = _effective_winner(series_map, (3, 1), predict_fn)
        else:
            home, away = None, None

        status = 'upcoming' if (home and away) else 'tbd'
        if home and away and predict_fn:
            return _series_obj(home, away, 0, 0, status, None, predict_fn)
        return _series_obj(home, away, 0, 0, 'tbd', None, None)

    # East/West each have 4 R1, 2 R2, 1 R3 series
    east_r1 = [get_series(1, m) for m in range(4)]
    east_r2 = [get_series(2, m) for m in range(2)]
    east_r3 = [get_series(3, 0)]
    east_finalist = _effective_winner(series_map, (3, 0), predict_fn)

    west_r1 = [get_series(1, m) for m in range(4, 8)]
    west_r2 = [get_series(2, m) for m in range(2, 4)]
    west_r3 = [get_series(3, 1)]
    west_finalist = _effective_winner(series_map, (3, 1), predict_fn)

    finals = get_series(4, 0)

    return {
        'generated_at': pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%dT%H:%M:%S'),
        'season': SEASON,
        'east': {
            'r1':       east_r1,
            'r2':       east_r2,
            'r3':       east_r3,
            'finalist': east_finalist,
        },
        'west': {
            'r1':       west_r1,
            'r2':       west_r2,
            'r3':       west_r3,
            'finalist': west_finalist,
        },
        'finals': finals,
    }


def build_bracket():
    """Main entry point: load data, run predictions, write bracket.json."""
    # 1. Load playoff games
    df_all = pd.read_csv(GAMES_CSV)
    playoff_mask = df_all['GAME_ID'].astype(str).str.startswith(SEASON_CODE)
    df_playoff = df_all[playoff_mask].copy()

    # 2. Build series standings
    series_map = build_series_dict(df_playoff)

    # 3. Load model and features
    model        = joblib.load(MODEL_PKL)
    feature_cols = pd.read_csv(FEATURE_CSV).squeeze().tolist()
    feature_cols = [c for c in feature_cols if c not in EXCLUDE_COLS]

    # 4. Load model data and build team snapshots
    df_model     = pd.read_csv(MODEL_DATA_CSV)
    home_snap, away_snap = get_team_snapshots(df_model)

    def predict_fn(home_team, away_team):
        return predict_series(home_team, away_team, home_snap, away_snap,
                              model, feature_cols)

    # 5. Assemble bracket JSON
    bracket = build_bracket_json(series_map, predict_fn)

    # 6. Write output
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(bracket, f, ensure_ascii=False, indent=2)

    total_series = (
        len(bracket['east']['r1']) + len(bracket['east']['r2']) +
        len(bracket['east']['r3']) + len(bracket['west']['r1']) +
        len(bracket['west']['r2']) + len(bracket['west']['r3']) + 1
    )
    print(f"OK: bracket.json written ({total_series} series, "
          f"{len(series_map)} with game data)")


if __name__ == '__main__':
    build_bracket()
