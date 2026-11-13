"""fetch_bracket.py — builds web/bracket.json (season projection + playoff bracket).

Modes (detected automatically from the data):
  projected            regular season running → full Monte-Carlo projection (season_simulator.py)
  post_season_pending  regular season over, no play-in yet → only play-in + playoffs simulated
  live                 real play-in/playoff games exist → real series (XGBoost, as before),
                       later rounds and round odds via ELO simulation
"""
import json
import os
import shutil
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_CSV      = os.path.join(BASE_DIR, "data",   "nba_api_games.csv")
PLAYIN_CSV     = os.path.join(BASE_DIR, "data",   "nba_api_playin.csv")
SCHEDULE_CSV   = os.path.join(BASE_DIR, "data",   "schedule_round_1.csv")
PREDICTIONS_XLSX = os.path.join(BASE_DIR, "output", "predictions.xlsx")
MODEL_DATA_CSV = os.path.join(BASE_DIR, "data",   "model_data.csv")
MODEL_PKL      = os.path.join(BASE_DIR, "models", "best_xgb_model.pkl")
FEATURE_CSV    = os.path.join(BASE_DIR, "models", "feature_cols.csv")
OUTPUT_JSON    = os.path.join(BASE_DIR, "web",    "bracket.json")
HISTORY_JSON   = os.path.join(BASE_DIR, "web",    "projection_history.json")
SIM_REPORT_JSON = os.path.join(BASE_DIR, "output", "season_sim_report.json")
ARCHIVE_DIR    = os.path.join(BASE_DIR, "data",   "archive")

N_SIMS = 50_000


def _current_season():
    try:
        from scrape_upcoming_games import current_nba_season
        return current_nba_season()
    except Exception:
        now = pd.Timestamp.now(tz="America/New_York")
        start = now.year if now.month >= 7 else now.year - 1
        return f"{start}-{str(start + 1)[-2:]}"


SEASON      = _current_season()            # e.g. "2026-27"
SEASON_CODE = f"4{SEASON[2:4]}00"          # prefix of this season's playoff game IDs, e.g. "42600"

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

    Returns (p_home_wins_series, mode_total_games, outcome_probs).
    mode_total_games: total games in the single most likely outcome (mode, not expected value).
    outcome_probs: dict mapping "h-a" strings (e.g. "4-2") to their probability.
    """
    if home_wins == 4:
        key = f"{home_wins}-{away_wins}"
        return 1.0, home_wins + away_wins, {key: 1.0}
    if away_wins == 4:
        key = f"{home_wins}-{away_wins}"
        return 0.0, home_wins + away_wins, {key: 1.0}

    total_p_home = 0.0
    outcome_probs = {}

    def dfs(h, a, prob):
        nonlocal total_p_home
        if h == 4 or a == 4:
            total_p_home += prob * (1.0 if h == 4 else 0.0)
            key = f"{h}-{a}"
            outcome_probs[key] = outcome_probs.get(key, 0.0) + prob
            return
        game_num = h + a + 1
        p = p1 if game_num in SERIES_HOME_GAMES else p2
        dfs(h + 1, a, prob * p)
        dfs(h, a + 1, prob * (1.0 - p))

    dfs(home_wins, away_wins, 1.0)

    best_key = max(outcome_probs, key=outcome_probs.get)
    mode_games = sum(int(x) for x in best_key.split("-"))
    return total_p_home, mode_games, outcome_probs


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
    "same_division", "away_opponent_strength",
    "home_injury_impact", "away_injury_impact", "injury_impact_diff",
    "home_playoff_exp", "away_playoff_exp", "playoff_exp_diff",
    "h2h_winrate_diff", "home_series_wins",
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
    ("last10_winrate_diff",         "home_last10_winrate",      "away_last10_winrate"),
    ("streak_diff",                 "home_current_streak",      "away_current_streak"),
    ("h2h_winrate_diff",            "home_h2h_winrate",         "away_h2h_winrate"),
    ("series_wins_diff",            "home_series_wins",         "away_series_wins"),
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


def build_feature_row(home_team, away_team, home_snap, away_snap, feature_cols, context=None):
    """
    Build a single feature vector (numpy array, length = len(feature_cols))
    for a game where `home_team` is hosting `away_team`.

    context: optional dict of game-context values (e.g. is_playoff=0, rest days,
    ELO) that override the snapshot/playoff defaults before diffs are derived.
    Without it the behaviour is unchanged (playoff game context).
    """
    h = home_snap.get(home_team, {})
    a = away_snap.get(away_team, {})
    merged = {**h, **a}
    merged["is_playoff"] = 1.0
    if context:
        merged.update(context)

    # Game-context defaults for features with no home_/away_ source columns.
    # Playoffs = end of season; series starts 0-0 so no leader and no elimination yet.
    merged.setdefault("season_progress", 1.0)
    merged.setdefault("home_series_wins", 0.0)
    merged.setdefault("away_series_wins", 0.0)
    merged.setdefault("series_wins_diff", 0.0)
    merged.setdefault("is_elimination_game", 0.0)
    merged.setdefault("market_prob_home_win", np.nan)  # no live odds for bracket games

    # Compute all derived difference features
    for diff_col, home_col, away_col in DERIVED_PAIRS:
        if home_col in merged and away_col in merged:
            merged[diff_col] = merged[home_col] - merged[away_col]

    return np.array([merged.get(c, np.nan) for c in feature_cols], dtype=float)


def predict_series(home_team, away_team, home_snap, away_snap, model, feature_cols,
                   cur_home_wins=0, cur_away_wins=0):
    """
    Predict win probability and series outcome distribution.

    Returns (p_home_wins_series, mode_total_games, outcome_probs).
    Starts simulation from (cur_home_wins, cur_away_wins) so active series
    are evaluated from their actual current state, not 0-0.
    """
    row_home_hosts = build_feature_row(home_team, away_team, home_snap, away_snap, feature_cols)
    row_away_hosts = build_feature_row(away_team, home_team, home_snap, away_snap, feature_cols)
    try:
        p1 = float(model.predict_proba(row_home_hosts.reshape(1, -1))[0, 1])
        p_away_wins_away_game = float(model.predict_proba(row_away_hosts.reshape(1, -1))[0, 1])
        p2 = 1.0 - p_away_wins_away_game
    except Exception as e:
        print(f"  WARNUNG: Modell-Vorhersage fehlgeschlagen ({home_team} vs {away_team}): {e}")
        return 0.5, 6, {}

    return simulate_series(cur_home_wins, cur_away_wins, p1, p2)


def _series_obj(home_team, away_team, home_wins, away_wins, status, winner,
                predict_fn):
    """
    Build the series dict for bracket.json.
    predict_fn(home, away, cur_hw, cur_aw) → (prob, mode_games, outcome_probs).
    Active series are evaluated from their current win state, not 0-0.
    """
    if status == 'tbd':
        return {
            'home_team': home_team or 'TBD',
            'away_team': away_team or 'TBD',
            'home_wins': 0, 'away_wins': 0,
            'status': 'tbd', 'winner': None,
            'prediction': None,
        }

    if predict_fn:
        # Completed series: predict from 0-0 to show what model expected pre-series.
        # Active/upcoming: predict from current state so remaining games are realistic.
        sim_hw = 0 if status == 'complete' else home_wins
        sim_aw = 0 if status == 'complete' else away_wins
        prob, mode_games, outcome_probs = predict_fn(home_team, away_team, sim_hw, sim_aw)
    else:
        prob, mode_games, outcome_probs = 0.5, 6, {}

    pred_winner = home_team if prob >= 0.5 else away_team
    pred_prob   = prob if prob >= 0.5 else 1.0 - prob

    # Most likely specific outcome consistent with the predicted winner.
    # In close series the overall mode can belong to the other team, so we filter
    # to only consider outcomes where the predicted winner actually wins.
    home_wins_series = prob >= 0.5
    winner_outcomes = {
        k: v for k, v in outcome_probs.items()
        if (k.startswith("4-")) == home_wins_series
    }
    best_score = (
        max(winner_outcomes, key=winner_outcomes.get) if winner_outcomes
        else max(outcome_probs, key=outcome_probs.get) if outcome_probs
        else ("4-2" if home_wins_series else "2-4")
    )

    # Total games in the predicted score (consistent with best_score)
    predicted_length = sum(int(x) for x in best_score.split("-"))

    # Top outcomes sorted by probability for frontend display
    top_outcomes = dict(
        sorted(outcome_probs.items(), key=lambda x: -x[1])[:5]
    )
    outcome_dist = {k: round(v, 3) for k, v in top_outcomes.items()}

    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_wins': home_wins,
        'away_wins': away_wins,
        'status':    status,
        'winner':    winner,
        'prediction': {
            'winner':               pred_winner,
            'win_probability':      round(pred_prob, 4),
            'predicted_length':     predicted_length,
            'predicted_score':      best_score,
            'outcome_distribution': outcome_dist,
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
        # Use actual wins for active series (same logic as _series_obj)
        sim_hw = s['home_wins'] if s['status'] == 'active' else 0
        sim_aw = s['away_wins'] if s['status'] == 'active' else 0
        prob, *_ = predict_fn(s['home_team'], s['away_team'], sim_hw, sim_aw) if predict_fn else (0.5, 6, {})
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
        prob, *_ = predict_fn(home, away)
        return home if prob >= 0.5 else away
    return home


def build_bracket_json(series_map, predict_fn=None, season=None):
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
        'season': season or SEASON,
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


def _read_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  WARNUNG: {os.path.basename(path)} nicht lesbar: {e}")
        return None


def detect_mode(df_all, season, schedule_df=None, playin_df=None):
    """projected | post_season_pending | live — rein aus den Daten abgeleitet."""
    import season_simulator as ss
    cur = df_all[df_all['season'] == season] if 'season' in df_all.columns else df_all.iloc[0:0]
    has_playoffs = (
        (cur['season_type'] == 'Playoffs').any() if 'season_type' in cur.columns else False
    ) or df_all['GAME_ID'].astype(str).str.startswith(f"4{season[2:4]}00").any()
    has_playin = playin_df is not None and not playin_df.empty and (
        (playin_df['season'] == season).any() if 'season' in playin_df.columns else True)
    if has_playoffs or has_playin:
        return 'live'
    played = ss.played_regular_season(df_all, season)
    remaining = ss.remaining_regular_season(schedule_df, played['GAME_ID'].tolist())
    if len(played) > 0 and remaining.empty:
        return 'post_season_pending'
    return 'projected'


def archive_previous_bracket(season, path=OUTPUT_JSON, archive_dir=ARCHIVE_DIR):
    """Alte Saison aus bracket.json nach data/archive/bracket_<saison>.json sichern."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            old_season = json.load(f).get('season')
    except Exception:
        return None
    if not old_season or old_season == season:
        return None
    os.makedirs(archive_dir, exist_ok=True)
    dest = os.path.join(archive_dir, f"bracket_{old_season}.json")
    if not os.path.exists(dest):          # erste Kopie behalten, nie ueberschreiben
        shutil.copy2(path, dest)
        print(f"  Archiviert: {dest}")
    return dest


def known_states(series_map, playin_df, season):
    """Echte Serienstaende / Play-In-Ergebnisse als Team-Index-Dicts fuer den Simulator."""
    import season_simulator as ss
    known_series = {}
    for s in series_map.values():
        h, a = s['home_team'], s['away_team']
        if h in ss.TEAM_IDX and a in ss.TEAM_IDX:
            hi, ai = ss.TEAM_IDX[h], ss.TEAM_IDX[a]
            known_series[frozenset((hi, ai))] = {hi: int(s['home_wins']), ai: int(s['away_wins'])}
    known_playin = {}
    if playin_df is not None and not playin_df.empty:
        df = playin_df[playin_df['season'] == season] if 'season' in playin_df.columns else playin_df
        for _, r in df.iterrows():
            h, a = r['hometeamName'], r['awayteamName']
            if h in ss.TEAM_IDX and a in ss.TEAM_IDX:
                hi, ai = ss.TEAM_IDX[h], ss.TEAM_IDX[a]
                known_playin[frozenset((hi, ai))] = hi if int(r['home_win']) == 1 else ai
    return known_series, known_playin


def _mark_live_series(bracket, series_map):
    """Live-Modus: team_probabilities + projected-Flag an die bestehenden Serienobjekte."""
    real_pairs = {frozenset((s['home_team'], s['away_team'])) for s in series_map.values()}

    def mark(s):
        if not s:
            return
        s['projected'] = frozenset((s.get('home_team'), s.get('away_team'))) not in real_pairs
        pred = s.get('prediction')
        if pred and pred.get('winner'):
            other = s['away_team'] if pred['winner'] == s['home_team'] else s['home_team']
            p = float(pred.get('win_probability', 0.5))
            pred['team_probabilities'] = {pred['winner']: round(p, 4), other: round(1 - p, 4)}

    for conf in ('east', 'west'):
        for rnd in ('r1', 'r2', 'r3'):
            for s in bracket[conf][rnd]:
                mark(s)
    mark(bracket['finals'])


def update_history(entry, path=HISTORY_JSON):
    """Ein Eintrag pro Tag; ein erneuter Lauf am selben Tag ueberschreibt den Tageseintrag."""
    history = []
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                history = json.load(f)
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []
    history = [h for h in history if h.get('date') != entry['date']]
    history.append(entry)
    history.sort(key=lambda h: h.get('date', ''))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=1)
    return history


def _write_json(path, obj, indent=2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def build_bracket(n_sims=N_SIMS):
    """Main entry point: load data, run predictions + season simulation, write bracket.json."""
    season = SEASON
    archive_previous_bracket(season, OUTPUT_JSON, ARCHIVE_DIR)

    # 1. Load games + playoff series of the current season
    df_all = pd.read_csv(GAMES_CSV)
    playoff_mask = df_all['GAME_ID'].astype(str).str.startswith(SEASON_CODE)
    df_playoff = df_all[playoff_mask].copy()
    series_map = build_series_dict(df_playoff)

    schedule_df = _read_csv(SCHEDULE_CSV)
    playin_df = _read_csv(PLAYIN_CSV)
    try:
        mode = detect_mode(df_all, season, schedule_df, playin_df)
    except Exception as e:
        mode = 'live' if series_map else 'projected'
        print(f"  WARNUNG: Modus-Erkennung fehlgeschlagen ({e}), nutze {mode}")
    print(f"  Modus: {mode}")

    # 2. Model + team snapshots (XGBoost) — failure must not stop the bracket
    predict_fn, xgb = None, None
    try:
        df_model = pd.read_csv(MODEL_DATA_CSV)
        home_snap, away_snap = get_team_snapshots(df_model)
        model        = joblib.load(MODEL_PKL)
        feature_cols = pd.read_csv(FEATURE_CSV).squeeze().tolist()
        feature_cols = [c for c in feature_cols if c not in EXCLUDE_COLS]

        def predict_fn(home_team, away_team, cur_home_wins=0, cur_away_wins=0):
            return predict_series(home_team, away_team, home_snap, away_snap,
                                  model, feature_cols, cur_home_wins, cur_away_wins)

        xgb = {"model": model, "feature_cols": feature_cols,
               "home_snap": home_snap, "away_snap": away_snap}
    except Exception as e:
        print(f"  WARNUNG: Modell/Snapshots nicht verfuegbar: {e}")

    # 3. Season simulation (errors are reported, never fatal)
    sim, t0 = None, time.time()
    try:
        import season_simulator as ss
        predictions_df = None
        if os.path.exists(PREDICTIONS_XLSX):
            try:
                predictions_df = pd.read_excel(PREDICTIONS_XLSX)
            except Exception as e:
                print(f"  WARNUNG: predictions.xlsx nicht lesbar: {e}")
        known_series, known_playin = known_states(series_map, playin_df, season)
        sim = ss.run_simulation(season, mode, df_all, schedule_df,
                                predictions_df=predictions_df, xgb=xgb, n_sims=n_sims,
                                known_playin=known_playin, known_series=known_series)
        report = sim['report']
    except Exception as e:
        traceback.print_exc()
        report = {
            "generated_at": pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%dT%H:%M:%S'),
            "season": season, "mode": mode, "sims": 0,
            "runtime_seconds": round(time.time() - t0, 1),
            "status": "critical", "passed": False, "checks": [],
            "issues": [f"Saison-Simulation fehlgeschlagen: {type(e).__name__}: {e}"],
            "warnings": [],
        }

    # 4. Assemble bracket JSON
    if mode == 'live' and series_map:
        bracket = build_bracket_json(series_map, predict_fn, season)   # bestehende Live-Logik
        _mark_live_series(bracket, series_map)
    elif sim is not None:
        bracket = {
            'generated_at': pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%dT%H:%M:%S'),
            'season': season,
            **sim['bracket'],
        }
    else:
        bracket = build_bracket_json(series_map, predict_fn, season)   # Rueckfall: Bestandslogik

    bracket['mode'] = mode
    bracket['sims'] = n_sims if sim is not None else 0
    if sim is not None:
        bracket['projection_date'] = sim['history_entry']['date']
        bracket['standings']  = sim['standings']
        bracket['play_in']    = sim['play_in']
        bracket['round_odds'] = sim['round_odds']

    # 5. Write outputs
    _write_json(OUTPUT_JSON, bracket)
    _write_json(SIM_REPORT_JSON, report)
    if sim is not None:
        update_history(sim['history_entry'], HISTORY_JSON)

    total_series = (
        len(bracket['east']['r1']) + len(bracket['east']['r2']) +
        len(bracket['east']['r3']) + len(bracket['west']['r1']) +
        len(bracket['west']['r2']) + len(bracket['west']['r3']) + 1
    )
    print(f"OK: bracket.json written ({total_series} series, "
          f"{len(series_map)} with game data, mode {mode}, "
          f"simulation {report.get('status')})")


if __name__ == '__main__':
    build_bracket()
