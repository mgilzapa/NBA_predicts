# NBA Playoffs Bracket Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Bracket" tab to the NBA Predictor web app that shows the full NBA Playoffs bracket with series standings and ML-powered win predictions for active/upcoming series.

**Architecture:** `fetch_bracket.py` reads already-fetched playoff game data from `data/nba_api_games.csv`, groups games into series by parsing the game ID format (`42500RMG` → round R, matchup M, game G), runs the XGBoost model to simulate remaining games, then writes `web/bracket.json`. The frontend loads this JSON and renders a two-conference bracket grid with matchup cards.

**Tech Stack:** Python 3.x · pandas · joblib/XGBoost (existing model) · vanilla JS · CSS Flexbox/Grid

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/fetch_bracket.py` | **Create** | API data → series standings → simulation → bracket.json |
| `tests/test_fetch_bracket.py` | **Create** | Tests for pure simulation + parsing logic |
| `run_all.py` | **Modify** | Add `fetch_bracket.py` at end of pipeline |
| `web/index.html` | **Modify** | Add fourth "Bracket" tab button + panel |
| `web/style.css` | **Modify** | Add bracket layout, matchup card, connector styles |
| `web/app.js` | **Modify** | Load bracket.json, render bracket tab |

---

## Task 1: Tests for core pure functions

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_fetch_bracket.py`

- [ ] **Step 1: Create tests directory and empty `__init__.py`**

```
mkdir tests
```
Create empty `tests/__init__.py`.

- [ ] **Step 2: Write tests for `parse_game_id`**

```python
# tests/test_fetch_bracket.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fetch_bracket import parse_game_id, get_conference, simulate_series


def test_parse_game_id_round1():
    assert parse_game_id(42500131) == (1, 3, 1)


def test_parse_game_id_round2():
    assert parse_game_id(42500224) == (2, 2, 4)


def test_parse_game_id_round3():
    assert parse_game_id(42500301) == (3, 0, 1)


def test_parse_game_id_finals():
    assert parse_game_id(42500401) == (4, 0, 1)
```

- [ ] **Step 3: Write tests for `get_conference`**

```python
def test_get_conference_r1_east():
    assert get_conference(1, 0) == 'east'
    assert get_conference(1, 3) == 'east'


def test_get_conference_r1_west():
    assert get_conference(1, 4) == 'west'
    assert get_conference(1, 7) == 'west'


def test_get_conference_r2():
    assert get_conference(2, 0) == 'east'
    assert get_conference(2, 1) == 'east'
    assert get_conference(2, 2) == 'west'
    assert get_conference(2, 3) == 'west'


def test_get_conference_r3():
    assert get_conference(3, 0) == 'east'
    assert get_conference(3, 1) == 'west'


def test_get_conference_finals():
    assert get_conference(4, 0) == 'finals'
```

- [ ] **Step 4: Write tests for `simulate_series`**

```python
def test_simulate_series_already_complete_home_won():
    p_win, exp_games = simulate_series(4, 2, 0.6, 0.4)
    assert p_win == 1.0
    assert exp_games == 6


def test_simulate_series_already_complete_away_won():
    p_win, exp_games = simulate_series(2, 4, 0.6, 0.4)
    assert p_win == 0.0
    assert exp_games == 6


def test_simulate_series_fair_coin_symmetric():
    p_win, exp_games = simulate_series(0, 0, 0.5, 0.5)
    assert abs(p_win - 0.5) < 1e-9
    # Expected: round(93/16) = round(5.8125) = 6
    assert exp_games == 6


def test_simulate_series_certain_home_wins_every_game():
    # p1=1.0, p2=0.0: home wins games 1,2,5,7; loses 3,4,6 → wins 4-3 in game 7
    p_win, exp_games = simulate_series(0, 0, 1.0, 0.0)
    assert p_win == 1.0
    assert exp_games == 7


def test_simulate_series_mid_series():
    # Home up 3-0, p1=p2=0.5: home wins next game (game 4 = away game)
    # From 3-0, game 4 (away game, p2=0.5 means 0.5 chance home wins):
    # 50% home wins in 4, 50% → 3-1 → game 5 (home game, p1=0.5) → etc.
    p_win, _ = simulate_series(3, 0, 0.5, 0.5)
    # P(home wins) = 1 - P(away wins from 3-0) = 1 - 0.5^4 = 1 - 0.0625 = 0.9375
    assert abs(p_win - 0.9375) < 1e-9
```

- [ ] **Step 5: Run tests (they must FAIL — module not yet created)**

```
python -m pytest tests/test_fetch_bracket.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'fetch_bracket'`

- [ ] **Step 6: Commit skeleton**

```bash
git add tests/
git commit -m "test: add fetch_bracket pure-function tests (red)"
```

---

## Task 2: `fetch_bracket.py` — parsing and series building

**Files:**
- Create: `src/fetch_bracket.py`

- [ ] **Step 1: Create `src/fetch_bracket.py` with pure parsing functions**

```python
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
```

- [ ] **Step 2: Run tests — parsing and simulation must pass**

```
python -m pytest tests/test_fetch_bracket.py -v
```
Expected: All 11 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/fetch_bracket.py tests/test_fetch_bracket.py
git commit -m "feat: add bracket parsing and series simulation core"
```

---

## Task 3: `fetch_bracket.py` — series extraction from game data

**Files:**
- Modify: `src/fetch_bracket.py`

- [ ] **Step 1: Add test for `build_series_dict`**

Add to `tests/test_fetch_bracket.py`:

```python
import pandas as pd
from fetch_bracket import build_series_dict


def _make_games():
    """Minimal DataFrame mimicking nba_api_games.csv rows."""
    return pd.DataFrame([
        # Round 1, matchup 3, game 1: CLE hosts TOR, CLE wins
        {'GAME_ID': 42500131, 'hometeamName': 'Cleveland Cavaliers',
         'awayteamName': 'Toronto Raptors', 'home_win': 1},
        # Round 1, matchup 3, game 2: CLE hosts TOR, TOR wins
        {'GAME_ID': 42500132, 'hometeamName': 'Cleveland Cavaliers',
         'awayteamName': 'Toronto Raptors', 'home_win': 0},
        # Round 1, matchup 3, game 3: TOR hosts CLE (away game), CLE wins
        {'GAME_ID': 42500133, 'hometeamName': 'Toronto Raptors',
         'awayteamName': 'Cleveland Cavaliers', 'home_win': 0},
    ])


def test_build_series_dict_wins():
    series_map = build_series_dict(_make_games())
    s = series_map[(1, 3)]
    assert s['home_team'] == 'Cleveland Cavaliers'   # series home = game-1 host
    assert s['away_team'] == 'Toronto Raptors'
    assert s['home_wins'] == 2   # CLE won games 1 and 3
    assert s['away_wins'] == 1   # TOR won game 2
    assert s['status'] == 'active'
    assert s['winner'] is None


def test_build_series_dict_complete():
    df = _make_games()
    # Add 3 more CLE wins
    for i in range(4, 7):
        df = pd.concat([df, pd.DataFrame([{
            'GAME_ID': int(f'4250013{i}'),
            'hometeamName': 'Cleveland Cavaliers' if i in (5,) else 'Toronto Raptors',
            'awayteamName': 'Toronto Raptors' if i in (5,) else 'Cleveland Cavaliers',
            'home_win': 1 if i in (5,) else 0,
        }])], ignore_index=True)
    series_map = build_series_dict(df)
    s = series_map[(1, 3)]
    assert s['status'] == 'complete'
    assert s['winner'] == 'Cleveland Cavaliers'
    assert s['home_wins'] == 4
```

- [ ] **Step 2: Run new tests (must FAIL)**

```
python -m pytest tests/test_fetch_bracket.py::test_build_series_dict_wins -v
```
Expected: `ImportError: cannot import name 'build_series_dict'`

- [ ] **Step 3: Add `build_series_dict` to `fetch_bracket.py`**

```python
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
```

- [ ] **Step 4: Run all tests**

```
python -m pytest tests/test_fetch_bracket.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fetch_bracket.py tests/test_fetch_bracket.py
git commit -m "feat: build series dict from playoff game data"
```

---

## Task 4: `fetch_bracket.py` — feature extraction and ML simulation

**Files:**
- Modify: `src/fetch_bracket.py`

- [ ] **Step 1: Add `get_team_snapshots` function**

Add to `src/fetch_bracket.py`, after the existing imports/constants:

```python
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
                row[col] = float(non_null.iloc[-1]) if not non_null.empty else np.nan
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
```

- [ ] **Step 2: Run existing tests (must still pass)**

```
python -m pytest tests/test_fetch_bracket.py -v
```
Expected: All tests PASS (new code is additive, no tests for snapshots needed at unit level).

- [ ] **Step 3: Commit**

```bash
git add src/fetch_bracket.py
git commit -m "feat: add team snapshot extraction and per-series prediction"
```

---

## Task 5: `fetch_bracket.py` — bracket assembly and JSON output

**Files:**
- Modify: `src/fetch_bracket.py`

- [ ] **Step 1: Add `build_bracket_json` and `build_bracket` functions**

Append to `src/fetch_bracket.py`:

```python
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
    Return the team expected to advance from a series:
    - If complete: the actual winner
    - If active/upcoming: the prediction winner
    - If not in series_map: None
    """
    s = series_map.get(key)
    if s is None:
        return None
    if s['status'] == 'complete':
        return s['winner']
    prob, _ = predict_fn(s['home_team'], s['away_team']) if predict_fn else (0.5, 6)
    return s['home_team'] if prob >= 0.5 else s['away_team']


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
        'generated_at': pd.Timestamp.now(tz='US/Eastern').strftime('%Y-%m-%dT%H:%M:%S'),
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
```

- [ ] **Step 2: Run `fetch_bracket.py` manually to verify output**

```
python src/fetch_bracket.py
```
Expected output: `OK: bracket.json written (15 series, N with game data)`
Check that `web/bracket.json` was created and has the correct structure:
```
python -c "import json; d=json.load(open('web/bracket.json')); print(list(d.keys())); print(len(d['east']['r1']), 'East R1 series')"
```
Expected: `['generated_at', 'season', 'east', 'west', 'finals']` and `4 East R1 series`

- [ ] **Step 3: Run full test suite**

```
python -m pytest tests/test_fetch_bracket.py -v
```
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/fetch_bracket.py web/bracket.json
git commit -m "feat: complete fetch_bracket.py with bracket assembly and JSON output"
```

---

## Task 6: `run_all.py` — pipeline integration

**Files:**
- Modify: `run_all.py`

- [ ] **Step 1: Add `fetch_bracket.py` to the scripts list in `run_all.py`**

Open `run_all.py`. The `scripts` list ends with `"src/export_json.py"`. Add `fetch_bracket.py` after it:

```python
        "src/export_json.py",    # JSON für Webseite exportieren
        "src/fetch_bracket.py",  # Bracket-Daten und Predictions
```

The full updated `scripts` list (lines 21–32 of `run_all.py`):

```python
    scripts = [
        "src/nba_api_test.py",
        "src/scrape_upcoming_games.py",
        "src/injury_reports.py",
        "src/fetch_player_stats.py",
        "src/tabelle.py",
        "src/feature_engineering.py",
        "src/predict.py",
        "src/clean_excel.py",
        "src/create_excel.py",
        "src/export_json.py",
        "src/fetch_bracket.py",
    ]
```

- [ ] **Step 2: Verify script exists check passes**

```
python -c "
import os
scripts = ['src/fetch_bracket.py']
for s in scripts:
    print(s, os.path.isfile(s))
"
```
Expected: `src/fetch_bracket.py True`

- [ ] **Step 3: Commit**

```bash
git add run_all.py
git commit -m "feat: add fetch_bracket.py to daily pipeline"
```

---

## Task 7: Frontend — HTML tab + CSS bracket layout

**Files:**
- Modify: `web/index.html`
- Modify: `web/style.css`

- [ ] **Step 1: Add Bracket tab button and panel to `index.html`**

In `web/index.html`, add `data-tab="bracket"` button after the "All Predictions" tab (line 38):

```html
    <button class="tab" data-tab="all" role="tab">All Predictions</button>
    <button class="tab" data-tab="bracket" role="tab">Bracket</button>
```

Add the bracket panel after `</section>` of `tab-all` (before `<script src="app.js">`):

```html
    <section id="tab-bracket" class="tab-content" role="tabpanel">
      <div class="bracket-header" id="bracket-header"></div>
      <div class="bracket-outer" id="bracket-outer">
        <div class="bracket-container" id="bracket-container">
          <!-- rendered by app.js -->
        </div>
      </div>
      <div class="bracket-mobile" id="bracket-mobile">
        <!-- rendered by app.js for mobile -->
      </div>
    </section>
```

- [ ] **Step 2: Add bracket CSS to `web/style.css`**

Append to the end of `web/style.css`:

```css
/* ─── Bracket Tab ─────────────────────────────────── */

.bracket-header {
  font-size: 0.82rem;
  color: var(--muted);
  margin-bottom: 20px;
}

/* Desktop bracket: hidden on mobile */
.bracket-outer {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}

.bracket-container {
  display: flex;
  align-items: stretch;
  gap: 0;
  min-width: 900px;
}

/* Each half fills half the container */
.bracket-half {
  display: flex;
  flex: 1;
  gap: 0;
}

.bracket-half.east { flex-direction: row; }
.bracket-half.west { flex-direction: row-reverse; }

/* Finals column in the center */
.bracket-finals-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 0 0 180px;
  padding: 0 8px;
}

.bracket-finals-label {
  font-family: 'Oswald', 'Arial Narrow', Arial, sans-serif;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 8px;
}

/* Each round column */
.bracket-round {
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  flex: 0 0 160px;
  padding: 4px 0;
  position: relative;
}

.bracket-round-label {
  font-family: 'Oswald', 'Arial Narrow', Arial, sans-serif;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: var(--muted);
  text-transform: uppercase;
  text-align: center;
  padding: 4px 0 8px;
}

/* Matchup card wrapper (handles connector lines) */
.bracket-matchup {
  position: relative;
  padding: 0 8px;
  display: flex;
  align-items: center;
}

/* Connector lines between rounds — right edge of a card connects rightward */
.bracket-matchup::after {
  content: '';
  position: absolute;
  right: -1px;
  top: 50%;
  width: 9px;
  height: 1px;
  background: var(--border);
}

.bracket-half.west .bracket-matchup::after {
  right: auto;
  left: -1px;
}

/* ─── Matchup Card ─────────────────────────────────── */

.matchup-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 10px;
  width: 100%;
  position: relative;
  transition: border-color 0.2s, background 0.2s;
}

.matchup-card:hover {
  border-color: rgba(88, 166, 255, 0.35);
  background: var(--surface-hover);
}

.matchup-card.status-tbd {
  opacity: 0.45;
}

.matchup-card.status-complete {
  border-color: rgba(63, 185, 80, 0.25);
}

/* Teams row inside card */
.mc-teams {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 5px;
}

.mc-team {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.72rem;
  color: var(--muted);
}

.mc-team.winner {
  color: var(--accent);
  font-weight: 700;
}

.mc-team.loser {
  color: var(--muted);
  opacity: 0.6;
}

.mc-team img.mc-logo {
  width: 16px;
  height: 16px;
  object-fit: contain;
  flex-shrink: 0;
}

.mc-score {
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  color: var(--text);
  font-weight: 600;
}

/* Prediction row */
.mc-pred {
  font-size: 0.67rem;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
}

.mc-pred .pred-winner {
  color: var(--accent);
  font-weight: 600;
}

.mc-pred .pred-prob {
  color: var(--muted);
}

/* Correct/Wrong badge on complete series */
.mc-badge {
  position: absolute;
  top: 6px;
  right: 6px;
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  padding: 2px 5px;
  border-radius: 4px;
}

.mc-badge.correct { background: rgba(63, 185, 80, 0.2); color: var(--correct); }
.mc-badge.wrong   { background: rgba(248, 81, 73, 0.2); color: var(--wrong); }

/* ─── Mobile bracket list ─────────────────────────── */

.bracket-mobile { display: none; }

.bm-round-title {
  font-family: 'Oswald', 'Arial Narrow', Arial, sans-serif;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text);
  letter-spacing: 0.04em;
  margin: 20px 0 10px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px;
}

.bm-conf-label {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: var(--muted);
  text-transform: uppercase;
  margin: 12px 0 6px;
}

.bm-series-list {
  display: grid;
  gap: 8px;
  margin-bottom: 8px;
}

@media (max-width: 640px) {
  .bracket-outer    { display: none; }
  .bracket-mobile   { display: block; }
}
```

- [ ] **Step 3: Verify HTML is valid (open browser)**

```
python -m http.server 8000
```
Open `http://localhost:8000/web/index.html`. Click the "Bracket" tab — it should appear as a blank tab with no JS errors.

- [ ] **Step 4: Commit**

```bash
git add web/index.html web/style.css
git commit -m "feat: add Bracket tab HTML structure and CSS layout"
```

---

## Task 8: Frontend — JS bracket rendering

**Files:**
- Modify: `web/app.js`

- [ ] **Step 1: Add bracket data loading to `loadData()`**

In `web/app.js`, find the `loadData` function (line 60). Add `bracket.json` to the parallel fetches:

```javascript
async function loadData() {
  try {
    const [predRes, oddsRes, bracketRes] = await Promise.all([
      fetch('predictions.json'),
      fetch('odds.json').catch(() => null),
      fetch('bracket.json').catch(() => null),
    ]);

    if (!predRes.ok) throw new Error(`HTTP ${predRes.status}`);
    const data = await predRes.json();

    if (oddsRes && oddsRes.ok) {
      try { oddsData = await oddsRes.json(); } catch { /* ignore */ }
    }

    let bracketData = null;
    if (bracketRes && bracketRes.ok) {
      try { bracketData = await bracketRes.json(); } catch { /* ignore */ }
    }

    currentData = data;
    render(data);
    renderBracket(bracketData);
  } catch (err) {
    showGlobalError(err.message);
  }
}
```

- [ ] **Step 2: Add `renderBracket`, `buildMatchupCard`, and `buildBracketMobile` functions**

Add the following at the end of `web/app.js`, before `loadData()`:

```javascript
// ─── Bracket Rendering ─────────────────────────────────────────────

function renderBracket(data) {
  const header    = document.getElementById('bracket-header');
  const container = document.getElementById('bracket-container');
  const mobile    = document.getElementById('bracket-mobile');

  if (!data) {
    header.textContent = 'Bracket data not available yet.';
    return;
  }

  header.innerHTML =
    `Season <strong>${esc(data.season)}</strong> · Updated ${esc(data.generated_at.slice(0, 16).replace('T', ' '))} ET`;

  container.innerHTML = buildBracketDesktop(data);
  mobile.innerHTML    = buildBracketMobile(data);
}

function buildMatchupCard(series) {
  if (!series) return '<div class="matchup-card status-tbd"><div class="mc-teams"><div class="mc-team">TBD</div></div></div>';

  const home = series.home_team || 'TBD';
  const away = series.away_team || 'TBD';

  if (series.status === 'tbd') {
    return `<div class="matchup-card status-tbd">
      <div class="mc-teams">
        <div class="mc-team">${esc(home)}</div>
        <div class="mc-team">${esc(away)}</div>
      </div>
    </div>`;
  }

  const pred   = series.prediction;
  const winner = series.winner;

  const homeCls = winner ? (winner === home ? 'winner' : 'loser') : '';
  const awayCls = winner ? (winner === away ? 'winner' : 'loser') : '';

  const scoreHtml = (series.status === 'complete' || series.home_wins + series.away_wins > 0)
    ? `<span class="mc-score">${series.home_wins}–${series.away_wins}</span>`
    : '';

  let predHtml = '';
  if (pred) {
    predHtml = `<div class="mc-pred">
      → <span class="pred-winner">${esc(pred.winner)}</span>
      <span class="pred-prob">${Math.round(pred.win_probability * 100)}% · in ${pred.predicted_length}G</span>
    </div>`;
  }

  let badge = '';
  if (series.status === 'complete' && pred) {
    const correct = pred.winner === winner;
    badge = `<span class="mc-badge ${correct ? 'correct' : 'wrong'}">${correct ? '✓' : '✗'}</span>`;
  }

  const cardCls = `matchup-card status-${series.status}`;
  return `<div class="${cardCls}">
    ${badge}
    <div class="mc-teams">
      <div class="mc-team ${awayCls}">
        ${logoHtml(away)}
        ${esc(away)}${scoreHtml && awayCls === 'winner' ? '' : ''}
      </div>
      <div class="mc-team ${homeCls}">
        ${logoHtml(home)}
        ${esc(home)}
        ${scoreHtml}
      </div>
    </div>
    ${predHtml}
  </div>`;
}

function buildRoundCol(series_list, label) {
  const cards = series_list.map(s =>
    `<div class="bracket-matchup">${buildMatchupCard(s)}</div>`
  ).join('');
  return `<div class="bracket-round">
    <div class="bracket-round-label">${esc(label)}</div>
    ${cards}
  </div>`;
}

function buildBracketDesktop(data) {
  const eastR1  = buildRoundCol(data.east.r1,  'R1');
  const eastR2  = buildRoundCol(data.east.r2,  'R2');
  const eastR3  = buildRoundCol(data.east.r3,  'Conf Finals');
  const westR3  = buildRoundCol(data.west.r3,  'Conf Finals');
  const westR2  = buildRoundCol(data.west.r2,  'R2');
  const westR1  = buildRoundCol(data.west.r1,  'R1');

  const finalsCard = buildMatchupCard(data.finals);
  const finalsCol  = `<div class="bracket-finals-col">
    <div class="bracket-finals-label">Finals</div>
    ${finalsCard}
  </div>`;

  return `
    <div class="bracket-half east">${eastR1}${eastR2}${eastR3}</div>
    ${finalsCol}
    <div class="bracket-half west">${westR3}${westR2}${westR1}</div>
  `;
}

function buildBracketMobile(data) {
  const rounds = [
    { label: 'First Round',        east: data.east.r1, west: data.west.r1 },
    { label: 'Second Round',       east: data.east.r2, west: data.west.r2 },
    { label: 'Conference Finals',  east: data.east.r3, west: data.west.r3 },
    { label: 'NBA Finals',         east: [data.finals], west: [] },
  ];

  return rounds.map(r => `
    <div class="bm-round-title">${esc(r.label)}</div>
    ${r.east.length ? `<div class="bm-conf-label">East</div><div class="bm-series-list">${r.east.map(s => buildMatchupCard(s)).join('')}</div>` : ''}
    ${r.west.length ? `<div class="bm-conf-label">West</div><div class="bm-series-list">${r.west.map(s => buildMatchupCard(s)).join('')}</div>` : ''}
  `).join('');
}
```

- [ ] **Step 3: Test in browser**

```
python -m http.server 8000
```
Open `http://localhost:8000/web/index.html`. Click "Bracket":
- Desktop (> 640px): horizontal bracket visible with East on left, Finals in center, West on right
- Cards show team names, series scores (e.g. "3–2"), prediction winner + probability
- Complete series show green ✓ or red ✗ badge
- TBD series appear grayed out

Resize window to < 640px: horizontal bracket hides, mobile list appears grouped by round.

- [ ] **Step 4: Check browser console for JS errors**

Open DevTools → Console. Must be zero errors on page load and tab switch.

- [ ] **Step 5: Run all Python tests**

```
python -m pytest tests/test_fetch_bracket.py -v
```
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add web/app.js
git commit -m "feat: render NBA bracket tab with matchup cards and mobile fallback"
```

---

## Self-Review Checklist

Spec sections checked against tasks:

| Spec requirement | Covered by |
|---|---|
| New "Bracket" tab in nav | Task 7 |
| `bracket.json` structure (east/west/finals, r1-r3, series objects) | Task 5 |
| `fetch_bracket.py` - NBA API / game data parsing | Tasks 2–3 |
| Series simulation (2-2-1-1-1, enumerate paths, p_home + exp_games) | Tasks 1–2 |
| TBD future rounds: use predicted winner from prior round | Task 5 |
| Matchup card: teams, scores, prediction, badge | Task 8 |
| Complete series: correct/wrong badge | Task 8 |
| TBD card: grayed out | Task 8 |
| Desktop horizontal bracket layout | Tasks 7–8 |
| Mobile vertical list | Tasks 7–8 |
| `run_all.py` integration, old bracket.json preserved on error | Task 6 |
| `bracket.json` missing → "not available" message | Task 8 |
| Uses existing CSS variables | Task 7 |

**Error handling note:** `run_all.py` already calls `sys.exit(1)` if any script fails, which stops the pipeline but does not overwrite previous outputs. `fetch_bracket.py` writes atomically only on success — if it crashes before `json.dump`, the old `bracket.json` is preserved.
