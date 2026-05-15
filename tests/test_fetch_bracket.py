import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fetch_bracket import parse_game_id, get_conference, simulate_series


# Tests for parse_game_id
def test_parse_game_id_round1():
    assert parse_game_id(42500131) == (1, 3, 1)


def test_parse_game_id_round2():
    assert parse_game_id(42500224) == (2, 2, 4)


def test_parse_game_id_round3():
    assert parse_game_id(42500301) == (3, 0, 1)


def test_parse_game_id_finals():
    assert parse_game_id(42500401) == (4, 0, 1)


# Tests for get_conference
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


# Tests for simulate_series
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
    # P(home wins) = 1 - P(away wins from 3-0) = 1 - 0.5^4 = 1 - 0.0625 = 0.9375
    p_win, _ = simulate_series(3, 0, 0.5, 0.5)
    assert abs(p_win - 0.9375) < 1e-9


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
    # Add 2 more CLE wins: game 4 (TOR hosts, CLE wins as visitor), game 5 (CLE hosts, CLE wins)
    extra = pd.DataFrame([
        {'GAME_ID': 42500134, 'hometeamName': 'Toronto Raptors',
         'awayteamName': 'Cleveland Cavaliers', 'home_win': 0},  # CLE wins
        {'GAME_ID': 42500135, 'hometeamName': 'Cleveland Cavaliers',
         'awayteamName': 'Toronto Raptors', 'home_win': 1},      # CLE wins
    ])
    df = pd.concat([df, extra], ignore_index=True)
    series_map = build_series_dict(df)
    s = series_map[(1, 3)]
    assert s['status'] == 'complete'
    assert s['winner'] == 'Cleveland Cavaliers'
    assert s['home_wins'] == 4
    assert s['away_wins'] == 1
