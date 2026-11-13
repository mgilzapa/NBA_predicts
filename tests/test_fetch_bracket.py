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


# ─── Saison-Prognose: Integration in fetch_bracket ──────────────────

import json as _json

import numpy as _np
import pytest as _pytest

import fetch_bracket as _fb
import season_simulator as _ss
from tests.test_season_simulator import synthetic_league as _synthetic_league

_SERIES_KEYS = {'home_team', 'away_team', 'home_wins', 'away_wins', 'status', 'winner', 'prediction'}
_PRED_KEYS = {'winner', 'win_probability', 'predicted_length', 'predicted_score', 'outcome_distribution'}


@_pytest.fixture
def bracket_env(tmp_path, monkeypatch):
    games, sched = _synthetic_league()
    games_csv = tmp_path / "nba_api_games.csv"
    sched_csv = tmp_path / "schedule.csv"
    games.to_csv(games_csv, index=False)
    sched.to_csv(sched_csv, index=False)
    paths = {
        "GAMES_CSV": games_csv, "SCHEDULE_CSV": sched_csv,
        "PLAYIN_CSV": tmp_path / "missing_playin.csv",
        "PREDICTIONS_XLSX": tmp_path / "missing_predictions.xlsx",
        "MODEL_DATA_CSV": tmp_path / "missing_model_data.csv",
        "OUTPUT_JSON": tmp_path / "web" / "bracket.json",
        "HISTORY_JSON": tmp_path / "web" / "projection_history.json",
        "SIM_REPORT_JSON": tmp_path / "output" / "season_sim_report.json",
        "ARCHIVE_DIR": tmp_path / "archive",
    }
    for k, v in paths.items():
        monkeypatch.setattr(_fb, k, str(v))
    monkeypatch.setattr(_fb, "SEASON", "2026-27")
    monkeypatch.setattr(_fb, "SEASON_CODE", "42600")
    monkeypatch.setattr(_ss, "load_elo", lambda path=None: _np.full(_ss.N_TEAMS, 1500.0))
    (tmp_path / "web").mkdir()
    with open(paths["OUTPUT_JSON"], "w", encoding="utf-8") as f:
        _json.dump({"season": "2025-26", "east": {}, "west": {}, "finals": None}, f)
    return paths


def _load(path):
    with open(path, encoding="utf-8") as f:
        return _json.load(f)


def test_bracket_json_new_fields_and_backward_compatible(bracket_env):
    _fb.build_bracket(n_sims=300)
    b = _load(bracket_env["OUTPUT_JSON"])
    # alte Struktur unveraendert
    for conf in ("east", "west"):
        assert len(b[conf]["r1"]) == 4 and len(b[conf]["r2"]) == 2 and len(b[conf]["r3"]) == 1
        assert isinstance(b[conf]["finalist"], str)
        for s in b[conf]["r1"] + b[conf]["r2"] + b[conf]["r3"]:
            assert _SERIES_KEYS <= set(s)
            assert _PRED_KEYS <= set(s["prediction"])
            assert s["status"] == "projected" and s["projected"] is True
            tp = s["prediction"]["team_probabilities"]
            assert set(tp) == {s["home_team"], s["away_team"]}
            assert abs(sum(tp.values()) - 1) < 1e-3
    assert _SERIES_KEYS <= set(b["finals"])
    assert b["season"] == "2026-27" and "generated_at" in b
    # neue Felder
    assert b["mode"] == "projected" and b["sims"] == 300
    for conf in ("east", "west"):
        assert len(b["standings"][conf]["current"]) == 15
        assert len(b["standings"][conf]["projected"]) == 15
        assert {"rank", "team", "w", "l", "pct", "gb", "l10", "streak"} <= set(b["standings"][conf]["current"][0])
        assert {"rank", "team", "exp_w", "exp_l", "w_p10", "w_p90", "pct_playoffs", "pct_playin",
                "pct_top_seed", "rank_change"} <= set(b["standings"][conf]["projected"][0])
        assert len(b["play_in"][conf]) == 3
    assert len(b["round_odds"]) == 30
    assert {"team", "conf", "play_in", "playoffs", "r2", "conf_finals", "finals", "champion"} <= set(b["round_odds"][0])
    # Report + Verlauf + Archiv
    rep = _load(bracket_env["SIM_REPORT_JSON"])
    assert rep["status"] in ("ok", "warning") and rep["passed"]
    assert len(_load(bracket_env["HISTORY_JSON"])) == 1
    assert os.path.exists(os.path.join(bracket_env["ARCHIVE_DIR"], "bracket_2025-26.json"))


def test_history_same_day_overwritten(bracket_env):
    _fb.build_bracket(n_sims=200)
    _fb.build_bracket(n_sims=200)
    hist = _load(bracket_env["HISTORY_JSON"])
    assert len(hist) == 1


def test_simulator_failure_keeps_bracket(bracket_env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaputt")
    monkeypatch.setattr(_ss, "run_simulation", boom)
    _fb.build_bracket(n_sims=100)
    b = _load(bracket_env["OUTPUT_JSON"])
    assert {"east", "west", "finals"} <= set(b)          # Bestandslogik geschrieben
    assert "standings" not in b and b["sims"] == 0
    rep = _load(bracket_env["SIM_REPORT_JSON"])
    assert rep["status"] == "critical" and not rep["passed"]
    assert "kaputt" in rep["issues"][0]


def test_detect_mode_variants():
    games, sched = _synthetic_league()
    assert _fb.detect_mode(games, "2026-27", sched) == "projected"
    assert _fb.detect_mode(games, "2026-27", sched.iloc[0:0]) == "post_season_pending"
    playoff = games.iloc[[0]].copy()
    playoff["GAME_ID"] = 42600101
    playoff["season_type"] = "Playoffs"
    assert _fb.detect_mode(pd.concat([games, playoff]), "2026-27", sched) == "live"
    playin = games.iloc[[0]].copy()
    playin["GAME_ID"] = 52600101
    playin["season_type"] = "PlayIn"
    assert _fb.detect_mode(games, "2026-27", sched, playin) == "live"


def test_archive_never_overwrites(tmp_path):
    out = tmp_path / "bracket.json"
    arch = tmp_path / "archive"
    out.write_text(_json.dumps({"season": "2025-26", "v": 1}), encoding="utf-8")
    _fb.archive_previous_bracket("2026-27", str(out), str(arch))
    out.write_text(_json.dumps({"season": "2025-26", "v": 2}), encoding="utf-8")
    _fb.archive_previous_bracket("2026-27", str(out), str(arch))
    assert _load(arch / "bracket_2025-26.json")["v"] == 1
    # gleiche Saison → kein Archiv
    out.write_text(_json.dumps({"season": "2026-27"}), encoding="utf-8")
    assert _fb.archive_previous_bracket("2026-27", str(out), str(arch)) is None


def test_build_feature_row_context_overrides_playoff_default():
    feature_cols = ["is_playoff", "rest_days_diff", "elo_diff"]
    home_snap = {"A": {"home_rest_days": 1.0, "home_elo": 1600.0}}
    away_snap = {"B": {"away_rest_days": 3.0, "away_elo": 1500.0}}
    old = _fb.build_feature_row("A", "B", home_snap, away_snap, feature_cols)
    assert list(old) == [1.0, -2.0, 100.0]                # Verhalten ohne context unveraendert
    new = _fb.build_feature_row("A", "B", home_snap, away_snap, feature_cols,
                                context={"is_playoff": 0.0, "home_rest_days": 2.0})
    assert list(new) == [0.0, -1.0, 100.0]
