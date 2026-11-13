import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import season_simulator as ss  # noqa: E402
from season_simulator import TEAM_IDX, TiebreakWorld, resolve_tie  # noqa: E402

SEASON = "2026-27"

# Teams fuer die Tiebreak-Tests (East)
BOS, NYK, PHI = TEAM_IDX["Boston Celtics"], TEAM_IDX["New York Knicks"], TEAM_IDX["Philadelphia 76ers"]
TOR, BKN = TEAM_IDX["Toronto Raptors"], TEAM_IDX["Brooklyn Nets"]
CHI, CLE = TEAM_IDX["Chicago Bulls"], TEAM_IDX["Cleveland Cavaliers"]
MIA, ORL = TEAM_IDX["Miami Heat"], TEAM_IDX["Orlando Magic"]
OKC, DEN = TEAM_IDX["Oklahoma City Thunder"], TEAM_IDX["Denver Nuggets"]


def make_world(h2h_entries=(), tb=None, div_leaders=(), eligible=None):
    """Alle Teams 41-41; h2h_entries = [(winner, loser, n), ...]."""
    h2h = np.zeros((ss.N_TEAMS, ss.N_TEAMS), dtype=np.int64)
    for w, l, n in h2h_entries:
        h2h[w, l] += n
    world = TiebreakWorld(np.full(ss.N_TEAMS, 41), np.full(ss.N_TEAMS, 41), h2h,
                          tb if tb is not None else np.zeros(ss.N_TEAMS))
    world.div_leaders = set(div_leaders)
    if eligible is not None:
        world.eligible = eligible
    return world


# ─── Tiebreaker: zwei Teams ─────────────────────────────────────────

def test_two_team_head_to_head():
    world = make_world([(NYK, BOS, 3), (BOS, NYK, 1)])
    assert resolve_tie([BOS, NYK], world) == [NYK, BOS]


def test_two_team_division_leader():
    world = make_world([(BOS, CHI, 2), (CHI, BOS, 2)], div_leaders=[CHI])
    assert resolve_tie([BOS, CHI], world) == [CHI, BOS]


def test_two_team_division_record_same_division():
    # BOS und NYK (Atlantic), h2h ausgeglichen; NYK besser gegen die Division
    world = make_world([(BOS, NYK, 2), (NYK, BOS, 2), (NYK, TOR, 2), (BOS, TOR, 1), (TOR, BOS, 1)])
    assert resolve_tie([BOS, NYK], world) == [NYK, BOS]


def test_two_team_division_record_skipped_for_different_divisions():
    # BOS (Atlantic) hat die bessere Division-Bilanz, CHI (Central) die bessere Conference-Bilanz
    world = make_world([(BOS, TOR, 3), (CHI, MIA, 3), (CHI, ORL, 2), (BOS, MIA, 0), (MIA, BOS, 2)])
    # Conference: BOS 3-2, CHI 5-0 → CHI vorn, Division-Bilanz darf nicht greifen
    assert resolve_tie([BOS, CHI], world) == [CHI, BOS]


def test_two_team_conference_record():
    world = make_world([(BOS, MIA, 2), (CHI, MIA, 1), (MIA, CHI, 1)])
    assert resolve_tie([BOS, CHI], world) == [BOS, CHI]


def test_two_team_vs_eligible_own_conference():
    # Conference-Bilanz gleich (je 1-0), BOS hat das Spiel gegen ein Playoff-Team gewonnen
    eligible = {0: {BOS, CHI, MIA}, 1: set()}
    world = make_world([(BOS, MIA, 1), (CHI, ORL, 1)], eligible=eligible)
    assert resolve_tie([BOS, CHI], world) == [BOS, CHI]


def test_two_team_vs_eligible_other_conference():
    eligible = {0: {BOS, CHI}, 1: {OKC}}
    world = make_world([(CHI, OKC, 1), (BOS, DEN, 1)], eligible=eligible)
    assert resolve_tie([BOS, CHI], world) == [CHI, BOS]


def test_two_team_last_step_point_diff_or_random():
    tb = np.zeros(ss.N_TEAMS)
    tb[CHI] = 5.0
    world = make_world(tb=tb)
    assert resolve_tie([BOS, CHI], world) == [CHI, BOS]


# ─── Tiebreaker: drei und mehr Teams ────────────────────────────────

def test_multi_team_division_leader_first():
    world = make_world([(BOS, CHI, 3), (BOS, MIA, 3)], div_leaders=[MIA])
    order = resolve_tie([BOS, CHI, MIA], world)
    assert order[0] == MIA
    assert order[1:] == [BOS, CHI]          # danach Zwei-Team-Verfahren: BOS schlug CHI


def test_multi_team_head_to_head_of_all():
    # BOS 4-0 gegen die Gruppe, CHI 2-2, MIA 0-4 (verschiedene Divisionen, keine Division-Sieger)
    world = make_world([(BOS, CHI, 2), (BOS, MIA, 2), (CHI, MIA, 2)])
    assert resolve_tie([MIA, CHI, BOS], world) == [BOS, CHI, MIA]


def test_multi_team_restart_with_two_teams():
    # Gruppen-H2H: BOS 3-1, CHI 1-2, MIA 1-2 → BOS vorn; CHI/MIA beginnen neu mit dem
    # Zwei-Team-Verfahren (direkter Vergleich CHI 1-0), obwohl MIA die bessere
    # Conference-Bilanz hat.
    world = make_world([(BOS, CHI, 2), (BOS, MIA, 1), (MIA, BOS, 1), (CHI, MIA, 1),
                        (MIA, ORL, 5)])
    assert resolve_tie([MIA, CHI, BOS], world) == [BOS, CHI, MIA]


def test_seed_world_division_leaders_and_order():
    w = np.full(ss.N_TEAMS, 30)
    l = np.full(ss.N_TEAMS, 52)
    w[BOS], l[BOS] = 60, 22
    w[NYK], l[NYK] = 50, 32
    world = TiebreakWorld(w, l, np.zeros((ss.N_TEAMS, ss.N_TEAMS), dtype=np.int64),
                          np.arange(ss.N_TEAMS, dtype=float))
    east, west = ss.seed_world(world)
    assert east[:2] == [BOS, NYK]
    assert BOS in world.div_leaders and NYK not in world.div_leaders
    assert len(set(east)) == 15 and len(set(west)) == 15
    assert set(east) == set(ss.CONF_TEAMS[0])


# ─── Aktuelle Tabelle ───────────────────────────────────────────────

def _game(gid, date, home, hs, away, as_):
    return {"GAME_ID": gid, "GAME_DATE": date, "hometeamName": home, "homeScore": hs,
            "season_type": "Regular Season", "awayteamName": away, "awayScore": as_,
            "home_win": int(hs > as_), "season": SEASON}


def test_current_standings_wl_gb_l10_streak():
    games = []
    gid = 22600001
    # Boston: 12 Spiele, die ersten 2 verloren, dann 10 gewonnen → L10 10-0, Streak W10
    for i in range(12):
        bos_wins = i >= 2
        games.append(_game(gid, f"2026-10-{20 + i:02d}", "Boston Celtics",
                           110 if bos_wins else 100, "Brooklyn Nets", 100 if bos_wins else 110))
        gid += 1
    df = ss.played_regular_season(pd.DataFrame(games), SEASON)
    standings, current_rank, _ = ss.current_standings(df)
    east = {r["team"]: r for r in standings["east"]}
    bos, bkn = east["Boston Celtics"], east["Brooklyn Nets"]
    assert (bos["w"], bos["l"]) == (10, 2)
    assert (bkn["w"], bkn["l"]) == (2, 10)
    assert bos["rank"] == 1 and bos["gb"] == 0
    assert bkn["gb"] == 8.0
    assert bos["l10"] == "10-0" and bos["streak"] == "W10"
    assert bkn["l10"] == "0-10" and bkn["streak"] == "L10"
    assert current_rank[TEAM_IDX["Boston Celtics"]] == 1


def test_cup_final_excluded():
    games = [_game(22600001, "2026-11-01", "Boston Celtics", 110, "Miami Heat", 100),
             _game(62600001, "2026-12-16", "Boston Celtics", 110, "Miami Heat", 100)]
    df = ss.played_regular_season(pd.DataFrame(games), SEASON)
    assert len(df) == 1


# ─── Play-In ────────────────────────────────────────────────────────

@pytest.mark.parametrize("g1_seed7_wins", [True, False])
@pytest.mark.parametrize("g2_seed9_wins", [True, False])
@pytest.mark.parametrize("g3_loser_wins", [True, False])
def test_play_in_seeds_7_and_8(g1_seed7_wins, g2_seed9_wins, g3_loser_wins):
    n = 4
    seeds = np.zeros((n, 2, 15), dtype=np.int64)
    seeds[:, 0, :] = ss.CONF_TEAMS[0]
    seeds[:, 1, :] = ss.CONF_TEAMS[1]
    s7, s8, s9, s10 = ss.CONF_TEAMS[0][6:10]
    w1, l1 = (s7, s8) if g1_seed7_wins else (s8, s7)
    w2 = s9 if g2_seed9_wins else s10
    w3 = l1 if g3_loser_wins else w2
    known_playin = {frozenset((s7, s8)): w1, frozenset((s9, s10)): w2, frozenset((l1, w2)): w3}
    rng = np.random.default_rng(1)
    elo = np.full((ss.N_TEAMS, n), 1500.0)
    pct = np.full((n, ss.N_TEAMS), 0.5)
    post = ss.simulate_postseason(rng, elo, seeds, pct, known_playin=known_playin)
    _, away_1v8, _, _ = post["series"][(0, 1, 0)]
    _, away_2v7, _, _ = post["series"][(0, 1, 1)]
    assert (away_2v7 == w1).all()        # Sieger 7v8 → Seed 7
    assert (away_1v8 == w3).all()        # Sieger Entscheidungsspiel → Seed 8


# ─── Serien ─────────────────────────────────────────────────────────

def test_series_home_court_format_2_2_1_1_1(monkeypatch):
    # Gastgeber gewinnt immer → Heimteam gewinnt Spiele 1,2,5,7, Gast 3,4,6 → immer 4-3
    monkeypatch.setattr(ss, "elo_home_prob", lambda rh, rg: np.ones_like(rh))
    n = 50
    home = np.full(n, BOS)
    away = np.full(n, NYK)
    elo = np.full((ss.N_TEAMS, n), 1500.0)
    hw, aw, win = ss._series(np.random.default_rng(0), elo, home, away)
    assert (hw == 4).all() and (aw == 3).all() and (win == BOS).all()


def test_series_continues_from_known_state(monkeypatch):
    monkeypatch.setattr(ss, "elo_home_prob", lambda rh, rg: np.ones_like(rh))
    n = 5
    home, away = np.full(n, BOS), np.full(n, NYK)
    elo = np.full((ss.N_TEAMS, n), 1500.0)
    known = {frozenset((BOS, NYK)): {BOS: 1, NYK: 3}}
    # Spiel 5 (Heim) gewinnt BOS, Spiel 6 (Gast NYK) gewinnt NYK → 2-4
    hw, aw, win = ss._series(np.random.default_rng(0), elo, home, away, known)
    assert (hw == 2).all() and (aw == 4).all() and (win == NYK).all()


def test_series_distribution_sums_to_one():
    n = 20_000
    home, away = np.full(n, BOS), np.full(n, NYK)
    elo = np.full((ss.N_TEAMS, n), 1500.0)
    elo[BOS] = 1600.0
    arrays = ss._series(np.random.default_rng(3), elo.copy(), home, away)
    hw, aw, _ = arrays
    elo0 = np.full(ss.N_TEAMS, 1500.0)
    p, dist, cnt, src = ss._series_stats((home, away, hw, aw), BOS, NYK, elo0)
    assert src == "simulation" and cnt == n
    assert abs(sum(dist.values()) - 1) < 1e-9
    assert set(dist) <= {"4-0", "4-1", "4-2", "4-3", "0-4", "1-4", "2-4", "3-4"}
    assert p > 0.5                                 # staerker + Heimvorteil


def test_series_stats_fallback_with_few_sims():
    n = 10
    home, away = np.full(n, BOS), np.full(n, NYK)
    hw, aw = np.full(n, 4, dtype=np.int8), np.zeros(n, dtype=np.int8)
    elo0 = np.full(ss.N_TEAMS, 1500.0)
    p, dist, cnt, src = ss._series_stats((home, away, hw, aw), BOS, NYK, elo0)
    assert src == "elo_analytic" and cnt == n
    assert 0.5 < p < 1.0                           # analytisch, nicht 100 % aus den 10 Sims
    assert abs(sum(dist.values()) - 1) < 1e-9


# ─── Komplette Simulation auf synthetischen Daten ───────────────────

def _split(team):
    city, name = team.split(" ", 1)
    return city, name


def synthetic_league(remaining_per_team=4, seed=0):
    """Jede Paarung einmal gespielt (435 Spiele) + kleiner Restspielplan."""
    rng = np.random.default_rng(seed)
    teams = ss.TEAMS
    games, gid = [], 22600001
    day = pd.Timestamp("2026-10-21")
    for i, h in enumerate(teams):
        for a in teams[i + 1:]:
            home_wins = rng.random() < 0.55
            games.append(_game(gid, (day + pd.Timedelta(days=gid % 60)).strftime("%Y-%m-%d"),
                               h, 110 if home_wins else 100, a, 100 if home_wins else 110))
            gid += 1
    sched = []
    start = pd.Timestamp("2027-01-10 19:00")
    for k in range(remaining_per_team):
        order = list(rng.permutation(len(teams)))
        for j in range(0, len(order), 2):
            h, a = teams[order[j]], teams[order[j + 1]]
            hc, hn = _split(h)
            ac, an = _split(a)
            sched.append({"gameId": gid, "gameDateTimeEst": (start + pd.Timedelta(days=k)).strftime("%Y-%m-%d %H:%M:%S"),
                          "homeTeamCity": hc, "homeTeamName": hn, "awayTeamCity": ac, "awayTeamName": an})
            gid += 1
    # ein TBD-Spiel (Cup-K.o.) wird uebersprungen
    sched.append({"gameId": gid, "gameDateTimeEst": "2027-01-20 19:00:00", "homeTeamCity": None,
                  "homeTeamName": None, "awayTeamCity": None, "awayTeamName": None})
    return pd.DataFrame(games), pd.DataFrame(sched)


@pytest.fixture(scope="module")
def sim_result():
    games, sched = synthetic_league()
    return ss.run_simulation(SEASON, "projected", games, sched, elo0=np.full(ss.N_TEAMS, 1500.0),
                             n_sims=600, seed=42, today="2027-01-01", log=lambda *a: None)


def test_simulation_invariants(sim_result):
    rep = sim_result["report"]
    assert rep["passed"], rep["issues"]
    assert rep["inputs"]["remaining_games"] == 60          # TBD-Spiel uebersprungen
    for conf in ("east", "west"):
        proj = sim_result["standings"][conf]["projected"]
        assert len(proj) == 15
        assert abs(sum(r["pct_top_seed"] for r in proj) - 1) <= 0.01
        assert abs(sum(r["pct_playoffs"] for r in proj) - 6) <= 0.01
        assert abs(sum(r["pct_playin"] for r in proj) - 4) <= 0.01
        for r in proj:
            assert r["exp_w"] + r["exp_l"] == 33              # 29 gespielt + 4 offen
            assert r["w_p10"] <= r["exp_w"] <= r["w_p90"] or r["w_p10"] == r["w_p90"]
    odds = sim_result["round_odds"]
    assert abs(sum(r["champion"] for r in odds) - 1) <= 0.01
    for r in odds:
        assert 0 <= r["champion"] <= r["finals"] <= r["conf_finals"] <= r["r2"] <= r["playoffs"] <= 1


def test_bracket_path_consistent(sim_result):
    b = sim_result["bracket"]
    for conf in ("east", "west"):
        r1_teams = [t for s in b[conf]["r1"] for t in (s["home_team"], s["away_team"])]
        assert len(set(r1_teams)) == 8
        r1_w = [s["prediction"]["winner"] for s in b[conf]["r1"]]
        assert {b[conf]["r2"][0]["home_team"], b[conf]["r2"][0]["away_team"]} == {r1_w[0], r1_w[3]}
        assert {b[conf]["r2"][1]["home_team"], b[conf]["r2"][1]["away_team"]} == {r1_w[1], r1_w[2]}
        r2_w = [s["prediction"]["winner"] for s in b[conf]["r2"]]
        assert {b[conf]["r3"][0]["home_team"], b[conf]["r3"][0]["away_team"]} == set(r2_w)
        assert b[conf]["finalist"] == b[conf]["r3"][0]["prediction"]["winner"]
        pi = sim_result["play_in"][conf]
        assert [g["game"] for g in pi] == ["7v8", "9v10", "final"]
        assert pi[0]["winner"] == b[conf]["r1"][1]["away_team"]     # Seed 7
        assert pi[2]["winner"] == b[conf]["r1"][0]["away_team"]     # Seed 8
        top6 = set(r1_teams) - {b[conf]["r1"][0]["away_team"], b[conf]["r1"][1]["away_team"]}
        pi_teams = {pi[0]["home_team"], pi[0]["away_team"], pi[1]["home_team"], pi[1]["away_team"]}
        assert len(pi_teams) == 4 and not (pi_teams & top6)
    assert {b["finals"]["home_team"], b["finals"]["away_team"]} == {b["east"]["finalist"], b["west"]["finalist"]}
    # 600 Sims → bei Paarungen mit < 500 Treffern greift der analytische Rueckfall
    assert any(s["prediction"]["source"] == "elo_analytic"
               for s in b["east"]["r2"] + b["west"]["r2"] + [b["finals"]])


def test_same_seed_identical_results():
    games, sched = synthetic_league(remaining_per_team=2, seed=5)
    kw = dict(elo0=np.full(ss.N_TEAMS, 1500.0), n_sims=200, seed=7, today="2027-01-01",
              log=lambda *a: None)
    a = ss.run_simulation(SEASON, "projected", games, sched, **kw)
    b = ss.run_simulation(SEASON, "projected", games, sched, **kw)
    assert a["round_odds"] == b["round_odds"]
    assert a["standings"] == b["standings"]
    assert a["bracket"] == b["bracket"]


def test_daily_seed_is_stable():
    assert ss.daily_seed("2026-11-12") == ss.daily_seed(pd.Timestamp("2026-11-12 23:59")) == 20261112


def test_post_season_mode_uses_final_table():
    games, sched = synthetic_league(seed=2)
    res = ss.run_simulation(SEASON, "post_season_pending", games, sched,
                            elo0=np.full(ss.N_TEAMS, 1500.0), n_sims=300, seed=1,
                            today="2027-04-15", log=lambda *a: None)
    assert res["report"]["inputs"]["remaining_games"] == 0
    for conf in ("east", "west"):
        cur = [r["team"] for r in res["standings"][conf]["current"]]
        proj = res["standings"][conf]["projected"]
        # Tabelle steht fest → jeder Seed mit 100 %
        assert [r["team"] for r in proj] == cur
        assert all(r["pct_top_seed"] in (0.0, 1.0) for r in proj)


def test_history_entry_format(sim_result):
    h = sim_result["history_entry"]
    assert h["date"] == "2027-01-01"
    assert len(h["teams"]) == 30
    assert set(next(iter(h["teams"].values()))) == {"playoffs", "conf_finals", "champion"}
    json.dumps(h)
