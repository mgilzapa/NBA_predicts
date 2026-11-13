"""season_simulator.py — Monte-Carlo-Saisonprognose mit Play-In und Playoff-Bracket.

Ablauf (aufgerufen von fetch_bracket.py, kein eigener Pipeline-Schritt):
  1. Aktuelle Tabelle aus den gespielten Regular-Season-Spielen
  2. Spielwahrscheinlichkeiten: heute aus predictions.xlsx, Tage 1-7 per XGBoost,
     danach ELO mit Heimvorteil (Konstanten aus elo_config.py)
  3. N Simulationen der Restsaison (ELO-Fortschreibung K=20 pro simuliertem Spiel)
  4. Seeding mit offiziellen NBA-Tiebreakern, Play-In, Playoffs Spiel fuer Spiel (2-2-1-1-1)
  5. Auswertung: Prognose-Tabelle, Rundenchancen, wahrscheinlichster Bracket-Pfad, Report
"""
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

from elo_config import ELO_K, ELO_SCALE, ELO_START, HCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_CSV       = os.path.join(BASE_DIR, "data",   "nba_api_games.csv")
PLAYIN_CSV      = os.path.join(BASE_DIR, "data",   "nba_api_playin.csv")
SCHEDULE_CSV    = os.path.join(BASE_DIR, "data",   "schedule_round_1.csv")
ELO_CSV         = os.path.join(BASE_DIR, "data",   "elo_current.csv")
PREDICTIONS_XLSX = os.path.join(BASE_DIR, "output", "predictions.xlsx")

DEFAULT_SIMS     = 50_000
XGB_WINDOW_DAYS  = 7        # Tage 1-7 nach heute per XGBoost
MIN_PATH_SIMS    = 500      # darunter: Serie analytisch per ELO berechnen
RUNTIME_WARN_S   = 180
CHUNK            = 5_000    # Sims pro Tiebreak-Block (Speicher fuer Head-to-Head-Matrizen)
PROB_TOL         = 0.01

# ─────────────────────────────────────────────────────────────
# Team-Stammdaten
# ─────────────────────────────────────────────────────────────
TEAM_INFO = {
    # East
    "Boston Celtics":         ("east", "Atlantic"),
    "Brooklyn Nets":          ("east", "Atlantic"),
    "New York Knicks":        ("east", "Atlantic"),
    "Philadelphia 76ers":     ("east", "Atlantic"),
    "Toronto Raptors":        ("east", "Atlantic"),
    "Chicago Bulls":          ("east", "Central"),
    "Cleveland Cavaliers":    ("east", "Central"),
    "Detroit Pistons":        ("east", "Central"),
    "Indiana Pacers":         ("east", "Central"),
    "Milwaukee Bucks":        ("east", "Central"),
    "Atlanta Hawks":          ("east", "Southeast"),
    "Charlotte Hornets":      ("east", "Southeast"),
    "Miami Heat":             ("east", "Southeast"),
    "Orlando Magic":          ("east", "Southeast"),
    "Washington Wizards":     ("east", "Southeast"),
    # West
    "Denver Nuggets":         ("west", "Northwest"),
    "Minnesota Timberwolves": ("west", "Northwest"),
    "Oklahoma City Thunder":  ("west", "Northwest"),
    "Portland Trail Blazers": ("west", "Northwest"),
    "Utah Jazz":              ("west", "Northwest"),
    "Golden State Warriors":  ("west", "Pacific"),
    "LA Clippers":            ("west", "Pacific"),
    "Los Angeles Lakers":     ("west", "Pacific"),
    "Phoenix Suns":           ("west", "Pacific"),
    "Sacramento Kings":       ("west", "Pacific"),
    "Dallas Mavericks":       ("west", "Southwest"),
    "Houston Rockets":        ("west", "Southwest"),
    "Memphis Grizzlies":      ("west", "Southwest"),
    "New Orleans Pelicans":   ("west", "Southwest"),
    "San Antonio Spurs":      ("west", "Southwest"),
}
TEAMS = sorted(TEAM_INFO)
N_TEAMS = len(TEAMS)
TEAM_IDX = {t: i for i, t in enumerate(TEAMS)}
CONF_NAMES = ("east", "west")
DIV_NAMES = sorted({d for _, d in TEAM_INFO.values()})
CONF_OF = np.array([CONF_NAMES.index(TEAM_INFO[t][0]) for t in TEAMS])
DIV_OF = np.array([DIV_NAMES.index(TEAM_INFO[t][1]) for t in TEAMS])
CONF_TEAMS = [[i for i in range(N_TEAMS) if CONF_OF[i] == c] for c in range(2)]
DIV_TEAMS = [[i for i in range(N_TEAMS) if DIV_OF[i] == d] for d in range(len(DIV_NAMES))]

# R1-Slots (Index wie in bracket.json): 1v8, 2v7, 3v6, 4v5 (Seeds 1-basiert)
R1_SLOTS = [(1, 8), (2, 7), (3, 6), (4, 5)]
R2_FROM = [(0, 3), (1, 2)]      # R2-Slot → welche R1-Slots
SERIES_HOME_GAMES = (1, 2, 5, 7)


# ─────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────
def elo_home_prob(r_home, r_away):
    """Siegwahrscheinlichkeit des Gastgebers inkl. Heimvorteil."""
    return 1.0 / (1.0 + 10.0 ** ((r_away - r_home - HCA) / ELO_SCALE))


def elo_expected(r_a, r_b):
    """Erwartungswert fuer das ELO-Update (ohne Heimvorteil, wie in feature_engineering.py)."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / ELO_SCALE))


def today_et():
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()


def daily_seed(date):
    """Fester Seed pro Tag → zwei Laeufe am selben Tag liefern identische Zahlen."""
    return int(pd.Timestamp(date).strftime("%Y%m%d"))


def _r(x):
    return round(float(x), 9)


def _rec_pct(w, l):
    g = w + l
    return w / g if g > 0 else 0.5


def _clean_datetime(series):
    clean = series.astype(str).str.replace(r"(Z|[+-]\d\d:\d\d)$", "", regex=True)
    return pd.to_datetime(clean, errors="coerce")


# ─────────────────────────────────────────────────────────────
# Daten laden
# ─────────────────────────────────────────────────────────────
def played_regular_season(games_df, season):
    """Gespielte Regular-Season-Spiele (ohne Cup-Finale, nur bekannte Teams)."""
    df = games_df[(games_df["season"] == season) &
                  (games_df["season_type"] == "Regular Season")].copy()
    df = df[df["GAME_ID"].astype(str).str.startswith("2")]          # 006 = Cup-Finale
    df = df[df["hometeamName"].isin(TEAM_IDX) & df["awayteamName"].isin(TEAM_IDX)]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.drop_duplicates(subset=["GAME_ID"])
    return df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def remaining_regular_season(schedule_df, played_ids):
    """Restspielplan: Regular Season (Praefix 002), TBD-Spiele werden uebersprungen."""
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame(columns=["gameId", "date", "home", "away"])
    s = schedule_df.copy()
    s["gameId"] = pd.to_numeric(s["gameId"], errors="coerce")
    s = s.dropna(subset=["gameId"])
    s["gameId"] = s["gameId"].astype(np.int64)
    s = s[s["gameId"].astype(str).str.startswith("2")]
    s["home"] = s["homeTeamCity"].astype(str) + " " + s["homeTeamName"].astype(str)
    s["away"] = s["awayTeamCity"].astype(str) + " " + s["awayTeamName"].astype(str)
    s = s[s["home"].isin(TEAM_IDX) & s["away"].isin(TEAM_IDX)]
    s = s[~s["gameId"].isin(set(played_ids))]
    s["date"] = _clean_datetime(s["gameDateTimeEst"])
    s = s.dropna(subset=["date"]).drop_duplicates(subset=["gameId"])
    s = s.sort_values(["date", "gameId"]).reset_index(drop=True)
    return s[["gameId", "date", "home", "away"]]


def load_elo(path=ELO_CSV):
    elo = np.full(N_TEAMS, float(ELO_START))
    if os.path.exists(path):
        df = pd.read_csv(path)
        for team, val in zip(df["team"], df["elo"]):
            if team in TEAM_IDX and pd.notna(val):
                elo[TEAM_IDX[team]] = float(val)
    return elo


# ─────────────────────────────────────────────────────────────
# Bilanzen
# ─────────────────────────────────────────────────────────────
def team_records(played):
    """W, L, Head-to-Head-Siege (h2h[i, j] = Siege von i gegen j), Punktdifferenz."""
    w = np.zeros(N_TEAMS, dtype=np.int64)
    l = np.zeros(N_TEAMS, dtype=np.int64)
    h2h = np.zeros((N_TEAMS, N_TEAMS), dtype=np.int64)
    pdiff = np.zeros(N_TEAMS, dtype=float)
    for home, away, hs, as_ in zip(played["hometeamName"], played["awayteamName"],
                                   played["homeScore"], played["awayScore"]):
        h, a = TEAM_IDX[home], TEAM_IDX[away]
        if hs > as_:
            w[h] += 1; l[a] += 1; h2h[h, a] += 1
        else:
            w[a] += 1; l[h] += 1; h2h[a, h] += 1
        pdiff[h] += hs - as_
        pdiff[a] += as_ - hs
    return w, l, h2h, pdiff


# ─────────────────────────────────────────────────────────────
# Offizielle NBA-Tiebreaker
# ─────────────────────────────────────────────────────────────
class TiebreakWorld:
    """Eine Liga-Situation (aktuelle Tabelle oder eine Simulation)."""
    __slots__ = ("w", "l", "h2h", "tb", "pct", "div_leaders", "eligible")

    def __init__(self, w, l, h2h, tb):
        self.w = np.asarray(w)
        self.l = np.asarray(l)
        self.h2h = h2h
        self.tb = np.asarray(tb, dtype=float)       # letzte Stufe: Punktdifferenz bzw. Zufall
        g = self.w + self.l
        self.pct = np.where(g > 0, self.w / np.maximum(g, 1), 0.0)
        self.div_leaders = set()
        self.eligible = {0: set(), 1: set()}


TWO_TEAM_CRITERIA = ("h2h", "div_leader", "div_record", "conf_record",
                     "vs_elig_own", "vs_elig_other", "tb")
MULTI_TEAM_CRITERIA = ("div_leader", "h2h", "div_record", "conf_record",
                       "vs_elig_own", "tb")


def _record_vs(world, t, opponents):
    opp = [o for o in opponents if o != t]
    if not opp:
        return 0.5
    return _rec_pct(world.h2h[t, opp].sum(), world.h2h[opp, t].sum())


def _criterion_values(name, group, world):
    if name == "h2h":
        return {t: _record_vs(world, t, group) for t in group}
    if name == "div_leader":
        return {t: 1.0 if t in world.div_leaders else 0.0 for t in group}
    if name == "div_record":
        divs = {DIV_OF[t] for t in group}
        if len(divs) != 1:
            return None                          # nur wenn alle in derselben Division
        members = DIV_TEAMS[DIV_OF[group[0]]]
        return {t: _record_vs(world, t, members) for t in group}
    if name == "conf_record":
        return {t: _record_vs(world, t, CONF_TEAMS[CONF_OF[t]]) for t in group}
    if name == "vs_elig_own":
        return {t: _record_vs(world, t, world.eligible[CONF_OF[t]]) for t in group}
    if name == "vs_elig_other":
        return {t: _record_vs(world, t, world.eligible[1 - CONF_OF[t]]) for t in group}
    if name == "tb":
        return {t: world.tb[t] for t in group}
    raise ValueError(name)


def resolve_tie(group, world):
    """
    Ordnet eine Gruppe mit gleicher Win% nach den offiziellen NBA-Tiebreakern.
    Trennt eine Stufe die Gruppe, wird jede Teilgruppe neu aufgeloest — zwei
    verbleibende Teams beginnen dabei mit dem Zwei-Team-Verfahren von vorn.
    """
    group = list(group)
    if len(group) <= 1:
        return group
    criteria = TWO_TEAM_CRITERIA if len(group) == 2 else MULTI_TEAM_CRITERIA
    for name in criteria:
        vals = _criterion_values(name, group, world)
        if vals is None:
            continue
        levels = sorted({_r(v) for v in vals.values()}, reverse=True)
        if len(levels) == 1:
            continue
        ordered = []
        for lvl in levels:
            sub = [t for t in group if _r(vals[t]) == lvl]
            ordered.extend(resolve_tie(sub, world))
        return ordered
    return sorted(group)                          # komplett gleich → stabil


def rank_by_pct(teams, world):
    """Sortiert nach Win%, Gleichstaende per resolve_tie."""
    by_pct = sorted(teams, key=lambda t: -world.pct[t])
    out, i = [], 0
    while i < len(by_pct):
        j = i + 1
        p = _r(world.pct[by_pct[i]])
        while j < len(by_pct) and _r(world.pct[by_pct[j]]) == p:
            j += 1
        grp = by_pct[i:j]
        out.extend(grp if len(grp) == 1 else resolve_tie(grp, world))
        i = j
    return out


def _compute_eligible(world):
    """Playoff-/Play-In-Teams (Seeds 1-10) fuer die Tiebreak-Stufen 5/6: vorlaeufig nach Win%."""
    for c in (0, 1):
        members = CONF_TEAMS[c]
        pcts = sorted((world.pct[m] for m in members), reverse=True)
        thr = _r(pcts[min(9, len(pcts) - 1)])
        world.eligible[c] = {m for m in members if _r(world.pct[m]) >= thr}


def _compute_div_leaders(world):
    world.div_leaders = set()
    leaders = set()
    for members in DIV_TEAMS:
        best = max(_r(world.pct[m]) for m in members)
        top = [m for m in members if _r(world.pct[m]) == best]
        leaders.add(top[0] if len(top) == 1 else resolve_tie(top, world)[0])
    world.div_leaders = leaders


def seed_world(world):
    """Liefert [east_order, west_order] (Team-Indizes, Seed 1 zuerst)."""
    _compute_eligible(world)
    _compute_div_leaders(world)
    return [rank_by_pct(CONF_TEAMS[c], world) for c in (0, 1)]


# ─────────────────────────────────────────────────────────────
# Aktuelle Tabelle
# ─────────────────────────────────────────────────────────────
def current_standings(played):
    w, l, h2h, pdiff = team_records(played)
    world = TiebreakWorld(w, l, h2h, pdiff)
    orders = seed_world(world)

    # L10 und Streak
    results = {i: [] for i in range(N_TEAMS)}
    for home, away, hw in zip(played["hometeamName"], played["awayteamName"],
                              played["homeScore"] > played["awayScore"]):
        results[TEAM_IDX[home]].append(bool(hw))
        results[TEAM_IDX[away]].append(not bool(hw))

    out = {}
    current_rank = {}
    for c, order in enumerate(orders):
        lead = order[0]
        rows = []
        for rank, t in enumerate(order, start=1):
            res = results[t]
            last10 = res[-10:]
            streak = ""
            if res:
                last = res[-1]
                n = 0
                for r in reversed(res):
                    if r != last:
                        break
                    n += 1
                streak = f"{'W' if last else 'L'}{n}"
            gb = ((w[lead] - w[t]) + (l[t] - l[lead])) / 2.0
            rows.append({
                "rank":   rank,
                "team":   TEAMS[t],
                "w":      int(w[t]),
                "l":      int(l[t]),
                "pct":    round(float(world.pct[t]), 3),
                "gb":     float(gb),
                "l10":    f"{sum(last10)}-{len(last10) - sum(last10)}",
                "streak": streak,
                "division_leader": t in world.div_leaders,
            })
            current_rank[t] = rank
        out[CONF_NAMES[c]] = rows
    return out, current_rank, (w, l, h2h, pdiff)


# ─────────────────────────────────────────────────────────────
# Spielwahrscheinlichkeiten (heute / XGBoost-Fenster / ELO)
# ─────────────────────────────────────────────────────────────
def _season_progress(date):
    year = date.year if date.month >= 8 else date.year - 1
    start = pd.Timestamp(f"{year}-10-01")
    return float(np.clip((date - start).days / 270, 0.0, 1.0))


def schedule_context(all_games_df, remaining, target_idx):
    """
    Rest-/Back-to-Back-/Fatigue-Features fuer die Spiele `target_idx` aus `remaining`,
    berechnet aus gespielten Spielen + Restspielplan.
    """
    timeline = {t: [] for t in TEAMS}   # team → [(date, is_home)]
    if all_games_df is not None and not all_games_df.empty:
        g = all_games_df
        dates = pd.to_datetime(g["GAME_DATE"]).dt.normalize()
        for d, h, a in zip(dates, g["hometeamName"], g["awayteamName"]):
            if h in timeline:
                timeline[h].append((d, 1))
            if a in timeline:
                timeline[a].append((d, 0))
    for d, h, a in zip(remaining["date"].dt.normalize(), remaining["home"], remaining["away"]):
        timeline[h].append((d, 1))
        timeline[a].append((d, 0))
    for t in timeline:
        timeline[t].sort(key=lambda x: x[0])

    def team_ctx(team, date):
        games = timeline[team]
        prev = [g for g in games if g[0] < date]
        rest = float((date - prev[-1][0]).days) if prev else np.nan
        last7 = sum(1 for g in prev if (date - g[0]).days <= 7)
        cons_away = 0
        for g in reversed(prev):
            if g[1] == 0:
                cons_away += 1
            else:
                break
        return rest, last7, cons_away

    contexts = {}
    for i in target_idx:
        row = remaining.iloc[i]
        d = row["date"].normalize()
        ctx = {"is_playoff": 0.0, "is_preseason": 0.0,
               "season_progress": _season_progress(d),
               "is_elimination_game": 0.0, "home_series_wins": 0.0,
               "away_series_wins": 0.0, "series_wins_diff": 0.0}
        for side, team in (("home", row["home"]), ("away", row["away"])):
            rest, last7, cons_away = team_ctx(team, d)
            ctx[f"{side}_rest_days"] = rest
            ctx[f"{side}_is_back_to_back"] = float(rest == 1) if not np.isnan(rest) else 0.0
            ctx[f"{side}_games_last7"] = float(last7)
            ctx[f"{side}_consecutive_away"] = float(cons_away)
        contexts[i] = ctx
    return contexts


def game_probabilities(remaining, today, elo0, predictions_df=None, xgb=None,
                       all_games_df=None, log=print):
    """
    Liefert (fixed_p, source): fixed_p[g] = feste Heimsiegwahrscheinlichkeit
    (heute / XGBoost) oder NaN → ELO in jeder Simulation.
    xgb: dict(model, feature_cols, home_snap, away_snap) oder None.
    """
    G = len(remaining)
    fixed = np.full(G, np.nan)
    source = np.array(["elo"] * G, dtype=object)
    if G == 0:
        return fixed, source

    dates = remaining["date"].dt.normalize()
    is_today = (dates == today).to_numpy()

    # Heute: dieselbe Zahl wie im Today-Tab
    if predictions_df is not None and not predictions_df.empty and is_today.any():
        p = predictions_df.copy()
        by_id = {}
        by_team = {}
        if "gameId" in p.columns:
            for gid, prob in zip(pd.to_numeric(p["gameId"], errors="coerce"), p["probability_home_win"]):
                if pd.notna(gid) and pd.notna(prob):
                    by_id[int(gid)] = float(prob)
        for h, a, prob in zip(p.get("Home Team", []), p.get("Away Team", []), p["probability_home_win"]):
            if pd.notna(prob):
                by_team[(h, a)] = float(prob)
        for g in np.where(is_today)[0]:
            row = remaining.iloc[g]
            prob = by_id.get(int(row["gameId"]), by_team.get((row["home"], row["away"])))
            if prob is not None:
                fixed[g] = prob
                source[g] = "predictions"

    # Heute ohne Prediction + Tage 1-7: XGBoost auf dem eingefrorenen Team-Snapshot
    window_end = today + pd.Timedelta(days=XGB_WINDOW_DAYS)
    xgb_idx = [g for g in range(G)
               if np.isnan(fixed[g]) and today <= dates.iloc[g] <= window_end]
    if xgb and xgb_idx:
        try:
            from fetch_bracket import build_feature_row
            ctxs = schedule_context(all_games_df, remaining, xgb_idx)
            rows = []
            for g in xgb_idx:
                r = remaining.iloc[g]
                ctx = dict(ctxs[g])
                ctx["home_elo"] = float(elo0[TEAM_IDX[r["home"]]])
                ctx["away_elo"] = float(elo0[TEAM_IDX[r["away"]]])
                rows.append(build_feature_row(r["home"], r["away"], xgb["home_snap"],
                                              xgb["away_snap"], xgb["feature_cols"], context=ctx))
            probs = xgb["model"].predict_proba(np.vstack(rows))[:, 1]
            for g, prob in zip(xgb_idx, probs):
                fixed[g] = float(prob)
                source[g] = "xgboost"
        except Exception as e:   # Modell-Problem → ELO-Rueckfall, Simulation laeuft weiter
            log(f"  WARNUNG: XGBoost-Wahrscheinlichkeiten fehlgeschlagen, nutze ELO: {e}")
    return fixed, source


# ─────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────
def simulate_regular_season(rng, n, home_idx, away_idx, fixed_p, elo0):
    """R[g, s] = Heimsieg in Spiel g, Sim s. ELO-Update K=20 nach jedem Spiel."""
    G = len(home_idx)
    elo = np.repeat(np.asarray(elo0, dtype=float)[:, None], n, axis=1)   # (30, n)
    R = np.empty((G, n), dtype=bool)
    for g in range(G):
        h, a = home_idx[g], away_idx[g]
        rh, ra = elo[h].copy(), elo[a].copy()
        p = fixed_p[g] if not np.isnan(fixed_p[g]) else elo_home_prob(rh, ra)
        win = rng.random(n) < p
        R[g] = win
        delta = ELO_K * (win - elo_expected(rh, ra))
        elo[h] = rh + delta
        elo[a] = ra - delta
    return R, elo


def _single_game(rng, elo, host, guest, known=None):
    """Ein Spiel pro Sim (Play-In). known: {frozenset(x,y): winner} echter Ergebnisse."""
    n = len(host)
    cols = np.arange(n)
    rh, rg = elo[host, cols], elo[guest, cols]
    win = rng.random(n) < elo_home_prob(rh, rg)
    fixed = np.zeros(n, dtype=bool)
    for pair, winner in (known or {}).items():
        x, y = tuple(pair)
        m = ((host == x) & (guest == y)) | ((host == y) & (guest == x))
        win[m] = host[m] == winner
        fixed |= m
    delta = ELO_K * (win - elo_expected(rh, rg)) * (~fixed)
    elo[host, cols] = rh + delta
    elo[guest, cols] = rg - delta
    return np.where(win, host, guest), np.where(win, guest, host)


def _series(rng, elo, home, away, known=None):
    """Best-of-7 Spiel fuer Spiel (2-2-1-1-1), startet ggf. vom echten Serienstand."""
    n = len(home)
    cols = np.arange(n)
    hw = np.zeros(n, dtype=np.int8)
    aw = np.zeros(n, dtype=np.int8)
    for pair, wins in (known or {}).items():
        x, y = tuple(pair)
        m = (home == x) & (away == y)
        hw[m], aw[m] = wins[x], wins[y]
        m = (home == y) & (away == x)
        hw[m], aw[m] = wins[y], wins[x]
    for _ in range(7):
        u = rng.random(n)
        active = (hw < 4) & (aw < 4)
        if not active.any():
            break
        gnum = hw + aw + 1
        home_hosts = np.isin(gnum, SERIES_HOME_GAMES)
        host = np.where(home_hosts, home, away)
        guest = np.where(home_hosts, away, home)
        rh, rg = elo[host, cols], elo[guest, cols]
        host_win = u < elo_home_prob(rh, rg)
        home_win = np.where(home_hosts, host_win, ~host_win)
        hw += (active & home_win).astype(np.int8)
        aw += (active & ~home_win).astype(np.int8)
        delta = ELO_K * (host_win - elo_expected(rh, rg)) * active
        elo[host, cols] = rh + delta
        elo[guest, cols] = rg - delta
    winner = np.where(hw == 4, home, away)
    return hw, aw, winner


def simulate_postseason(rng, elo, seeds, pct, known_playin=None, known_series=None):
    """
    seeds: (n, 2, 15) Team-Indizes nach Seed (vor dem Play-In).
    Liefert dict mit Play-In-Ergebnissen und allen 15 Serien-Slots.
    """
    n = seeds.shape[0]
    out = {"playin": {}, "series": {}}
    champs = []
    for c in (0, 1):
        s = seeds[:, c, :]
        w1, l1 = _single_game(rng, elo, s[:, 6], s[:, 7], known_playin)
        w2, l2 = _single_game(rng, elo, s[:, 8], s[:, 9], known_playin)
        w3, l3 = _single_game(rng, elo, l1, w2, known_playin)
        out["playin"][c] = {"g1": (s[:, 6], s[:, 7], w1), "g2": (s[:, 8], s[:, 9], w2),
                            "g3": (l1, w2, w3)}
        final = s[:, :8].copy()
        final[:, 6] = w1
        final[:, 7] = w3

        r1_win, r1_wseed = [], []
        for slot, (hi, lo) in enumerate(R1_SLOTS):
            home, away = final[:, hi - 1], final[:, lo - 1]
            hw, aw, win = _series(rng, elo, home, away, known_series)
            out["series"][(c, 1, slot)] = (home, away, hw, aw)
            r1_win.append(win)
            r1_wseed.append(np.where(win == home, hi, lo))

        r2_win, r2_wseed = [], []
        for slot, (a, b) in enumerate(R2_FROM):
            ta, tb_, sa, sb = r1_win[a], r1_win[b], r1_wseed[a], r1_wseed[b]
            home = np.where(sa < sb, ta, tb_)
            away = np.where(sa < sb, tb_, ta)
            hw, aw, win = _series(rng, elo, home, away, known_series)
            out["series"][(c, 2, slot)] = (home, away, hw, aw)
            r2_win.append(win)
            r2_wseed.append(np.minimum(sa, sb) * (win == home) + np.maximum(sa, sb) * (win != home))

        sa, sb = r2_wseed
        home = np.where(sa < sb, r2_win[0], r2_win[1])
        away = np.where(sa < sb, r2_win[1], r2_win[0])
        hw, aw, win = _series(rng, elo, home, away, known_series)
        out["series"][(c, 3, 0)] = (home, away, hw, aw)
        champs.append(win)

    cols = np.arange(n)
    pe, pw = pct[cols, champs[0]], pct[cols, champs[1]]
    coin = rng.random(n) < 0.5
    east_home = (pe > pw) | ((pe == pw) & coin)
    home = np.where(east_home, champs[0], champs[1])
    away = np.where(east_home, champs[1], champs[0])
    hw, aw, win = _series(rng, elo, home, away, known_series)
    out["series"][(None, 4, 0)] = (home, away, hw, aw)
    out["champion"] = win
    return out


def seed_all_sims(R, home_idx, away_idx, base_w, base_l, base_h2h, tb, log=print):
    """Seeding fuer alle Sims (Tiebreaker in Python, H2H blockweise)."""
    G, n = R.shape
    Wt = np.repeat(base_w[:, None], n, axis=1).astype(np.int32)
    Lt = np.repeat(base_l[:, None], n, axis=1).astype(np.int32)
    for g in range(G):
        h, a = home_idx[g], away_idx[g]
        r = R[g]
        Wt[h] += r; Lt[a] += r
        Wt[a] += ~r; Lt[h] += ~r
    W, L = Wt.T.copy(), Lt.T.copy()                 # (n, 30)

    seeds = np.empty((n, 2, 15), dtype=np.int16)
    if G == 0:                                      # Tabelle steht fest → einmal seeden
        world = TiebreakWorld(base_w, base_l, base_h2h, tb[0])
        orders = seed_world(world)
        seeds[:, 0, :] = orders[0]
        seeds[:, 1, :] = orders[1]
        return seeds, W, L

    for start in range(0, n, CHUNK):
        stop = min(n, start + CHUNK)
        m = stop - start
        H = np.repeat(base_h2h[:, :, None], m, axis=2).astype(np.int16)   # (30, 30, m)
        for g in range(G):
            h, a = home_idx[g], away_idx[g]
            r = R[g, start:stop]
            H[h, a] += r
            H[a, h] += ~r
        H = np.ascontiguousarray(H.transpose(2, 0, 1))                     # (m, 30, 30)
        for i in range(m):
            s = start + i
            world = TiebreakWorld(W[s], L[s], H[i], tb[s])
            orders = seed_world(world)
            seeds[s, 0, :] = orders[0]
            seeds[s, 1, :] = orders[1]
    return seeds, W, L


# ─────────────────────────────────────────────────────────────
# Auswertung
# ─────────────────────────────────────────────────────────────
def _pair_mask(home, away, x, y):
    return ((home == x) & (away == y)) | ((home == y) & (away == x))


def _series_stats(slot_arrays, x, y, elo0, mask=None):
    """Statistik einer Paarung x (Heim im Pfad) vs y aus den passenden Sims, sonst analytisch."""
    from fetch_bracket import simulate_series
    home, away, hw, aw = slot_arrays
    m = _pair_mask(home, away, x, y)
    if mask is not None:
        m &= mask
    cnt = int(m.sum())
    if cnt >= MIN_PATH_SIMS:
        wx = np.where(home == x, hw, aw)[m]
        wy = np.where(home == x, aw, hw)[m]
        p_x = float(np.mean(wx == 4))
        counts = Counter(zip(wx.tolist(), wy.tolist()))
        dist = {f"{a}-{b}": c / cnt for (a, b), c in counts.items()}
        return p_x, dist, cnt, "simulation"
    p1 = float(elo_home_prob(elo0[x], elo0[y]))
    p2 = 1.0 - float(elo_home_prob(elo0[y], elo0[x]))
    p_x, _, dist = simulate_series(0, 0, p1, p2)
    return float(p_x), dist, cnt, "elo_analytic"


def projected_series_obj(home, away, p_home, dist, n_matched, source):
    """Serienobjekt im bracket.json-Format, markiert als Prognose."""
    home_wins_series = p_home >= 0.5
    winner = home if home_wins_series else away
    winner_outcomes = {k: v for k, v in dist.items() if k.startswith("4-") == home_wins_series}
    best = (max(winner_outcomes, key=winner_outcomes.get) if winner_outcomes
            else ("4-2" if home_wins_series else "2-4"))
    ordered = dict(sorted(dist.items(), key=lambda kv: -kv[1]))
    return {
        "home_team": home,
        "away_team": away,
        "home_wins": 0,
        "away_wins": 0,
        "status":    "projected",
        "winner":    None,
        "projected": True,
        "prediction": {
            "winner":               winner,
            "win_probability":      round(max(p_home, 1 - p_home), 4),
            "predicted_length":     sum(int(v) for v in best.split("-")),
            "predicted_score":      best,
            "outcome_distribution": {k: round(v, 3) for k, v in ordered.items()},
            "team_probabilities":   {home: round(p_home, 4), away: round(1 - p_home, 4)},
            "sims_matched":         n_matched,
            "source":               source,
        },
    }


def _most_common(codes):
    vals, counts = np.unique(codes, return_counts=True)
    return int(vals[np.argmax(counts)]), int(counts.max())


def _game_prob(host_arr, guest_arr, winner_arr, x, y, elo0, mask=None):
    """P(x gewinnt ein Einzelspiel gegen y) aus passenden Sims, sonst ELO (x als Gastgeber)."""
    m = (host_arr == x) & (guest_arr == y)
    if mask is not None:
        m &= mask
    cnt = int(m.sum())
    if cnt >= MIN_PATH_SIMS:
        return float(np.mean(winner_arr[m] == x)), cnt, "simulation"
    return float(elo_home_prob(elo0[x], elo0[y])), cnt, "elo_analytic"


def build_bracket_path(post, seeds, elo0, mean_pct, known_playin=None):
    """
    Wahrscheinlichster Bracket-Pfad: R1-Slots nacheinander als haeufigste Paarung
    (bedingt auf die bereits gewaehlten Slots), Sieger = hoehere Siegchance in den
    Sims mit genau dieser Paarung. Play-In wird auf denselben Sims bedingt.
    """
    known_playin = known_playin or {}
    n = seeds.shape[0]
    bracket = {}
    play_in = {}
    conf_champ = {}
    for c in (0, 1):
        conf = CONF_NAMES[c]
        mask = np.ones(n, dtype=bool)
        r1_pairs = []
        for slot in range(4):
            home, away, _, _ = post["series"][(c, 1, slot)]
            code, _ = _most_common((home * 100 + away)[mask])
            h, a = divmod(code, 100)
            mask &= (home == h) & (away == a)
            r1_pairs.append((h, a))

        # Play-In-Belegung (Seeds 7-10 vor dem Play-In), bedingt auf den R1-Pfad
        s = seeds[:, c, :].astype(np.int64)
        pi_code = s[:, 6] * 1_000_000 + s[:, 7] * 10_000 + s[:, 8] * 100 + s[:, 9]
        code, _ = _most_common(pi_code[mask])
        p7, p8, p9, p10 = code // 1_000_000, (code // 10_000) % 100, (code // 100) % 100, code % 100
        seeding_mask = pi_code == code
        final7, final8 = r1_pairs[1][1], r1_pairs[0][1]
        g1h, g1a, g1w = post["playin"][c]["g1"]
        g2h, g2a, g2w = post["playin"][c]["g2"]
        g3h, g3a, g3w = post["playin"][c]["g3"]
        w1 = final7
        l1 = p8 if w1 == p7 else p7
        if final8 in (p9, p10):
            w2 = final8
        else:
            code2, _ = _most_common(g2w[mask & seeding_mask]) if (mask & seeding_mask).any() else (p9, 0)
            w2 = code2
        w3 = final8

        def pi_game(label, key, host, guest, winner, seed_h, seed_g, arrays, cond):
            pair = frozenset((host, guest))
            if pair in known_playin:
                real_w = known_playin[pair]
                probs = {TEAMS[host]: 1.0 if real_w == host else 0.0,
                         TEAMS[guest]: 1.0 if real_w == guest else 0.0}
                return {"game": key, "label": label, "home_team": TEAMS[host],
                        "away_team": TEAMS[guest], "home_seed": seed_h, "away_seed": seed_g,
                        "winner": TEAMS[real_w], "status": "complete", "projected": False,
                        "team_probabilities": probs}, real_w
            p, cnt, src = _game_prob(*arrays, host, guest, elo0, cond)
            return {"game": key, "label": label, "home_team": TEAMS[host],
                    "away_team": TEAMS[guest], "home_seed": seed_h, "away_seed": seed_g,
                    "winner": TEAMS[winner], "status": "projected", "projected": True,
                    "team_probabilities": {TEAMS[host]: round(p, 4), TEAMS[guest]: round(1 - p, 4)},
                    "sims_matched": cnt, "source": src}, winner

        # Siegchance pro Spiel: alle Sims mit genau dieser Paarung in diesem Spiel
        g1, _ = pi_game("7 vs 8", "7v8", p7, p8, w1, 7, 8, (g1h, g1a, g1w), None)
        g2, _ = pi_game("9 vs 10", "9v10", p9, p10, w2, 9, 10, (g2h, g2a, g2w), None)
        g3, _ = pi_game("Final 8th seed", "final", l1, w2, w3,
                        8 if l1 == p8 else 7, 9 if w2 == p9 else 10, (g3h, g3a, g3w), None)
        play_in[conf] = [g1, g2, g3]

        # Serien
        r1 = []
        seed_of = {}
        for slot, (h, a) in enumerate(r1_pairs):
            seed_of[h], seed_of[a] = R1_SLOTS[slot]
            p, dist, cnt, src = _series_stats(post["series"][(c, 1, slot)], h, a, elo0)
            r1.append(projected_series_obj(TEAMS[h], TEAMS[a], p, dist, cnt, src))
        r1_winners = [TEAM_IDX[x["prediction"]["winner"]] for x in r1]

        r2 = []
        for slot, (sa, sb) in enumerate(R2_FROM):
            ta, tb_ = r1_winners[sa], r1_winners[sb]
            h, a = (ta, tb_) if seed_of[ta] < seed_of[tb_] else (tb_, ta)
            p, dist, cnt, src = _series_stats(post["series"][(c, 2, slot)], h, a, elo0)
            r2.append(projected_series_obj(TEAMS[h], TEAMS[a], p, dist, cnt, src))
        r2_winners = [TEAM_IDX[x["prediction"]["winner"]] for x in r2]

        ta, tb_ = r2_winners
        h, a = (ta, tb_) if seed_of[ta] < seed_of[tb_] else (tb_, ta)
        p, dist, cnt, src = _series_stats(post["series"][(c, 3, 0)], h, a, elo0)
        r3 = [projected_series_obj(TEAMS[h], TEAMS[a], p, dist, cnt, src)]
        conf_champ[c] = TEAM_IDX[r3[0]["prediction"]["winner"]]
        bracket[conf] = {"r1": r1, "r2": r2, "r3": r3, "finalist": r3[0]["prediction"]["winner"]}

    e, w = conf_champ[0], conf_champ[1]
    fh, fa, _, _ = post["series"][(None, 4, 0)]
    m = _pair_mask(fh, fa, e, w)
    if m.sum() > 0:
        h, a = (e, w) if np.mean(fh[m] == e) >= 0.5 else (w, e)
    else:
        h, a = (e, w) if mean_pct[e] >= mean_pct[w] else (w, e)
    p, dist, cnt, src = _series_stats(post["series"][(None, 4, 0)], h, a, elo0)
    bracket["finals"] = projected_series_obj(TEAMS[h], TEAMS[a], p, dist, cnt, src)
    return bracket, play_in


def _round_odds(seeds, post):
    n = seeds.shape[0]
    reach = {k: np.zeros(N_TEAMS) for k in ("play_in", "playoffs", "r2", "conf_finals",
                                            "finals", "champion")}
    for c in (0, 1):
        for k in range(6, 10):
            reach["play_in"] += np.bincount(seeds[:, c, k], minlength=N_TEAMS)
    for (c, rnd, slot), (home, away, _, _) in post["series"].items():
        key = {1: "playoffs", 2: "r2", 3: "conf_finals", 4: "finals"}[rnd]
        reach[key] += np.bincount(home, minlength=N_TEAMS) + np.bincount(away, minlength=N_TEAMS)
    reach["champion"] += np.bincount(post["champion"], minlength=N_TEAMS)
    return {k: v / n for k, v in reach.items()}


# ─────────────────────────────────────────────────────────────
# Plausibilitaetschecks
# ─────────────────────────────────────────────────────────────
def run_checks(mean_w, mean_l, total_games, standings_proj, odds, bracket, play_in,
               runtime_s, n_sims):
    checks, issues, warnings = [], [], []

    def check(name, ok, detail, severity="issue"):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})
        if not ok:
            (issues if severity == "issue" else warnings).append(f"{name}: {detail}")

    diff = np.abs(mean_w + mean_l - total_games)
    check("games_per_team", diff.max() < 1e-6,
          f"max. Abweichung erwartete W+L vs. Spiele: {diff.max():.6f}")
    short = [TEAMS[t] for t in range(N_TEAMS) if total_games[t] < 82]
    checks.append({"name": "tbd_games", "passed": True,
                   "detail": (f"{len(short)} Teams mit < 82 Spielen (TBD-Cup-Spiele im Spielplan)"
                              if short else "alle Teams 82 Spiele")})
    check("league_wins_equal_losses", abs(mean_w.sum() - mean_l.sum()) < 1e-6,
          f"Summe W {mean_w.sum():.3f} vs. Summe L {mean_l.sum():.3f}")

    all_probs = [v for arr in odds.values() for v in arr]
    for conf in CONF_NAMES:
        for row in standings_proj[conf]:
            all_probs += [row["pct_playoffs"], row["pct_playin"], row["pct_top_seed"]]
    check("probabilities_in_range", all(0 <= p <= 1 for p in all_probs),
          "alle Wahrscheinlichkeiten in [0, 1]")
    for conf in CONF_NAMES:
        rows = standings_proj[conf]
        for key, target in (("pct_top_seed", 1), ("pct_playoffs", 6), ("pct_playin", 4)):
            s = sum(r[key] for r in rows)
            check(f"{conf}_{key}_sum", abs(s - target) <= PROB_TOL,
                  f"Summe {s:.4f} (Soll {target})")
    champ = float(odds["champion"].sum())
    check("champion_sum", abs(champ - 1) <= PROB_TOL, f"Summe {champ:.4f} (Soll 1)")

    # Bracket-Pfad: keine Doppelbelegung, Sieger kommen weiter
    ok, detail = True, "keine Doppelbelegung, Pfad konsistent"
    for conf in CONF_NAMES:
        r1_teams = [t for s in bracket[conf]["r1"] for t in (s["home_team"], s["away_team"])]
        if len(set(r1_teams)) != 8:
            ok, detail = False, f"{conf}: R1 enthaelt doppelte Teams"
        pi_teams = {play_in[conf][0]["home_team"], play_in[conf][0]["away_team"],
                    play_in[conf][1]["home_team"], play_in[conf][1]["away_team"]}
        top6 = {s[k] for s in bracket[conf]["r1"] for k in ("home_team", "away_team")} - {
            bracket[conf]["r1"][0]["away_team"], bracket[conf]["r1"][1]["away_team"]}
        if len(pi_teams) != 4 or pi_teams & top6:
            ok, detail = False, f"{conf}: Play-In-Teams doppelt"
        for rnd, prev, pairs in (("r2", "r1", R2_FROM), ("r3", "r2", [(0, 1)])):
            winners = [s["prediction"]["winner"] for s in bracket[conf][prev]]
            for s, (a, b) in zip(bracket[conf][rnd], pairs):
                if {s["home_team"], s["away_team"]} != {winners[a], winners[b]}:
                    ok, detail = False, f"{conf} {rnd}: Paarung passt nicht zu den Vorrunden-Siegern"
    fin = bracket["finals"]
    if {fin["home_team"], fin["away_team"]} != {bracket["east"]["finalist"], bracket["west"]["finalist"]}:
        ok, detail = False, "Finals passen nicht zu den Conference-Siegern"
    check("bracket_path_consistent", ok, detail)

    check("runtime", runtime_s <= RUNTIME_WARN_S,
          f"{runtime_s:.1f} s fuer {n_sims} Sims (Schwelle {RUNTIME_WARN_S} s)", severity="warning")
    return checks, issues, warnings


# ─────────────────────────────────────────────────────────────
# Einstieg
# ─────────────────────────────────────────────────────────────
def run_simulation(season, mode, games_df, schedule_df=None, elo0=None, predictions_df=None,
                   xgb=None, n_sims=DEFAULT_SIMS, seed=None, today=None,
                   known_playin=None, known_series=None, log=print):
    """
    Fuehrt die komplette Saison-Prognose aus.
    Rueckgabe: dict(standings, play_in, round_odds, bracket, history_entry, report)
    """
    t0 = time.time()
    today = pd.Timestamp(today).normalize() if today is not None else today_et()
    seed = daily_seed(today) if seed is None else seed
    rng = np.random.default_rng(seed)
    elo0 = load_elo() if elo0 is None else np.asarray(elo0, dtype=float)

    played = played_regular_season(games_df, season)
    if mode == "projected":
        remaining = remaining_regular_season(schedule_df, played["GAME_ID"].tolist())
    else:
        remaining = remaining_regular_season(None, [])

    standings_cur, current_rank, (bw, bl, bh2h, pdiff) = current_standings(played)

    fixed_p, source = game_probabilities(remaining, today, elo0, predictions_df, xgb,
                                         all_games_df=games_df, log=log)
    home_idx = np.array([TEAM_IDX[t] for t in remaining["home"]], dtype=np.int64)
    away_idx = np.array([TEAM_IDX[t] for t in remaining["away"]], dtype=np.int64)
    log(f"  Saison {season} | Modus {mode} | {len(played)} gespielt | {len(remaining)} offen "
        f"| heute {int((source == 'predictions').sum())} | XGBoost {int((source == 'xgboost').sum())} "
        f"| ELO {int((source == 'elo').sum())} | {n_sims} Sims | Seed {seed}")

    R, elo = simulate_regular_season(rng, n_sims, home_idx, away_idx, fixed_p, elo0)
    t_reg = time.time()
    if len(remaining) > 0:
        tb = rng.random((n_sims, N_TEAMS))              # Punktdifferenz unbekannt → Zufall
    else:
        tb = np.repeat(pdiff[None, :], n_sims, axis=0)  # Tabelle steht fest → echte Punktdifferenz
    seeds, W, L = seed_all_sims(R, home_idx, away_idx, bw, bl, bh2h, tb, log=log)
    t_seed = time.time()
    del R

    games_tot = np.maximum(W + L, 1)
    pct = W / games_tot
    post = simulate_postseason(rng, elo, seeds.astype(np.int64), pct,
                               known_playin=known_playin, known_series=known_series)
    t_post = time.time()

    # ── Prognose-Tabelle ─────────────────────────────────────
    seed_pos = np.empty((n_sims, N_TEAMS), dtype=np.int16)
    for c in (0, 1):
        for k in range(15):
            seed_pos[np.arange(n_sims), seeds[:, c, k]] = k + 1
    mean_w = W.mean(axis=0)
    mean_l = L.mean(axis=0)
    total_games = (bw + bl + np.bincount(home_idx, minlength=N_TEAMS)
                   + np.bincount(away_idx, minlength=N_TEAMS))
    mean_pct = mean_w / np.maximum(total_games, 1)
    standings_proj = {}
    for c in (0, 1):
        rows = []
        for t in CONF_TEAMS[c]:
            sp = seed_pos[:, t]
            exp_w = int(round(mean_w[t]))
            rows.append({
                "team":         TEAMS[t],
                "exp_w":        exp_w,
                "exp_l":        int(total_games[t] - exp_w),
                "w_p10":        int(np.percentile(W[:, t], 10)),
                "w_p90":        int(np.percentile(W[:, t], 90)),
                "avg_seed":     round(float(sp.mean()), 2),
                "pct_playoffs": round(float(np.mean(sp <= 6)), 4),
                "pct_playin":   round(float(np.mean((sp >= 7) & (sp <= 10))), 4),
                "pct_top_seed": round(float(np.mean(sp == 1)), 4),
                "_mean_w":      float(mean_w[t]),
            })
        rows.sort(key=lambda r: (r["avg_seed"], -r["_mean_w"]))
        for rank, r in enumerate(rows, start=1):
            r["rank"] = rank
            r["rank_change"] = int(current_rank[TEAM_IDX[r["team"]]] - rank)
            del r["_mean_w"]
        standings_proj[CONF_NAMES[c]] = [{"rank": r.pop("rank"), **r} for r in rows]

    odds = _round_odds(seeds, post)
    round_odds = sorted(
        [{"team": TEAMS[t], "conf": CONF_NAMES[CONF_OF[t]],
          **{k: round(float(v[t]), 4) for k, v in odds.items()}} for t in range(N_TEAMS)],
        key=lambda r: (-r["champion"], -r["finals"], r["team"]))

    bracket, play_in = build_bracket_path(post, seeds, elo0, mean_pct, known_playin)
    runtime = time.time() - t0
    log(f"  Laufzeit: Regular Season {t_reg - t0:.1f}s | Seeding {t_seed - t_reg:.1f}s | "
        f"Postseason {t_post - t_seed:.1f}s | gesamt {runtime:.1f}s")

    checks, issues, warnings = run_checks(mean_w, mean_l, total_games, standings_proj, odds,
                                          bracket, play_in, runtime, n_sims)
    status = "critical" if issues else ("warning" if warnings else "ok")
    report = {
        "generated_at":    pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%dT%H:%M:%S"),
        "season":          season,
        "mode":            mode,
        "sims":            n_sims,
        "seed":            seed,
        "runtime_seconds": round(runtime, 1),
        "status":          status,
        "passed":          not issues,
        "checks":          checks,
        "issues":          issues,
        "warnings":        warnings,
        "inputs": {
            "played_games":            int(len(played)),
            "remaining_games":         int(len(remaining)),
            "today_from_predictions":  int((source == "predictions").sum()),
            "xgboost_games":           int((source == "xgboost").sum()),
            "elo_games":               int((source == "elo").sum()),
        },
    }
    history_entry = {
        "date": today.strftime("%Y-%m-%d"),
        "teams": {r["team"]: {"playoffs": r["playoffs"], "conf_finals": r["conf_finals"],
                              "champion": r["champion"]} for r in round_odds},
    }
    return {
        "standings": {c: {"current": standings_cur[c], "projected": standings_proj[c]}
                      for c in CONF_NAMES},
        "play_in":   play_in,
        "round_odds": round_odds,
        "bracket":   bracket,
        "history_entry": history_entry,
        "report":    report,
    }
