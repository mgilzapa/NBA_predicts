import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

base_dir = os.path.dirname(os.path.dirname(__file__))
PLAYER_BOX = os.path.join(base_dir, "data", "player_boxscores.csv")
base_games_path = os.path.join(base_dir, "data", "base_games.csv")

df = pd.read_csv(base_games_path)
df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
df["is_playoff"] = df.get("gameSubtype", pd.Series("", index=df.index)).str.contains(
    "Playoff|playoff|First Round|Conference", na=False
).astype(int)

# ─────────────────────────────────────────────────────────────
# BASIS: Team-History (Heim + Auswärts zusammengeführt)
# ─────────────────────────────────────────────────────────────
home_games = pd.DataFrame({
    "date":           df["gameDateTimeEst"],
    "game_id":        df["gameId"],
    "team":           df["hometeamName"],
    "opponent":       df["awayteamName"],
    "win":            (df["homeScore"] > df["awayScore"]).astype(int),
    "points":         df["homeScore"],
    "points_allowed": df["awayScore"],
    "is_home":        1,
    "is_playoff":     df["is_playoff"],
})
away_games = pd.DataFrame({
    "date":           df["gameDateTimeEst"],
    "game_id":        df["gameId"],
    "team":           df["awayteamName"],
    "opponent":       df["hometeamName"],
    "win":            (df["awayScore"] > df["homeScore"]).astype(int),
    "points":         df["awayScore"],
    "points_allowed": df["homeScore"],
    "is_home":        0,
    "is_playoff":     df["is_playoff"],
})

team_history = (
    pd.concat([home_games, away_games], ignore_index=True)
    .sort_values(["team", "date"])
    .reset_index(drop=True)
)

# ─────────────────────────────────────────────────────────────
# ELO-RATING
# ─────────────────────────────────────────────────────────────
ELO_START = 1500
ELO_K = 20

def expected_elo(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

# Alle Spiele chronologisch sortiert
elo_games = (
    df[["gameDateTimeEst", "gameId", "hometeamName", "awayteamName", "homeScore", "awayScore"]]
    .sort_values("gameDateTimeEst")
    .reset_index(drop=True)
)

elo_ratings = {}  # team → aktuelles ELO
elo_games_played = {}
home_elo_before = []
away_elo_before = []
home_elo_games_played = []
away_elo_games_played = []

for _, row in elo_games.iterrows():
    home = row["hometeamName"]
    away = row["awayteamName"]

    r_home = elo_ratings.get(home, ELO_START)
    r_away = elo_ratings.get(away, ELO_START)

    # ELO VOR dem Spiel speichern (kein Leakage)
    home_elo_before.append(r_home)
    away_elo_before.append(r_away)
    home_elo_games_played.append(elo_games_played.get(home, 0))  
    away_elo_games_played.append(elo_games_played.get(away, 0))  

    # Ergebnis
    home_won = 1 if row["homeScore"] > row["awayScore"] else 0
    exp_home = expected_elo(r_home, r_away)

    # Update
    elo_ratings[home] = r_home + ELO_K * (home_won - exp_home)
    elo_ratings[away] = r_away + ELO_K * ((1 - home_won) - (1 - exp_home))
    elo_games_played[home] = elo_games_played.get(home, 0) + 1  
    elo_games_played[away] = elo_games_played.get(away, 0) + 1  

elo_games["home_elo"] = home_elo_before
elo_games["away_elo"] = away_elo_before
elo_games["elo_diff"]  = elo_games["home_elo"] - elo_games["away_elo"]
elo_games["home_elo_games_played"] = home_elo_games_played  
elo_games["away_elo_games_played"] = away_elo_games_played  

# In df mergen
df = df.merge(
    elo_games[["gameId", "home_elo", "away_elo", "elo_diff",
               "home_elo_games_played", "away_elo_games_played"]],  # NEU
    on="gameId", how="left"
)

ELO_WARMUP = 20
mask = (df["home_elo_games_played"] < ELO_WARMUP) | (df["away_elo_games_played"] < ELO_WARMUP)
df.loc[mask, ["home_elo", "away_elo", "elo_diff"]] = np.nan

# ─────────────────────────────────────────────────────────────
# FEATURE 1: Letzte 5 / letzte 3 / letzte 10 Winrate + Trend
# ─────────────────────────────────────────────────────────────
def shifted_rolling(series, window, min_p=None):
    return series.shift(1).rolling(window, min_periods=min_p or max(1, window // 2))

for w, col in [(5, "last5_winrate"), (3, "last3_winrate"), (10, "last10_winrate")]:
    team_history[col] = (
        team_history.groupby("team")["win"]
        .transform(lambda x, w=w: shifted_rolling(x, w).mean())
    )

# Trend: verbessert oder verschlechtert sich das Team gerade?
team_history["winrate_trend"] = team_history["last3_winrate"] - team_history["last10_winrate"]

# ─────────────────────────────────────────────────────────────
# FEATURE 2: Punkte (letzte 5, min_periods=3)
# ─────────────────────────────────────────────────────────────
team_history["last5_avg_points"] = (
    team_history.groupby("team")["points"]
    .transform(lambda x: shifted_rolling(x, 5, 3).mean())
)
team_history["last5_avg_points_allowed"] = (
    team_history.groupby("team")["points_allowed"]
    .transform(lambda x: shifted_rolling(x, 5, 3).mean())
)

# ─────────────────────────────────────────────────────────────
# FEATURE 3: Ruhetage & Back-to-Back
# ─────────────────────────────────────────────────────────────
team_history["rest_days"] = (
    team_history.groupby("team")["date"].diff().dt.days
)
team_history["is_back_to_back"] = (team_history["rest_days"] == 1).astype(int)

# ─────────────────────────────────────────────────────────────
# FEATURE 3b: Fatigue – Spiele in letzten 7 Tagen & consecutive away
# ─────────────────────────────────────────────────────────────
team_history["games_last7"] = (
    team_history.groupby("team", group_keys=False)
    .apply(lambda g: (
        g.set_index("date")["win"]
        .shift(1, freq="D")  # kein Leakage
        .rolling("7D")
        .count()
        .reset_index(drop=True)
    ))
    .reset_index(level=0, drop=True)
)

# Consecutive Away Games
def count_consecutive_away(g):
    result = []
    count = 0
    for is_home in g["is_home"].shift(1):  # shift: kein Leakage
        if is_home == 0:
            count += 1
        else:
            count = 0
        result.append(count)
    return pd.Series(result, index=g.index)

team_history["consecutive_away"] = (
    team_history.groupby("team", group_keys=False)
    .apply(count_consecutive_away)
    .reset_index(level=0, drop=True)
)

# ─────────────────────────────────────────────────────────────
# FEATURE 4: Gegner-Stärke (SOS)
# ─────────────────────────────────────────────────────────────
# Erst eigene Rolling-Winrate berechnen und in separater Spalte speichern
team_history["own_last5_winrate"] = (
    team_history.groupby("team")["win"]
    .transform(lambda x: shifted_rolling(x, 5, 3).mean())
)

# Lookup: für jeden Row die Winrate des Gegners holen
opp_map = team_history.set_index(["date", "team"])["own_last5_winrate"].to_dict()
team_history["opponent_strength"] = team_history.apply(
    lambda r: opp_map.get((r["date"], r["opponent"]), np.nan), axis=1
)

# ─────────────────────────────────────────────────────────────
# FEATURE 5: Heim/Auswärts-Winrate (ohne Leakage)
# ─────────────────────────────────────────────────────────────
team_history["home_winrate"] = (
    team_history.groupby("team", group_keys=False)
    .apply(lambda g: g["win"].where(g["is_home"] == 1).shift(1).expanding().mean())
    .reset_index(level=0, drop=True)
)
team_history["away_winrate"] = (
    team_history.groupby("team", group_keys=False)
    .apply(lambda g: g["win"].where(g["is_home"] == 0).shift(1).expanding().mean())
    .reset_index(level=0, drop=True)
)

# ffill wie vorher, aber dann mit globaler Winrate als Fallback auffüllen
team_history["home_winrate"] = team_history.groupby("team")["home_winrate"].ffill()
team_history["away_winrate"] = team_history.groupby("team")["away_winrate"].ffill()

# Fallback: globale last5_winrate wenn immer noch NaN
team_history["home_winrate"] = team_history["home_winrate"].fillna(team_history["last5_winrate"])
team_history["away_winrate"] = team_history["away_winrate"].fillna(team_history["last5_winrate"])

# ─────────────────────────────────────────────────────────────
# FEATURE 6: Head-to-Head Record (letzte 2 Saisons)
# ─────────────────────────────────────────────────────────────
# NBA-Saison: Oktober–Dezember gehören zur neuen Saison, Januar–Juni zur alten
team_history["season"] = team_history["date"].apply(
    lambda d: d.year if d.month >= 10 else d.year - 1
)
current_season = team_history["season"].max()

h2h = (
    team_history[team_history["season"] >= current_season - 1]
    .groupby(["team", "opponent"], group_keys=False)["win"]
    .apply(lambda x: x.shift(1).expanding().mean())
)
team_history["h2h_winrate"] = h2h

# ─────────────────────────────────────────────────────────────
# FEATURE 7: Enges Spiel (close game) statt falschem OT-Indikator
# ─────────────────────────────────────────────────────────────
df["close_game"] = (abs(df["homeScore"] - df["awayScore"]) <= 5).astype(int)

close_home = df[["gameDateTimeEst", "hometeamName", "close_game"]].rename(
    columns={"hometeamName": "team", "gameDateTimeEst": "date"}
)
close_away = df[["gameDateTimeEst", "awayteamName", "close_game"]].rename(
    columns={"awayteamName": "team", "gameDateTimeEst": "date"}
)
close_all = pd.concat([close_home, close_away], ignore_index=True)
team_history = team_history.merge(close_all, on=["date", "team"], how="left")
team_history["last_game_close"] = team_history.groupby("team")["close_game"].shift(1)

# ─────────────────────────────────────────────────────────────
# FEATURE 8: Division-Rivalität
# ─────────────────────────────────────────────────────────────
divisions = {
    "Boston Celtics": "Atlantic",    "Brooklyn Nets": "Atlantic",
    "New York Knicks": "Atlantic",   "Philadelphia 76ers": "Atlantic",
    "Toronto Raptors": "Atlantic",
    "Chicago Bulls": "Central",      "Cleveland Cavaliers": "Central",
    "Detroit Pistons": "Central",    "Indiana Pacers": "Central",
    "Milwaukee Bucks": "Central",
    "Atlanta Hawks": "Southeast",    "Charlotte Hornets": "Southeast",
    "Miami Heat": "Southeast",       "Orlando Magic": "Southeast",
    "Washington Wizards": "Southeast",
    "Denver Nuggets": "Northwest",   "Minnesota Timberwolves": "Northwest",
    "Oklahoma City Thunder": "Northwest", "Portland Trail Blazers": "Northwest",
    "Utah Jazz": "Northwest",
    "Golden State Warriors": "Pacific", "LA Clippers": "Pacific",
    "Los Angeles Lakers": "Pacific", "Phoenix Suns": "Pacific",
    "Sacramento Kings": "Pacific",
    "Dallas Mavericks": "Southwest", "Houston Rockets": "Southwest",
    "Memphis Grizzlies": "Southwest","New Orleans Pelicans": "Southwest",
    "San Antonio Spurs": "Southwest",
}
df["home_division"] = df["hometeamName"].map(divisions)
df["away_division"] = df["awayteamName"].map(divisions)
df["same_division"]  = (df["home_division"] == df["away_division"]).astype(int)

# ─────────────────────────────────────────────────────────────
# FEATURE 9: Playoff Experience (Spieler-basiert, falls Box vorhanden)
# ─────────────────────────────────────────────────────────────
playoff_exp_home = pd.Series(dtype=float)
playoff_exp_away = pd.Series(dtype=float)

if os.path.exists(PLAYER_BOX):
    print("Lade Spieler-Boxscores...")
    box = pd.read_csv(PLAYER_BOX)

    if "personId" in box.columns:
        box.rename(columns={"personId": "PLAYER_ID"}, inplace=True)

    team_name_mapping = {
        "76ers": "Philadelphia 76ers",   "Bucks": "Milwaukee Bucks",
        "Bulls": "Chicago Bulls",        "Cavaliers": "Cleveland Cavaliers",
        "Celtics": "Boston Celtics",     "Clippers": "LA Clippers",
        "Grizzlies": "Memphis Grizzlies","Hawks": "Atlanta Hawks",
        "Heat": "Miami Heat",            "Hornets": "Charlotte Hornets",
        "Jazz": "Utah Jazz",             "Kings": "Sacramento Kings",
        "Knicks": "New York Knicks",     "Lakers": "Los Angeles Lakers",
        "Magic": "Orlando Magic",        "Mavericks": "Dallas Mavericks",
        "Nets": "Brooklyn Nets",         "Nuggets": "Denver Nuggets",
        "Pacers": "Indiana Pacers",      "Pelicans": "New Orleans Pelicans",
        "Pistons": "Detroit Pistons",    "Raptors": "Toronto Raptors",
        "Rockets": "Houston Rockets",    "Spurs": "San Antonio Spurs",
        "Suns": "Phoenix Suns",          "Thunder": "Oklahoma City Thunder",
        "Timberwolves": "Minnesota Timberwolves",
        "Trail Blazers": "Portland Trail Blazers",
        "Warriors": "Golden State Warriors", "Wizards": "Washington Wizards",
    }
    box["teamNameFull"] = box["teamName"].map(team_name_mapping)
    box = box.dropna(subset=["teamNameFull"])

    # Minuten parsen
    def parse_minutes(m):
        if isinstance(m, str) and ":" in m:
            parts = m.split(":")
            return int(parts[0]) + int(parts[1]) / 60
        try:
            return float(m)
        except Exception:
            return 0.0

    box["minutes"] = box["minutes"].apply(parse_minutes)

    # ── Aggregierte Team-Stats pro Spiel ──────────────────────
    stat_cols = [
        "points", "reboundsDefensive", "reboundsOffensive",
        "assists", "steals", "blocks", "turnovers", "minutes",
    ]
    team_game_stats = (
        box.groupby(["GAME_ID", "teamNameFull"])[stat_cols].sum().reset_index()
    )
    team_game_stats["player_count"] = (
        box.groupby(["GAME_ID", "teamNameFull"]).size().values
    )

    df = df.merge(
        team_game_stats.add_prefix("home_"),
        left_on=["gameId", "hometeamName"],
        right_on=["home_GAME_ID", "home_teamNameFull"],
        how="left",
    )
    df = df.merge(
        team_game_stats.add_prefix("away_"),
        left_on=["gameId", "awayteamName"],
        right_on=["away_GAME_ID", "away_teamNameFull"],
        how="left",
    )
    df.drop(
        columns=["home_GAME_ID", "home_teamNameFull", "away_GAME_ID", "away_teamNameFull"],
        inplace=True, errors="ignore",
    )

    # ── Team-Stats unified für Rolling ───────────────────────
    def build_team_stats(side):
        prefix = f"{side}_"
        opp    = "away" if side == "home" else "home"
        name_col = f"{side}teamName"
        s = df[["gameDateTimeEst", name_col,
                 f"{prefix}points", f"{prefix}reboundsDefensive",
                 f"{prefix}reboundsOffensive", f"{prefix}assists",
                 f"{prefix}minutes", f"{prefix}player_count"]].copy()
        s["rebounds"] = s[f"{prefix}reboundsDefensive"] + s[f"{prefix}reboundsOffensive"]
        s.rename(columns={
            name_col:              "team",
            f"{prefix}points":     "PTS",
            f"{prefix}assists":    "AST",
            f"{prefix}minutes":    "MIN",
            f"{prefix}player_count": "player_count",
        }, inplace=True)
        return s[["gameDateTimeEst", "team", "PTS", "rebounds", "AST", "MIN", "player_count"]]

    team_stats_all = (
        pd.concat([build_team_stats("home"), build_team_stats("away")], ignore_index=True)
        .sort_values(["team", "gameDateTimeEst"])
    )

    metrics = ["PTS", "rebounds", "AST", "MIN", "player_count"]
    for m in metrics:
        team_stats_all[m] = pd.to_numeric(team_stats_all[m], errors="coerce").fillna(0)
        col = f"last5_{m.lower()}"
        team_stats_all[col] = (
            team_stats_all.groupby("team")[m]
            .transform(lambda x: shifted_rolling(x, 5, 3).mean())
        )

    last5_cols = [f"last5_{m.lower()}" for m in metrics]

    home_last5 = team_stats_all[["gameDateTimeEst", "team"] + last5_cols].rename(
        columns={"team": "hometeamName",
                 **{c: f"home_{c}" for c in last5_cols}}
    )
    away_last5 = team_stats_all[["gameDateTimeEst", "team"] + last5_cols].rename(
        columns={"team": "awayteamName",
                 **{c: f"away_{c}" for c in last5_cols}}
    )

    df = df.merge(home_last5, on=["gameDateTimeEst", "hometeamName"], how="left")
    df = df.merge(away_last5, on=["gameDateTimeEst", "awayteamName"], how="left")

    df["pts_diff_last5"]          = df["home_last5_pts"]          - df["away_last5_pts"]
    df["reb_diff_last5"]          = df["home_last5_rebounds"]      - df["away_last5_rebounds"]
    df["ast_diff_last5"]          = df["home_last5_ast"]           - df["away_last5_ast"]
    df["min_diff_last5"]          = df["home_last5_min"]           - df["away_last5_min"]
    df["player_count_diff_last5"] = df["home_last5_player_count"]  - df["away_last5_player_count"]

    # ── Playoff Experience pro Team ───────────────────────────
    # Anzahl Playoff-Spiele pro Spieler (letzten 3 Saisons) → Teamdurchschnitt
    if "is_playoff" in box.columns and "GAME_ID" in box.columns:
        box_playoff = box[box.get("is_playoff", 0) == 1] if "is_playoff" in box.columns else pd.DataFrame()
    else:
        # Fallback: alle Spiele mit gameId in Playoff-df
        playoff_ids = df[df["is_playoff"] == 1]["gameId"].astype(str).tolist()
        box_playoff = box[box["GAME_ID"].astype(str).isin(playoff_ids)]

    if not box_playoff.empty:
        player_playoff_exp = (
            box_playoff.groupby("PLAYER_ID")  # kein teamNameFull → kein Doppelzählen
            .size()
            .reset_index(name="playoff_games")
        )
        # Team-Zuordnung: letztes bekanntes Team pro Spieler
        last_team = (
            box_playoff.sort_values("GAME_ID")
            .groupby("PLAYER_ID")["teamNameFull"]
            .last()
            .reset_index()
        )
        player_playoff_exp = player_playoff_exp.merge(last_team, on="PLAYER_ID")
        
        team_playoff_exp = (
            player_playoff_exp.groupby("teamNameFull")["playoff_games"]
            .mean()
            .reset_index(name="avg_playoff_exp")
        )
        df = df.merge(
            team_playoff_exp.rename(columns={"teamNameFull": "hometeamName",
                                             "avg_playoff_exp": "home_playoff_exp"}),
            on="hometeamName", how="left",
        )
        df = df.merge(
            team_playoff_exp.rename(columns={"teamNameFull": "awayteamName",
                                             "avg_playoff_exp": "away_playoff_exp"}),
            on="awayteamName", how="left",
        )
        df[["home_playoff_exp", "away_playoff_exp"]] = (
            df[["home_playoff_exp", "away_playoff_exp"]].fillna(0)
        )
        df["playoff_exp_diff"] = df["home_playoff_exp"] - df["away_playoff_exp"]

    print("Spieler-Features erfolgreich hinzugefügt.")
else:
    print(f"Warnung: {PLAYER_BOX} nicht gefunden – überspringe Spieler-Features.")

# ─────────────────────────────────────────────────────────────
# FEATURE 10: Injury Impact (aus current_injuries + player_matches)
# ─────────────────────────────────────────────────────────────
INJURY_MATCHES = os.path.join(base_dir, "data", "injury_player_matches.csv")

if os.path.exists(INJURY_MATCHES):
    print("Lade Injury-Daten...")
    inj = pd.read_csv(INJURY_MATCHES)

    # Injury-Score pro Team: Summe der importance_score aller verletzten Spieler
    # status_weight ist bereits in importance_score eingerechnet? Nein → manuell gewichten
    inj["weighted_impact"] = inj["importance_score"] * inj["status_weight"]

    team_injury_score = (
        inj.groupby("teamNameFull")["weighted_impact"]
        .sum()
        .reset_index(name="injury_impact")
    )

    # Heimteam
    df = df.merge(
        team_injury_score.rename(columns={
            "teamNameFull": "hometeamName",
            "injury_impact": "home_injury_impact"
        }),
        on="hometeamName", how="left"
    )
    # Auswärtsteam
    df = df.merge(
        team_injury_score.rename(columns={
            "teamNameFull": "awayteamName",
            "injury_impact": "away_injury_impact"
        }),
        on="awayteamName", how="left"
    )

    df["home_injury_impact"] = df["home_injury_impact"].fillna(0)
    df["away_injury_impact"] = df["away_injury_impact"].fillna(0)

    # Differenz: positiv = Heimteam hat mehr verletzte Spieler (schlechter)
    df["injury_impact_diff"] = df["home_injury_impact"] - df["away_injury_impact"]

    print("Injury-Features hinzugefügt.")
else:
    print(f"Warnung: {INJURY_MATCHES} nicht gefunden – überspringe Injury-Features.")

# ─────────────────────────────────────────────────────────────
# ALLE TEAM-HISTORY FEATURES IN df MERGEN
# ─────────────────────────────────────────────────────────────
history_features = [
    "last5_winrate", "last3_winrate", "last10_winrate", "winrate_trend",
    "last5_avg_points", "last5_avg_points_allowed",
    "rest_days", "is_back_to_back",
    "opponent_strength", "h2h_winrate",
    "home_winrate", "away_winrate",
    "last_game_close",
    "games_last7", "consecutive_away",
]

def merge_history(df, side):
    name_col = f"{side}teamName"
    prefix   = f"{side}_"
    cols     = ["date", "team", "game_id"] + history_features  # game_id hinzufügen
    sub      = team_history[cols].rename(columns={
        "date":    "gameDateTimeEst",
        "team":    name_col,
        "game_id": "gameId",
        **{c: f"{prefix}{c}" for c in history_features},
    })
    return df.merge(sub, on=["gameDateTimeEst", name_col, "gameId"], how="left")  

df = merge_history(df, "home")
df = merge_history(df, "away")

# Differenz-Features
df["winrate_diff"]                  = df["home_last5_winrate"]           - df["away_last5_winrate"]
df["winrate_trend_diff"]            = df["home_winrate_trend"]           - df["away_winrate_trend"]
df["average_points_diff"]           = df["home_last5_avg_points"]        - df["away_last5_avg_points"]
df["average_points_allowed_diff"]   = df["home_last5_avg_points_allowed"]- df["away_last5_avg_points_allowed"]
df["rest_days_diff"]                = df["home_rest_days"]               - df["away_rest_days"]
df["h2h_winrate_diff"]              = df["home_h2h_winrate"]             - df["away_h2h_winrate"]

# ─────────────────────────────────────────────────────────────
# FEATURE: Offensive/Defensive Rating & Net Rating
# ─────────────────────────────────────────────────────────────
# Nur wenn Spieler-Features vorhanden (MIN > 0)
if "home_last5_min" in df.columns and "away_last5_min" in df.columns:
    # Minuten als Divisor absichern
    home_min = df["home_last5_min"].replace(0, np.nan)
    away_min = df["away_last5_min"].replace(0, np.nan)

    df["home_off_rating"] = df["home_last5_pts"]               / home_min * 100
    df["away_off_rating"] = df["away_last5_pts"]               / away_min * 100
    df["home_def_rating"] = df["home_last5_avg_points_allowed"] / home_min * 100
    df["away_def_rating"] = df["away_last5_avg_points_allowed"] / away_min * 100

    df["home_net_rating"] = df["home_off_rating"] - df["home_def_rating"]
    df["away_net_rating"] = df["away_off_rating"] - df["away_def_rating"]
    df["net_rating_diff"] = df["home_net_rating"] - df["away_net_rating"]
    df["off_rating_diff"] = df["home_off_rating"] - df["away_off_rating"]
    df["def_rating_diff"] = df["home_def_rating"] - df["away_def_rating"]

# ─────────────────────────────────────────────────────────────
# HOME WIN LABEL
# ─────────────────────────────────────────────────────────────
df["home_win"] = (df["homeScore"] > df["awayScore"]).astype(int)

# ─────────────────────────────────────────────────────────────
# ZEITBASIERTE CROSS-VALIDATION (Diagnose)
# ─────────────────────────────────────────────────────────────
feature_cols = [
    "home_last5_winrate", "away_last5_winrate", "winrate_diff",
    "home_last3_winrate", "away_last3_winrate",
    "home_winrate_trend", "away_winrate_trend", "winrate_trend_diff",
    "home_last5_avg_points", "away_last5_avg_points", "average_points_diff",
    "home_last5_avg_points_allowed", "away_last5_avg_points_allowed", "average_points_allowed_diff",
    "home_rest_days", "away_rest_days", "rest_days_diff",
    "home_is_back_to_back", "away_is_back_to_back",
    "home_opponent_strength", "away_opponent_strength",
    "home_h2h_winrate", "away_h2h_winrate", "h2h_winrate_diff",
    "home_home_winrate", "away_home_winrate",
    "home_away_winrate", "away_away_winrate",
    "home_last_game_close", "away_last_game_close",
    "same_division", "is_playoff",
    "home_elo", "away_elo", "elo_diff",
    "home_consecutive_away", "away_consecutive_away",
    "home_games_last7", "away_games_last7",
    "home_off_rating", "away_off_rating", "off_rating_diff",
    "home_def_rating", "away_def_rating", "def_rating_diff",
    "home_net_rating", "away_net_rating", "net_rating_diff",
]

# Player-Features dynamisch anhängen falls vorhanden
player_feature_cols = [
    "home_last5_pts", "away_last5_pts",
    "home_last5_rebounds", "away_last5_rebounds",
    "home_last5_ast", "away_last5_ast",
    "home_last5_min", "away_last5_min",
    "home_last5_player_count", "away_last5_player_count",
    "pts_diff_last5", "reb_diff_last5", "ast_diff_last5",
    "min_diff_last5", "player_count_diff_last5",
]
playoff_exp_cols = ["home_playoff_exp", "away_playoff_exp", "playoff_exp_diff"]

for col in player_feature_cols + playoff_exp_cols:
    if col in df.columns:
        feature_cols.append(col)

# NEU: Injury-Features
injury_cols = ["home_injury_impact", "away_injury_impact", "injury_impact_diff"]

# Deduplizieren
feature_cols = list(dict.fromkeys(feature_cols))

df_model = df.dropna(subset=feature_cols + ["home_win"]).copy()

print(f"\nModel-Datensatz: {len(df_model)} Spiele, {len(feature_cols)} Features")

# ─────────────────────────────────────────────────────────────
# SPEICHERN
# ─────────────────────────────────────────────────────────────
model_output_path = os.path.join(base_dir, "data", "model_data.csv")
df.to_csv(model_output_path, index=False)
print(f"\nGespeichert: {model_output_path}")

feat_path = os.path.join(base_dir, "models", "feature_cols.csv")
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
pd.Series(feature_cols).to_csv(feat_path, index=False)
print(f"Feature-Liste gespeichert: {feat_path} ({len(feature_cols)} Features)")
