import pandas as pd
import numpy as np

df = pd.read_csv("data/base_games.csv")
df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])

# ------------------------------------------------------------
# BASIS: Heim- und Auswärtsspiele für Team-History
# ------------------------------------------------------------
home_games = pd.DataFrame({
    "date": df["gameDateTimeEst"],
    "team": df["hometeamName"],
    "opponent": df["awayteamName"],
    "win": (df["homeScore"] > df["awayScore"]).astype(int),
    "points": df["homeScore"],
    "points_allowed": df["awayScore"],
    "is_home": 1
})

away_games = pd.DataFrame({
    "date": df["gameDateTimeEst"],
    "team": df["awayteamName"],
    "opponent": df["hometeamName"],
    "win": (df["awayScore"] > df["homeScore"]).astype(int),
    "points": df["awayScore"],
    "points_allowed": df["homeScore"],
    "is_home": 0
})

team_history = pd.concat([home_games, away_games], ignore_index=True)
team_history = team_history.sort_values(["team", "date"]).reset_index(drop=True)

# ------------------------------------------------------------
# FEATURE 1: Letzte 5 Spiele Winrate (vorhanden)
# ------------------------------------------------------------
team_history["last5_winrate"] = (
    team_history.groupby("team")["win"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# ------------------------------------------------------------
# FEATURE 2: Letzte 5 Spiele Punkte (vorhanden)
# ------------------------------------------------------------
team_history["last5_avg_points"] = (
    team_history.groupby("team")["points"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

team_history["last5_avg_points_allowed"] = (
    team_history.groupby("team")["points_allowed"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# ------------------------------------------------------------
# FEATURE 3: Ruhetage (vorhanden)
# ------------------------------------------------------------
team_history["rest_days"] = (
    team_history.groupby("team")["date"]
    .diff()
    .dt.days
)
team_history["is_back_to_back"] = (team_history["rest_days"] == 1).astype(int)

# ------------------------------------------------------------
# FEATURE 4: Gegner-Stärke (Strength of Schedule) - NEU
# ------------------------------------------------------------
# Winrate der letzten 5 Gegner
team_history["opponent_last5_winrate"] = (
    team_history.groupby("opponent")["win"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# ------------------------------------------------------------
# FEATURE 5: Heim/Auswärts getrennte Winrate - NEU
# ------------------------------------------------------------
# Nur Heimspiele
team_history["home_winrate"] = (
    team_history[team_history["is_home"]==1]
    .groupby("team")["win"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

# Nur Auswärtsspiele
team_history["away_winrate"] = (
    team_history[team_history["is_home"]==0]
    .groupby("team")["win"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

# Für Zeilen, wo keine Heim-/Auswärts-Historie existiert, mit Gesamt-Winrate füllen
team_history["home_winrate"] = team_history.groupby("team")["home_winrate"].ffill()
team_history["away_winrate"] = team_history.groupby("team")["away_winrate"].ffill()

# ------------------------------------------------------------
# FEATURE 6: Saison-Phase (Spielnummer) - NEU
# ------------------------------------------------------------
# Spielnummer pro Team und Saison berechnen
team_history["season"] = team_history["date"].dt.year
team_history["game_number"] = team_history.groupby(["team", "season"]).cumcount() + 1

# Saison-Phase als Kategorie (0-20, 21-60, 61-82)
team_history["season_phase"] = pd.cut(
    team_history["game_number"], 
    bins=[0, 20, 60, 100], 
    labels=["Anfang", "Mitte", "Ende"]
)
# One-Hot-Encoding für die Phasen
season_dummies = pd.get_dummies(team_history["season_phase"], prefix="phase")
team_history = pd.concat([team_history, season_dummies], axis=1)

# ------------------------------------------------------------
# FEATURE 7: Overtime-Indikator - NEU
# ------------------------------------------------------------
# Grober Overtime-Indikator: Summe > 220 Punkte (ca. 110 pro Team)
df["overtime"] = (df["homeScore"] + df["awayScore"] > 220).astype(int)

# Overtime-Info in team_history mergen
overtime_info = df[["gameDateTimeEst", "hometeamName", "awayteamName", "overtime"]].copy()
overtime_info["date"] = overtime_info["gameDateTimeEst"]
overtime_info_home = overtime_info[["date", "hometeamName", "overtime"]].rename(columns={"hometeamName": "team"})
overtime_info_away = overtime_info[["date", "awayteamName", "overtime"]].rename(columns={"awayteamName": "team"})
overtime_all = pd.concat([overtime_info_home, overtime_info_away], ignore_index=True)

team_history = team_history.merge(
    overtime_all[["date", "team", "overtime"]], 
    on=["date", "team"], 
    how="left"
)
team_history["last_game_overtime"] = team_history.groupby("team")["overtime"].shift(1)

# ------------------------------------------------------------
# FEATURE 8: Division/Rivalität - NEU
# ------------------------------------------------------------
# Team zu Division Mapping (Quelle: NBA)
divisions = {
    # Atlantic
    "Boston Celtics": "Atlantic", "Brooklyn Nets": "Atlantic", "New York Knicks": "Atlantic", 
    "Philadelphia 76ers": "Atlantic", "Toronto Raptors": "Atlantic",
    # Central
    "Chicago Bulls": "Central", "Cleveland Cavaliers": "Central", "Detroit Pistons": "Central",
    "Indiana Pacers": "Central", "Milwaukee Bucks": "Central",
    # Southeast
    "Atlanta Hawks": "Southeast", "Charlotte Hornets": "Southeast", "Miami Heat": "Southeast",
    "Orlando Magic": "Southeast", "Washington Wizards": "Southeast",
    # Northwest
    "Denver Nuggets": "Northwest", "Minnesota Timberwolves": "Northwest", 
    "Oklahoma City Thunder": "Northwest", "Portland Trail Blazers": "Northwest", 
    "Utah Jazz": "Northwest",
    # Pacific
    "Golden State Warriors": "Pacific", "LA Clippers": "Pacific", "Los Angeles Lakers": "Pacific",
    "Phoenix Suns": "Pacific", "Sacramento Kings": "Pacific",
    # Southwest
    "Dallas Mavericks": "Southwest", "Houston Rockets": "Southwest", "Memphis Grizzlies": "Southwest",
    "New Orleans Pelicans": "Southwest", "San Antonio Spurs": "Southwest"
}

df["home_division"] = df["hometeamName"].map(divisions)
df["away_division"] = df["awayteamName"].map(divisions)
df["same_division"] = (df["home_division"] == df["away_division"]).astype(int)

# ------------------------------------------------------------
# ALLE FEATURES FÜR HEIM UND AUSWÄRTS VORBEREITEN
# ------------------------------------------------------------
# Winrate
home_features = team_history[["date", "team", "last5_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "last5_winrate": "home_last5_winrate"
})
away_features = team_history[["date", "team", "last5_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "last5_winrate": "away_last5_winrate"
})

# Punkte
home_pts = team_history[["date", "team", "last5_avg_points"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "last5_avg_points": "home_last5_avg_points"
})
away_pts = team_history[["date", "team", "last5_avg_points"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "last5_avg_points": "away_last5_avg_points"
})

# Gegnerpunkte
home_pts_allowed = team_history[["date", "team", "last5_avg_points_allowed"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "last5_avg_points_allowed": "home_last5_avg_points_allowed"
})
away_pts_allowed = team_history[["date", "team", "last5_avg_points_allowed"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "last5_avg_points_allowed": "away_last5_avg_points_allowed"
})

# Ruhetage
home_rest = team_history[["date", "team", "rest_days", "is_back_to_back"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "rest_days": "home_rest_days", 
    "is_back_to_back": "home_is_back_to_back"
})
away_rest = team_history[["date", "team", "rest_days", "is_back_to_back"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "rest_days": "away_rest_days",
    "is_back_to_back": "away_is_back_to_back"
})

# Gegner-Stärke
home_sos = team_history[["date", "team", "opponent_last5_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "opponent_last5_winrate": "home_opponent_strength"
})
away_sos = team_history[["date", "team", "opponent_last5_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "opponent_last5_winrate": "away_opponent_strength"
})

# Heim/Auswärts Winrate
home_home_winrate = team_history[["date", "team", "home_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "home_winrate": "home_home_winrate"
})
away_home_winrate = team_history[["date", "team", "home_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "home_winrate": "away_home_winrate"
})
home_away_winrate = team_history[["date", "team", "away_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "away_winrate": "home_away_winrate"
})
away_away_winrate = team_history[["date", "team", "away_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "away_winrate": "away_away_winrate"
})

# Saison-Phase
phase_cols = [col for col in team_history.columns if col.startswith("phase_")]
home_phase = team_history[["date", "team"] + phase_cols].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName"
})
away_phase = team_history[["date", "team"] + phase_cols].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName"
})

# Overtime
home_ot = team_history[["date", "team", "last_game_overtime"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "last_game_overtime": "home_last_game_overtime"
})
away_ot = team_history[["date", "team", "last_game_overtime"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "last_game_overtime": "away_last_game_overtime"
})

# ------------------------------------------------------------
# ALLE FEATURES IN DAS HAUPTDATAFRAME MERGEN
# ------------------------------------------------------------
df = df.merge(home_features, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_features, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_pts, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_pts, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_pts_allowed, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_pts_allowed, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_rest, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_rest, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_sos, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_sos, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_home_winrate, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_home_winrate, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_away_winrate, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_away_winrate, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_phase, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_phase, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_ot, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_ot, on=["gameDateTimeEst", "awayteamName"], how="left")

# Division-Feature direkt aus df
df["same_division"] = (df["home_division"] == df["away_division"]).astype(int)


df.to_csv("data/model_data.csv", index=False)
print("Gespeichert: data/model_data.csv")
