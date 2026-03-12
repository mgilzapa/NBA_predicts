import pandas as pd

df = pd.read_csv("data/base_games.csv")
df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])


# Heimteam-Spiele
home_games = pd.DataFrame({
    "date": df["gameDateTimeEst"],
    "team": df["hometeamName"],
    "opponent": df["awayteamName"],
    "win": (df["homeScore"] > df["awayScore"]).astype(int)
})

# Auswärtsteam-Spiele
away_games = pd.DataFrame({
    "date": df["gameDateTimeEst"],
    "team": df["awayteamName"],
    "opponent": df["hometeamName"],
    "win": (df["awayScore"] > df["homeScore"]).astype(int)
})

# zusammenführen
team_history = pd.concat([home_games, away_games], ignore_index=True)

# sortieren
team_history = team_history.sort_values(["team", "date"]).reset_index(drop=True)
# letzte 5 Spiele Winrate
team_history["last5_winrate"] = (
    team_history.groupby("team")["win"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

home_features = team_history[["date", "team", "last5_winrate"]].rename(columns={
    "date": "gameDateTimeEst",
    "team": "hometeamName",
    "last5_winrate": "home_last5_winrate"
})

away_features = team_history[["date", "team", "last5_winrate"]].rename(columns={
    "date": "gameDateTimeEst",
    "team": "awayteamName",
    "last5_winrate": "away_last5_winrate"
})

# punkte statt Winrate
home_games = pd.DataFrame({
    "date": df["gameDateTimeEst"],
    "team": df["hometeamName"],
    "opponent": df["awayteamName"],
    "win": (df["homeScore"] > df["awayScore"]).astype(int),
    "points": df["homeScore"],
    "points_allowed": df["awayScore"]
})

away_games = pd.DataFrame({
    "date": df["gameDateTimeEst"],
    "team": df["awayteamName"],
    "opponent": df["hometeamName"],
    "win": (df["awayScore"] > df["homeScore"]).astype(int),
    "points": df["awayScore"],
    "points_allowed": df["homeScore"]
}).sort_values(["team", "date"])

team_history = pd.concat([home_games, away_games], ignore_index=True).sort_values(["team", "date"])

team_history["last5_avg_points"] = (
    team_history.groupby("team")["points"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

home_pts = team_history[["date", "team", "last5_avg_points"]].rename(columns={
    "date": "gameDateTimeEst",
    "team": "hometeamName",
    "last5_avg_points": "home_last5_avg_points"
})

away_pts = team_history[["date", "team", "last5_avg_points"]].rename(columns={
    "date": "gameDateTimeEst",
    "team": "awayteamName",
    "last5_avg_points": "away_last5_avg_points"
})

df = df.merge(home_features, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_features, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_pts, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_pts, on=["gameDateTimeEst", "awayteamName"], how="left")


team_history["rest_days"] = (
    team_history.groupby("team")["date"]
    .diff()
    .dt.days
)

home_rest = team_history[["date", "team", "rest_days"]].rename(columns={
    "date": "gameDateTimeEst",
    "team": "hometeamName",
    "rest_days": "home_rest_days"
})

away_rest = team_history[["date", "team", "rest_days"]].rename(columns={
    "date": "gameDateTimeEst",
    "team": "awayteamName",
    "rest_days": "away_rest_days"
})

df = df.merge(home_rest, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_rest, on=["gameDateTimeEst", "awayteamName"], how="left")

team_history["last5_avg_points_allowed"] = (
    team_history.groupby("team")["points_allowed"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)
home_points_allowed = team_history[["date", "team", "last5_avg_points_allowed"]].rename(columns={
    "date": "gameDateTimeEst",
    "team": "hometeamName",
    "last5_avg_points_allowed": "home_last5_avg_points_allowed"
})

away_points_allowed = team_history[["date", "team", "last5_avg_points_allowed"]].rename(columns={
    "date": "gameDateTimeEst",
    "team": "awayteamName",
    "last5_avg_points_allowed": "away_last5_avg_points_allowed"
})

df = df.merge(home_points_allowed, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_points_allowed, on=["gameDateTimeEst", "awayteamName"], how="left")


df.to_csv("data/model_data.csv", index=False)
print("Gespeichert: data/model_data.csv")