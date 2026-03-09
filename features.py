import pandas as pd

df = pd.read_csv("data/base_games.csv")



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
    "points": df["homeScore"]
})

away_games = pd.DataFrame({
    "date": df["gameDateTimeEst"],
    "team": df["awayteamName"],
    "opponent": df["hometeamName"],
    "win": (df["awayScore"] > df["homeScore"]).astype(int),
    "points": df["awayScore"]
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

print(df.columns)
print(df[[
    "gameDateTimeEst",
    "hometeamName",
    "awayteamName",
    "home_last5_avg_points",
    "away_last5_avg_points"
]].head(10))

df.to_csv("data/model_data.csv", index=False)