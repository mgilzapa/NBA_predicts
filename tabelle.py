import pandas as pd

df = pd.read_csv("data/Games.csv", low_memory=False)

# nur wichtige Spalten
df = df[["gameDateTimeEst", "hometeamName", "awayteamName", "homeScore", "awayScore"]].copy()

# Datum
df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])

# nur letzte Jahre
df = df[df["gameDateTimeEst"] >= "2021-01-01"].copy()

# Zielvariable
df["home_win"] = (df["homeScore"] > df["awayScore"]).astype(int)

# sortieren
df = df.sort_values("gameDateTimeEst").reset_index(drop=True)

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

df = df.merge(home_features, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_features, on=["gameDateTimeEst", "awayteamName"], how="left")

print(df[["gameDateTimeEst", "hometeamName", "awayteamName", "home_last5_winrate", "away_last5_winrate", "home_win"]].head(10))

df.to_csv("data/model_data.csv", index=False)