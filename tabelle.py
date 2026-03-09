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

df.to_csv("data/base_games.csv", index=False)