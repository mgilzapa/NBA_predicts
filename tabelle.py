import pandas as pd

df = pd.read_csv("data/nba_api_games.csv")

df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

df = df.rename(columns={
    "GAME_ID": "gameId",
    "GAME_DATE": "gameDateTimeEst"
})

df = df[[
    "gameId",
    "gameDateTimeEst",
    "hometeamName",
    "awayteamName",
    "homeScore",
    "awayScore",
    "home_win"
]].copy()

df = df.sort_values("gameDateTimeEst").reset_index(drop=True)

df.to_csv("data/base_games.csv", index=False)
print(len(df))
print("Gespeichert: data/base_games.csv")