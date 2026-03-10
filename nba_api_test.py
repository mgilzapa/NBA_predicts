from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
import time

seasons = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

all_games = []

for season in seasons:
    print(f"Lade Saison {season}...")

    gamelog = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season"
    )

    df = gamelog.get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["is_home"] = df["MATCHUP"].str.contains("vs.").astype(int)

    home_df = df[df["is_home"] == 1].copy()
    away_df = df[df["is_home"] == 0].copy()

    home_df = home_df.rename(columns={
        "TEAM_NAME": "hometeamName",
        "PTS": "homeScore"
    })

    away_df = away_df.rename(columns={
        "TEAM_NAME": "awayteamName",
        "PTS": "awayScore"
    })

    home_df = home_df[["GAME_ID", "GAME_DATE", "hometeamName", "homeScore"]]
    away_df = away_df[["GAME_ID", "awayteamName", "awayScore"]]

    games = home_df.merge(away_df, on="GAME_ID", how="inner")
    games["home_win"] = (games["homeScore"] > games["awayScore"]).astype(int)
    games["season"] = season

    all_games.append(games)

    time.sleep(1)

all_games_df = pd.concat(all_games, ignore_index=True)
all_games_df = all_games_df.sort_values("GAME_DATE").reset_index(drop=True)


print(len(all_games_df))
all_games_df.to_csv("data/nba_api_games.csv", index=False)
print("Gespeichert: data/nba_api_games.csv")