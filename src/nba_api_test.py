import os
import pandas as pd
import time
from nba_api.stats.endpoints import BoxScoreTraditionalV3, leaguegamelog

from create_all_predictions import update_all_predictions

def fetch_actual_winner(game_id):
    try:
        boxscore = BoxScoreTraditionalV3(game_id=game_id, timeout=30)
        frames = boxscore.get_data_frames()
        for df in frames:
            if len(df) == 2 and 'teamName' in df.columns and 'points' in df.columns:
                team0 = df.iloc[0]
                team1 = df.iloc[1]
                print(f"   Team 0: {team0['teamName']}, Points: {team0['points']} (Typ: {type(team0['points'])})")
                print(f"   Team 1: {team1['teamName']}, Points: {team1['points']} (Typ: {type(team1['points'])})")
                points0 = pd.to_numeric(team0['points'], errors='coerce')
                points1 = pd.to_numeric(team1['points'], errors='coerce')
                if points0 > points1:
                    return team0['teamName']
                else:
                    return team1['teamName']
        return None
    except Exception as e:
        print(f"Fehler: {e}")
        return None

def update_actual_winners_from_csv(target_file, nba_games_file):
    """
    Liest die Gewinner aus nba_api_games.csv und trägt sie in all_predictions.xlsx ein.
    """
    if not os.path.exists(nba_games_file):
        print(f"❌ {nba_games_file} nicht gefunden.")
        return

    # all_predictions laden
    df_pred = pd.read_excel(target_file)
    if 'gameId' not in df_pred.columns:
        print("❌ Keine Spalte 'gameId' in all_predictions.")
        return

    # nba_api_games laden
    df_res = pd.read_csv(nba_games_file)

    # gameId normalisieren (als String ohne .0)
    df_pred['gameId'] = df_pred['gameId'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_res['GAME_ID'] = df_res['GAME_ID'].astype(str).str.replace(r'\.0$', '', regex=True)

    # Gewinner berechnen (actual_winner)
    df_res['actual_winner'] = df_res.apply(
        lambda row: row['hometeamName'] if row['home_win'] == 1 else row['awayteamName'],
        axis=1
    )

    # Merge: für jede Zeile in df_pred den actual_winner aus df_res holen
    merged = df_pred.merge(
        df_res[['GAME_ID', 'actual_winner']],
        left_on='gameId',
        right_on='GAME_ID',
        how='left'
    )

    # Spalte 'Actual Winner' aktualisieren (falls vorhanden)
    if 'Actual Winner' in merged.columns:
        merged['Actual Winner'] = merged['actual_winner'].combine_first(merged['Actual Winner'])
    else:
        merged.rename(columns={'actual_winner': 'Actual Winner'}, inplace=True)

    # Hilfsspalten entfernen
    merged.drop(columns=['GAME_ID', 'actual_winner'], inplace=True, errors='ignore')

    # Speichern
    merged.to_excel(target_file, index=False)
    print(f" Actual Winners in {target_file} aktualisiert.")

def update_nba_games(csv_path="data/nba_api_games.csv"):
    """
    Lädt neue NBA-Spiele von der API und hängt sie an die bestehende CSV an.
    Falls die CSV nicht existiert, wird sie komplett ab 2016 neu erstellt.
    """
    # Seasons, die noch nicht abgeschlossen sind (aktuell 2025-26)
    current_season = "2025-26"
    # Falls du auch vorherige Seasons nachladen willst, falls Lücken bestehen, kannst du das erweitern.
    seasons_to_check = [current_season]

    # Prüfen, ob CSV existiert
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        existing_df["GAME_DATE"] = pd.to_datetime(existing_df["GAME_DATE"])
        last_date = existing_df["GAME_DATE"].max()
        print(f"Letztes Spiel in CSV: {last_date.date()}")
        # Nur Spiele ab dem Tag nach dem letzten Spiel laden (um Überschneidungen zu vermeiden)
        # Da die API nur saisonweise liefert, holen wir die ganze Saison und filtern später.
    else:
        existing_df = pd.DataFrame()
        last_date = None
        # Falls keine CSV existiert, lade alle Seasons ab 2016
        seasons_to_check = ["2016-17", "2017-18", "2018-19", "2019-20", 
                            "2020-21", "2021-22", "2022-23", "2023-24", 
                            "2024-25", "2025-26"]
        print("Keine vorhandene CSV gefunden – erstelle neue Datei mit allen Seasons.")

    all_new_games = []

    for season in seasons_to_check:
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

        home_df = home_df.rename(columns={"TEAM_NAME": "hometeamName", "PTS": "homeScore"})
        away_df = away_df.rename(columns={"TEAM_NAME": "awayteamName", "PTS": "awayScore"})

        home_df = home_df[["GAME_ID", "GAME_DATE", "hometeamName", "homeScore"]]
        away_df = away_df[["GAME_ID", "awayteamName", "awayScore"]]

        games = home_df.merge(away_df, on="GAME_ID", how="inner")
        games["home_win"] = (games["homeScore"] > games["awayScore"]).astype(int)
        games["season"] = season

        # Falls wir eine bestehende CSV haben, nur Spiele ab dem letzten Datum behalten
        if last_date is not None and season == current_season:
            games = games[games["GAME_DATE"] > last_date].copy()
            if len(games) == 0:
                print(f"Keine neuen Spiele in Saison {season} seit {last_date.date()}.")
            else:
                print(f"{len(games)} neue Spiele in Saison {season} gefunden.")
        all_new_games.append(games)

        time.sleep(1)  # Netzlast reduzieren

    if all_new_games:
        new_games_df = pd.concat(all_new_games, ignore_index=True)
        # Mit vorhandenen Daten kombinieren
        if not existing_df.empty:
            updated_df = pd.concat([existing_df, new_games_df], ignore_index=True)
        else:
            updated_df = new_games_df

        # Sortieren und Duplikate entfernen (sicherheitshalber)
        updated_df = updated_df.sort_values("GAME_DATE").drop_duplicates(subset=["GAME_ID"]).reset_index(drop=True)
        updated_df.to_csv(csv_path, index=False)
        print(f"CSV aktualisiert: {csv_path} | Gesamtanzahl Spiele: {len(updated_df)}")
    else:
        print("Keine neuen Spiele zum Hinzufügen.")

if __name__ == "__main__":
    update_nba_games("data/nba_api_games.csv")
    df = pd.read_csv("data/model_data.csv")
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    base_dir = os.path.dirname(os.path.dirname(__file__))
    target_file = os.path.join(base_dir, "output", "all_predictions.xlsx")
    nba_games_file = os.path.join(base_dir, "data", "nba_api_games.csv")

    # 1. Neue Vorhersagen aus predictions_today übernehmen
    update_all_predictions()

    # 2. Fehlende Actual Winners aus nba_api_games.csv nachtragen
    update_actual_winners_from_csv(target_file, nba_games_file)