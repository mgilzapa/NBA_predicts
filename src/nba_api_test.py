import os
import pandas as pd
import time
from nba_api.stats.endpoints import BoxScoreTraditionalV3, leaguegamelog

from create_all_predictions import update_all_predictions
from scrape_upcoming_games import NBA_TEAM_ID_PREFIX, current_nba_season

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
        print(f"FEHLER: {nba_games_file} nicht gefunden.")
        return

    if not os.path.exists(target_file):
        print(f"INFO: {target_file} nicht gefunden - ueberspringe Actual-Winner-Update.")
        return

    # all_predictions laden
    df_pred = pd.read_excel(target_file)
    if 'gameId' not in df_pred.columns:
        print("FEHLER: Keine Spalte 'gameId' in all_predictions.")
        return

    # nba_api_games laden
    df_res = pd.read_csv(nba_games_file)

    # datum normalisieren
    df_pred['_date'] = pd.to_datetime(df_pred['Date']).dt.normalize()
    df_res['_date'] = pd.to_datetime(df_res['GAME_DATE']).dt.normalize()

    # Gewinner berechnen (actual_winner)
    df_res['actual_winner'] = df_res.apply(
        lambda row: row['hometeamName'] if row['home_win'] == 1 else row['awayteamName'],
        axis=1
    )

    # Merge über Datum + Home Team (erst dedup damit kein 1:n-Merge entsteht)
    df_res_dedup = df_res[['_date', 'hometeamName', 'actual_winner']].drop_duplicates(subset=['_date', 'hometeamName'])
    merged = df_pred.merge(
        df_res_dedup,
        left_on=['_date', 'Home Team'],
        right_on=['_date', 'hometeamName'],
        how='left'
    )

    # Spalte 'Actual Winner' aktualisieren (falls vorhanden)
    if 'Actual Winner' in merged.columns:
        merged['Actual Winner'] = merged['actual_winner'].combine_first(merged['Actual Winner'])
    else:
        merged.rename(columns={'actual_winner': 'Actual Winner'}, inplace=True)

    # Hilfsspalten entfernen
    merged.drop(columns=['_date','hometeamName', 'actual_winner'], inplace=True, errors='ignore')

    # Speichern
    merged.to_excel(target_file, index=False)
    print(f" Actual Winners in {target_file} aktualisiert.")

SEASON_TYPES = ["Pre Season", "Regular Season", "Playoffs"]
ALL_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24",
    "2024-25", "2025-26",
]


def fetch_season_games(season, season_type):
    """Laedt alle abgeschlossenen Spiele einer Saison/eines Spieltyps als Heim/Auswaerts-Zeile."""
    print(f"  -> Lade {season} {season_type}")
    gamelog = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        timeout=60,
    )
    df = gamelog.get_data_frames()[0]
    time.sleep(1)
    if df.empty:
        return pd.DataFrame()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["is_home"] = df["MATCHUP"].str.contains("vs.").astype(int)

    # Preseason enthaelt Spiele gegen internationale Clubs -> nur Spiele NBA vs. NBA behalten
    df["is_nba"] = df["TEAM_ID"].astype(str).str.startswith(NBA_TEAM_ID_PREFIX)
    nba_only = df.groupby("GAME_ID")["is_nba"].transform("all") & (df.groupby("GAME_ID")["GAME_ID"].transform("size") == 2)
    df = df[nba_only].copy()

    home_df = df[df["is_home"] == 1].rename(columns={"TEAM_NAME": "hometeamName", "PTS": "homeScore"})
    away_df = df[df["is_home"] == 0].rename(columns={"TEAM_NAME": "awayteamName", "PTS": "awayScore"})
    home_df = home_df[["GAME_ID", "GAME_DATE", "hometeamName", "homeScore"]]
    away_df = away_df[["GAME_ID", "awayteamName", "awayScore"]]

    # inner merge: Preseason-Spiele gegen Nicht-NBA-Teams (nur eine Seite im Log) fallen raus
    games = home_df.merge(away_df, on="GAME_ID", how="inner")
    # API liefert "0012600009", die CSV speichert 12600009 -> einheitlich als int
    games["GAME_ID"] = games["GAME_ID"].astype(int)
    games["season_type"] = season_type
    games["home_win"] = (games["homeScore"] > games["awayScore"]).astype(int)
    games["season"] = season
    return games


def update_nba_games(csv_path="data/nba_api_games.csv"):
    """
    Laedt neue NBA-Spiele (Preseason, Regular Season, Playoffs) und haengt sie an die CSV an.
    Falls die CSV nicht existiert, wird sie komplett ab 2016 neu erstellt.
    Fehlende Preseason-Daten frueherer Saisons werden einmalig nachgeladen.
    """
    current_season = current_nba_season()
    seasons = list(dict.fromkeys(ALL_SEASONS + [current_season]))

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        existing_df["GAME_DATE"] = pd.to_datetime(existing_df["GAME_DATE"])
        existing_df["GAME_ID"] = existing_df["GAME_ID"].astype(int)
        print(f"Letztes Spiel in CSV: {existing_df['GAME_DATE'].max().date()}")
        have = set(zip(existing_df["season"], existing_df["season_type"]))
        # aktuelle Saison immer neu pruefen, aeltere nur wenn der Spieltyp komplett fehlt
        to_fetch = [
            (season, season_type)
            for season in seasons
            for season_type in SEASON_TYPES
            if season == current_season or (season, season_type) not in have
        ]
    else:
        existing_df = pd.DataFrame()
        to_fetch = [(season, season_type) for season in seasons for season_type in SEASON_TYPES]
        print("Keine vorhandene CSV gefunden – erstelle neue Datei mit allen Seasons.")

    known_ids = set(existing_df["GAME_ID"]) if not existing_df.empty else set()
    all_new_games = []
    for season, season_type in to_fetch:
        games = fetch_season_games(season, season_type)
        if games.empty:
            continue
        games = games[~games["GAME_ID"].isin(known_ids)]
        if len(games):
            print(f"     {len(games)} neue Spiele ({season} {season_type})")
            all_new_games.append(games)

    if all_new_games:
        new_games_df = pd.concat(all_new_games, ignore_index=True)

        if not existing_df.empty:
            updated_df = pd.concat([existing_df, new_games_df], ignore_index=True)
        else:
            updated_df = new_games_df

        updated_df = (
            updated_df
            .sort_values("GAME_DATE")
            .drop_duplicates(subset=["GAME_ID"])
            .reset_index(drop=True)
        )

        updated_df.to_csv(csv_path, index=False)
        print(f"CSV aktualisiert: {csv_path} | Gesamtanzahl Spiele: {len(updated_df)}")
    else:
        print("Keine neuen Spiele zum Hinzufügen.")

def update_playin_games(csv_path="data/nba_api_playin.csv"):
    """
    Laedt die Play-In-Spiele (GAME_ID-Praefix 005) der aktuellen Saison in eine eigene CSV.
    Bewusst NICHT in nba_api_games.csv, damit Feature Engineering und Training unveraendert
    bleiben. Wird von fetch_bracket.py fuer den Live-Modus des Brackets genutzt.
    """
    current_season = current_nba_season()
    games = fetch_season_games(current_season, "PlayIn")
    existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    if games.empty:
        print(f"Keine Play-In-Spiele fuer {current_season}.")
        return
    combined = pd.concat([existing, games], ignore_index=True) if not existing.empty else games
    combined["GAME_ID"] = combined["GAME_ID"].astype(int)
    combined = combined.drop_duplicates(subset=["GAME_ID"], keep="last").sort_values("GAME_DATE")
    combined.to_csv(csv_path, index=False)
    print(f"Play-In-Spiele gespeichert: {csv_path} ({len(combined)} Spiele)")


if __name__ == "__main__":
    update_nba_games("data/nba_api_games.csv")
    try:
        update_playin_games("data/nba_api_playin.csv")
    except Exception as e:  # Play-In ist optional, darf den Pipeline-Lauf nicht stoppen
        print(f"WARNUNG: Play-In-Spiele konnten nicht geladen werden: {e}")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    target_file = os.path.join(base_dir, "output", "all_predictions.xlsx")
    nba_games_file = os.path.join(base_dir, "data", "nba_api_games.csv")

    # 1. Neue Vorhersagen aus predictions_today übernehmen
    update_all_predictions()

    # 2. Fehlende Actual Winners aus nba_api_games.csv nachtragen
    update_actual_winners_from_csv(target_file, nba_games_file)