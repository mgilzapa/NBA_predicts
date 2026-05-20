import os
import sys
import time
import argparse
import pandas as pd
from nba_api.stats.endpoints import BoxScoreTraditionalV3
from datetime import datetime

# ------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------
base_dir = os.path.dirname(os.path.dirname(__file__))
GAME_LIST_CSV = os.path.join(base_dir, "data", "nba_api_games.csv")
PLAYER_BOX_CSV = os.path.join(base_dir, "data", "player_boxscores.csv")
SLEEP_SECONDS = 1

os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)

# ------------------------------------------------------------
# Argumente
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--backfill",
    action="store_true",
    help="Alle fehlenden Spiele der aktuellen Saison nachholen (nicht nur gestern)",
)
parser.add_argument(
    "--season-start",
    default="2025-10-01",
    help="Saisonbeginn für Backfill (Standard: 2025-10-01)",
)
args, _ = parser.parse_known_args()

# ------------------------------------------------------------
# 1. Bereits verarbeitete gameIds ermitteln
# ------------------------------------------------------------
if os.path.exists(PLAYER_BOX_CSV):
    existing = pd.read_csv(PLAYER_BOX_CSV)
    processed_ids = set(existing["GAME_ID"].unique())
    print(f"Bereits verarbeitete gameIds: {len(processed_ids)}")
else:
    processed_ids = set()
    print("Keine vorhandene Spielerdatei – starte neu.")

# ------------------------------------------------------------
# 2. Zielspiele bestimmen
# ------------------------------------------------------------
if not os.path.exists(GAME_LIST_CSV):
    print(f"Fehler -> Datei nicht gefunden: {GAME_LIST_CSV}")
    sys.exit(1)

games = pd.read_csv(GAME_LIST_CSV)
games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
games["GAME_ID"] = games["GAME_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)

if args.backfill:
    season_start = pd.Timestamp(args.season_start)
    target = games[games["GAME_DATE"] >= season_start].copy()
    print(f"Backfill-Modus: Spiele ab {season_start.date()} ({len(target)} gesamt)")
else:
    yesterday = pd.Timestamp(datetime.now().date() - pd.Timedelta(days=1))
    target = games[games["GAME_DATE"].dt.date == yesterday.date()].copy()
    print(f"Lade Boxscores für Spiele vom: {yesterday.date()} ({len(target)} Spiele)")

to_fetch = target[~target["GAME_ID"].isin(processed_ids)].copy()
print(f"Spiele zum Verarbeiten: {len(to_fetch)}")

if to_fetch.empty:
    print("Keine neuen Spiele zum Verarbeiten – Skript wird beendet.")
    sys.exit(0)

# ------------------------------------------------------------
# 3. Für jedes Spiel den Boxscore holen
# ------------------------------------------------------------
all_player_rows = []
failed_games = []

for i, (idx, row) in enumerate(to_fetch.iterrows(), 1):
    game_id = row["GAME_ID"]
    print(f"[{i}/{len(to_fetch)}] Spiel {game_id} ({row['GAME_DATE'].date()})...")

    for attempt in range(3):
        try:
            boxscore = BoxScoreTraditionalV3(game_id=game_id, timeout=90)
            player_df = boxscore.get_data_frames()[0]
            if not player_df.empty:
                player_df.rename(columns={"personId": "PLAYER_ID"}, inplace=True)
                player_df["GAME_ID"] = game_id
                all_player_rows.append(player_df)
            break
        except Exception as e:
            print(f"  Versuch {attempt+1}/3 fehlgeschlagen: {e}")
            if attempt < 2:
                time.sleep(SLEEP_SECONDS * (2 ** attempt))
            else:
                print(f"  Spiel {game_id} endgültig fehlgeschlagen.")
                failed_games.append(game_id)

    # Zwischenspeichern alle 50 Spiele
    if args.backfill and len(all_player_rows) > 0 and i % 50 == 0:
        new_data = pd.concat(all_player_rows, ignore_index=True)
        if os.path.exists(PLAYER_BOX_CSV):
            old_data = pd.read_csv(PLAYER_BOX_CSV)
            combined = pd.concat([old_data, new_data], ignore_index=True)
        else:
            combined = new_data
        combined.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"], keep="last", inplace=True)
        combined.to_csv(PLAYER_BOX_CSV, index=False)
        print(f"  >> Zwischenstand gespeichert: {len(combined)} Einträge")
        all_player_rows = []

    time.sleep(SLEEP_SECONDS)

# Fehlgeschlagene Spiele speichern
if failed_games:
    failed_file = os.path.join(base_dir, "data", "failed_boxscores.csv")
    pd.Series(failed_games, name="GAME_ID").to_csv(failed_file, index=False)
    print(f"{len(failed_games)} Spiele fehlgeschlagen – IDs gespeichert in {failed_file}")

# ------------------------------------------------------------
# 4. Restliche Daten speichern
# ------------------------------------------------------------
if all_player_rows:
    new_data = pd.concat(all_player_rows, ignore_index=True)
    if os.path.exists(PLAYER_BOX_CSV):
        old_data = pd.read_csv(PLAYER_BOX_CSV)
        combined = pd.concat([old_data, new_data], ignore_index=True)
    else:
        combined = new_data
    combined.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"], keep="last", inplace=True)
    combined.to_csv(PLAYER_BOX_CSV, index=False)
    print(f"Gespeichert: {PLAYER_BOX_CSV} mit {len(combined)} Einträgen.")
else:
    print("Keine neuen Daten hinzugefügt.")
