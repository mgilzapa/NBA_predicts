import os
import sys
import time
import pandas as pd
from nba_api.stats.endpoints import BoxScoreTraditionalV3
from datetime import datetime

# ------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------
base_dir = os.path.dirname(os.path.dirname(__file__))
GAME_LIST_CSV = os.path.join(base_dir, "data", "nba_api_games.csv")
PLAYER_BOX_CSV = os.path.join(base_dir, "data", "player_boxscores.csv")
SLEEP_SECONDS = 1                                # Pause zwischen API-Aufrufen

os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)

# ------------------------------------------------------------
# Batch-Zeitraum aus Kommandozeile lesen
# ------------------------------------------------------------
if len(sys.argv) >= 3:
    batch_start = pd.to_datetime(sys.argv[1])
    batch_end = pd.to_datetime(sys.argv[2])
    print(f"Lade Batch von {batch_start.date()} bis {batch_end.date()}")
else:
    print("Fehler: Bitte Start- und Enddatum angeben, z.B. python fetch_player_stats.py 2025-10-01 2025-11-01")
    sys.exit(1)

# Nur Spiele bis gestern (da zukünftige keine Boxscores haben)
yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)

# ------------------------------------------------------------
# 1. Bereits verarbeitete gameIds ermitteln
# ------------------------------------------------------------
if os.path.exists(PLAYER_BOX_CSV):
    existing = pd.read_csv(PLAYER_BOX_CSV)
    processed_ids = set(existing['GAME_ID'].unique())
    print(f"Bereits verarbeitete gameIds: {len(processed_ids)}")
else:
    processed_ids = set()
    print("Keine vorhandene Spielerdatei – starte neu.")

# ------------------------------------------------------------
# 2. Alle Spiele laden und auf Batch-Zeitraum filtern
# ------------------------------------------------------------
if not os.path.exists(GAME_LIST_CSV):
    print(f"Fehler -> Datei nicht gefunden: {GAME_LIST_CSV}")
    exit(1)

games = pd.read_csv(GAME_LIST_CSV)
games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
games['GAME_ID'] = games['GAME_ID'].astype(str).str.replace(r'\.0$', '', regex=True)
games['GAME_ID'] = games['GAME_ID'].str.zfill(10)

# Filter: Nur Spiele im Batch-Zeitraum und nicht in der Zukunft
mask = (games['GAME_DATE'] >= batch_start) & (games['GAME_DATE'] <= batch_end) & (games['GAME_DATE'] <= yesterday)
games_batch = games.loc[mask].copy()

print(f"Spiele im Batch-Zeitraum: {len(games_batch)}")

to_fetch = games_batch[~games_batch['GAME_ID'].isin(processed_ids)].copy()
print(f"Zu verarbeitende neue Spiele im Batch: {len(to_fetch)}")

if to_fetch.empty:
    print("Alles aktuell – nichts zu tun.")
    exit(0)

# ------------------------------------------------------------
# 3. Für jedes Spiel den Boxscore holen
# ------------------------------------------------------------
all_player_rows = []
failed_games = []

for idx, row in to_fetch.iterrows():
    game_id = row['GAME_ID']
    print(f"Verarbeite Spiel {game_id} ({idx+1}/{len(to_fetch)})...")

    max_retries = 3
    success = False

    for attempt in range(max_retries):
        try:
            boxscore = BoxScoreTraditionalV3(game_id=game_id, timeout=90)
            player_df = boxscore.get_data_frames()[0]  # Spielerstatistiken
            if not player_df.empty:
                player_df.rename(columns={'personId': 'PLAYER_ID'}, inplace=True)
                player_df['GAME_ID'] = game_id
                all_player_rows.append(player_df)
                success = True
                break
        except Exception as e:
            print(f"  Versuch {attempt+1}/{max_retries} fehlgeschlagen: {e}")
            if attempt < max_retries - 1:
                wait_time = SLEEP_SECONDS * (2 ** attempt)
                print(f"  Warte {wait_time} Sekunden vor nächstem Versuch...")
                time.sleep(wait_time)
            else:
                print(f"Fehler: Spiel {game_id} endgültig fehlgeschlagen – überspringe.")
                failed_games.append(game_id)

    time.sleep(SLEEP_SECONDS)

# Fehlgeschlagene Spiele speichern
if failed_games:
    failed_file = os.path.join(base_dir, "data", "failed_boxscores.csv")
    pd.Series(failed_games, name="GAME_ID").to_csv(failed_file, index=False)
    print(f"⚠️ {len(failed_games)} Spiele fehlgeschlagen – IDs gespeichert in {failed_file}")

# ------------------------------------------------------------
# 4. Alle gesammelten Daten kombinieren und speichern
# ------------------------------------------------------------
if all_player_rows:
    new_data = pd.concat(all_player_rows, ignore_index=True)

    if os.path.exists(PLAYER_BOX_CSV):
        old_data = pd.read_csv(PLAYER_BOX_CSV)
        combined = pd.concat([old_data, new_data], ignore_index=True)
    else:
        combined = new_data

    combined.drop_duplicates(subset=['GAME_ID', 'PLAYER_ID'], keep='last', inplace=True)
    combined.to_csv(PLAYER_BOX_CSV, index=False)
    print(f"Gespeichert: {PLAYER_BOX_CSV} mit {len(combined)} Einträgen.")
else:
    print("Keine neuen Daten hinzugefügt.")