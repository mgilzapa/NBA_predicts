import os
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

def update_all_predictions():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    source_file = os.path.join(base_dir, "output", "predictions.xlsx")
    target_file = os.path.join(base_dir, "output", "all_predictions.xlsx")

    if not os.path.exists(source_file):
        print(f"Quelldatei {source_file} nicht gefunden. Überspringe.")
        return

    
    try:
        df_source = pd.read_excel(source_file, sheet_name="predictions_today")
    except ValueError:
        print("Blatt 'predictions_today' nicht in der Quelldatei vorhanden.")
        return

    if df_source.empty:
        print("Keine Einträge im Blatt 'predictions_today' gefunden.")
        return

    # Prüfe ob gameID vorhanden ist
    if "gameID" not in df_source.columns:
        print("Warnung: Keine gameID-Spalte gefunden – verwende Datum+Teams als Fallback.")
        # Fallback auf Datum+Teams
        df_source["_temp_key"] = df_source["Date"].astype(str) + "|" + df_source["Home Team"] + "|" + df_source["Away Team"]
        key_column = "_temp_key"
    else:
        # Konvertiere gameID zu String (sicherheitshalber)
        df_source["gameId"] = df_source["gameId"].astype(str)
        key_column = "gameId"

    # Lade vorhandene Zieldatei, falls existent
    if os.path.exists(target_file):
        df_target = pd.read_excel(target_file)
        
        # Gleiche Vorbereitung für Ziel-Daten
        if "gameId" in df_target.columns:
            df_target["gameId"] = df_target["gameId"].astype(str)
            target_key_column = "gameId"
        else:
            df_target["_temp_key"] = df_target["Date"].astype(str) + "|" + df_target["Home Team"] + "|" + df_target["Away Team"]
            target_key_column = "_temp_key"
        
        # Kombinieren
        combined = pd.concat([df_target, df_source], ignore_index=True)
    else:
        combined = df_source
        target_key_column = key_column

    # Duplikate basierend auf gameID (oder Fallback) entfernen
    before = len(combined)
    combined.drop_duplicates(subset=[target_key_column if target_key_column in combined.columns else key_column], 
                            keep="last", inplace=True)
    after = len(combined)
    
    if before > after:
        print(f" Info: {before - after} Duplikate entfernt.")

    # Temporäre Schlüsselspalte entfernen falls vorhanden
    if "_temp_key" in combined.columns:
        combined.drop(columns=["_temp_key"], inplace=True)

    # Nach Datum sortieren
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined.sort_values("Date", inplace=True)

    # Speichern
    combined.to_excel(target_file, index=False)
    print(f"OK: Alle Vorhersagen gespeichert in {target_file} mit {len(combined)} Einträgen.")

if __name__ == "__main__":
    update_all_predictions()