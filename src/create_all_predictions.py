import os
import pandas as pd

def update_all_predictions():

    base_dir = os.path.dirname(os.path.dirname(__file__))  # geht von src/ ins Hauptverzeichnis
    source_file = os.path.join(base_dir, "output", "predictions.xlsx")
    target_file = os.path.join(base_dir, "output", "all_predictions.xlsx")
    """
    Liest das Blatt 'yesterday' aus source_file und fügt alle Einträge in target_file ein,
    wobei Duplikate (basierend auf Datum + Teams) entfernt werden.
    """
    if not os.path.exists(source_file):
        print(f"Quelldatei {source_file} nicht gefunden. Überspringe.")
        return

    # Lade yesterday-Blatt aus der Quelle
    try:
        df_source = pd.read_excel(source_file, sheet_name="yesterday")
    except ValueError:
        print("Blatt 'yesterday' nicht in der Quelldatei vorhanden.")
        return

    if df_source.empty:
        print("Keine Einträge im Blatt 'yesterday' gefunden.")
        return

    # Lade vorhandene Zieldatei, falls existent
    if os.path.exists(target_file):
        df_target = pd.read_excel(target_file)
        combined = pd.concat([df_target, df_source], ignore_index=True)
    else:
        combined = df_source

    # Duplikate entfernen (basierend auf Date, Home Team, Away Team)
    combined.drop_duplicates(subset=["Date", "Home Team", "Away Team"], keep="last", inplace=True)

    # Nach Datum sortieren
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined.sort_values("Date", inplace=True)

    # Speichern
    combined.to_excel(target_file, index=False)
    print(f"OK: Alle Vorhersagen gespeichert in {target_file} mit {len(combined)} Einträgen.")

if __name__ == "__main__":
    update_all_predictions()