import os
import pandas as pd
from predict import output_today, output_yesterday

excel_file = "output/predictions.xlsx"
sheet_today = "predictions_today"
sheet_yesterday = "yesterday"

# Bestehende Daten laden (falls vorhanden)
if os.path.exists(excel_file):
    with pd.ExcelFile(excel_file) as xls:
        sheet_names = xls.sheet_names
        existing_today = pd.read_excel(xls, sheet_today) if sheet_today in sheet_names else pd.DataFrame()
        existing_yesterday = pd.read_excel(xls, sheet_yesterday) if sheet_yesterday in sheet_names else pd.DataFrame()
else:
    existing_today = pd.DataFrame()
    existing_yesterday = pd.DataFrame()

# Heutige Spiele kombinieren und Duplikate entfernen
if not output_today.empty:
    combined_today = pd.concat([existing_today, output_today], ignore_index=True)
    combined_today.drop_duplicates(subset=["Date", "Home Team", "Away Team"], keep="last", inplace=True)
else:
    combined_today = existing_today

# Gestrige Spiele kombinieren und Duplikate entfernen
if not output_yesterday.empty:
    combined_yesterday = pd.concat([existing_yesterday, output_yesterday], ignore_index=True)
    combined_yesterday.drop_duplicates(subset=["Date", "Home Team", "Away Team"], keep="last", inplace=True)
else:
    combined_yesterday = existing_yesterday

# Beide Sheets in eine Excel-Datei schreiben
try:
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        combined_today.to_excel(writer, sheet_name=sheet_today, index=False)
        combined_yesterday.to_excel(writer, sheet_name=sheet_yesterday, index=False)
    print(f"OK: Excel-Datei erfolgreich geschrieben: {excel_file}")
    print("Info: Vorhandene Sheets:", pd.ExcelFile(excel_file).sheet_names)
except PermissionError:
    print("FEHLER: Die Datei ist möglicherweise geöffnet. Bitte schließen Sie sie und führen Sie das Skript erneut aus.")
except Exception as e:
    print(f"FEHLER beim Schreiben der Excel-Datei: {e}")