import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name):
    print(f"[{datetime.now()}] Starte {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Fehler in {script_name}:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"✅ {script_name} erfolgreich beendet.")
        if result.stdout:
            print(result.stdout)
    print("-" * 50)

if __name__ == "__main__":
    # Reihenfolge der Skripte (angepasst an deine tatsächlichen Dateinamen)
    scripts = [
        "src/nba_api_test.py",   # Ergebnisse von der API holen
        "src/fetch_player_stats.py",  # Spielerstatistiken holen
        "src/tabelle.py",        # Tabelle bereinigen
        "src/feature_engineering.py",       # Features hinzufügen
        "src/predict.py",        # Vorhersage für heute
        "src/clean_excel.py",    # Excel-Datei bereinigen
        "src/create_excel.py",   # Excel-Datei aktualisieren
    ]

    # Prüfen, ob alle Skripte existieren
    for script in scripts:
        if not os.path.isfile(script):
            print(f"❌ Skript {script} nicht gefunden!")
            sys.exit(1)

    # Skripte nacheinander ausführen
    for script in scripts:
        run_script(script)

    print(f"[{datetime.now()}] Alle Skripte erfolgreich ausgeführt.")