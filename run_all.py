import subprocess
import sys
import os
from datetime import datetime

import pandas as pd


def run_script(script_name):
    print(f"[{datetime.now()}] Starte {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FEHLER] {script_name}:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"[OK] {script_name} erfolgreich beendet.")
        if result.stdout:
            print(result.stdout)
    print("-" * 50)


def is_playoff_season():
    """Return True if today's schedule contains playoff games."""
    schedule_path = os.path.join("data", "schedule_round_1.csv")
    if not os.path.exists(schedule_path):
        return False
    try:
        df = pd.read_csv(schedule_path)
        return (
            "seriesGameNumber" in df.columns and
            df["seriesGameNumber"].notna().any()
        )
    except Exception:
        return False


if __name__ == "__main__":
    # Odds API Key setzen falls noch nicht als Umgebungsvariable vorhanden
    if not os.environ.get("ODDS_API_KEY"):
        os.environ["ODDS_API_KEY"] = "373d3a5bd6b6dad8581de9490f258177"

    # Reihenfolge der Skripte (angepasst an deine tatsächlichen Dateinamen)
    scripts = [
        "src/nba_api_test.py",                      # Ergebnisse von der API holen
        "src/scrape_upcoming_games.py",             # Kommende Spiele scrapen
        "src/injury_reports.py",                    # Verletzungsbericht aktualisieren
        "src/fetch_player_stats.py",                # Spielerstatistiken holen
        "src/tabelle.py",                           # Tabelle bereinigen
        "src/feature_engineering.py",               # Features hinzufügen
        "src/agents/data_quality_checker.py",       # Datenqualität prüfen
        "src/agents/feature_drift_detector.py",     # Feature-Drift erkennen
    ]

    # Steps 6b + 6c: playoff-only, run after feature_engineering and before predict.py
    if is_playoff_season():
        print(f"[{datetime.now()}] Playoff-Saison erkannt — füge Playoff-Schritte hinzu.")
        scripts += [
            "src/fetch_playoff_stats.py",           # [6b] Playoff-Statistiken holen
            "src/playoff_feature_engineering.py",   # [6c] Playoff-Features berechnen
        ]

    scripts += [
        "src/predict.py",                           # Vorhersage für heute
        "src/clean_excel.py",                       # Excel-Datei bereinigen
        "src/create_excel.py",                      # Excel-Datei aktualisieren
        "src/create_all_predictions.py",            # Heutige Vorhersagen sofort in all_predictions.xlsx
        "src/fetch_odds.py",                        # Wettquoten holen (braucht ODDS_API_KEY)
        "src/agents/odds_feature_injector.py",      # Wettquoten in Vorhersagen einblenden
        "src/agents/prediction_auditor.py",         # Predictions auf Plausibilitaet pruefen
        "src/export_json.py",                       # JSON für Webseite exportieren
        "src/fetch_bracket.py",                     # Bracket predictions für Webseite
        "src/agents/model_evaluator.py",            # Modell-Performance auswerten
        "src/agents/auto_retrainer.py",             # Neu trainieren falls Accuracy sinkt
        "src/agents/dashboard_exporter.py",         # Dashboard-JSON fuer Webapp exportieren
    ]

    # Prüfen, ob alle Skripte existieren
    for script in scripts:
        if not os.path.isfile(script):
            print(f"[FEHLER] Skript {script} nicht gefunden!")
            sys.exit(1)

    # Skripte nacheinander ausführen
    for script in scripts:
        run_script(script)

    print(f"[{datetime.now()}] Alle Skripte erfolgreich ausgeführt.")