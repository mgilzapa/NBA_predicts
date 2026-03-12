# NBA_predicts

Dieses Projekt sammelt täglich NBA-Spielergebnisse, trainiert ein logistisches Regressionsmodell und sagt die Spiele des aktuellen Tages (US Eastern Time) vorher. Die Vorhersagen sowie die Auswertung der Vortagesspiele werden übersichtlich in einer Excel-Datei gespeichert.

📌 Funktionsweise
  Der gesamte Workflow läuft täglich automatisch ab und besteht aus mehreren aufeinander aufbauenden Skripten:
  
  Daten aktualisieren – nba_api_test.py
  -Ruft über die nba_api alle Spiele der aktuellen Saison ab und speichert sie in nba_api_games.csv. Neue Spiele werden angehängt, bereits vorhandene nicht      dupliziert.
  
  Daten bereinigen – tabelle.py
  -Bereinigt die Rohdaten (z.B. korrekte Spaltentypen, Entfernen von Duplikaten) und erzeugt eine konsistente Tabelle Games.csv.
  
  Features berechnen – features.py
  -Erstellt aus den bereinigten Spielen die Feature‑Datei model_data.csv mit den für das Modell notwendigen Merkmalen (z.B. last5_winrate, durchschnittliche Punkte etc.).
  
  Vorhersage & Export – predict.py 
    
  -Trainiert das Modell auf allen Daten bis heute (US/Eastern).
    
  -Sagt die für heute geplanten Spiele vorher (Quelle: LeagueSchedule25-26.csv).
    
  -Holt die gestrigen Spiele aus den historischen Daten und vergleicht Vorhersage mit tatsächlichem Ergebnis.
    
  -Schreibt beides in die Excel‑Datei predictions_today.xlsx –
  Blatt predictions_today (heutige Vorhersagen) und Blatt yesterday (gestrige Ergebnisse inkl. tatsächlichem Gewinner).

Das Master‑Skript run_all.py führt diese vier Schritte nacheinander aus. Über einen Cron‑Job (Linux/macOS) oder die Windows‑Aufgabenplanung wird es täglich um 9:00 Uhr MEZ gestartet.
