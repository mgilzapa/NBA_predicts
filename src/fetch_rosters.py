"""Holt die aktuellen Kader aller 30 NBA-Teams (CommonTeamRoster).

Ausgabe: data/current_rosters.csv mit PLAYER_ID -> aktuelles Team.
Wird genutzt fuer
  - Spieler-Importance / Injury-Features (Spieler zaehlen fuer ihr NEUES Team)
  - Roster-Continuity, die steuert wie stark das ELO zum Saisonstart regressiert
"""
import os
import time

import pandas as pd
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams as static_teams

from scrape_upcoming_games import current_nba_season

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ROSTER_OUTPUT = os.path.join(BASE_DIR, "data", "current_rosters.csv")
SLEEP_SECONDS = 0.7


def fetch_team_roster(team_id, season, retries=3):
    for attempt in range(1, retries + 1):
        try:
            frame = commonteamroster.CommonTeamRoster(
                team_id=team_id, season=season, timeout=30
            ).get_data_frames()[0]
            return frame
        except Exception as exc:
            print(f"  Versuch {attempt}/{retries} fuer Team {team_id} fehlgeschlagen: {exc}")
            time.sleep(SLEEP_SECONDS * (2 ** attempt))
    return pd.DataFrame()


def fetch_current_rosters(output_path=ROSTER_OUTPUT, season=None):
    season = season or current_nba_season()
    rows = []
    failed = []

    for team in sorted(static_teams.get_teams(), key=lambda t: t["full_name"]):
        roster = fetch_team_roster(team["id"], season)
        if roster.empty:
            failed.append(team["full_name"])
            continue
        roster = roster[["PLAYER_ID", "PLAYER", "POSITION", "EXP"]].copy()
        roster["team_id"] = team["id"]
        # gleiche Schreibweise wie nba_api_games.csv (z.B. "LA Clippers")
        roster["teamNameFull"] = team["full_name"].replace("Los Angeles Clippers", "LA Clippers")
        roster["season"] = season
        rows.append(roster)
        time.sleep(SLEEP_SECONDS)

    if not rows:
        print("WARNUNG: Keine Kader geladen – vorhandene Datei bleibt unveraendert.")
        return pd.DataFrame()

    if failed:
        # Lieber alte Daten behalten als halbe Kader schreiben
        if os.path.exists(output_path):
            old = pd.read_csv(output_path)
            old = old[old["teamNameFull"].isin(failed)]
            rows.append(old)
        print(f"WARNUNG: Kader fehlgeschlagen fuer: {failed}")

    rosters = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rosters.to_csv(output_path, index=False)
    print(f"OK: {len(rosters)} Spieler in {rosters['teamNameFull'].nunique()} Kadern ({season}) gespeichert: {output_path}")
    return rosters


def load_current_rosters(path=ROSTER_OUTPUT):
    if not os.path.exists(path):
        return pd.DataFrame()
    rosters = pd.read_csv(path)
    rosters["PLAYER_ID"] = rosters["PLAYER_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    return rosters


if __name__ == "__main__":
    fetch_current_rosters()
