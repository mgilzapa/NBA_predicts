import argparse
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "data", "schedule_round_1.csv")
EASTERN_TZ = ZoneInfo("America/New_York")
NBA_TEAM_ID_PREFIX = "161061"


def current_nba_season(now=None):
    """Return the NBA season string for the current date, e.g. 2025-26."""
    eastern_now = now or datetime.now(EASTERN_TZ)
    start_year = eastern_now.year if eastern_now.month >= 7 else eastern_now.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def normalize_est_datetime(series):
    # The NBA schedule endpoint labels gameDateTimeEst values with "Z", but the
    # value itself is already Eastern time. Strip timezone markers and keep ET.
    clean = series.astype(str).str.replace(r"(Z|[+-]\d\d:\d\d)$", "", regex=True)
    return pd.to_datetime(clean, errors="coerce")


def clean_schedule_frame(frame):
    frame = frame.copy()
    frame.rename(
        columns={
            "homeTeam_teamId": "homeTeamId",
            "awayTeam_teamId": "awayTeamId",
            "homeTeam_teamCity": "homeTeamCity",
            "awayTeam_teamCity": "awayTeamCity",
            "homeTeam_teamName": "homeTeamName",
            "awayTeam_teamName": "awayTeamName",
            "homeTeam_teamTricode": "homeTeamTricode",
            "awayTeam_teamTricode": "awayTeamTricode",
            "day": "gameDay",
        },
        inplace=True,
    )

    required_cols = ["gameId", "gameDateTimeEst", "homeTeamId", "awayTeamId"]
    missing_cols = [col for col in required_cols if col not in frame.columns]
    if missing_cols:
        raise ValueError(f"Fehlende Spalten im NBA-Schedule: {missing_cols}")

    frame["gameDateTimeEst"] = normalize_est_datetime(frame["gameDateTimeEst"])
    frame.dropna(subset=["gameDateTimeEst"], inplace=True)

    frame["homeTeamId"] = frame["homeTeamId"].astype(str)
    frame["awayTeamId"] = frame["awayTeamId"].astype(str)
    nba_mask = (
        frame["homeTeamId"].str.startswith(NBA_TEAM_ID_PREFIX)
        & frame["awayTeamId"].str.startswith(NBA_TEAM_ID_PREFIX)
    )
    frame = frame.loc[nba_mask].copy()

    frame["gameId"] = frame["gameId"].astype(str).str.replace(r"\.0$", "", regex=True)
    frame.sort_values("gameDateTimeEst", inplace=True)
    frame.drop_duplicates(subset=["gameId"], keep="last", inplace=True)

    output_cols = [
        "gameId",
        "gameDateTimeEst",
        "homeTeamId",
        "awayTeamId",
        "homeTeamCity",
        "homeTeamName",
        "awayTeamCity",
        "awayTeamName",
        "gameDay",
        "arenaName",
        "arenaCity",
        "arenaState",
        "gameLabel",
        "gameSubLabel",
        "gameSubtype",
        "seriesGameNumber",
        "weekNumber",
    ]
    for col in output_cols:
        if col not in frame.columns:
            frame[col] = pd.NA

    return frame[output_cols].reset_index(drop=True)


def fetch_league_schedule(season, retries=3, delay_seconds=2):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            schedule = scheduleleaguev2.ScheduleLeagueV2(
                league_id="00",
                season=season,
                timeout=30,
            )
            frames = schedule.get_data_frames()
            if not frames or frames[0].empty:
                raise RuntimeError(f"Keine Schedule-Daten fuer Saison {season} erhalten.")
            return clean_schedule_frame(frames[0])
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                print(f"Versuch {attempt} fehlgeschlagen: {exc}. Neuer Versuch...")
                time.sleep(delay_seconds)

    raise RuntimeError(f"NBA-Schedule konnte nicht geladen werden: {last_error}")


def upcoming_games(season=None, days=None):
    eastern_now = datetime.now(EASTERN_TZ)
    selected_season = season or current_nba_season(eastern_now)

    schedule = fetch_league_schedule(selected_season)
    now_naive = pd.Timestamp(eastern_now.replace(tzinfo=None))
    upcoming = schedule[schedule["gameDateTimeEst"] >= now_naive].copy()

    if days is not None:
        end_time = now_naive + pd.Timedelta(days=days)
        upcoming = upcoming[upcoming["gameDateTimeEst"] < end_time].copy()

    return upcoming.sort_values("gameDateTimeEst").reset_index(drop=True)


def save_upcoming_games(output_path=DEFAULT_OUTPUT, season=None, days=None):
    games = upcoming_games(season=season, days=days)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not games.empty:
        games["gameDateTimeEst"] = games["gameDateTimeEst"].dt.strftime("%Y-%m-%d %H:%M:%S")

    games.to_csv(output_path, index=False)
    print(f"OK: {len(games)} kommende Spiele gespeichert: {output_path}")
    return games


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape kommende NBA-Spiele und speichere sie in US Eastern Time."
    )
    parser.add_argument("--season", default=None, help="NBA-Saison, z.B. 2025-26.")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Optional: nur Spiele innerhalb der naechsten N Tage speichern.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Zieldatei. Standard: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_upcoming_games(output_path=args.output, season=args.season, days=args.days)
