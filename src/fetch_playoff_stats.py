"""fetch_playoff_stats.py

Computes per-team playoff rolling stats for the current season from
data/nba_api_games.csv. Writes data/playoff_stats.csv.

Run as step 6b in run_all.py (playoffs only).
"""
import os
import json
import sys

import numpy as np
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NBA_GAMES_CSV = os.path.join(BASE_DIR, "data", "nba_api_games.csv")
BRACKET_JSON  = os.path.join(BASE_DIR, "web", "bracket.json")
OUTPUT_CSV    = os.path.join(BASE_DIR, "data", "playoff_stats.csv")

CURRENT_SEASON = "2025-26"


def _season_label(model_season_int):
    """Convert model_data season int (e.g. 2025) to nba_api season string ('2025-26')."""
    y = int(model_season_int)
    return f"{y}-{str(y + 1)[-2:]}"


def _round_from_game_id(game_id):
    """Extract playoff round number from GAME_ID like 42500231 → 2."""
    return int(str(int(game_id))[5])


def _compute_rolling(games_df, current_season):
    """Per-team current-season playoff rolling stats (pts and margin)."""
    curr = games_df[
        (games_df["season"] == current_season) &
        (games_df["season_type"] == "Playoffs")
    ].copy()

    if curr.empty:
        return pd.DataFrame(columns=[
            "team_name", "playoff_games_played",
            "playoff_pts_last3", "playoff_margin_last3",
            "prev_series_games",
        ])

    curr["GAME_DATE"] = pd.to_datetime(curr["GAME_DATE"])
    curr = curr.sort_values("GAME_DATE")
    curr["round"] = curr["GAME_ID"].apply(_round_from_game_id)

    # Build per-team flat game log
    rows = []
    for _, row in curr.iterrows():
        h_pts = float(row["homeScore"])
        a_pts = float(row["awayScore"])
        for team, pts, margin, rnd in [
            (row["hometeamName"], h_pts, h_pts - a_pts, row["round"]),
            (row["awayteamName"], a_pts, a_pts - h_pts, row["round"]),
        ]:
            rows.append({"team": team, "date": row["GAME_DATE"],
                         "pts": pts, "margin": margin, "round": rnd})

    team_log = pd.DataFrame(rows).sort_values("date")

    stats = []
    for team, grp in team_log.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)
        last3 = grp.tail(3)
        curr_round = int(grp["round"].max())
        prev_games = int((grp["round"] == curr_round - 1).sum()) if curr_round > 1 else 0
        stats.append({
            "team_name": team,
            "playoff_games_played": len(grp),
            "playoff_pts_last3": round(float(last3["pts"].mean()), 2),
            "playoff_margin_last3": round(float(last3["margin"].mean()), 2),
            "prev_series_games": prev_games,
        })

    return pd.DataFrame(stats)


def _compute_hist_winrate(games_df):
    """Per-team historical home win rate in playoff games (all seasons)."""
    hist = games_df[games_df["season_type"] == "Playoffs"].copy()
    if hist.empty:
        return pd.DataFrame(columns=["team_name", "playoff_home_winrate_hist"])

    agg = hist.groupby("hometeamName").agg(
        total=("home_win", "count"),
        wins=("home_win", "sum"),
    ).reset_index()
    agg["playoff_home_winrate_hist"] = (agg["wins"] / agg["total"]).round(4)
    return agg[["hometeamName", "playoff_home_winrate_hist"]].rename(
        columns={"hometeamName": "team_name"}
    )


def main():
    if not os.path.exists(NBA_GAMES_CSV):
        print(f"ERROR: {NBA_GAMES_CSV} not found")
        return 1

    games = pd.read_csv(NBA_GAMES_CSV)

    rolling  = _compute_rolling(games, CURRENT_SEASON)
    hist_wr  = _compute_hist_winrate(games)

    if rolling.empty:
        print(f"No playoff games found for season {CURRENT_SEASON}")
        return 1

    result = rolling.merge(hist_wr, on="team_name", how="left")
    result["playoff_home_winrate_hist"] = result["playoff_home_winrate_hist"].fillna(0.5)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"OK: playoff_stats.csv written ({len(result)} teams)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
