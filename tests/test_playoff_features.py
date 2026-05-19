import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Test 1: is_playoff fix ────────────────────────────────────────────────────
class TestIsPlayoff:
    def _make_df(self):
        return pd.DataFrame({
            "gameId":          [1001, 1002, 2001],
            "gameDateTimeEst": pd.to_datetime(["2025-04-15", "2025-04-17", "2025-01-10"]),
            "hometeamName":    ["Spurs", "Spurs", "Celtics"],
            "awayteamName":    ["Thunder", "Thunder", "Lakers"],
            "homeScore":       [110, 105, 120],
            "awayScore":       [100, 108, 115],
        })

    def _make_nba_api(self):
        return pd.DataFrame({
            "GAME_ID":     [1001, 1002, 2001],
            "season_type": ["Playoffs", "Playoffs", "Regular Season"],
        })

    def test_playoff_games_flagged(self):
        df = self._make_df()
        nba_api = self._make_nba_api()
        df = df.merge(nba_api.rename(columns={"GAME_ID": "gameId"}), on="gameId", how="left")
        df["is_playoff"] = (df["season_type"].fillna("") == "Playoffs").astype(int)
        assert df.loc[df["gameId"] == 1001, "is_playoff"].iloc[0] == 1
        assert df.loc[df["gameId"] == 2001, "is_playoff"].iloc[0] == 0

    def test_no_nulls_after_join(self):
        df = self._make_df()
        nba_api = self._make_nba_api()
        df = df.merge(nba_api.rename(columns={"GAME_ID": "gameId"}), on="gameId", how="left")
        df["is_playoff"] = (df["season_type"].fillna("") == "Playoffs").astype(int)
        assert df["is_playoff"].isna().sum() == 0


# ── Test 2: H2H per-season ────────────────────────────────────────────────────
class TestH2HPerSeason:
    def _make_team_history(self):
        rows = [
            # Season 2024: Spurs beat Thunder 3 times
            {"team": "Spurs",   "opponent": "Thunder", "season": 2024, "win": 1,
             "date": pd.Timestamp("2024-11-01")},
            {"team": "Thunder", "opponent": "Spurs",   "season": 2024, "win": 0,
             "date": pd.Timestamp("2024-11-01")},
            {"team": "Spurs",   "opponent": "Thunder", "season": 2024, "win": 1,
             "date": pd.Timestamp("2024-12-01")},
            {"team": "Thunder", "opponent": "Spurs",   "season": 2024, "win": 0,
             "date": pd.Timestamp("2024-12-01")},
            {"team": "Spurs",   "opponent": "Thunder", "season": 2024, "win": 1,
             "date": pd.Timestamp("2025-01-01")},
            {"team": "Thunder", "opponent": "Spurs",   "season": 2024, "win": 0,
             "date": pd.Timestamp("2025-01-01")},
            # Season 2025: starts fresh
            {"team": "Spurs",   "opponent": "Thunder", "season": 2025, "win": 0,
             "date": pd.Timestamp("2025-11-01")},
            {"team": "Thunder", "opponent": "Spurs",   "season": 2025, "win": 1,
             "date": pd.Timestamp("2025-11-01")},
        ]
        th = pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)
        return th

    def _compute_h2h(self, th):
        h2h = (
            th.groupby(["team", "opponent", "season"], group_keys=False)["win"]
            .apply(lambda x: x.shift(1).expanding().mean())
        )
        th["h2h_winrate"] = h2h.fillna(0.5)
        return th

    def test_first_meeting_is_neutral(self):
        th = self._compute_h2h(self._make_team_history())
        first_spurs_2024 = th[(th["team"] == "Spurs") & (th["season"] == 2024)].iloc[0]
        assert first_spurs_2024["h2h_winrate"] == 0.5

    def test_season_resets(self):
        th = self._compute_h2h(self._make_team_history())
        first_spurs_2025 = th[(th["team"] == "Spurs") & (th["season"] == 2025)].iloc[0]
        assert first_spurs_2025["h2h_winrate"] == 0.5

    def test_h2h_accumulates_within_season(self):
        th = self._compute_h2h(self._make_team_history())
        third_spurs_2024 = th[(th["team"] == "Spurs") & (th["season"] == 2024)].iloc[2]
        assert third_spurs_2024["h2h_winrate"] == 1.0

    def test_no_nulls_after_fillna(self):
        th = self._compute_h2h(self._make_team_history())
        assert th["h2h_winrate"].isna().sum() == 0


# ── Test 3: Series Record ─────────────────────────────────────────────────────
class TestSeriesRecord:
    def _apply_series_record(self, df):
        playoffs = df[df["is_playoff"] == 1].copy()
        if not playoffs.empty:
            playoffs = playoffs.sort_values("gameDateTimeEst")
            playoffs["series_key"] = playoffs.apply(
                lambda r: str(sorted([r["hometeamName"], r["awayteamName"]])) + str(r["season"]),
                axis=1,
            )

            def compute_series_wins(group):
                group = group.sort_values("gameDateTimeEst")
                group["home_series_wins"] = (
                    group["home_win"].shift(1).expanding().sum().fillna(0).astype(int)
                )
                group["away_series_wins"] = (
                    (1 - group["home_win"]).shift(1).expanding().sum().fillna(0).astype(int)
                )
                return group

            playoffs = playoffs.groupby("series_key", group_keys=False).apply(compute_series_wins, include_groups=False)
            playoffs["series_wins_diff"] = playoffs["home_series_wins"] - playoffs["away_series_wins"]
            playoffs["is_elimination_game"] = (
                (playoffs["home_series_wins"] == 3) | (playoffs["away_series_wins"] == 3)
            ).astype(int)
            df = df.merge(
                playoffs[["gameId", "home_series_wins", "away_series_wins",
                           "series_wins_diff", "is_elimination_game"]],
                on="gameId", how="left",
            )
        for col in ["home_series_wins", "away_series_wins", "series_wins_diff", "is_elimination_game"]:
            df[col] = df.get(col, pd.Series(0, index=df.index)).fillna(0).astype(int)
        return df

    def _make_playoff_df(self):
        return pd.DataFrame({
            "gameId":          [1001, 1002, 1003],
            "gameDateTimeEst": pd.to_datetime(["2025-04-15", "2025-04-17", "2025-04-19"]),
            "hometeamName":    ["Spurs",   "Spurs",   "Thunder"],
            "awayteamName":    ["Thunder", "Thunder", "Spurs"],
            "home_win":        [1, 1, 0],
            "is_playoff":      [1, 1, 1],
            "season":          [2025, 2025, 2025],
        })

    def test_no_leakage_game1(self):
        df = self._apply_series_record(self._make_playoff_df())
        g1 = df[df["gameId"] == 1001].iloc[0]
        assert g1["home_series_wins"] == 0 and g1["away_series_wins"] == 0

    def test_series_wins_accumulate(self):
        df = self._apply_series_record(self._make_playoff_df())
        g2 = df[df["gameId"] == 1002].iloc[0]
        assert g2["home_series_wins"] == 1 and g2["away_series_wins"] == 0

    def test_regular_season_all_zeros(self):
        df = pd.DataFrame({
            "gameId":          [9001, 9002],
            "gameDateTimeEst": pd.to_datetime(["2025-01-10", "2025-01-12"]),
            "hometeamName":    ["Lakers", "Lakers"],
            "awayteamName":    ["Celtics", "Celtics"],
            "home_win":        [1, 0],
            "is_playoff":      [0, 0],
            "season":          [2024, 2024],
        })
        df = self._apply_series_record(df)
        for col in ["home_series_wins", "away_series_wins", "series_wins_diff", "is_elimination_game"]:
            assert (df[col] == 0).all(), f"{col} should be 0 for regular season"

    def test_elimination_game_flagged(self):
        df = pd.DataFrame({
            "gameId":          [1, 2, 3, 4],
            "gameDateTimeEst": pd.date_range("2025-04-15", periods=4, freq="2D"),
            "hometeamName":    ["A", "A", "A", "B"],
            "awayteamName":    ["B", "B", "B", "A"],
            "home_win":        [1, 1, 1, 0],
            "is_playoff":      [1, 1, 1, 1],
            "season":          [2025] * 4,
        })
        df = self._apply_series_record(df)
        g4 = df[df["gameId"] == 4].iloc[0]
        assert g4["is_elimination_game"] == 1
        g1 = df[df["gameId"] == 1].iloc[0]
        assert g1["is_elimination_game"] == 0
