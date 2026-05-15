import json
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import fetch_odds


def test_normalize_team():
    assert fetch_odds.normalize_team("Los Angeles Lakers") == "los angeles lakers"
    assert fetch_odds.normalize_team("  Boston Celtics  ") == "boston celtics"


def test_make_key():
    key = fetch_odds.make_key("2026-05-15", "Los Angeles Lakers", "Golden State Warriors")
    assert key == "2026-05-15|los angeles lakers|golden state warriors"


def test_parse_bookmakers_extracts_prices():
    api_game = {
        "home_team": "Los Angeles Lakers",
        "bookmakers": [
            {
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Los Angeles Lakers", "price": 1.85},
                            {"name": "Golden State Warriors", "price": 2.10},
                        ]
                    }
                ]
            }
        ]
    }
    result = fetch_odds.parse_bookmakers(api_game)
    assert len(result) == 1
    assert result[0] == {"name": "DraftKings", "home": 1.85, "away": 2.1}


def test_parse_bookmakers_skips_non_h2h():
    api_game = {
        "home_team": "Los Angeles Lakers",
        "bookmakers": [
            {
                "title": "DraftKings",
                "markets": [{"key": "spreads", "outcomes": []}]
            }
        ]
    }
    assert fetch_odds.parse_bookmakers(api_game) == []


def test_parse_bookmakers_skips_incomplete_outcomes():
    api_game = {
        "home_team": "Los Angeles Lakers",
        "bookmakers": [
            {
                "title": "BetMGM",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Los Angeles Lakers", "price": 1.75}
                        ]
                    }
                ]
            }
        ]
    }
    assert fetch_odds.parse_bookmakers(api_game) == []


def test_fetch_and_save_writes_valid_json():
    mock_api_response = [
        {
            "home_team": "Los Angeles Lakers",
            "away_team": "Golden State Warriors",
            "commence_time": "2026-05-15T23:10:00Z",
            "bookmakers": [
                {
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Los Angeles Lakers", "price": 1.90},
                                {"name": "Golden State Warriors", "price": 2.05},
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "odds.json")
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_api_response
        mock_resp.raise_for_status.return_value = None

        with patch("fetch_odds.requests.get", return_value=mock_resp):
            result = fetch_odds.fetch_and_save("test_key", output_path)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            data = json.load(f)

        expected_key = "2026-05-15|los angeles lakers|golden state warriors"
        assert expected_key in data["games"]
        game = data["games"][expected_key]
        assert game["home_team"] == "Los Angeles Lakers"
        assert game["bookmakers"][0]["name"] == "FanDuel"
        assert game["bookmakers"][0]["home"] == 1.90
        assert game["bookmakers"][0]["away"] == 2.05
        assert "generated_at" in data
