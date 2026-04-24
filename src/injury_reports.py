import html
import os
import re
from datetime import datetime, timezone

import pandas as pd
import requests


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PLAYER_BOX = os.path.join(BASE_DIR, "data", "player_boxscores.csv")
BASE_GAMES = os.path.join(BASE_DIR, "data", "base_games.csv")
INJURY_OUTPUT = os.path.join(BASE_DIR, "data", "current_injuries.csv")
INJURY_MATCH_OUTPUT = os.path.join(BASE_DIR, "data", "injury_player_matches.csv")
SOURCE_URL = "https://www.covers.com/sport/basketball/nba/injuries"

TEAM_NAMES = [
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "LA Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards",
]

SHORT_TO_FULL_TEAM = {
    "76ers": "Philadelphia 76ers",
    "Bucks": "Milwaukee Bucks",
    "Bulls": "Chicago Bulls",
    "Cavaliers": "Cleveland Cavaliers",
    "Celtics": "Boston Celtics",
    "Clippers": "LA Clippers",
    "Grizzlies": "Memphis Grizzlies",
    "Hawks": "Atlanta Hawks",
    "Heat": "Miami Heat",
    "Hornets": "Charlotte Hornets",
    "Jazz": "Utah Jazz",
    "Kings": "Sacramento Kings",
    "Knicks": "New York Knicks",
    "Lakers": "Los Angeles Lakers",
    "Magic": "Orlando Magic",
    "Mavericks": "Dallas Mavericks",
    "Nets": "Brooklyn Nets",
    "Nuggets": "Denver Nuggets",
    "Pacers": "Indiana Pacers",
    "Pelicans": "New Orleans Pelicans",
    "Pistons": "Detroit Pistons",
    "Raptors": "Toronto Raptors",
    "Rockets": "Houston Rockets",
    "Spurs": "San Antonio Spurs",
    "Suns": "Phoenix Suns",
    "Thunder": "Oklahoma City Thunder",
    "Timberwolves": "Minnesota Timberwolves",
    "Trail Blazers": "Portland Trail Blazers",
    "Warriors": "Golden State Warriors",
    "Wizards": "Washington Wizards",
}

POSITION_TOKENS = {"PG", "SG", "SF", "PF", "C", "G", "F"}
STATUS_KEYWORDS = [
    "out for season",
    "out",
    "doubtful",
    "questionable",
    "game time decision",
    "day to day",
    "probable",
]
TEAM_NAME_LOOKUP = {name: name for name in TEAM_NAMES}
TEAM_NAME_LOOKUP.update(SHORT_TO_FULL_TEAM)
INJURY_COLUMNS = [
    "team_name",
    "player_report_name",
    "position",
    "status",
    "updated",
    "note",
    "status_weight",
    "source_url",
]
DEBUG_ENV_VAR = "INJURY_DEBUG"


def parse_minutes(value):
    if isinstance(value, str) and ":" in value:
        minutes, seconds = value.split(":", 1)
        return int(minutes) + int(seconds) / 60
    try:
        return float(value)
    except Exception:
        return 0.0


def shifted_rolling(series, window, min_p=None):
    return series.shift(1).rolling(window, min_periods=min_p or max(1, window // 2))


def status_to_weight(status_text):
    status = str(status_text).strip().lower()
    if "out for season" in status:
        return 1.0
    if status.startswith("out"):
        return 1.0
    if "doubtful" in status:
        return 0.75
    if "game time decision" in status:
        return 0.50
    if "day to day" in status:
        return 0.35
    if "probable" in status:
        return 0.15
    return 0.0


def normalize_spaces(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def debug_enabled():
    return os.getenv(DEBUG_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def debug_print(message):
    if debug_enabled():
        print(f"[injury-debug] {message}")


def canonical_team_name(text):
    return TEAM_NAME_LOOKUP.get(normalize_spaces(text))


def looks_like_player_name(text):
    text = normalize_spaces(text)
    if not text:
        return False
    if canonical_team_name(text):
        return False
    if text.lower() in {"player", "pos", "status", "date", "notes", "note", "injuries"}:
        return False
    if text in POSITION_TOKENS:
        return False
    if re.match(r"^[A-Z]\.\s+[A-Za-z].*", text):
        return True
    if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$", text):
        return True
    return False


def looks_like_status(text):
    lowered = normalize_spaces(text).lower()
    return any(keyword in lowered for keyword in STATUS_KEYWORDS)


def clean_html_to_lines(raw_html):
    text = re.sub(r"<script.*?</script>", " ", raw_html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = html.unescape(text)
    lines = [normalize_spaces(line) for line in text.splitlines()]
    return [line for line in lines if line]


def parse_injury_lines(lines):
    rows = []
    current_team = None
    i = 0

    while i < len(lines):
        line = normalize_spaces(lines[i])
        canonical_team = canonical_team_name(line)

        if canonical_team:
            current_team = canonical_team
            i += 1
            continue

        if current_team is None:
            i += 1
            continue

        if line == "No injuries to report.":
            i += 1
            continue

        # Covers often splits player / position / status into separate text rows.
        if not looks_like_player_name(line):
            i += 1
            continue

        player = line
        position = ""
        status = ""
        updated = ""
        note = ""
        j = i + 1

        if j < len(lines) and normalize_spaces(lines[j]) in POSITION_TOKENS:
            position = normalize_spaces(lines[j])
            j += 1

        if j < len(lines) and looks_like_status(lines[j]):
            status = normalize_spaces(lines[j])
            j += 1
        elif j < len(lines):
            combined = f"{position} {normalize_spaces(lines[j])}".strip()
            if looks_like_status(combined):
                status = normalize_spaces(lines[j])
                j += 1

        if not status:
            i += 1
            continue

        if j < len(lines) and normalize_spaces(lines[j]).startswith("("):
            updated = normalize_spaces(lines[j]).strip("() ")
            j += 1

        note_parts = []
        while j < len(lines):
            next_line = normalize_spaces(lines[j])
            if not next_line:
                j += 1
                continue
            if canonical_team_name(next_line) or looks_like_player_name(next_line):
                break
            if next_line in POSITION_TOKENS or looks_like_status(next_line):
                break
            if next_line.lower() in {"player", "pos", "status", "date", "notes", "note"}:
                j += 1
                continue
            note_parts.append(next_line)
            j += 1
        note = " ".join(note_parts).strip()

        rows.append({
            "team_name": current_team,
            "player_report_name": player,
            "position": position,
            "status": status,
            "updated": updated,
            "note": note,
            "status_weight": status_to_weight(status),
            "source_url": SOURCE_URL,
        })
        i = j

    parsed = pd.DataFrame(rows, columns=INJURY_COLUMNS)
    debug_print(f"parse_injury_lines: {len(lines)} Textzeilen, {len(parsed)} erkannte Injury-Eintraege")
    if debug_enabled() and not parsed.empty:
        preview = parsed[["team_name", "player_report_name", "status"]].head(5).to_dict("records")
        debug_print(f"Beispiel-Eintraege: {preview}")
    return parsed


def fetch_current_injuries(output_path=INJURY_OUTPUT):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    }
    try:
        response = requests.get(SOURCE_URL, headers=headers, timeout=30)
        response.raise_for_status()
        debug_print(f"HTTP-Status: {response.status_code}")
        debug_print(f"HTML-Laenge: {len(response.text)} Zeichen")
        lines = clean_html_to_lines(response.text)
        debug_print(f"Nach HTML-Bereinigung: {len(lines)} Textzeilen")
        if debug_enabled() and lines:
            debug_print(f"Erste Zeilen: {lines[:15]}")
            status_hits = [
                (idx, line) for idx, line in enumerate(lines)
                if looks_like_status(line)
            ][:20]
            debug_print(f"Zeilen mit Status-Keywords: {status_hits}")
        injuries = parse_injury_lines(lines)
    except requests.RequestException as exc:
        debug_print(f"Netzwerkfehler beim Abruf: {exc}")
        injuries = pd.DataFrame(columns=INJURY_COLUMNS)

    injuries["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
    if "status_weight" not in injuries.columns:
        injuries["status_weight"] = pd.Series(dtype=float)
    injuries = injuries[injuries["status_weight"] > 0].copy()
    debug_print(f"Nach Status-Filter > 0: {len(injuries)} Eintraege")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if injuries.empty and len(injuries.columns) == 0:
        injuries = pd.DataFrame(columns=INJURY_COLUMNS + ["fetched_at_utc"])
    injuries.to_csv(output_path, index=False)
    debug_print(f"CSV geschrieben: {output_path}")
    return injuries


def build_player_importance_snapshot(
    base_games_path=BASE_GAMES,
    player_box_path=PLAYER_BOX,
):
    if not os.path.exists(base_games_path) or not os.path.exists(player_box_path):
        return pd.DataFrame()

    games = pd.read_csv(base_games_path)
    games["gameDateTimeEst"] = pd.to_datetime(games["gameDateTimeEst"])
    games["gameId"] = games["gameId"].astype(str).str.replace(r"\.0$", "", regex=True)

    box = pd.read_csv(player_box_path)
    if "personId" in box.columns:
        if "PLAYER_ID" in box.columns:
            box["PLAYER_ID"] = box["PLAYER_ID"].fillna(box["personId"])
        else:
            box.rename(columns={"personId": "PLAYER_ID"}, inplace=True)

    box["teamNameFull"] = box["teamName"].map(SHORT_TO_FULL_TEAM)
    box = box.dropna(subset=["teamNameFull"]).copy()
    box["minutes"] = box["minutes"].apply(parse_minutes)

    for stat_col in ["points", "reboundsTotal", "assists"]:
        box[stat_col] = pd.to_numeric(box[stat_col], errors="coerce").fillna(0)

    player_games = box[[
        "GAME_ID",
        "teamNameFull",
        "PLAYER_ID",
        "firstName",
        "familyName",
        "minutes",
        "points",
        "reboundsTotal",
        "assists",
    ]].copy()
    player_games["gameId"] = player_games["GAME_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    player_games["PLAYER_ID"] = player_games["PLAYER_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    player_games["firstName"] = player_games["firstName"].fillna("").astype(str).str.strip()
    player_games["familyName"] = player_games["familyName"].fillna("").astype(str).str.strip()
    player_games["player_name"] = (
        player_games["firstName"] + " " + player_games["familyName"]
    ).str.strip()
    player_games["report_name"] = (
        player_games["firstName"].str[:1] + ". " + player_games["familyName"]
    ).str.strip()
    player_games = player_games.merge(
        games[["gameId", "gameDateTimeEst"]],
        on="gameId",
        how="inner",
    )
    player_games.sort_values(["teamNameFull", "PLAYER_ID", "gameDateTimeEst"], inplace=True)

    for metric in ["minutes", "points", "reboundsTotal", "assists"]:
        player_games[f"prev_{metric}"] = (
            player_games.groupby(["teamNameFull", "PLAYER_ID"])[metric]
            .transform(lambda x: shifted_rolling(x, 5, 3).mean())
        )

    player_games["prev_games_played"] = (
        player_games.groupby(["teamNameFull", "PLAYER_ID"]).cumcount()
    )
    player_games["importance_score"] = (
        0.50 * player_games["prev_minutes"].fillna(0) +
        0.30 * player_games["prev_points"].fillna(0) +
        0.12 * player_games["prev_assists"].fillna(0) +
        0.08 * player_games["prev_reboundsTotal"].fillna(0)
    )
    player_games.loc[player_games["prev_games_played"] < 3, "importance_score"] = float("nan")

    latest = (
        player_games.sort_values("gameDateTimeEst")
        .groupby(["teamNameFull", "PLAYER_ID"], as_index=False)
        .last()
    )
    latest = latest.dropna(subset=["importance_score"]).copy()
    latest["family_name_norm"] = latest["familyName"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    latest["report_name_norm"] = latest["report_name"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    latest["player_name_norm"] = latest["player_name"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    debug_print(f"Player-Snapshot gebaut: {len(latest)} Spieler mit Importance")
    return latest


def merge_injuries_with_players(injuries, player_snapshot):
    if injuries.empty or player_snapshot.empty:
        debug_print(
            f"merge_injuries_with_players uebersprungen: injuries={len(injuries)}, player_snapshot={len(player_snapshot)}"
        )
        return pd.DataFrame()

    injuries = injuries.copy()
    injuries["report_name_norm"] = injuries["player_report_name"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    injuries["family_name_norm"] = (
        injuries["player_report_name"]
        .str.replace(r"^[A-Z]\.\s*", "", regex=True)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )

    direct = injuries.merge(
        player_snapshot,
        left_on=["team_name", "report_name_norm"],
        right_on=["teamNameFull", "report_name_norm"],
        how="left",
        suffixes=("", "_player"),
    )
    if "family_name_norm" not in direct.columns and "family_name_norm_player" in direct.columns:
        direct["family_name_norm"] = (
            injuries.reindex(direct.index)["family_name_norm"].values
        )

    unmatched = direct[direct["PLAYER_ID"].isna()].copy()
    matched = direct[direct["PLAYER_ID"].notna()].copy()

    if unmatched.empty:
        debug_print(f"Direkt gematcht: {len(matched)} Verletzungen")
        return matched

    player_family = player_snapshot.copy()
    family_counts = (
        player_family.groupby(["teamNameFull", "family_name_norm"])["PLAYER_ID"]
        .nunique()
        .reset_index(name="family_matches")
    )
    player_family = player_family.merge(
        family_counts,
        on=["teamNameFull", "family_name_norm"],
        how="left",
    )
    player_family = player_family[player_family["family_matches"] == 1].copy()

    fallback = unmatched.drop(
        columns=[
            col for col in player_snapshot.columns
            if col not in {"team_name", "player_report_name", "status", "status_weight", "report_name_norm", "family_name_norm"}
        ],
        errors="ignore",
    ).merge(
        player_family,
        left_on=["team_name", "family_name_norm"],
        right_on=["teamNameFull", "family_name_norm"],
        how="left",
    )
    fallback = fallback[fallback["PLAYER_ID"].notna()].copy()

    combined = pd.concat([matched, fallback], ignore_index=True)
    combined.to_csv(INJURY_MATCH_OUTPUT, index=False)
    debug_print(
        f"Matching abgeschlossen: direkt={len(matched)}, fallback={len(fallback)}, gesamt={len(combined)}"
    )
    if debug_enabled() and not unmatched.empty:
        unmatched_preview = unmatched[["team_name", "player_report_name", "status"]].head(10).to_dict("records")
        debug_print(f"Nicht direkt gematchte Injury-Eintraege: {unmatched_preview}")
    return combined


def compute_injury_features(injuries, player_snapshot):
    matched = merge_injuries_with_players(injuries, player_snapshot)
    if matched.empty:
        debug_print("compute_injury_features: keine gematchten Injuries")
        return pd.DataFrame()

    matched = matched.sort_values(["team_name", "importance_score"], ascending=[True, False]).copy()

    rows = []
    for team_name, group in player_snapshot.groupby("teamNameFull", sort=False):
        roster = group.sort_values("importance_score", ascending=False).copy()
        team_injuries = matched[matched["team_name"] == team_name].copy()

        if team_injuries.empty:
            continue

        injury_weight_map = (
            team_injuries.groupby("PLAYER_ID")["status_weight"]
            .max()
            .to_dict()
        )

        roster["injury_weight"] = roster["PLAYER_ID"].map(injury_weight_map).fillna(0.0)
        top3 = roster.head(3).copy()
        top5 = roster.head(5).copy()

        rows.append({
            "team_name": team_name,
            "injury_reported_count": float((roster["injury_weight"] > 0).sum()),
            "injury_weighted_count": float(roster["injury_weight"].sum()),
            "top3_missing_count": float(top3["injury_weight"].sum()),
            "top5_missing_count": float(top5["injury_weight"].sum()),
            "missing_top3_importance": float((top3["importance_score"] * top3["injury_weight"]).sum()),
            "missing_top5_importance": float((top5["importance_score"] * top5["injury_weight"]).sum()),
            "top3_availability_ratio": float(max(0.0, 1.0 - top3["injury_weight"].sum() / max(len(top3), 1))),
            "top5_availability_ratio": float(max(0.0, 1.0 - top5["injury_weight"].sum() / max(len(top5), 1))),
        })

    features = pd.DataFrame(rows)
    debug_print(f"compute_injury_features: {len(features)} Teams mit Injury-Features")
    if debug_enabled() and not features.empty:
        preview = features.head(10).to_dict("records")
        debug_print(f"Injury-Feature-Beispiel: {preview}")
    return features


if __name__ == "__main__":
    if debug_enabled():
        print(f"[injury-debug] Debug-Modus aktiv ueber {DEBUG_ENV_VAR}=1")
    injuries = fetch_current_injuries()
    print(f"Gespeichert: {INJURY_OUTPUT} ({len(injuries)} Eintraege)")
