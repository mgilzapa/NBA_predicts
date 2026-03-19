import os
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.dirname(__file__))
PLAYER_BOX = os.path.join(base_dir, "data", "player_boxscores.csv")

base_games_path = os.path.join(base_dir, "data", "base_games.csv")
df = pd.read_csv(base_games_path)

df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])


# ------------------------------------------------------------
# BASIS: Heim- und Auswärtsspiele für Team-History
# ------------------------------------------------------------
home_games = pd.DataFrame({
    "date": df["gameDateTimeEst"],
    "team": df["hometeamName"],
    "opponent": df["awayteamName"],
    "win": (df["homeScore"] > df["awayScore"]).astype(int),
    "points": df["homeScore"],
    "points_allowed": df["awayScore"],
    "is_home": 1
})

away_games = pd.DataFrame({
    "date": df["gameDateTimeEst"],
    "team": df["awayteamName"],
    "opponent": df["hometeamName"],
    "win": (df["awayScore"] > df["homeScore"]).astype(int),
    "points": df["awayScore"],
    "points_allowed": df["homeScore"],
    "is_home": 0
})

team_history = pd.concat([home_games, away_games], ignore_index=True)
team_history = team_history.sort_values(["team", "date"]).reset_index(drop=True)

# ------------------------------------------------------------
# FEATURE 1: Letzte 5 Spiele Winrate 
# ------------------------------------------------------------
team_history["last5_winrate"] = (
    team_history.groupby("team")["win"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# ------------------------------------------------------------
# FEATURE 2: Letzte 5 Spiele Punkte 
# ------------------------------------------------------------
team_history["last5_avg_points"] = (
    team_history.groupby("team")["points"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

team_history["last5_avg_points_allowed"] = (
    team_history.groupby("team")["points_allowed"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# ------------------------------------------------------------
# FEATURE 3: Ruhetage 
# ------------------------------------------------------------
team_history["rest_days"] = (
    team_history.groupby("team")["date"]
    .diff()
    .dt.days
)
team_history["is_back_to_back"] = (team_history["rest_days"] == 1).astype(int)

# ------------------------------------------------------------
# FEATURE 4: Gegner-Stärke (Strength of Schedule) 
# ------------------------------------------------------------
# Winrate der letzten 5 Gegner
team_history["opponent_last5_winrate"] = (
    team_history.groupby("opponent")["win"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# ------------------------------------------------------------
# FEATURE 5: Heim/Auswärts getrennte Winrate 
# ------------------------------------------------------------
# Nur Heimspiele
team_history["home_winrate"] = (
    team_history[team_history["is_home"]==1]
    .groupby("team")["win"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

# Nur Auswärtsspiele
team_history["away_winrate"] = (
    team_history[team_history["is_home"]==0]
    .groupby("team")["win"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

# Für Zeilen, wo keine Heim-/Auswärts-Historie existiert, mit Gesamt-Winrate füllen
team_history["home_winrate"] = team_history.groupby("team")["home_winrate"].ffill()
team_history["away_winrate"] = team_history.groupby("team")["away_winrate"].ffill()

# ------------------------------------------------------------
# FEATURE 6: Saison-Phase (Spielnummer) 
# ------------------------------------------------------------
# Spielnummer pro Team und Saison berechnen
team_history["season"] = team_history["date"].dt.year
team_history["game_number"] = team_history.groupby(["team", "season"]).cumcount() + 1

# Saison-Phase als Kategorie (0-20, 21-60, 61-82)
team_history["season_phase"] = pd.cut(
    team_history["game_number"], 
    bins=[0, 20, 60, 100], 
    labels=["Anfang", "Mitte", "Ende"]
)
# One-Hot-Encoding für die Phasen
season_dummies = pd.get_dummies(team_history["season_phase"], prefix="phase")
team_history = pd.concat([team_history, season_dummies], axis=1)

# ------------------------------------------------------------
# FEATURE 7: Overtime-Indikator 
# ------------------------------------------------------------
# Grober Overtime-Indikator: Summe > 220 Punkte (ca. 110 pro Team)
df["overtime"] = (df["homeScore"] + df["awayScore"] > 220).astype(int)

# Overtime-Info in team_history mergen
overtime_info = df[["gameDateTimeEst", "hometeamName", "awayteamName", "overtime"]].copy()
overtime_info["date"] = overtime_info["gameDateTimeEst"]
overtime_info_home = overtime_info[["date", "hometeamName", "overtime"]].rename(columns={"hometeamName": "team"})
overtime_info_away = overtime_info[["date", "awayteamName", "overtime"]].rename(columns={"awayteamName": "team"})
overtime_all = pd.concat([overtime_info_home, overtime_info_away], ignore_index=True)

team_history = team_history.merge(
    overtime_all[["date", "team", "overtime"]], 
    on=["date", "team"], 
    how="left"
)
team_history["last_game_overtime"] = team_history.groupby("team")["overtime"].shift(1)

# ------------------------------------------------------------
# FEATURE 8: Division/Rivalität 
# ------------------------------------------------------------
# Team zu Division Mapping (Quelle: NBA)
divisions = {
    # Atlantic
    "Boston Celtics": "Atlantic", "Brooklyn Nets": "Atlantic", "New York Knicks": "Atlantic", 
    "Philadelphia 76ers": "Atlantic", "Toronto Raptors": "Atlantic",
    # Central
    "Chicago Bulls": "Central", "Cleveland Cavaliers": "Central", "Detroit Pistons": "Central",
    "Indiana Pacers": "Central", "Milwaukee Bucks": "Central",
    # Southeast
    "Atlanta Hawks": "Southeast", "Charlotte Hornets": "Southeast", "Miami Heat": "Southeast",
    "Orlando Magic": "Southeast", "Washington Wizards": "Southeast",
    # Northwest
    "Denver Nuggets": "Northwest", "Minnesota Timberwolves": "Northwest", 
    "Oklahoma City Thunder": "Northwest", "Portland Trail Blazers": "Northwest", 
    "Utah Jazz": "Northwest",
    # Pacific
    "Golden State Warriors": "Pacific", "LA Clippers": "Pacific", "Los Angeles Lakers": "Pacific",
    "Phoenix Suns": "Pacific", "Sacramento Kings": "Pacific",
    # Southwest
    "Dallas Mavericks": "Southwest", "Houston Rockets": "Southwest", "Memphis Grizzlies": "Southwest",
    "New Orleans Pelicans": "Southwest", "San Antonio Spurs": "Southwest"
}

df["home_division"] = df["hometeamName"].map(divisions)
df["away_division"] = df["awayteamName"].map(divisions)
df["same_division"] = (df["home_division"] == df["away_division"]).astype(int)

# ------------------------------------------------------------
# ALLE FEATURES FÜR HEIM UND AUSWÄRTS VORBEREITEN
# ------------------------------------------------------------
# Winrate
home_features = team_history[["date", "team", "last5_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "last5_winrate": "home_last5_winrate"
})
away_features = team_history[["date", "team", "last5_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "last5_winrate": "away_last5_winrate"
})

# Punkte
home_pts = team_history[["date", "team", "last5_avg_points"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "last5_avg_points": "home_last5_avg_points"
})
away_pts = team_history[["date", "team", "last5_avg_points"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "last5_avg_points": "away_last5_avg_points"
})

# Gegnerpunkte
home_pts_allowed = team_history[["date", "team", "last5_avg_points_allowed"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "last5_avg_points_allowed": "home_last5_avg_points_allowed"
})
away_pts_allowed = team_history[["date", "team", "last5_avg_points_allowed"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "last5_avg_points_allowed": "away_last5_avg_points_allowed"
})

# Ruhetage
home_rest = team_history[["date", "team", "rest_days", "is_back_to_back"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "rest_days": "home_rest_days", 
    "is_back_to_back": "home_is_back_to_back"
})
away_rest = team_history[["date", "team", "rest_days", "is_back_to_back"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "rest_days": "away_rest_days",
    "is_back_to_back": "away_is_back_to_back"
})

# Gegner-Stärke
home_sos = team_history[["date", "team", "opponent_last5_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "opponent_last5_winrate": "home_opponent_strength"
})
away_sos = team_history[["date", "team", "opponent_last5_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "opponent_last5_winrate": "away_opponent_strength"
})

# Heim/Auswärts Winrate
home_home_winrate = team_history[["date", "team", "home_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "home_winrate": "home_home_winrate"
})
away_home_winrate = team_history[["date", "team", "home_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "home_winrate": "away_home_winrate"
})
home_away_winrate = team_history[["date", "team", "away_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "away_winrate": "home_away_winrate"
})
away_away_winrate = team_history[["date", "team", "away_winrate"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "away_winrate": "away_away_winrate"
})

# Saison-Phase
phase_cols = [col for col in team_history.columns if col.startswith("phase_")]
home_phase = team_history[["date", "team"] + phase_cols].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName"
})
away_phase = team_history[["date", "team"] + phase_cols].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName"
})

# Overtime
home_ot = team_history[["date", "team", "last_game_overtime"]].rename(columns={
    "date": "gameDateTimeEst", "team": "hometeamName", "last_game_overtime": "home_last_game_overtime"
})
away_ot = team_history[["date", "team", "last_game_overtime"]].rename(columns={
    "date": "gameDateTimeEst", "team": "awayteamName", "last_game_overtime": "away_last_game_overtime"
})

# ------------------------------------------------------------
# ALLE FEATURES IN DAS HAUPTDATAFRAME MERGEN
# ------------------------------------------------------------
df = df.merge(home_features, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_features, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_pts, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_pts, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_pts_allowed, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_pts_allowed, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_rest, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_rest, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_sos, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_sos, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_home_winrate, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_home_winrate, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_away_winrate, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_away_winrate, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_phase, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_phase, on=["gameDateTimeEst", "awayteamName"], how="left")
df = df.merge(home_ot, on=["gameDateTimeEst", "hometeamName"], how="left")
df = df.merge(away_ot, on=["gameDateTimeEst", "awayteamName"], how="left")

# Division-Feature direkt aus df
df["same_division"] = (df["home_division"] == df["away_division"]).astype(int)

if os.path.exists(PLAYER_BOX):
    print("Lade Spieler-Boxscores und berechne Team-Aggregate...")
    box = pd.read_csv(PLAYER_BOX)

    # Spalten umbenennen, falls nötig (personId -> PLAYER_ID)
    if 'personId' in box.columns:
        box.rename(columns={'personId': 'PLAYER_ID'}, inplace=True)

    # Teamname-Mapping (Kurz -> Vollständig)
    team_name_mapping = {
        '76ers': 'Philadelphia 76ers',
        'Bucks': 'Milwaukee Bucks',
        'Bulls': 'Chicago Bulls',
        'Cavaliers': 'Cleveland Cavaliers',
        'Celtics': 'Boston Celtics',
        'Clippers': 'LA Clippers',
        'Grizzlies': 'Memphis Grizzlies',
        'Hawks': 'Atlanta Hawks',
        'Heat': 'Miami Heat',
        'Hornets': 'Charlotte Hornets',
        'Jazz': 'Utah Jazz',
        'Kings': 'Sacramento Kings',
        'Knicks': 'New York Knicks',
        'Lakers': 'Los Angeles Lakers',
        'Magic': 'Orlando Magic',
        'Mavericks': 'Dallas Mavericks',
        'Nets': 'Brooklyn Nets',
        'Nuggets': 'Denver Nuggets',
        'Pacers': 'Indiana Pacers',
        'Pelicans': 'New Orleans Pelicans',
        'Pistons': 'Detroit Pistons',
        'Raptors': 'Toronto Raptors',
        'Rockets': 'Houston Rockets',
        'Spurs': 'San Antonio Spurs',
        'Suns': 'Phoenix Suns',
        'Thunder': 'Oklahoma City Thunder',
        'Timberwolves': 'Minnesota Timberwolves',
        'Trail Blazers': 'Portland Trail Blazers',
        'Warriors': 'Golden State Warriors',
        'Wizards': 'Washington Wizards'
    }
    box['teamNameFull'] = box['teamName'].map(team_name_mapping)

    # Fehlende Teams entfernen
    box = box.dropna(subset=['teamNameFull'])

    # Minuten parsen (falls als String wie "27:15")
    if box['minutes'].dtype == 'object':
        def parse_minutes(m):
            if isinstance(m, str) and ':' in m:
                parts = m.split(':')
                return int(parts[0]) + int(parts[1])/60
            try:
                return float(m)
            except:
                return 0.0
        box['minutes'] = box['minutes'].apply(parse_minutes)

    # Aggregierte Team-Statistiken pro Spiel (Summe)
    stat_cols = ['points', 'reboundsDefensive', 'reboundsOffensive', 'assists', 'steals', 'blocks', 'turnovers', 'minutes']
    team_game_stats = box.groupby(['GAME_ID', 'teamNameFull'])[stat_cols].sum().reset_index()
    team_game_stats['player_count'] = box.groupby(['GAME_ID', 'teamNameFull']).size().values

    # Diese Werte an df mergen (Heim und Auswärts)
    df = df.merge(
        team_game_stats.add_prefix('home_'),
        left_on=['gameId', 'hometeamName'],
        right_on=['home_GAME_ID', 'home_teamNameFull'],
        how='left'
    )
    df = df.merge(
        team_game_stats.add_prefix('away_'),
        left_on=['gameId', 'awayteamName'],
        right_on=['away_GAME_ID', 'away_teamNameFull'],
        how='left'
    )
    # Hilfsspalten entfernen
    df.drop(columns=['home_GAME_ID', 'home_teamNameFull', 'away_GAME_ID', 'away_teamNameFull'], inplace=True, errors='ignore')

        # Heim-Statistiken
    home_stats = df[['gameDateTimeEst', 'hometeamName', 
                    'home_points', 'home_reboundsDefensive', 'home_reboundsOffensive', 
                    'home_assists', 'home_minutes', 'home_player_count']].copy()

    # Gesamt-Rebounds berechnen (defensive + offensive)
    home_stats['home_rebounds'] = home_stats['home_reboundsDefensive'] + home_stats['home_reboundsOffensive']

    # Nur benötigte Spalten behalten und umbenennen
    home_stats = home_stats[['gameDateTimeEst', 'hometeamName', 'home_points', 'home_rebounds', 'home_assists', 'home_minutes', 'home_player_count']]
    home_stats.rename(columns={
        'hometeamName': 'team',
        'home_points': 'PTS',
        'home_rebounds': 'REB',
        'home_assists': 'AST',
        'home_minutes': 'MIN',
        'home_player_count': 'player_count'
    }, inplace=True)

    # Auswärts-Statistiken
    away_stats = df[['gameDateTimeEst', 'awayteamName', 
                    'away_points', 'away_reboundsDefensive', 'away_reboundsOffensive', 
                    'away_assists', 'away_minutes', 'away_player_count']].copy()

    away_stats['away_rebounds'] = away_stats['away_reboundsDefensive'] + away_stats['away_reboundsOffensive']

    away_stats = away_stats[['gameDateTimeEst', 'awayteamName', 'away_points', 'away_rebounds', 'away_assists', 'away_minutes', 'away_player_count']]
    away_stats.rename(columns={
        'awayteamName': 'team',
        'away_points': 'PTS',
        'away_rebounds': 'REB',
        'away_assists': 'AST',
        'away_minutes': 'MIN',
        'away_player_count': 'player_count'
    }, inplace=True)

    team_stats_all = pd.concat([home_stats, away_stats], ignore_index=True)
    team_stats_all.sort_values(['team', 'gameDateTimeEst'], inplace=True)
    

    metrics = ['PTS', 'REB', 'AST', 'MIN', 'player_count']

    for metric in metrics:
        if metric in team_stats_all.columns:
            # Konvertiere zu numeric, setze Fehler auf NaN und fülle NaN mit 0
            team_stats_all[metric] = pd.to_numeric(team_stats_all[metric], errors='coerce').fillna(0)
            #print(f"Metrik {metric} konvertiert. Neuer Typ: {team_stats_all[metric].dtype}")
        else:
            break
            #print(f"Warnung: Metrik {metric} nicht in team_stats_all – überspringe.")

    # Gleitende Mittelwerte der letzten 5 Spiele für jede Metrik
    
    for metric in metrics:
        if metric in team_stats_all.columns:
            col_name = f'last5_{metric.lower()}'
            team_stats_all[col_name] = (
                team_stats_all.groupby('team')[metric]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            )
            #print(f"{col_name} berechnet.")

    # Aufteilen in Heim/Auswärts
    home_last5 = team_stats_all[['gameDateTimeEst', 'team', 'last5_pts', 'last5_reb', 'last5_ast', 'last5_min', 'last5_player_count']].rename(
        columns={'team': 'hometeamName', 'last5_pts': 'home_last5_pts', 'last5_reb': 'home_last5_reb',
                 'last5_ast': 'home_last5_ast', 'last5_min': 'home_last5_min', 'last5_player_count': 'home_last5_player_count'}
    )
    away_last5 = team_stats_all[['gameDateTimeEst', 'team', 'last5_pts', 'last5_reb', 'last5_ast', 'last5_min', 'last5_player_count']].rename(
        columns={'team': 'awayteamName', 'last5_pts': 'away_last5_pts', 'last5_reb': 'away_last5_reb',
                 'last5_ast': 'away_last5_ast', 'last5_min': 'away_last5_min', 'last5_player_count': 'away_last5_player_count'}
    )

    df = df.merge(home_last5, on=['gameDateTimeEst', 'hometeamName'], how='left')
    df = df.merge(away_last5, on=['gameDateTimeEst', 'awayteamName'], how='left')

    # Differenz-Features
    df['pts_diff_last5'] = df['home_last5_pts'] - df['away_last5_pts']
    df['reb_diff_last5'] = df['home_last5_reb'] - df['away_last5_reb']
    df['ast_diff_last5'] = df['home_last5_ast'] - df['away_last5_ast']
    df['min_diff_last5'] = df['home_last5_min'] - df['away_last5_min']
    df['player_count_diff_last5'] = df['home_last5_player_count'] - df['away_last5_player_count']

    print("Spieler-Features erfolgreich hinzugefügt.")
else:
    print(f"Warnung: {PLAYER_BOX} nicht gefunden – überspringe Spieler-Features.")


model_output_path = os.path.join(base_dir, "data", "model_data.csv")
df.to_csv(model_output_path, index=False)
print(f"Gespeichert: {model_output_path}")
