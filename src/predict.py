import os
import pandas as pd
from xgboost import XGBClassifier


# -----------------------------
# 1. Historische Modelldaten laden und bereinigen
# -----------------------------
df = pd.read_csv("data/model_data.csv")
'''
df_model = df.dropna(subset=["home_last5_winrate",
                            "away_last5_winrate",
                            "home_last5_avg_points",
                            "away_last5_avg_points",
                            "home_rest_days",
                            "away_rest_days",
                            "home_last5_avg_points_allowed",
                            "away_last5_avg_points_allowed",
                            "home_is_back_to_back",
                            "away_is_back_to_back",
                            "home_opponent_strength",
                            "away_opponent_strength",
                            "home_home_winrate",
                            "away_home_winrate",
                            "home_away_winrate",
                            "away_away_winrate"
                            ]).copy()
'''
df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
#df_model = df.dropna(subset=feature_cols).copy()
df["winrate_diff"] = df["home_last5_winrate"] - df["away_last5_winrate"]
df["average_points_diff"] = df["home_last5_avg_points"] - df["away_last5_avg_points"]
df["average_points_allowed_diff"] = df["home_last5_avg_points_allowed"] - df["away_last5_avg_points_allowed"]
df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]

feature_cols =["home_last5_winrate",
                 "away_last5_winrate",
                 "winrate_diff",
                 "home_last5_avg_points",
                 "away_last5_avg_points",
                 "home_rest_days",
                 "away_rest_days",
                "home_last5_avg_points_allowed",
                "away_last5_avg_points_allowed",
                "average_points_diff",
                "average_points_allowed_diff",
                "rest_days_diff",
                "home_is_back_to_back",
                "away_is_back_to_back",
                "home_opponent_strength",
                "away_opponent_strength",
                "home_home_winrate", 
                "away_home_winrate",
                "home_away_winrate", 
                "away_away_winrate",
                "winrate_diff",
                "average_points_diff",
                "average_points_allowed_diff",
                "rest_days_diff",
                "home_last5_pts",
                "away_last5_pts",
                "home_last5_reb",
                "away_last5_reb",
                "home_last5_ast",
                "away_last5_ast",
                "home_last5_min",
                "away_last5_min",
                "home_last5_player_count",
                "away_last5_player_count",
                "pts_diff_last5",
                "reb_diff_last5",
                "ast_diff_last5",
                "min_diff_last5",
                "player_count_diff_last5"
                 ]

df_model = df.dropna(subset=feature_cols).copy()

# Aktuelles Datum in US Eastern Time (zeitzonenfrei)
eastern_now = pd.Timestamp.now(tz='US/Eastern')
today_naive = eastern_now.tz_localize(None).normalize()
yesterday_naive = (eastern_now - pd.Timedelta(days=1)).tz_localize(None).normalize()

train = df_model[df_model["gameDateTimeEst"] < today_naive].copy()
X_train = train[feature_cols].values
y_train = train["home_win"].values

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# 3. Zukünftige Spiele laden und auf NBA-Teams beschränken
# -----------------------------
future = pd.read_csv("data/LeagueSchedule25_26.csv")


# Spalten vereinheitlichen
future.rename(columns={
    "homeTeamName": "hometeamName",
    "awayTeamName": "awayteamName"
}, inplace=True, errors="ignore")

# Datum konvertieren und ungültige entfernen
future["gameDateTimeEst"] = pd.to_datetime(future["gameDateTimeEst"], errors="coerce")
future.dropna(subset=["gameDateTimeEst"], inplace=True)

# Mapping: Kurzname → vollständiger Teamname (NBA)
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
# Zusätzlicher Fall für "Phoenix" (falls als Teamname vorkommt)
if 'Phoenix' in future['hometeamName'].values or 'Phoenix' in future['awayteamName'].values:
    team_name_mapping['Phoenix'] = 'Phoenix Suns'

# Nur Zeilen, bei denen beide Teams im Mapping sind (echte NBA-Spiele)
nba_mask = future['hometeamName'].isin(team_name_mapping.keys()) & future['awayteamName'].isin(team_name_mapping.keys())
future_nba = future.loc[nba_mask].copy()
print(f"Anzahl NBA-Spiele in Schedule: {len(future_nba)}")

# Teamnamen ersetzen
future_nba['hometeamName'] = future_nba['hometeamName'].map(team_name_mapping)
future_nba['awayteamName'] = future_nba['awayteamName'].map(team_name_mapping)
future = future_nba

# -----------------------------
# 4. Letzte bekannte Team-Features aus Modelldaten extrahieren
# -----------------------------
home_cols = [col for col in feature_cols if col.startswith('home_')]
away_cols = [col for col in feature_cols if col.startswith('away_')]

# Letzte Werte für Heim-Teams
home_latest = (
    df_model.sort_values("gameDateTimeEst")
    .groupby("hometeamName")[home_cols]
    .last()
    .reset_index()
)

# Letzte Werte für Auswärts-Teams
away_latest = (
    df_model.sort_values("gameDateTimeEst")
    .groupby("awayteamName")[away_cols]
    .last()
    .reset_index()
)
# -----------------------------
# 5. Features mit den kommenden Spielen mergen
# -----------------------------
future = future.merge(home_latest, on="hometeamName", how="left")
future = future.merge(away_latest, on="awayteamName", how="left")

future["winrate_diff"] = future["home_last5_winrate"] - future["away_last5_winrate"]
future["average_points_diff"] = future["home_last5_avg_points"] - future["away_last5_avg_points"]
future["average_points_allowed_diff"] = future["home_last5_avg_points_allowed"] - future["away_last5_avg_points_allowed"]
future["rest_days_diff"] = future["home_rest_days"] - future["away_rest_days"]

future["pts_diff_last5"] = future["home_last5_pts"] - future["away_last5_pts"]
future["reb_diff_last5"] = future["home_last5_reb"] - future["away_last5_reb"]
future["ast_diff_last5"] = future["home_last5_ast"] - future["away_last5_ast"]
future["min_diff_last5"] = future["home_last5_min"] - future["away_last5_min"]
future["player_count_diff_last5"] = future["home_last5_player_count"] - future["away_last5_player_count"]

# Prüfen, wie viele vollständige Datensätze wir haben
complete_mask = future[feature_cols].notna().all(axis=1)
print(f"Anzahl Spiele mit vollständigen Features: {complete_mask.sum()} von {len(future)}")

# -----------------------------
# 6. Vorhersage für gültige Spiele
# -----------------------------
future_valid = future.loc[complete_mask].copy()

if future_valid.empty:
    print("WARNUNG: Keine Spiele mit vollständigen Features – Vorhersage übersprungen.")
else:
    X_future = future_valid[feature_cols].values
    future_valid["prediction"] = model.predict(X_future)
    future_valid["probability_home_win"] = model.predict_proba(X_future)[:, 1]
    future_valid["predicted_winner"] = future_valid.apply(
        lambda row: row["hometeamName"] if row["prediction"] == 1 else row["awayteamName"],
        axis=1
    )

    # -----------------------------
    # 7. Auswahl der relevanten Spiele (ab heute)
    # -----------------------------
    # Aktuelles Datum in US Eastern Time (Zeitzone der Spieldaten)
    today_us_eastern = pd.Timestamp.now(tz='US/Eastern').tz_localize(None).normalize()
    future_today = future_valid[future_valid["gameDateTimeEst"].dt.normalize() == today_naive].copy()

    if future_today.empty:
        print("Keine Spiele ab heute gefunden. Zeige stattdessen die nächsten 5 anstehenden Spiele:")
        future_today = future_valid.nsmallest(5, "gameDateTimeEst")

    future_today.sort_values("gameDateTimeEst", inplace=True)

    # Ausgabe vorbereiten
    output = future_today[[
        "gameDateTimeEst", 
        "hometeamName", 
        "awayteamName",
        "predicted_winner", 
        "probability_home_win", 
        "gameId"
    ]].copy()
    output["gameDateTimeEst"] = output["gameDateTimeEst"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output.rename(columns={
        "gameDateTimeEst": "Date",
        "hometeamName": "Home Team",
        "awayteamName": "Away Team",
        "predicted_winner": "Predicted Winner"
    }, inplace=True)
    output_today = output

yesterday_us_eastern = (pd.Timestamp.now(tz='US/Eastern') - pd.Timedelta(days=1)).tz_localize(None).normalize()
yesterday_games = df_model[df_model["gameDateTimeEst"].dt.normalize() == yesterday_naive].copy()

output_yesterday = pd.DataFrame()
if not yesterday_games.empty:
    X_yesterday = yesterday_games[feature_cols].values
    yesterday_games["prediction"] = model.predict(X_yesterday)
    yesterday_games["probability_home_win"] = model.predict_proba(X_yesterday)[:, 1]
    yesterday_games["predicted_winner"] = yesterday_games.apply(
        lambda row: row["hometeamName"] if row["prediction"] == 1 else row["awayteamName"],
        axis=1
    )
    yesterday_games["actual_winner"] = yesterday_games.apply(
        lambda row: row["hometeamName"] if row["home_win"] == 1 else row["awayteamName"],
        axis=1
    )
    output_yesterday = yesterday_games[[
        "gameDateTimeEst",
        "hometeamName", 
        "awayteamName",
        "predicted_winner", 
        "probability_home_win", 
        "actual_winner",
        "gameId"
    ]].copy()

    output_yesterday["gameDateTimeEst"] = output_yesterday["gameDateTimeEst"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_yesterday.rename(columns={
        "gameDateTimeEst": "Date",
        "hometeamName": "Home Team",
        "awayteamName": "Away Team",
        "predicted_winner": "Predicted Winner",
        "actual_winner": "Actual Winner"
    }, inplace=True)
    print(f"Gefundene Spiele für gestern: {len(output_yesterday)}")
else:
    print("Keine Spiele von gestern (US Eastern) gefunden.")

print("Vorhersagen für heute:")
print(output_today.to_string(index=False))
print("\nVorhersagen für gestern:")
print(output_yesterday.to_string(index=False))
    



    
    