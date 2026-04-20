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
future = pd.read_csv("data/schedule_round_1.csv")


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

# Nur echte NBA-Spiele
nba_mask = future['hometeamName'].isin(team_name_mapping.keys()) & future['awayteamName'].isin(team_name_mapping.keys())
future = future.loc[nba_mask].copy()

# Teamnamen mappen
future['hometeamName'] = future['hometeamName'].map(team_name_mapping)
future['awayteamName'] = future['awayteamName'].map(team_name_mapping)

# Nach Datum sortieren und die nächsten 8 Spiele nehmen
future = future.sort_values("gameDateTimeEst").head(8).copy()

# -----------------------------
# 4. Letzte bekannte Team-Features aus Modelldaten extrahieren
# -----------------------------
home_cols = [col for col in feature_cols if col.startswith('home_')]
away_cols = [col for col in feature_cols if col.startswith('away_')]

# Home-Features: aus home-Spielen direkt + aus away-Spielen (umbenannt)
home_as_home = df_model[["gameDateTimeEst", "hometeamName"] + home_cols].rename(
    columns={"hometeamName": "team_name"}
)
away_as_home = df_model[["gameDateTimeEst", "awayteamName"] + away_cols].rename(
    columns={"awayteamName": "team_name", **{a: h for a, h in zip(away_cols, home_cols)}}
)
all_as_home = pd.concat([home_as_home, away_as_home])
home_latest = (
    all_as_home.sort_values("gameDateTimeEst")
    .groupby("team_name")[home_cols]
    .last()
    .reset_index()
    .rename(columns={"team_name": "hometeamName"})
)

# Away-Features: aus away-Spielen direkt + aus home-Spielen (umbenannt)
home_as_away = df_model[["gameDateTimeEst", "hometeamName"] + home_cols].rename(
    columns={"hometeamName": "team_name", **{h: a for h, a in zip(home_cols, away_cols)}}
)
away_as_away = df_model[["gameDateTimeEst", "awayteamName"] + away_cols].rename(
    columns={"awayteamName": "team_name"}
)
all_as_away = pd.concat([home_as_away, away_as_away])
away_latest = (
    all_as_away.sort_values("gameDateTimeEst")
    .groupby("team_name")[away_cols]
    .last()
    .reset_index()
    .rename(columns={"team_name": "awayteamName"})
)

# -----------------------------
# 5. Features mit den kommenden Spielen mergen
# -----------------------------
future = future.sort_values("gameDateTimeEst").copy()  # kein head(8) hier!

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
# 6. Vorhersage für die nächsten 8 gültigen Spiele
# -----------------------------
future_valid = future.loc[complete_mask].sort_values("gameDateTimeEst").head(8).copy()

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
    eastern_now = pd.Timestamp.now(tz='US/Eastern')
    today_naive = eastern_now.tz_localize(None).normalize()

    # Nächste 8 anstehende Spiele ab heute (inkl. heute)
    future_today = future_valid[
        future_valid["gameDateTimeEst"].dt.normalize() >= today_naive
    ].sort_values("gameDateTimeEst").head(8).copy()

    if future_today.empty:
        print("Keine Spiele ab heute gefunden. Zeige die nächsten 5:")
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


    



    
    