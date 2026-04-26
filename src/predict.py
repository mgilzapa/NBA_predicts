import joblib
import os

import pandas as pd
from xgboost import XGBClassifier
from injury_reports import INJURY_OUTPUT, build_player_importance_snapshot, compute_injury_features

output_today = pd.DataFrame()
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def latest_team_snapshot(frame, team_col, role_cols, rename_map=None):
    team_frame = frame[["gameDateTimeEst", team_col] + role_cols].copy()
    if rename_map:
        team_frame = team_frame.rename(columns=rename_map)
    team_frame = team_frame.rename(columns={team_col: "team_name"})
    value_cols = [col for col in team_frame.columns if col not in {"gameDateTimeEst", "team_name"}]

    rows = []
    for team_name, group in team_frame.groupby("team_name", sort=False):
        group = group.sort_values("gameDateTimeEst")
        row = {
            "team_name": team_name,
            "gameDateTimeEst": group["gameDateTimeEst"].max(),
        }
        for col in value_cols:
            non_null = group[col].dropna()
            row[col] = non_null.iloc[-1] if not non_null.empty else pd.NA
        rows.append(row)

    return pd.DataFrame(rows)

# -----------------------------
# 1. Historische Modelldaten laden und bereinigen
# -----------------------------
df = pd.read_csv("data/model_data.csv")
df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
df["winrate_diff"] = df["home_last5_winrate"] - df["away_last5_winrate"]
df["average_points_diff"] = df["home_last5_avg_points"] - df["away_last5_avg_points"]
df["average_points_allowed_diff"] = df["home_last5_avg_points_allowed"] - df["away_last5_avg_points_allowed"]
df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]

feat_path = os.path.join(BASE_DIR, "models", "feature_cols.csv")
feature_cols = pd.read_csv(feat_path).squeeze().tolist()
feature_cols = [c for c in feature_cols if c in df.columns]

exclude_cols = [
    "home_off_rating", "away_off_rating", "off_rating_diff",
    "home_def_rating", "away_def_rating", "def_rating_diff",
    "home_net_rating", "away_net_rating", "net_rating_diff",
    "home_h2h_winrate", "away_h2h_winrate", "h2h_winrate_diff",
    "same_division", "is_playoff", "away_opponent_strength",
    ]
feature_cols = [c for c in feature_cols if c not in exclude_cols]
derived_feature_dependencies = {
    "winrate_diff": ["home_last5_winrate", "away_last5_winrate"],
    "winrate_trend_diff": ["home_winrate_trend", "away_winrate_trend"],
    "average_points_diff": ["home_last5_avg_points", "away_last5_avg_points"],
    "average_points_allowed_diff": ["home_last5_avg_points_allowed", "away_last5_avg_points_allowed"],
    "rest_days_diff": ["home_rest_days", "away_rest_days"],
    "h2h_winrate_diff": ["home_h2h_winrate", "away_h2h_winrate"],
    "pts_diff_last5": ["home_last5_pts", "away_last5_pts"],
    "reb_diff_last5": ["home_last5_rebounds", "away_last5_rebounds"],
    "ast_diff_last5": ["home_last5_ast", "away_last5_ast"],
    "min_diff_last5": ["home_last5_min", "away_last5_min"],
    "player_count_diff_last5": ["home_last5_player_count", "away_last5_player_count"],
    "playoff_exp_diff": ["home_playoff_exp", "away_playoff_exp"],
    "top1_importance_diff": ["home_top1_importance", "away_top1_importance"],
    "missing_top3_importance_diff": ["home_missing_top3_importance", "away_missing_top3_importance"],
    "missing_top5_importance_diff": ["home_missing_top5_importance", "away_missing_top5_importance"],
    "top3_availability_diff": ["home_top3_availability_ratio", "away_top3_availability_ratio"],
    "top5_availability_diff": ["home_top5_availability_ratio", "away_top5_availability_ratio"],
}

support_feature_cols = feature_cols + [
    "home_winrate_trend",
    "away_winrate_trend",
    "home_rest_days",
    "away_rest_days",
    "home_top1_importance",
    "away_top1_importance",
    "home_missing_top3_importance",
    "away_missing_top3_importance",
    "home_top3_availability_ratio",
    "away_top3_availability_ratio",
    "home_top5_availability_ratio",
    "away_top5_availability_ratio",
]
for derived_col in feature_cols:
    support_feature_cols.extend(derived_feature_dependencies.get(derived_col, []))

support_feature_cols = list(dict.fromkeys([
    col for col in support_feature_cols + ["home_elo", "away_elo"]
    if col in df.columns
]))

df_model = df.dropna(subset=feature_cols).copy()

# Aktuelles Datum in US Eastern Time (zeitzonenfrei)
eastern_now = pd.Timestamp.now(tz='US/Eastern')
today_naive = eastern_now.tz_localize(None).normalize()
yesterday_naive = (eastern_now - pd.Timedelta(days=1)).tz_localize(None).normalize()
prediction_date = today_naive
#prediction_end = pd.Timestamp("2026-04-27")

model_path = os.path.join(BASE_DIR, "models", "best_xgb_model.pkl")
model = joblib.load(model_path)

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

eastern_now = pd.Timestamp.now(tz='US/Eastern')
today_naive = eastern_now.tz_localize(None).normalize()

#für alle spiele für heute

schedule_dates = pd.DatetimeIndex(future["gameDateTimeEst"].dt.normalize().dropna().unique()).sort_values()
if len(schedule_dates) > 0 and today_naive not in schedule_dates:
    future_dates = schedule_dates[schedule_dates >= today_naive]
    prediction_date = future_dates[0] if len(future_dates) > 0 else schedule_dates[-1]
    print(
        f"Hinweis: Keine Spiele fuer {today_naive.date()} im Schedule, nutze {pd.Timestamp(prediction_date).date()}."
    )
else:
    prediction_date = today_naive

future = future[future["gameDateTimeEst"].dt.normalize() == prediction_date].copy()

'''
#für genaue Date
future = future[
    (future["gameDateTimeEst"].dt.normalize() >= prediction_date)
    & (future["gameDateTimeEst"].dt.normalize() <= prediction_end)
].copy()
'''
if future.empty:
    print("Keine Spiele heute gefunden.")
    exit()


# -----------------------------
# 4. Letzte bekannte Team-Features aus Modelldaten extrahieren
# -----------------------------
home_cols = [col for col in support_feature_cols if col.startswith('home_')]
away_cols = [col for col in support_feature_cols if col.startswith('away_')]
away_to_home = {col: col.replace("away_", "home_", 1) for col in away_cols}
home_to_away = {col: col.replace("home_", "away_", 1) for col in home_cols}

# Home-Features: aus home-Spielen direkt + aus away-Spielen (umbenannt)
history_source = df[df["gameDateTimeEst"] < today_naive].copy()

home_as_home = latest_team_snapshot(history_source, "hometeamName", home_cols)
away_as_home = latest_team_snapshot(history_source, "awayteamName", away_cols, away_to_home)
all_as_home = pd.concat([home_as_home, away_as_home], ignore_index=True)
home_latest = latest_team_snapshot(all_as_home, "team_name", home_cols).rename(
    columns={"team_name": "hometeamName", "gameDateTimeEst": "home_feature_asof"}
)

# Away-Features: aus away-Spielen direkt + aus home-Spielen (umbenannt)
home_as_away = latest_team_snapshot(history_source, "hometeamName", home_cols, home_to_away)
away_as_away = latest_team_snapshot(history_source, "awayteamName", away_cols)
all_as_away = pd.concat([home_as_away, away_as_away], ignore_index=True)
away_latest = latest_team_snapshot(all_as_away, "team_name", away_cols).rename(
    columns={"team_name": "awayteamName", "gameDateTimeEst": "away_feature_asof"}
)

# -----------------------------
# 5. Features mit den kommenden Spielen mergen
# -----------------------------
future = future.sort_values("gameDateTimeEst").copy()  # kein head(8) hier!

future = future.merge(home_latest, on="hometeamName", how="left")
future = future.merge(away_latest, on="awayteamName", how="left")

home_elo_from_home = history_source[["gameDateTimeEst", "hometeamName", "home_elo"]].rename(
    columns={"hometeamName": "team", "home_elo": "elo"}
)
home_elo_from_away = history_source[["gameDateTimeEst", "awayteamName", "away_elo"]].rename(
    columns={"awayteamName": "team", "away_elo": "elo"}
)
all_elo = pd.concat([home_elo_from_home, home_elo_from_away], ignore_index=True)
latest_elo = (
    all_elo.sort_values("gameDateTimeEst")
    .groupby("team")["elo"]
    .last()
    .reset_index()
)
future["home_elo"] = future["hometeamName"].map(latest_elo.set_index("team")["elo"])
future["away_elo"] = future["awayteamName"].map(latest_elo.set_index("team")["elo"])
future["elo_diff"] = future["home_elo"] - future["away_elo"]

future["winrate_diff"] = future["home_last5_winrate"] - future["away_last5_winrate"]
if {"home_winrate_trend", "away_winrate_trend"}.issubset(future.columns):
    future["winrate_trend_diff"] = future["home_winrate_trend"] - future["away_winrate_trend"]
future["average_points_diff"] = future["home_last5_avg_points"] - future["away_last5_avg_points"]
future["average_points_allowed_diff"] = future["home_last5_avg_points_allowed"] - future["away_last5_avg_points_allowed"]
future["rest_days_diff"] = future["home_rest_days"] - future["away_rest_days"]
if {"home_h2h_winrate", "away_h2h_winrate"}.issubset(future.columns):
    future["h2h_winrate_diff"] = future["home_h2h_winrate"] - future["away_h2h_winrate"]
if {"home_last5_pts", "away_last5_pts"}.issubset(future.columns):
    future["pts_diff_last5"] = future["home_last5_pts"] - future["away_last5_pts"]
if {"home_last5_rebounds", "away_last5_rebounds"}.issubset(future.columns):
    future["reb_diff_last5"] = future["home_last5_rebounds"] - future["away_last5_rebounds"]
if {"home_last5_ast", "away_last5_ast"}.issubset(future.columns):
    future["ast_diff_last5"] = future["home_last5_ast"] - future["away_last5_ast"]
if {"home_last5_min", "away_last5_min"}.issubset(future.columns):
    future["min_diff_last5"] = future["home_last5_min"] - future["away_last5_min"]
if {"home_last5_player_count", "away_last5_player_count"}.issubset(future.columns):
    future["player_count_diff_last5"] = future["home_last5_player_count"] - future["away_last5_player_count"]
if {"home_playoff_exp", "away_playoff_exp"}.issubset(future.columns):
    future["playoff_exp_diff"] = future["home_playoff_exp"] - future["away_playoff_exp"]
if {"home_top1_importance", "away_top1_importance"}.issubset(future.columns):
    future["top1_importance_diff"] = future["home_top1_importance"] - future["away_top1_importance"]
if {"home_missing_top3_importance", "away_missing_top3_importance"}.issubset(future.columns):
    future["missing_top3_importance_diff"] = (
        future["home_missing_top3_importance"] - future["away_missing_top3_importance"]
    )
if {"home_missing_top5_importance", "away_missing_top5_importance"}.issubset(future.columns):
    future["missing_top5_importance_diff"] = (
        future["home_missing_top5_importance"] - future["away_missing_top5_importance"]
    )
if {"home_top3_availability_ratio", "away_top3_availability_ratio"}.issubset(future.columns):
    future["top3_availability_diff"] = (
        future["home_top3_availability_ratio"] - future["away_top3_availability_ratio"]
    )
if {"home_top5_availability_ratio", "away_top5_availability_ratio"}.issubset(future.columns):
    future["top5_availability_diff"] = (
        future["home_top5_availability_ratio"] - future["away_top5_availability_ratio"]
    )

divisions = {
    "Boston Celtics": "Atlantic", "Brooklyn Nets": "Atlantic", "New York Knicks": "Atlantic",
    "Philadelphia 76ers": "Atlantic", "Toronto Raptors": "Atlantic",
    "Chicago Bulls": "Central", "Cleveland Cavaliers": "Central", "Detroit Pistons": "Central",
    "Indiana Pacers": "Central", "Milwaukee Bucks": "Central",
    "Atlanta Hawks": "Southeast", "Charlotte Hornets": "Southeast", "Miami Heat": "Southeast",
    "Orlando Magic": "Southeast", "Washington Wizards": "Southeast",
    "Denver Nuggets": "Northwest", "Minnesota Timberwolves": "Northwest",
    "Oklahoma City Thunder": "Northwest", "Portland Trail Blazers": "Northwest", "Utah Jazz": "Northwest",
    "Golden State Warriors": "Pacific", "LA Clippers": "Pacific", "Los Angeles Lakers": "Pacific",
    "Phoenix Suns": "Pacific", "Sacramento Kings": "Pacific",
    "Dallas Mavericks": "Southwest", "Houston Rockets": "Southwest", "Memphis Grizzlies": "Southwest",
    "New Orleans Pelicans": "Southwest", "San Antonio Spurs": "Southwest",
}
future["same_division"] = (
    future["hometeamName"].map(divisions) == future["awayteamName"].map(divisions)
).astype(int)
future["is_playoff"] = 1

if os.path.exists(INJURY_OUTPUT):
    try:
        injuries = pd.read_csv(INJURY_OUTPUT)
        player_snapshot = build_player_importance_snapshot()
        injury_features = compute_injury_features(injuries, player_snapshot)

        if not injury_features.empty:
            injury_base_cols = [
                "home_missing_top3_importance", "away_missing_top3_importance",
                "home_missing_top5_importance", "away_missing_top5_importance",
                "home_top3_availability_ratio", "away_top3_availability_ratio",
                "home_top5_availability_ratio", "away_top5_availability_ratio",
            ]
            for col in injury_base_cols:
                if col not in future.columns:
                    future[col] = 0.0
                    
            home_live = injury_features.rename(columns={
                "team_name": "hometeamName",
                "top3_missing_count": "home_top3_missing_count_live",
                "top5_missing_count": "home_top5_missing_count_live",
                "missing_top3_importance": "home_missing_top3_importance_live",
                "missing_top5_importance": "home_missing_top5_importance_live",
                "top3_availability_ratio": "home_top3_availability_ratio_live",
                "top5_availability_ratio": "home_top5_availability_ratio_live",
            })
            away_live = injury_features.rename(columns={
                "team_name": "awayteamName",
                "top3_missing_count": "away_top3_missing_count_live",
                "top5_missing_count": "away_top5_missing_count_live",
                "missing_top3_importance": "away_missing_top3_importance_live",
                "missing_top5_importance": "away_missing_top5_importance_live",
                "top3_availability_ratio": "away_top3_availability_ratio_live",
                "top5_availability_ratio": "away_top5_availability_ratio_live",
            })

            future = future.merge(
                home_live[[
                    "hometeamName",
                    "home_top3_missing_count_live",
                    "home_top5_missing_count_live",
                    "home_missing_top3_importance_live",
                    "home_missing_top5_importance_live",
                    "home_top3_availability_ratio_live",
                    "home_top5_availability_ratio_live",
                ]],
                on="hometeamName",
                how="left",
            )
            future = future.merge(
                away_live[[
                    "awayteamName",
                    "away_top3_missing_count_live",
                    "away_top5_missing_count_live",
                    "away_missing_top3_importance_live",
                    "away_missing_top5_importance_live",
                    "away_top3_availability_ratio_live",
                    "away_top5_availability_ratio_live",
                ]],
                on="awayteamName",
                how="left",
            )

            override_pairs = [
                ("home_top3_missing_count", "home_top3_missing_count_live"),
                ("home_top5_missing_count", "home_top5_missing_count_live"),
                ("home_missing_top3_importance", "home_missing_top3_importance_live"),
                ("home_missing_top5_importance", "home_missing_top5_importance_live"),
                ("home_top3_availability_ratio", "home_top3_availability_ratio_live"),
                ("home_top5_availability_ratio", "home_top5_availability_ratio_live"),
                ("away_top3_missing_count", "away_top3_missing_count_live"),
                ("away_top5_missing_count", "away_top5_missing_count_live"),
                ("away_missing_top3_importance", "away_missing_top3_importance_live"),
                ("away_missing_top5_importance", "away_missing_top5_importance_live"),
                ("away_top3_availability_ratio", "away_top3_availability_ratio_live"),
                ("away_top5_availability_ratio", "away_top5_availability_ratio_live"),
            ]
            for base_col, live_col in override_pairs:
                if live_col in future.columns and base_col in future.columns:
                    future[base_col] = future[live_col].combine_first(future[base_col])

            future["missing_top3_importance_diff"] = (
                future["home_missing_top3_importance"] - future["away_missing_top3_importance"]
            )
            future["missing_top5_importance_diff"] = (
                future["home_missing_top5_importance"] - future["away_missing_top5_importance"]
            )
            future["top3_availability_diff"] = (
                future["home_top3_availability_ratio"] - future["away_top3_availability_ratio"]
            )
            future["top5_availability_diff"] = (
                future["home_top5_availability_ratio"] - future["away_top5_availability_ratio"]
            )
    except Exception as exc:
        print(f"WARNUNG: Injury-Reports konnten nicht eingerechnet werden: {exc}")


# Prüfen, wie viele vollständige Datensätze wir haben
for col in feature_cols:
    if col not in future.columns:
        future[col] = pd.NA

print("\nFehlende Features in future:")
for col in feature_cols:
    null_count = future[col].isna().sum() if col in future.columns else "SPALTE FEHLT"
    if null_count != 0:
        print(f"  {col}: {null_count}")

complete_mask = future[feature_cols].notna().all(axis=1)
print(f"Anzahl Spiele mit vollständigen Features: {complete_mask.sum()} von {len(future)}")

# -----------------------------
# 6. Vorhersage für die nächsten 8 gültigen Spiele
# -----------------------------
future_valid = future.loc[complete_mask].sort_values("gameDateTimeEst").copy()

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
    #spiele für heute abend
    
    future_today = future_valid[
        future_valid["gameDateTimeEst"].dt.normalize() == prediction_date
    ].sort_values("gameDateTimeEst").copy()
    '''
    #spiele bis x datum
    future_today = future_valid[
    (future_valid["gameDateTimeEst"].dt.normalize() >= prediction_date)
    & (future_valid["gameDateTimeEst"].dt.normalize() <= prediction_end)
    ].sort_values("gameDateTimeEst").copy()
'''
    if future_today.empty:
        print("Keine Spiele mit vollständigen Features heute gefunden.")

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



    



    
    
