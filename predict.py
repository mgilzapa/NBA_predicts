import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Aktuelles Datum in US Eastern Time (zeitzonenfrei)
eastern_now = pd.Timestamp.now(tz='US/Eastern')
today_naive = eastern_now.tz_localize(None).normalize()
yesterday_naive = (eastern_now - pd.Timedelta(days=1)).tz_localize(None).normalize()

# -----------------------------
# 1. Historische Modelldaten laden und bereinigen
# -----------------------------
df = pd.read_csv("data/model_data.csv")
df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])

feature_cols = [
    "home_last5_winrate",
    "away_last5_winrate",
    "home_last5_avg_points",
    "away_last5_avg_points",
    "home_last5_avg_points_allowed",
    "away_last5_avg_points_allowed"
]

df_model = df.dropna(subset=feature_cols).copy()


train = df_model[df_model["gameDateTimeEst"] < today_naive].copy()
X_train = train[feature_cols]
y_train = train["home_win"]

model = LogisticRegression(max_iter=1000)
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
# Letzte Werte für Heim-Teams
home_latest = (
    df_model.sort_values("gameDateTimeEst")
    .groupby("hometeamName")[['home_last5_winrate', 'home_last5_avg_points', 'home_last5_avg_points_allowed']]
    .last()
    .reset_index()
)

# Letzte Werte für Auswärts-Teams
away_latest = (
    df_model.sort_values("gameDateTimeEst")
    .groupby("awayteamName")[['away_last5_winrate', 'away_last5_avg_points', 'away_last5_avg_points_allowed']]
    .last()
    .reset_index()
)

# -----------------------------
# 5. Features mit den kommenden Spielen mergen
# -----------------------------
future = future.merge(home_latest, on="hometeamName", how="left")
future = future.merge(away_latest, on="awayteamName", how="left")

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
    X_future = future_valid[feature_cols]
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
        "gameDateTimeEst", "hometeamName", "awayteamName",
        "predicted_winner", "probability_home_win"
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
    X_yesterday = yesterday_games[feature_cols]
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
        "gameDateTimeEst", "hometeamName", "awayteamName",
        "predicted_winner", "probability_home_win", "actual_winner"
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
    
# -----------------------------
# 9. Excel-Datei aktualisieren (beide Sheets mit Duplikatsbereinigung)
# -----------------------------
excel_file = "data/predictions.xlsx"
sheet_today = "predictions_today"
sheet_yesterday = "yesterday"

# Bestehende Daten laden (falls vorhanden)
if os.path.exists(excel_file):
    with pd.ExcelFile(excel_file) as xls:
        sheet_names = xls.sheet_names
        existing_today = pd.read_excel(xls, sheet_today) if sheet_today in sheet_names else pd.DataFrame()
        existing_yesterday = pd.read_excel(xls, sheet_yesterday) if sheet_yesterday in sheet_names else pd.DataFrame()
else:
    existing_today = pd.DataFrame()
    existing_yesterday = pd.DataFrame()

# Heutige Spiele kombinieren und Duplikate entfernen
if not output_today.empty:
    combined_today = pd.concat([existing_today, output_today], ignore_index=True)
    combined_today.drop_duplicates(subset=["Date", "Home Team", "Away Team"], keep="last", inplace=True)
else:
    combined_today = existing_today

# Gestrige Spiele kombinieren und Duplikate entfernen
if not output_yesterday.empty:
    combined_yesterday = pd.concat([existing_yesterday, output_yesterday], ignore_index=True)
    combined_yesterday.drop_duplicates(subset=["Date", "Home Team", "Away Team"], keep="last", inplace=True)
else:
    combined_yesterday = existing_yesterday

# Beide Sheets in eine Excel-Datei schreiben
try:
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        combined_today.to_excel(writer, sheet_name=sheet_today, index=False)
        combined_yesterday.to_excel(writer, sheet_name=sheet_yesterday, index=False)
    print(f"OK: Excel-Datei erfolgreich geschrieben: {excel_file}")
    print("Info: Vorhandene Sheets:", pd.ExcelFile(excel_file).sheet_names)
except PermissionError:
    print("FEHLER: Die Datei ist möglicherweise geöffnet. Bitte schließen Sie sie und führen Sie das Skript erneut aus.")
except Exception as e:
    print(f"FEHLER beim Schreiben der Excel-Datei: {e}")


    
    