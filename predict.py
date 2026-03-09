import pandas as pd
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 1. Historische Modelldaten laden
# -----------------------------
df = pd.read_csv("data/model_data.csv")
df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])

# nur das beste Feature-Setup benutzen
df_model = df.dropna(subset=[
    "home_last5_winrate",
    "away_last5_winrate",
    "home_last5_avg_points",
    "away_last5_avg_points",
    "home_last5_avg_points_allowed",
    "away_last5_avg_points_allowed"
]).copy()

# -----------------------------
# 2. Modell trainieren
# -----------------------------
train = df_model[df_model["gameDateTimeEst"] < "2025-01-01"].copy()

X_train = train[[
    "home_last5_winrate",
    "away_last5_winrate",
    "home_last5_avg_points",
    "away_last5_avg_points",
    "home_last5_avg_points_allowed",
    "away_last5_avg_points_allowed"
]]

y_train = train["home_win"]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# 3. Zukünftige Spiele laden
# -----------------------------
future = pd.read_csv("data/LeagueSchedule25_26.csv")
future["gameDateTimeEst"] = pd.to_datetime(future["gameDateTimeEst"])

# Spaltennamen an dein anderes Format anpassen
future = future.rename(columns={
    "homeTeamName": "hometeamName",
    "awayTeamName": "awayteamName"
})

# -----------------------------
# 4. Letzte bekannte Team-Features holen
# -----------------------------
home_latest = (
    df_model.sort_values("gameDateTimeEst")
    .groupby("hometeamName")[[
        "home_last5_winrate",
        "home_last5_avg_points",
        "home_last5_avg_points_allowed"
    ]]
    .last()
    .reset_index()
)

away_latest = (
    df_model.sort_values("gameDateTimeEst")
    .groupby("awayteamName")[[
        "away_last5_winrate",
        "away_last5_avg_points",
        "away_last5_avg_points_allowed"
    ]]
    .last()
    .reset_index()
)

# -----------------------------
# 5. Auf kommende Spiele mergen
# -----------------------------
future = future.merge(home_latest, on="hometeamName", how="left")
future = future.merge(away_latest, on="awayteamName", how="left")

# -----------------------------
# 6. Vorhersage machen
# -----------------------------
X_future = future[[
    "home_last5_winrate",
    "away_last5_winrate",
    "home_last5_avg_points",
    "away_last5_avg_points",
    "home_last5_avg_points_allowed",
    "away_last5_avg_points_allowed"
]]

valid_rows = X_future.notna().all(axis=1)

future_valid = future[valid_rows].copy()
X_future_valid = X_future[valid_rows].copy()

future_valid["prediction"] = model.predict(X_future_valid)
future_valid["probability_home_win"] = model.predict_proba(X_future_valid)[:, 1]

future_valid["predicted_winner"] = future_valid.apply(
    lambda row: row["hometeamName"] if row["prediction"] == 1 else row["awayteamName"],
    axis=1
)

tomorrow = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

future_tomorrow = future_valid[
    future_valid["gameDateTimeEst"].dt.normalize() == tomorrow
].copy()

print(future_tomorrow[[
    "gameDateTimeEst",
    "hometeamName",
    "awayteamName",
    "predicted_winner",
    "probability_home_win"
]])

