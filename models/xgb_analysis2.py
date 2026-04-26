import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

# ─────────────────────────────────────────────────────────────
# PFADE
# ─────────────────────────────────────────────────────────────
base_dir   = os.path.dirname(os.path.dirname(__file__))
data_path  = os.path.join(base_dir, "data",   "model_data.csv")
feat_path  = os.path.join(base_dir, "models", "feature_cols.csv")
output_dir = os.path.join(base_dir, "output")
models_dir = os.path.join(base_dir, "models")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. DATEN LADEN
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(data_path)
df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])

# Feature-Liste aus feature_engineering.py laden (single source of truth)
if os.path.exists(feat_path):
    feature_cols = pd.read_csv(feat_path).squeeze().tolist()
    print(f"✅ Feature-Liste geladen: {len(feature_cols)} Features")
else:
    raise FileNotFoundError(
        f"feature_cols.csv nicht gefunden unter {feat_path}.\n"
        "Bitte zuerst feature_engineering.py ausführen."
    )

# Nur Spalten behalten die auch im DataFrame existieren
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
print(f"✅ Verfügbare Features im DataFrame: {len(feature_cols)}")

df_model = df.dropna(subset=feature_cols + ["home_win"]).copy()

# ─────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
#    - Test  : letzte 60 Tage
#    - Train : letzte 3 Saisons (ab Okt 2018), neuere Spiele stärker gewichtet
# ─────────────────────────────────────────────────────────────
eastern_now  = pd.Timestamp.now(tz="US/Eastern").tz_localize(None).normalize()
test_start   = (eastern_now - pd.Timedelta(days=60)).normalize()
train_cutoff = pd.Timestamp("2018-10-01")  # letzte 3 Saisons

train = df_model[
    (df_model["gameDateTimeEst"] < test_start) &
    (df_model["home_elo_games_played"] > 20) &
    (df_model["away_elo_games_played"] > 20)
].copy()
test = df_model[
    (df_model["gameDateTimeEst"] >= test_start) &
    (df_model["gameDateTimeEst"] <  eastern_now)
].copy()

X_train = train[feature_cols]
y_train = train["home_win"].values
X_test  = test[feature_cols]
y_test  = test["home_win"].values

# Sample Weights: neuere Spiele stärker gewichten
# Spiele von vor 3 Jahren bekommen ~25% Gewicht, aktuelle ~100%
days_old     = (test_start - train["gameDateTimeEst"]).dt.days
sample_weights = 1 / (1 + days_old / 365)

print(f"\n📊 Trainingsdaten : {len(train)} Spiele (ab {train_cutoff.date()})")
print(f"📊 Testdaten      : {len(test)} Spiele")
print(f"📊 Features       : {len(feature_cols)}")
print(f"📊 Gewichtung     : älteste {sample_weights.min():.2f}x → neueste {sample_weights.max():.2f}x")

# ─────────────────────────────────────────────────────────────
# 3. BASELINE – aktuelles Modell (falls gespeichert)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("🔵 BASELINE")
print("="*60)

baseline_path = os.path.join(models_dir, "best_xgb_model.pkl")
if os.path.exists(baseline_path):
    baseline = joblib.load(baseline_path)
    # Nur Features verwenden die das alte Modell kennt
    try:
        baseline_acc = baseline.score(X_test, y_test)
        print(f"Gespeichertes Modell Accuracy: {baseline_acc:.2%}")
    except Exception as e:
        print(f"Baseline konnte nicht evaluiert werden: {e}")
        baseline_acc = None
else:
    print("Kein gespeichertes Modell gefunden – überspringe Baseline.")
    baseline_acc = None

# ─────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE (schnelles XGB auf allen Trainingsdaten)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("📊 FEATURE IMPORTANCE")
print("="*60)

xgb_quick = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="logloss",
    random_state=42, verbosity=0,
)
xgb_quick.fit(X_train, y_train, sample_weight=sample_weights)
quick_acc = xgb_quick.score(X_test, y_test)
print(f"Quick XGB Test Accuracy: {quick_acc:.2%}")

feat_imp = pd.Series(xgb_quick.feature_importances_, index=feature_cols).sort_values(ascending=False)

print("\n🏆 Top 15 Features:")
for feat, imp in feat_imp.head(15).items():
    print(f"   {feat:<45} {imp:.4f}")

low_importance = feat_imp[feat_imp < 0.01].index.tolist()
print(f"\n📉 Features mit Importance < 0.01 ({len(low_importance)}):")
print(f"   {low_importance}")

# Plot
plt.figure(figsize=(12, 8))
feat_imp.head(20).sort_values().plot(kind="barh", color="steelblue")
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance (Top 20)")
plt.tight_layout()
plot_path = os.path.join(output_dir, "feature_importance.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nPlot gespeichert: {plot_path}")

# ─────────────────────────────────────────────────────────────
# 5. ENSEMBLE – XGBoost + LightGBM (soft voting)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("🤝 ENSEMBLE MODELL")
print("="*60)

best_xgb = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.5,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)
best_xgb.fit(X_train, y_train, sample_weight=sample_weights.values)
best_acc = best_xgb.score(X_test, y_test)
print(f"✅ XGB Test Accuracy: {best_acc:.2%}")

# ─────────────────────────────────────────────────────────────
# 6. FEHLERANALYSE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("🔍 FEHLERANALYSE")
print("="*60)

test_copy = test.copy()
test_copy["pred"]        = best_xgb.predict(X_test)
test_copy["correct"]     = test_copy["pred"] == y_test
test_copy["probability"] = best_xgb.predict_proba(X_test)[:, 1]

falsche  = test_copy[~test_copy["correct"]]
richtige = test_copy[test_copy["correct"]]

print(f"\n📊 Testdaten gesamt : {len(test_copy)} Spiele")
print(f"✅ Richtig          : {len(richtige)} ({len(richtige)/len(test_copy)*100:.1f}%)")
print(f"❌ Falsch           : {len(falsche)} ({len(falsche)/len(test_copy)*100:.1f}%)")

print("\n🏀 Teams mit den meisten Fehlvorhersagen:")
team_fehler = pd.concat([
    falsche.groupby("hometeamName").size().rename("als_Heim"),
    falsche.groupby("awayteamName").size().rename("als_Auswärts"),
], axis=1).fillna(0)
team_fehler["gesamt"] = team_fehler["als_Heim"] + team_fehler["als_Auswärts"]
print(team_fehler.sort_values("gesamt", ascending=False).head(10).to_string())

print("\n😲 Überraschende Fehler (hohe Siegwahrscheinlichkeit, aber falsch):")
ueberraschung = falsche.nlargest(10, "probability")[
    ["gameDateTimeEst", "hometeamName", "awayteamName", "home_win", "pred", "probability"]
]
for _, row in ueberraschung.iterrows():
    predicted_winner = row["hometeamName"] if row["pred"] == 1 else row["awayteamName"]
    actual_winner    = row["hometeamName"] if row["home_win"] == 1 else row["awayteamName"]
    print(f"   {row['gameDateTimeEst'].date()}: {row['hometeamName']} vs {row['awayteamName']}")
    print(f"      Vorhergesagt: {predicted_winner} ({row['probability']:.1%}) → Tatsächlich: {actual_winner}")

# ─────────────────────────────────────────────────────────────
# 7. ZUSAMMENFASSUNG
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("📈 ZUSAMMENFASSUNG")
print("="*60)
print(f"   Quick XGB    : {quick_acc:.2%}")
print(f"   Bestes XGB   : {best_acc:.2%}")

# ─────────────────────────────────────────────────────────────
# 8. MODELL SPEICHERN
# ─────────────────────────────────────────────────────────────
save_model = input("\n💾 Modell speichern? (j/n): ")
if save_model.lower() == "j":
    ensemble_path = os.path.join(models_dir, "best_xgb_model.pkl")
    joblib.dump(best_xgb, ensemble_path)
    print(f"✅ Modell gespeichert: {ensemble_path}")

print("\n✅ Analyse abgeschlossen!")
