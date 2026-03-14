import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import joblib  # zum Speichern des besten Modells

# ------------------------------------------------------------
# 1. DATEN LADEN (gleicher Code wie in train.py)
# ------------------------------------------------------------
df = pd.read_csv('data/model_data.csv')

feature_cols = [
    "home_last5_winrate", "away_last5_winrate",
    "home_last5_avg_points", "away_last5_avg_points",
    "home_rest_days", "away_rest_days",
    "home_last5_avg_points_allowed", "away_last5_avg_points_allowed",
    "home_is_back_to_back", "away_is_back_to_back",
    "home_opponent_strength", "away_opponent_strength",
    "home_home_winrate", "away_home_winrate",
    "home_away_winrate", "away_away_winrate",
    "same_division",
    "home_last_game_overtime", "away_last_game_overtime"
]

# Phase-Spalten hinzufügen falls vorhanden
phase_cols = [col for col in df.columns if col.startswith("phase_")]
feature_cols.extend(phase_cols)

# Daten vorbereiten
df_model = df.dropna(subset=feature_cols).copy()
df_model["gameDateTimeEst"] = pd.to_datetime(df_model["gameDateTimeEst"])

# Train/Test Split
eastern_now = pd.Timestamp.now(tz='US/Eastern')
today_naive = eastern_now.tz_localize(None).normalize()
fourteen_days_ago = (eastern_now - pd.Timedelta(days=14)).tz_localize(None).normalize()

train = df_model[df_model["gameDateTimeEst"] < today_naive].copy()
test = df_model[
    (df_model["gameDateTimeEst"] >= fourteen_days_ago) & 
    (df_model["gameDateTimeEst"] < today_naive)
].copy()

X_train = train[feature_cols]
y_train = train["home_win"]
X_test = test[feature_cols]
y_test = test["home_win"]

print(f"\n📊 Trainingsdaten: {len(train)} Spiele")
print(f"📊 Testdaten: {len(test)} Spiele")
print(f"📊 Features: {len(feature_cols)}")
print("="*60)

# ------------------------------------------------------------
# 2. BESTEHENDES MODELL LADEN ODER TRAINIEREN
# ------------------------------------------------------------
from xgboost import XGBClassifier

# Dein aktuelles Modell (mit deinen Parametern)
xgb_current = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
xgb_current.fit(X_train, y_train)
current_acc = xgb_current.score(X_test, y_test)
print(f"\n🔵 Aktuelles Modell Accuracy: {current_acc:.2%}")

# ------------------------------------------------------------
# 3. SCHRITT 1: FEATURE IMPORTANCE ANALYSE
# ------------------------------------------------------------
print("\n" + "="*60)
print("📊 SCHRITT 1: FEATURE IMPORTANCE")
print("="*60)

importance = xgb_current.feature_importances_
feat_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\n🏆 TOP 10 WICHTIGSTE FEATURES:")
for i, row in feat_importance.head(10).iterrows():
    print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")

print("\n📉 WENIGER WICHTIGE FEATURES:")
for i, row in feat_importance.tail(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")

# Plot speichern
plt.figure(figsize=(12, 8))
plt.barh(feat_importance.head(15)['feature'], feat_importance.head(15)['importance'])
plt.xlabel('Importance')
plt.title('XGBoost Feature Importance (Top 15)')
plt.tight_layout()
plt.savefig('output/feature_importance.png', dpi=150)
print("\n📸 Plot gespeichert: output/feature_importance.png")

# ------------------------------------------------------------
# 4. SCHRITT 2: HYPERPARAMETER-TUNING
# ------------------------------------------------------------
print("\n" + "="*60)
print("⚙️  SCHRITT 2: HYPERPARAMETER-TUNING")
print("="*60)

# Kleinere Grid-Search für erste Tests (kannst du erweitern)
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.05, 0.07],
    'n_estimators': [200, 300],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
}

print("\n🔄 Teste verschiedene Parameter...")
print(f"   {len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['n_estimators']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree'])} Kombinationen")

xgb_tuned = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)
grid = GridSearchCV(xgb_tuned, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"\n✅ Beste Parameter: {grid.best_params_}")
print(f"✅ Beste CV Accuracy: {grid.best_score_:.2%}")

# Bestes Modell auf Testdaten evaluieren
best_xgb = grid.best_estimator_
tuned_acc = best_xgb.score(X_test, y_test)
print(f"✅ Test Accuracy mit besten Parametern: {tuned_acc:.2%}")
print(f"✅ Verbesserung: {tuned_acc - current_acc:+.2%}")

# ------------------------------------------------------------
# 5. SCHRITT 3: FEHLERANALYSE
# ------------------------------------------------------------
print("\n" + "="*60)
print("🔍 SCHRITT 3: FEHLERANALYSE")
print("="*60)

# Vorhersagen mit dem getunten Modell
best_preds = best_xgb.predict(X_test)
test_copy = test.copy()
test_copy['pred'] = best_preds
test_copy['correct'] = (test_copy['pred'] == y_test)
test_copy['probability'] = best_xgb.predict_proba(X_test)[:, 1]

falsche = test_copy[~test_copy['correct']]
richtige = test_copy[test_copy['correct']]

print(f"\n📊 Testdaten insgesamt: {len(test_copy)} Spiele")
print(f"✅ Richtig: {len(richtige)} ({len(richtige)/len(test_copy)*100:.1f}%)")
print(f"❌ Falsch: {len(falsche)} ({len(falsche)/len(test_copy)*100:.1f}%)")

# Teams mit den meisten Fehlern
print("\n🏀 Teams mit den meisten Fehlvorhersagen:")
team_fehler = pd.concat([
    falsche.groupby('hometeamName').size().rename('als_Heim'),
    falsche.groupby('awayteamName').size().rename('als_Auswärts')
], axis=1).fillna(0)
team_fehler['gesamt'] = team_fehler['als_Heim'] + team_fehler['als_Auswärts']
print(team_fehler.sort_values('gesamt', ascending=False).head(10))

# Überraschende Fehler (hohe Wahrscheinlichkeit, aber falsch)
print("\n😲 Überraschende Fehler (hohe Siegwahrscheinlichkeit, aber verloren):")
ueberraschung = falsche.nlargest(10, 'probability')[['gameDateTimeEst', 'hometeamName', 'awayteamName', 'home_win', 'pred', 'probability']]
for _, row in ueberraschung.iterrows():
    winner = row['hometeamName'] if row['home_win'] == 1 else row['awayteamName']
    print(f"   {row['gameDateTimeEst'].date()}: {row['hometeamName']} vs {row['awayteamName']}")
    print(f"      Vorhersage: {row['pred']} mit {row['probability']:.1%} -> Tatsächlich: {winner}")

# ------------------------------------------------------------
# 6. MODELL VERGLEICH UND SPEICHERN
# ------------------------------------------------------------
print("\n" + "="*60)
print("📈 ZUSAMMENFASSUNG")
print("="*60)

print(f"""
📊 MODELLVERGLEICH:
   • Aktuelles Modell:  {current_acc:.2%}
   • Getuntes Modell:    {tuned_acc:.2%}
   • Verbesserung:       {tuned_acc - current_acc:+.2%}

🏆 Beste Parameter:
   {grid.best_params_}

📁 Ausgaben:
   • Feature Importance: output/feature_importance.png
""")

# Optional: Bestes Modell speichern
save_model = input("\n💾 Bestes Modell speichern? (j/n): ")
if save_model.lower() == 'j':
    joblib.dump(best_xgb, 'models/best_xgb_model.pkl')
    print("✅ Modell gespeichert: models/best_xgb_model.pkl")
    
    # Feature-Liste auch speichern
    pd.Series(feature_cols).to_csv('models/feature_cols.csv', index=False)
    print("✅ Feature-Liste gespeichert: models/feature_cols.csv")

print("\n✅ Analyse abgeschlossen!")