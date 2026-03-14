import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv('data/model_data.csv')

df_model = df.dropna(subset=["home_last5_winrate",
                            "away_last5_winrate",
                            "home_last5_avg_points",
                            "away_last5_avg_points",
                            "home_rest_days",
                            "away_rest_days",
                            "home_last5_avg_points_allowed",
                            "away_last5_avg_points_allowed",
                            "home_is_back_to_back",
                            "away_is_back_to_back"
                            ]).copy()

# Datum wieder als echtes Datum
df_model["gameDateTimeEst"] = pd.to_datetime(df_model["gameDateTimeEst"])
# Differenz der Winraten
df_model["winrate_diff"] = df_model["home_last5_winrate"] - df_model["away_last5_winrate"]
df_model["average_points_diff"] = df_model["home_last5_avg_points"] - df_model["away_last5_avg_points"]
df_model["average_points_allowed_diff"] = df_model["home_last5_avg_points_allowed"] - df_model["away_last5_avg_points_allowed"]
df_model["rest_days_diff"] = df_model["home_rest_days"] - df_model["away_rest_days"]

# Aktuelles Datum in US Eastern Time (zeitzonenfrei)
eastern_now = pd.Timestamp.now(tz='US/Eastern')
today_naive = eastern_now.tz_localize(None).normalize()
fourteen_days_ago = (eastern_now - pd.Timedelta(days=14)).tz_localize(None).normalize()

train = df_model[df_model["gameDateTimeEst"] < today_naive].copy()
test = df_model[
    (df_model["gameDateTimeEst"] >= fourteen_days_ago) & 
    (df_model["gameDateTimeEst"] < today_naive)].copy()

future_cols =["home_last5_winrate",
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
                "away_is_back_to_back"
                 ]
X_train = train[future_cols]
y_train = train["home_win"]

X_test = test[future_cols]
y_test = test["home_win"]

lr = LogisticRegression(max_iter=1000)
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

lr.fit(X_train, y_train)
xgb.fit(X_train, y_train)

xgb_preds = xgb.predict(X_test)
xgb_probs = xgb.predict_proba(X_test)[:, 1]
lr_preds = lr.predict(X_test)

# Baseline (immer Heimteam tippen)
baseline_preds = np.ones(len(y_test))  # 1 = Heimteam gewinnt
baseline_acc = accuracy_score(y_test, baseline_preds)

# Metriken berechnen
lr_acc = accuracy_score(y_test, lr_preds)
xgb_acc = accuracy_score(y_test, xgb_preds)

lr_correct = (lr_preds == y_test).sum()
xgb_correct = (xgb_preds == y_test).sum()
lr_wrong = (lr_preds != y_test).sum()
xgb_wrong = (xgb_preds != y_test).sum()

# Verbesserung gegenüber Baseline
lr_improvement = lr_acc - baseline_acc
xgb_improvement = xgb_acc - baseline_acc

# Vergleich XGBoost vs Logistic Regression
xgb_vs_lr = xgb_acc - lr_acc
xgb_vs_lr_percent = (xgb_vs_lr / lr_acc * 100) if lr_acc > 0 else 0


print("\n" + "="*60)
print(f"   • Spiele insgesamt: {len(test)}")
print(f"   • Zeitraum: {test['gameDateTimeEst'].min().date()} bis {test['gameDateTimeEst'].max().date()}")
print(f"   • Baseline (immer Heim): {baseline_acc:.2%}")
print("\n" + "-"*60)

# Logistic Regression
print(f"\n🔵 LOGISTIC REGRESSION:")
print(f"   • Richtige Vorhersagen:  {lr_correct:3d} von {len(test)} ({lr_correct/len(test)*100:5.1f}%)")
print(f"   • Fehlvorhersagen:        {lr_wrong:3d} von {len(test)} ({lr_wrong/len(test)*100:5.1f}%)")
print(f"   • Accuracy:               {lr_acc:.2%}")
print(f"   • vs. Baseline:           {lr_improvement:+.2%}")
print("\n" + "-"*60)

# XGBoost
print(f"\n🟢 XGBOOST:")
print(f"   • Richtige Vorhersagen:  {xgb_correct:3d} von {len(test)} ({xgb_correct/len(test)*100:5.1f}%)")
print(f"   • Fehlvorhersagen:        {xgb_wrong:3d} von {len(test)} ({xgb_wrong/len(test)*100:5.1f}%)")
print(f"   • Accuracy:               {xgb_acc:.2%}")
print(f"   • vs. Baseline:           {xgb_improvement:+.2%}")
print("\n" + "-"*60)

print(f"\n🏆 BESTES MODELL:")
if xgb_acc > lr_acc:
    besser = "XGBoost"
    diff = xgb_acc - lr_acc
    prozent_besser = (diff / lr_acc * 100)
    print(f"   • Gewinner:            {besser}")
    print(f"   • Accuracy:            {xgb_acc:.2%}")
    print(f"   • Vorsprung:           {diff:+.2%} absolut")
    print(f"   • Relativ besser:      +{prozent_besser:.1f}%")
    
    # Sternchen-Skala für visuellen Eindruck
    staerke = int(prozent_besser / 5)  # 5% pro Stern
    print(f"   • Überlegenheit:       {'⭐' * min(staerke, 5)}")
    
elif lr_acc > xgb_acc:
    besser = "Logistic Regression"
    diff = lr_acc - xgb_acc
    prozent_besser = (diff / xgb_acc * 100)
    print(f"   • Gewinner:            {besser}")
    print(f"   • Accuracy:            {lr_acc:.2%}")
    print(f"   • Vorsprung:           {diff:+.2%} absolut")
    print(f"   • Relativ besser:      +{prozent_besser:.1f}%")
    staerke = int(prozent_besser / 5)
    print(f"   • Überlegenheit:       {'⭐' * min(staerke, 5)}")
else:
    print(f"   • Unentschieden!       Beide Modelle gleich gut ({xgb_acc:.2%})")
print("\n" + "="*60)