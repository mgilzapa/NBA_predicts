import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/model_data.csv')

df_model = df.dropna(subset=["home_last5_winrate",
                            "away_last5_winrate",
                            "home_last5_avg_points",
                            "away_last5_avg_points",
                            "home_rest_days",
                            "away_rest_days",
                            "home_last5_avg_points_allowed",
                            "away_last5_avg_points_allowed"
                            ]).copy()

# Datum wieder als echtes Datum
df_model["gameDateTimeEst"] = pd.to_datetime(df_model["gameDateTimeEst"])
# Differenz der Winraten
df_model["winrate_diff"] = df_model["home_last5_winrate"] - df_model["away_last5_winrate"]
df_model["average_points_diff"] = df_model["home_last5_avg_points"] - df_model["away_last5_avg_points"]
df_model["average_points_allowed_diff"] = df_model["home_last5_avg_points_allowed"] - df_model["away_last5_avg_points_allowed"]
df_model["rest_days_diff"] = df_model["home_rest_days"] - df_model["away_rest_days"]

train = df_model[df_model["gameDateTimeEst"] < "2026-02-24"]
test = df_model[df_model["gameDateTimeEst"] >= "2026-02-24"]

X_train = train[["home_last5_winrate",
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
                "rest_days_diff"
                 ]]
y_train = train["home_win"]

X_test = test[["home_last5_winrate",
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
                "rest_days_diff"
               ]]
y_test = test["home_win"]

model = LogisticRegression(max_iter=1000)
'''
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
'''
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
baseline_preds = [1] * len(y_test)

#tabelle mit Vorhersagen
'''
results = test.copy()
results["prediction"] = preds
results["probability"] = probs
print(results[["gameDateTimeEst", "hometeamName", "awayteamName", "home_win", "prediction", "probability"]].tail(10))
'''

#tabelle mit Fehlvorhersagen
'''
results = test[["gameDateTimeEst", "hometeamName", "awayteamName", "home_win"]].copy()
results["prediction"] = preds
results["probability"] = probs

wrong = results[results["home_win"] != results["prediction"]]
print(wrong.head(20))
print("Fehlvorhersagen:", len(wrong))
'''

print("Tesspiele gesamt:", len(test))
print("Richtige Vorhersagen:", (preds == y_test).sum())
print("Fehlvorhersagen:", (preds != y_test).sum())
print("Baseline Accuracy:", accuracy_score(y_test, baseline_preds))
print("Accuracy:", accuracy_score(y_test, preds))
print("Diff:", accuracy_score(y_test, preds) - accuracy_score(y_test, baseline_preds))
    