import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/model_data.csv')

df_model = df.dropna(subset=["home_last5_winrate", "away_last5_winrate"]).copy()
# Datum wieder als echtes Datum
df_model["gameDateTimeEst"] = pd.to_datetime(df_model["gameDateTimeEst"])

# Features und Ziel
X = df_model[["home_last5_winrate", "away_last5_winrate"]]
y = df_model["home_win"]

train = df_model[df_model["gameDateTimeEst"] < "2025-01-01"]
test = df_model[df_model["gameDateTimeEst"] >= "2025-01-01"]

X_train = train[["home_last5_winrate", "away_last5_winrate"]]
y_train = train["home_win"]

X_test = test[["home_last5_winrate", "away_last5_winrate"]]
y_test = test["home_win"]

model = LogisticRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(probs[:10])
print("Accuracy:", accuracy_score(y_test, preds))
print(df_model.head())
print(len(df_model))