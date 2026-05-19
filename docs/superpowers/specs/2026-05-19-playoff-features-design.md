# Playoff Feature-Erweiterung

**Datum:** 2026-05-19  
**Ziel:** Playoff-Predictions verbessern durch H2H-Saison-Bilanz und Series Record als Modell-Features.

## Motivation

Das Modell hat zwei blinde Flecken bei Playoff-Spielen:

1. **H2H wird ignoriert** — `home_h2h_winrate` ist in `EXCLUDE_COLS`, weil es historisch 83% null ist. Aber z.B. Spurs 5-0 gegen OKC diese Saison ist ein starkes Signal das das Modell nie sieht.
2. **Series Record fehlt komplett** — Ob ein Team in der Serie führt (z.B. Spurs 1-0 vs OKC) ist einer der stärksten Playoff-Indikatoren. Dieses Feature existiert nicht im Modell.
3. **`is_playoff` ist immer 0** — `base_games.csv` hat keine `season_type` Spalte, daher wurde `is_playoff` nie korrekt berechnet.

## Scope

Drei Kernänderungen in `feature_engineering.py` und `predict.py`, einmaliges Retraining.

---

## Feature 1: `is_playoff` reparieren

**Problem:** `base_games.csv` hat keine `season_type` Spalte. `feature_engineering.py` versucht `gameSubtype` zu lesen, was nicht existiert → `is_playoff` ist immer 0.

**Fix:** In `feature_engineering.py` nach dem Laden von `base_games.csv` einen Left-Join mit `nba_api_games.csv` auf `gameId = GAME_ID` durchführen. `is_playoff = (season_type == 'Playoffs')`.

**Konsequenz:** `is_playoff` wird aus `EXCLUDE_COLS` in `auto_retrainer.py` und `predict.py` entfernt (es ist jetzt informativ, nicht nutzlos).

---

## Feature 2: Season H2H (`home_h2h_winrate`, `away_h2h_winrate`, `h2h_winrate_diff`)

**Problem:** Aktuelle Berechnung:
- Nutzt 2 Saisonen (current + previous) → verdünnt das Signal
- NaN bei erstem Aufeinandertreffen → row wird bei `dropna` verworfen → 83% null

**Fix in `feature_engineering.py`:**
```python
# Alt: seasons >= current_season - 1 (2 Saisonen)
# Neu: pro (team, opponent, season) — innerhalb jeder Saison separat
h2h = (
    team_history
    .groupby(["team", "opponent", "season"], group_keys=False)["win"]
    .apply(lambda x: x.shift(1).expanding().mean())
)
team_history["h2h_winrate"] = h2h.fillna(0.5)  # 0.5 = neutral, kein Prior
```

**Bedeutung von 0.5:** Erstes Aufeinandertreffen in der Saison = kein Vorteil bekannt. Spurs 5-0 vs OKC = 0.0 für OKC (aus OKC-Perspektive als home).

**EXCLUDE_COLS Anpassung:** `home_h2h_winrate`, `away_h2h_winrate`, `h2h_winrate_diff` aus EXCLUDE_COLS in `auto_retrainer.py` und `predict.py` entfernen.

---

## Feature 3: Series Record (4 neue Features)

### Neue Features

| Feature | Beschreibung | Bereich |
|---|---|---|
| `home_series_wins` | Siege des Home-Teams in dieser Serie vor diesem Spiel | 0–3 |
| `away_series_wins` | Siege des Away-Teams in dieser Serie vor diesem Spiel | 0–3 |
| `series_wins_diff` | home_series_wins - away_series_wins | -3 bis +3 |
| `is_elimination_game` | 1 wenn eines der Teams bei Niederlage ausscheidet (3-x oder x-3) | 0 oder 1 |

Für Regular-Season-Spiele sind alle 4 Features = 0.

### Training: Berechnung in `feature_engineering.py`

Serie = eindeutig durch `(team_a, team_b, saison)` (Teams treffen sich max. einmal pro Saison in Playoffs).

```python
# Nur Playoff-Spiele
playoffs = df[df["is_playoff"] == 1].copy()
playoffs = playoffs.sort_values("gameDateTimeEst")

# Series-Key: frozenset der Teams + Saison
playoffs["series_key"] = playoffs.apply(
    lambda r: str(sorted([r["hometeamName"], r["awayteamName"]])) + str(r["season"]),
    axis=1
)

# Kumulierte Siege vor jedem Spiel (shift(1) für kein Leakage)
def compute_series_wins(group):
    group = group.sort_values("gameDateTimeEst")
    group["home_series_wins"] = group["home_win"].shift(1).expanding().sum().fillna(0).astype(int)
    group["away_series_wins"] = (1 - group["home_win"]).shift(1).expanding().sum().fillna(0).astype(int)
    return group

playoffs = playoffs.groupby("series_key", group_keys=False).apply(compute_series_wins)
playoffs["series_wins_diff"] = playoffs["home_series_wins"] - playoffs["away_series_wins"]
playoffs["is_elimination_game"] = (
    (playoffs["home_series_wins"] == 3) | (playoffs["away_series_wins"] == 3)
).astype(int)

# Merge zurück in df (Regular Season bekommt 0)
df = df.merge(
    playoffs[["gameId", "home_series_wins", "away_series_wins", "series_wins_diff", "is_elimination_game"]],
    on="gameId", how="left"
)
for col in ["home_series_wins", "away_series_wins", "series_wins_diff", "is_elimination_game"]:
    df[col] = df[col].fillna(0).astype(int)
```

### Prediction: Lesen aus `bracket.json`

`bracket.json` hat bereits `home_wins` / `away_wins` pro aktiver Serie (geschrieben von `fetch_bracket.py`).

In `predict.py`:
```python
import json

with open("web/bracket.json") as f:
    bracket = json.load(f)

# Series-Lookup: {frozenset([team1, team2]): (home_team, home_wins, away_wins)}
series_map = {}
for conf in ["east", "west"]:
    for rnd in bracket[conf]:
        if isinstance(bracket[conf][rnd], list):
            for series in bracket[conf][rnd]:
                key = frozenset([series["home_team"], series["away_team"]])
                series_map[key] = {
                    "home_team": series["home_team"],
                    "home_wins": series.get("home_wins", 0),
                    "away_wins": series.get("away_wins", 0),
                }

def get_series_wins(row):
    key = frozenset([row["hometeamName"], row["awayteamName"]])
    if key not in series_map:
        return 0, 0
    sr = series_map[key]
    if sr["home_team"] == row["hometeamName"]:
        return sr["home_wins"], sr["away_wins"]
    else:
        return sr["away_wins"], sr["home_wins"]

future[["home_series_wins", "away_series_wins"]] = future.apply(
    lambda r: pd.Series(get_series_wins(r)), axis=1
)
future["series_wins_diff"] = future["home_series_wins"] - future["away_series_wins"]
future["is_elimination_game"] = (
    (future["home_series_wins"] == 3) | (future["away_series_wins"] == 3)
).astype(int)
```

---

## Dateien und Änderungen

| Datei | Art der Änderung |
|---|---|
| `src/feature_engineering.py` | is_playoff fix (join nba_api_games), H2H pro Saison, Series Record Berechnung, 4 neue Features in feature_cols |
| `src/predict.py` | Series Record aus bracket.json, H2H aus exclude_cols, derived_feature_dependencies ergänzen |
| `src/agents/auto_retrainer.py` | `home_h2h_winrate`, `away_h2h_winrate`, `h2h_winrate_diff`, `is_playoff` aus EXCLUDE_COLS |
| `src/agents/feature_drift_detector.py` | Neues EXCLUDE_COLS-Set anpassen (optional) |

## Retraining

Nach dem Feature Engineering: `python src/agents/auto_retrainer.py --force`

Erwartete Auswirkung auf Test-Accuracy: gering bis moderat (+0–2%), da das Modell weiterhin überwiegend auf Regular-Season-Patterns trainiert. Der Haupteffekt ist Playoff-Korrektheit: das Modell kann jetzt auf H2H-Dominanz und Series-Momentum reagieren.

## Nicht in Scope

- Separates Playoff-Modell (eigenes Training nur auf Playoff-Daten)
- Series-Length-Prediction Verbesserungen (fetch_bracket.py)
- Live-Odds-Integration für Playoff-Series
