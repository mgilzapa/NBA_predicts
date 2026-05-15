# Design: NBA Playoffs Bracket Prediction

**Datum:** 2026-05-15  
**Status:** Entwurf

---

## Ziel

Ein neuer „Bracket"-Tab in der bestehenden Web-App zeigt den vollständigen NBA-Playoffs-Bracket (alle 4 Runden, beide Conferences). Abgeschlossene Serien zeigen den echten Gewinner; aktive und zukünftige Serien zeigen den aktuellen Stand plus eine Modell-Prediction (Gewinner, Wahrscheinlichkeit, erwartete Spielanzahl).

---

## Architektur

```
run_all.py (täglich)
  └── src/fetch_bracket.py   → liest NBA API + simuliert Serien
                             → schreibt web/bracket.json

web/
  ├── bracket.json           ← täglich generiert
  ├── app.js                 ← neuer Tab + Rendering
  ├── index.html             ← neuer Tab-Button
  └── style.css              ← Bracket-Styles
```

### Datenfluss

1. `fetch_bracket.py` ruft die NBA API ab (Serien-Stände aller Runden)
2. Für jede aktive/zukünftige Serie: Simulation via ML-Modell
3. Schreibt `web/bracket.json`
4. `app.js` lädt `bracket.json` parallel zu `predictions.json` und rendert den Bracket-Tab

---

## `web/bracket.json` — Struktur

```json
{
  "generated_at": "2026-05-15T09:00:00",
  "season": "2025-26",
  "east": {
    "r1": [ ...series... ],
    "r2": [ ...series... ],
    "r3": [ ...series... ],
    "finalist": null
  },
  "west": {
    "r1": [ ...series... ],
    "r2": [ ...series... ],
    "r3": [ ...series... ],
    "finalist": null
  },
  "finals": { ...series... }
}
```

### Series-Objekt

```json
{
  "home_team": "Boston Celtics",
  "away_team": "New York Knicks",
  "home_wins": 3,
  "away_wins": 2,
  "status": "active",
  "winner": null,
  "prediction": {
    "winner": "Boston Celtics",
    "win_probability": 0.68,
    "predicted_length": 6
  }
}
```

**`status`-Werte:**
- `"complete"` — Serie beendet, `winner` gesetzt, `prediction` zeigt Rückblick
- `"active"` — läuft gerade, `home_wins + away_wins >= 1`
- `"upcoming"` — Matchup bekannt, noch kein Spiel gespielt
- `"tbd"` — Matchup noch unbekannt (Vorrunde läuft noch)

---

## Python: `src/fetch_bracket.py`

### Teil 1 — Bracket-Stand via NBA API

Verwendet `nba_api.stats.endpoints.playoffbracket.PlayoffBracket` oder `SeriesLeaderboard` um für jede Serie abzurufen:
- Welche Teams spielen gegeneinander
- Aktueller Serien-Stand (Siege pro Team)
- Welche Runde / Conference

### Teil 2 — Serien-Simulation

Für jede Serie mit `status != "complete"`:

**Schritt 1 — Spielwahrscheinlichkeit bestimmen:**  
Lädt das gespeicherte XGBoost-Modell (`models/`) und die letzten Team-Features aus `data/model_data.csv`. Berechnet `p_home` (Wahrscheinlichkeit, dass das Heimteam ein einzelnes Spiel gewinnt) für beide möglichen Heimteams (Heim/Auswärts der Serie wechseln je nach Spielnummer).

NBA-Heimrecht-Rotation (Best-of-7, 2-2-1-1-1):
```
Spiel 1 → Home-Team
Spiel 2 → Home-Team
Spiel 3 → Away-Team
Spiel 4 → Away-Team
Spiel 5 → Home-Team
Spiel 6 → Away-Team
Spiel 7 → Home-Team
```

**Schritt 2 — Seriensimulation:**  
Ausgehend vom aktuellen Stand (`home_wins`, `away_wins`) werden alle verbleibenden möglichen Spielverläufe aufgezählt. Pro Pfad wird die Wahrscheinlichkeit berechnet (Produkt der Einzel-Spielwahrscheinlichkeiten). Summiert über alle Pfade:
- `P(home gewinnt Serie)` → `win_probability`
- Erwartete Gesamtspielanzahl → `predicted_length` (Erwartungswert, gerundet)

Maximale verbleibende Spiele: 7 − (home_wins + away_wins), also ≤ 7 Äste → schnell berechenbar.

**Schritt 3 — Zukünftige Runden:**  
Für Serien mit `status == "tbd"`: Nimm jeweils den `prediction.winner` beider Vorrunden-Serien als hypothetisches Matchup (also: wahrscheinlichster Gewinner trifft auf wahrscheinlichsten Gewinner der anderen Serie). Führe dann Schritt 1–2 durch. Die resultierende `win_probability` ist eine bedingte Erwartung unter dieser Annahme.

### Ausgabe

Schreibt `web/bracket.json` mit vollständiger Struktur.

---

## Frontend — Bracket-Tab

### Navigation

```
[ TODAY ]  [ YESTERDAY ]  [ ALL PREDICTIONS ]  [ BRACKET ]
```

### Layout

Klassischer NBA-Bracket mit zwei Hälften:

```
EAST                              WEST
──────────────────────────────────────────────────────────
R1  R2  Conf Finals │ FINALS │ Conf Finals  R2  R1
[A] ─┐              │        │              ┌─ [E]
     ├─ [AB] ─┐     │        │     ┌─ [EF] ─┤
[B] ─┘        │     │        │     │         └─ [F]
              ├─ [East] ─── [West] ┤
[C] ─┐        │     │        │     │         ┌─ [G]
     ├─ [CD] ─┘     │        │     └─ [GH] ─┤
[D] ─┘              │        │              └─ [H]
```

East-Seite links, West-Seite rechts, Finals in der Mitte. Verbindungslinien zwischen den Runden.

### Matchup-Card

Pro Serie eine kompakte Card:

```
┌─────────────────────────────┐
│  [Logo] BOS  3 ── 2  NYK [Logo]  │  ← aktueller Stand
│  → Boston Celtics          │
│     68% · in 6 Spielen     │
└─────────────────────────────┘
```

- **Abgeschlossen:** Gewinner fett/blau hervorgehoben, grüner „✓ Correct"-Badge wenn unsere Prediction stimmte
- **Aktiv:** Aktueller Serien-Stand + Prediction
- **Upcoming/TBD:** Teams (oder „TBD") + Prediction falls Matchup bekannt

### Farben

Bestehende CSS-Variablen werden weiterverwendet:
- Predicted Winner: `var(--accent)` (#58a6ff)
- Abgeschlossene Serien-Karte: gedämpft (`var(--muted)` für den Verlierer)
- Korrekter Tipp: `var(--correct)` (#3fb950)
- Falscher Tipp: `var(--wrong)` (#f85149)

---

## `run_all.py` — Integration

`fetch_bracket.py` wird am Ende der Pipeline aufgerufen (nach `export_json.py`). Bei einem Fehler (API nicht erreichbar, kein API-Key) wird `bracket.json` nicht überschrieben — der alte Stand bleibt erhalten.

---

## Responsiveness

- Desktop: Voller horizontaler Bracket
- Mobile (< 640px): Vertikale Liste der Serien, gruppiert nach Runde (kein horizontales Bracket, da zu schmal)

---

## Fehlerbehandlung

| Fehler | Verhalten |
|---|---|
| `bracket.json` fehlt | Tab zeigt „Bracket data not available yet" |
| API-Timeout in `fetch_bracket.py` | Skript bricht mit Fehlermeldung ab, alte `bracket.json` bleibt |
| Teamname aus API unbekannt im TEAM_LOGOS-Mapping | Logo wird ausgeblendet (wie bestehende `onerror`-Logik) |
| `status == "tbd"` | Card zeigt „TBD vs TBD" mit grauem Hintergrund |

---

## Implementierungsschritte (Reihenfolge)

1. **`src/fetch_bracket.py`** — NBA API anbinden, Serien-Stand lesen, Simulation, `bracket.json` schreiben
2. **`run_all.py`** — `fetch_bracket.py` ans Ende der Pipeline hängen
3. **`web/index.html`** — vierten Tab „Bracket" hinzufügen
4. **`web/app.js`** — `bracket.json` laden, Bracket-Rendering, Matchup-Cards
5. **`web/style.css`** — Bracket-Layout (Grid/Flexbox), Matchup-Card-Styles, Responsive
6. **Lokal testen** — `python -m http.server 8000`, alle Serien-Status-Varianten prüfen

---

## Offene Punkte / spätere Erweiterungen

- Simulation-Konfidenzintervalle anzeigen (z.B. „in 5–7 Spielen")
- Animierter Champion-Reveal wenn Finals-Prediction sich ändert
- Historischer Bracket-Verlauf (wie unsere Predictions sich über die Runden verändert haben)
