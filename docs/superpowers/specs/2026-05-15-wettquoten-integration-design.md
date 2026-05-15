# Design: Wettquoten-Integration

**Datum:** 2026-05-15  
**Status:** Genehmigt

---

## Überblick

Wettquoten von The Odds API werden alle 5 Stunden via GitHub Actions automatisch abgerufen und in `web/odds.json` gespeichert. Die Webseite zeigt pro Spielkarte die Top 3 besten Quoten für das vorhergesagte Gewinner-Team sowie eine aufklappbare Liste aller Buchmacher. Der Nutzer kann zwischen Dezimal- und amerikanischer Quotenformat per Toggle wechseln.

---

## Datenquelle

- **API:** The Odds API — `GET /v4/sports/basketball_nba/odds`
- **Parameter:** `regions=us&markets=h2h&oddsFormat=decimal`
- **Free Tier:** 500 Requests/Monat — bei 1 Call alle 5h = ~144 Calls/Monat, ausreichend
- **API-Key:** Wird als GitHub Secret `ODDS_API_KEY` gespeichert, nie im Code

---

## Neue Dateien

### `src/fetch_odds.py`
- Ruft The Odds API auf
- Matcht jedes API-Spiel mit bestehenden Predictions über Teamnamen + Datum (normalisiert, case-insensitive)
- Sortiert Buchmacher absteigend nach Quote für das vorhergesagte Gewinner-Team
- Schreibt `web/odds.json`

### `.github/workflows/fetch-odds.yml`
- Trigger: `schedule: cron: '0 */5 * * *'` (alle 5 Stunden)
- Steps: Python setup → `pip install requests` → `python src/fetch_odds.py` → commit + push wenn `web/odds.json` geändert
- Secrets: `ODDS_API_KEY`

---

## Datenformat `web/odds.json`

```json
{
  "generated_at": "2026-05-15T14:00:00Z",
  "games": {
    "2026-05-15_LAL_GSW": {
      "home_team": "Los Angeles Lakers",
      "away_team": "Golden State Warriors",
      "commence_time": "2026-05-15T23:10:00Z",
      "bookmakers": [
        { "name": "FanDuel",    "home": 1.90, "away": 2.05 },
        { "name": "DraftKings", "home": 1.85, "away": 2.10 },
        { "name": "BetMGM",     "home": 1.75, "away": 2.20 }
      ]
    }
  }
}
```

Schlüssel-Format: `YYYY-MM-DD_HOMECODE_AWAYCODE` (3-Buchstaben-Kürzel aus Teamnamen).

---

## Frontend-Änderungen

### Toggle (Dezimal / Amerikanisch)
- Globaler State `oddsFormat` = `'decimal'` | `'american'`
- Toggle-Button im Header neben dem Meta-Timestamp
- Beim Umschalten: alle sichtbaren Quoten sofort neu rendern (kein Reload)
- Umrechnung Dezimal → Amerikanisch:
  - Quote ≥ 2.0: `+(Math.round((odds - 1) * 100))`
  - Quote < 2.0: `-(Math.round(100 / (odds - 1)))`

### Spielkarte — Odds-Sektion
Unterhalb der bestehenden ELO-Zeile, nur wenn Odds-Daten für das Spiel vorhanden:

```
── Trennlinie ──────────────────────────
Top 3 für [Predicted Winner]
  FanDuel     1.90  ★ (beste)
  DraftKings  1.85
  BetMGM      1.75

[Alle Buchmacher ▼]  (aufklappbar)
  Heim: ...  |  Auswärts: ...
```

- **Top 3:** Die 3 Buchmacher mit der höchsten Quote für den vorhergesagten Gewinner, ★-Markierung beim Besten
- **Aufklappbar:** Zeigt alle Buchmacher mit Heim- UND Auswärtsquote
- **Kein Match:** Sektion wird ausgeblendet (kein Fehler)

### `app.js`
- `loadData()` lädt `predictions.json` und `odds.json` parallel via `Promise.all`
- `buildCard(g, showResult, oddsData)` bekommt Odds als optionalen Parameter
- Matching-Funktion: findet passenden `odds.json`-Eintrag für ein Spiel über Datum + Teamnamen

### `style.css`
- `.odds-section`: Trennlinie + Padding
- `.odds-top3`: Flex-Liste der Top-3-Quoten
- `.odds-best`: Hervorhebung (Akzentfarbe + ★)
- `.odds-toggle`: Aufklapp-Button
- `.odds-all`: Versteckte, aufklappbare Liste
- `.format-toggle`: Header-Toggle-Button (Dec / Am)

---

## Fehlerbehandlung

- `odds.json` nicht ladbar → Predictions werden normal angezeigt, Odds-Sektion fehlt still
- Spiel nicht in `odds.json` → Karte ohne Odds-Sektion
- API-Key ungültig / Limit erreicht → GitHub Action schlägt fehl (Notification via GitHub), bestehende `odds.json` bleibt erhalten

---

## Was nicht gebaut wird

- Live-Quoten (Sekunden-Aktualisierung)
- Eigene Buchmacher-Auswahl durch den Nutzer
- Historische Quoten-Archivierung
