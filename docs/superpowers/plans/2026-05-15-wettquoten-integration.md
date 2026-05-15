# Wettquoten-Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wettquoten von The Odds API alle 5 Stunden via GitHub Actions abrufen und pro Spielkarte die Top-3-Quoten + alle Buchmacher anzeigen, mit Dezimal/Amerikanisch-Toggle.

**Architecture:** `src/fetch_odds.py` ruft The Odds API ab und schreibt `web/odds.json`. Ein GitHub-Actions-Workflow führt das Skript alle 5 Stunden aus und commitet die Datei. Das Frontend lädt `odds.json` parallel zu `predictions.json` und rendert eine Odds-Sektion in jede Spielkarte.

**Tech Stack:** Python 3.12, `requests`, pytest, GitHub Actions, Vanilla JS, CSS `<details>/<summary>`

---

## Dateiübersicht

| Datei | Aktion | Zweck |
|---|---|---|
| `src/fetch_odds.py` | Neu | API-Abruf, Parsing, Schreiben von `web/odds.json` |
| `tests/test_fetch_odds.py` | Neu | Unit-Tests für Parsing- und Matching-Logik |
| `.github/workflows/fetch-odds.yml` | Neu | Cron-Job alle 5 Stunden |
| `web/index.html` | Ändern | Toggle-Button im Header |
| `web/style.css` | Ändern | Styles für Odds-Sektion und Toggle |
| `web/app.js` | Ändern | Odds laden, matchen, rendern, Toggle-Handler |

---

## Task 1: `src/fetch_odds.py` — Helper-Funktionen + Tests

**Files:**
- Create: `src/fetch_odds.py`
- Create: `tests/test_fetch_odds.py`

- [ ] **Step 1: Test-Datei schreiben (schlägt noch fehl)**

Erstelle `tests/test_fetch_odds.py`:

```python
import json
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import fetch_odds


def test_normalize_team():
    assert fetch_odds.normalize_team("Los Angeles Lakers") == "los angeles lakers"
    assert fetch_odds.normalize_team("  Boston Celtics  ") == "boston celtics"


def test_make_key():
    key = fetch_odds.make_key("2026-05-15", "Los Angeles Lakers", "Golden State Warriors")
    assert key == "2026-05-15|los angeles lakers|golden state warriors"


def test_parse_bookmakers_extracts_prices():
    api_game = {
        "home_team": "Los Angeles Lakers",
        "bookmakers": [
            {
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Los Angeles Lakers", "price": 1.85},
                            {"name": "Golden State Warriors", "price": 2.10},
                        ]
                    }
                ]
            }
        ]
    }
    result = fetch_odds.parse_bookmakers(api_game)
    assert len(result) == 1
    assert result[0] == {"name": "DraftKings", "home": 1.85, "away": 2.1}


def test_parse_bookmakers_skips_non_h2h():
    api_game = {
        "home_team": "Los Angeles Lakers",
        "bookmakers": [
            {
                "title": "DraftKings",
                "markets": [{"key": "spreads", "outcomes": []}]
            }
        ]
    }
    assert fetch_odds.parse_bookmakers(api_game) == []


def test_parse_bookmakers_skips_incomplete_outcomes():
    api_game = {
        "home_team": "Los Angeles Lakers",
        "bookmakers": [
            {
                "title": "BetMGM",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Los Angeles Lakers", "price": 1.75}
                        ]
                    }
                ]
            }
        ]
    }
    assert fetch_odds.parse_bookmakers(api_game) == []


def test_fetch_and_save_writes_valid_json():
    mock_api_response = [
        {
            "home_team": "Los Angeles Lakers",
            "away_team": "Golden State Warriors",
            "commence_time": "2026-05-15T23:10:00Z",
            "bookmakers": [
                {
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Los Angeles Lakers", "price": 1.90},
                                {"name": "Golden State Warriors", "price": 2.05},
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "odds.json")
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_api_response
        mock_resp.raise_for_status.return_value = None

        with patch("fetch_odds.requests.get", return_value=mock_resp):
            result = fetch_odds.fetch_and_save("test_key", output_path)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            data = json.load(f)

        expected_key = "2026-05-15|los angeles lakers|golden state warriors"
        assert expected_key in data["games"]
        game = data["games"][expected_key]
        assert game["home_team"] == "Los Angeles Lakers"
        assert game["bookmakers"][0]["name"] == "FanDuel"
        assert game["bookmakers"][0]["home"] == 1.90
        assert game["bookmakers"][0]["away"] == 2.05
        assert "generated_at" in data
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

```
cd C:\Users\Miguel\Desktop\NBA_predicts
.venv\Scripts\python -m pytest tests/test_fetch_odds.py -v
```

Erwartet: `ModuleNotFoundError: No module named 'fetch_odds'`

- [ ] **Step 3: `src/fetch_odds.py` implementieren**

Erstelle `src/fetch_odds.py`:

```python
import json
import os
import requests
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "web", "odds.json")
API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


def normalize_team(name: str) -> str:
    return name.strip().lower()


def make_key(date: str, home_team: str, away_team: str) -> str:
    return f"{date}|{normalize_team(home_team)}|{normalize_team(away_team)}"


def parse_bookmakers(api_game: dict) -> list:
    home = api_game["home_team"]
    result = []
    for bm in api_game.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market["key"] != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
            home_price = outcomes.get(home)
            away_prices = [v for k, v in outcomes.items() if k != home]
            away_price = away_prices[0] if away_prices else None
            if home_price and away_price:
                result.append({
                    "name": bm["title"],
                    "home": round(home_price, 3),
                    "away": round(away_price, 3),
                })
    return result


def fetch_and_save(api_key: str, output_path: str) -> dict:
    resp = requests.get(API_URL, params={
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }, timeout=15)
    resp.raise_for_status()

    games = {}
    for g in resp.json():
        commence = g.get("commence_time", "")
        date = commence[:10] if commence else ""
        key = make_key(date, g["home_team"], g["away_team"])
        games[key] = {
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "date": date,
            "bookmakers": parse_bookmakers(g),
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "games": games,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


if __name__ == "__main__":
    api_key = os.environ["ODDS_API_KEY"]
    result = fetch_and_save(api_key, OUTPUT_PATH)
    print(f"OK: odds.json geschrieben ({len(result['games'])} Spiele)")
```

- [ ] **Step 4: Tests laufen lassen — müssen alle bestehen**

```
.venv\Scripts\python -m pytest tests/test_fetch_odds.py -v
```

Erwartet: `5 passed`

- [ ] **Step 5: Commit**

```
git add src/fetch_odds.py tests/test_fetch_odds.py
git commit -m "feat: add fetch_odds.py with unit tests"
```

---

## Task 2: GitHub Actions Workflow

**Files:**
- Create: `.github/workflows/fetch-odds.yml`

- [ ] **Step 1: Verzeichnis anlegen und Workflow schreiben**

Erstelle `.github/workflows/fetch-odds.yml`:

```yaml
name: Fetch NBA Odds

on:
  schedule:
    - cron: '0 */5 * * *'
  workflow_dispatch:

jobs:
  fetch-odds:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install requests

      - name: Fetch odds
        env:
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
        run: python src/fetch_odds.py

      - name: Commit odds.json
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add web/odds.json
          git diff --staged --quiet || git commit -m "chore: update odds.json [skip ci]"
          git push
```

- [ ] **Step 2: GitHub Secret anlegen**

Gehe zu: `https://github.com/<dein-username>/<dein-repo>/settings/secrets/actions`

Klicke „New repository secret":
- Name: `ODDS_API_KEY`
- Value: dein API-Key von [the-odds-api.com](https://the-odds-api.com)

- [ ] **Step 3: Commit**

```
git add .github/workflows/fetch-odds.yml
git commit -m "feat: add GitHub Actions workflow for odds fetching"
```

---

## Task 3: HTML + CSS

**Files:**
- Modify: `web/index.html`
- Modify: `web/style.css`

- [ ] **Step 1: Toggle-Button in `web/index.html` einfügen**

Ersetze den `<header>`-Block:

```html
  <header>
    <div class="header-inner">
      <div class="logo">
        <svg class="logo-icon" width="20" height="20" viewBox="0 0 20 20" fill="none">
          <circle cx="10" cy="10" r="9" stroke="currentColor" stroke-width="1.5"/>
          <path d="M10 1 Q14 5 14 10 Q14 15 10 19" stroke="currentColor" stroke-width="1.5"/>
          <path d="M10 1 Q6 5 6 10 Q6 15 10 19" stroke="currentColor" stroke-width="1.5"/>
          <line x1="1" y1="10" x2="19" y2="10" stroke="currentColor" stroke-width="1.5"/>
        </svg>
        <span>NBA<strong>PREDICT</strong></span>
      </div>
      <div class="header-right">
        <div class="meta" id="meta"></div>
        <div class="fmt-toggle" id="fmt-toggle">
          <button class="fmt-btn active" data-fmt="decimal">Dec</button>
          <button class="fmt-btn" data-fmt="american">Am</button>
        </div>
      </div>
    </div>
  </header>
```

- [ ] **Step 2: CSS an das Ende von `web/style.css` anfügen**

Füge am Ende der Datei hinzu:

```css
/* ─── Header Right ───────────────────────────────── */

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

/* ─── Format Toggle ──────────────────────────────── */

.fmt-toggle {
  display: flex;
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}

.fmt-btn {
  background: none;
  border: none;
  color: var(--muted);
  cursor: pointer;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.05em;
  padding: 4px 10px;
  transition: background 0.15s, color 0.15s;
}

.fmt-btn.active {
  background: var(--accent);
  color: #fff;
}

.fmt-btn:not(.active):hover {
  background: var(--surface-hover);
  color: var(--text);
}

/* ─── Odds Section ───────────────────────────────── */

.odds-section {
  border-top: 1px solid var(--border);
  margin-top: 12px;
  padding-top: 12px;
}

.odds-label {
  color: var(--muted);
  font-size: 11px;
  letter-spacing: 0.04em;
  margin-bottom: 8px;
  text-transform: uppercase;
}

.odds-top3 {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.odds-row {
  align-items: center;
  display: flex;
  justify-content: space-between;
}

.odds-bm {
  color: var(--muted);
  font-size: 13px;
}

.odds-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
}

.odds-best .odds-bm,
.odds-best .odds-val {
  color: var(--text);
}

.odds-star {
  color: var(--accent);
  font-size: 11px;
  margin-left: 4px;
}

/* ─── Odds Details (aufklappbar) ─────────────────── */

.odds-details {
  margin-top: 10px;
}

.odds-details summary {
  color: var(--muted);
  cursor: pointer;
  font-size: 12px;
  list-style: none;
  user-select: none;
}

.odds-details summary::-webkit-details-marker { display: none; }

.odds-details summary::before {
  content: '▶ ';
  font-size: 9px;
}

.odds-details[open] summary::before {
  content: '▼ ';
}

.odds-details summary:hover {
  color: var(--text);
}

.odds-all-header,
.odds-all-row {
  align-items: center;
  display: grid;
  font-size: 12px;
  grid-template-columns: 1fr auto 12px auto;
  gap: 6px;
  margin-top: 6px;
}

.odds-all-header {
  color: var(--muted);
  font-size: 11px;
  margin-top: 8px;
}

.odds-all-row .odds-bm {
  font-size: 12px;
}

.odds-home,
.odds-away {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  text-align: right;
}

.odds-sep {
  color: var(--border);
  text-align: center;
}
```

- [ ] **Step 3: Commit**

```
git add web/index.html web/style.css
git commit -m "feat: add odds toggle button and CSS styles"
```

---

## Task 4: JavaScript — Odds laden, matchen, rendern

**Files:**
- Modify: `web/app.js`

- [ ] **Step 1: Globale State-Variablen oben in `app.js` einfügen**

Füge direkt nach `'use strict';` ein:

```javascript
let currentData = null;
let oddsData = { games: {} };
let oddsFormat = 'decimal';
```

- [ ] **Step 2: `loadData()` ersetzen — lädt beide JSONs parallel**

Ersetze die bestehende `loadData()`-Funktion:

```javascript
async function loadData() {
  try {
    const [predRes, oddsRes] = await Promise.all([
      fetch('predictions.json'),
      fetch('odds.json').catch(() => null),
    ]);

    if (!predRes.ok) throw new Error(`HTTP ${predRes.status}`);
    const data = await predRes.json();

    if (oddsRes && oddsRes.ok) {
      oddsData = await oddsRes.json();
    }

    currentData = data;
    render(data);
  } catch (err) {
    showGlobalError(err.message);
  }
}
```

- [ ] **Step 3: `render()` und `renderToday()` / `renderYesterday()` aktualisieren**

Ersetze `render()`:

```javascript
function render(data) {
  updateMeta(data.generated_at);
  renderToday(data.today || []);
  renderYesterday(data.yesterday || []);
  renderAll(data.all || []);
}
```

Ersetze in `renderToday()` die Zeile `grid.innerHTML = games.map(g => buildCard(g, false)).join('');`:

```javascript
  grid.innerHTML = games.map(g => buildCard(g, false, matchOdds(g))).join('');
```

Ersetze in `renderYesterday()` die Zeile `grid.innerHTML = games.map(g => buildCard(g, true)).join('');`:

```javascript
  grid.innerHTML = games.map(g => buildCard(g, true, matchOdds(g))).join('');
```

- [ ] **Step 4: `matchOdds()` nach `render()` einfügen**

```javascript
function matchOdds(game) {
  if (!oddsData.games || !game.date) return null;
  const key = `${game.date}|${game.home_team.toLowerCase().trim()}|${game.away_team.toLowerCase().trim()}`;
  return oddsData.games[key] || null;
}
```

- [ ] **Step 5: `decimalToAmerican()` und `formatOdds()` nach `matchOdds()` einfügen**

```javascript
function decimalToAmerican(odds) {
  if (odds >= 2.0) return '+' + Math.round((odds - 1) * 100);
  return '-' + Math.round(100 / (odds - 1));
}

function formatOdds(odds) {
  if (oddsFormat === 'american') return decimalToAmerican(odds);
  return odds.toFixed(2);
}
```

- [ ] **Step 6: `buildOddsSection()` nach `formatOdds()` einfügen**

```javascript
function buildOddsSection(oddsEntry, game) {
  if (!oddsEntry || !oddsEntry.bookmakers || oddsEntry.bookmakers.length === 0) return '';

  const predictedIsHome = game.predicted_winner === game.home_team;
  const priceKey = predictedIsHome ? 'home' : 'away';

  const sorted = [...oddsEntry.bookmakers].sort((a, b) => b[priceKey] - a[priceKey]);
  const top3 = sorted.slice(0, 3);

  const top3Html = top3.map((bm, i) => {
    const star = i === 0 ? ' <span class="odds-star">★</span>' : '';
    return `<div class="odds-row${i === 0 ? ' odds-best' : ''}">
      <span class="odds-bm">${esc(bm.name)}</span>
      <span class="odds-val">${formatOdds(bm[priceKey])}${star}</span>
    </div>`;
  }).join('');

  const allHtml = sorted.map(bm => `
    <div class="odds-all-row">
      <span class="odds-bm">${esc(bm.name)}</span>
      <span class="odds-home">${formatOdds(bm.home)}</span>
      <span class="odds-sep">·</span>
      <span class="odds-away">${formatOdds(bm.away)}</span>
    </div>`).join('');

  return `
<div class="odds-section">
  <div class="odds-label">Best odds — ${esc(game.predicted_winner)}</div>
  <div class="odds-top3">${top3Html}</div>
  <details class="odds-details">
    <summary>All bookmakers</summary>
    <div class="odds-all-header">
      <span></span>
      <span>${esc(game.home_team)}</span>
      <span>·</span>
      <span>${esc(game.away_team)}</span>
    </div>
    ${allHtml}
  </details>
</div>`;
}
```

- [ ] **Step 7: `buildCard()` aktualisieren — Odds-Parameter hinzufügen**

Ersetze die Signatur und den Rückgabe-String:

```javascript
function buildCard(g, showResult, oddsEntry = null) {
```

Füge am Ende des Template-Strings (nach `${actualInfo}` und vor dem schließenden `</div>`) ein:

```javascript
  ${buildOddsSection(oddsEntry, g)}
```

Der komplette Return-Block sieht dann so aus:

```javascript
  return `
<div class="game-card" data-home="${homeProb}" data-away="${awayProb}">
  ${badge}
  <div class="teams-row">
    <div class="team away-side">
      <span class="${awayCls}">${esc(g.away_team)}</span>
      <span class="team-role">Away</span>
    </div>
    <div class="versus">
      <span class="at-sym">@</span>
      <span class="game-time">${time}</span>
    </div>
    <div class="team home-side">
      <span class="${homeCls}">${esc(g.home_team)}</span>
      <span class="team-role">Home</span>
    </div>
  </div>
  <div class="prob-row">
    <span class="pct away-pct">${awayProb}%</span>
    <div class="prob-bar">
      <div class="bar-away"></div>
      <div class="bar-home"></div>
    </div>
    <span class="pct home-pct">${homeProb}%</span>
  </div>
  <div class="elo-row">
    <span>${esc(awayElo)}</span>
    <span>${esc(homeElo)}</span>
  </div>
  ${actualInfo}
  ${buildOddsSection(oddsEntry, g)}
</div>`;
```

- [ ] **Step 8: Toggle-Handler am Ende von `app.js` vor `loadData()` einfügen**

```javascript
document.getElementById('fmt-toggle').addEventListener('click', e => {
  const btn = e.target.closest('.fmt-btn');
  if (!btn) return;
  oddsFormat = btn.dataset.fmt;
  document.querySelectorAll('.fmt-btn').forEach(b => b.classList.toggle('active', b === btn));
  if (currentData) render(currentData);
});
```

- [ ] **Step 9: Commit**

```
git add web/app.js
git commit -m "feat: integrate odds display with toggle in game cards"
```

---

## Task 5: Verifikation

- [ ] **Step 1: Lokalen Test mit Dummy-`odds.json` durchführen**

Erstelle temporär `web/odds.json` mit echten Teamnamen aus `web/predictions.json`:

```json
{
  "generated_at": "2026-05-15T12:00:00Z",
  "games": {
    "2026-05-15|cleveland cavaliers|detroit pistons": {
      "home_team": "Cleveland Cavaliers",
      "away_team": "Detroit Pistons",
      "date": "2026-05-15",
      "bookmakers": [
        { "name": "FanDuel",    "home": 1.72, "away": 2.15 },
        { "name": "DraftKings", "home": 1.70, "away": 2.20 },
        { "name": "BetMGM",     "home": 1.68, "away": 2.25 },
        { "name": "Caesars",    "home": 1.65, "away": 2.30 }
      ]
    },
    "2026-05-15|minnesota timberwolves|san antonio spurs": {
      "home_team": "Minnesota Timberwolves",
      "away_team": "San Antonio Spurs",
      "date": "2026-05-15",
      "bookmakers": [
        { "name": "FanDuel",    "home": 2.80, "away": 1.45 },
        { "name": "DraftKings", "home": 2.75, "away": 1.48 },
        { "name": "BetMGM",     "home": 2.70, "away": 1.50 }
      ]
    }
  }
}
```

- [ ] **Step 2: Webseite im Browser öffnen**

```
start web\index.html
```

Prüfe:
- Beide Spielkarten zeigen die Odds-Sektion
- Top-3 wird mit ★ beim besten Anbieter angezeigt
- „All bookmakers" klappt per Klick auf Details auf
- Toggle „Dec" / „Am" wechselt das Format sofort
- Karten ohne Odds-Eintrag zeigen keine Odds-Sektion (teste mit leerem `games: {}`)

- [ ] **Step 3: Python-Tests nochmal laufen lassen**

```
.venv\Scripts\python -m pytest tests/test_fetch_odds.py -v
```

Erwartet: `5 passed`

- [ ] **Step 4: Finaler Commit + Push**

```
git add web/odds.json
git commit -m "test: add dummy odds.json for local testing"
git push
```

Danach auf GitHub unter Actions prüfen ob der Workflow sichtbar ist. Einmalig manuell via „Run workflow" triggern um die erste echte `odds.json` zu generieren (benötigt gesetztes Secret).
