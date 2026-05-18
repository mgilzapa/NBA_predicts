# Betting Simulation Feature — Design Spec

**Date:** 2026-05-18  
**Status:** Approved  

---

## Overview

A new "Simulation" tab in the existing SPA allows users to run multiple independent bet simulations in parallel. Each simulation has its own starting capital, tracks daily bets against NBA predictions, and shows a capital-over-time chart. All state lives in browser `localStorage` — no backend changes required.

---

## Architecture

**Files changed:**
- `web/index.html` — add 6th tab button + panel + new-simulation modal
- `web/sim.js` — all simulation logic (new file, loaded after `app.js`)
- `web/style.css` — simulation-specific styles appended

**No changes to:** `app.js`, Python pipeline, or any data-producing scripts.

**Data source:** `predictions.json` and `odds.json`. `sim.js` fetches both independently on tab activation — no dependency on `app.js` internals. (`let currentData` in `app.js` is not a `window` property, so cross-script access is avoided by design.)

---

## Data Model

Stored under localStorage key `nba_sim` (version-stamped for future migrations):

```json
{
  "version": 1,
  "simulations": [
    {
      "id": "sim_1716012800000",
      "name": "Sim #1",
      "started_at": "2026-05-18",
      "starting_capital": 500,
      "capital": 612.50,
      "bets": [
        {
          "date": "2026-05-18",
          "gameId": "42500311",
          "home_team": "Oklahoma City Thunder",
          "away_team": "San Antonio Spurs",
          "predicted_winner": "Oklahoma City Thunder",
          "stake": 50,
          "odds": 1.44,
          "odds_source": "bookmaker",
          "status": "pending",
          "payout": null
        }
      ]
    }
  ]
}
```

**Bet status values:**
- `pending` — result not yet in `predictions.json`
- `won` — `correct: true` found; `payout = stake × odds`
- `lost` — `correct: false` found; `payout = 0`

**Odds source values:**
- `bookmaker` — pre-filled from `odds.json` best bookmaker for predicted winner
- `manual` — user entered a custom quote

---

## Bet Resolution Logic

On every page load of the Simulation tab, `sim.js`:

1. Loads all simulations from localStorage
2. For each simulation, iterates `pending` bets
3. Looks up the bet's `gameId` in `currentData.all`
4. If found AND `correct !== null && correct !== undefined`:
   - Sets `status` to `won` or `lost`
   - Sets `payout` to `stake × odds` (won) or `0` (lost)
   - Updates `simulation.capital` accordingly
5. Saves updated state back to localStorage
6. Re-renders the UI

Resolution is **idempotent** — running it multiple times on the same data produces the same result.

---

## UI — Overview (default view)

Grid of simulation cards. Each card shows:
- Simulation name
- Start date + starting capital
- Current capital + absolute and percentage P&L (color-coded green/red)
- Number of bets placed
- Pending bets count (if any)

One card is the "New Simulation" button (dashed border, `+` icon).

---

## UI — Detail View (after clicking a card)

Back arrow returns to overview. Layout from top to bottom:

1. **Header:** simulation name, delete button, back link
2. **Capital summary:** current capital, absolute P&L, percentage P&L
3. **Line chart (SVG, no external library):** one point per day that has at least one resolved bet or the starting day; x-axis = dates, y-axis = capital; horizontal dashed line at starting capital
4. **"Heute tippen" section:** shown only if `predictions.json → today` has games
   - One row per game: team names, predicted winner + probability, odds input (pre-filled or blank), stake input
   - Checkbox to include this game in the bet
   - Only the predicted winner is bettable (no toggle)
   - If today's bets are already placed: shown as read-only with `pending` badge
   - "Bets bestätigen" button — disabled until at least one game is checked with valid stake + odds
5. **Bet history table:** all bets newest-first; columns: date, matchup, predicted winner, stake, odds, result badge, P&L

---

## UI — New Simulation Modal

Triggered by "+" card or "Neue Simulation" button.

Fields:
- **Name** (text, optional, default `Sim #N` where N = count + 1)
- **Startkapital** (number, required, > 0)

Actions: Cancel / Erstellen

---

## Edge Cases

| Scenario | Behavior |
|---|---|
| No games today | "Heute tippen" section hidden, message: "Keine Spiele heute." |
| Bets already placed today | Section shown as read-only, no re-betting |
| Odds not in `odds.json` | Odds input pre-filled blank, `type="number"`, user must enter manually |
| Simulation capital reaches 0 | Still displayed, can continue betting (stake capped to current capital) |
| `predictions.json` not yet loaded | Resolution skipped, re-triggered when data arrives |
| Delete simulation | Confirmation dialog; removes from localStorage array |

---

## Styling

Follows existing brand: Oswald + JetBrains Mono, dark theme (`--bg`, `--surface`, `--accent` CSS vars). Cards use `var(--surface)` background with `var(--accent)` hover border. P&L green = `#22c55e`, red = `var(--wrong-color)`. Chart line = `var(--accent)`.

No external chart library — chart rendered as inline SVG using path + polyline elements.

---

## Out of Scope

- Parlay / combined bets
- Kelly Criterion auto-staking
- Cross-device sync
- Historical simulation of past days (only real-time from today forward)
