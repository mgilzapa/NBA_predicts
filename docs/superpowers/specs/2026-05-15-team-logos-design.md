# Design: Team Logos in Spielkarten

**Datum:** 2026-05-15  
**Status:** Genehmigt

---

## Überblick

Auf den Spielkarten (Heute + Gestern) wird über jedem Teamnamen ein 64px-Logo angezeigt. Logos kommen vom ESPN CDN via statischem JS-Mapping. Lädt ein Logo nicht, wird es per `onerror` ausgeblendet — der Teamname bleibt sichtbar.

---

## Änderungen

### `web/app.js`

**1. `TEAM_LOGOS`-Konstante** (alle 30 NBA-Teams, Teamname → ESPN-Kürzel):

```javascript
const TEAM_LOGOS = {
  'Atlanta Hawks': 'atl',
  'Boston Celtics': 'bos',
  'Brooklyn Nets': 'bkn',
  'Charlotte Hornets': 'cha',
  'Chicago Bulls': 'chi',
  'Cleveland Cavaliers': 'cle',
  'Dallas Mavericks': 'dal',
  'Denver Nuggets': 'den',
  'Detroit Pistons': 'det',
  'Golden State Warriors': 'gs',
  'Houston Rockets': 'hou',
  'Indiana Pacers': 'ind',
  'LA Clippers': 'lac',
  'Los Angeles Lakers': 'lal',
  'Memphis Grizzlies': 'mem',
  'Miami Heat': 'mia',
  'Milwaukee Bucks': 'mil',
  'Minnesota Timberwolves': 'min',
  'New Orleans Pelicans': 'no',
  'New York Knicks': 'ny',
  'Oklahoma City Thunder': 'okc',
  'Orlando Magic': 'orl',
  'Philadelphia 76ers': 'phi',
  'Phoenix Suns': 'phx',
  'Portland Trail Blazers': 'por',
  'Sacramento Kings': 'sac',
  'San Antonio Spurs': 'sa',
  'Toronto Raptors': 'tor',
  'Utah Jazz': 'utah',
  'Washington Wizards': 'wsh',
};
```

**2. `teamLogoUrl(name)`** — gibt ESPN-Bild-URL zurück oder `null`:

```javascript
function teamLogoUrl(name) {
  const abbr = TEAM_LOGOS[name];
  return abbr ? `https://a.espncdn.com/i/teamlogos/nba/500/${abbr}.png` : null;
}
```

**3. `buildCard()`** — Logo-HTML über dem Teamnamen einfügen:

```javascript
function logoHtml(teamName) {
  const url = teamLogoUrl(teamName);
  if (!url) return '';
  return `<img class="team-logo" src="${url}" alt="${esc(teamName)}" onerror="this.style.display='none'">`;
}
```

Im Team-Block (Away + Home):
```html
<div class="team away-side">
  ${logoHtml(g.away_team)}
  <span class="${awayCls}">${esc(g.away_team)}</span>
  <span class="team-role">Away</span>
</div>
```

### `web/style.css`

```css
.team-logo {
  display: block;
  width: 64px;
  height: 64px;
  object-fit: contain;
  margin: 0 auto 6px;
  filter: drop-shadow(0 2px 4px rgba(0,0,0,0.4));
}
```

---

## Scope

- **Heute-Tab:** ✅ (nutzt `buildCard()`)
- **Gestern-Tab:** ✅ (nutzt `buildCard()`)
- **All Predictions-Tabelle:** ❌ (nutzt `buildCard()` nicht)

---

## Fehlerbehandlung

- Unbekannter Teamname → `teamLogoUrl()` gibt `null` zurück → kein `<img>` gerendert
- Bild lädt nicht (CDN offline) → `onerror="this.style.display='none'"` blendet das Element aus
- In beiden Fällen bleibt der Teamname sichtbar
