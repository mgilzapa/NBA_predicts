'use strict';

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

let currentData = null;
let oddsData = { games: {} };
let oddsFormat = 'decimal';
let activeFilters = { team: '', dateFrom: '', dateTo: '', quickDate: '', result: '', seasonType: '', confidence: '' };

// ─── Tab Switching ─────────────────────────────────────────────────

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;

    document.querySelectorAll('.tab').forEach(t => {
      t.classList.remove('active');
      t.setAttribute('aria-selected', 'false');
    });
    document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));

    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');
    const panel = document.getElementById('tab-' + target);
    panel.classList.add('active');

    updateTabIndicator(btn);
    animateBars(panel);

    if (target === 'bracket') {
      requestAnimationFrame(() => {
        document.querySelectorAll('#bracket-container .matchup-card').forEach((card, i) => {
          card.style.animation = 'none';
          void card.offsetWidth;
          card.style.animation = `cardReveal 260ms cubic-bezier(0.22,1,0.36,1) ${Math.min(i * 22, 320)}ms both`;
        });
        drawBracketConnectors();
      });
    }
  });
});

// ─── Data Loading ──────────────────────────────────────────────────

async function loadData() {
  try {
    const [predRes, oddsRes, bracketRes, dashRes, histRes] = await Promise.all([
      fetch('predictions.json', { cache: 'no-store' }),
      fetch('odds.json', { cache: 'no-store' }).catch(() => null),
      fetch('bracket.json', { cache: 'no-store' }).catch(() => null),
      fetch('dashboard.json', { cache: 'no-store' }).catch(() => null),
      fetch('projection_history.json', { cache: 'no-store' }).catch(() => null),
    ]);

    if (!predRes.ok) throw new Error(`HTTP ${predRes.status}`);
    const data = await predRes.json();

    if (oddsRes && oddsRes.ok) {
      try { oddsData = await oddsRes.json(); } catch { /* ignore */ }
    }

    let bracketData = null;
    if (bracketRes && bracketRes.ok) {
      try { bracketData = await bracketRes.json(); } catch { /* ignore */ }
    }

    let dashData = null;
    if (dashRes && dashRes.ok) {
      try { dashData = await dashRes.json(); } catch { /* ignore */ }
    }

    let histData = null;
    if (histRes && histRes.ok) {
      try { histData = await histRes.json(); } catch { /* ignore */ }
    }

    currentData = data;
    render(data);
    try {
      renderBracket(bracketData, histData);
    } catch (e) {
      // Bracket-Fehler duerfen den Rest der Seite nicht mitreissen
      const bh = document.getElementById('bracket-header');
      if (bh) bh.textContent = 'Bracket could not be rendered.';
      console.error(e);
    }
    renderDashboard(dashData);
    initFilters();
  } catch (err) {
    showGlobalError(err.message);
  }
}

function showGlobalError(msg) {
  ['today-cards', 'yesterday-cards'].forEach(id => {
    document.getElementById(id).innerHTML =
      `<div class="error-msg">Failed to load predictions.json — ${msg}</div>`;
  });
  const tbody = document.getElementById('all-tbody');
  if (tbody) {
    tbody.innerHTML =
      `<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:48px">Failed to load predictions.json — ${msg}</td></tr>`;
  }
}

// ─── Render ────────────────────────────────────────────────────────

function render(data) {
  updateMeta(data.generated_at);
  renderToday(data.today || []);
  renderYesterday(data.yesterday || []);
  renderAll(applyFilters(data.all || []));
}

function applyFilters(games) {
  return games.filter(g => {
    if (activeFilters.team && g.home_team !== activeFilters.team && g.away_team !== activeFilters.team) return false;
    if (activeFilters.dateFrom && g.date < activeFilters.dateFrom) return false;
    if (activeFilters.dateTo && g.date > activeFilters.dateTo) return false;
    if (activeFilters.result === 'correct' && g.correct !== true) return false;
    if (activeFilters.result === 'wrong' && g.correct !== false) return false;
    if (activeFilters.result === 'pending' && (g.correct === true || g.correct === false)) return false;
    if (activeFilters.seasonType) {
      const isPlayoff = String(g.gameId || '').startsWith('4');
      if (activeFilters.seasonType === 'playoffs' && !isPlayoff) return false;
      if (activeFilters.seasonType === 'regular' && isPlayoff) return false;
    }
    if (activeFilters.confidence && gameConfidenceLevel(g) !== activeFilters.confidence) return false;
    return true;
  });
}

function initFilters() {
  const allGames = () => currentData ? currentData.all || [] : [];

  // Populate team dropdown
  const teams = [...new Set((currentData.all || []).flatMap(g => [g.home_team, g.away_team]))].sort();
  const teamSelect = document.getElementById('f-team');
  teams.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t; opt.textContent = t;
    teamSelect.appendChild(opt);
  });

  function syncAndRender() {
    activeFilters.team       = document.getElementById('f-team').value;
    activeFilters.result     = document.getElementById('f-result').value;
    activeFilters.seasonType = document.getElementById('f-season').value;
    activeFilters.confidence = document.getElementById('f-confidence').value;
    const hasAny = Object.values(activeFilters).some(v => v !== '');
    document.getElementById('f-reset').style.display = hasAny ? '' : 'none';
    renderAll(applyFilters(allGames()));
  }

  document.getElementById('f-team').addEventListener('change', syncAndRender);
  document.getElementById('f-result').addEventListener('change', syncAndRender);
  document.getElementById('f-season').addEventListener('change', syncAndRender);
  document.getElementById('f-confidence').addEventListener('change', syncAndRender);

  document.getElementById('f-date-from').addEventListener('change', e => {
    activeFilters.dateFrom = e.target.value;
    activeFilters.quickDate = '';
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    syncAndRender();
  });

  document.getElementById('f-date-to').addEventListener('change', e => {
    activeFilters.dateTo = e.target.value;
    activeFilters.quickDate = '';
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    syncAndRender();
  });

  document.querySelector('.f-chips').addEventListener('click', e => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    const wasActive = chip.classList.contains('active');
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));

    if (wasActive) {
      activeFilters.quickDate = activeFilters.dateFrom = activeFilters.dateTo = '';
      document.getElementById('f-date-from').value = '';
      document.getElementById('f-date-to').value = '';
    } else {
      chip.classList.add('active');
      chip.style.animation = 'none';
      void chip.offsetWidth;
      chip.style.animation = 'chipActivate 180ms cubic-bezier(0.22,1,0.36,1)';
      activeFilters.quickDate = chip.dataset.chip;
      const today = new Date();
      const fmt = d => d.toISOString().slice(0, 10);
      if (chip.dataset.chip === 'last7') {
        const from = new Date(today); from.setDate(from.getDate() - 7);
        activeFilters.dateFrom = fmt(from); activeFilters.dateTo = '';
      } else if (chip.dataset.chip === 'last30') {
        const from = new Date(today); from.setDate(from.getDate() - 30);
        activeFilters.dateFrom = fmt(from); activeFilters.dateTo = '';
      } else if (chip.dataset.chip === 'thisMonth') {
        activeFilters.dateFrom = fmt(new Date(today.getFullYear(), today.getMonth(), 1));
        activeFilters.dateTo = '';
      }
      document.getElementById('f-date-from').value = activeFilters.dateFrom;
      document.getElementById('f-date-to').value = activeFilters.dateTo;
    }
    syncAndRender();
  });

  document.getElementById('f-reset').addEventListener('click', () => {
    activeFilters = { team: '', dateFrom: '', dateTo: '', quickDate: '', result: '', seasonType: '', confidence: '' };
    document.getElementById('f-team').value = '';
    document.getElementById('f-result').value = '';
    document.getElementById('f-season').value = '';
    document.getElementById('f-confidence').value = '';
    document.getElementById('f-date-from').value = '';
    document.getElementById('f-date-to').value = '';
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    document.getElementById('f-reset').style.display = 'none';
    renderAll(allGames());
  });
}

function updateMeta(generatedAt) {
  const el = document.getElementById('meta');
  if (!generatedAt) return;
  const dt = new Date(generatedAt);
  const fmt = dt.toLocaleString('en-US', {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
  el.textContent = 'Updated ' + fmt + ' ET';
}

// ─── Odds Matching & Formatting ────────────────────

function matchOdds(game) {
  if (!oddsData.games || !game.date || !game.home_team || !game.away_team) return null;
  const home = game.home_team.toLowerCase().trim();
  const away = game.away_team.toLowerCase().trim();
  const key = `${game.date}|${home}|${away}`;
  if (oddsData.games[key]) return oddsData.games[key];
  // Odds API uses UTC — late ET games appear as next day in UTC
  const nextDay = new Date(game.date + 'T12:00:00Z');
  nextDay.setUTCDate(nextDay.getUTCDate() + 1);
  const nextDate = nextDay.toISOString().slice(0, 10);
  return oddsData.games[`${nextDate}|${home}|${away}`] || null;
}

function decimalToAmerican(odds) {
  if (odds <= 1.0) return '—';
  if (odds >= 2.0) return '+' + Math.round((odds - 1) * 100);
  return '-' + Math.round(100 / (odds - 1));
}

function formatOdds(odds) {
  if (oddsFormat === 'american') return decimalToAmerican(odds);
  return odds.toFixed(2);
}

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

function renderToday(games) {
  const header  = document.getElementById('today-header');
  const grid    = document.getElementById('today-cards');

  if (games.length === 0) {
    header.innerHTML = '<span class="date-label">Today</span>';
    grid.innerHTML   = '<div class="empty">No games scheduled for today.</div>';
    return;
  }

  const dateLabel = games[0].date ? humanDate(games[0].date) : 'Today';
  const count     = games.length;
  header.innerHTML =
    `<span class="date-label">${dateLabel}</span>` +
    `<span class="game-count">${count} Game${count !== 1 ? 's' : ''}</span>`;

  grid.innerHTML = games.map(g => buildCard(g, false, matchOdds(g))).join('');
  grid.querySelectorAll('.game-card').forEach((card, i) => {
    card.style.animation = `cardReveal 260ms cubic-bezier(0.22,1,0.36,1) ${i * 50}ms both`;
  });
  animateBars(grid);
}

function renderYesterday(games) {
  const header = document.getElementById('yesterday-header');
  const grid   = document.getElementById('yesterday-cards');

  if (games.length === 0) {
    header.innerHTML = '<span class="date-label">Yesterday</span>';
    grid.innerHTML   = '<div class="empty">No game data for yesterday.</div>';
    return;
  }

  const dateLabel = games[0].date ? humanDate(games[0].date) : 'Yesterday';
  const count     = games.length;
  const correct   = games.filter(g => g.correct === true).length;

  header.innerHTML =
    `<span class="date-label">${dateLabel}</span>` +
    `<span class="game-count">${count} Game${count !== 1 ? 's' : ''}</span>` +
    `<span class="accuracy-pill">${correct}/${count} correct</span>`;

  grid.innerHTML = games.map(g => buildCard(g, true, matchOdds(g))).join('');
  grid.querySelectorAll('.game-card').forEach((card, i) => {
    card.style.animation = `cardReveal 260ms cubic-bezier(0.22,1,0.36,1) ${i * 50}ms both`;
  });
  animateBars(grid);
}

function gameConfidenceLevel(g) {
  const prob = g.predicted_winner === g.home_team
    ? (g.probability_home_win || 0) * 100
    : (1 - (g.probability_home_win || 0)) * 100;
  if (!prob) return '';
  if (prob >= 65) return 'high';
  if (prob >= 55) return 'medium';
  return 'low';
}

function confidenceBadge(g) {
  const level = gameConfidenceLevel(g);
  if (!level) return '';
  if (level === 'high')   return '<span class="conf-badge conf-high">High</span>';
  if (level === 'medium') return '<span class="conf-badge conf-med">Medium</span>';
  return '<span class="conf-badge conf-low">Low</span>';
}

function renderAll(games) {
  const header = document.getElementById('all-header');
  const tbody  = document.getElementById('all-tbody');

  // Cross-fade header on every update
  header.style.transition = 'none';
  header.style.opacity    = '0';

  if (games.length === 0) {
    header.textContent = '';
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:48px">No predictions yet.</td></tr>';
    requestAnimationFrame(() => requestAnimationFrame(() => {
      header.style.transition = 'opacity 180ms ease';
      header.style.opacity    = '1';
    }));
    return;
  }

  const total   = games.length;
  const decided = games.filter(g => g.correct !== null && g.correct !== undefined);
  const correct = decided.filter(g => g.correct === true).length;
  const accuracy = decided.length > 0
    ? Math.round((correct / decided.length) * 100)
    : null;

  header.innerHTML = accuracy !== null
    ? `${total} predictions · <span class="accuracy-pill">${accuracy}% accuracy</span> · ${correct}/${decided.length} decided`
    : `${total} predictions`;

  requestAnimationFrame(() => requestAnimationFrame(() => {
    header.style.transition = 'opacity 180ms ease';
    header.style.opacity    = '1';
  }));

  tbody.innerHTML = games.map((g, index) => {
    const date       = g.date ? humanDate(g.date) : '—';
    const actual     = g.actual_winner || '—';
    const resultHtml = g.correct === true  ? '<span class="result-pill correct">Correct</span>' :
                       g.correct === false ? '<span class="result-pill wrong">Wrong</span>' :
                                            '<span class="r-pending">—</span>';
    const oddsEntry  = matchOdds(g);
    const oddsCell   = bestOddsCell(g, oddsEntry);
    const animStyle  = index < 15
      ? ` style="animation:rowSlideIn 200ms cubic-bezier(0.22,1,0.36,1) ${index * 16}ms both"`
      : '';

    return `<tr${animStyle}>
      <td class="td-date">${date}</td>
      <td>${esc(g.home_team)}</td>
      <td>${esc(g.away_team)}</td>
      <td style="color:var(--accent)">${esc(g.predicted_winner)} ${confidenceBadge(g)}</td>
      <td>${esc(actual)}</td>
      <td class="td-odds">${oddsCell}</td>
      <td class="td-result">${resultHtml}</td>
    </tr>`;
  }).join('');
}

// ─── Card Builder ──────────────────────────────────────────────────

function buildCard(g, showResult, oddsEntry = null) {
  const homeProb = Math.round(g.probability_home_win * 100);
  const awayProb = 100 - homeProb;

  const homeWins = g.predicted_winner === g.home_team;
  const homeCls  = homeWins ? 'team-name winner' : 'team-name';
  const awayCls  = !homeWins ? 'team-name winner' : 'team-name';

  const time = g.time && g.time !== '00:00' ? g.time + ' ET' : '—';

  const homeElo = g.home_elo > 0 ? 'ELO ' + Math.round(g.home_elo) : '';
  const awayElo = g.away_elo > 0 ? 'ELO ' + Math.round(g.away_elo) : '';

  let badge = '';
  if (showResult && g.correct !== null && g.correct !== undefined) {
    badge = g.correct
      ? '<span class="badge correct">✓ Correct</span>'
      : '<span class="badge wrong">✗ Wrong</span>';
  }

  let actualInfo = '';
  if (showResult && g.actual_winner) {
    actualInfo = `<div class="actual-info">Actual winner: <strong>${esc(g.actual_winner)}</strong></div>`;
  }

  // Bet value (only for upcoming games that have odds)
  let betHtml = '';
  if (g.bet_value !== null && g.bet_value !== undefined) {
    const evPct  = (g.bet_value * 100).toFixed(1);
    const isPos  = g.bet_value >= 0;
    const sign   = isPos ? '+' : '';
    const oddsStr = g.best_odds
      ? ` · ${g.best_odds.toFixed(2)}${g.best_bookmaker ? ' <span class="bet-bm">' + esc(g.best_bookmaker) + '</span>' : ''}`
      : '';
    betHtml = `<div class="bet-row ${isPos ? 'bet-value' : 'bet-no-value'}">
      <span class="bet-label"><span aria-hidden="true">${isPos ? '▲' : '▼'}</span> ${isPos ? 'VALUE BET' : 'NO VALUE'}</span>
      <span class="bet-ev">${sign}${evPct}% EV${oddsStr}</span>
    </div>`;
  }

  return `
<div class="game-card" data-home="${homeProb}" data-away="${awayProb}">
  ${badge}
  <div class="teams-row">
    <div class="team away-side">
      ${logoHtml(g.away_team)}
      <span class="${awayCls}">${esc(g.away_team)}</span>
      <span class="team-role">Away</span>
    </div>
    <div class="versus">
      <span class="at-sym">@</span>
      <span class="game-time">${time}</span>
    </div>
    <div class="team home-side">
      ${logoHtml(g.home_team)}
      <span class="${homeCls}">${esc(g.home_team)}</span>
      <span class="team-role">Home</span>
    </div>
  </div>
  <div class="prob-label">Model confidence</div>
  <div class="prob-row">
    <span class="pct away-pct${awayProb > homeProb ? ' pct-winner' : ''}">${awayProb}%</span>
    <div class="prob-bar">
      <div class="bar-away"></div>
      <div class="bar-home"></div>
    </div>
    <span class="pct home-pct${homeProb >= awayProb ? ' pct-winner' : ''}">${homeProb}%</span>
  </div>
  <div class="elo-row">
    <span>${esc(awayElo)}</span>
    <span>${esc(homeElo)}</span>
  </div>
  ${betHtml}
  ${actualInfo}
  ${buildOddsSection(oddsEntry, g)}
</div>`;
}

// ─── Bar Animation ─────────────────────────────────────────────────

function animateBars(container) {
  const cards = container.querySelectorAll('.game-card');
  cards.forEach((card, i) => {
    const homePct = card.dataset.home;
    const awayPct = card.dataset.away;
    const barHome = card.querySelector('.bar-home');
    const barAway = card.querySelector('.bar-away');
    setTimeout(() => {
      if (barHome) barHome.style.width = homePct + '%';
      if (barAway) barAway.style.width = awayPct + '%';
    }, 30 + i * 55);
  });
}

// ─── Helpers ───────────────────────────────────────────────────────

function bestOddsCell(game, oddsEntry) {
  if (!oddsEntry || !oddsEntry.bookmakers || oddsEntry.bookmakers.length === 0) return '—';
  const key = game.predicted_winner === game.home_team ? 'home' : 'away';
  const best = oddsEntry.bookmakers.reduce((a, b) => b[key] > a[key] ? b : a);
  return `<span class="td-odds-val">${formatOdds(best[key])}</span> <span class="td-odds-bm">${esc(best.name)}</span>`;
}

function teamAbbr(name) {
  const abbr = TEAM_LOGOS[name];
  return abbr ? abbr.toUpperCase() : name;
}

function teamLogoUrl(name) {
  const abbr = TEAM_LOGOS[name];
  return abbr ? `https://a.espncdn.com/i/teamlogos/nba/500/${abbr}.png` : null;
}

function logoHtml(teamName, px = 64) {
  const url = teamLogoUrl(teamName);
  if (!url) return '';
  const cls = px <= 20 ? 'mc-logo' : 'team-logo';
  return `<img class="${cls}" src="${url}" alt="${esc(teamName)}" width="${px}" height="${px}" loading="lazy" decoding="async" onerror="this.style.display='none'">`;
}

function humanDate(dateStr) {
  const d = new Date(dateStr + 'T12:00:00');
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function esc(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function pct(val) {
  if (val == null || isNaN(val)) return '—';
  return Math.round(val * 100) + '%';
}

// ─── Format Toggle ─────────────────────────────────

document.getElementById('fmt-toggle').addEventListener('click', e => {
  const btn = e.target.closest('.fmt-btn');
  if (!btn) return;
  oddsFormat = btn.dataset.fmt;
  document.querySelectorAll('.fmt-btn').forEach(b => b.classList.toggle('active', b === btn));
  if (currentData) render(currentData);
});

// ─── Bracket Rendering ─────────────────────────────────────────────

function renderBracket(data, history = null) {
  const header    = document.getElementById('bracket-header');
  const container = document.getElementById('bracket-container');
  const mobile    = document.getElementById('bracket-mobile');
  const extra     = document.getElementById('bracket-extra');

  seriesTips.clear();

  if (!data) {
    header.textContent = 'Bracket data not available yet.';
    if (extra) extra.innerHTML = '';
    return;
  }

  const acc = calcBracketAccuracy(data);
  const accHtml = acc.total > 0
    ? ` · <span class="accuracy-pill">${acc.correct}/${acc.total} series correct</span>`
    : '';
  const updated = data.generated_at ? ` · Updated ${esc(data.generated_at.slice(0, 16).replace('T', ' '))} ET` : '';
  header.innerHTML = `Season <strong>${esc(data.season)}</strong>${updated}${accHtml}${projectionBadge(data)}`;

  container.classList.toggle('has-playin', hasPlayIn(data));
  const outer = document.getElementById('bracket-outer');
  if (outer) outer.classList.toggle('has-playin', hasPlayIn(data));
  container.innerHTML = buildBracketDesktop(data);
  mobile.innerHTML    = buildBracketMobile(data);
  initSeriesTooltip();

  if (extra) {
    // Fehlende Bereiche werden ausgeblendet, das Bracket bleibt wie bisher sichtbar
    try {
      extra.innerHTML = buildProjectionSections(data, history);
      initProjectionSections(extra, data, history);
    } catch (e) {
      extra.innerHTML = '';
      console.error(e);
    }
  }
}

function isProjectionMode(data) {
  return data && (data.mode === 'projected' || data.mode === 'post_season_pending');
}

function projectionBadge(data) {
  if (!data || !data.mode || !data.sims) return '';
  const sims = Number(data.sims).toLocaleString('en-US');
  const asOf = data.projection_date ? ` · as of ${esc(humanDateLong(data.projection_date))}` : '';
  if (isProjectionMode(data)) {
    return ` <span class="proj-badge"><span class="proj-badge-tag">Projection</span>${sims} simulations${asOf}</span>`;
  }
  return ` <span class="proj-badge proj-badge--live"><span class="proj-badge-tag">Live</span>round odds from ${sims} simulations${asOf}</span>`;
}

function hasPlayIn(data) {
  return !!(data && data.play_in && Array.isArray(data.play_in.east) && Array.isArray(data.play_in.west)
    && data.play_in.east.length && data.play_in.west.length);
}

function humanDateLong(dateStr) {
  const d = new Date(dateStr + 'T12:00:00');
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function fmtProb(p) {
  if (p == null || isNaN(p)) return '—';
  if (p <= 0) return '0%';
  if (p >= 1) return '100%';
  if (p < 0.005) return '<1%';
  if (p > 0.995) return '>99%';
  return Math.round(p * 100) + '%';
}

function isProjectedSeries(series) {
  if (!series || series.status === 'tbd') return false;
  if (series.status === 'projected') return true;
  return series.projected === true && series.status !== 'complete' && series.status !== 'active';
}

// Serien-Tooltips (Ergebnisverteilung): ein gemeinsames, fixiertes Element
const seriesTips = new Map();
let seriesTipSeq = 0;

function buildMatchupCard(series) {
  if (!series) return '<div class="matchup-card status-tbd"><div class="mc-teams"><div class="mc-team">TBD</div></div></div>';

  const home = series.home_team || 'TBD';
  const away = series.away_team || 'TBD';

  if (series.status === 'tbd') {
    return `<div class="matchup-card status-tbd">
      <div class="mc-teams">
        <div class="mc-team">${esc(home)}</div>
        <div class="mc-team">${esc(away)}</div>
      </div>
    </div>`;
  }

  const pred   = series.prediction;
  const winner = series.winner;

  if (isProjectedSeries(series)) return buildProjectedCard(series);

  const homeCls = winner ? (winner === home ? 'winner' : 'loser') : '';
  const awayCls = winner ? (winner === away ? 'winner' : 'loser') : '';

  let badge = '';
  if (series.status === 'complete' && pred) {
    const correct = pred.winner === winner;
    badge = `<span class="mc-badge ${correct ? 'correct' : 'wrong'}">${correct ? '✓' : '✗'}</span>`;
  }

  // Actual result line
  let resultHtml = '';
  if (series.status === 'complete' && winner) {
    const wWins = winner === home ? series.home_wins : series.away_wins;
    const lWins = winner === home ? series.away_wins : series.home_wins;
    resultHtml = `<div class="mc-result"><span class="mc-result-winner">${esc(teamAbbr(winner))}</span><span class="mc-result-text"> Wins ${wWins}–${lWins}</span></div>`;
  } else if (series.status === 'active' && (series.home_wins + series.away_wins > 0)) {
    const hw = series.home_wins, aw = series.away_wins;
    if (hw !== aw) {
      const leader = hw > aw ? home : away;
      resultHtml = `<div class="mc-result"><span class="mc-result-winner">${esc(teamAbbr(leader))}</span><span class="mc-result-text"> leads ${Math.max(hw, aw)}–${Math.min(hw, aw)}</span></div>`;
    } else {
      resultHtml = `<div class="mc-result"><span class="mc-result-text">Tied ${hw}–${aw}</span></div>`;
    }
  }

  // Prediction line
  let predHtml = '';
  if (pred) {
    predHtml = `<div class="mc-pred-row"><span class="mc-pred-label">PRED</span><span class="mc-pred-winner">${esc(teamAbbr(pred.winner))}</span><span class="mc-pred-score"> Wins 4–${pred.predicted_length - 4}</span></div>`;
  }

  const footerHtml = (resultHtml || predHtml)
    ? `<div class="mc-footer">${resultHtml}${predHtml}</div>`
    : '';

  const cardCls = `matchup-card status-${series.status}`;
  return `<div class="${cardCls}">
    ${badge}
    <div class="mc-teams">
      <div class="mc-team ${awayCls}">
        ${logoHtml(away, 16)}
        ${esc(away)}
      </div>
      <div class="mc-team ${homeCls}">
        ${logoHtml(home, 16)}
        ${esc(home)}
      </div>
    </div>
    ${footerHtml}
  </div>`;
}

// Teamname ohne Stadt ("San Antonio Spurs" -> "Spurs") fuer die schmalen Bracket-Spalten
function teamNickname(name) {
  if (!name) return '';
  if (name.endsWith('Trail Blazers')) return 'Trail Blazers';
  const parts = name.split(' ');
  return parts[parts.length - 1];
}

// Prognose-Serie: gestrichelter Rahmen, Serien-Siegchance pro Team, wahrscheinlichstes Ergebnis,
// Ergebnisverteilung im Tooltip (Hover oder Tastaturfokus)
function buildProjectedCard(series) {
  const home = series.home_team;
  const away = series.away_team;
  const pred = series.prediction || {};
  const tp   = pred.team_probabilities || {};
  const pHome = tp[home] != null ? tp[home] : (pred.winner === home ? pred.win_probability : 1 - (pred.win_probability || 0.5));
  const pAway = tp[away] != null ? tp[away] : 1 - pHome;

  const tipId = 'st' + (++seriesTipSeq);
  seriesTips.set(tipId, series);

  const row = (team, p) => {
    const isPred = pred.winner === team;
    return `<div class="mc-team${isPred ? ' pred-winner' : ''}">
        ${logoHtml(team, 16)}
        <span class="mc-team-name"><span class="tn-full">${esc(team)}</span><span class="tn-short" aria-hidden="true">${esc(teamNickname(team))}</span></span>
        <span class="mc-prob" aria-label="${esc(team)} series win chance ${fmtProb(p)}">${fmtProb(p)}</span>
      </div>`;
  };

  let predHtml = '';
  if (pred.winner) {
    const lost = (pred.predicted_length || 6) - 4;
    predHtml = `<div class="mc-footer"><div class="mc-pred-row"><span class="mc-pred-label">PROJ</span><span class="mc-pred-winner">${esc(teamAbbr(pred.winner))}</span><span class="mc-pred-score"> Wins 4–${lost}</span><span class="mc-dist-hint" aria-hidden="true">i</span></div></div>`;
  }

  const label = `Projected series: ${away} at ${home}. ${home} ${fmtProb(pHome)}, ${away} ${fmtProb(pAway)}. Focus for outcome distribution.`;
  return `<div class="matchup-card status-projected" role="group" tabindex="0" data-tip="${tipId}" aria-label="${esc(label)}">
    <div class="mc-teams">
      ${row(away, pAway)}
      ${row(home, pHome)}
    </div>
    ${predHtml}
  </div>`;
}

function buildPlayInCard(game) {
  if (!game) return '';
  const tp = game.team_probabilities || {};
  const done = game.status === 'complete';
  const row = (team, seed) => {
    const isWin = game.winner === team;
    const cls = done ? (isWin ? 'winner' : 'loser') : (isWin ? 'pred-winner' : '');
    return `<div class="mc-team ${cls}">
        <span class="pi-seed" aria-label="seed ${seed}">${seed}</span>
        ${logoHtml(team, 16)}
        <span class="mc-team-name">${esc(teamAbbr(team))}</span>
        <span class="mc-prob">${done ? (isWin ? 'W' : 'L') : fmtProb(tp[team])}</span>
      </div>`;
  };
  const statusCls = done ? 'status-complete' : 'status-projected';
  const label = `Play-in ${game.label}: ${game.away_team} at ${game.home_team}` +
    (done ? `, ${game.winner} won` : `, ${game.home_team} ${fmtProb(tp[game.home_team])}, ${game.away_team} ${fmtProb(tp[game.away_team])}`);
  return `<div class="matchup-card playin-card ${statusCls}" role="group" aria-label="${esc(label)}">
    <div class="pi-label">${esc(game.label || '')}</div>
    <div class="mc-teams">
      ${row(game.home_team, game.home_seed)}
      ${row(game.away_team, game.away_seed)}
    </div>
  </div>`;
}

function buildPlayInCol(games, conf) {
  if (!games || !games.length) return '';
  return `<div class="bracket-playin ${conf}">
    <div class="bracket-round-label">Play-In</div>
    <div class="bracket-playin-list">${games.map(buildPlayInCard).join('')}</div>
  </div>`;
}

function buildRoundCol(series_list, label) {
  const cards = series_list.map(s =>
    `<div class="bracket-matchup">${buildMatchupCard(s)}</div>`
  ).join('');
  return `<div class="bracket-round">
    <div class="bracket-round-label">${esc(label)}</div>
    ${cards}
  </div>`;
}

// NBA bracket visual order: 1v8 (idx 0), 4v5 (idx 3), 3v6 (idx 2), 2v7 (idx 1)
function bracketOrderR1(arr) {
  return [arr[0], arr[3], arr[2], arr[1]];
}

function buildFinalsWinner(series, data = null) {
  if (!series) return '';
  const champion = series.status === 'complete' ? series.winner
    : (series.prediction ? series.prediction.winner : null);
  if (!champion) return '';
  const url = teamLogoUrl(champion);
  const logoEl = url
    ? `<img class="finals-winner-logo" src="${url}" alt="${esc(champion)}" width="72" height="72" loading="lazy" decoding="async" onerror="this.style.display='none'">`
    : '';
  const projected = isProjectedSeries(series);
  const label = series.status === 'complete' ? 'Champion' : (projected ? 'Projected Champion' : 'Predicted Winner');
  let oddsHtml = '';
  if (projected && data && Array.isArray(data.round_odds)) {
    const r = data.round_odds.find(x => x.team === champion);
    if (r) oddsHtml = `<div class="finals-winner-odds">${fmtProb(r.champion)} title chance</div>`;
  }
  return `<div class="finals-winner-banner">
    ${logoEl}
    <div class="finals-winner-label">${esc(label)}</div>
    <div class="finals-winner-name">${esc(champion)}</div>
    ${oddsHtml}
  </div>`;
}

function buildBracketDesktop(data) {
  const westR1  = buildRoundCol(bracketOrderR1(data.west.r1), 'R1');
  const westR2  = buildRoundCol(data.west.r2,  'R2');
  const westR3  = buildRoundCol(data.west.r3,  'Conf Finals');
  const eastR3  = buildRoundCol(data.east.r3,  'Conf Finals');
  const eastR2  = buildRoundCol(data.east.r2,  'R2');
  const eastR1  = buildRoundCol(bracketOrderR1(data.east.r1), 'R1');

  const winnerBanner = buildFinalsWinner(data.finals, data);
  const finalsCard   = buildMatchupCard(data.finals);
  const finalsCol    = `<div class="bracket-finals-col">
    ${winnerBanner}
    <div class="bracket-finals-label">NBA Finals</div>
    ${finalsCard}
  </div>`;

  const withPlayIn = hasPlayIn(data);
  const piWest = withPlayIn ? buildPlayInCol(data.play_in.west, 'west') : '';
  const piEast = withPlayIn ? buildPlayInCol(data.play_in.east, 'east') : '';

  return `
    ${piWest}
    <div class="bracket-half west">${westR1}${westR2}${westR3}</div>
    ${finalsCol}
    <div class="bracket-half east">${eastR3}${eastR2}${eastR1}</div>
    ${piEast}
  `;
}

function buildBracketMobile(data) {
  const withPlayIn = hasPlayIn(data);
  const rounds = [
    { label: 'First Round',        east: bracketOrderR1(data.east.r1), west: bracketOrderR1(data.west.r1),
      playIn: withPlayIn ? data.play_in : null },
    { label: 'Second Round',       east: data.east.r2, west: data.west.r2 },
    { label: 'Conference Finals',  east: data.east.r3, west: data.west.r3 },
    { label: 'NBA Finals',         east: [data.finals], west: [] },
  ];

  // Play-In erscheint als eigener Block vor R1 der jeweiligen Conference
  const piBlock = (games) => games && games.length
    ? `<div class="bm-playin"><div class="bm-playin-label">Play-In</div><div class="bm-series-list">${games.map(buildPlayInCard).join('')}</div></div>`
    : '';

  return rounds.map(r => `
    <div class="bm-round-title">${esc(r.label)}</div>
    ${r.east.length ? `<div class="bm-conf-label">East</div>${r.playIn ? piBlock(r.playIn.east) : ''}<div class="bm-series-list">${r.east.map(s => buildMatchupCard(s)).join('')}</div>` : ''}
    ${r.west.length ? `<div class="bm-conf-label">West</div>${r.playIn ? piBlock(r.playIn.west) : ''}<div class="bm-series-list">${r.west.map(s => buildMatchupCard(s)).join('')}</div>` : ''}
  `).join('');
}

// ─── Series Tooltip (outcome distribution) ─────────────────────────

function outcomeLabel(key, home, away) {
  const [h, a] = key.split('-').map(Number);
  return h === 4 ? `${teamAbbr(home)} 4–${a}` : `${teamAbbr(away)} 4–${h}`;
}

function buildSeriesTipHtml(series) {
  const pred = series.prediction || {};
  const tp = pred.team_probabilities || {};
  const home = series.home_team, away = series.away_team;
  const dist = Object.entries(pred.outcome_distribution || {}).sort((a, b) => b[1] - a[1]);
  const max = dist.length ? dist[0][1] : 1;
  const rows = dist.map(([k, v]) => `
    <div class="tip-row">
      <span class="tip-outcome">${esc(outcomeLabel(k, home, away))}</span>
      <span class="tip-bar" aria-hidden="true"><span style="width:${Math.round((v / max) * 100)}%"></span></span>
      <span class="tip-val">${fmtProb(v)}</span>
    </div>`).join('');
  const src = pred.source === 'elo_analytic'
    ? 'ELO estimate (few matching simulations)'
    : (pred.sims_matched ? `${Number(pred.sims_matched).toLocaleString('en-US')} simulations with this matchup` : '');
  return `
    <div class="tip-title">${esc(teamAbbr(away))} @ ${esc(teamAbbr(home))}</div>
    <div class="tip-teams">
      <span>${esc(teamAbbr(home))} <strong>${fmtProb(tp[home])}</strong></span>
      <span>${esc(teamAbbr(away))} <strong>${fmtProb(tp[away])}</strong></span>
    </div>
    <div class="tip-sub">Series outcome</div>
    ${rows}
    ${src ? `<div class="tip-foot">${esc(src)}</div>` : ''}`;
}

function initSeriesTooltip() {
  let tip = document.getElementById('series-tip');
  if (tip) return;
  tip = document.createElement('div');
  tip.id = 'series-tip';
  tip.className = 'series-tip';
  tip.setAttribute('role', 'tooltip');
  tip.hidden = true;
  document.body.appendChild(tip);

  let current = null;

  function show(card) {
    const series = seriesTips.get(card.dataset.tip);
    if (!series) return;
    current = card;
    tip.innerHTML = buildSeriesTipHtml(series);
    tip.hidden = false;
    card.setAttribute('aria-describedby', 'series-tip');
    const r = card.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = r.left + r.width / 2 - tw / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - tw - 8));
    let top = r.bottom + 8;
    if (top + th > window.innerHeight - 8) top = Math.max(8, r.top - th - 8);
    tip.style.left = left + 'px';
    tip.style.top  = top + 'px';
  }

  function hide(card) {
    if (card && card !== current) return;
    if (current) current.removeAttribute('aria-describedby');
    current = null;
    tip.hidden = true;
  }

  document.addEventListener('mouseover', e => {
    const card = e.target.closest && e.target.closest('[data-tip]');
    if (card && card !== current) show(card);
  });
  document.addEventListener('mouseout', e => {
    const card = e.target.closest && e.target.closest('[data-tip]');
    if (card && !card.contains(e.relatedTarget)) hide(card);
  });
  document.addEventListener('focusin', e => {
    const card = e.target.closest && e.target.closest('[data-tip]');
    if (card) show(card);
  });
  document.addEventListener('focusout', e => {
    const card = e.target.closest && e.target.closest('[data-tip]');
    if (card) hide(card);
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') hide(); });
  window.addEventListener('scroll', () => hide(), { passive: true });
  document.addEventListener('scroll', () => hide(), { passive: true, capture: true });
}

// ─── Projection Sections (standings, round odds, history) ──────────

const standingsView = { east: 'projected', west: 'projected' };   // Standard: Prognose
let roundOddsSort = { key: 'champion', dir: -1 };
let historyMetric = 'playoffs';

function buildProjectionSections(data, history) {
  const parts = [];
  if (data.standings && data.standings.east && data.standings.west) {
    parts.push(`
      <section class="proj-section" aria-labelledby="standings-title">
        <h2 class="proj-title" id="standings-title">Standings</h2>
        <p class="proj-note">${isProjectionMode(data)
          ? 'Projected final table from the Monte Carlo simulation. Seeds 1–6 go to the playoffs, 7–10 to the play-in.'
          : 'Final regular season table.'}</p>
        <div class="standings-grid">
          <div class="standings-card" data-conf="west">${buildStandingsCard(data, 'west')}</div>
          <div class="standings-card" data-conf="east">${buildStandingsCard(data, 'east')}</div>
        </div>
      </section>`);
  }
  if (Array.isArray(data.round_odds) && data.round_odds.length) {
    parts.push(`
      <section class="proj-section" aria-labelledby="odds-title">
        <h2 class="proj-title" id="odds-title">Round Odds</h2>
        <p class="proj-note">Chance of each team to reach each round, from ${Number(data.sims || 0).toLocaleString('en-US')} simulations. Click a column to sort.</p>
        <div class="table-scroll" id="round-odds-wrap">${buildRoundOddsTable(data.round_odds)}</div>
      </section>`);
  }
  if (Array.isArray(history)) {
    parts.push(`
      <section class="proj-section" aria-labelledby="history-title">
        <div class="proj-title-row">
          <h2 class="proj-title" id="history-title">Odds History</h2>
          <label class="hist-metric-label">
            <span class="visually-hidden">Metric</span>
            <select class="f-select" id="hist-metric">
              <option value="playoffs"${historyMetric === 'playoffs' ? ' selected' : ''}>% Playoffs</option>
              <option value="conf_finals"${historyMetric === 'conf_finals' ? ' selected' : ''}>% Conf Finals</option>
              <option value="champion"${historyMetric === 'champion' ? ' selected' : ''}>% Title</option>
            </select>
          </label>
        </div>
        <div class="hist-wrap" id="hist-wrap">${buildHistoryChart(history, historyMetric)}</div>
      </section>`);
  }
  return parts.join('');
}

function buildStandingsCard(data, conf) {
  const view = standingsView[conf];
  const confLabel = conf === 'east' ? 'East' : 'West';
  const btn = (v, label) =>
    `<button type="button" class="seg-btn${view === v ? ' active' : ''}" data-view="${v}" aria-pressed="${view === v}">${label}</button>`;
  const table = view === 'current'
    ? buildCurrentTable(data.standings[conf].current || [])
    : buildProjectedTable(data.standings[conf].projected || []);
  return `
    <div class="standings-head">
      <h3 class="standings-conf">${confLabel}</h3>
      <div class="seg-toggle" role="group" aria-label="${confLabel} standings view">
        ${btn('current', 'Current')}${btn('projected', 'Projected')}
      </div>
    </div>
    <div class="table-scroll">${table}</div>`;
}

function rowSepClass(rank) {
  if (rank === 6) return ' class="row-sep"';
  if (rank === 10) return ' class="row-sep row-sep--playin"';
  return '';
}

function teamCell(team) {
  return `<td class="st-team"><span class="st-team-inner">${logoHtml(team, 16)}<span>${esc(team)}</span></span></td>`;
}

function buildCurrentTable(rows) {
  const body = rows.map(r => `
    <tr${rowSepClass(r.rank)}>
      <td class="st-num">${r.rank}</td>
      ${teamCell(r.team)}
      <td class="st-num">${r.w}-${r.l}</td>
      <td class="st-num">${(r.pct != null ? r.pct.toFixed(3) : '—').replace(/^0\./, '.')}</td>
      <td class="st-num">${r.gb ? r.gb.toFixed(1) : '—'}</td>
      <td class="st-num">${esc(r.l10 || '—')}</td>
      <td class="st-num">${esc(r.streak || '—')}</td>
    </tr>`).join('');
  return `<table class="standings-table">
    <thead><tr>
      <th scope="col" class="st-num">#</th><th scope="col">Team</th><th scope="col" class="st-num">W-L</th>
      <th scope="col" class="st-num">Win %</th><th scope="col" class="st-num">GB</th>
      <th scope="col" class="st-num">L10</th><th scope="col" class="st-num">Strk</th>
    </tr></thead>
    <tbody>${body}</tbody>
  </table>`;
}

function rankChangeHtml(c) {
  if (!c) return '<span class="rc rc-same" aria-label="no change">–</span>';
  return c > 0
    ? `<span class="rc rc-up" aria-label="up ${c}"><span aria-hidden="true">▲</span>${c}</span>`
    : `<span class="rc rc-down" aria-label="down ${-c}"><span aria-hidden="true">▼</span>${-c}</span>`;
}

function buildProjectedTable(rows) {
  const body = rows.map(r => `
    <tr${rowSepClass(r.rank)}>
      <td class="st-num">${r.rank}</td>
      ${teamCell(r.team)}
      <td class="st-num">${r.exp_w}-${r.exp_l}</td>
      <td class="st-num st-muted">${r.w_p10}–${r.w_p90}</td>
      <td class="st-num">${fmtProb(r.pct_playoffs)}</td>
      <td class="st-num">${fmtProb(r.pct_playin)}</td>
      <td class="st-num">${fmtProb(r.pct_top_seed)}</td>
      <td class="st-num">${rankChangeHtml(r.rank_change)}</td>
    </tr>`).join('');
  return `<table class="standings-table">
    <thead><tr>
      <th scope="col" class="st-num">#</th><th scope="col">Team</th>
      <th scope="col" class="st-num"><abbr title="Expected wins-losses">Exp W-L</abbr></th>
      <th scope="col" class="st-num"><abbr title="Wins, 10th to 90th percentile">Range</abbr></th>
      <th scope="col" class="st-num"><abbr title="Seed 1-6">Playoffs</abbr></th>
      <th scope="col" class="st-num"><abbr title="Seed 7-10">Play-In</abbr></th>
      <th scope="col" class="st-num"><abbr title="Seed 1">#1 Seed</abbr></th>
      <th scope="col" class="st-num"><abbr title="Change vs. current rank">Δ</abbr></th>
    </tr></thead>
    <tbody>${body}</tbody>
  </table>`;
}

const ROUND_ODDS_COLS = [
  { key: 'play_in',     label: 'Play-In' },
  { key: 'playoffs',    label: 'Playoffs' },
  { key: 'r2',          label: 'R2' },
  { key: 'conf_finals', label: 'Conf Finals' },
  { key: 'finals',      label: 'Finals' },
  { key: 'champion',    label: 'Title' },
];

function buildRoundOddsTable(rows) {
  const { key, dir } = roundOddsSort;
  const sorted = [...rows].sort((a, b) => {
    if (key === 'team') return dir * a.team.localeCompare(b.team);
    return dir * ((a[key] || 0) - (b[key] || 0)) || a.team.localeCompare(b.team);
  });
  const th = (k, label, cls = 'st-num') => {
    const active = key === k;
    const aria = active ? (dir === -1 ? 'descending' : 'ascending') : 'none';
    const arrow = active ? (dir === -1 ? '▼' : '▲') : '';
    return `<th scope="col" class="${cls}" aria-sort="${aria}"><button type="button" class="sort-btn" data-sort="${k}">${label}<span class="sort-arrow" aria-hidden="true">${arrow}</span></button></th>`;
  };
  const body = sorted.map(r => `
    <tr>
      ${teamCell(r.team)}
      <td class="st-conf">${r.conf === 'east' ? 'E' : 'W'}</td>
      ${ROUND_ODDS_COLS.map(c => `<td class="st-num${c.key === 'champion' ? ' st-strong' : ''}">${fmtProb(r[c.key])}</td>`).join('')}
    </tr>`).join('');
  return `<table class="standings-table odds-table">
    <thead><tr>
      ${th('team', 'Team', '')}
      <th scope="col" class="st-conf"><abbr title="Conference">Conf</abbr></th>
      ${ROUND_ODDS_COLS.map(c => th(c.key, c.label)).join('')}
    </tr></thead>
    <tbody>${body}</tbody>
  </table>`;
}

const HIST_LABELS = { playoffs: '% Playoffs', conf_finals: '% Conf Finals', champion: '% Title' };

function buildHistoryChart(history, metric) {
  const pts = history.filter(h => h && h.date && h.teams);
  if (pts.length < 2) {
    return `<div class="sim-chart-empty">The history chart appears after the second daily run${pts.length ? ` (first data point: ${esc(humanDateLong(pts[0].date))})` : ''}.</div>`;
  }
  const teams = [...new Set(pts.flatMap(p => Object.keys(p.teams)))];
  const last = pts[pts.length - 1].teams;
  const val = (p, t) => (p.teams[t] && p.teams[t][metric] != null) ? p.teams[t][metric] : null;
  const ranked = [...teams].sort((a, b) => ((last[b] || {})[metric] || 0) - ((last[a] || {})[metric] || 0));
  const top = ranked.slice(0, 5);

  const W = 760, H = 260, PL = 44, PR = 16, PT = 12, PB = 28;
  const cW = W - PL - PR, cH = H - PT - PB;
  const maxV = metric === 'playoffs' ? 1 : Math.min(1, Math.max(0.1, Math.ceil(Math.max(...pts.flatMap(p => teams.map(t => val(p, t) || 0))) * 10) / 10));
  const cx = i => PL + (i / (pts.length - 1)) * cW;
  const cy = v => PT + cH - (v / maxV) * cH;

  const pathFor = t => {
    let d = '', pen = false;
    pts.forEach((p, i) => {
      const v = val(p, t);
      if (v == null) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${cx(i).toFixed(1)},${cy(v).toFixed(1)} `;
      pen = true;
    });
    return d.trim();
  };

  const grid = [0, 0.25, 0.5, 0.75, 1].map(f => {
    const v = f * maxV, y = cy(v).toFixed(1);
    return `<line x1="${PL}" y1="${y}" x2="${W - PR}" y2="${y}" class="hist-grid"/>` +
      `<text x="${PL - 6}" y="${(+y + 3).toFixed(1)}" text-anchor="end" class="sim-ch-label">${Math.round(v * 100)}%</text>`;
  }).join('');

  const labelIdx = new Set([0, pts.length - 1]);
  if (pts.length >= 5) labelIdx.add(Math.floor((pts.length - 1) / 2));
  const xLabels = [...labelIdx].map(i =>
    `<text x="${cx(i).toFixed(1)}" y="${H - 6}" text-anchor="middle" class="sim-ch-label">${esc(humanDate(pts[i].date))}</text>`).join('');

  const muted = ranked.slice(5).map(t =>
    `<g class="hist-team" data-team="${esc(t)}"><path d="${pathFor(t)}" class="hist-line hist-line--muted"/><path d="${pathFor(t)}" class="hist-hit"/></g>`).join('');
  const colored = top.map((t, i) =>
    `<g class="hist-team" data-team="${esc(t)}"><path d="${pathFor(t)}" class="hist-line hist-c${i + 1}"/><path d="${pathFor(t)}" class="hist-hit"/></g>`).join('');

  const legend = top.map((t, i) => `
    <button type="button" class="hist-legend-item" data-team="${esc(t)}">
      <span class="hist-swatch hist-c${i + 1}" aria-hidden="true"></span>
      <span>${esc(t)}</span>
      <span class="hist-legend-val">${fmtProb(val(pts[pts.length - 1], t))}</span>
    </button>`).join('');

  return `
    <svg class="hist-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img"
         aria-label="${esc(HIST_LABELS[metric])} over time for all teams. Top 5: ${esc(top.map(t => `${t} ${fmtProb(val(pts[pts.length - 1], t))}`).join(', '))}">
      ${grid}${muted}${colored}${xLabels}
    </svg>
    <div class="hist-readout" id="hist-readout" aria-live="polite"></div>
    <div class="hist-legend">${legend}</div>`;
}

function initProjectionSections(root, data, history) {
  root.querySelectorAll('.standings-card').forEach(card => {
    card.addEventListener('click', e => {
      const btn = e.target.closest('.seg-btn');
      if (!btn) return;
      const conf = card.dataset.conf;
      standingsView[conf] = btn.dataset.view;
      card.innerHTML = buildStandingsCard(data, conf);
      const again = card.querySelector(`.seg-btn[data-view="${standingsView[conf]}"]`);
      if (again) again.focus();
    });
  });

  const oddsWrap = root.querySelector('#round-odds-wrap');
  if (oddsWrap) {
    oddsWrap.addEventListener('click', e => {
      const btn = e.target.closest('.sort-btn');
      if (!btn) return;
      const k = btn.dataset.sort;
      roundOddsSort = roundOddsSort.key === k
        ? { key: k, dir: -roundOddsSort.dir }
        : { key: k, dir: k === 'team' ? 1 : -1 };
      oddsWrap.innerHTML = buildRoundOddsTable(data.round_odds);
      const again = oddsWrap.querySelector(`.sort-btn[data-sort="${k}"]`);
      if (again) again.focus();
    });
  }

  const sel = root.querySelector('#hist-metric');
  const histWrap = root.querySelector('#hist-wrap');
  if (sel && histWrap) {
    sel.addEventListener('change', () => {
      historyMetric = sel.value;
      histWrap.innerHTML = buildHistoryChart(history, historyMetric);
    });

    const highlight = team => {
      const svg = histWrap.querySelector('.hist-svg');
      const readout = histWrap.querySelector('#hist-readout');
      if (!svg) return;
      svg.classList.toggle('has-hl', !!team);
      svg.querySelectorAll('.hist-team').forEach(g => g.classList.toggle('is-hl', g.dataset.team === team));
      if (readout) {
        if (team && Array.isArray(history) && history.length) {
          const lastPt = history[history.length - 1];
          const v = lastPt.teams && lastPt.teams[team] ? lastPt.teams[team][historyMetric] : null;
          readout.textContent = `${team}: ${fmtProb(v)} (${HIST_LABELS[historyMetric]}, latest)`;
        } else {
          readout.textContent = '';
        }
      }
    };
    histWrap.addEventListener('mouseover', e => {
      const el = e.target.closest('[data-team]');
      highlight(el ? el.dataset.team : null);
    });
    histWrap.addEventListener('mouseleave', () => highlight(null));
    histWrap.addEventListener('focusin', e => {
      const el = e.target.closest('[data-team]');
      if (el) highlight(el.dataset.team);
    });
    histWrap.addEventListener('focusout', () => highlight(null));
  }
}

// ─── Bracket Accuracy ──────────────────────────────────────────────

function calcBracketAccuracy(data) {
  const allSeries = [
    ...(data.west.r1 || []), ...(data.west.r2 || []), ...(data.west.r3 || []),
    ...(data.east.r1 || []), ...(data.east.r2 || []), ...(data.east.r3 || []),
    data.finals,
  ].filter(s => s && s.status === 'complete' && s.winner && s.prediction);
  const correct = allSeries.filter(s => s.prediction.winner === s.winner).length;
  return { correct, total: allSeries.length };
}

// ─── Bracket Connectors (SVG) ───────────────────────────────────────

function drawBracketConnectors() {
  const container = document.getElementById('bracket-container');
  if (!container || !container.children.length) return;

  const old = container.querySelector('.bracket-svg');
  if (old) old.remove();

  container.style.position = 'relative';
  const cRect = container.getBoundingClientRect();
  if (cRect.width === 0) return;

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.classList.add('bracket-svg');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('width', cRect.width);
  svg.setAttribute('height', cRect.height);
  svg.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;overflow:visible;';
  container.insertBefore(svg, container.firstChild);

  function rel(r) {
    return {
      left:  r.left  - cRect.left,
      right: r.right - cRect.left,
      cy:    (r.top  + r.bottom) / 2 - cRect.top,
    };
  }

  function addPath(d) {
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', d);
    p.setAttribute('class', 'bracket-connector-line');
    svg.appendChild(p);
  }

  function connectRounds(fromRound, toRound, dir) {
    const fromMs = Array.from(fromRound.querySelectorAll('.bracket-matchup'));
    const toMs   = Array.from(toRound.querySelectorAll('.bracket-matchup'));
    if (!fromMs.length || !toMs.length) return;
    const ratio = Math.max(1, Math.round(fromMs.length / toMs.length));

    toMs.forEach((toM, ti) => {
      const toCard = toM.querySelector('.matchup-card');
      if (!toCard) return;
      const to = rel(toCard.getBoundingClientRect());

      for (let fi = ti * ratio; fi < (ti + 1) * ratio; fi++) {
        const fromCard = fromMs[fi]?.querySelector('.matchup-card');
        if (!fromCard) continue;
        const from = rel(fromCard.getBoundingClientRect());

        if (dir === 'right') {
          const midX = (from.right + to.left) / 2;
          addPath(`M ${from.right} ${from.cy} H ${midX} V ${to.cy} H ${to.left}`);
        } else {
          const midX = (from.left + to.right) / 2;
          addPath(`M ${from.left} ${from.cy} H ${midX} V ${to.cy} H ${to.right}`);
        }
      }
    });
  }

  function connectToFinals(confCard, finalsCard, dir) {
    const from = rel(confCard.getBoundingClientRect());
    const to   = rel(finalsCard.getBoundingClientRect());
    if (dir === 'right') {
      const midX = (from.right + to.left) / 2;
      addPath(`M ${from.right} ${from.cy} H ${midX} V ${to.cy} H ${to.left}`);
    } else {
      const midX = (from.left + to.right) / 2;
      addPath(`M ${from.left} ${from.cy} H ${midX} V ${to.cy} H ${to.right}`);
    }
  }

  const finalsCard = container.querySelector('.bracket-finals-col .matchup-card');

  const westHalf = container.querySelector('.bracket-half.west');
  if (westHalf) {
    const rounds = Array.from(westHalf.querySelectorAll('.bracket-round'));
    for (let r = 0; r < rounds.length - 1; r++) connectRounds(rounds[r], rounds[r + 1], 'right');
    if (finalsCard) {
      const last = rounds[rounds.length - 1]?.querySelector('.matchup-card');
      if (last) connectToFinals(last, finalsCard, 'right');
    }
  }

  const eastHalf = container.querySelector('.bracket-half.east');
  if (eastHalf) {
    const rounds = Array.from(eastHalf.querySelectorAll('.bracket-round'));
    for (let r = rounds.length - 1; r > 0; r--) connectRounds(rounds[r], rounds[r - 1], 'left');
    if (finalsCard) {
      const first = rounds[0]?.querySelector('.matchup-card');
      if (first) connectToFinals(first, finalsCard, 'left');
    }
  }
}

// ─── Dashboard ─────────────────────────────────────────────────────

function renderDashboard(data) {
  const root = document.getElementById('dashboard-root');
  if (!root) return;

  if (!data) {
    root.innerHTML = '<div class="db-empty">dashboard.json not found. Run the pipeline first.</div>';
    return;
  }

  const status = data.overall_status || 'ok';
  const statusLabel = { ok: 'All Systems OK', warning: 'Warnings', critical: 'Issues Detected' }[status] || status;
  const statusClass = { ok: 'db-status--ok', warning: 'db-status--warn', critical: 'db-status--crit' }[status] || '';

  const fmt = ts => {
    if (!ts) return '—';
    const d = new Date(ts);
    return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) + ' ET';
  };

  // ── Status banner ──────────────────────────────────────────────────
  const banner = `
    <div class="db-banner ${statusClass}">
      <span class="db-banner-dot"></span>
      <span class="db-banner-label">${esc(statusLabel)}</span>
      <span class="db-banner-time">Updated ${fmt(data.generated_at)}</span>
    </div>`;


  // ── Health list ────────────────────────────────────────────────────
  const h = data.health || {};

  function healthRow(title, item, statusHtml) {
    if (!item) return '';
    const ok = item.passed === true;
    const unknown = item.passed == null;
    const rowClass = unknown ? '' : (ok ? 'db-health--ok' : 'db-health--fail');
    const dotClass = unknown ? 'db-health-dot--unknown' : (ok ? 'db-health-dot--ok' : 'db-health-dot--fail');
    return `
      <div class="db-health-row ${rowClass}">
        <span class="db-health-dot ${dotClass}"></span>
        <span class="db-health-name">${esc(title)}</span>
        <span class="db-health-status">${statusHtml}</span>
        <span class="db-health-time">${fmt(item.generated_at)}</span>
      </div>`;
  }

  const dq = h.data_quality;
  const fd = h.feature_drift;
  const pa = h.prediction_audit;
  const rt = h.retraining;
  const sm = h.season_simulation && h.season_simulation.generated_at ? h.season_simulation : null;
  const smStatus = sm ? (
    sm.passed === false
      ? `<span class="db-fail-text">${sm.issues} issue${sm.issues !== 1 ? 's' : ''}</span>`
      : `<span class="db-ok-text">${sm.warnings ? 'OK with warnings' : 'OK'}</span>`
  ) + ` <span class="db-muted">(${Number(sm.sims || 0).toLocaleString('en-US')} sims · ${esc(sm.mode || '—')}${sm.runtime_seconds != null ? ` · ${sm.runtime_seconds}s` : ''})</span>` : '—';

  const healthSection = `
    <div class="db-section-title">System Health</div>
    <div class="db-health-list">
      ${healthRow('Data Quality', dq, dq ? (dq.issues === 0 ? `<span class="db-ok-text">OK</span>` + (dq.warnings ? ` <span class="db-muted">(${dq.warnings} warnings)</span>` : '') : `<span class="db-fail-text">${dq.issues} issue${dq.issues !== 1 ? 's' : ''}</span>`) : '—')}
      ${healthRow('Feature Drift', fd, fd ? (fd.issues === 0 ? `<span class="db-ok-text">No drift</span>` + (fd.warnings ? ` <span class="db-muted">(${fd.warnings} warnings)</span>` : '') : `<span class="db-fail-text">${fd.issues} issue${fd.issues !== 1 ? 's' : ''}</span>`) : '—')}
      ${healthRow('Prediction Audit', pa, pa ? `<span class="db-muted">${pa.predictions} prediction${pa.predictions !== 1 ? 's' : ''}:</span> ` + (pa.flags === 0 ? `<span class="db-ok-text">no flags</span>` : `<span class="db-fail-text">${pa.flags} flag${pa.flags !== 1 ? 's' : ''}</span>`) : '—')}
      ${healthRow('Retraining', rt, rt ? (rt.retrained ? `<span class="db-ok-text">Retrained</span> <span class="db-muted">(${pct(rt.test_accuracy)})</span>` : `<span class="db-muted">${esc(rt.reason || 'Not needed')}</span>`) : '—')}
      ${sm ? healthRow('Season Simulation', sm, smStatus) : ''}
    </div>`;

  // ── Observations ───────────────────────────────────────────────────
  const obs = data.observations || [];
  let obsSection = '';
  if (obs.length) {
    const severityOrder = { high: 0, medium: 1, low: 2 };
    const sorted = [...obs].sort((a, b) => (severityOrder[a.severity] ?? 9) - (severityOrder[b.severity] ?? 9));
    const items = sorted.map(o => {
      const cls = { high: 'db-obs--high', medium: 'db-obs--med', low: 'db-obs--low' }[o.severity] || '';
      const badge = { high: 'Critical', medium: 'Warning', low: 'Info' }[o.severity] || o.severity;
      const src = { prediction_audit: 'Prediction Audit', feature_drift: 'Feature Drift', data_quality: 'Data Quality', model_evaluator: 'Model', season_simulation: 'Season Simulation' }[o.source] || o.source;
      return `
        <div class="db-obs ${cls}">
          <div class="db-obs-top">
            <span class="db-obs-badge">${esc(badge)}</span>
            <span class="db-obs-src">${esc(src)}</span>
            <span class="db-obs-title">${esc(o.title)}</span>
          </div>
          <div class="db-obs-detail">${esc(o.detail)}</div>
        </div>`;
    }).join('');
    obsSection = `
      <div class="db-section-title">Observations <span class="db-obs-count">${obs.length}</span></div>
      <div class="db-obs-list">${items}</div>`;
  } else {
    obsSection = `
      <div class="db-section-title">Observations</div>
      <div class="db-obs-empty">No flags or issues. Everything looks good.</div>`;
  }

  root.innerHTML = `
    <div class="db-root">
      ${banner}
      ${healthSection}
      ${obsSection}
    </div>`;
}

// ─── Tab Indicator ─────────────────────────────────────────────────

function initTabIndicator() {
  const bar    = document.querySelector('.tab-bar');
  const active = bar.querySelector('.tab.active');
  if (!active) return;
  const ind = document.createElement('span');
  ind.className = 'tab-indicator';
  bar.appendChild(ind);
  const barRect = bar.getBoundingClientRect();
  const btnRect = active.getBoundingClientRect();
  ind.style.width     = btnRect.width + 'px';
  ind.style.transform = `translateX(${btnRect.left - barRect.left}px)`;
  window.addEventListener('resize', () => updateTabIndicator(bar.querySelector('.tab.active')));
}

function updateTabIndicator(btn) {
  if (!btn) return;
  const bar = document.querySelector('.tab-bar');
  const ind = bar.querySelector('.tab-indicator');
  if (!ind) return;
  const barRect = bar.getBoundingClientRect();
  const btnRect = btn.getBoundingClientRect();
  ind.style.transition = 'transform 220ms cubic-bezier(0.22,1,0.36,1), width 220ms cubic-bezier(0.22,1,0.36,1)';
  ind.style.width     = btnRect.width + 'px';
  ind.style.transform = `translateX(${btnRect.left - barRect.left}px)`;
}

// ─── Init ──────────────────────────────────────────────────────────

initTabIndicator();
loadData();

let _connDebounce;
window.addEventListener('resize', () => {
  clearTimeout(_connDebounce);
  _connDebounce = setTimeout(() => {
    if (document.querySelector('.tab[data-tab="bracket"].active')) drawBracketConnectors();
  }, 150);
});
