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
let activeFilters = { team: '', dateFrom: '', dateTo: '', quickDate: '', result: '', seasonType: '' };

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
    const [predRes, oddsRes, bracketRes, dashRes] = await Promise.all([
      fetch('predictions.json'),
      fetch('odds.json').catch(() => null),
      fetch('bracket.json').catch(() => null),
      fetch('dashboard.json').catch(() => null),
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

    currentData = data;
    render(data);
    renderBracket(bracketData);
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
    const hasAny = Object.values(activeFilters).some(v => v !== '');
    document.getElementById('f-reset').style.display = hasAny ? '' : 'none';
    renderAll(applyFilters(allGames()));
  }

  document.getElementById('f-team').addEventListener('change', syncAndRender);
  document.getElementById('f-result').addEventListener('change', syncAndRender);
  document.getElementById('f-season').addEventListener('change', syncAndRender);

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
    activeFilters = { team: '', dateFrom: '', dateTo: '', quickDate: '', result: '', seasonType: '' };
    document.getElementById('f-team').value = '';
    document.getElementById('f-result').value = '';
    document.getElementById('f-season').value = '';
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
      <td style="color:var(--accent)">${esc(g.predicted_winner)}</td>
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

function renderBracket(data) {
  const header    = document.getElementById('bracket-header');
  const container = document.getElementById('bracket-container');
  const mobile    = document.getElementById('bracket-mobile');

  if (!data) {
    header.textContent = 'Bracket data not available yet.';
    return;
  }

  const acc = calcBracketAccuracy(data);
  const accHtml = acc.total > 0
    ? ` · <span class="accuracy-pill">${acc.correct}/${acc.total} series correct</span>`
    : '';
  header.innerHTML = `Season <strong>${esc(data.season)}</strong> · Updated ${esc(data.generated_at.slice(0, 16).replace('T', ' '))} ET${accHtml}`;

  container.innerHTML = buildBracketDesktop(data);
  mobile.innerHTML    = buildBracketMobile(data);
}

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

function buildFinalsWinner(series) {
  if (!series) return '';
  const champion = series.status === 'complete' ? series.winner
    : (series.prediction ? series.prediction.winner : null);
  if (!champion) return '';
  const url = teamLogoUrl(champion);
  const logoEl = url
    ? `<img class="finals-winner-logo" src="${url}" alt="${esc(champion)}" width="72" height="72" loading="lazy" decoding="async" onerror="this.style.display='none'">`
    : '';
  const label = series.status === 'complete' ? 'Champion' : 'Predicted Winner';
  return `<div class="finals-winner-banner">
    ${logoEl}
    <div class="finals-winner-label">${esc(label)}</div>
    <div class="finals-winner-name">${esc(champion)}</div>
  </div>`;
}

function buildBracketDesktop(data) {
  const westR1  = buildRoundCol(bracketOrderR1(data.west.r1), 'R1');
  const westR2  = buildRoundCol(data.west.r2,  'R2');
  const westR3  = buildRoundCol(data.west.r3,  'Conf Finals');
  const eastR3  = buildRoundCol(data.east.r3,  'Conf Finals');
  const eastR2  = buildRoundCol(data.east.r2,  'R2');
  const eastR1  = buildRoundCol(bracketOrderR1(data.east.r1), 'R1');

  const winnerBanner = buildFinalsWinner(data.finals);
  const finalsCard   = buildMatchupCard(data.finals);
  const finalsCol    = `<div class="bracket-finals-col">
    ${winnerBanner}
    <div class="bracket-finals-label">NBA Finals</div>
    ${finalsCard}
  </div>`;

  return `
    <div class="bracket-half west">${westR1}${westR2}${westR3}</div>
    ${finalsCol}
    <div class="bracket-half east">${eastR3}${eastR2}${eastR1}</div>
  `;
}

function buildBracketMobile(data) {
  const rounds = [
    { label: 'First Round',        east: bracketOrderR1(data.east.r1), west: bracketOrderR1(data.west.r1) },
    { label: 'Second Round',       east: data.east.r2, west: data.west.r2 },
    { label: 'Conference Finals',  east: data.east.r3, west: data.west.r3 },
    { label: 'NBA Finals',         east: [data.finals], west: [] },
  ];

  return rounds.map(r => `
    <div class="bm-round-title">${esc(r.label)}</div>
    ${r.east.length ? `<div class="bm-conf-label">East</div><div class="bm-series-list">${r.east.map(s => buildMatchupCard(s)).join('')}</div>` : ''}
    ${r.west.length ? `<div class="bm-conf-label">West</div><div class="bm-series-list">${r.west.map(s => buildMatchupCard(s)).join('')}</div>` : ''}
  `).join('');
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

  const healthSection = `
    <div class="db-section-title">System Health</div>
    <div class="db-health-list">
      ${healthRow('Data Quality', dq, dq ? (dq.issues === 0 ? `<span class="db-ok-text">OK</span>` + (dq.warnings ? ` <span class="db-muted">(${dq.warnings} warnings)</span>` : '') : `<span class="db-fail-text">${dq.issues} issue${dq.issues !== 1 ? 's' : ''}</span>`) : '—')}
      ${healthRow('Feature Drift', fd, fd ? (fd.issues === 0 ? `<span class="db-ok-text">No drift</span>` + (fd.warnings ? ` <span class="db-muted">(${fd.warnings} warnings)</span>` : '') : `<span class="db-fail-text">${fd.issues} issue${fd.issues !== 1 ? 's' : ''}</span>`) : '—')}
      ${healthRow('Prediction Audit', pa, pa ? `<span class="db-muted">${pa.predictions} prediction${pa.predictions !== 1 ? 's' : ''}:</span> ` + (pa.flags === 0 ? `<span class="db-ok-text">no flags</span>` : `<span class="db-fail-text">${pa.flags} flag${pa.flags !== 1 ? 's' : ''}</span>`) : '—')}
      ${healthRow('Retraining', rt, rt ? (rt.retrained ? `<span class="db-ok-text">Retrained</span> <span class="db-muted">(${pct(rt.test_accuracy)})</span>` : `<span class="db-muted">${esc(rt.reason || 'Not needed')}</span>`) : '—')}
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
      const src = { prediction_audit: 'Prediction Audit', feature_drift: 'Feature Drift', data_quality: 'Data Quality', model_evaluator: 'Model' }[o.source] || o.source;
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
