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

// ─── Tab Switching ─────────────────────────────────────────────────

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;

    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));

    btn.classList.add('active');
    const panel = document.getElementById('tab-' + target);
    panel.classList.add('active');

    // Re-animate bars when switching to a tab with cards
    animateBars(panel);
  });
});

// ─── Data Loading ──────────────────────────────────────────────────

async function loadData() {
  try {
    const [predRes, oddsRes, bracketRes] = await Promise.all([
      fetch('predictions.json'),
      fetch('odds.json').catch(() => null),
      fetch('bracket.json').catch(() => null),
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

    currentData = data;
    render(data);
    renderBracket(bracketData);
  } catch (err) {
    showGlobalError(err.message);
  }
}

function showGlobalError(msg) {
  ['today-cards', 'yesterday-cards'].forEach(id => {
    document.getElementById(id).innerHTML =
      `<div class="error-msg">Failed to load predictions.json — ${msg}</div>`;
  });
}

// ─── Render ────────────────────────────────────────────────────────

function render(data) {
  updateMeta(data.generated_at);
  renderToday(data.today || []);
  renderYesterday(data.yesterday || []);
  renderAll(data.all || []);
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
  const key = `${game.date}|${game.home_team.toLowerCase().trim()}|${game.away_team.toLowerCase().trim()}`;
  return oddsData.games[key] || null;
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
    `<span class="game-count">${count} Game${count !== 1 ? 's' : ''} · ${correct}/${count} Correct</span>`;

  grid.innerHTML = games.map(g => buildCard(g, true, matchOdds(g))).join('');
  animateBars(grid);
}

function renderAll(games) {
  const header = document.getElementById('all-header');
  const tbody  = document.getElementById('all-tbody');

  if (games.length === 0) {
    header.textContent = '';
    tbody.innerHTML    = '<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:48px">No predictions yet.</td></tr>';
    return;
  }

  const total   = games.length;
  const decided = games.filter(g => g.correct !== null && g.correct !== undefined);
  const correct = decided.filter(g => g.correct === true).length;
  const accuracy = decided.length > 0
    ? Math.round((correct / decided.length) * 100)
    : null;

  header.innerHTML = accuracy !== null
    ? `${total} predictions · <strong style="color:var(--accent)">${accuracy}%</strong> accuracy (${correct}/${decided.length} decided)`
    : `${total} predictions`;

  tbody.innerHTML = games.map(g => {
    const date      = g.date ? humanDate(g.date) : '—';
    const actual    = g.actual_winner || '—';
    const resultCls = g.correct === true  ? 'r-correct' :
                      g.correct === false ? 'r-wrong'   : 'r-pending';
    const resultTxt = g.correct === true  ? '✓' :
                      g.correct === false ? '✗' : '—';

    const oddsEntry = matchOdds(g);
    const oddsCell  = bestOddsCell(g, oddsEntry);

    return `<tr>
      <td class="td-date">${date}</td>
      <td>${esc(g.home_team)}</td>
      <td>${esc(g.away_team)}</td>
      <td style="color:var(--accent)">${esc(g.predicted_winner)}</td>
      <td>${esc(actual)}</td>
      <td class="td-odds">${oddsCell}</td>
      <td class="td-result ${resultCls}">${resultTxt}</td>
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

function teamLogoUrl(name) {
  const abbr = TEAM_LOGOS[name];
  return abbr ? `https://a.espncdn.com/i/teamlogos/nba/500/${abbr}.png` : null;
}

function logoHtml(teamName) {
  const url = teamLogoUrl(teamName);
  if (!url) return '';
  return `<img class="team-logo" src="${url}" alt="${esc(teamName)}" onerror="this.style.display='none'">`;
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

  header.innerHTML =
    `Season <strong>${esc(data.season)}</strong> · Updated ${esc(data.generated_at.slice(0, 16).replace('T', ' '))} ET`;

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

  const scoreHtml = (series.status === 'complete' || series.home_wins + series.away_wins > 0)
    ? `<span class="mc-score">${series.home_wins}–${series.away_wins}</span>`
    : '';

  let predHtml = '';
  if (pred) {
    predHtml = `<div class="mc-pred">
      → <span class="pred-winner">${esc(pred.winner)}</span>
      <span class="pred-prob">${Math.round(pred.win_probability * 100)}% · in ${pred.predicted_length}G</span>
    </div>`;
  }

  let badge = '';
  if (series.status === 'complete' && pred) {
    const correct = pred.winner === winner;
    badge = `<span class="mc-badge ${correct ? 'correct' : 'wrong'}">${correct ? '✓' : '✗'}</span>`;
  }

  const cardCls = `matchup-card status-${series.status}`;
  return `<div class="${cardCls}">
    ${badge}
    <div class="mc-teams">
      <div class="mc-team ${awayCls}">
        ${logoHtml(away)}
        ${esc(away)}
      </div>
      <div class="mc-team ${homeCls}">
        ${logoHtml(home)}
        ${esc(home)}
        ${scoreHtml}
      </div>
    </div>
    ${predHtml}
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

function buildBracketDesktop(data) {
  const eastR1  = buildRoundCol(data.east.r1,  'R1');
  const eastR2  = buildRoundCol(data.east.r2,  'R2');
  const eastR3  = buildRoundCol(data.east.r3,  'Conf Finals');
  const westR3  = buildRoundCol(data.west.r3,  'Conf Finals');
  const westR2  = buildRoundCol(data.west.r2,  'R2');
  const westR1  = buildRoundCol(data.west.r1,  'R1');

  const finalsCard = buildMatchupCard(data.finals);
  const finalsCol  = `<div class="bracket-finals-col">
    <div class="bracket-finals-label">Finals</div>
    ${finalsCard}
  </div>`;

  return `
    <div class="bracket-half east">${eastR1}${eastR2}${eastR3}</div>
    ${finalsCol}
    <div class="bracket-half west">${westR3}${westR2}${westR1}</div>
  `;
}

function buildBracketMobile(data) {
  const rounds = [
    { label: 'First Round',        east: data.east.r1, west: data.west.r1 },
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

// ─── Init ──────────────────────────────────────────────────────────

loadData();
