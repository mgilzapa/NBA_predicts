'use strict';

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
    const [predRes, oddsRes] = await Promise.all([
      fetch('predictions.json'),
      fetch('odds.json').catch(() => null),
    ]);

    if (!predRes.ok) throw new Error(`HTTP ${predRes.status}`);
    const data = await predRes.json();

    if (oddsRes && oddsRes.ok) {
      try { oddsData = await oddsRes.json(); } catch { /* ignore */ }
    }

    currentData = data;
    render(data);
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
    tbody.innerHTML    = '<tr><td colspan="6" style="text-align:center;color:var(--muted);padding:48px">No predictions yet.</td></tr>';
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

    return `<tr>
      <td class="td-date">${date}</td>
      <td>${esc(g.home_team)}</td>
      <td>${esc(g.away_team)}</td>
      <td style="color:var(--accent)">${esc(g.predicted_winner)}</td>
      <td>${esc(actual)}</td>
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

// ─── Init ──────────────────────────────────────────────────────────

loadData();
