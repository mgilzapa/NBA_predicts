'use strict';

// ─── Constants ───────────────────────────────────────────────────────────────

const SIM_KEY = 'nba_sim';
const SIM_VERSION = 1;

// ─── Module state ────────────────────────────────────────────────────────────

let _state = null;
let _pred  = null;
let _odds  = null;
let _view  = 'overview'; // 'overview' | 'detail'
let _detailId = null;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function fmtEur(val) {
  const n = parseFloat(val) || 0;
  return n.toLocaleString('de-DE', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' €';
}

function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

function genId() {
  return 'sim_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6);
}

// ─── Storage ─────────────────────────────────────────────────────────────────

function stLoad() {
  try {
    const raw = localStorage.getItem(SIM_KEY);
    if (!raw) return { version: SIM_VERSION, simulations: [] };
    const parsed = JSON.parse(raw);
    return parsed.version === SIM_VERSION ? parsed : { version: SIM_VERSION, simulations: [] };
  } catch {
    return { version: SIM_VERSION, simulations: [] };
  }
}

function stSave() {
  localStorage.setItem(SIM_KEY, JSON.stringify(_state));
}

// ─── Data fetch ──────────────────────────────────────────────────────────────

async function fetchSimData() {
  const [predRes, oddsRes] = await Promise.all([
    fetch('predictions.json').catch(() => null),
    fetch('odds.json').catch(() => null),
  ]);
  _pred = null;
  _odds = null;
  if (predRes && predRes.ok) try { _pred = await predRes.json(); } catch { /* ignore */ }
  if (oddsRes && oddsRes.ok) try { _odds = await oddsRes.json(); } catch { /* ignore */ }
}

// ─── Odds lookup ─────────────────────────────────────────────────────────────

function findBestOdds(game) {
  if (!_odds || !_odds.games) return null;
  const key = `${game.date}|${game.home_team.toLowerCase()}|${game.away_team.toLowerCase()}`;
  const entry = _odds.games[key];
  if (!entry || !entry.bookmakers || !entry.bookmakers.length) return null;
  const isHome = game.predicted_winner === game.home_team;
  let best = null;
  entry.bookmakers.forEach(bm => {
    const odds = isHome ? bm.home : bm.away;
    if (odds && (!best || odds > best.odds)) best = { odds, bookmaker: bm.name };
  });
  return best;
}

// ─── Bet resolution ──────────────────────────────────────────────────────────

function resolveBets() {
  if (!_pred || !_pred.all) return;
  const gameMap = {};
  _pred.all.forEach(g => { gameMap[g.gameId] = g; });

  let changed = false;
  _state.simulations.forEach(sim => {
    sim.bets.forEach(bet => {
      if (bet.status !== 'pending') return;
      const g = gameMap[bet.gameId];
      if (!g || g.correct === null || g.correct === undefined) return;
      const won = g.actual_winner === bet.predicted_winner;
      bet.status = won ? 'won' : 'lost';
      bet.payout = won ? parseFloat((bet.stake * bet.odds).toFixed(2)) : 0;
      // At placement stake was already deducted. Add payout back (0 if lost).
      sim.capital = parseFloat((sim.capital + bet.payout).toFixed(2));
      changed = true;
    });
  });
  if (changed) stSave();
}

// ─── Chart ───────────────────────────────────────────────────────────────────

function buildChartPoints(sim) {
  const byDate = {};
  sim.bets.filter(b => b.status !== 'pending').forEach(b => {
    byDate[b.date] = parseFloat(((byDate[b.date] || 0) + (b.payout - b.stake)).toFixed(2));
  });
  const allDates = [...new Set([sim.started_at, ...Object.keys(byDate)])].sort();
  let cap = sim.starting_capital;
  return allDates.map(d => {
    cap = parseFloat((cap + (byDate[d] || 0)).toFixed(2));
    return { date: d, capital: cap };
  });
}

function renderChart(sim) {
  const pts = buildChartPoints(sim);
  if (pts.length < 2) {
    return '<div class="sim-chart-empty">Platziere Wetten um den Chart zu sehen.</div>';
  }

  const W = 600, H = 180;
  const PL = 52, PR = 16, PT = 14, PB = 32;
  const cW = W - PL - PR, cH = H - PT - PB;

  const vals = pts.map(p => p.capital);
  const rawMin = Math.min(...vals, sim.starting_capital);
  const rawMax = Math.max(...vals, sim.starting_capital);
  const pad = (rawMax - rawMin) * 0.08 || rawMin * 0.02 || 1;
  const minV = rawMin - pad;
  const maxV = rawMax + pad;
  const rng = maxV - minV;

  const cx = i => PL + (i / Math.max(pts.length - 1, 1)) * cW;
  const cy = v => PT + cH - ((v - minV) / rng) * cH;

  const baseY = cy(sim.starting_capital).toFixed(1);
  const polyPts = pts.map((p, i) => `${cx(i).toFixed(1)},${cy(p.capital).toFixed(1)}`).join(' ');

  // X labels: first + last + middle if ≥ 4 points
  const labelIdx = new Set([0, pts.length - 1]);
  if (pts.length >= 4) labelIdx.add(Math.floor(pts.length / 2));
  const xLabels = [...labelIdx].map(i =>
    `<text x="${cx(i).toFixed(1)}" y="${H - 4}" text-anchor="middle" class="sim-ch-label">${pts[i].date.slice(5)}</text>`
  ).join('');

  // Y labels
  const yLabelTop = `<text x="${PL - 5}" y="${(PT + 8).toFixed(1)}" text-anchor="end" class="sim-ch-label">${Math.round(maxV)}</text>`;
  const yLabelBot = `<text x="${PL - 5}" y="${(PT + cH + 4).toFixed(1)}" text-anchor="end" class="sim-ch-label">${Math.round(minV)}</text>`;

  // Dots
  const dots = pts.map((p, i) => {
    const cls = p.capital >= sim.starting_capital ? 'sim-dot-pos' : 'sim-dot-neg';
    return `<circle cx="${cx(i).toFixed(1)}" cy="${cy(p.capital).toFixed(1)}" r="3.5" class="${cls}" />`;
  }).join('');

  return `
<svg class="sim-chart-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
  <line x1="${PL}" y1="${baseY}" x2="${W - PR}" y2="${baseY}" class="sim-baseline" stroke-dasharray="5,4" />
  <polyline points="${polyPts}" class="sim-chartline" fill="none" />
  ${dots}
  ${xLabels}
  ${yLabelTop}
  ${yLabelBot}
</svg>`;
}

// ─── Overview ────────────────────────────────────────────────────────────────

function renderOverview() {
  const root = document.getElementById('sim-root');
  const sims = _state.simulations;

  const cards = sims.map(sim => {
    const pl = parseFloat((sim.capital - sim.starting_capital).toFixed(2));
    const plPct = ((pl / sim.starting_capital) * 100).toFixed(1);
    const sign = pl >= 0 ? '+' : '';
    const polarity = pl >= 0 ? 'pos' : 'neg';
    const pending = sim.bets.filter(b => b.status === 'pending').length;
    const pendingBadge = pending
      ? `<span class="sim-pending-badge">${pending} ausstehend</span>` : '';

    return `
<div class="sim-card" role="button" tabindex="0" data-simid="${esc(sim.id)}">
  <div class="sim-card-name">${esc(sim.name)}</div>
  <div class="sim-card-meta">ab&nbsp;${esc(sim.started_at)}&nbsp;&middot;&nbsp;Start:&nbsp;<span class="sim-mono">${fmtEur(sim.starting_capital)}</span></div>
  <div class="sim-capital-row">
    <span class="sim-capital-val">${fmtEur(sim.capital)}</span>
    <span class="sim-pl sim-pl-${polarity}">${sign}${fmtEur(pl)}&nbsp;(${sign}${plPct}%)</span>
  </div>
  <div class="sim-card-stats">
    <span>${sim.bets.length}&nbsp;Wette${sim.bets.length !== 1 ? 'n' : ''}</span>
    ${pendingBadge}
  </div>
</div>`;
  }).join('');

  root.innerHTML = `
<div class="sim-overview-header">
  <div class="sim-section-title">Simulationen</div>
  <button class="sim-new-btn" id="sim-new-btn">+ Neue Simulation</button>
</div>
<div class="sim-grid">
  ${cards}
  <div class="sim-card sim-card-new" role="button" tabindex="0" id="sim-card-new" aria-label="Neue Simulation erstellen">
    <div class="sim-card-new-icon">+</div>
    <div class="sim-card-new-label">Neue Simulation</div>
  </div>
</div>`;

  root.querySelectorAll('.sim-card:not(.sim-card-new)').forEach(card => {
    const go = () => showDetail(card.dataset.simid);
    card.addEventListener('click', go);
    card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
  });

  const newCard = document.getElementById('sim-card-new');
  newCard.addEventListener('click', openModal);
  newCard.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openModal(); } });
  document.getElementById('sim-new-btn').addEventListener('click', openModal);
}

// ─── Detail ──────────────────────────────────────────────────────────────────

function showDetail(id) {
  _view = 'detail';
  _detailId = id;
  renderDetail();
}

function showOverview() {
  _view = 'overview';
  _detailId = null;
  renderOverview();
}

function renderDetail() {
  const root = document.getElementById('sim-root');
  const sim = _state.simulations.find(s => s.id === _detailId);
  if (!sim) { showOverview(); return; }

  const pl = parseFloat((sim.capital - sim.starting_capital).toFixed(2));
  const plPct = ((pl / sim.starting_capital) * 100).toFixed(1);
  const sign = pl >= 0 ? '+' : '';
  const polarity = pl >= 0 ? 'pos' : 'neg';

  const today = todayStr();
  const todayBets = sim.bets.filter(b => b.date === today);
  const alreadyBetToday = todayBets.length > 0;
  const todayGames = _pred && _pred.today ? _pred.today : [];

  const todaySection = buildTodaySection(sim, todayGames, alreadyBetToday, todayBets);
  const historyHtml = buildHistoryTable(sim);
  const chartHtml = renderChart(sim);

  root.innerHTML = `
<div class="sim-detail-header">
  <button class="sim-back-btn" id="sim-back-btn">&#8592; Zurück</button>
  <div class="sim-detail-name">${esc(sim.name)}</div>
  <button class="sim-delete-btn" id="sim-delete-btn">Löschen</button>
</div>
<div class="sim-summary-row">
  <div class="sim-summary-item">
    <div class="sim-summary-label">Kapital</div>
    <div class="sim-summary-val sim-mono">${fmtEur(sim.capital)}</div>
  </div>
  <div class="sim-summary-item">
    <div class="sim-summary-label">P&amp;L</div>
    <div class="sim-summary-val sim-mono sim-pl-${polarity}">${sign}${fmtEur(pl)}</div>
  </div>
  <div class="sim-summary-item">
    <div class="sim-summary-label">Rendite</div>
    <div class="sim-summary-val sim-mono sim-pl-${polarity}">${sign}${plPct}%</div>
  </div>
  <div class="sim-summary-item sim-summary-item--sm">
    <div class="sim-summary-label">Startkapital</div>
    <div class="sim-summary-val sim-mono">${fmtEur(sim.starting_capital)}</div>
  </div>
  <div class="sim-summary-item sim-summary-item--sm">
    <div class="sim-summary-label">Start</div>
    <div class="sim-summary-val">${esc(sim.started_at)}</div>
  </div>
</div>
<div class="sim-chart-wrap">${chartHtml}</div>
${todaySection}
${historyHtml}`;

  document.getElementById('sim-back-btn').addEventListener('click', showOverview);
  document.getElementById('sim-delete-btn').addEventListener('click', () => deleteSim(sim.id));

  if (!alreadyBetToday && todayGames.length > 0) {
    wireConfirm(sim);
  }
}

// ─── Today section ───────────────────────────────────────────────────────────

function buildTodaySection(sim, todayGames, alreadyBetToday, todayBets) {
  const header = '<div class="sim-section-title sim-section-title--today">Heute tippen</div>';

  if (todayGames.length === 0) {
    return `<div class="sim-today-section">${header}<div class="sim-empty">Keine Spiele heute.</div></div>`;
  }

  if (alreadyBetToday) {
    const rows = todayBets.map(bet => `
<div class="sim-today-row sim-today-readonly">
  <div class="sim-today-teams">${esc(bet.away_team)} @ ${esc(bet.home_team)}</div>
  <div class="sim-today-pick">Tipp: <strong>${esc(bet.predicted_winner)}</strong></div>
  <div class="sim-today-placed">
    <span class="sim-mono">${fmtEur(bet.stake)} &times; ${bet.odds.toFixed(2)}</span>
    <span class="sim-status-badge sim-badge-pending">Ausstehend</span>
  </div>
</div>`).join('');
    return `<div class="sim-today-section">${header}${rows}</div>`;
  }

  const rows = todayGames.map(game => {
    const bestOdds = findBestOdds(game);
    const oddsVal = bestOdds ? bestOdds.odds.toFixed(2) : '';
    const oddsSource = bestOdds ? 'bookmaker' : 'manual';
    const predIsHome = game.predicted_winner === game.home_team;
    const predProb = predIsHome ? game.probability_home_win : (1 - game.probability_home_win);
    const probPct = Math.round(predProb * 100);

    return `
<div class="sim-today-row" data-gameid="${esc(game.gameId)}">
  <label class="sim-today-check-wrap" title="Auswählen">
    <input type="checkbox" class="sim-game-check"
      data-gameid="${esc(game.gameId)}"
      data-home="${esc(game.home_team)}"
      data-away="${esc(game.away_team)}"
      data-winner="${esc(game.predicted_winner)}"
      data-odds-source="${oddsSource}"
    />
  </label>
  <div class="sim-today-info-col">
    <div class="sim-today-teams">${esc(game.away_team)} @ ${esc(game.home_team)}</div>
    <div class="sim-today-pick">Tipp:&nbsp;<strong>${esc(game.predicted_winner)}</strong>&nbsp;<span class="sim-prob">${probPct}%</span></div>
  </div>
  <div class="sim-today-inputs">
    <div class="sim-input-group">
      <label class="sim-input-label" for="stake-${esc(game.gameId)}">Einsatz&nbsp;(€)</label>
      <input type="number" id="stake-${esc(game.gameId)}" class="sim-input sim-stake-input"
        data-gameid="${esc(game.gameId)}" min="0.01" step="0.01" placeholder="0.00" />
    </div>
    <div class="sim-input-group">
      <label class="sim-input-label" for="odds-${esc(game.gameId)}">Quote</label>
      <input type="number" id="odds-${esc(game.gameId)}" class="sim-input sim-odds-input"
        data-gameid="${esc(game.gameId)}" min="1.01" step="0.01" placeholder="1.00"
        value="${esc(oddsVal)}" data-source="${oddsSource}" />
    </div>
  </div>
</div>`;
  }).join('');

  return `
<div class="sim-today-section" id="sim-today-section">
  ${header}
  ${rows}
  <div class="sim-today-footer">
    <button class="sim-confirm-btn" id="sim-confirm-btn" disabled>Bets bestätigen</button>
  </div>
</div>`;
}

function wireConfirm(sim) {
  const section = document.getElementById('sim-today-section');
  if (!section) return;
  const confirmBtn = document.getElementById('sim-confirm-btn');

  function validate() {
    const checked = section.querySelectorAll('.sim-game-check:checked');
    if (!checked.length) { confirmBtn.disabled = true; return; }
    let ok = true;
    checked.forEach(cb => {
      const id = cb.dataset.gameid;
      const stake = parseFloat(section.querySelector(`.sim-stake-input[data-gameid="${id}"]`).value);
      const odds  = parseFloat(section.querySelector(`.sim-odds-input[data-gameid="${id}"]`).value);
      if (!stake || stake <= 0 || !odds || odds <= 1) ok = false;
    });
    confirmBtn.disabled = !ok;
  }

  section.addEventListener('change', validate);
  section.addEventListener('input', validate);

  confirmBtn.addEventListener('click', () => {
    placeBets(sim, section);
  });
}

function placeBets(sim, section) {
  const checked = section.querySelectorAll('.sim-game-check:checked');
  const today = todayStr();

  checked.forEach(cb => {
    const id = cb.dataset.gameid;
    const stakeRaw = parseFloat(section.querySelector(`.sim-stake-input[data-gameid="${id}"]`).value);
    const oddsRaw  = parseFloat(section.querySelector(`.sim-odds-input[data-gameid="${id}"]`).value);
    const stake    = parseFloat(Math.min(stakeRaw, sim.capital).toFixed(2));
    const odds     = parseFloat(oddsRaw.toFixed(2));
    if (stake <= 0) return;

    sim.bets.push({
      date: today,
      gameId: id,
      home_team: cb.dataset.home,
      away_team: cb.dataset.away,
      predicted_winner: cb.dataset.winner,
      stake,
      odds,
      odds_source: cb.dataset.oddsSource,
      status: 'pending',
      payout: null,
    });
    sim.capital = parseFloat((sim.capital - stake).toFixed(2));
  });

  stSave();
  renderDetail();
}

// ─── Bet history ─────────────────────────────────────────────────────────────

function buildHistoryTable(sim) {
  const title = '<div class="sim-section-title">Wettenhistorie</div>';
  if (!sim.bets.length) {
    return `<div class="sim-history">${title}<div class="sim-empty">Noch keine Wetten platziert.</div></div>`;
  }

  const rows = [...sim.bets]
    .sort((a, b) => b.date.localeCompare(a.date))
    .map(bet => {
      const statusClass = bet.status === 'won' ? 'sim-badge-won'
        : bet.status === 'lost' ? 'sim-badge-lost' : 'sim-badge-pending';
      const statusLabel = bet.status === 'won' ? 'Gewonnen'
        : bet.status === 'lost' ? 'Verloren' : 'Ausstehend';
      const netPl = bet.status === 'won' ? parseFloat((bet.payout - bet.stake).toFixed(2))
        : bet.status === 'lost' ? -bet.stake : null;
      const plText = netPl !== null
        ? `<span class="sim-mono ${netPl >= 0 ? 'sim-pl-pos' : 'sim-pl-neg'}">${netPl >= 0 ? '+' : ''}${fmtEur(netPl)}</span>`
        : '<span class="sim-muted">—</span>';

      return `
<tr>
  <td class="sim-td-date">${esc(bet.date)}</td>
  <td>${esc(bet.away_team)}&nbsp;@&nbsp;${esc(bet.home_team)}</td>
  <td>${esc(bet.predicted_winner)}</td>
  <td class="sim-mono">${fmtEur(bet.stake)}</td>
  <td class="sim-mono">${bet.odds.toFixed(2)}</td>
  <td><span class="sim-status-badge ${statusClass}">${statusLabel}</span></td>
  <td>${plText}</td>
</tr>`;
    }).join('');

  return `
<div class="sim-history">
  ${title}
  <div class="sim-table-wrap">
    <table class="sim-table">
      <thead>
        <tr>
          <th>Datum</th><th>Spiel</th><th>Tipp</th>
          <th>Einsatz</th><th>Quote</th><th>Ergebnis</th><th>P&amp;L</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  </div>
</div>`;
}

// ─── Delete ──────────────────────────────────────────────────────────────────

function deleteSim(id) {
  if (!confirm('Simulation wirklich löschen?\nDiese Aktion kann nicht rückgängig gemacht werden.')) return;
  _state.simulations = _state.simulations.filter(s => s.id !== id);
  stSave();
  showOverview();
}

// ─── Modal ───────────────────────────────────────────────────────────────────

function openModal() {
  const modal = document.getElementById('sim-modal');
  modal.removeAttribute('hidden');
  modal.setAttribute('aria-hidden', 'false');
  document.getElementById('sim-modal-name').value = '';
  document.getElementById('sim-modal-capital').value = '';
  requestAnimationFrame(() => document.getElementById('sim-modal-capital').focus());
}

function closeModal() {
  const modal = document.getElementById('sim-modal');
  modal.setAttribute('hidden', '');
  modal.setAttribute('aria-hidden', 'true');
}

function handleCreate(e) {
  e.preventDefault();
  const capitalEl = document.getElementById('sim-modal-capital');
  const capital = parseFloat(capitalEl.value);
  if (!capital || capital <= 0) {
    capitalEl.focus();
    return;
  }
  const nameEl = document.getElementById('sim-modal-name');
  const n = _state.simulations.length + 1;
  const name = nameEl.value.trim() || `Sim #${n}`;

  const sim = {
    id: genId(),
    name,
    started_at: todayStr(),
    starting_capital: parseFloat(capital.toFixed(2)),
    capital: parseFloat(capital.toFixed(2)),
    bets: [],
  };
  _state.simulations.push(sim);
  stSave();
  closeModal();
  showDetail(sim.id);
}

// ─── Tab activation ──────────────────────────────────────────────────────────

async function onSimTabActivated() {
  const root = document.getElementById('sim-root');
  root.innerHTML = '<div class="sim-empty sim-loading">Lade Daten…</div>';

  _state = stLoad();
  await fetchSimData();
  resolveBets();

  if (_view === 'detail' && _detailId && _state.simulations.find(s => s.id === _detailId)) {
    renderDetail();
  } else {
    _view = 'overview';
    _detailId = null;
    renderOverview();
  }
}

// ─── Init ────────────────────────────────────────────────────────────────────

function simInit() {
  // Modal
  const modal = document.getElementById('sim-modal');
  document.getElementById('sim-modal-form').addEventListener('submit', handleCreate);
  document.getElementById('sim-modal-cancel').addEventListener('click', closeModal);
  document.getElementById('sim-modal-close').addEventListener('click', closeModal);
  modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && !modal.hasAttribute('hidden')) closeModal(); });

  // Tab button
  const tabBtn = document.getElementById('tab-btn-simulation');
  if (tabBtn) tabBtn.addEventListener('click', onSimTabActivated);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', simInit);
} else {
  simInit();
}
