/**
 * March Madness Interactive Bracket Renderer
 *
 * Renders a full NCAA-style bracket as inline SVG and wires up
 * click-to-pick, auto-fill, random, and reset interactions.
 */

/* ═══════════════════════════════════════════════════════════════════════
   Constants
   ═══════════════════════════════════════════════════════════════════════ */
const NS = 'http://www.w3.org/2000/svg';

const C = {
  BOX_W:  110,
  BOX_H:  20,
  GAME_H: 40,   // BOX_H * 2
  R1_GAP: 6,
  CONN_W: 14,
  STEP:   124,  // BOX_W + CONN_W
  HEADER_H: 36,
  REGION_GAP: 24,
  MARGIN: 14,
  COLS: 11,
};

// Computed
C.REGION_H = 8 * C.GAME_H + 7 * C.R1_GAP;   // 362
C.SIDE_H   = 2 * C.REGION_H + C.REGION_GAP;  // 748
C.SVG_W    = 2 * C.MARGIN + C.COLS * C.STEP - C.CONN_W; // 1392
C.SVG_H    = C.HEADER_H + C.SIDE_H + 2 * C.MARGIN;     // 812

// Which region sits where
const REGION_CFG = {
  W: { side: 'left',  pos: 'top' },
  X: { side: 'left',  pos: 'bottom' },
  Y: { side: 'right', pos: 'top' },
  Z: { side: 'right', pos: 'bottom' },
};

// Region CSS colours (match CSS vars)
const REGION_COLORS = {
  W: '#3498db',
  X: '#e74c3c',
  Y: '#2ecc71',
  Z: '#f39c12',
};

// Round header labels, columns left→right
const ROUND_HEADERS = [
  'Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8',
  'Final Four', 'Championship', 'Final Four',
  'Elite 8', 'Sweet 16', 'Round of 32', 'Round of 64',
];

/* ═══════════════════════════════════════════════════════════════════════
   SVG helpers
   ═══════════════════════════════════════════════════════════════════════ */
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) el.setAttribute(k, v);
  }
  return el;
}

function svgText(x, y, text, cls, extra = {}) {
  const t = svgEl('text', { x, y, class: cls, ...extra });
  t.textContent = text;
  return t;
}

/* ═══════════════════════════════════════════════════════════════════════
   Layout helpers
   ═══════════════════════════════════════════════════════════════════════ */

/** Column index (0-10) for a given round and side */
function colForRound(round, side) {
  if (side === 'left')  return round - 1;       // R1→0, R2→1, R3→2, R4→3
  if (side === 'right') return C.COLS - round;   // R1→10, R2→9, R3→8, R4→7
  return 5; // center
}

/** X position for a column */
function colX(col) { return C.MARGIN + col * C.STEP; }

/** Top-Y offset for a region */
function regionY(pos) {
  const baseY = C.HEADER_H + C.MARGIN;
  return pos === 'top' ? baseY : baseY + C.REGION_H + C.REGION_GAP;
}

/** Centre-Y for a game in a given round (recursive centering) */
function gameCenterY(round, idx, regY) {
  if (round === 1) {
    return regY + idx * (C.GAME_H + C.R1_GAP) + C.GAME_H / 2;
  }
  const a = gameCenterY(round - 1, idx * 2,     regY);
  const b = gameCenterY(round - 1, idx * 2 + 1, regY);
  return (a + b) / 2;
}

/* ═══════════════════════════════════════════════════════════════════════
   Main application class
   ═══════════════════════════════════════════════════════════════════════ */
class BracketApp {
  constructor() {
    this.gender  = 'M';
    this.season  = null;
    this.seasons = {};
    this.data    = null;        // BracketResponse
    this.teamIdx = {};          // teamId → TeamInfo
    this.gameIdx = {};          // slot   → GameInfo
    this.overrides = {};        // slot   → teamId
  }

  /* ── Bootstrap ───────────────────────────────────────────────────── */
  async init() {
    this.showLoading(true, 'Loading seasons…');
    try {
      this.seasons = await API.getSeasons();
      this.populateSeasons();
      this.bindEvents();
      await this.loadBracket();
    } catch (e) {
      console.error('Init failed:', e);
      this.showError(e.message);
    }
    this.showLoading(false);
  }

  /* ── Season dropdown ─────────────────────────────────────────────── */
  populateSeasons() {
    const sel = document.getElementById('season-select');
    sel.innerHTML = '';
    const list = this.seasons[this.gender] || [];
    for (const s of list) {
      if (s === 2020) continue;            // COVID — skip
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = `${s - 1}–${s}`;
      sel.appendChild(opt);
    }
    // Default to latest
    this.season = list.filter(s => s !== 2020).at(-1) || list.at(-1);
    sel.value = this.season;
  }

  /* ── Event wiring ────────────────────────────────────────────────── */
  bindEvents() {
    // Gender toggle
    document.querySelectorAll('.toggle-group button').forEach(btn => {
      btn.addEventListener('click', () => {
        if (btn.dataset.gender === this.gender) return;
        document.querySelectorAll('.toggle-group button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.gender = btn.dataset.gender;
        this.overrides = {};
        this.populateSeasons();
        this.loadBracket();
      });
    });

    // Season select
    document.getElementById('season-select').addEventListener('change', e => {
      this.season = parseInt(e.target.value);
      this.overrides = {};
      this.loadBracket();
    });

    // Action buttons
    document.getElementById('btn-auto').addEventListener('click', () => {
      this.overrides = {};
      this.loadBracket('deterministic');
    });
    document.getElementById('btn-random').addEventListener('click', () => {
      this.overrides = {};
      this.loadBracket('probabilistic');
    });
    document.getElementById('btn-reset').addEventListener('click', () => {
      this.overrides = {};
      this.loadBracket();
    });

    // Matchup explorer
    document.getElementById('btn-explore').addEventListener('click', () => this.runExplorer());
  }

  /* ── Data fetching ───────────────────────────────────────────────── */
  async loadBracket(mode = 'deterministic') {
    this.showLoading(true, 'Simulating bracket…');
    try {
      this.data = await API.getBracket(this.gender, this.season, mode);
      this.buildIndices();
      this.render();
    } catch (e) {
      console.error('loadBracket failed:', e);
      this.showError(e.message);
    }
    this.showLoading(false);
  }

  async simulate() {
    this.showLoading(true, 'Updating bracket…');
    try {
      this.data = await API.simulate(this.gender, this.season, 'deterministic', this.overrides);
      this.buildIndices();
      this.render();
    } catch (e) {
      console.error('simulate failed:', e);
      this.showError(e.message);
    }
    this.showLoading(false);
  }

  /* ── Index builders ──────────────────────────────────────────────── */
  buildIndices() {
    this.teamIdx = {};
    for (const t of this.data.teams) this.teamIdx[t.team_id] = t;
    this.gameIdx = {};
    for (const g of this.data.games) this.gameIdx[g.slot] = g;
  }

  seedNum(teamId) {
    return this.teamIdx[teamId]?.seed_num ?? null;
  }

  /* ── Click handler ───────────────────────────────────────────────── */
  onTeamClick(slot, teamId) {
    const game = this.gameIdx[slot];
    if (!game || !game.strong_id || !game.weak_id) return;

    // Determine the loser
    const loserId = teamId === game.strong_id ? game.weak_id : game.strong_id;

    // Set override
    this.overrides[slot] = teamId;

    // Cascade: remove loser from any later-round override
    if (loserId) {
      for (const [s, tid] of Object.entries(this.overrides)) {
        if (s !== slot && tid === loserId) delete this.overrides[s];
      }
    }

    this.simulate();
  }

  /* ═══════════════════════════════════════════════════════════════════
     Rendering
     ═══════════════════════════════════════════════════════════════════ */

  render() {
    const vp = document.getElementById('bracket-viewport');
    vp.innerHTML = '';

    const svg = svgEl('svg', {
      width: C.SVG_W, height: C.SVG_H,
      viewBox: `0 0 ${C.SVG_W} ${C.SVG_H}`,
      class: 'bracket-svg',
    });

    this.addRoundHeaders(svg);

    // Regional rounds (R1–R4)
    for (const [reg, cfg] of Object.entries(REGION_CFG)) {
      this.addRegion(svg, reg, cfg);
    }

    // Final Four + Championship
    this.addFinalRounds(svg);

    vp.appendChild(svg);

    this.updateChampion();
    this.updateStats();
    this.populateExplorerTeams();
  }

  /* ── Round headers ───────────────────────────────────────────────── */
  addRoundHeaders(svg) {
    for (let col = 0; col < C.COLS; col++) {
      const x = colX(col) + C.BOX_W / 2;
      const y = C.MARGIN + 10;
      const txt = svgText(x, y, ROUND_HEADERS[col], 'round-header');
      svg.appendChild(txt);
    }
  }

  /* ── One region (R1–R4) ──────────────────────────────────────────── */
  addRegion(svg, region, cfg) {
    const { side, pos } = cfg;
    const rY = regionY(pos);
    const color = REGION_COLORS[region];
    const regionName = this.data.regions[region] || region;

    // Region label
    const labelX = side === 'left'
      ? colX(colForRound(1, side)) + 2
      : colX(colForRound(1, side)) + C.BOX_W - 2;
    const labelY = rY - 6;
    const lbl = svgText(labelX, labelY, regionName, 'region-label', {
      fill: color,
      'text-anchor': side === 'left' ? 'start' : 'end',
    });
    svg.appendChild(lbl);

    // R1–R4 games + connectors
    for (let round = 1; round <= 4; round++) {
      const gamesInRound = 8 / Math.pow(2, round - 1); // 8,4,2,1
      const col = colForRound(round, side);
      const x = colX(col);

      for (let i = 0; i < gamesInRound; i++) {
        const slot = `R${round}${region}${i + 1}`;
        const game = this.gameIdx[slot];
        if (!game) continue;

        const cy = gameCenterY(round, i, rY);
        const y = cy - C.GAME_H / 2;
        this.addGameBox(svg, x, y, game, side, color);
      }

      // Connectors from this round to next (except R4 — handled in final rounds)
      if (round < 4) {
        const nextCol = colForRound(round + 1, side);
        for (let i = 0; i < gamesInRound; i += 2) {
          const cy0 = gameCenterY(round, i,     rY);
          const cy1 = gameCenterY(round, i + 1, rY);
          this.addConnector(svg, col, nextCol, cy0, cy1, side);
        }
      }
    }

    // Connector from R4 → Final Four slot
    const r4cy = gameCenterY(4, 0, rY);
    // Will be connected in addFinalRounds
    this._r4Centers = this._r4Centers || {};
    this._r4Centers[region] = r4cy;
  }

  /* ── Final Four & Championship ───────────────────────────────────── */
  addFinalRounds(svg) {
    const topY  = regionY('top');
    const botY  = regionY('bottom');

    // R5WX (Final Four left, col 4)
    const ffLeftCY = (this._r4Centers['W'] + this._r4Centers['X']) / 2;
    const ffLeftGame = this.gameIdx['R5WX'];
    if (ffLeftGame) {
      const x = colX(4);
      const y = ffLeftCY - C.GAME_H / 2;
      this.addGameBox(svg, x, y, ffLeftGame, 'left', '#9b59b6');

      // Connector: R4W1 → R5WX
      this.addConnector(svg, 3, 4, this._r4Centers['W'], this._r4Centers['X'], 'left');
    }

    // R5YZ (Final Four right, col 6)
    const ffRightCY = (this._r4Centers['Y'] + this._r4Centers['Z']) / 2;
    const ffRightGame = this.gameIdx['R5YZ'];
    if (ffRightGame) {
      const x = colX(6);
      const y = ffRightCY - C.GAME_H / 2;
      this.addGameBox(svg, x, y, ffRightGame, 'right', '#9b59b6');

      // Connector: R4Y1 → R5YZ
      this.addConnector(svg, 7, 6, this._r4Centers['Y'], this._r4Centers['Z'], 'right');
    }

    // R6CH (Championship, col 5)
    const champCY = (ffLeftCY + ffRightCY) / 2;
    const champGame = this.gameIdx['R6CH'];
    if (champGame) {
      const x = colX(5);
      const y = champCY - C.GAME_H / 2;
      this.addGameBox(svg, x, y, champGame, 'center', '#ff6b35');

      // Connectors: R5WX → R6CH (left) and R5YZ → R6CH (right)
      // Left connector (single line)
      const exitL = colX(4) + C.BOX_W;
      const entryL = colX(5);
      svg.appendChild(svgEl('line', {
        x1: exitL, y1: ffLeftCY, x2: entryL, y2: champCY,
        class: 'connector', stroke: '#363a4a',
      }));

      // Right connector (single line)
      const exitR = colX(6);
      const entryR = colX(5) + C.BOX_W;
      svg.appendChild(svgEl('line', {
        x1: exitR, y1: ffRightCY, x2: entryR, y2: champCY,
        class: 'connector', stroke: '#363a4a',
      }));
    }

    // Champion crown above the championship game
    if (this.data.champion_id) {
      const crownY = champCY - C.GAME_H / 2 - 22;
      svg.appendChild(svgText(colX(5) + C.BOX_W / 2, crownY, '🏆', 'champion-crown'));
      svg.appendChild(svgText(colX(5) + C.BOX_W / 2, crownY - 14,
        this.data.champion_name, 'champion-name-svg'));
    }

    delete this._r4Centers;
  }

  /* ── Single game box (two teams stacked) ─────────────────────────── */
  addGameBox(svg, x, y, game, side, regionColor) {
    const g = svgEl('g', { class: 'game-box', 'data-slot': game.slot });

    // Background rects
    const bgColor = '#1a1d28';
    const borderColor = '#2a2d3a';

    // Top team (strong)
    this._addTeamRow(g, x, y, game, 'strong', game.strong_id, game.strong_name,
      game.strong_seed_num, game.probability, regionColor, bgColor, borderColor, side);

    // Divider
    g.appendChild(svgEl('line', {
      x1: x, y1: y + C.BOX_H, x2: x + C.BOX_W, y2: y + C.BOX_H,
      stroke: borderColor, 'stroke-width': 0.5,
    }));

    // Bottom team (weak)
    const weakProb = game.probability != null ? 1 - game.probability : null;
    this._addTeamRow(g, x, y + C.BOX_H, game, 'weak', game.weak_id, game.weak_name,
      game.weak_seed_num, weakProb, regionColor, bgColor, borderColor, side);

    // Outer border
    g.appendChild(svgEl('rect', {
      x, y, width: C.BOX_W, height: C.GAME_H,
      fill: 'none', stroke: borderColor, rx: 3,
    }));

    svg.appendChild(g);
  }

  /** Render one team row inside a game box */
  _addTeamRow(g, x, y, game, role, teamId, teamName, seedNum, prob, regionColor, bgColor, borderColor, side) {
    const isWinner = game.winner_id != null && game.winner_id === teamId;
    const isLoser  = game.winner_id != null && game.winner_id !== teamId;
    const hasTeam  = teamId != null;

    // Row background
    const rowBg = isWinner ? 'rgba(46,204,113,0.12)' : bgColor;
    const row = svgEl('rect', {
      x, y, width: C.BOX_W, height: C.BOX_H,
      fill: rowBg, rx: role === 'strong' ? 3 : 0,
    });
    // Click area needs to be on top, but let's make the group clickable
    const rowGroup = svgEl('g', {
      class: `team-box ${isWinner ? 'winner' : ''} ${isLoser ? 'loser' : ''} ${hasTeam ? 'pickable' : ''}`,
      'data-slot': game.slot,
      'data-team': teamId,
      style: hasTeam ? 'cursor:pointer' : '',
    });

    rowGroup.appendChild(row);

    if (hasTeam) {
      // Winner left-edge indicator
      if (isWinner) {
        rowGroup.appendChild(svgEl('rect', {
          x, y, width: 3, height: C.BOX_H, fill: '#2ecc71', rx: 1,
        }));
      }

      // Seed badge
      const seedTxt = seedNum != null ? String(seedNum) : '';
      if (seedTxt) {
        const seedEl = svgText(x + 5, y + 14, seedTxt, 'seed-badge');
        if (isLoser) seedEl.setAttribute('opacity', '0.4');
        rowGroup.appendChild(seedEl);
      }

      // Team name (truncate if needed)
      const nameX = x + (seedTxt ? 20 : 6);
      const maxChars = seedTxt ? 11 : 14;
      const dispName = teamName.length > maxChars ? teamName.slice(0, maxChars - 1) + '…' : teamName;
      const nameEl = svgText(nameX, y + 14, dispName,
        `team-name ${isWinner ? 'winner' : ''} ${!hasTeam ? 'tbd' : ''}`);
      if (isLoser) nameEl.setAttribute('opacity', '0.4');
      rowGroup.appendChild(nameEl);

      // Probability
      if (prob != null) {
        const pctStr = `${Math.round(prob * 100)}%`;
        const probEl = svgText(x + C.BOX_W - 5, y + 14, pctStr, 'prob-text', {
          'text-anchor': 'end',
        });
        if (isLoser) probEl.setAttribute('opacity', '0.35');
        rowGroup.appendChild(probEl);
      }

      // Click handler
      rowGroup.addEventListener('click', () => this.onTeamClick(game.slot, teamId));

      // Hover tooltip
      rowGroup.addEventListener('mouseenter', (e) => {
        const seed = this.teamIdx[teamId];
        this.showTooltip(e, teamName, seedNum, prob);
      });
      rowGroup.addEventListener('mouseleave', () => this.hideTooltip());
    } else {
      // TBD placeholder
      const tbd = svgText(x + C.BOX_W / 2, y + 14, 'TBD', 'team-name tbd', {
        'text-anchor': 'middle',
      });
      rowGroup.appendChild(tbd);
    }

    g.appendChild(rowGroup);
  }

  /* ── Bracket connectors ──────────────────────────────────────────── */
  addConnector(svg, fromCol, toCol, cy0, cy1, side) {
    let exitX, entryX, midX;

    if (side === 'left') {
      exitX  = colX(fromCol) + C.BOX_W;
      entryX = colX(toCol);
      midX   = exitX + C.CONN_W / 2;
    } else {
      exitX  = colX(fromCol);
      entryX = colX(toCol) + C.BOX_W;
      midX   = entryX + C.CONN_W / 2;
    }

    const midY = (cy0 + cy1) / 2;
    const color = '#363a4a';

    // Top feeder → mid → across
    svg.appendChild(svgEl('path', {
      d: `M ${exitX} ${cy0} H ${midX} V ${midY} H ${entryX}`,
      fill: 'none', stroke: color, class: 'connector',
    }));

    // Bottom feeder → mid
    svg.appendChild(svgEl('path', {
      d: `M ${exitX} ${cy1} H ${midX} V ${midY}`,
      fill: 'none', stroke: color, class: 'connector',
    }));
  }

  /* ═══════════════════════════════════════════════════════════════════
     UI updates
     ═══════════════════════════════════════════════════════════════════ */

  updateChampion() {
    const el = document.getElementById('champion-name');
    const wrap = document.getElementById('champion-display');
    if (this.data.champion_name) {
      el.textContent = this.data.champion_name;
      wrap.classList.remove('empty');
    } else {
      el.textContent = 'Pick a champion';
      wrap.classList.add('empty');
    }

    // Header title
    document.getElementById('header-title').textContent =
      `${this.data.gender_label} ${this.data.season} NCAA Tournament`;
  }

  updateStats() {
    const games = this.data.games.filter(g => g.round_name !== 'Play-In');
    const decided = games.filter(g => g.winner_id != null).length;
    const upsets = games.filter(g => {
      if (!g.winner_id || !g.strong_seed_num || !g.weak_seed_num) return false;
      // An upset is when the weaker seed (higher number) wins
      return g.winner_id === g.weak_id && g.weak_seed_num > g.strong_seed_num;
    }).length;

    document.getElementById('stat-decided').textContent = `${decided}/${games.length}`;
    document.getElementById('stat-upsets').textContent = String(upsets);
    document.getElementById('stat-overrides').textContent = String(Object.keys(this.overrides).length);
  }

  /* ── Matchup Explorer ────────────────────────────────────────────── */
  populateExplorerTeams() {
    const selA = document.getElementById('explore-team-a');
    const selB = document.getElementById('explore-team-b');
    selA.innerHTML = '<option value="">Select team…</option>';
    selB.innerHTML = '<option value="">Select team…</option>';

    const sorted = [...this.data.teams].sort((a, b) => a.seed_num - b.seed_num);
    for (const t of sorted) {
      const opt = `<option value="${t.team_id}">(${t.seed_num}) ${t.team_name}</option>`;
      selA.innerHTML += opt;
      selB.innerHTML += opt;
    }
  }

  async runExplorer() {
    const aId = parseInt(document.getElementById('explore-team-a').value);
    const bId = parseInt(document.getElementById('explore-team-b').value);
    if (!aId || !bId || aId === bId) return;

    try {
      const res = await API.predict(aId, bId, this.season, this.gender);
      const wrap = document.getElementById('explorer-result');
      wrap.classList.remove('hidden');

      document.getElementById('explore-name-a').textContent = res.team1_name;
      document.getElementById('explore-name-b').textContent = res.team2_name;

      const pA = res.probability;
      const pB = 1 - pA;

      const elA = document.getElementById('explore-prob-a');
      const elB = document.getElementById('explore-prob-b');

      elA.textContent = `${Math.round(pA * 100)}%`;
      elB.textContent = `${Math.round(pB * 100)}%`;

      elA.className = `prob ${pA >= pB ? 'favored' : 'underdog'}`;
      elB.className = `prob ${pB >= pA ? 'favored' : 'underdog'}`;

      document.getElementById('explore-bar-fill').style.width = `${Math.round(pA * 100)}%`;
    } catch (e) {
      console.error('Explorer error:', e);
    }
  }

  /* ── Tooltip ─────────────────────────────────────────────────────── */
  showTooltip(e, name, seed, prob) {
    const tt = document.getElementById('tooltip');
    tt.innerHTML = `
      <div class="tt-team">${name}</div>
      ${seed != null ? `<div class="tt-label">Seed: ${seed}</div>` : ''}
      ${prob != null ? `<div class="tt-prob">Win: ${Math.round(prob * 100)}%</div>` : ''}
    `;
    tt.style.left = (e.clientX + 12) + 'px';
    tt.style.top  = (e.clientY - 10) + 'px';
    tt.classList.add('show');
  }

  hideTooltip() {
    document.getElementById('tooltip').classList.remove('show');
  }

  /* ── Loading overlay ─────────────────────────────────────────────── */
  showLoading(show, msg = 'Loading…') {
    const ol = document.getElementById('loading-overlay');
    const txt = ol.querySelector('.loading-text');
    if (txt) txt.textContent = msg;
    ol.classList.toggle('hidden', !show);
  }

  showError(msg) {
    const vp = document.getElementById('bracket-viewport');
    vp.innerHTML = `<div style="text-align:center;padding:40px;color:#e74c3c;">
      <h3>Error</h3><p>${msg}</p>
      <p style="color:#8a8f9d;margin-top:8px;">Check that the server is running and models are trained.</p>
    </div>`;
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   Boot
   ═══════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  window.app = new BracketApp();
  window.app.init();
});
