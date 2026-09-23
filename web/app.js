"use strict";

// Coloring by Cooling: simulated annealing for map coloring and Sudoku.
// A JavaScript port of graph_coloring.annealing and graph_coloring.apps.sudoku.

const $ = (sel) => document.querySelector(sel);
const rand = Math.random;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---------------------------------------------------------------------------
// Map coloring annealer: recolor one conflicting node at a time, Metropolis
// acceptance, conflicting nodes kept in an indexed set (O(1) updates).
// ---------------------------------------------------------------------------

class MapAnnealer {
  constructor(adj, k) {
    this.adj = adj;
    this.n = adj.length;
    this.k = k;
    this.randomize();
  }

  randomize() {
    const { n, k } = this;
    this.col = new Int32Array(n);
    for (let v = 0; v < n; v++) this.col[v] = Math.floor(rand() * k);
    this.conf = new Int32Array(n);
    this.items = [];
    this.pos = new Int32Array(n).fill(-1);
    let twice = 0;
    for (let v = 0; v < n; v++) {
      for (const w of this.adj[v]) if (this.col[w] === this.col[v]) this.conf[v]++;
      twice += this.conf[v];
      if (this.conf[v]) this.add(v);
    }
    this.cost = twice / 2;
    this.best = this.cost;
    this.moves = 0;
  }

  add(v) {
    if (this.pos[v] < 0) {
      this.pos[v] = this.items.length;
      this.items.push(v);
    }
  }

  remove(v) {
    const i = this.pos[v];
    if (i >= 0) {
      const last = this.items.pop();
      if (last !== v) {
        this.items[i] = last;
        this.pos[last] = i;
      }
      this.pos[v] = -1;
    }
  }

  step(T) {
    this.moves++;
    if (this.cost === 0 || this.k < 2 || this.n === 0) return;
    const { adj, col, conf, items } = this;
    const v = items.length ? items[Math.floor(rand() * items.length)] : Math.floor(rand() * this.n);
    const old = col[v];
    let nw = Math.floor(rand() * (this.k - 1));
    if (nw >= old) nw++;
    let delta = 0;
    for (const w of adj[v]) {
      if (col[w] === nw) delta++;
      else if (col[w] === old) delta--;
    }
    if (delta > 0 && rand() >= Math.exp(-delta / T)) return;
    col[v] = nw;
    conf[v] = 0;
    for (const w of adj[v]) {
      if (col[w] === old) {
        if (--conf[w] === 0) this.remove(w);
      } else if (col[w] === nw) {
        conf[w]++;
        conf[v]++;
        this.add(w);
      }
    }
    if (conf[v]) this.add(v);
    else this.remove(v);
    this.cost += delta;
    if (this.cost < this.best) this.best = this.cost;
  }
}

// ---------------------------------------------------------------------------
// Sudoku: the Sudoku graph, constraint propagation and the box-swap annealer.
// ---------------------------------------------------------------------------

const PEERS = [];
for (let i = 0; i < 81; i++) {
  const r = Math.floor(i / 9), c = i % 9, br = r - (r % 3), bc = c - (c % 3);
  const set = new Set();
  for (let j = 0; j < 9; j++) {
    set.add(r * 9 + j);
    set.add(j * 9 + c);
  }
  for (let dr = 0; dr < 3; dr++) for (let dc = 0; dc < 3; dc++) set.add((br + dr) * 9 + bc + dc);
  set.delete(i);
  PEERS.push([...set]);
}
const UNITS = [];
for (let r = 0; r < 9; r++) UNITS.push([...Array(9)].map((_, c) => r * 9 + c));
for (let c = 0; c < 9; c++) UNITS.push([...Array(9)].map((_, r) => r * 9 + c));
for (let b = 0; b < 9; b++) {
  const br = 3 * Math.floor(b / 3), bc = 3 * (b % 3);
  UNITS.push([...Array(9)].map((_, j) => (br + Math.floor(j / 3)) * 9 + bc + (j % 3)));
}
const BOXES = UNITS.slice(18);
const ALL = 0x3fe; // bits 1..9

function candidates(grid, i) {
  let used = 0;
  for (const p of PEERS[i]) used |= 1 << grid[p];
  return ALL & ~used;
}

const popcount = (x) => {
  let n = 0;
  while (x) { x &= x - 1; n++; }
  return n;
};

function clueClash(grid) {
  const names = ["row", "column", "box"];
  for (let u = 0; u < 27; u++) {
    const seen = new Map();
    for (const i of UNITS[u]) {
      const d = grid[i];
      if (!d) continue;
      if (seen.has(d)) return `The clues clash: ${names[Math.floor(u / 9)]} ${(u % 9) + 1} has two ${d}s.`;
      seen.set(d, i);
    }
  }
  return null;
}

// Naked and hidden singles. Returns a new grid or throws with a reason.
function propagate(start) {
  const grid = Int8Array.from(start);
  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 0; i < 81; i++) {
      if (grid[i]) continue;
      const cand = candidates(grid, i);
      if (!cand) throw new Error(`Cell ${Math.floor(i / 9) + 1},${(i % 9) + 1} has no digit left, so this puzzle has no solution.`);
      if (popcount(cand) === 1) {
        grid[i] = 31 - Math.clz32(cand);
        changed = true;
      }
    }
    for (const unit of UNITS) {
      for (let d = 1; d <= 9; d++) {
        let spot = -1, count = 0, present = false;
        for (const i of unit) {
          if (grid[i] === d) { present = true; break; }
          if (!grid[i] && candidates(grid, i) & (1 << d)) { spot = i; count++; }
        }
        if (present) continue;
        if (count === 0) throw new Error(`Digit ${d} has nowhere to go, so this puzzle has no solution.`);
        if (count === 1) { grid[spot] = d; changed = true; }
      }
    }
  }
  return grid;
}

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

class SudokuAnnealer {
  constructor(start) {
    this.start = Int8Array.from(start);
    this.cand = new Int32Array(81);
    for (let i = 0; i < 81; i++) this.cand[i] = start[i] ? 0 : candidates(start, i);
    this.freeBoxes = BOXES.map((box) => box.filter((i) => !start[i])).filter((f) => f.length >= 2);
    this.moves = 0;
    this.restarts = 0;
    this.best = Infinity;
    this.restart();
  }

  restart() {
    const g = (this.grid = Int8Array.from(this.start));
    for (const box of BOXES) {
      const free = box.filter((i) => !this.start[i]);
      const present = new Set(box.map((i) => this.start[i]));
      const missing = [1, 2, 3, 4, 5, 6, 7, 8, 9].filter((d) => !present.has(d));
      const fill = this.matching(shuffle(free.slice()), shuffle(missing)) ||
        Object.fromEntries(free.map((c, j) => [c, missing[j]]));
      for (const c of free) g[c] = fill[c];
    }
    this.rowCnt = new Int32Array(90);
    this.colCnt = new Int32Array(90);
    for (let i = 0; i < 81; i++) {
      this.rowCnt[Math.floor(i / 9) * 10 + g[i]]++;
      this.colCnt[(i % 9) * 10 + g[i]]++;
    }
    let cost = 0;
    for (let i = 0; i < 90; i++) {
      cost += (this.rowCnt[i] * (this.rowCnt[i] - 1)) / 2 + (this.colCnt[i] * (this.colCnt[i] - 1)) / 2;
    }
    this.cost = cost;
    this.best = Math.min(this.best, cost);
    this.runMoves = 0;
  }

  // Assign digits to cells bijectively, each digit a candidate of its cell.
  matching(cells, digits) {
    const out = {};
    const used = new Set();
    const go = (j) => {
      if (j === cells.length) return true;
      for (const d of digits) {
        if (used.has(d) || !(this.cand[cells[j]] & (1 << d))) continue;
        used.add(d);
        out[cells[j]] = d;
        if (go(j + 1)) return true;
        used.delete(d);
      }
      return false;
    };
    return go(0) ? out : null;
  }

  step(T) {
    this.moves++;
    if (this.cost === 0 || !this.freeBoxes.length) return;
    if (++this.runMoves > 1_000_000) {
      this.restarts++;
      this.restart();
      return;
    }
    const box = this.freeBoxes[Math.floor(rand() * this.freeBoxes.length)];
    const i = Math.floor(rand() * box.length);
    let j = Math.floor(rand() * (box.length - 1));
    if (j >= i) j++;
    const a = box[i], b = box[j], g = this.grid;
    const d1 = g[a], d2 = g[b];
    if (!(this.cand[a] & (1 << d2)) || !(this.cand[b] & (1 << d1))) return;
    const r1 = Math.floor(a / 9), c1 = a % 9, r2 = Math.floor(b / 9), c2 = b % 9;
    const R = this.rowCnt, C = this.colCnt;
    let delta = 0;
    if (r1 !== r2) {
      delta += R[r1 * 10 + d2] - (R[r1 * 10 + d1] - 1) + R[r2 * 10 + d1] - (R[r2 * 10 + d2] - 1);
    }
    if (c1 !== c2) {
      delta += C[c1 * 10 + d2] - (C[c1 * 10 + d1] - 1) + C[c2 * 10 + d1] - (C[c2 * 10 + d2] - 1);
    }
    if (delta > 0 && rand() >= Math.exp(-delta / T)) return;
    g[a] = d2;
    g[b] = d1;
    if (r1 !== r2) {
      R[r1 * 10 + d1]--; R[r1 * 10 + d2]++; R[r2 * 10 + d2]--; R[r2 * 10 + d1]++;
    }
    if (c1 !== c2) {
      C[c1 * 10 + d1]--; C[c1 * 10 + d2]++; C[c2 * 10 + d2]--; C[c2 * 10 + d1]++;
    }
    this.cost += delta;
    if (this.cost < this.best) this.best = this.cost;
  }

  clashes(i) {
    const d = this.grid[i];
    return this.rowCnt[Math.floor(i / 9) * 10 + d] > 1 || this.colCnt[(i % 9) * 10 + d] > 1;
  }
}

const PUZZLES = {
  easy: "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79",
  medium: "...26.7.168..7..9.19...45..82.1...4...46.29...5...3.28..93...74.4..5..367.3.18...",
  hard: "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..",
};

// ---------------------------------------------------------------------------
// Maps: loading, adjacency, projection and drawing.
// ---------------------------------------------------------------------------

const MAPS = {
  santiago_communes: {
    title: "Santiago communes",
    chi: 4,
    why: "Calera de Tango is ringed by five communes (Maipú, Padre Hurtado, Peñaflor, Talagante, San Bernardo). An odd ring already needs 3 colors, and Calera de Tango touches all five.",
    labels: false,
    zoom: [-70.86, -70.47, -33.66, -33.31],
  },
  chile_regions: {
    title: "Regions of Chile",
    chi: 3,
    why: "Valparaíso, Metropolitana and O'Higgins all border each other.",
    labels: true,
  },
  south_america: {
    title: "South America",
    chi: 4,
    why: "Argentina, Bolivia, Brazil and Paraguay all border each other.",
    labels: true,
  },
  us_states: {
    title: "Contiguous United States",
    chi: 4,
    why: "Nevada is ringed by five states (California, Oregon, Idaho, Utah, Arizona), an odd ring it touches entirely.",
    labels: false,
  },
};

const NAME_KEYS = ["name", "NAME", "Name", "nombre", "NOMBRE", "Comuna", "NOM_COM", "Region", "admin", "ADMIN"];

function rings(geometry) {
  if (!geometry) return [];
  if (geometry.type === "Polygon") return geometry.coordinates;
  if (geometry.type === "MultiPolygon") return geometry.coordinates.flat();
  return [];
}

function parseGeoJSON(data) {
  const features = (data.features || []).filter((f) => rings(f.geometry).length);
  if (!features.length) throw new Error("No polygons found. The file needs Polygon or MultiPolygon features.");
  if (features.length > 1500) throw new Error(`That map has ${features.length} regions; the limit here is 1500.`);
  const seen = new Map();
  const names = features.map((f, i) => {
    const p = f.properties || {};
    const key = NAME_KEYS.find((k) => typeof p[k] === "string" && p[k].trim());
    let name = key ? p[key].trim() : `Region ${i + 1}`;
    const count = (seen.get(name) || 0) + 1;
    seen.set(name, count);
    if (count > 1) name = `${name} (${count})`;
    return name;
  });
  const index = new Map(names.map((n, i) => [n, i]));
  const adj = names.map(() => new Set());
  if (Array.isArray(data.adjacency)) {
    for (const [a, b] of data.adjacency) {
      const i = index.get(a), j = index.get(b);
      if (i !== undefined && j !== undefined && i !== j) { adj[i].add(j); adj[j].add(i); }
    }
  } else {
    sharedBorders(features, adj);
  }
  return { names, rings: features.map((f) => rings(f.geometry)), adj: adj.map((s) => [...s]) };
}

// Two regions are neighbours when they share a border segment (after
// rounding coordinates to about a metre).
function sharedBorders(features, adj) {
  const q = (x) => Math.round(x * 1e5);
  const owner = new Map();
  features.forEach((f, i) => {
    for (const ring of rings(f.geometry)) {
      for (let k = 0; k + 1 < ring.length; k++) {
        const a = `${q(ring[k][0])},${q(ring[k][1])}`, b = `${q(ring[k + 1][0])},${q(ring[k + 1][1])}`;
        if (a === b) continue;
        const key = a < b ? `${a}|${b}` : `${b}|${a}`;
        const j = owner.get(key);
        if (j === undefined) owner.set(key, i);
        else if (j !== i) { adj[i].add(j); adj[j].add(i); }
      }
    }
  });
}

function project(map) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const rs of map.rings) for (const r of rs) for (const [x, y] of r) {
    if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const kx = Math.cos((((minY + maxY) / 2) * Math.PI) / 180);
  const W = 1000, S = W / ((maxX - minX) * kx || 1);
  const H = (maxY - minY) * S || 1;
  const P = ([x, y]) => [(x - minX) * kx * S, (maxY - y) * S];
  return { P, W, H };
}

function ringArea(pts) {
  let a = 0;
  for (let i = 0; i + 1 < pts.length; i++) a += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1];
  return a / 2;
}

function labelPoint(pts) {
  const a = ringArea(pts);
  if (!a) return pts[0];
  let cx = 0, cy = 0;
  for (let i = 0; i + 1 < pts.length; i++) {
    const cr = pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1];
    cx += (pts[i][0] + pts[i + 1][0]) * cr;
    cy += (pts[i][1] + pts[i + 1][1]) * cr;
  }
  return [cx / (6 * a), cy / (6 * a)];
}

const SVGNS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}) => {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};

// ---------------------------------------------------------------------------
// App state and UI
// ---------------------------------------------------------------------------

const state = {
  mode: "maps",
  running: false,
  mapKey: "santiago_communes",
  map: null, // {names, rings, adj, meta}
  k: 4,
  engine: null,
  sched: { t: 0, L: 1000, reheats: 0 },
  history: [],
  zoomed: false,
  sudokuGivens: new Int8Array(81),
  sudokuEngine: null,
  sudokuFixed: null,
  speed: { maps: 22, sudoku: 72 },
  error: null,
};

const ui = {
  svg: $("#map"),
  hNow: $("#h-now"),
  tNow: $("#t-now"),
  tMarker: $("#t-marker"),
  status: $("#status"),
  moves: $("#moves"),
  restarts: $("#restarts"),
  restartsLabel: $("#restarts-label"),
  best: $("#best"),
  run: $("#run"),
  reset: $("#reset"),
  speed: $("#speed"),
  speedOut: $("#speed-out"),
  autoT: $("#auto-t"),
  temp: $("#temp"),
  tempOut: $("#temp-out"),
  spark: $("#spark"),
  mapNote: $("#map-note"),
  mapMeta: $("#map-meta"),
  mapTitle: $("#map-title"),
  labels: $("#labels"),
  zoom: $("#zoom"),
  sudoku: $("#sudoku"),
  sudokuNote: $("#sudoku-note"),
};

const T_MIN = 0.02, T_MAX = 2;
const sliderToT = (v) => T_MIN * Math.pow(T_MAX / T_MIN, v / 100);
const tToSlider = (T) => (100 * Math.log(T / T_MIN)) / Math.log(T_MAX / T_MIN);
const MAX_MOVES = { maps: 3000, sudoku: 60000 };
const movesPerFrame = () => Math.max(1, Math.round(Math.pow(MAX_MOVES[state.mode], ui.speed.value / 100)));
const fmt = (x) => x.toLocaleString("en-US");

function setStatus(kind, text) {
  ui.status.dataset.state = kind;
  ui.status.textContent = text;
}

function currentT() {
  if (!ui.autoT.checked) return sliderToT(Number(ui.temp.value));
  if (state.mode === "sudoku") return 0.5;
  const { t, L } = state.sched;
  return 0.8 * Math.pow(0.05 / 0.8, Math.min(t / L, 1));
}

// ---- maps ----------------------------------------------------------------

async function loadMap(key) {
  stop();
  $("#map-loading").hidden = false;
  try {
    const res = await fetch(`data/${key}.geojson`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const map = parseGeoJSON(await res.json());
    map.meta = MAPS[key];
    useMap(key, map);
  } catch (err) {
    ui.mapNote.textContent = `The map could not be loaded (${err.message}).`;
    ui.mapNote.classList.add("warn");
  } finally {
    $("#map-loading").hidden = true;
  }
}

function useMap(key, map) {
  state.mapKey = key;
  state.map = map;
  state.zoomed = false;
  ui.labels.checked = map.meta ? map.meta.labels : map.names.length <= 30;
  ui.zoom.hidden = !(map.meta && map.meta.zoom);
  ui.zoom.textContent = "Zoom to Greater Santiago";
  ui.mapTitle.textContent = map.meta ? map.meta.title : "Your map";
  const edges = map.adj.reduce((s, a) => s + a.length, 0) / 2;
  ui.mapMeta.textContent = `${map.names.length} regions · ${edges} borders`;
  drawMap();
  newMapEngine();
}

function drawMap() {
  const { svg } = ui;
  const map = state.map;
  svg.replaceChildren();
  const { P, W, H } = project(map);
  map.W = W;
  map.H = H;
  map.P = P;
  map.points = [];
  const regions = el("g");
  map.paths = map.rings.map((rs, i) => {
    let best = null, bestArea = -1;
    const d = rs.map((r) => {
      const pts = r.map(P);
      const a = Math.abs(ringArea(pts));
      if (a > bestArea) { bestArea = a; best = pts; }
      return "M" + pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("L") + "Z";
    }).join("");
    map.points.push(labelPoint(best));
    const path = el("path", { d, "fill-rule": "evenodd" });
    const title = el("title");
    title.textContent = `${map.names[i]} · ${map.adj[i].length} neighbours`;
    path.appendChild(title);
    regions.appendChild(path);
    return path;
  });
  map.clashLayer = el("g");
  map.labelLayer = el("g");
  map.names.forEach((name, i) => {
    const t = el("text", { x: map.points[i][0], y: map.points[i][1], "text-anchor": "middle", "dominant-baseline": "middle" });
    t.textContent = name;
    map.labelLayer.appendChild(t);
  });
  svg.append(regions, map.clashLayer, map.labelLayer);
  setView();
}

function setView() {
  const map = state.map;
  let x = 0, y = 0, w = map.W, h = map.H;
  if (state.zoomed && map.meta && map.meta.zoom) {
    const [x0, x1, y0, y1] = map.meta.zoom;
    const a = map.P([x0, y1]), b = map.P([x1, y0]);
    [x, y, w, h] = [a[0], a[1], b[0] - a[0], b[1] - a[1]];
  }
  const pad = w * 0.02;
  ui.svg.setAttribute("viewBox", `${x - pad} ${y - pad} ${w + 2 * pad} ${h + 2 * pad}`);
  // keep labels the same size on screen whatever the zoom
  const fs = (w / 1000) * (state.zoomed ? 19 : 12);
  map.labelLayer.style.fontSize = `${fs.toFixed(2)}px`;
  map.labelLayer.style.strokeWidth = `${(fs / 4).toFixed(2)}px`;
  map.labelLayer.style.display = ui.labels.checked ? "" : "none";
}

function newMapEngine() {
  state.engine = new MapAnnealer(state.map.adj, state.k);
  state.sched = { t: 0, L: Math.max(400, 30 * state.map.names.length), reheats: 0 };
  state.history = [];
  state.map.lastCol = new Int32Array(state.map.names.length).fill(-1);
  explainK();
  renderMap();
  setStatus("idle", "Ready");
}

function explainK() {
  const meta = state.map.meta;
  ui.mapNote.classList.remove("warn");
  if (!meta) {
    ui.mapNote.textContent = "Nobody has told this page the chromatic number of your map. Lower k until the search gets stuck.";
  } else if (state.k < meta.chi) {
    ui.mapNote.classList.add("warn");
    ui.mapNote.innerHTML = `<strong>${state.k} colors can never work here.</strong> ${meta.why} Watch <span class="mono">H</span> refuse to reach 0.`;
  } else {
    ui.mapNote.innerHTML = `This map needs exactly <strong>${meta.chi} colors</strong>, a result proven by exact search in the Python library. ${meta.why}`;
  }
}

function renderMap() {
  const map = state.map, eng = state.engine;
  if (!map || !eng) return;
  for (let i = 0; i < map.paths.length; i++) {
    if (map.lastCol[i] !== eng.col[i]) {
      map.paths[i].style.fill = `var(--c${eng.col[i]})`;
      map.lastCol[i] = eng.col[i];
    }
  }
  map.clashLayer.replaceChildren();
  for (let v = 0; v < map.adj.length; v++) {
    if (!eng.conf[v]) continue;
    for (const w of map.adj[v]) {
      if (w > v && eng.col[w] === eng.col[v]) {
        const [x1, y1] = map.points[v], [x2, y2] = map.points[w];
        map.clashLayer.append(
          el("line", { x1, y1, x2, y2, class: "clash-halo" }),
          el("line", { x1, y1, x2, y2, class: "clash" }),
        );
      }
    }
  }
  if (state.mode === "maps") renderReadout(eng.cost, eng.best, eng.moves, state.sched.reheats);
}

// ---- sudoku ----------------------------------------------------------------

const cells = [];
for (let i = 0; i < 81; i++) {
  const c = document.createElement("div");
  c.className = "cell";
  c.tabIndex = 0;
  c.setAttribute("role", "gridcell");
  c.dataset.i = i;
  ui.sudoku.appendChild(c);
  cells.push(c);
}

function loadPuzzle(str) {
  stop();
  const digits = [...str].filter((ch) => /[0-9.]/.test(ch)).map((ch) => (ch === "." ? 0 : Number(ch)));
  if (digits.length !== 81) {
    ui.sudokuNote.textContent = `A puzzle needs 81 cells; that text has ${digits.length}.`;
    ui.sudokuNote.classList.add("warn");
    return;
  }
  state.sudokuGivens = Int8Array.from(digits);
  resetSudoku();
}

function resetSudoku() {
  stop();
  state.sudokuEngine = null;
  state.history = [];
  ui.sudokuNote.classList.remove("warn");
  const clues = state.sudokuGivens.reduce((s, d) => s + (d ? 1 : 0), 0);
  ui.sudokuNote.textContent = `${clues} clues. Click a cell and type 1–9 to edit, or paste an 81-character puzzle (. or 0 for blanks).`;
  renderSudoku();
  setStatus("idle", "Ready");
}

function prepareSudoku() {
  const clash = clueClash(state.sudokuGivens);
  if (clash) throw new Error(clash);
  let start = state.sudokuGivens;
  let forced = 0;
  if ($("#propagate").checked) {
    start = propagate(start);
    forced = start.reduce((s, d) => s + (d ? 1 : 0), 0) - state.sudokuGivens.reduce((s, d) => s + (d ? 1 : 0), 0);
  }
  state.sudokuEngine = new SudokuAnnealer(start);
  state.history = [];
  ui.sudokuNote.classList.remove("warn");
  ui.sudokuNote.textContent = forced
    ? `Constraint propagation filled ${forced} forced cells; annealing handles the rest.`
    : "Annealing from a random fill of every box.";
}

function renderSudoku() {
  const eng = state.sudokuEngine;
  const g = eng ? eng.grid : state.sudokuGivens;
  const solved = eng && eng.cost === 0;
  for (let i = 0; i < 81; i++) {
    const c = cells[i];
    const d = g[i];
    const txt = d ? String(d) : "";
    if (c.textContent !== txt) c.textContent = txt;
    c.classList.toggle("given", state.sudokuGivens[i] > 0);
    c.classList.toggle("clash", !!eng && !solved && eng.clashes(i));
    c.classList.toggle("solved", !!solved);
  }
  if (state.mode !== "sudoku") return;
  if (eng) renderReadout(eng.cost, eng.best, eng.moves, eng.restarts);
  else renderReadout(null, null, 0, 0);
}

// ---- shared readout --------------------------------------------------------

function renderReadout(cost, best, moves, restarts) {
  ui.hNow.textContent = cost === null ? "–" : String(cost);
  ui.best.textContent = best === null || best === Infinity ? "–" : String(best);
  ui.moves.textContent = fmt(moves);
  ui.restarts.textContent = String(restarts);
  const T = currentT();
  ui.tNow.textContent = T.toFixed(3);
  ui.tMarker.style.left = `calc(${Math.min(100, Math.max(0, tToSlider(T)))}% - 2px)`;
  drawSpark();
}

function drawSpark() {
  const cv = ui.spark;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const ink = css.getPropertyValue("--ink").trim(), rule = css.getPropertyValue("--rule").trim();
  const hist = state.history;
  ctx.strokeStyle = rule;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h - 0.5);
  ctx.lineTo(w, h - 0.5);
  ctx.stroke();
  if (hist.length < 2) return;
  const max = Math.max(1, ...hist);
  const x = (i) => (i / (hist.length - 1)) * (w - 6) + 1;
  const y = (v) => h - 4 - (v / max) * (h - 12);
  ctx.beginPath();
  hist.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.lineTo(x(hist.length - 1), h);
  ctx.lineTo(x(0), h);
  ctx.closePath();
  ctx.globalAlpha = 0.12;
  ctx.fillStyle = ink;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.beginPath();
  hist.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.strokeStyle = ink;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x(hist.length - 1), y(hist[hist.length - 1]), 3, 0, 2 * Math.PI);
  ctx.fillStyle = ink;
  ctx.fill();
  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.fillStyle = css.getPropertyValue("--muted").trim();
  ctx.fillText(`H over time · max ${max}`, 4, 11);
}

// ---- run loop ----------------------------------------------------------------

function start() {
  if (state.running) return;
  if (state.mode === "sudoku") {
    if (!state.sudokuEngine || state.sudokuEngine.cost === 0) {
      try {
        prepareSudoku();
      } catch (err) {
        ui.sudokuNote.textContent = err.message;
        ui.sudokuNote.classList.add("warn");
        setStatus("stuck", "No solution");
        return;
      }
    }
  } else if (!state.engine) return;
  else if (state.engine.cost === 0) newMapEngine();
  state.running = true;
  ui.run.textContent = "Pause";
  setStatus("running", "Annealing");
  requestAnimationFrame(frame);
}

function autoStart() {
  if (!reduceMotion && state.mode === "maps" && state.engine) start();
}

function stop() {
  state.running = false;
  ui.run.textContent = "Start";
}

function frame() {
  if (!state.running) return;
  const n = movesPerFrame();
  if (state.mode === "maps") {
    const eng = state.engine, s = state.sched;
    for (let i = 0; i < n && eng.cost > 0; i++) {
      eng.step(currentT());
      if (++s.t >= s.L && eng.cost > 0) { s.t = 0; s.reheats++; }
    }
    pushHistory(eng.cost);
    renderMap();
    if (eng.cost === 0) finish(`Solved: ${state.k} colors, no clashes`);
    else if (state.map.meta && state.k < state.map.meta.chi && s.reheats > 0) setStatus("stuck", "Stuck, as predicted");
  } else {
    const eng = state.sudokuEngine;
    const T = currentT();
    for (let i = 0; i < n && eng.cost > 0; i++) eng.step(T);
    pushHistory(eng.cost);
    renderSudoku();
    if (eng.cost === 0) {
      finish("Solved");
      ui.sudokuNote.textContent = `Solved in ${fmt(eng.moves)} moves${eng.restarts ? ` and ${eng.restarts} restart${eng.restarts > 1 ? "s" : ""}` : ""}.`;
    }
  }
  if (state.running) requestAnimationFrame(frame);
}

function pushHistory(cost) {
  state.history.push(cost);
  if (state.history.length > 480) state.history.splice(0, state.history.length - 480);
}

function finish(text) {
  stop();
  setStatus("solved", text);
  ui.run.textContent = "Run again";
}

// ---- controls ------------------------------------------------------------------

function setMode(mode) {
  stop();
  state.speed[state.mode] = Number(ui.speed.value);
  state.mode = mode;
  ui.speed.value = state.speed[mode];
  for (const m of ["maps", "sudoku"]) {
    $(`#tab-${m}`).setAttribute("aria-selected", String(m === mode));
    $(`#panel-${m}`).hidden = m !== mode;
    $(`#controls-${m}`).hidden = m !== mode;
  }
  ui.restartsLabel.textContent = mode === "maps" ? "reheats" : "restarts";
  $("label[for='auto-t']").lastChild.textContent = mode === "maps" ? " Automatic cooling" : " Automatic temperature (T = 0.5)";
  ui.reset.textContent = mode === "maps" ? "Shuffle" : "Reset";
  state.history = [];
  updateSpeed();
  if (mode === "maps") {
    if (state.engine) renderMap();
    setStatus(state.engine && state.engine.cost === 0 ? "solved" : "idle", state.engine && state.engine.cost === 0 ? "Solved" : "Ready");
  } else {
    renderSudoku();
    setStatus("idle", "Ready");
  }
}

function updateSpeed() {
  ui.speedOut.textContent = `${fmt(movesPerFrame())} moves/frame`;
}

function updateTemp() {
  ui.temp.disabled = ui.autoT.checked;
  ui.tempOut.textContent = sliderToT(Number(ui.temp.value)).toFixed(2);
  if (state.mode === "maps") renderMap();
  else renderSudoku();
}

$("#tab-maps").addEventListener("click", () => setMode("maps"));
$("#tab-sudoku").addEventListener("click", () => setMode("sudoku"));
$(".modes").addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const next = state.mode === "maps" ? "sudoku" : "maps";
    setMode(next);
    $(`#tab-${next}`).focus();
  }
});
ui.run.addEventListener("click", () => (state.running ? (stop(), setStatus("idle", "Paused")) : start()));
ui.reset.addEventListener("click", () => {
  if (state.mode === "maps") {
    stop();
    newMapEngine();
  } else {
    resetSudoku();
  }
});
ui.speed.addEventListener("input", updateSpeed);
ui.autoT.addEventListener("change", updateTemp);
ui.temp.addEventListener("input", updateTemp);

$("#map-select").addEventListener("change", (e) => {
  if (e.target.value !== "custom") loadMap(e.target.value).then(autoStart);
});
$("#k-group").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-k]");
  if (!b) return;
  for (const x of $("#k-group").children) x.setAttribute("aria-checked", String(x === b));
  state.k = Number(b.dataset.k);
  stop();
  if (state.map) {
    newMapEngine();
    autoStart();
  }
});
ui.labels.addEventListener("change", () => state.map && setView());
ui.zoom.addEventListener("click", () => {
  state.zoomed = !state.zoomed;
  ui.zoom.textContent = state.zoomed ? "Show the whole region" : "Zoom to Greater Santiago";
  setView();
  renderMap();
});
$("#upload").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const map = parseGeoJSON(JSON.parse(reader.result));
      map.meta = null;
      const opt = $("#map-select option[value='custom']");
      opt.hidden = false;
      opt.textContent = `${file.name} (${map.names.length})`;
      $("#map-select").value = "custom";
      useMap("custom", map);
      ui.mapTitle.textContent = file.name.replace(/\.(geo)?json$/i, "");
      autoStart();
    } catch (err) {
      ui.mapNote.textContent = `That file could not be used: ${err.message}`;
      ui.mapNote.classList.add("warn");
    }
  };
  reader.readAsText(file);
  e.target.value = "";
});

$("#presets").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-p]");
  if (b) loadPuzzle(PUZZLES[b.dataset.p]);
});
$("#clear").addEventListener("click", () => {
  state.sudokuGivens = new Int8Array(81);
  resetSudoku();
  cells[0].focus();
});
ui.sudoku.addEventListener("keydown", (e) => {
  const c = e.target.closest(".cell");
  if (!c) return;
  const i = Number(c.dataset.i);
  const moves = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -9, ArrowDown: 9 };
  if (e.key in moves) {
    const j = i + moves[e.key];
    if (j >= 0 && j < 81) cells[j].focus();
    e.preventDefault();
    return;
  }
  let d = null;
  if (/^[1-9]$/.test(e.key)) d = Number(e.key);
  else if (["Backspace", "Delete", "0", ".", " "].includes(e.key)) d = 0;
  if (d === null) return;
  e.preventDefault();
  state.sudokuGivens[i] = d;
  resetSudoku();
  const clash = clueClash(state.sudokuGivens);
  if (clash) {
    ui.sudokuNote.textContent = clash;
    ui.sudokuNote.classList.add("warn");
  }
  if (d && i < 80) cells[i + 1].focus();
  else cells[i].focus();
});
document.addEventListener("paste", (e) => {
  if (state.mode !== "sudoku") return;
  const text = (e.clipboardData || window.clipboardData).getData("text");
  if (text) {
    e.preventDefault();
    loadPuzzle(text);
  }
});
window.addEventListener("resize", () => drawSpark());

// ---- boot ----------------------------------------------------------------------

ui.speed.value = state.speed.maps;
ui.temp.value = Math.round(tToSlider(0.3));
updateSpeed();
ui.tempOut.textContent = sliderToT(Number(ui.temp.value)).toFixed(2);
state.sudokuGivens = Int8Array.from(PUZZLES.hard, (ch) => (ch === "." ? 0 : Number(ch)));
renderSudoku();
if (location.hash === "#sudoku") setMode("sudoku");
loadMap(state.mapKey).then(autoStart);
