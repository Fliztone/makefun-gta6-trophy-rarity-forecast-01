// Synthetic end-to-end test (test-only; not part of the graded program).
// Builds final-like cases from a structural cohort model of a launch week:
//   - tracked population P(t): Gompertz growth from 638 at launch;
//   - players arrive in cohorts; a cohort of age u has earned trophy i with probability f_i*(1-exp(-(u/tau_i)^k));
//   - Exophase refreshes every ~3h; reads every 30min early / 10min later show the last refresh's page;
//   - the case is cut at 2026-11-21T11:30Z, target 2026-11-26T00:00Z (the final's shape).
// Runs main.ts on the case, scores with the contest's exact pinball loss on the logit scale, and compares
// against reference implementations of the two house baselines (persistence; ln-time curve over last 24h).
// The synthetic truth is a plausible dynamics, NOT the real answer: this validates mechanics and calibration.
// Usage: node tests/synth.mjs

import { mkdtempSync, rmSync, writeFileSync, readFileSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

const ROOT = join(import.meta.dirname, '..');
const KIT = 'D:/Project X/makefun/gta6-trophy-rarity-forecast-01-kit';
const base = JSON.parse(readFileSync(join(KIT, 'input', 'p00.in'), 'utf8'));
const trophies = base.trophies;

const clampPct = (x) => Math.min(99.99, Math.max(0.01, x));
const logit = (pct) => { const c = clampPct(pct); return Math.log(c / (100 - c)); };
const sigmoid = (l) => 100 / (1 + Math.exp(-l));
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];

// Deterministic LCG for scenario jitter.
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

// Per-trophy progression parameters [f (lifetime earn fraction of a cohort), tau (hours), k (shape)].
const PARAMS = {
  1: [0.004, 230, 1.6], 2: [0.997, 0.05, 1.5], 3: [0.88, 7, 1.6], 4: [0.74, 18, 1.6],
  5: [0.58, 33, 1.6], 6: [0.44, 52, 1.6], 7: [0.32, 72, 1.7], 8: [0.6, 9, 1.6],
  9: [0.55, 16, 1.6], 10: [0.5, 24, 1.6], 11: [0.46, 33, 1.6], 12: [0.42, 44, 1.6],
  13: [0.38, 58, 1.6], 14: [0.011, 210, 1.5], 15: [0.03, 170, 1.6], 16: [0.35, 12, 1.6],
  17: [0.05, 90, 1.6], 18: [0.04, 110, 1.6], 19: [0.08, 100, 1.6], 20: [0.3, 8, 1.6],
  21: [0.4, 6, 1.6], 22: [0.2, 60, 1.6], 23: [0.18, 75, 1.6], 24: [0.16, 90, 1.6],
  25: [0.07, 130, 1.6], 26: [0.14, 100, 1.6], 27: [0.12, 110, 1.6], 28: [0.06, 120, 1.6],
  29: [0.18, 40, 1.6], 30: [0.04, 150, 1.6], 31: [0.05, 140, 1.6], 32: [0.06, 130, 1.6],
  33: [0.22, 15, 1.6], 34: [0.28, 10, 1.6], 35: [0.12, 70, 1.6], 36: [0.3, 12, 1.6],
  37: [0.25, 20, 1.6], 38: [0.15, 35, 1.6], 39: [0.2, 25, 1.6], 40: [0.07, 95, 1.6],
  41: [0.25, 18, 1.6], 42: [0.08, 110, 1.6], 43: [0.1, 85, 1.6], 44: [0.5, 3, 1.6],
  45: [0.35, 8, 1.6], 46: [0.4, 5, 1.6], 47: [0.3, 15, 1.6], 48: [0.35, 20, 1.6],
  49: [0.25, 25, 1.6], 50: [0.2, 30, 1.6], 51: [0.18, 20, 1.6], 52: [0.06, 140, 1.6],
  53: [0.1, 100, 1.6], 54: [0.05, 150, 1.6], 55: [0.15, 60, 1.6],
};

const T0 = Date.parse('2026-11-18T23:50:00Z');
const CHECKPOINT = Date.parse('2026-11-21T11:30:00Z');
const TARGET = Date.parse('2026-11-26T00:00:00Z');
const DT = 0.25; // hours
const END = (TARGET - T0) / 3600000 + 2;

function simulate(scenario) {
  const { pmax, r, seed, fScale } = scenario;
  const rand = lcg(seed);
  const params = trophies.map((t) => {
    const [f, tau, k] = PARAMS[t.id];
    return [Math.min(0.999, f * fScale * (0.85 + 0.3 * rand())), tau * (0.85 + 0.3 * rand()), k];
  });
  const P = (t) => pmax * Math.exp(-Math.log(pmax / 638) * Math.exp(-r * t));
  const steps = Math.ceil(END / DT);
  // Cohort arrivals: initial 638 with mixed ages, then increments of P.
  const cohorts = [{ n: 638 * 0.97, born: -0.2 }, { n: 638 * 0.025, born: -4 }, { n: 638 * 0.005, born: -12 }];
  let prevP = 638;
  for (let s = 1; s <= steps; s++) {
    const t = s * DT;
    const p = P(t);
    cohorts.push({ n: p - prevP, born: t });
    prevP = p;
  }
  // Earn counts per trophy over time: E_i(t) = sum_cohorts n * F_i(t - born).
  const F = (u, pr) => (u <= 0 ? 0 : pr[0] * (1 - Math.exp(-Math.pow(u / pr[1], pr[2]))));
  const ratesAt = (t) => {
    const p = P(Math.max(0, t));
    return params.map((pr) => {
      let e = 0;
      for (const c of cohorts) {
        if (c.born > t) break;
        e += c.n * F(t - c.born, pr);
      }
      return Math.round((100 * e) / p * 100) / 100;
    });
  };
  const earnedAt = (t) => {
    const p = P(Math.max(0, t));
    let e = 0;
    for (const c of cohorts) {
      if (c.born > t) break;
      const age = t - c.born;
      for (const pr of params) e += c.n * F(age, pr);
    }
    return Math.round(e);
  };
  // Refresh schedule: every ~3h with jitter; page state = last refresh <= read time.
  const refreshes = [0];
  let rt = 0;
  while (rt < END) {
    rt += 2.5 + rand() * 1.5;
    refreshes.push(rt);
  }
  const pageAt = (t) => {
    let last = 0;
    for (const x of refreshes) { if (x <= t) last = x; else break; }
    return last;
  };
  // Reads: 30min cadence for the first 12h, 10min after, strictly before the checkpoint.
  const history = [];
  let slot = 0;
  while (T0 + slot * 3600000 < CHECKPOINT) {
    const t = slot;
    const rf = pageAt(t);
    const at = new Date(T0 + t * 3600000 + 300).toISOString();
    history.push({
      slot: new Date(T0 + t * 3600000).toISOString(),
      at,
      players_tracked: Math.round(P(rf)),
      trophies_earned: earnedAt(rf),
      rates: ratesAt(rf),
    });
    slot += t < 12 ? 0.5 : 1 / 6;
  }
  // Answer: first read in [target, target+1h) -> page state at its last refresh.
  const answerT = pageAt((TARGET - T0) / 3600000 + 1 / 60);
  const truth = ratesAt(answerT);
  const caseObj = {
    schema: 'makefun/gta6-trophies-case/v1',
    id: scenario.id,
    at: new Date(CHECKPOINT).toISOString(),
    horizon_minutes: (TARGET - CHECKPOINT) / 60000,
    target: new Date(TARGET).toISOString(),
    levels: LEVELS,
    game: base.game,
    trophies,
    history,
    notes: 'synthetic test case (tests/synth.mjs), not house data',
  };
  return { caseObj, truth };
}

function pinball(truth, rows) {
  let loss = 0, miss = 0, cover = 0, items = 0;
  const byId = new Map(rows.map((r) => [r.id, r.q]));
  trophies.forEach((t, i) => {
    const y = logit(truth[i]);
    const q = byId.get(t.id);
    for (let k = 0; k < 5; k++) {
      const d = y - logit(q[k]);
      loss += 2 * Math.max(LEVELS[k] * d, (LEVELS[k] - 1) * d);
      items++;
    }
    miss += Math.abs(truth[i] - q[2]);
    if (q[0] <= truth[i] && truth[i] <= q[4]) cover++;
  });
  return { loss: loss / items, miss: miss / trophies.length, cover: cover / trophies.length };
}

// Reference house baselines (own implementations of the rules' descriptions).
function persistenceBaseline(c) {
  const last = c.history[c.history.length - 1];
  const spread = 0.5 * Math.sqrt(c.horizon_minutes / 1440);
  return trophies.map((t, i) => ({
    id: t.id,
    q: Z.map((z) => Math.round(sigmoid(logit(last.rates[i] ?? 0) + z * spread) * 10000) / 10000),
  }));
}
function curveBaseline(c) {
  const last = c.history[c.history.length - 1];
  const d = c.horizon_minutes / 1440;
  const walk = 0.5 * Math.sqrt(d);
  const origin = Date.parse('2026-11-19T00:00:00Z');
  const tTarget = Math.log(1 + (Date.parse(c.target) - origin) / 3600000);
  const cutoff = Date.parse(c.at) - 24 * 3600000;
  const reads = c.history.filter((h) => Date.parse(h.at) >= cutoff);
  if (reads.length < 3) return persistenceBaseline(c);
  return trophies.map((t, i) => {
    const xs = [], ys = [];
    for (const h of reads) {
      const v = h.rates[i];
      if (v === null) continue;
      xs.push(Math.log(1 + Math.max(0, (Date.parse(h.at) - origin) / 3600000)));
      ys.push(logit(v));
    }
    if (xs.length < 3) return { id: t.id, q: Z.map((z) => Math.round(sigmoid(logit(last.rates[i] ?? 0) + z * walk) * 10000) / 10000) };
    const n = xs.length;
    const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
    let sxx = 0, sxy = 0;
    for (let k = 0; k < n; k++) { sxx += (xs[k] - mx) ** 2; sxy += (xs[k] - mx) * (ys[k] - my); }
    const b = sxx > 1e-12 ? sxy / sxx : 0;
    const a = my - b * mx;
    let sse = 0;
    for (let k = 0; k < n; k++) sse += (ys[k] - a - b * xs[k]) ** 2;
    const resid = Math.sqrt(sse / Math.max(1, n - 2));
    const m = a + b * tTarget;
    const s = Math.sqrt(walk * walk + resid * resid);
    return { id: t.id, q: Z.map((z) => Math.round(sigmoid(m + z * s) * 10000) / 10000) };
  });
}

const scenarios = [
  { id: 'syn-base', pmax: 400000, r: 0.07, seed: 42, fScale: 1.0 },
  { id: 'syn-fast', pmax: 600000, r: 0.09, seed: 7, fScale: 1.0 },
  { id: 'syn-slow', pmax: 250000, r: 0.05, seed: 99, fScale: 0.85 },
];

export { simulate, pinball, trophies, PARAMS };

// Run the comparison only when executed directly (importing for debug does not rerun it).
if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {

console.log('scenario      entry-loss  baseline-loss  curve-loss  entry-miss  entry-cover  players@cp');
for (const sc of scenarios) {
  const { caseObj, truth } = simulate(sc);
  const tmpIn = mkdtempSync(join(tmpdir(), 'gta6syn-in-'));
  const tmpOut = mkdtempSync(join(tmpdir(), 'gta6syn-out-'));
  try {
    writeFileSync(join(tmpIn, caseObj.id + '.in'), JSON.stringify(caseObj));
    execFileSync(process.execPath, [join(ROOT, 'main.ts')], {
      env: { ...process.env, MAKEFUN_INPUT_DIR: tmpIn, MAKEFUN_OUTPUT_DIR: tmpOut },
      stdio: 'pipe',
    });
    const rows = JSON.parse(readFileSync(join(tmpOut, caseObj.id + '.out'), 'utf8')).trophies;
    const ours = pinball(truth, rows);
    const pers = pinball(truth, persistenceBaseline(caseObj));
    const curv = pinball(truth, curveBaseline(caseObj));
    const last = caseObj.history[caseObj.history.length - 1];
    console.log(
      sc.id.padEnd(13),
      ours.loss.toFixed(4).padStart(10), pers.loss.toFixed(4).padStart(14), curv.loss.toFixed(4).padStart(11),
      ours.miss.toFixed(2).padStart(11), ours.cover.toFixed(2).padStart(12),
      String(last.players_tracked).padStart(11),
    );
    // Per-trophy detail for the base scenario: where do we miss most?
    if (sc.id === 'syn-base') {
      const byId = new Map(rows.map((r) => [r.id, r.q]));
      const detail = trophies.map((t, i) => ({
        id: t.id, name: t.name, last: last.rates[i], truth: truth[i], med: byId.get(t.id)[2],
        err: Math.abs(logit(truth[i]) - logit(byId.get(t.id)[2])),
      })).sort((a, b) => b.err - a.err);
      console.log('\n  worst 12 median misses (logit):');
      for (const d of detail.slice(0, 12)) {
        console.log(`   ${String(d.id).padStart(2)} ${d.name.slice(0, 30).padEnd(30)} last=${String(d.last).padStart(6)} truth=${String(d.truth).padStart(6)} med=${String(d.med).padStart(7)} |dlogit|=${d.err.toFixed(2)}`);
      }
      console.log('');
    }
  } finally {
    rmSync(tmpIn, { recursive: true, force: true });
    rmSync(tmpOut, { recursive: true, force: true });
  }
}
}