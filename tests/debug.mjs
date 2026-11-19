// Debug aid (test-only): replicates main.ts's series construction and ensemble for selected trophies
// on the synthetic base scenario, printing every intermediate so misses can be diagnosed.
import { simulate } from './synth.mjs';

const clampPct = (x) => Math.min(99.99, Math.max(0.01, x));
const logit = (pct) => { const c = clampPct(pct); return Math.log(c / (100 - c)); };
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

function fitLine(xs, ys, ws) {
  const n = xs.length;
  let sw = 0, sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { sw += ws[i]; sx += ws[i] * xs[i]; sy += ws[i] * ys[i]; }
  const mx = sx / sw, my = sy / sw;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { const dx = xs[i] - mx; sxx += ws[i] * dx * dx; sxy += ws[i] * dx * (ys[i] - my); }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  let sse = 0;
  for (let i = 0; i < n; i++) { const r = ys[i] - (a + b * xs[i]); sse += ws[i] * r * r; }
  const dof = Math.max(1, n - 2);
  const resid = Math.sqrt(sse / (sw * dof / n));
  const seB = sxx > 1e-12 ? resid / Math.sqrt(sxx) : 1e3;
  return { a, b, resid, seB, n };
}
const shrink = (b, seB) => { const t2 = (b * b) / Math.max(1e-9, seB * seB); return b * (t2 / (t2 + 4)); };

const { caseObj, truth } = simulate({ id: 'syn-base', pmax: 400000, r: 0.07, seed: 42, fScale: 1.0 });
const c = caseObj;
const t0 = Date.parse(c.history[0].at);
const tNow = (Date.parse(c.at) - t0) / 3600000;
const tTarget = (Date.parse(c.target) - t0) / 3600000;
const H = c.horizon_minutes / 60;

// page points + growth fit (as in main.ts)
const pagePts = [];
let prevKey = '';
for (const r of c.history) {
  const key = `${r.players_tracked}|${JSON.stringify(r.rates)}`;
  if (key !== prevKey) { pagePts.push({ t: (Date.parse(r.at) - t0) / 3600000, p: r.players_tracked }); prevKey = key; }
}
const lastPlayers = c.history[c.history.length - 1].players_tracked;
const recent = pagePts.filter((p) => p.t >= tNow - 24);
const growthPts = recent.length >= 3 ? recent : pagePts;
const gf = fitLine(growthPts.map((p) => Math.log(1 + Math.max(0, p.t))), growthPts.map((p) => Math.log(Math.max(1, p.p))), growthPts.map((p) => Math.exp(-(tNow - p.t) / 24)));
const gb = shrink(gf.b, gf.seB);
const lnPGrowth = Math.min(Math.log(12), Math.max(0, gb * (Math.log(1 + tTarget) - Math.log(1 + tNow))));
console.log(`tNow=${tNow.toFixed(2)}h tTarget=${tTarget.toFixed(2)}h H=${H.toFixed(1)}h lastPlayers=${lastPlayers} lnPGrowth=${lnPGrowth.toFixed(3)} (growth b=${gf.b.toFixed(3)} se=${gf.seB.toFixed(3)})`);
console.log('page refresh points:', pagePts.map((p) => `${p.t.toFixed(1)}h:${p.p}`).join(' '));

for (const tid of [1, 14, 15, 30, 2]) {
  const i = c.trophies.findIndex((t) => t.id === tid);
  // raw series
  const raw = [];
  let prev = '';
  for (const r of c.history) {
    const v = r.rates[i];
    if (v === null) continue;
    const key = `${r.players_tracked}|${v}`;
    if (key !== prev) { raw.push({ t: (Date.parse(r.at) - t0) / 3600000, p: r.players_tracked, y: logit(v), v }); prev = key; }
  }
  let start = 0;
  if (raw.length >= 2) {
    const last = raw[raw.length - 1];
    if (last.y > -9.2) for (let k = raw.length - 2; k >= 0; k--) if (raw[k].y <= -9.2) { start = k; break; }
    if (last.y < 9.2) for (let k = raw.length - 2; k >= 0; k--) if (raw[k].y >= 9.2) { start = Math.max(start, k); break; }
  }
  const pts = raw.slice(start);
  console.log(`\n=== trophy ${tid} (${c.trophies[i].name}) truth=${truth[i]} raw=${raw.length}pts censoredStart=${start} kept=${pts.length}pts`);
  console.log('  kept points:', pts.map((p) => `${p.t.toFixed(1)}h:${p.v}%`).join(' '));
  let win = pts.filter((p) => p.t >= tNow - 30);
  if (win.length < 3) win = pts.filter((p) => p.t >= tNow - 48);
  if (win.length < 3) win = pts;
  console.log(`  window: ${win.length}pts`);
  if (win.length < 2) { console.log('  (too few points)'); continue; }
  const xsT = win.map((p) => p.t), xsL = win.map((p) => Math.log(1 + Math.max(0, p.t))), xsP = win.map((p) => Math.log(Math.max(1, p.p)));
  const ys = win.map((p) => p.y), ws = win.map((p) => Math.exp(-(tNow - p.t) / 12));
  const fL = fitLine(xsL, ys, ws), fT = fitLine(xsT, ys, ws), fP = fitLine(xsP, ys, ws);
  const bL = shrink(fL.b, fL.seB), bT = shrink(fT.b, fT.seB), bP = shrink(fP.b, fP.seB);
  const mLn = fL.a + bL * Math.log(1 + tTarget);
  const mLin = fT.a + bT * tNow + bT * H * 0.55;
  const lnPRange = Math.max(...xsP) - Math.min(...xsP);
  const popMove = Math.max(-1.5, Math.min(1.5, bP * lnPGrowth));
  const mPop = fP.a + bP * Math.log(Math.max(1, lastPlayers)) + popMove;
  let wLn = 0.45, wLin = 0.25, wPop = 0.3;
  if (win.length < 3) { wLn = 0; wLin = 0.6; wPop = 0.4; }
  if (lnPRange < 0.15) wPop = 0;
  const m = (wLn * mLn + wLin * mLin + wPop * mPop) / (wLn + wLin + wPop);
  const yLast = pts[pts.length - 1].y;
  const anchor = win.length >= 4 ? fL.a + bL * Math.log(1 + tNow) : yLast;
  const capped = anchor + Math.max(-3, Math.min(3, m - anchor));
  const blended = 0.75 * capped + 0.25 * yLast;
  console.log(`  fL: b=${fL.b.toFixed(3)} se=${fL.seB.toFixed(3)} -> bL=${bL.toFixed(3)} resid=${fL.resid.toFixed(3)} mLn=${mLn.toFixed(2)}`);
  console.log(`  fT: b=${fT.b.toFixed(4)} se=${fT.seB.toFixed(4)} -> bT=${bT.toFixed(4)} mLin=${mLin.toFixed(2)}`);
  console.log(`  fP: b=${fP.b.toFixed(3)} se=${fP.seB.toFixed(3)} -> bP=${bP.toFixed(3)} lnPRange=${lnPRange.toFixed(3)} mPop=${mPop.toFixed(2)} wPop=${wPop}`);
  console.log(`  ensemble m=${m.toFixed(2)} anchor=${anchor.toFixed(2)} capped=${capped.toFixed(2)} blended=${blended.toFixed(2)} -> med=${(100 / (1 + Math.exp(-blended))).toFixed(4)}% (yLast=${yLast.toFixed(2)})`);
}