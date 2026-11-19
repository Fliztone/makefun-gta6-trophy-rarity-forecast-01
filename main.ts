// gta6-trophy-rarity-forecast-01 entry: week-one Exophase earn-rate forecasts for every GTA VI PS5 trophy.
//
// Node 22+, standard library only, offline, deterministic (no clock reads, no randomness, stable iteration).
// Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output):
//   {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}
// five quantiles of the trophy's earn rate (percent 0..100) at the case's target, non-decreasing, <= 65536 bytes.
//
// METHOD
// ------
// The case history is a sequence of reads of Exophase's trophy page. Exophase refreshes a game's counts on its
// own schedule (hours apart), so consecutive reads are usually identical: the information lives in the refresh
// steps. We collapse duplicates into per-trophy series of (hours, players_tracked, logit(rate)) and forecast in
// logit space, where the contest scores (pinball loss on logit(y) vs logit(q)).
//
// Median: an ensemble of three damped extrapolations fitted to a recent window of refresh points (last 30h,
// extended when thin) with recency weights -- launch-night points mislead trophies that sat at the 0.00% floor
// while the base exploded, so the window keeps the fit on the current regime:
//   M_ln  : logit(rate) ~ a + b*ln(1+t)      -- log-time trend; dilution slows as the player base matures,
//                                               and ln(1+t) damps the extrapolation naturally.
//   M_lin : logit(rate) ~ a + b*t            -- linear trend, damped by 0.55 over the horizon (trends in
//                                               earn rates decay: early adopters progress, casuals dilute,
//                                               and both effects saturate within a week).
//   M_pop : logit(rate) ~ a + b*ln(players)  -- dilution/progress driven by the tracked population; players
//                                               extrapolated with a log-time growth fit capped at 12x.
// Slopes are shrunk toward zero by t^2/(t^2+4) (noisy slopes move the median less) and every shrunk line is
// re-anchored at its weighted centroid; the blend is capped at +/-3 logits from the last fitted value and
// blended 25% toward persistence (trends partially revert over a week). Structural constraints are enforced:
// story trophies are nested (prologue >= chapter 1 >= ... >= final mission, isotonic PAV pass), platinum <=
// every other trophy, a trophy printed at 100% can only fall and one printed at 0.00% can only rise, saturated
// trophies (>= 90%) stay near their last value, and runs printed at the 0.00%/100.00% clamp are censored out
// of the fits once the trophy leaves the boundary.
//
// Stale-launch regime: a flat page (no refresh yet) with a tiny tracked base (< 2000 players) early in launch
// means the first big refresh is imminent and will dilute every rate: the tracked population grows several-fold
// with players who have barely started (launch night: 638 players, 943 trophies earned in total; the first
// refresh multiplied the base ~6x and lowered every rate but the 100% one). There we shift rare-trophy medians
// down ~0.5 logits and widen the spread; the 100% trophy stays pinned near 100 (the rules record that it
// survived the first refresh unchanged).
//
// Spread: sigma^2 = walk + fit + structure.
//   walk : robust per-trophy volatility of DETRENDED refresh steps (MAD of (dy - b*dx)/sqrt(dt)), shrunk toward
//          the cross-trophy pool, times sqrt(horizon hours) -- a random walk over the unobserved refreshes;
//          the predictable part of each move belongs to the median, not the spread.
//   fit  : slope standard error times the extrapolation distance, plus 0.4x the fit residual scale.
//   struct: 0.22*sqrt(horizon days) + 0.07 -- irreducible model error (refresh timing, population composition).
// Two-sided logit-normal quantiles; boundary trophies get asymmetric sigmas (100%: wide downside, 0%: wide
// upside). A calibrated logit-normal forecast scores ~0.50*sigma in mean pinball, so sigma is an honest
// predictive spread, never tuned narrow.
//
// FALLBACK PRIORS (only when a trophy has < 2 usable refresh points): week-one expectations anchored to
// historical Rockstar/open-world launches -- GTA V "100%" trophy ~4.5% lifetime on Exophase, GTA V platinum
// ~0.24% (PSNProfiles), RDR2 chapter trophies ~42-59% and story completion ~22-30% lifetime; GTA VI's campaign
// is ~42-50h (91 missions, 5 chapters) and its platinum ~200h of play, so at day 7 only the hardcore tail has
// finished the story: prologue ~95%, early chapters ~35-55%, late chapters ~10-20%, story end ~8%, 100% ~0.6%,
// platinum ~0.25%, grind/collectible silvers ~1-4%, ordinary side bronzes ~3-8%.
//
// Sources consulted (research, not code): exophase.com GTA V PS5 trophy page; psnprofiles.com GTA V / RDR2
// trophy lists; howlongtobeat.com/game/104609; Rock Paper Shotgun / Kotaku / GameSpot GTA VI length coverage;
// truetrophies.com Spider-Man 2 vs Hogwarts Legacy launch comparison (player counts peak ~1 week post-launch).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}
interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}
interface Pt {
  t: number; // hours since the first read
  p: number; // players tracked
  y: number; // logit(rate), rate clamped to 0.01..99.99
  v: number; // the printed rate (percent), for boundary censoring
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const c = clampPct(pct);
  return Math.log(c / (100 - c));
};
const sigmoid = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Weighted least squares y = a + b*x; returns slope, intercept, weighted centroid, residual scale, slope stderr. */
function fitLine(xs: number[], ys: number[], ws: number[]): {
  a: number; b: number; mx: number; my: number; resid: number; seB: number; n: number;
} {
  const n = xs.length;
  let sw = 0, sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { sw += ws[i]; sx += ws[i] * xs[i]; sy += ws[i] * ys[i]; }
  const mx = sx / sw, my = sy / sw;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += ws[i] * dx * dx;
    sxy += ws[i] * dx * (ys[i] - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    sse += ws[i] * r * r;
  }
  const dof = Math.max(1, n - 2);
  const resid = Math.sqrt(sse / (sw * dof / n));
  const seB = sxx > 1e-12 ? resid / Math.sqrt(sxx) : 1e3;
  return { a, b, mx, my, resid, seB, n };
}

/** Shrink a slope by its t-statistic: t²/(t²+4). Well-estimated slopes pass ~unchanged, noisy ones fade. */
const shrink = (b: number, seB: number): number => {
  const t2 = (b * b) / Math.max(1e-9, seB * seB);
  return b * (t2 / (t2 + 4));
};

/** Pool adjacent violators: enforce non-increasing values along a chain (weighted). */
function pavNonIncreasing(vals: number[], ws: number[]): number[] {
  const blocks: Array<{ v: number; w: number; k: number }> = [];
  for (let i = 0; i < vals.length; i++) {
    let cur = { v: vals[i], w: ws[i], k: 1 };
    while (blocks.length > 0 && blocks[blocks.length - 1].v < cur.v) {
      const prev = blocks.pop()!;
      const w = prev.w + cur.w;
      cur = { v: (prev.v * prev.w + cur.v * cur.w) / w, w, k: prev.k + cur.k };
    }
    blocks.push(cur);
  }
  const out: number[] = [];
  for (const b of blocks) for (let i = 0; i < b.k; i++) out.push(b.v);
  return out;
}

/** Fallback week-one prior (percent) from tier + description; used only with < 2 refresh points. */
function priorRate(t: Trophy, prologueId: number | null): number {
  const d = t.description.toLowerCase();
  if (t.tier === 'platinum') return 0.25;
  if (/100% completion/.test(d)) return 0.6;
  if (t.id === prologueId) return 95;
  const ch = d.match(/^complete chapter (\d+)/);
  if (ch) {
    const k = parseInt(ch[1], 10);
    return [55, 38, 24, 14, 9][Math.min(k - 1, 4)];
  }
  if (/^complete the final mission\.?$/.test(d)) return 8;
  if (/final mission for/.test(d)) return 5; // stranger-strand finales
  if (/complete (gone fishing|breakfast run|the cartel heist|child's play|headliners|vcia takedown|operation savage glory)/.test(d)) return 12;
  if (/\b(250|100|50|20)\b/.test(d) || /all (locations|shops|garage|of leonida)|every (spot|location)|each gang/.test(d)) return 2.5;
  if (/earn \$|spend \$|laundering/.test(d)) return 3;
  if (/purchase all|register \d+ vehicles/.test(d)) return 2;
  if (/explore all|discover \d+|meditate/.test(d)) return 4;
  if (t.tier === 'silver') return 3.5;
  if (t.secret) return 6;
  return 5;
}

/** Classify the story chain: prologue (the ~100% "Complete ..." trophy), chapters in order, main finale. */
function storyChain(trophies: Trophy[], firstRates: Array<number | null>): number[] {
  const idx: number[] = [];
  let prologue = -1;
  const chapters: Array<{ i: number; k: number }> = [];
  let finale = -1;
  trophies.forEach((t, i) => {
    const d = t.description.toLowerCase();
    const ch = d.match(/^complete chapter (\d+)/);
    if (ch) chapters.push({ i, k: parseInt(ch[1], 10) });
    else if (/^complete the final mission\.?$/.test(d)) finale = i;
    else if (prologue < 0 && /^complete /.test(d) && (firstRates[i] ?? 0) >= 99) prologue = i;
  });
  if (prologue >= 0) idx.push(prologue);
  chapters.sort((a, b) => a.k - b.k);
  for (const c of chapters) idx.push(c.i);
  if (finale >= 0) idx.push(finale);
  return idx;
}

interface Forecast {
  m: number; // median logit
  sDown: number;
  sUp: number;
}

function forecastCase(c: Case): Forecast[] {
  const n = c.trophies.length;
  const hist = c.history;
  const t0 = hist.length > 0 ? Date.parse(hist[0].at) : Date.parse(c.at);
  const tNow = (Date.parse(c.at) - t0) / 3600000;
  const tTarget = (Date.parse(c.target) - t0) / 3600000;
  const H = c.horizon_minutes / 60; // horizon hours
  const Hd = H / 24;

  // Page-level refresh events: reads where the tracked base or any rate changed.
  const pagePts: Array<{ t: number; p: number }> = [];
  let prevKey = '';
  for (const r of hist) {
    const key = `${r.players_tracked}|${JSON.stringify(r.rates)}`;
    if (key !== prevKey) {
      pagePts.push({ t: (Date.parse(r.at) - t0) / 3600000, p: r.players_tracked });
      prevKey = key;
    }
  }
  const lastRead = hist.length > 0 ? hist[hist.length - 1] : null;
  const lastPlayers = lastRead ? lastRead.players_tracked : 0;
  const staleLaunch = pagePts.length <= 1 && lastPlayers > 0 && lastPlayers < 2000 && tNow < 12;

  // Per-trophy refresh series (skip nulls; a point is new when rate or tracked base changed).
  // Floor/ceiling censoring: a run printed at 0.00% (or 100.00%) is censored -- the true logit is beyond the
  // clamp, not at it. Once a trophy leaves the boundary, fits start at the last boundary point; the flat
  // boundary run carries no slope information and would drag extrapolations toward the clamp.
  const series: Pt[][] = c.trophies.map(() => []);
  const lastRate: number[] = new Array(n).fill(0);
  c.trophies.forEach((_t, i) => {
    let prev = '';
    const raw: Pt[] = [];
    for (const r of hist) {
      const v = r.rates[i];
      if (v === null || v === undefined) continue;
      const key = `${r.players_tracked}|${v}`;
      if (key !== prev) {
        raw.push({ t: (Date.parse(r.at) - t0) / 3600000, p: r.players_tracked, y: logit(v), v });
        prev = key;
      }
      lastRate[i] = v;
    }
    let start = 0;
    if (raw.length >= 2) {
      const last = raw[raw.length - 1];
      // Only a printed 0.00% (resp. 100.00%) is censored; 0.01% is a real, if coarse, observation.
      if (last.v > 0.005) {
        for (let k = raw.length - 2; k >= 0; k--) {
          if (raw[k].v <= 0.005) { start = k; break; }
        }
      }
      if (last.v < 99.995) {
        for (let k = raw.length - 2; k >= 0; k--) {
          if (raw[k].v >= 99.995) { start = Math.max(start, k); break; }
        }
      }
    }
    series[i] = raw.slice(start);
  });

  // Player-growth extrapolation: ln(P) ~ ln(1+t) over the last 24h of page refreshes (growth saturates).
  let lnPGrowth = Math.log(2); // default: 2x over the horizon if no growth signal
  const recent = pagePts.filter((p) => p.t >= tNow - 24);
  const growthPts = recent.length >= 3 ? recent : pagePts;
  if (growthPts.length >= 2 && lastPlayers > 0) {
    const gx = growthPts.map((p) => Math.log(1 + Math.max(0, p.t)));
    const gy = growthPts.map((p) => Math.log(Math.max(1, p.p)));
    const gw = growthPts.map((p) => Math.exp(-(tNow - p.t) / 24));
    const gf = fitLine(gx, gy, gw);
    const gb = shrink(gf.b, gf.seB);
    const dLnP = gb * (Math.log(1 + tTarget) - Math.log(1 + tNow));
    lnPGrowth = Math.min(Math.log(12), Math.max(0, dLnP));
  }

  // Cross-trophy volatility pool from raw refresh steps (robust, per sqrt-hour). Used only as a shrinkage
  // target / fallback; raw steps include the predictable trend, so the pool is discounted where it is used.
  const volSamples: number[] = [];
  series.forEach((pts) => {
    for (let k = 1; k < pts.length; k++) {
      const dt = Math.max(0.25, pts[k].t - pts[k - 1].t);
      volSamples.push((pts[k].y - pts[k - 1].y) / Math.sqrt(dt));
    }
  });
  const poolVol = volSamples.length >= 4
    ? Math.min(0.9, 1.4826 * median(volSamples.map((v) => Math.abs(v - median(volSamples)))) + 0.02)
    : 0.35;
  const poolVolEff = poolVol * 0.7;

  const prologueId = (() => {
    const chain = storyChain(c.trophies, hist.length > 0 ? hist[0].rates : []);
    return chain.length > 0 ? c.trophies[chain[0]].id : null;
  })();

  const out: Forecast[] = [];
  for (let i = 0; i < n; i++) {
    const pts = series[i];
    const trophy = c.trophies[i];
    const yLast = pts.length > 0 ? pts[pts.length - 1].y : logit(priorRate(trophy, prologueId));
    const atBound100 = lastRate[i] >= 99.99;
    const atBound0 = lastRate[i] <= 0.005 && pts.length > 0;

    let m = yLast;
    let sigma = 0.3 * Math.sqrt(Hd) + 0.1;
    let sDown = sigma;
    let sUp = sigma;

    if (staleLaunch) {
      // First refresh imminent: several-fold dilution by players who have barely started.
      const dil = 0.55 * (1 - clampPct(lastRate[i]) / 100); // logit shift, softened near 100%
      m = yLast - Math.min(1.5, dil * Math.min(2, 0.6 + Hd));
      sigma = 0.95 + 0.35 * Math.sqrt(Hd);
      if (atBound100) {
        // The rules record the 100% trophy surviving the first refresh unchanged.
        m = yLast;
        sDown = 0.8;
        sUp = 0.15;
      } else if (atBound0) {
        // A 0.00% trophy can only rise; the first players to reach it are already tracked.
        m = yLast + 0.15;
        sDown = 0.2;
        sUp = sigma;
      } else {
        sDown = sigma;
        sUp = sigma;
      }
    } else if (pts.length >= 2) {
      // Fit on a recent window: launch-night points are misleading for trophies that sat at the 0.00% floor
      // while the player base exploded (the early ln(P) correlation reads as dilution even for trophies that
      // are now rising as cohorts age). 30h window, extended to 48h / all history only when points are thin.
      let win = pts.filter((p) => p.t >= tNow - 30);
      if (win.length < 3) win = pts.filter((p) => p.t >= tNow - 48);
      if (win.length < 3) win = pts;
      const xsT = win.map((p) => p.t);
      const xsL = win.map((p) => Math.log(1 + Math.max(0, p.t)));
      const xsP = win.map((p) => Math.log(Math.max(1, p.p)));
      const ys = win.map((p) => p.y);
      const ws = win.map((p) => Math.exp(-(tNow - p.t) / 12));

      const fL = fitLine(xsL, ys, ws);
      const fT = fitLine(xsT, ys, ws);
      const fP = fitLine(xsP, ys, ws);
      const bL = shrink(fL.b, fL.seB);
      const bT = shrink(fT.b, fT.seB);
      const bP = shrink(fP.b, fP.seB);

      // Evaluate every shrunk line re-anchored at its weighted centroid: the intercept belongs to the
      // unshrunk slope, so a + b_shrunk*x would bias the line by (b - b_shrunk)*xbar.
      const mLn = fL.my + bL * (Math.log(1 + tTarget) - fL.mx);
      const fitNow = fT.my + bT * (tNow - fT.mx);
      const mLin = fitNow + bT * H * 0.55;

      let wLn = 0.45, wLin = 0.25, wPop = 0.3;
      if (win.length < 3) { wLn = 0; wLin = 0.6; wPop = 0.4; }
      if (lastPlayers <= 1) wPop = 0;
      // The ln(P) fit is ill-conditioned when the tracked base barely moved inside the window: a tiny
      // x-range makes its slope enormous and noisy. Drop it there, and cap its extrapolation contribution.
      const lnPRange = Math.max(...xsP) - Math.min(...xsP);
      if (lnPRange < 0.15) wPop = 0;
      const popMove = Math.max(-1.5, Math.min(1.5, bP * lnPGrowth));
      const mPopSafe = fP.my + bP * (Math.log(Math.max(1, lastPlayers)) - fP.mx) + popMove;
      const wSum = wLn + wLin + wPop;
      m = (wLn * mLn + wLin * mLin + wPop * mPopSafe) / wSum;

      // Cap the extrapolation move; keep the last fitted value as the anchor when data is thin.
      const anchor = win.length >= 4 ? fL.my + bL * (Math.log(1 + tNow) - fL.mx) : yLast;
      m = anchor + Math.max(-3, Math.min(3, m - anchor));
      // Trends partially revert over a week-long horizon: keep a quarter of the belief at the last value.
      m = 0.75 * m + 0.25 * yLast;
      // Near-floor trophies with thin data: the scorer's clamp makes 0.00% and 0.01% the same logit, hiding
      // the start of a rise. Blend toward the week-one prior (never below 3x the last rate) for upside.
      if (win.length <= 3 && lastRate[i] <= 0.05 && !atBound0) {
        const pr = Math.max(priorRate(trophy, prologueId), lastRate[i] * 3);
        m = 0.6 * m + 0.4 * logit(pr);
      }

      // Volatility from detrended refresh steps: the predictable part of each move belongs to the median,
      // not the spread. Steps outside the window keep their raw move (launch-night dynamics are volatile).
      const vs: number[] = [];
      for (let k = 1; k < pts.length; k++) {
        const dt = Math.max(0.25, pts[k].t - pts[k - 1].t);
        let dy = pts[k].y - pts[k - 1].y;
        if (pts[k - 1].t >= tNow - 30) {
          dy -= bL * (Math.log(1 + Math.max(0, pts[k].t)) - Math.log(1 + Math.max(0, pts[k - 1].t)));
        }
        vs.push(dy / Math.sqrt(dt));
      }
      let vT = poolVol;
      if (vs.length >= 2) {
        const mv = median(vs);
        vT = Math.min(0.9, 1.4826 * median(vs.map((v) => Math.abs(v - mv))) + 0.02);
      } else if (vs.length === 1) {
        vT = Math.min(0.9, Math.abs(vs[0]));
      }
      const kShrink = 5;
      const vEff = Math.sqrt((vs.length * vT * vT + kShrink * poolVolEff * poolVolEff) / (vs.length + kShrink));
      const sWalk = vEff * Math.sqrt(H);
      const dx = Math.log(1 + tTarget) - Math.log(1 + tNow);
      const sFit = Math.sqrt((fL.seB * dx) ** 2 + (0.4 * fL.resid) ** 2);
      const sStruct = 0.22 * Math.sqrt(Hd) + 0.07;
      sigma = Math.min(2.2, Math.sqrt(sWalk * sWalk + sFit * sFit + sStruct * sStruct));
      sigma = Math.max(0.25, sigma);
      sDown = sigma;
      sUp = sigma;

      if (atBound100) {
        m = Math.min(m, yLast);
        if (m >= yLast - 0.02) m = yLast - 0.12 * Math.sqrt(Hd); // casuals will slip it below 100 eventually
        sDown = Math.max(sigma, 0.6);
        sUp = 0.2;
      } else if (atBound0) {
        m = Math.max(m, yLast);
        if (m <= yLast + 0.02) m = yLast + 0.25 * Math.sqrt(Hd); // players will reach it
        sDown = 0.2;
        sUp = Math.max(sigma, 0.8);
      } else if (lastRate[i] >= 90) {
        // Saturated trophy (e.g. the prologue): nearly everyone who plays earns it, so it stays near 100;
        // logit extrapolation is unstable this close to the boundary, so hold it near the last value.
        m = Math.max(yLast - 0.8, Math.min(yLast + 0.3, 0.3 * m + 0.7 * yLast));
        sDown = Math.max(0.5, Math.min(sigma, 0.9));
        sUp = 0.25;
      }
    } else {
      // One or zero refresh points outside the stale regime: persistence with a wide structural spread.
      sigma = Math.max(0.6, 0.5 * Math.sqrt(Hd) + 0.3);
      sDown = sigma;
      sUp = sigma;
      if (atBound100) { m = yLast - 0.1 * Math.sqrt(Hd); sDown = Math.max(sigma, 0.6); sUp = 0.2; }
      if (atBound0) { m = yLast + 0.25 * Math.sqrt(Hd); sDown = 0.2; sUp = Math.max(sigma, 0.8); }
    }

    out.push({ m, sDown, sUp });
  }

  // Structural pass: nested story earn sets (isotonic, non-increasing along the chain) and platinum <= all.
  const chain = storyChain(c.trophies, hist.length > 0 ? hist[0].rates : []);
  if (chain.length >= 2) {
    const fixed = pavNonIncreasing(chain.map((i) => out[i].m), chain.map(() => 1));
    chain.forEach((i, k) => { out[i].m = fixed[k]; });
  }
  const platIdx = c.trophies.findIndex((t) => t.tier === 'platinum');
  if (platIdx >= 0) {
    let minOther = Infinity;
    out.forEach((f, i) => { if (i !== platIdx) minOther = Math.min(minOther, f.m); });
    if (Number.isFinite(minOther)) out[platIdx].m = Math.min(out[platIdx].m, minOther);
  }

  return out;
}

function quantiles(f: Forecast): number[] {
  const q = Z.map((z, k) => {
    const s = k < 2 ? f.sDown : k > 2 ? f.sUp : 0;
    return round4(Math.min(100, Math.max(0, sigmoid(f.m + z * s))));
  });
  for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
  return q;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file}: unexpected levels ${JSON.stringify(c.levels)}`);
  }
  const fs = forecastCase(c);
  const rows = c.trophies.map((t, i) => ({ id: t.id, q: quantiles(fs[i]) }));
  const body = `${JSON.stringify({ trophies: rows })}\n`;
  if (Buffer.byteLength(body, 'utf8') > 65536) throw new Error(`${file}: output exceeds 65536 bytes`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), body);
}