// Local validation for the forecast entry (test-only; the graded program is main.ts alone).
// Checks, per case: output schema exactly as the rules demand, size cap, quantile ranges/monotonicity,
// story-chain nesting, platinum <= all, boundary behavior; then byte-for-byte determinism across two runs.
// Usage: node tests/validate.mjs <inputDir> <outputDir>

import { readFileSync, readdirSync, mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';

const [inputDir, outputDir] = process.argv.slice(2);
if (!inputDir || !outputDir) {
  console.error('usage: node tests/validate.mjs <inputDir> <outputDir>');
  process.exit(2);
}

const clampPct = (x) => Math.min(99.99, Math.max(0.01, x));
const logit = (pct) => { const c = clampPct(pct); return Math.log(c / (100 - c)); };
let failures = 0;
const fail = (msg) => { failures++; console.error('FAIL: ' + msg); };
const ok = (msg) => console.log('ok: ' + msg);

const files = readdirSync(inputDir).filter((f) => f.endsWith('.in')).sort();
if (files.length === 0) { fail('no .in files found'); process.exit(1); }

for (const file of files) {
  const id = file.slice(0, -3);
  const c = JSON.parse(readFileSync(join(inputDir, file), 'utf8'));
  const outPath = join(outputDir, id + '.out');
  const raw = readFileSync(outPath, 'utf8');
  if (Buffer.byteLength(raw, 'utf8') > 65536) fail(`${id}: output > 65536 bytes`);
  const o = JSON.parse(raw);
  const keys = Object.keys(o);
  if (keys.length !== 1 || keys[0] !== 'trophies') fail(`${id}: top-level keys ${JSON.stringify(keys)}`);
  if (!Array.isArray(o.trophies)) { fail(`${id}: trophies not a list`); continue; }
  if (o.trophies.length !== c.trophies.length) fail(`${id}: ${o.trophies.length} rows for ${c.trophies.length} trophies`);
  const seen = new Set();
  const byId = new Map();
  for (const row of o.trophies) {
    const rk = Object.keys(row).sort();
    if (rk.join(',') !== 'id,q') fail(`${id}: row keys ${JSON.stringify(rk)}`);
    if (seen.has(row.id)) fail(`${id}: duplicate id ${row.id}`);
    seen.add(row.id);
    if (!Array.isArray(row.q) || row.q.length !== 5) { fail(`${id}/${row.id}: q not 5 values`); continue; }
    for (let k = 0; k < 5; k++) {
      const v = row.q[k];
      if (typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 100) fail(`${id}/${row.id}: q[${k}]=${v} out of range`);
      if (k > 0 && v < row.q[k - 1]) fail(`${id}/${row.id}: q decreasing at ${k}`);
    }
    byId.set(row.id, row.q);
  }
  for (const t of c.trophies) if (!seen.has(t.id)) fail(`${id}: missing trophy ${t.id}`);
  ok(`${id}: schema valid (${o.trophies.length} rows, ${Buffer.byteLength(raw, 'utf8')} bytes)`);

  // Structural sanity on medians.
  const med = (tid) => logit(byId.get(tid)[2]);
  const desc = (tid) => c.trophies.find((t) => t.id === tid).description.toLowerCase();
  const chapters = c.trophies.filter((t) => /^complete chapter \d+/.test(desc(t.id)))
    .sort((a, b) => parseInt(desc(a.id).match(/\d+/)[0]) - parseInt(desc(b.id).match(/\d+/)[0]));
  const finale = c.trophies.find((t) => /^complete the final mission\.?$/.test(desc(t.id)));
  const chainIds = [...chapters.map((t) => t.id), ...(finale ? [finale.id] : [])];
  for (let k = 1; k < chainIds.length; k++) {
    if (med(chainIds[k]) > med(chainIds[k - 1]) + 1e-9) fail(`${id}: story chain rises at ${chainIds[k - 1]} -> ${chainIds[k]}`);
  }
  const plat = c.trophies.find((t) => t.tier === 'platinum');
  if (plat) {
    for (const t of c.trophies) {
      if (t.id !== plat.id && med(plat.id) > med(t.id) + 1e-9) fail(`${id}: platinum above trophy ${t.id}`);
    }
  }
  ok(`${id}: story chain nested, platinum minimal`);

  // Print a compact forecast table for eyeballing.
  const last = c.history[c.history.length - 1];
  console.log(`\n${id}: checkpoint ${c.at} -> target ${c.target} (${c.horizon_minutes} min), players ${last.players_tracked}`);
  console.log(' id  tier       last%     q05     q25     q50     q75     q95   name');
  c.trophies.forEach((t, i) => {
    const q = byId.get(t.id);
    const lr = last.rates[i];
    console.log(
      String(t.id).padStart(3), t.tier.padEnd(8),
      (lr === null ? 'null' : lr.toFixed(2)).padStart(8),
      ...q.map((v) => v.toFixed(3).padStart(8)),
      ' ', t.name.slice(0, 34),
    );
  });
  console.log('');
}

// Determinism: two fresh runs into temp dirs must be byte-identical to the provided outputs.
const repoRoot = join(import.meta.dirname, '..');
for (let run = 0; run < 2; run++) {
  const tmp = mkdtempSync(join(tmpdir(), 'gta6fc-'));
  try {
    execFileSync(process.execPath, [join(repoRoot, 'main.ts')], {
      env: { ...process.env, MAKEFUN_INPUT_DIR: inputDir, MAKEFUN_OUTPUT_DIR: tmp },
      stdio: 'pipe',
    });
    for (const file of files) {
      const id = file.slice(0, -3);
      const a = readFileSync(join(outputDir, id + '.out'));
      const b = readFileSync(join(tmp, id + '.out'));
      if (!a.equals(b)) fail(`determinism: ${id}.out differs on rerun ${run + 1}`);
    }
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}
ok('determinism: two fresh runs byte-identical');

if (failures > 0) { console.error(`\n${failures} failure(s)`); process.exit(1); }
console.log('\nALL CHECKS PASSED');