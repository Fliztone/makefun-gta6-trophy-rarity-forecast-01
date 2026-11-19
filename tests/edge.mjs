// Edge-case tests (test-only): empty history, single read, nulls, floor/ceiling trophies, tiny horizons.
// Each must produce a schema-valid forecast without crashing. Usage: node tests/edge.mjs
import { mkdtempSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';

const ROOT = join(import.meta.dirname, '..');
const KIT = 'D:/Project X/makefun/gta6-trophy-rarity-forecast-01-kit';
const base = JSON.parse(readFileSync(join(KIT, 'input', 'p00.in'), 'utf8'));

function mkCase(id, over) {
  return {
    schema: 'makefun/gta6-trophies-case/v1',
    id, at: '2026-11-21T11:30:00Z', horizon_minutes: 6510, target: '2026-11-26T00:00:00Z',
    levels: [0.05, 0.25, 0.5, 0.75, 0.95], game: base.game, trophies: base.trophies,
    history: [], notes: 'edge test', ...over,
  };
}
const read = (at, players, rates) => ({ slot: at, at, players_tracked: players, trophies_earned: 1000, rates });
const flat = base.history[0].rates;

const cases = [
  mkCase('e-empty', { history: [] }),
  mkCase('e-single', { history: [read('2026-11-21T11:00:00Z', 50000, flat)] }),
  mkCase('e-nulls', {
    history: [
      read('2026-11-20T11:00:00Z', 40000, flat.map((v, i) => (i % 3 === 0 ? null : v))),
      read('2026-11-21T11:00:00Z', 50000, flat),
    ],
  }),
  mkCase('e-bounds', {
    history: [
      read('2026-11-20T11:00:00Z', 40000, flat.map((v, i) => (i === 3 ? 0.0 : i === 4 ? 100.0 : v))),
      read('2026-11-21T11:00:00Z', 50000, flat.map((v, i) => (i === 3 ? 0.0 : i === 4 ? 100.0 : v))),
    ],
  }),
  mkCase('e-short-horizon', { horizon_minutes: 10, target: '2026-11-21T11:40:00Z', history: [read('2026-11-21T11:00:00Z', 50000, flat)] }),
  mkCase('e-allnull', { history: [read('2026-11-21T11:00:00Z', 50000, flat.map(() => null))] }),
];

let failures = 0;
const tmpIn = mkdtempSync(join(tmpdir(), 'gta6edge-in-'));
const tmpOut = mkdtempSync(join(tmpdir(), 'gta6edge-out-'));
try {
  for (const c of cases) writeFileSync(join(tmpIn, c.id + '.in'), JSON.stringify(c));
  execFileSync(process.execPath, [join(ROOT, 'main.ts')], {
    env: { ...process.env, MAKEFUN_INPUT_DIR: tmpIn, MAKEFUN_OUTPUT_DIR: tmpOut },
    stdio: 'pipe',
  });
  for (const c of cases) {
    const raw = readFileSync(join(tmpOut, c.id + '.out'), 'utf8');
    const o = JSON.parse(raw);
    const ids = new Set(o.trophies.map((r) => r.id));
    let good = Object.keys(o).length === 1 && o.trophies.length === c.trophies.length && ids.size === c.trophies.length;
    for (const row of o.trophies) {
      if (Object.keys(row).sort().join(',') !== 'id,q') good = false;
      if (row.q.length !== 5) good = false;
      for (let k = 0; k < 5; k++) {
        const v = row.q[k];
        if (!Number.isFinite(v) || v < 0 || v > 100 || (k > 0 && v < row.q[k - 1])) good = false;
      }
    }
    if (Buffer.byteLength(raw, 'utf8') > 65536) good = false;
    console.log(`${good ? 'ok' : 'FAIL'}: ${c.id} (${Buffer.byteLength(raw, 'utf8')} bytes)`);
    if (!good) failures++;
  }
  // Determinism across the whole edge set.
  const tmpOut2 = mkdtempSync(join(tmpdir(), 'gta6edge-out2-'));
  try {
    execFileSync(process.execPath, [join(ROOT, 'main.ts')], {
      env: { ...process.env, MAKEFUN_INPUT_DIR: tmpIn, MAKEFUN_OUTPUT_DIR: tmpOut2 },
      stdio: 'pipe',
    });
    for (const c of cases) {
      const a = readFileSync(join(tmpOut, c.id + '.out'));
      const b = readFileSync(join(tmpOut2, c.id + '.out'));
      if (!a.equals(b)) { console.log(`FAIL: determinism ${c.id}`); failures++; }
    }
    console.log('ok: edge determinism');
  } finally {
    rmSync(tmpOut2, { recursive: true, force: true });
  }
} finally {
  rmSync(tmpIn, { recursive: true, force: true });
  rmSync(tmpOut, { recursive: true, force: true });
}
if (failures) { console.error(`${failures} failure(s)`); process.exit(1); }
console.log('ALL EDGE CHECKS PASSED');