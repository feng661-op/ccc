"""Bounded, reproducible replay of the existing Q3 experiment matrix after physical repair."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
import os
import time
import json
import hashlib

HERE = Path(__file__).resolve().parent
OUT = HERE / 'physical_fix_evidence'
OUT.mkdir(exist_ok=True)
RUNS = [(n, '运行单策略.py', n) for n in ['D', 'A', 'B', 'C', 'D_point', 'D_no6', 'D_no12', 'D_no18', 'D_K5']]
RUNS += [('D_sunk', '运行语义单策略.py', 'sunk'), ('D_stepwise', '运行语义单策略.py', 'stepwise')]


def run_one(item):
    name, script, arg = item
    start = time.time()
    log = OUT / f'replay_{name}.log'
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    print('START', name, flush=True)
    with log.open('w', encoding='utf-8') as f:
        result = subprocess.run([sys.executable, str(HERE/script), arg], cwd=HERE, env=env,
                                stdout=f, stderr=subprocess.STDOUT, timeout=7200)
    row = {'name': name, 'exit_code': result.returncode, 'seconds': time.time()-start,
           'log': str(log), 'source_script_sha256': hashlib.sha256((HERE/script).read_bytes()).hexdigest()}
    print('END', json.dumps(row, ensure_ascii=False), flush=True)
    return row


if __name__ == '__main__':
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(run_one, x) for x in RUNS]):
            rows.append(future.result())
            (OUT/'replay_status.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    if any(r['exit_code'] for r in rows):
        raise SystemExit('Replay failure: inspect per-strategy logs; no acceptance claimed')
    print('ALL_11_REPLAYS_PASSED', flush=True)
