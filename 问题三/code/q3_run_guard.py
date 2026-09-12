"""Coordinate disjoint local replay processes and reuse only this audit's outputs."""
from pathlib import Path
import atexit
import os
import time
import numpy as np

_LOCKS=[]


def claim(name):
    here=Path(__file__).resolve().parent
    cutoff=here/'cross_problem_evidence'/'parallel_replay_epoch.txt'
    if not cutoff.exists():
        return
    # This optional local replay lock must not make ordinary Linux/macOS
    # execution depend on the Windows runtime when no shared replay is active.
    import msvcrt
    epoch=float(cutoff.read_text(encoding='ascii'))
    directory=here/'cross_problem_evidence'/'run_locks';directory.mkdir(exist_ok=True)
    stream=(directory/(name+'.lock')).open('a+b');stream.seek(0)
    if os.fstat(stream.fileno()).st_size==0:stream.write(b'0');stream.flush()
    while True:
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            break
        except OSError:
            time.sleep(.5)
    _LOCKS.append(stream)
    def release():
        stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1);stream.close()
    atexit.register(release)
    arr=here/f'run_{name}.npz';metric=here/f'metric_{name}.json'
    if arr.exists() and metric.exists() and min(arr.stat().st_mtime,metric.stat().st_mtime)>epoch:
        with np.load(arr) as z:
            complete=(str(z['execution_mode'])=='q2_reserve_feedback' and z['B'].shape==(365,144)
                      and z['A_stage'].shape==(365,4,144) and np.isfinite(z['soc_path']).all())
        if complete:
            print('Reusing completed current-audit replay:',name,flush=True)
            raise SystemExit(0)
