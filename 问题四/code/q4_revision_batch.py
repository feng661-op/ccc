"""Independent finite numerical jobs; three subprocesses, no model delegation."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import os,sys,subprocess,json
C=Path(__file__).resolve().parent;L=C/'inheritance_revision/logs';L.mkdir(exist_ok=True)
def launch(name,args):
    env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (L/(name+'.log')).open('w',encoding='utf-8') as f:
        for a in args:
            r=subprocess.run([sys.executable,'-B','-u','q4_revision_run.py',*a],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError(name+' failed; see '+str(L/(name+'.log')))
    print(json.dumps({'finished':name}),flush=True)
if __name__=='__main__':
    tasks=[('main2',[['formal','--branch','q4_2']]),('main3',[['formal','--branch','q4_3']]),
        ('semantic3',[['january','--branch','q4_3','--clock','adjustment_time'],['formal','--branch','q4_3','--clock','adjustment_time']]),
        ('matched2',[['formal','--branch','q4_2','--mode','matched_fixed_decision']]),
        ('matched3',[['formal','--branch','q4_3','--mode','matched_fixed_decision']]),
        ('backup3',[['formal','--branch','q4_3','--mode','backup']])]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(launch,*t) for t in tasks]
        for f in as_completed(futures):f.result()
    print('ALL_FRESH_FORMAL_AND_SEMANTIC_RUNS_DONE',flush=True)
