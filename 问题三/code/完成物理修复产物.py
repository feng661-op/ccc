"""Rebuild and verify all Q3 artifacts after the bounded replay matrix.
Usage: python 完成物理修复产物.py [first_step_index]
"""
from pathlib import Path
import subprocess,sys,os,json,time,shutil
HERE=Path(__file__).resolve().parent;OUT=HERE/'physical_fix_evidence'
replays=json.loads((OUT/'replay_status.json').read_text(encoding='utf-8'))
assert len(replays)==11 and all(x['exit_code']==0 for x in replays), 'Complete all replay jobs first'
steps=[
 ['汇总主结果.py'],['扩展建议验证.py'],['语义鲁棒性.py'],['边际价值审计.py'],
 ['写出result3.py'],['验证问题3.py'],['验证物理修复矩阵.py'],
 ['-m','pytest','-q','test_physical_semantics.py','--junitxml=physical_fix_evidence/targeted_pytest.xml'],
 ['汇总最终指标.py'],['同步问题3论文输出.py'],['生成问题3图表.py'],['同步问题3图表.py'],
 ['最终总验收.py'],['验证问题3论文输出.py']]
start=int(sys.argv[1]) if len(sys.argv)>1 else 0
assert 0<=start<len(steps)
env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
results=json.loads((OUT/'postprocess_status.json').read_text(encoding='utf-8')) if start and (OUT/'postprocess_status.json').exists() else []
results=[x for x in results if x['index']<start]
for i,args in enumerate(steps):
 if i<start:continue
 if i==13:
  rob=HERE.parent/'output'/'robustness'
  shutil.copy2(HERE/'final_acceptance.json',rob/'final_acceptance.json')
 log=OUT/f'post_{i:02}.log';t=time.time();print('START',i,args,flush=True)
 with log.open('w',encoding='utf-8') as f:
  p=subprocess.run([sys.executable,*args],cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 row={'index':i,'args':args,'exit_code':p.returncode,'seconds':time.time()-t,'log':str(log)}
 results.append(row);(OUT/'postprocess_status.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
 print('END',i,p.returncode,round(row['seconds'],2),flush=True)
 if p.returncode:raise SystemExit(f'Failed step {i}; inspect {log}; do not publish')
rob=HERE.parent/'output'/'robustness'
for name in ['physical_matrix_validation.json','legacy_physical_audit.json','targeted_pytest.xml']:
 shutil.copy2(OUT/name,rob/name)
shutil.copy2(HERE/'final_acceptance.json',rob/'final_acceptance.json')
print('ALL_Q3_ARTIFACTS_REBUILT_AND_VERIFIED',flush=True)
