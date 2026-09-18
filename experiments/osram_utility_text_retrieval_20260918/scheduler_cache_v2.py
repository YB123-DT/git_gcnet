"""Small GPU queue limited to GPUs 5 and 6, max 6 cache jobs per card."""
from __future__ import annotations
import json, os, subprocess, sys, time
from collections import deque
from pathlib import Path

REPO=Path('/data2/yb/paper/GCNet_TPAMI_pam_episodic_text_20260917')
ROOT=Path('/data2/yb/remote_experiments/osram_utility_text_retrieval_20260918')
GPUS=(5,6); MAX_PER_GPU=6
SEEDS=(66,67,68,69,70); RATES=(0.1,0.2,0.3,0.4,0.5,0.6,0.7)

def tag(rate): return f'{rate:.1f}'.replace('.','p')

def main():
    jobs=[(seed,rate) for seed in SEEDS for rate in RATES]
    queue=deque(jobs); running=[]; counts={gpu:0 for gpu in GPUS}
    manifest={'gpus':list(GPUS),'max_per_gpu':MAX_PER_GPU,'jobs':[]}
    log_dir=ROOT/'cache_v2_logs'; log_dir.mkdir(parents=True,exist_ok=True)
    while queue or running:
        # start jobs
        started=False
        for gpu in GPUS:
            while queue and counts[gpu] < MAX_PER_GPU:
                seed,rate=queue.popleft()
                log=log_dir/f'seed{seed}_rate{tag(rate)}.log'
                env=dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES=str(gpu))
                cmd=[sys.executable,'-u',str(REPO/'experiments/osram_utility_text_retrieval_20260918/build_seed_cache.py'),
                     '--seed',str(seed),'--rate',str(rate),'--device','cuda',
                     '--output-root',str(ROOT/'utility_cache_v2')]
                proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log.open('w'),stderr=subprocess.STDOUT)
                row={'job':f'{seed}_{tag(rate)}','seed':seed,'rate':rate,'gpu':gpu,'pid':proc.pid,'status':'running','retry':0}
                running.append([proc,row]); manifest['jobs'].append(row); counts[gpu]+=1; started=True
        # reap
        still=[]
        for proc,row in running:
            code=proc.poll()
            if code is None:
                still.append([proc,row]); continue
            row['status']='complete' if code==0 else 'failed'; row['exit_code']=code
            counts[row['gpu']]-=1
        running=still
        (ROOT/'GPU_QUEUE.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if not started and not running:
            break
        time.sleep(2)
    print('CACHE_QUEUE_DONE')

if __name__=='__main__': main()
