"""Finite E0–E4 MOSI suite, no sweeps or test-driven configuration changes."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

UUIDS={1:'GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049',
       2:'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4',
       3:'GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a',
       6:'GPU-e4cafb17-818e-216a-b94a-7440063a9153'}


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)


def read(path): return json.loads(Path(path).read_text())


def gpu_free(gpu):
    row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free',
                                 '--format=csv,noheader,nounits'],text=True).strip().split(',')
    if row[0].strip()!=UUIDS[gpu]: raise RuntimeError('GPU identity mismatch')
    return int(row[1])


def summarize(root):
    rows=[];checks={}
    for seed in (66,67,68):
        histories={};initial=set();counts=set()
        for mode in ('E0','E1','E2','E3','E4'):
            d=root/'runs'/f'{mode}_seed{seed}'
            completed=read(d/'COMPLETED.json');m=read(d/'metrics.json');p=read(d/'provenance.json')
            h=read(d/'history.json')
            if completed['epochs']!=150 or len(h)!=150 or m['selection_split']!='validation':
                raise ValueError('incomplete/incorrect protocol')
            steps=[b for epoch in h for b in epoch['train']]
            if len(steps)!=300: raise ValueError('unexpected optimizer step count')
            initial.add(p['initialization_sha256']);counts.add(p['parameters'])
            histories[mode]=[(b['mask_hash'],b['pair_hash']) for b in steps]
            rows.append(dict(seed=seed,mode=mode,selected_epoch=m['selected_epoch'],
                wf1=100*m['eight_rate_mean_wf1'],high=100*m['high_missing_mean_wf1'],
                acc=100*sum(v['accuracy'] for v in m['test']['random'].values())/8,
                per_rate=m['test']['random'],persistent=m['test']['persistent'],
                transition=m['test']['transition'],seconds=m['elapsed_seconds']))
        if len(initial)!=1 or len(counts)!=1: raise ValueError('initialization/parameter mismatch')
        if any(histories[m]!=histories['E1'] for m in ('E2','E3','E4')):
            raise ValueError('paired trajectories/pairs differ')
        checks[str(seed)]=dict(initialization_equal=True,parameters_equal=True,
                               mixed_masks_and_pairs_equal=True,optimizer_steps=300)
    means={}
    for mode in ('E0','E1','E2','E3','E4'):
        selected=[r for r in rows if r['mode']==mode]
        means[mode]={k:dict(mean=statistics.mean(r[k] for r in selected),
                            std=statistics.stdev(r[k] for r in selected)) for k in ('wf1','high','acc')}
    write(root/'SUMMARY.json',dict(status='complete',rows=rows,means=means,checks=checks,
        protocol='one validation-selected checkpoint per run; fixed config; no test selection'))
    lines=['# MC-FWC MOSI E0–E4, three seeds','',
        'One checkpoint selected by eight-rate validation mean W-F1; test evaluated only after training.',
        '150 epochs = 100 at 1e-3 + 50 at 1e-4. No method efficacy claim from implementation success.',
        '', '| Mode | ACC mean±sd | W-F1 mean±sd | High W-F1 mean±sd |', '|---|---:|---:|---:|']
    for mode,m in means.items():
        values=[f"{m[k]['mean']:.3f} ± {m[k]['std']:.3f}" for k in ('acc','wf1','high')]
        lines.append('| '+mode+' | '+' | '.join(values)+' |')
    lines+=['','Per-seed/per-rate/persistent/transition/write-utility details: SUMMARY.json and each run metrics.json.',
        'The optional shuffled-utility implementation was not included in this fixed 15-run suite.',
        'Three seeds do not by themselves establish statistical significance. Historical per-rate Test-oracle scores are not direct comparators.']
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return means


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--dataset-root',required=True)
    p.add_argument('--commit',required=True);p.add_argument('--gpus',type=int,nargs='+',default=[1,2])
    p.add_argument('--per-gpu',type=int,default=2);p.add_argument('--smoke',action='store_true')
    p.add_argument('--summarize',action='store_true');args=p.parse_args()
    if args.summarize: summarize(args.root);return
    if len(set(args.gpus))!=len(args.gpus) or any(g not in UUIDS for g in args.gpus):
        raise ValueError('Only verified healthy GPU whitelist; 4 forbidden and7 drained')
    if not 1<=args.per_gpu<=3: raise ValueError('bounded concurrency required')
    args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'LAUNCH.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.root/'LAUNCH.json').exists(): raise ValueError('do not double-submit existing suite')
        specs=[(mode,seed) for seed in ((66,) if args.smoke else (66,67,68))
               for mode in ('E0','E4','E2','E3','E1')]
        active={};records=[];pending=list(specs)
        state=dict(commit=args.commit,source=str(args.source.resolve()),pid=os.getpid(),
            started_utc=datetime.now(timezone.utc).isoformat(),smoke=args.smoke,
            gpu_whitelist={str(g):UUIDS[g] for g in args.gpus},per_gpu=args.per_gpu,
            status='running',records=records,pending=pending)
        write(args.root/'LAUNCH.json',state)
        while pending or active:
            for pid,(child,rec) in list(active.items()):
                code=child.poll()
                if code is not None:
                    expected=Path(rec['output'])/('SMOKE.json' if args.smoke else 'COMPLETED.json')
                    rec.update(exit_code=code,status='complete' if code==0 and expected.exists() else 'failed')
                    del active[pid]
            for gpu in args.gpus:
                while pending and sum(rec['gpu']==gpu for _,rec in active.values())<args.per_gpu:
                    if gpu_free(gpu)<6000: break
                    mode,seed=pending.pop(0);name=f'{mode}_seed{seed}'
                    out=args.root/'runs'/name
                    if out.exists(): raise ValueError('existing run directory; explicit recovery required')
                    command=[sys.executable,'-u','-m','gcnet_missing_m3.train_mc_fwc',
                        '--mode',mode,'--seed',str(seed),'--epochs','150','--device','cuda',
                        '--dataset-root',args.dataset_root,'--output',str(out)]
                    if args.smoke: command+=['--smoke-steps','3']
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUIDS[gpu],OMP_NUM_THREADS='1',
                        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                    with (args.root/f'{name}.log').open('x') as log:
                        child=subprocess.Popen(command,cwd=args.source,env=env,stdout=log,
                            stderr=subprocess.STDOUT,start_new_session=True)
                    rec=dict(mode=mode,seed=seed,gpu=gpu,uuid=UUIDS[gpu],pid=child.pid,
                        output=str(out),command=command,status='running')
                    records.append(rec);active[child.pid]=(child,rec)
                    write(args.root/'LAUNCH.json',state)
                    time.sleep(3)
            state['pending']=list(pending);write(args.root/'LAUNCH.json',state)
            if pending or active: time.sleep(10)
        state['status']='complete' if all(r['status']=='complete' for r in records) else 'failed'
        state['completed_utc']=datetime.now(timezone.utc).isoformat()
        write(args.root/'LAUNCH.json',state)
        if state['status']!='complete': raise RuntimeError('suite includes failed run; inspect preserved logs')
        if not args.smoke: summarize(args.root)
        else:
            reports={mode:read(args.root/'runs'/f'{mode}_seed66/SMOKE.json') for mode,_ in specs}
            if len({v['initialization_sha256'] for v in reports.values()})!=1:
                raise ValueError('smoke initialization mismatch')
            sequences={mode:[(s['mask_hash'],s['pair_hash']) for s in r['steps']] for mode,r in reports.items()}
            if any(sequences[m]!=sequences['E1'] for m in ('E2','E3','E4')):
                raise ValueError('smoke pairing mismatch')
            write(args.root/'SUMMARY.json',dict(initialization_equal=True,mixed_masks_pairs_equal=True,
                modes={m:dict(seconds=r['elapsed_seconds'],peak_cuda_bytes=r['peak_cuda_bytes'],
                    losses=[s['task_loss'] for s in r['steps']],
                    extra_norms=[s['diagnostics']['extra_gradient_norm'] for s in r['steps']]) for m,r in reports.items()}))


if __name__=='__main__': main()
