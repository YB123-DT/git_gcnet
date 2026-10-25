"""Persistent finite queue: four smokes, then four concurrent formal runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MODES = ('neumiss','signature','ridge','csdi')


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args()
    args.root.mkdir(parents=True,exist_ok=False)
    state=dict(status='preflight',pid=os.getpid(),gpu_uuid=args.gpu_uuid,
               source_commit=args.source_commit,python=sys.executable,runs={})
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_uuid
    if 'cmerc-repro' not in sys.executable:
        raise RuntimeError('Use the existing cmerc-repro environment')
    import torch
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    state['gpu']=guard_gpu(args.gpu_uuid)
    free=int(subprocess.check_output(['nvidia-smi','-i',args.gpu_uuid,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free < 16000:
        raise RuntimeError('Require16GiB free for four measured lightweight branches')
    write(args.root/'QUEUE.json',state)
    for phase in ('smoke','train'):
        state['status']=phase
        running={}
        for mode in MODES:
            output=args.root/(phase+'_'+mode)
            cmd=[sys.executable,'-u','-m','experiments.osram_four_block_mosi_20261025.run',
                 '--mode',mode,'--output',str(output),'--gpu-uuid',args.gpu_uuid,
                 '--source-commit',args.source_commit]
            if phase=='smoke':cmd.append('--smoke')
            stream=(args.root/(phase+'_'+mode+'.log')).open('w')
            proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
            running[mode]=(proc,stream)
            state['runs'][phase+'_'+mode]=dict(pid=proc.pid,command=cmd,output=str(output),status='running')
            write(args.root/'QUEUE.json',state)
        while running:
            for mode,(proc,stream) in list(running.items()):
                code=proc.poll()
                if code is None:continue
                stream.close()
                result=args.root/(phase+'_'+mode)/'SUMMARY.json'
                summary=json.loads(result.read_text()) if result.exists() else {}
                expected='smoke_passed' if phase=='smoke' else 'complete'
                ok=code==0 and summary.get('status')==expected
                state['runs'][phase+'_'+mode].update(status=expected if ok else 'failed',exit_code=code)
                del running[mode]
                write(args.root/'QUEUE.json',state)
            if running:time.sleep(5)
        if any(state['runs'][phase+'_'+m]['status']=='failed' for m in MODES):
            state['status']='failed';write(args.root/'QUEUE.json',state)
            raise RuntimeError('At least one child failed; see QUEUE.json and per-method logs')
    state['status']='complete';write(args.root/'QUEUE.json',state)
    results={m:json.loads((args.root/('train_'+m)/'SUMMARY.json').read_text()) for m in MODES}
    baseline=results['neumiss']['baseline_aggregate']
    table=['# Four-branch MOSI results','','INTERNAL TEST-ORACLE SCREEN','',
           '| Method | Mean ACC | Mean W-F1 | High W-F1 | Δ mean W-F1 |','|---|---:|---:|---:|---:|']
    table.append(f"| Frozen Flat | {baseline['mean']['ACC']:.4f} | {baseline['mean']['WF1']:.4f} | {baseline['high']['WF1']:.4f} | 0 |")
    for mode,result in results.items():
        assert result['baseline_aggregate']==baseline
        a=result['aggregate']
        table.append(f"| {mode} | {a['mean']['ACC']:.4f} | {a['mean']['WF1']:.4f} | {a['high']['WF1']:.4f} | {a['mean']['WF1']-baseline['mean']['WF1']:+.4f} |")
    (args.root/'RESULT.md').write_text('\n'.join(table)+'\n')
    write(args.root/'RESULT.json',{m:dict(aggregate=r['aggregate'],best_by_rate=r['best_by_rate']) for m,r in results.items()})


if __name__=='__main__':main()
