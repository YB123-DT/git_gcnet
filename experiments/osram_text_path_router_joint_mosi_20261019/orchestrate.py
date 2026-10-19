"""MOSI seed66: one smoke, then matched Flat/Router runs on one healthy GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MODULE='experiments.osram_text_path_router_joint_mosi_20261019.run'
BAD_UUID='GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'


def command(mode,phase,source,output,uuid):
    return [sys.executable,'-m',MODULE,'--mode',mode,'--phase',phase,
            '--source-commit',source,'--output',str(output),'--gpu-uuid',uuid]


def gpu_info(uuid):
    if uuid==BAD_UUID:
        raise ValueError('biggpu GPU4 is forbidden')
    text=subprocess.check_output(['nvidia-smi','-i',uuid,
        '--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    index,actual,used,total=[value.strip() for value in text.strip().split(',')]
    if int(index)==4 or actual!=uuid:
        raise ValueError('Unverified or forbidden physical GPU')
    return dict(index=int(index),uuid=uuid,used_mib=int(used),total_mib=int(total))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output, never overwrite old results')
    info=gpu_info(args.gpu_uuid)
    args.output.mkdir(parents=True)
    state=dict(status='preflight',source_commit=args.source_commit,pid=os.getpid(),gpu=info,jobs=[],
               scope='MOSI seed66 only; one joint smoke followed by Flat and Joint150')

    def record():
        path=args.output/'QUEUE.json';temporary=path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state,indent=2,sort_keys=True,allow_nan=False)+'\n')
        temporary.replace(path)

    def start(mode,phase):
        output=args.output/('smoke' if phase=='smoke' else 'runs')/mode
        output.parent.mkdir(parents=True,exist_ok=True)
        log=output.with_suffix('.log')
        argv=command(mode,phase,args.source_commit,output,args.gpu_uuid)
        with log.open('x') as stream:
            process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu_uuid,
                         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        job=dict(mode=mode,phase=phase,pid=process.pid,output=str(output),log=str(log),
                 command=argv,status='running')
        state['jobs'].append(job);record()
        return process,job

    def finish(process,job):
        code=process.wait()
        path=Path(job['output'])/'SUMMARY.json'
        summary=json.loads(path.read_text()) if path.exists() else {}
        job.update(exit_code=code,status='complete' if code==0 and summary.get('status')=='complete' else 'failed')
        record()
        if job['status']!='complete':
            raise RuntimeError('Child failed; inspect '+job['log'])
        return summary

    try:
        state['status']='smoke';record()
        smoke=finish(*start('joint','smoke'))
        current=gpu_info(args.gpu_uuid)
        required=2*smoke['peak_memory_mib']*1.5+2048
        if current['total_mib']-current['used_mib']<required:
            raise RuntimeError('Not enough VRAM for paired jobs; no batch/protocol change permitted')
        state.update(status='training',memory_required_mib=required);record()
        children=[start(mode,'train') for mode in ('flat','joint')]
        results={job['mode']:finish(process,job) for process,job in children}
        if results['flat']['initial_backbone_sha256']!=results['joint']['initial_backbone_sha256']:
            raise ValueError('MOSI initializations differ')
        report=args.output/'SUMMARY.json'
        report.write_text(json.dumps(results,indent=2,sort_keys=True,allow_nan=False)+'\n')
        state.update(status='complete',completed_at=time.time())
    except BaseException as error:
        # No retries or killing of unrelated/other already-started experiments.
        state.update(status='failed',error=repr(error));raise
    finally:
        record()


if __name__=='__main__':
    main()
