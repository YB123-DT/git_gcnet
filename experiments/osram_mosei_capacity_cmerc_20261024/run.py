"""The user-corrected eight-run queue: cmerc, constant1e-4, GPUs0/1 only."""
from pathlib import Path

from experiments.osram_mosei_flat_nested_capacity_20261024 import run as common
from .worker import MODES

PYTHON='/data2/yb/conda_envs/cmerc-repro/bin/python'


def plan_jobs(output,source_commit):
    jobs=[]
    for i,mode in enumerate(MODES):
        uuid=common.GPU0 if i<4 else common.GPU1
        path=Path(output)/'runs'/mode
        command=[PYTHON,'-u','-m','experiments.osram_mosei_capacity_cmerc_20261024.worker',
            '--mode',mode,'--phase','train','--seed','66','--gpu-uuid',uuid,
            '--source-commit',source_commit,'--output',str(path)]
        jobs.append(dict(mode=mode,output=str(path),command=command,log=str(path)+'.log',
            gpu_uuid=uuid,reserve_mib=8192 if mode=='f512_n256' else 6144,status='pending'))
    return jobs


def main(argv=None):
    from .report import summarize
    common.main(argv,planner=plan_jobs,reporter=summarize,external=None)


if __name__=='__main__':main()
