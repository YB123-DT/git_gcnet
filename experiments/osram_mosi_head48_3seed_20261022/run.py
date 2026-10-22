"""Only two new head48 jobs, concurrent on one healthy GPU; reuse seed66."""
from pathlib import Path

from experiments.osram_mosi_mid_lr_20261021 import run as common
from experiments.osram_mosi_gcnet_router_3seed_20261019.run import make_job,BAD_UUID
from .report import write_report


def plan_jobs(output,sha,gpus):
    if len(gpus)!=1 or not gpus[0].startswith('GPU-') or gpus[0]==BAD_UUID:
        raise ValueError('Require one healthy GPU; physical GPU4 forbidden')
    jobs=[]
    for seed in (67,68):
        job=make_job('experiments.osram_mosi_head48_3seed_20261022.worker',
            Path(output)/'runs'/f'seed_{seed}',sha,gpus[0],'head_48',seed,'CMUMOSI')
        job.update(backbone_lr=3e-4,final_backbone_lr=3e-4/10,
                   checkpoint_selection='test-oracle-per-rate; weighted_f1')
        jobs.append(job)
    return jobs


if __name__=='__main__':
    common.main(planner=plan_jobs,reporter=write_report)
