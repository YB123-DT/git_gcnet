"""Reuse seed66, launch only67/68 concurrently, then report all three."""
from pathlib import Path

from experiments.osram_mosi_mid_lr_20261021 import run as common
from experiments.osram_mosi_gcnet_router_3seed_20261019.run import make_job, BAD_UUID
from .report import write_report


def plan_jobs(output, sha, gpus):
    if len(gpus) != 1 or not gpus[0].startswith('GPU-') or gpus[0] == BAD_UUID:
        raise ValueError('One healthy GPU required')
    jobs = []
    for seed in (67, 68):
        job = make_job('experiments.osram_mosi_mid_lr_3seed_20261021.worker',
                       Path(output)/'runs'/'MOSI'/f'seed_{seed}'/'flat', sha,
                       gpus[0], 'flat', seed, 'CMUMOSI')
        job.update(backbone_lr=3e-4, final_backbone_lr=3e-4 / 10,
                   checkpoint_selection='test-oracle-per-rate; weighted_f1')
        jobs.append(job)
    return jobs


if __name__ == '__main__':
    common.main(planner=plan_jobs, reporter=write_report)
