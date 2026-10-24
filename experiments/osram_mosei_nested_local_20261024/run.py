"""Four finite MOSEI tests around Nested32; reuse the proven GPU2 controller."""
from pathlib import Path

from experiments.osram_mosei_nested_combinations_20261024 import run as common
from experiments.osram_nested_cmerc_tuning_20261023 import run as screen

GPU = common.GPU
VARIANTS = (
    ('nested32_lr5e5', 'nested_dim32', 5e-5),
    ('nested32_lr2e4', 'nested_dim32', 2e-4),
    ('nested32_depth2', 'mosei_nested_dim32_depth2', None),
    ('nested32_groups4', 'mosei_nested_d32_l3_g4_m32', None),
)


def plan_jobs(output, source_commit, gpu_uuid):
    if gpu_uuid != GPU:
        raise ValueError('Only authorized biggpu physical GPU2 may admit these jobs')
    jobs = []
    for variant, method, nested_lr in VARIANTS:
        path = Path(output) / 'runs/CMUMOSEI/fold_1/seed_66' / variant
        command = [screen.PYTHON, '-u', '-m', screen.MODULE,
            '--dataset', 'CMUMOSEI', '--method', method, '--seed', '66', '--fold', '1',
            '--phase', 'train', '--gpu-uuid', gpu_uuid, '--source-commit', source_commit,
            '--output', str(path)]
        if nested_lr is not None:
            command += ['--nested-learning-rate', str(nested_lr)]
        jobs.append(dict(variant=variant, dataset='CMUMOSEI',method=method,seed=66,fold=1,
            epochs=60,learning_rate=1e-4,nested_learning_rate=nested_lr,
            phase='train',source_commit=source_commit,gpu_uuid=gpu_uuid,
            output=str(path),log=str(path)+'.log',status='pending',command=command,
            mask_protocol='gcnet-random-mask-v1'))
    return jobs


def main(argv=None):
    from .report import write_report, validate_source_extension
    common.main(argv, planner=plan_jobs, report_writer=write_report,
                source_validator=validate_source_extension)


if __name__ == '__main__':
    main()
