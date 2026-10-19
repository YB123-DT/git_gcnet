"""After MOSI succeeds, complete only the missing IEMOCAP6 folds1..4."""
import hashlib
import json
from pathlib import Path

from experiments.osram_mosi_gcnet_router_3seed_20261019.run import make_job, parser, run_queue
from experiments.osram_mosi_gcnet_router_3seed_20261019.report import CORE

MODULE = 'experiments.osram_iemocap6_router_fivefold_20261019.worker'
OLD66 = Path('/home/qinjinghui/remote_experiments/osram_text_path_router_joint_20261019/execution_v1/runs/IEMOCAPSix')
OLD6768 = Path('/home/qinjinghui/remote_experiments/osram_text_path_router_3seed_20261019/execution_v1/runs/IEMOCAPSix')


def fold5_references():
    return [dict(dataset='IEMOCAPSix', fold=5, seed=seed, mode=mode, reused=True,
                 output=str((OLD66 if seed == 66 else OLD6768/f'seed_{seed}')/mode))
            for seed in (66, 67, 68) for mode in ('flat', 'joint')]


def plan_jobs(output, source_commit, gpus):
    if len(gpus) != 3:
        raise ValueError('This finite plan expects three GPU UUIDs')
    return [make_job(MODULE, Path(output)/'runs'/'IEMOCAPSix'/f'fold_{fold}'/f'seed_{seed}'/mode,
                     source_commit, gpus[index], mode, seed, 'IEMOCAPSix', fold=fold)
            for fold in (1, 2, 3, 4) for index, seed in enumerate((66, 67, 68)) for mode in ('flat', 'joint')]


def verify_references(references):
    """Fail before training if a protected fold5 result cannot be reused safely."""
    root = Path(__file__).resolve().parents[2]
    for ref in references:
        path = Path(ref['output'])/'SUMMARY.json'
        summary = json.loads(path.read_text())
        cfg = summary['identity']['config']
        if (summary.get('status') != 'complete' or summary.get('phase') != 'train'
                or summary.get('dataset') != 'IEMOCAPSix' or summary.get('seed') != ref['seed']
                or summary.get('mode') != ref['mode'] or cfg.get('fold') != 5
                or cfg.get('epochs') != 100 or summary.get('epoch') != 100
                or cfg.get('missing_mask_protocol') != 'gcnet-random-mask-v1'):
            raise ValueError('Protected fold5 reference incomplete or mismatched: ' + str(path))
        for name in CORE:
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != summary['identity']['source_hashes'][name]:
                raise ValueError('Protected fold5 numerical core changed: ' + name)
        ref.update(summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   source_commit=summary['source_commit'])
    return references


def main():
    args = parser(__doc__).parse_args()
    if args.after_queue is None:
        raise ValueError('Require --after-queue MOSI/QUEUE.json to enforce the approved sequential suites')
    if args.poll_seconds <= 0:
        raise ValueError('poll-seconds must be positive')
    from .report import write_report
    run_queue(args, plan_jobs(args.output, args.source_commit, args.gpus), write_report,
              references=verify_references(fold5_references()))


if __name__ == '__main__':
    main()
