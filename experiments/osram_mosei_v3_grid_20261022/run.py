"""Four new MOSEI arms; reuse two completed ones, two healthy GPUs."""
import sys

from experiments.osram_mosi_v3_capacity_20261022 import run as queue
from .worker import MODES
from .report import summarize


def worker_command(args, mode, phase, uuid):
    if mode not in MODES or phase not in ('smoke', 'train'):
        raise ValueError('Only four untrained MOSEI grid arms')
    return [sys.executable, '-u', '-m', 'experiments.osram_mosei_v3_grid_20261022.worker',
            '--mode', mode, '--phase', phase, '--seed', '66', '--gpu-uuid', uuid,
            '--source-commit', args.source_commit,
            '--output', str(args.output / ('smoke' if phase == 'smoke' else 'runs') / mode)]


def assignment(args, mode, index):
    return args.gpu_uuids[index % 2]


def main(argv=None):
    return queue.main(argv, modes=MODES, command_factory=worker_command,
        report_factory=summarize, assignment_factory=assignment,
        default_per_gpu=2, max_per_gpu=2)


if __name__ == '__main__':
    main()
