"""Nine fresh MOSI memory arms, real smokes and measured GPU7/GPU0 admission."""
import sys

from experiments.osram_mosi_v3_capacity_20261022 import run as queue
from .config import MODES
from .report import summarize


def worker_command(args, mode, phase, uuid):
    if mode not in MODES or phase not in ('smoke', 'train'):
        raise ValueError('Only nine approved memory arms and smoke/train phases')
    return [sys.executable, '-u', '-m', 'experiments.osram_mosi_memory_grid_20261022.worker',
            '--mode', mode, '--phase', phase, '--seed', '66', '--gpu-uuid', uuid,
            '--source-commit', args.source_commit,
            '--output', str(args.output / ('smoke' if phase == 'smoke' else 'runs') / mode)]


def assignment(args, mode, index):
    slot = index % 2
    uuid = args.gpu_uuids[slot]
    if queue.gpu_free(uuid)[0] != (7, 0)[slot]:
        raise ValueError('Memory queue requires physical GPU7 then GPU0; GPU4 is forbidden')
    return uuid


def main(argv=None):
    return queue.main(argv, modes=MODES, command_factory=worker_command,
                      report_factory=summarize, assignment_factory=assignment,
                      default_per_gpu=5, max_per_gpu=5)


if __name__ == '__main__':
    main()
