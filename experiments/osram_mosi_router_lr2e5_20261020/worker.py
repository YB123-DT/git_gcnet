"""Finite MOSI seed66 experiment: backbone constant2e-5; all else unchanged."""
from pathlib import Path

from experiments.osram_mosi_router_lr1e5_20261020 import worker as common

BACKBONE_LR = 2e-5
EXTRA_SOURCES = tuple(str(Path(__file__).with_name(name).relative_to(common.ROOT))
                      for name in ('worker.py', 'run.py', 'report.py'))


def main(argv=None):
    common.main(argv, backbone_lr=BACKBONE_LR, extra_source_files=EXTRA_SOURCES)


if __name__ == '__main__':
    main()
