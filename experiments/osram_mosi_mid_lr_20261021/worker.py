"""Original MOSI Flat seed66; two explicitly approved 100+50 LR schedules."""
import argparse
from pathlib import Path

from experiments.osram_mosi_router_lr1e5_20261020 import worker as common

LEARNING_RATES = (5e-4, 3e-4)
EXTRA_SOURCES = tuple(str(Path(__file__).with_name(name).relative_to(common.ROOT))
                      for name in ('worker.py', 'run.py', 'report.py'))


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backbone-lr', type=float, choices=LEARNING_RATES, required=True)
    args, remainder = parser.parse_known_args(argv)
    common.main(remainder, backbone_lr=args.backbone_lr, final_backbone_lr=args.backbone_lr / 10,
                extra_source_files=EXTRA_SOURCES, allowed_modes=('flat',))


if __name__ == '__main__':
    main()
