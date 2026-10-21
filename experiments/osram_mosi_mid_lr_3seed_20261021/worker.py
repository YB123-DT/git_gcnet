"""Complete only seeds67/68 of the original Flat3e-4/3e-5 MOSI150 screen."""
from pathlib import Path

from experiments.osram_mosi_router_lr1e5_20261020 import worker as common
from experiments.osram_mosi_mid_lr_20261021.worker import validate_reference, EXTRA_SOURCES

SOURCES = EXTRA_SOURCES + tuple(str(Path(__file__).with_name(name).relative_to(common.ROOT))
                                for name in ('worker.py', 'run.py', 'report.py'))


def main(argv=None):
    common.main(argv, backbone_lr=3e-4, final_backbone_lr=3e-4 / 10,
                extra_source_files=SOURCES, allowed_modes=('flat',),
                allowed_seeds=(67, 68), reference_seed_parent=common.REFERENCE.parent,
                reference_validator=validate_reference)


if __name__ == '__main__':
    main()
