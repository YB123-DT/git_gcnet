"""Same direct-source MOSI algorithm; explicitly authorized seeds66/67/68."""
from pathlib import Path
from experiments.osram_coupled_source_role_20261020 import worker as common

EXTRA_SOURCES=tuple(str(Path(__file__).with_name(n).relative_to(common.ROOT))
                    for n in ('worker.py','run.py','report.py'))


def main(argv=None):
    common.main('CMUMOSI',argv,source_role_mode='direct',
        extra_source_files=EXTRA_SOURCES,allowed_seeds=(66,67,68))


if __name__=='__main__':main()
