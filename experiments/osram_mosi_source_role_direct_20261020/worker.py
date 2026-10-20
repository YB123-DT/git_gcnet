"""Only MOSI seed66: directly weighted sources with identity-centred sigmoid."""
from pathlib import Path
from experiments.osram_coupled_source_role_20261020 import worker as common

EXTRA_SOURCES=tuple(str(Path(__file__).with_name(n).relative_to(common.ROOT))
                    for n in ('worker.py','run.py','report.py'))


def main(argv=None):
    common.main('CMUMOSI',argv,source_role_mode='direct',extra_source_files=EXTRA_SOURCES)


if __name__=='__main__':main()
