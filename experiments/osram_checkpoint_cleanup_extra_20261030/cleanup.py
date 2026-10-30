"""Eight-family follow-up; reuse the prior exact-manifest safety executor."""
import argparse
from contextlib import contextmanager
from pathlib import Path

from experiments.osram_checkpoint_cleanup_20261030 import cleanup as legacy

FAMILIES = {
    'osram_v3_classification_20261021',
    'osram_mosi_v3_capacity_20261022',
    'osram_mosei_v3_grid_20261022',
    'osram_mosei_v3_capacity_20261022',
    'osram_mosi_hparam_verify_20260918',
    'osram_mosi_write_gate_gap_strength_20260920',
    'osram_nojepa_capacity2400_20260919',
    'osram_nojepa_capacity2400_200_20260919_retry',
}


def preserve_run(family, run):
    del family
    if run.name in ('none', 'flat', 'baseline', 'small_flat', 'f512_v128'):
        return True
    if run.name.endswith('_plain'):
        return True
    return any(p.startswith(('cfg01_', 'cfg11_', 'cfg84_', 'cfg88_'))
               or p == 'baseline' or p.endswith('_baseline') for p in run.parts)


@contextmanager
def scope():
    """Single-process CLI scope; never alter the first pass's persisted source."""
    previous_families, previous_rule = legacy.FAMILIES, legacy.preserve_run
    legacy.FAMILIES, legacy.preserve_run = FAMILIES, preserve_run
    try:
        yield
    finally:
        legacy.FAMILIES, legacy.preserve_run = previous_families, previous_rule


def inventory(output):
    raw = output.with_name('RAW_PLAN.json')
    with scope():
        legacy.inventory(raw)
    plan = legacy.common.read(raw)
    plan.update(driver_sha256=legacy.common.sha(__file__), families=sorted(FAMILIES),
        authorization='2026-10-30 follow-up user request: remove more obsolete checkpoints; eight extra explicitly recorded families')
    legacy.common.save_new(output, plan)
    print('Final eight-family manifest: ' + str(output), flush=True)


def execute(manifest, output):
    plan = legacy.common.read(manifest)
    if (plan.get('driver_sha256') != legacy.common.sha(__file__)
            or plan.get('families') != sorted(FAMILIES)):
        raise ValueError('Require unchanged authorized eight-family driver/manifest')
    with scope():
        legacy.execute(manifest, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('inventory').add_argument('--output', required=True, type=Path)
    ex = commands.add_parser('execute')
    ex.add_argument('--manifest', required=True, type=Path)
    ex.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'inventory':
        inventory(args.output)
    else:
        execute(args.manifest, args.output)


if __name__ == '__main__':
    main()
