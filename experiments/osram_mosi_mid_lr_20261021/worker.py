"""Original MOSI Flat seed66; two explicitly approved 100+50 LR schedules."""
import argparse
import copy
from pathlib import Path

from experiments.osram_mosi_router_lr1e5_20261020 import worker as common

LEARNING_RATES = (5e-4, 3e-4)
# Exactly the opt-in source-role integration since the original27b8c1f Flat.
# All added computations/parameters are gated OFF; unknown edits still fail.
DISABLED_SOURCE_ROLE_UPDATES = {
    'gcnet_missing_m3/osram.py': (
        '63d22598ec2948ecabb0e357c84bc5b5d5465ed225973ff09d29c2288458a97e',
        'd02e9b5da370caf264c0a26496cda9e2dceb28c887f425bfe73d7f496c19a838'),
    'gcnet_missing_m3/model.py': (
        '1d36aa12108c9c3c0e65e1ec774c87b8fd65e8fb0bbecfb4c72b85aac4118af2',
        'ce1afb7b98279cb1d62afc1a3ab1420d167580d02afdbd38d38f7e9e454b3ee2'),
    'gcnet_missing_m3/train_gcnet.py': (
        'f93230ff7e876ddb2f81db2d464330da2eec2f82b30087aa68a4cf77b6e15d16',
        '12267fc232895296be802c3d8f673003190ad1c88e9a7dbe4216879a89b6751f'),
}
EXTRA_SOURCES = tuple(str(Path(__file__).with_name(name).relative_to(common.ROOT))
                      for name in ('worker.py', 'run.py', 'report.py'))


def validate_reference(identity, prior, initial, *, backbone_lr):
    """Document exact disabled integration, never claim source byte identity."""
    if (identity['config'].get('osram_source_role') is not False
            or identity['config'].get('osram_source_role_mode') != 'residual'):
        raise ValueError('Only original Flat with source-role disabled is approved')
    normalized = copy.deepcopy(prior)
    normalized['identity']['config'].setdefault('osram_source_role', False)
    normalized['identity']['config'].setdefault('osram_source_role_mode', 'residual')
    changes = {}
    for name, (before, after) in DISABLED_SOURCE_ROLE_UPDATES.items():
        old = prior['identity']['source_hashes'][name]
        new = identity['source_hashes'][name]
        if old != new:
            if (old, new) != (before, after):
                raise ValueError('Unverified numerical source evolution: ' + name)
            changes[name] = dict(reference=old, current=new)
            normalized['identity']['source_hashes'][name] = new
    proof = common.validate_reference(identity, normalized, initial, backbone_lr=backbone_lr)
    proof.update(core_sources_exact=not bool(changes), disabled_source_role_compatible=True,
                 disabled_source_role_source_changes=changes)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backbone-lr', type=float, choices=LEARNING_RATES, required=True)
    args, remainder = parser.parse_known_args(argv)
    common.main(remainder, backbone_lr=args.backbone_lr, final_backbone_lr=args.backbone_lr / 10,
                extra_source_files=EXTRA_SOURCES, allowed_modes=('flat',),
                reference_validator=validate_reference)


if __name__ == '__main__':
    main()
