"""Four new MOSEI seed66 capacity arms; the two existing 1024 arms are reused."""
from dataclasses import replace

from experiments.osram_mosei_v3_capacity_20261022 import worker as base

MODES = {'f512_plain': (512, 0, 0), 'f512_v128': (512, 128, 256),
         'f512_v256': (512, 256, 512), 'f1024_v128': (1024, 128, 256)}
RATES = base.RATES
EXPECTED_COUNTS = {
    'f512_plain': dict(total=7032929, flat_adapter=3058240, v3=0),
    'f512_v128': dict(total=7864639, flat_adapter=3058240, v3=831710),
    'f512_v256': dict(total=9349567, flat_adapter=3058240, v3=2316638),
    'f1024_v128': dict(total=10912575, flat_adapter=6106176, v3=831710),
}
SOURCE_CATALOG_ADDITIONS = tuple('experiments/osram_mosei_v3_grid_20261022/' + name
    for name in ('worker.py', 'run.py', 'report.py')) + (
    'experiments/osram_mosei_gcnet_router_20261020/worker.py',
    'experiments/osram_source_role_v3_20261021/worker.py',
    'experiments/osram_source_relation_attention_20261021/worker.py',
    'experiments/osram_mosi_gcnet_testbest_20261020/worker.py',
    'experiments/osram_text_path_router_joint_mosi_20261019/run.py',
    'experiments/osram_text_path_router_joint_20261019/run.py',
    'experiments/osram_coupled_source_role_20261020/worker.py',
    'experiments/osram_12_source_evidence_audit_20261019/run.py',
    'experiments/osram_tasr_20261018/runner_helpers.py',
    'experiments/osram_mosi_v3_capacity_20261022/run.py',
)


def candidate_config(cfg, mode):
    if mode not in MODES:
        raise ValueError('Only the four new MOSEI seed66 capacity arms are approved')
    baseline = base.candidate_config(cfg, 'small_flat')
    flat, width, pair = MODES[mode]
    return replace(baseline, osram_adapter_hidden_dim=flat, osram_source_role_v3=bool(width),
                   osram_source_role_v3_dim=width or 64, osram_source_role_v3_pair_hidden=pair or 128)


def expected_counts(mode):
    if mode not in MODES:
        raise ValueError('Only the four new MOSEI seed66 capacity arms are approved')
    return dict(EXPECTED_COUNTS[mode])


def main(argv=None):
    return base.main(argv, allowed_modes=MODES, config_factory=candidate_config,
        expected_counts_factory=expected_counts, source_catalog_additions=SOURCE_CATALOG_ADDITIONS,
        capture_first_epoch_only=True)


if __name__ == '__main__':
    main()
