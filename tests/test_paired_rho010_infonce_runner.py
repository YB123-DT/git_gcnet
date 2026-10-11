import json
from pathlib import Path
from experiments.osram_paired_rho010_infonce_20261011.run import configuration


def test_only_contrast_weight_changes():
    root=Path(__file__).resolve().parents[1]
    control=json.loads((root/'experiments/osram_paired_history_rho025_20261002/results/C02/config.json').read_text())
    changed=configuration(control)
    assert [k for k in control if changed[k]!=control[k]]==['history_contrast_weight']
    assert changed['history_contrast_weight']==.1
    assert control['history_contrast_weight']==0


def test_fixed_patterns_keep_original_checkpoint_mapping():
    from experiments.osram_paired_rho010_infonce_20261011.fixed import PATTERNS,checkpoint_rate
    assert tuple(PATTERNS)==('A','T','V','AT','AV','TV')
    assert [checkpoint_rate(b) for b in PATTERNS.values()]==[.7,.7,.7,.3,.3,.3]
