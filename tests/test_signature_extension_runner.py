import pytest

from experiments.osram_signature_stage2_extension_20261025.run import budget, learning_rate
from experiments.osram_signature_stage2_extension_20261025.scheduler import jobs_for, OLD_ROOT


def test_stage2_budgets_and_learning_rates():
    assert budget('CMUMOSI') == 150 and budget('CMUMOSEI') == 60
    assert learning_rate('CMUMOSI',99) == 3e-4
    assert learning_rate('CMUMOSI',100) == 3e-4/10
    assert learning_rate('CMUMOSEI',0) == learning_rate('CMUMOSEI',59) == 3e-4
    with pytest.raises(ValueError):
        learning_rate('CMUMOSEI',60)
    with pytest.raises(ValueError):
        budget('MELD')


def test_finite_queue_reuses_mosi66_and_only_extends_approved_scope():
    jobs = jobs_for('train')
    actual = {(j['dataset'],j['seed'],j['rate']) for j in jobs}
    expected = {(d,s,str(r/10)) for d,seeds in [('CMUMOSI',(67,68)),('CMUMOSEI',(66,))]
                for s in seeds for r in range(8)}
    assert actual == expected and len(jobs) == 24
    assert len(jobs_for('smoke')) == 2
    assert str(OLD_ROOT) == '/data2/yb/remote_experiments/osram_neumiss_fixed_rate_stage2_20261025/execution_v1'
