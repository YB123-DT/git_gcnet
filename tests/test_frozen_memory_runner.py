"""Cache semantics before any diagnostic fitting."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / 'experiments/osram_frozen_memory_audit_20261009/run.py'


def load_runner():
    assert MODULE.exists(), 'frozen-memory cache runner is not implemented'
    spec = importlib.util.spec_from_file_location('frozen_memory_runner', MODULE)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def rows():
    return dict(local=np.ones((3,256)), memory=np.zeros((3,4,512)),
                availability=np.array([[1,1,1],[1,0,1],[0,1,1]]),
                labels=np.array([1.,0.,-1.]), prediction=np.array([.5,.1,-.8]),
                conversation_ids=np.array(['a','a','b']),utterance_indices=np.array([0,1,0]),
                speaker=np.zeros(3,dtype=int))


def test_cache_schema_and_active_mask():
    module = load_runner()
    data = rows()
    data['memory'][1,0] = 1
    data['memory'][1,2] = 1
    assert module.validate_cache(data, {'a','b'}) == 3
    data['memory'][1,1,0] = 1
    with pytest.raises(ValueError, match='inactive'):
        module.validate_cache(data, {'a','b'})


def test_cache_rejects_first_history_and_duplicate_ids():
    module = load_runner()
    data = rows()
    data['memory'][0,0,0] = 1
    with pytest.raises(ValueError, match='first'):
        module.validate_cache(data, {'a','b'})
    data = rows()
    data['utterance_indices'][1] = 0
    with pytest.raises(ValueError, match='duplicate'):
        module.validate_cache(data, {'a','b'})


def test_cache_rejects_split_coverage_and_nonfinite():
    module = load_runner()
    data = rows()
    with pytest.raises(ValueError, match='coverage'):
        module.validate_cache(data, {'a','c'})
    data['local'][0,0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        module.validate_cache(data, {'a','b'})
