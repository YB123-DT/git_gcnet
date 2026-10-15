"""Run CPU-only review tests, with optional read-only local feature validation."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-data', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if args.with_data:
        os.environ['BASELINE_TEST_DATA'] = '1'
    else:
        os.environ.pop('BASELINE_TEST_DATA', None)
    try:
        import pytest
    except ModuleNotFoundError:
        # Reuse the installed local pure-Python pytest without replacing the
        # training environment's NumPy or sklearn. No dependency installation.
        local_pytest = Path('/home/yangbin/miniconda3/lib/python3.13/site-packages')
        if not local_pytest.exists():
            raise SystemExit('pytest is required; see requirements.txt')
        sys.path.append(str(local_pytest))
        import pytest
    root = Path(__file__).resolve().parent
    return pytest.main([str(root / 'tests'),
                        str(root / 'adapters/graph_models/test_graph_models.py'), '-q'])


if __name__ == '__main__':
    raise SystemExit(main())
