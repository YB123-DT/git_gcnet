"""Round two uses the unchanged shared batch32 CUDA smoke template."""
from experiments.osram_meaningful20_20261003.cuda_check import (
    artifact_budget, check_finite_state, check_gpu, main, parser, profile)

if __name__ == '__main__': main()
