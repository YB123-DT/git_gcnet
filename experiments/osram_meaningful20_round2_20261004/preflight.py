"""Use the same minimal readiness and live resource gates as round one."""
from experiments.osram_meaningful20_20261003.preflight import (
    BANNED_UUID, HEALTHY_INDICES, admission, main, query_gpus,
    validate_gpu, validate_readiness)

if __name__ == '__main__': main()
