"""Run updated scheduling logic without changing immutable model snapshots."""
import importlib.util
from pathlib import Path
import sys

if __name__ == '__main__':
    source = sys.argv.pop(1)
    sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location(
        'experiments.osram_meaningful20_20261003.queue', Path(__file__).with_name('queue.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = module.parser().parse_args()
    if not 1 <= args.max_concurrent <= 6 or args.poll_seconds < 1:
        raise ValueError('Invalid concurrency or polling interval')
    module.coordinate(args)
