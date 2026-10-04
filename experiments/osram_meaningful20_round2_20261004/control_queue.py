"""Load the round-two controller against an explicit immutable source tree."""
import importlib.util
from pathlib import Path
import sys

if __name__ == '__main__':
    source = sys.argv.pop(1)
    sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location(
        'experiments.osram_meaningful20_round2_20261004.queue', Path(__file__).with_name('queue.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.coordinate(module.parser().parse_args())
