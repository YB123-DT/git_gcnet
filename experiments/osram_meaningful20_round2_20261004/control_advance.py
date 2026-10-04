"""Update only dispatch, keeping the trained/profiled model snapshot immutable."""
import importlib.util
from pathlib import Path
import sys


if __name__=='__main__':
    source=sys.argv.pop(1)
    sys.path.insert(0,source)
    package='experiments.osram_meaningful20_round2_20261004'
    for name in ('dispatch','advance'):
        spec=importlib.util.spec_from_file_location(package+'.'+name,Path(__file__).with_name(name+'.py'))
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
    module.advance(module.parser().parse_args())
