"""Overlay scheduling files only; the immutable training snapshot is untouched."""
import importlib
import importlib.util
from pathlib import Path
import sys

def load_control(name, path):
    if not path.is_file(): raise FileNotFoundError(f'Missing external control file: {path}')
    parent, attribute = name.rsplit('.', 1)
    package = importlib.import_module(parent)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    setattr(package, attribute, module)
    return module


if __name__ == '__main__':
    source = sys.argv.pop(1)
    sys.path.insert(0, source)
    directory = Path(__file__).resolve().parent
    load_control('experiments.osram_meaningful20_20261003.queue', directory / 'shared_queue.py')
    dispatch = directory / 'dispatch.py'
    if dispatch.is_file():
        load_control('experiments.osram_meaningful20_round2_20261004.dispatch', dispatch)
    elif '--dispatch-policy' in sys.argv or any(arg.startswith('--dispatch-policy=') for arg in sys.argv):
        raise FileNotFoundError(f'Dispatch policy requires external control file: {dispatch}')
    module = load_control('experiments.osram_meaningful20_round2_20261004.queue', directory / 'queue.py')
    module.coordinate(module.parser().parse_args())
