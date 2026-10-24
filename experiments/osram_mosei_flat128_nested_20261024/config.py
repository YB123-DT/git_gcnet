"""Exactly24 approved result slots,23 fresh runs; graph grouping is not CHM heads."""
from dataclasses import replace
import hashlib
from itertools import product
from pathlib import Path

REUSED_CONTROL = Path('/data2/yb/remote_experiments/osram_mosei_capacity_cmerc_20261024/execution_v1/runs/f128')
BASE_SOURCE = Path('/data2/yb/remote_experiments/osram_mosei_capacity_cmerc_20261024/source_v1')
VARIANTS = {f'N{i:02d}': dict(dim=d, depth=l, groups=g, mlp_hidden=128)
            for i, (d, l, g) in enumerate(product((32,64,128), (2,3), (2,4,8)), 1)}
VARIANTS.update({f'N{i}': dict(dim=d, depth=3, groups=8, mlp_hidden=m)
                 for i, (d,m) in enumerate(((32,32), (64,64), (128,256)), 19)})


def method_name(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown approved Nested configuration: '+variant)
    return 'mosei_f128_'+variant.lower()


def run_spec(variant):
    if variant in ('F66', 'F67', 'F68'):
        return dict(method='none', seed=int(variant[1:]))
    return dict(method=method_name(variant), seed=66)


def candidate_config(cfg, variant):
    from experiments.osram_nested_cmerc_tuning_20261023.worker import candidate_config as validate_original
    if cfg.osram_adapter_hidden_dim != 0:
        raise ValueError('Start from the original accepted large Flat configuration')
    validate_original(cfg, 'none', seed=66, fold=1)
    return replace(cfg, osram_adapter_hidden_dim=128, **{
        'seed': run_spec(variant)['seed'], 'osram_meaningful_block': run_spec(variant)['method']})


def verify_registry_extension(root, base_root=BASE_SOURCE):
    relative = 'gcnet_missing_m3/meaningful_new40_registry.py'
    root, base_root = Path(root), Path(base_root)
    methods = {method_name(v) for v in VARIANTS}
    kept, removed = [], []
    for line in (root/relative).read_text().splitlines(keepends=True):
        found = [m for m in methods if "'"+m+"':" in line]
        if found:
            removed.extend(found)
        else:
            kept.append(line)
    if set(removed) != methods or len(removed) != 21 or ''.join(kept) != (base_root/relative).read_text():
        raise ValueError('Registry differs beyond the21 approved configuration entries')
    checked = 0
    for package in ('gcnet_missing_m3', 'gcnet_modality_jepa'):
        old = {p.name for p in (base_root/package).glob('*.py')}
        if old != {p.name for p in (root/package).glob('*.py')}:
            raise ValueError('Numerical source addition/removal')
        for name in old:
            rel = package+'/'+name
            if rel != relative and (root/rel).read_bytes() != (base_root/rel).read_bytes():
                raise ValueError('Numerical source changed: '+rel)
            checked += rel != relative
    return dict(unchanged_numerical_files=checked, registry_entries=sorted(removed),
                registry_sha256=hashlib.sha256((root/relative).read_bytes()).hexdigest(),
                base_registry_sha256=hashlib.sha256((base_root/relative).read_bytes()).hexdigest())
