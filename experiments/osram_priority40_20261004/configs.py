"""Deliver forty reference-derived configurations, never launch training.

python -m experiments.osram_priority40_20261004.configs \
    --reference /explicit/cfg84/config.json --output /new/delivery/directory
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.osram_meaningful20_20261003.run import candidate_config
from gcnet_missing_m3.priority40_registry import (
    PRIORITY_INPUT_METHODS, PRIORITY_METHODS, PRIORITY_NORM_METHODS,
)


# Mechanism matches, not assertions that the complete networks are identical.
HISTORICAL_MECHANISMS = {
    'm02_dat': ('dual_attention_symbolic_relations',),
    'm04_edgeconv': ('dgcnn_dynamic_edgeconv',),
    'm05_pna': ('pna_evidence',),
    'm08_mfb': ('mfb',),
    'm10_block': ('block',),
    'm11_film': ('film',),
    'm14_gct': ('gct',),
    'm17_janossy': ('janossy_full_role_symmetrization',),
    'm18_fspool': ('fspool_fsunpool_evidence',),
    'm19_netvlad': ('netvlad_residual_encoding',),
    'm22_dcnv2': ('dcnv2',),
    'm23_cin': ('cin',),
    'm25_kan': ('repr_kan_function_composition',),
    'm38_soft_moe': ('routing_soft_moe_dispatch_expert_combine',),
    'm39_node': ('node',),
    'm40_capsule': ('capsule_dynamic_routing',),
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + '\n').encode('utf-8')


def code_provenance(repo=None):
    """Report actual HEAD and dirty state, without claiming an immutable snapshot."""
    repo = Path(repo or Path(__file__).resolve().parents[2])

    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(repo), *arguments])

    try:
        commit = git('rev-parse', 'HEAD').decode().strip()
        status = git('status', '--porcelain=v1', '--untracked-files=all').decode()
        diff = git('diff', '--binary', 'HEAD', '--')
        untracked = git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
        return {
            'code_commit': commit,
            'worktree_dirty': bool(status),
            'worktree_status': status.splitlines(),
            'tracked_diff_sha256': _sha(diff),
            'untracked_paths': [name for name in untracked if name],
            'tool_sha256': _sha(Path(__file__).read_bytes()),
            'immutable_snapshot': False,
            'note': 'HEAD alone does not identify dirty/untracked source contents; no snapshot created.',
        }
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError('Cannot establish repository code provenance') from error


def placement(method):
    if method in PRIORITY_INPUT_METHODS:
        return 'I'
    if method in PRIORITY_NORM_METHODS:
        return 'N'
    return 'R'


def parameter_summary(method, config):
    """Count CPU constructors only, no model/data import or forward/training."""
    import torch
    from torch import nn
    from gcnet_missing_m3.priority40_common import DynamicTanh, PriorityFeatureCore
    from gcnet_missing_m3.meaningful_input_priority40 import build_priority40

    def count(module):
        return sum(parameter.numel() for parameter in module.parameters())

    # Existing validation fixes these values. Read them rather than substitute defaults.
    latent = config['latent_dim']
    heads, value = config['osram_num_heads'], config['osram_value_dim']
    where = placement(method)
    with torch.random.fork_rng(devices=[]):
        if where == 'R':
            core = PriorityFeatureCore(method, latent, heads, value)
            operator = count(core.operator)
            bridge = nn.Linear(core.output_dim, config['osram_output_dim'])
            shared = count(core) - operator + count(bridge)
            return {
                'new_parameters_excluding_shared': operator,
                'shared_parameters': shared,
                'net_added_parameters': operator + shared,
                'count_basis': 'actual operator; excludes common typed projections, role embedding and zero bridge',
            }
        if where == 'I':
            number = count(build_priority40(method, latent, heads, value))
            return {
                'new_parameters_excluding_shared': number,
                'shared_parameters': 0,
                'net_added_parameters': number,
                'count_basis': 'actual raw-slot input adapter; no R token projection or bridge',
            }
        # cfg84 Flat concatenates Local with four doubled forward/backward contexts.
        width = latent + 4 * (2 * heads * value)
        replacement = count(DynamicTanh(width))
        removed = count(nn.LayerNorm(width))
        return {
            'new_parameters_excluding_shared': replacement - removed,
            'shared_parameters': 0,
            'net_added_parameters': replacement - removed,
            'replacement_parameters': replacement,
            'removed_layernorm_parameters': removed,
            'normalization_width': width,
            'count_basis': 'net DyT-minus-first-LayerNorm; not a residual branch',
        }


def generate(reference, output):
    reference, output = Path(reference), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite output: {output}')
    raw = reference.read_bytes()
    baseline = json.loads(raw)
    if not isinstance(baseline, dict):
        raise ValueError('Reference must be a JSON object')
    if 'seed' not in baseline:
        raise ValueError('Reference must specify its seed; no defaults are invented')
    if len(PRIORITY_METHODS) != 40 or len(set(PRIORITY_METHODS)) != 40:
        raise RuntimeError('Priority registry must contain exactly forty distinct IDs')

    configs, records = {}, []
    for method in PRIORITY_METHODS:
        config = candidate_config(baseline, method, seed=baseline['seed'])
        # Guard any future expansion of candidate_config against accidental protocol edits.
        expected = dict(baseline, osram_meaningful_block=method)
        if config != expected:
            raise RuntimeError('candidate_config changed fields other than osram_meaningful_block')
        configs[method] = _json_bytes(config)
        historical = list(HISTORICAL_MECHANISMS.get(method, ()))
        records.append({
            'id': method,
            'placement': placement(method),
            'config_file': method + '.json',
            'config_sha256': _sha(configs[method]),
            'parameters': parameter_summary(method, config),
            'historical_mechanism': historical,
            'historical_status': 'known_mechanism_comparison' if historical else 'not_exhaustively_audited',
            'historical_training_status': 'not_inferred_from_mechanism_match',
            'count_as_new_method': False if historical else None,
        })

    summary = {
        'schema_version': 1,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'reference': {'path': str(reference.resolve()), 'sha256': _sha(raw),
                      'seed': baseline['seed']},
        'code': code_provenance(),
        'configuration_count': len(configs),
        'only_changed_field': 'osram_meaningful_block',
        'training_started': False,
        'historical_comparison_count': sum(bool(row['historical_mechanism']) for row in records),
        'new_method_count': None,
        'novelty_note': 'Historical comparisons do not count as new; unlisted methods are not certified novel.',
        'methods': records,
    }
    # Validate/construct all forty before creating output. mkdir and exclusive opens
    # reject both pre-existing destinations and concurrent attempts to overwrite.
    output.mkdir(parents=True, exist_ok=False)
    for method, payload in configs.items():
        with (output / (method + '.json')).open('xb') as stream:
            stream.write(payload)
    with (output / 'SUMMARY.json').open('xb') as stream:
        stream.write(_json_bytes(summary))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    summary = generate(args.reference, args.output)
    print(json.dumps({'output': str(args.output), 'configuration_count': summary['configuration_count'],
                      'training_started': False}, sort_keys=True))


if __name__ == '__main__':
    main()
