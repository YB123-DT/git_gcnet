"""One MOSEI Flat256 run; delegate all numerical training to the width screen."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from experiments.osram_mosei_v3_capacity_20261022 import worker as base

MODES = ('flat256',)
NUMERICAL_COMMIT = '7ec78786b12b6006bd8f6ebcfab32ee9f6026a23'


def candidate_config(cfg, mode):
    if mode not in MODES:
        raise ValueError('Only the single Flat256 arm is approved')
    checked = base.candidate_config(cfg, 'small_flat')
    return replace(checked, osram_adapter_hidden_dim=256)


def expected_counts(mode):
    if mode not in MODES:
        raise ValueError('Only the single Flat256 arm is approved')
    return dict(total=5508961, flat_adapter=1534272, v3=0)


def write_report(output):
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import validate_selected
    from experiments.osram_tasr_20261018 import runner_helpers as h
    controls = [('1600', base.REFERENCE),
        ('1024', Path('/data2/yb/remote_experiments/osram_mosei_v3_capacity_20261022/execution_v1/runs/small_flat')),
        ('512', Path('/data2/yb/remote_experiments/osram_mosei_v3_grid_20261022/execution_v1/runs/f512_plain')),
        ('256', Path(output))]
    lines = ['# MOSEI Flat width comparison', '', 'INTERNAL TEST-ORACLE SCREEN', '',
        'Seed66, s0, constant1e-3,60epochs,batch32,Adam/WD1e-5,MSE,new GCNet missing.',
        'Only Flat hidden width changes; output1600/LocalSkip/CHM unchanged. No extra Block.',
        'Controls are reused, not retrained. Per-rate TEST-best W-F1 is not single-checkpoint deployment performance.', '',
        '| Hidden width | ACC | W-F1 | High W-F1 | MSE |', '|---:|---:|---:|---:|---:|']
    rows = []
    for width,path in controls:
        d = json.loads((path/'SUMMARY.json').read_text())
        c = d.get('config') or d['identity']['config']
        if d['status'] != 'complete' or d['epoch'] != 60 or c['seed'] != 66:
            raise ValueError('Complete60epoch seed66 control required: '+str(path))
        if (c['learning_rate'] != .001 or c['missing_mask_protocol'] != 'gcnet-random-mask-v1'
                or c['osram_meaningful_block'] != 'none' or c.get('osram_source_role_v3',False)):
            raise ValueError('Do not mix controls from another protocol or module')
        validate_selected(path, d['best_by_rate'])
        a,b = d['aggregate']['router']['mean'], d['aggregate']['router']['high']
        lines.append(f"| {width} | {a['ACC']:.6f} | {a['WF1']:.6f} | {b['WF1']:.6f} | {a['MSE']:.6f} |")
        rows.append(dict(width=int(width), path=str(path), summary_sha256=h.sha(path/'SUMMARY.json'),
                         aggregate=d['aggregate']['router']))
    h.write_json(Path(output)/'RESULT.json',dict(status='complete',label='INTERNAL TEST-ORACLE SCREEN',runs=rows))
    (Path(output)/'RESULT.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, unused = parser.parse_known_args(argv)
    base.main(argv, allowed_modes=MODES, config_factory=candidate_config,
        expected_counts_factory=expected_counts,
        source_catalog_additions=('experiments/osram_mosei_flat256_20261024/worker.py',),
        capture_first_epoch_only=True)
    if args.phase == 'train':
        write_report(args.output)


if __name__ == '__main__':
    main()
