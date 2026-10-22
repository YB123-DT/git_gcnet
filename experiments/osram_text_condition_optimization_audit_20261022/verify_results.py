"""Validate finished artifacts without model inference or parameter updates."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def verify(root):
    result = {}
    for dataset in ('CMUMOSI', 'CMUMOSEI'):
        folder = root/dataset
        read = lambda name: json.loads((folder/name).read_text())
        state, summary, pairs = read('STATUS.json'), read('SUMMARY.json'), read('GRADIENTS.json')
        assert state['status'] == 'complete' and state['new_training_steps'] == 0
        assert state['weights_restored_exact'] and summary['weights_restored_exact']
        assert state['config']['missing_mask_protocol'] == 'gcnet-random-mask-v1'
        assert len(pairs) == summary['N_pairs'] == state['paired_targets'] == 64
        assert all(r['split'] == 'train' and r['target'] > 0 for r in pairs)
        assert all(sum(r['rate'] == f'{i/10:.1f}' for r in pairs) == 8 for i in range(8))
        assert max(r['prewrite_max_error'] for r in pairs) == 0
        assert max(r['past_prediction_max_error'] for r in pairs) == 0
        assert not list(folder.rglob('*.pt')), 'Diagnostic must not save changed weights'
        strata = summary['strata']
        for groups in [strata['pooled_test_slots'], *strata['per_rate'].values()]:
            assert sum(groups[g]['N'] for g in ('Textvisible', 'Textmissing_seen', 'Textmissing_never')) == groups['overall']['N']
            for row in groups.values():
                assert sum(row[k] for k in ('corrections','harms','both_correct','both_wrong')) == row['polarity_N']
        jacobian_groups = {}
        for group in ('overall', *pairs[0]['geometry']['groups']):
            values = []
            for pair in pairs:
                geometry = pair['prediction_jacobian_geometry']
                if geometry is not None:
                    value = geometry['overall' if group == 'overall' else 'groups']
                    if group != 'overall':
                        value = value[group]
                    if value['cosine'] is not None:
                        values.append(value['cosine'])
            jacobian_groups[group] = dict(N=len(values),mean_cosine=statistics.mean(values),
                negative_count=sum(x < 0 for x in values))
        joint_counts = {}
        for opposed in (False, True):
            selected = [r for r in pairs if r['residual_sign_opposed'] == opposed]
            joint_counts['opposed_residuals' if opposed else 'same_residual_sign'] = dict(
                N=len(selected),negative_total_gradient=sum(r['geometry']['overall']['cosine'] < 0 for r in selected),
                **{f'{update}_cross_harms':sum(
                    r['virtual_steps'][update][delta] > 1e-6*max(1,abs(r[base])) for r in selected)
                    for update,delta,base in (('visible','delta_missing','missing_loss'),
                                             ('missing','delta_visible','visible_loss'))})
        hashes = {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in
            ('STATUS.json','SUMMARY.json','GRADIENTS.json','TEST_STRATA.json','GRADIENT_SUMMARY.csv','RESULT.md')}
        result[dataset] = dict(source_commit=state['source_commit'],pid=state['pid'],seconds=state['seconds'],
            paired_targets=len(pairs),prewrite_max_error=0.,past_prediction_max_error=0.,
            weights_restored_exact=True,no_new_checkpoints=True,all_checks_passed=True,
            test_polarity_N=strata['pooled_test_slots']['overall']['polarity_N'],
            prediction_jacobian_groups=jacobian_groups,residual_sign_cross_counts=joint_counts,
            artifact_sha256=hashes)
    assert len({r['source_commit'] for r in result.values()}) == 1
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing validation receipts')
    result = verify(args.root)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
