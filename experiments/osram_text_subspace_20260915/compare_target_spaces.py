"""One Student forward, two frozen target spaces; no Stage2 optimizer updates."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from gcnet_missing_m3.text_subspace import text_jepa_loss, load_frozen_subspace
from gcnet_missing_m3.pretrained_teacher import file_sha256, state_sha256
from experiments.osram_supervised_teacher_20260914.smoke import build
from experiments.osram_text_subspace_20260915.audit_dynamics import gradient_summary
from experiments.osram_text_subspace_20260915.run import ROOT, TEACHER_ROOT, tr, configuration, stage2_config, loaders


def compare_losses(predictions, targets, projector, emotion, groups):
    """The same graph/tensors are reused, not replayed with merely the same seed."""
    results = {}
    for name, projection in (('full-text', None), ('predictable-subspace', projector)):
        jepa = text_jepa_loss(predictions, targets, projection, temperature=.03)
        comparisons = {}
        for group, parameters in groups.items():
            joint = gradient_summary({'emotion': emotion, 'weighted_jepa': .1*jepa.total},
                                     parameters, common_support=True)
            branches = gradient_summary({'weighted_reg': .05*jepa.regression,
                                         'weighted_nce': .05*jepa.contrastive},
                                        parameters, common_support=True)
            jnorm, enorm = joint['norms']['weighted_jepa'], joint['norms']['emotion']
            rnorm, cnorm = branches['norms']['weighted_reg'], branches['norms']['weighted_nce']
            comparisons[group] = dict(joint=joint, branches=branches,
                jepa_emotion_norm_ratio=jnorm/enorm if enorm else None,
                nce_reg_norm_ratio=cnorm/rnorm if rnorm else None)
        results[name] = dict(losses=dict(emotion=float(emotion.detach()), regression=float(jepa.regression.detach()),
                                         nce=float(jepa.contrastive.detach()), jepa=float(jepa.total.detach())),
                             target_count=jepa.target_count, groups=comparisons)
    for group in groups:
        assert (results['full-text']['groups'][group]['joint']['parameter_names'] ==
                results['predictable-subspace']['groups'][group]['joint']['parameter_names'])
    return results


def export_history(history, path):
    if [row['epoch'] for row in history] != list(range(1, 101)):
        raise ValueError('Expected the complete 100-epoch Stage1 history')
    rows = []
    for entry in history:
        for split, metrics in entry['metrics'].items():
            if split not in {'train', 'validation'}:
                raise ValueError('Stage1 history must not include Test')
            for pattern, p in metrics['patterns'].items():
                rows.append(dict(epoch=entry['epoch'], split=split, pattern=pattern,
                    target_sentiment_wf1=metrics['sentiment']['weighted_f1'],
                    **{f'loss_{k}': v for k,v in metrics['losses'].items()},
                    target_effective_rank=metrics['target']['effective_rank'],
                    target_std_mean=metrics['target']['mean_std'],
                    pattern_smooth_l1=p['smooth_l1'], centered_cosine=p['centered_cosine'],
                    real_vs_shuffle_cosine_gap=p['real_vs_shuffle_cosine_gap'],
                    centered_retrieval=p['centered_retrieval'], chance=p['retrieval_chance'],
                    prediction_effective_rank=p['prediction']['effective_rank'],
                    std_ratio=p['prediction_teacher_std_ratio'],
                    prediction_sentiment_wf1=p['sentiment']['weighted_f1']))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run(stage1_root, output, device):
    stage1 = json.loads((stage1_root/'STAGE1.json').read_text())
    history = json.loads((stage1_root/'history.json').read_text())
    checkpoint_path = stage1_root/'subspace.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    selected = min(history, key=lambda x:x['metrics']['validation']['losses']['total'])['epoch']
    assert selected == stage1['selected_epoch'] == checkpoint['epoch']
    assert not stage1['smoke_only'] and checkpoint['selection_split'] == 'validation'
    assert stage1['optimizer_steps'] == 200 and stage1['test_batches_consumed'] == 0
    output.mkdir(parents=True, exist_ok=False)
    base, _, _ = configuration(66)
    cfg = replace(stage2_config(base, TEACHER_ROOT/'teacher'/'seed_66'/'teacher_projectors.pt',
                                None, 'full-text'), device=device)
    tr.set_random_seed(66)
    train, _, _unused_test, *dims = loaders(cfg)
    # Reset model RNG after obtaining dimensions, matching the old smoke initialization.
    tr.set_random_seed(66)
    model = build(cfg, dims)
    train, _, _unused_test, *_ = loaders(cfg)
    view = tr._prepare_view(tr._move_batch(next(iter(train[0])), torch.device(device)),
                            tr._build_schedule(cfg, 'train', .5), 0, tuple(dims))
    r, provenance = load_frozen_subspace(checkpoint_path, model.teacher_integrity(), cfg.latent_dim)
    r.to(device)
    before = state_sha256(model.state_dict())
    r_before = state_sha256(r.state_dict())
    model.train()
    scores, _, _, predictions = model([view['incomplete']], view['availability'], view['qmask'],
                                      view['umask'], view['lengths'], predict_missing=True)
    targets = model.encode_teacher_targets([view['complete']])
    emotion = tr._task_loss(cfg.dataset, scores, view['labels'], view['umask'], cfg.mosi_task_mode,
                            cfg.task_regression_loss, cfg.task_smooth_l1_beta)
    groups = {'encoder': [(f'encoder.{n}',p) for n,p in model.observed_set.named_parameters()],
              'osram': [(f'osram.{n}',p) for n,p in model.osram.named_parameters()]}
    groups['combined'] = groups['encoder'] + groups['osram']
    result = compare_losses(predictions, targets, r, emotion, groups)
    assert before == state_sha256(model.state_dict()) and r_before == state_sha256(r.state_dict())
    assert all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in r.parameters())
    full = result['full-text']['groups']['combined']['joint']['norms']['weighted_jepa']
    sub = result['predictable-subspace']['groups']['combined']['joint']['norms']['weighted_jepa']
    payload = dict(seed=66, missing_rate=.5, model_forward_calls=1, dropout_realizations=1,
        stage2_optimizer_steps=0, test_batches_consumed=0, model_and_R_unchanged=True,
        stage1=stage1, stage1_checkpoint=provenance, comparison=result,
        subspace_full_weighted_jepa_norm_ratio=sub/full if full else None,
        stage1_history_sha256=file_sha256(stage1_root/'history.json'),
        selection_rule='Stage1 minimum validation composite loss; no Stage2 checkpoint selection yet',
        scientific_scope='One fresh Student batch, matched forward; not a training-time norm guarantee')
    (output/'TARGET_SPACE_CONTROL.json').write_text(json.dumps(payload, indent=2))
    export_history(history, output/'stage1_per_epoch.csv')
    print(json.dumps(dict(stage1_selected_epoch=selected, stage1_elapsed_seconds=stage1['elapsed_seconds'],
        subspace_full_gradient_ratio=payload['subspace_full_weighted_jepa_norm_ratio'],
        spaces={name: dict(losses=v['losses'], combined=v['groups']['combined']) for name,v in result.items()}), indent=2))
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1-root', type=Path, default=ROOT/'stage1_seed66_full100')
    parser.add_argument('--output', type=Path, default=ROOT/'target_space_control_seed66')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.stage1_root, args.output, args.device)
