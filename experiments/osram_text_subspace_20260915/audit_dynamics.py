"""No-update audit: inherited anisotropy and objective gradients, never training."""
import argparse
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from gcnet_missing_m3.text_subspace import TextSubspacePretrainer, representation_stats, text_jepa_loss
from gcnet_missing_m3.pretrained_teacher import state_sha256, file_sha256
from experiments.osram_supervised_teacher_20260914.smoke import build
from experiments.osram_text_subspace_20260915.run import (
    ROOT, TEACHER_ROOT, tr, configuration, stage1_data, stage2_config, loaders,
)


def gradient_summary(losses, named_parameters, common_support=False):
    """Fixed parameter coordinates, no .backward()/grad accumulation or clipping."""
    named_parameters = [(n, p) for n, p in named_parameters if p.requires_grad]
    params = tuple(p for _, p in named_parameters)
    gradients = {name: torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
                 for name, value in losses.items()}
    selected = [i for i in range(len(params)) if not common_support
                or all(gs[i] is not None for gs in gradients.values())]
    vectors = {}
    for name, gs in gradients.items():
        chunks = [(gs[i].detach() if gs[i] is not None else torch.zeros_like(params[i])).flatten().double().cpu()
                  for i in selected]
        vectors[name] = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.double)
    norms = {k: float(v.norm()) for k, v in vectors.items()}
    cosine = {}
    for a, b in itertools.combinations(vectors, 2):
        denominator = norms[a] * norms[b]
        cosine[f'{a}__{b}'] = float(vectors[a].dot(vectors[b]) / denominator) if denominator > 0 else None
    return dict(norms=norms, cosines=cosine,
                norm_of_sum=float(sum(vectors.values()).norm()),
                common_support=common_support, parameter_names=[named_parameters[i][0] for i in selected],
                parameter_count=sum(params[i].numel() for i in selected))


@torch.no_grad()
def rank_baselines(text, initial_r, trained_r):
    return {name: representation_stats(z) for name, z in (
        ('teacher_text', text), ('initial_R0', initial_r(text)), ('trained_R', trained_r(text)))}


def stage1_gradients(model, batch):
    before = state_sha256(model.state_dict())
    losses = model.loss(batch, batch['labels'], beta=1.)
    terms = {k: losses[k] for k in ('sentiment', 'predictability', 'variance', 'covariance')}
    r = gradient_summary(terms, list(model.projector.named_parameters()))
    full = gradient_summary({'total': losses['total']}, list(model.named_parameters()))
    q = gradient_summary({'predictability': losses['predictability']}, list(model.predictor.named_parameters()))
    assert all(p.grad is None for p in model.parameters())
    assert before == state_sha256(model.state_dict())
    return dict(losses={k: float(v.detach()) for k, v in losses.items()}, R=r, Q=q,
                total_R_C_Q_gradient_norm=full['norms']['total'],
                hypothetical_global_clip_coefficient=min(1., 1. / (full['norms']['total'] + 1e-6)),
                clipping_applied=False, parameter_hash=before)


def audit(output, smoke_root, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    archive = json.loads((smoke_root/'SMOKE.json').read_text())
    checkpoint_path = smoke_root/'stage1'/'subspace.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    tr.set_random_seed(66)
    data, _, dims, teacher_hash, provenance = stage1_data(66, device)
    initial = TextSubspacePretrainer().to(device)
    # Original fit_stage1 initializes Adam (no random draw), then randperm of batches.
    order = torch.randperm(len(data['train'])).tolist()
    batch = {k: v.to(device) for k, v in data['train'][order[0]].items()}
    before = stage1_gradients(initial, batch)
    expected = archive['stage1']['first_step_gradient_l2']
    actual = dict(R_from_sentiment=before['R']['norms']['sentiment'],
                  R_from_predictability=before['R']['norms']['predictability'],
                  Q_from_predictability=before['Q']['norms']['predictability'])
    for name in actual:
        torch.testing.assert_close(torch.tensor(actual[name]), torch.tensor(expected[name]), rtol=2e-6, atol=1e-6)
    if teacher_hash != checkpoint['teacher_projector_sha256']:
        raise ValueError('Teacher checkpoint provenance mismatch')
    trained = TextSubspacePretrainer().to(device)
    trained.load_state_dict(checkpoint['stage1_model'], strict=True)
    assert state_sha256(trained.projector.state_dict()) == checkpoint['projector_sha256']
    after = stage1_gradients(trained, batch)
    ranks = {split: rank_baselines(torch.cat([b['text'] for b in rows]).to(device),
                                   initial.projector, trained.projector) for split, rows in data.items()}
    for split, stats in ranks.items():
        torch.testing.assert_close(torch.tensor(stats['trained_R']['effective_rank']),
            torch.tensor(archive['stage1']['selected_metrics'][split]['target']['effective_rank']))
    del initial, trained, batch, data

    base, _, _ = configuration(66)
    cfg = stage2_config(base, TEACHER_ROOT/'teacher'/'seed_66'/'teacher_projectors.pt',
                        checkpoint_path, 'predictable-subspace')
    from dataclasses import replace
    cfg = replace(cfg, device=device)
    tr.set_random_seed(66)
    model = build(cfg, dims)
    train, _, _unused_test, *loaded_dims = loaders(cfg)
    assert loaded_dims == dims
    view = tr._prepare_view(tr._move_batch(next(iter(train[0])), torch.device(device)),
                            tr._build_schedule(cfg, 'train', .5), 0, tuple(dims))
    original_hash = state_sha256(model.state_dict())
    model.train()
    scores, _, _, prediction = model([view['incomplete']], view['availability'],
        view['qmask'], view['umask'], view['lengths'], predict_missing=True)
    jepa = text_jepa_loss(prediction, model.encode_teacher_targets([view['complete']]),
                          model.text_subspace, cfg.temperature)
    emotion = tr._task_loss(cfg.dataset, scores, view['labels'], view['umask'], cfg.mosi_task_mode,
                            cfg.task_regression_loss, cfg.task_smooth_l1_beta)
    reproduced = dict(emotion_loss=float(emotion.detach()), regression=float(jepa.regression.detach()),
                       contrastive=float(jepa.contrastive.detach()))
    for name, value in reproduced.items():
        torch.testing.assert_close(torch.tensor(value), torch.tensor(archive['stage2'][name]), rtol=2e-6, atol=1e-6)
    terms = {'emotion': emotion, 'weighted_jepa': .1*jepa.total}
    groups = {'encoder': [(f'encoder.{n}', p) for n, p in model.observed_set.named_parameters()],
              'osram': [(f'osram.{n}', p) for n, p in model.osram.named_parameters()]}
    groups['combined'] = groups['encoder'] + groups['osram']
    comparisons = {name: {
        'common_used_parameters': gradient_summary(terms, parameters, common_support=True),
        'whole_module_parameters': gradient_summary(terms, parameters)} for name, parameters in groups.items()}
    branch_terms = {'weighted_reg': .05*jepa.regression, 'weighted_nce': .05*jepa.contrastive}
    branch_gradients = gradient_summary(branch_terms, groups['combined'], common_support=True)
    assert original_hash == state_sha256(model.state_dict())
    assert all(p.grad is None for p in model.parameters())
    result = dict(seed=66, optimizer_steps=0, test_batches_consumed=0, model_state_unchanged=True,
        source_commit='ea62697', checkpoint_sha256=file_sha256(checkpoint_path),
        teacher_projector_sha256=teacher_hash, initial_R0_reconstruction='Original seed/data/projector init order; not a saved snapshot',
        initial_first_batch_index=order[0], initial_gradient_reference=expected,
        initial_gradient_reproduced=actual, rank_baselines=ranks,
        stage1={'initial': before, 'after_two_updates_same_batch': after},
        stage2={'state': 'fresh Student before original smoke update; same batch/dropout loss realization',
                'reproduced_losses': reproduced, 'comparisons': comparisons,
                'weighted_reg_vs_nce_shared': branch_gradients},
        scope='Single-seed no-update diagnostic, no loss/temperature/architecture changes',
        stage1_condition='Current frozen Teacher A/V only, not Stage2 Student+causal Base/Gap information',
        teacher_provenance=provenance)
    (output/'DYNAMICS.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(ranks={s: {n: v['effective_rank'] for n, v in rs.items()} for s, rs in ranks.items()},
        stage1={s: {'norms': v['R']['norms'], 'cosines': v['R']['cosines'],
                     'global_clip': v['hypothetical_global_clip_coefficient']} for s,v in result['stage1'].items()},
        stage2={s:v['common_used_parameters'] for s,v in comparisons.items()}), indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dynamics_seed66')
    parser.add_argument('--smoke-root', type=Path, default=ROOT/'smoke_seed66_verified')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    audit(args.output, args.smoke_root, args.device)
