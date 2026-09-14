"""Matched initialization/two-update diagnostic, not a training experiment."""
import inspect
import json
from dataclasses import asdict
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.model import MissingM3GraphModel
from experiments.osram_causal_nojepa_20260910.run import configuration, runner


def rng():
    return torch.get_rng_state(), torch.cuda.get_rng_state_all()


def restore(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state_all(state[1])


def norm(grads):
    return sum(float(g.detach().double().square().sum())
               for g in grads if g is not None) ** .5


def main():
    torch.set_num_threads(4)
    cfg, _, _ = configuration(66)
    tr.set_random_seed(cfg.seed)
    roots = [str(runner.FEATURES / n) for n in
             ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    train, _, _, *dims = tr.get_loaders(
        audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=1, dataset=cfg.dataset, batch_size=cfg.batch_size,
        num_workers=0, seed=cfg.seed, evaluation_protocol=cfg.evaluation_protocol,
        validation_fraction=cfg.validation_fraction)
    view = tr._prepare_view(
        tr._move_batch(next(iter(train[0])), torch.device(cfg.device)),
        tr._build_schedule(cfg, 'train', .5), 0, tuple(dims))
    names = inspect.signature(MissingM3GraphModel.__init__).parameters
    kwargs = {k: v for k, v in asdict(cfg).items() if k in names}
    kwargs.update(adim=dims[0], tdim=dims[1], vdim=dims[2], D_e=cfg.hidden,
                  graph_hidden_size=cfg.hidden // 2, n_speakers=1, n_classes=1,
                  time_attn=cfg.time_attention)
    initialization_rng = rng()
    models = {}
    for name in ('flat_joint', 'wsc'):
        restore(initialization_rng)
        models[name] = MissingM3GraphModel(
            **kwargs, write_state_completion=name == 'wsc').to(cfg.device)
    shared = set(models['flat_joint'].state_dict()) & set(models['wsc'].state_dict())
    mismatches = [k for k in shared if not torch.equal(
        models['flat_joint'].state_dict()[k], models['wsc'].state_dict()[k])]
    assert not mismatches, mismatches
    optimizers = {name: torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        for name, model in models.items()}
    report = dict(seed=66, rate=.5, config=asdict(cfg),
                  shared_tensors_exact=len(shared), shared_mismatches=mismatches,
                  conversations=len(view['lengths']), utterances=int(view['umask'].sum()),
                  note='Flat uses original Joint JEPA; WSC uses write-token loss. No architecture changes.',
                  steps=[])
    call = ([view['incomplete']], view['availability'], view['qmask'],
            view['umask'], view['lengths'])
    forward_rng = rng()
    for step in (1, 2):
        for name, model in models.items():
            restore(forward_rng)
            model.train()
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            memory_norms = []
            original_write = model.osram.block_write

            def observe_write(*args, **kw):
                result = original_write(*args, **kw)
                memory_norms.append(result.detach().float().norm(dim=(-2, -1)).cpu())
                return result

            with patch.object(model.osram, 'block_write', side_effect=observe_write):
                logits, _, _, predictions = model(*call, predict_missing=name == 'flat_joint')
            cls = tr._task_loss(cfg.dataset, logits, view['labels'], view['umask'],
                                cfg.mosi_task_mode, cfg.task_regression_loss,
                                cfg.task_smooth_l1_beta)
            if name == 'wsc':
                aux, count = model.write_state_loss(
                    view['complete'], view['availability'], view['qmask'], view['umask'])
                predictor = model.write_state.predictor
                weight = cfg.jepa_weight
            else:
                teacher = model.encode_teacher_targets([view['complete']])
                jepa = tr.missing_m3_loss(
                    predictions, teacher, temperature=cfg.temperature,
                    regression_aggregation=cfg.jepa_regression_aggregation,
                    contrastive_prediction_source=cfg.jepa_contrastive_source)
                aux, count = jepa.total, jepa.target_count
                predictor = model.missing_predictor
                weight = cfg.jepa_weight * tr._jepa_rate_weight(.5, cfg.jepa_rate_weighting)
            predictor_params = [p for p in predictor.parameters() if p.requires_grad]
            task_grads = torch.autograd.grad(cls, predictor_params, retain_graph=True, allow_unused=True)
            loss = cls + weight * aux
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            groups = dict(encoder=model.observed_set, osram=model.osram,
                          classifier=model.smax_fc, predictor=predictor)
            norms = {k: norm(p.grad for p in module.parameters()) for k, module in groups.items()}
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            memories = torch.cat([v.flatten() for v in memory_norms])
            row = dict(model=name, step=step, emotion_loss=float(cls), auxiliary_loss=float(aux),
                       weighted_auxiliary_loss=float(weight * aux), total_loss=float(loss),
                       targets=int(count), gradient_l2_preclip=norms,
                       total_gradient_l2_preclip=float(total_norm),
                       total_gradient_l2_postclip=norm(p.grad for p in model.parameters()),
                       emotion_predictor_gradient_l2=norm(task_grads),
                       actual_postwrite_memory_frobenius_mean=float(memories.mean()),
                       actual_postwrite_memory_frobenius_max=float(memories.max()))
            report['steps'].append(row)
            optimizer.step()
            model.update_teacher(cfg.ema_tau)
            print(json.dumps(row), flush=True)
            if name == 'wsc':
                next_rng = rng()
        forward_rng = next_rng
    output = Path('/data2/yb/remote_experiments/osram_write_state_20260914/MATCHED_SMOKE.json')
    output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
