"""Frozen Flat readout patching: one scan per batch, no training or memory intervention."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
MODES = ('normal', 'no_gap', 'no_base', 'base_to_gap', 'gap_to_base')


def patched_inputs(local, base, gap, availability, valid):
    """Copy the donor, retain its original slot; intervene only on one-missing rows."""
    missing = ~availability.bool()
    eligible = valid & (missing.sum(-1) == 1)
    active_gap = gap * missing[..., None]
    out = {}
    for mode in MODES:
        b, g = base.clone(), active_gap.clone()
        if mode == 'no_gap':
            g[eligible] = 0
        elif mode == 'no_base':
            b[eligible] = 0
        elif mode == 'base_to_gap':
            g[eligible] = (base.unsqueeze(-2).expand_as(g) * missing[..., None])[eligible]
        elif mode == 'gap_to_base':
            b[eligible] = active_gap.sum(-2)[eligible]
        out[mode] = torch.cat([local, b, g.flatten(-2)], -1)
    return out


@torch.no_grad()
def run(seed, output):
    from gcnet_missing_m3 import train_gcnet as tr
    from experiments.osram_local_gated_20260910.smoke import make
    from experiments.osram_causal_readout_20260910.run import FULL, FEATURES
    torch.set_num_threads(4)
    path = FULL / f'seed_{seed}' / 'best.pt'
    cp = torch.load(path, map_location='cpu', weights_only=False)
    cfg = tr.TrainConfig(**dict(cp['config'], device='cuda'))
    assert cfg.osram_write_step == .6 and not cfg.osram_bidirectional
    assert cfg.osram_readout_fusion == 'flat' and cfg.osram_emotion_ablation == 'full'
    assert not cfg.classification_completion and not cfg.local_context_residual
    assert cfg.readout_type == 'shared' and cfg.mosi_task_mode == 'regression'
    assert cfg.completion_path == 'none'
    tr.set_random_seed(seed)
    roots = [str(FEATURES/n) for n in ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    _, _, test, *dims = tr.get_loaders(audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=1, dataset=cfg.dataset, batch_size=cfg.batch_size, num_workers=0, seed=seed,
        evaluation_protocol=cfg.evaluation_protocol, validation_fraction=cfg.validation_fraction)
    model = make(cfg, dims).eval()
    model.load_state_dict(cp['model'], strict=True)
    reference_metrics = json.loads((path.parent/'metrics.json').read_text())
    report = dict(seed=seed, checkpoint=str(path), checkpoint_epoch=cp.get('epoch'),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint_selection=cp.get('selection_protocol', cfg.checkpoint_selection),
        protocol='fixed existing checkpoint; no intervention-specific selection',
        scope='exactly one missing modality; AT / AV / TV; nonzero labels for W-F1',
        label='INTERNAL DIAGNOSTIC ONLY', rows=[], mask_matches={}, max_normal_logit_error=0.)
    for ri in range(8):
        rate = ri/10
        schedule = tr._build_schedule(cfg, 'test', rate)
        labels, masks = [], []
        predictions = {m: [] for m in MODES}
        for raw in test[cfg.fold-1]:
            view = tr._prepare_view(tr._move_batch(raw, torch.device('cuda')), schedule, 0, tuple(dims))
            original = model([view['incomplete']], view['availability'], view['qmask'],
                             view['umask'], view['lengths'], predict_missing=False)[0]
            ctx = model.last_osram_context
            valid = view['umask'].T.bool()
            inputs = patched_inputs(ctx['local'], ctx['base'], ctx['gap'], view['availability'], valid)
            # All following computations are readout-only; the scan is never called again.
            for mode, value in inputs.items():
                hidden = model.osram.emotion_norm(model.osram.local_skip(ctx['local']) +
                                                  model.osram.emotion_adapter(value))
                hidden = hidden * valid[...,None]
                logits = model.smax_fc(hidden)
                if mode == 'normal':
                    assert torch.equal(logits, original), 'normal replay differs from original forward'
                predictions[mode].append(logits.squeeze(-1)[valid].cpu().numpy())
            labels.append(view['labels'].T[valid].cpu().numpy())
            masks.append(view['availability'][valid].cpu().numpy())
        y, a = np.concatenate(labels), np.concatenate(masks)
        mask_hash = tr._sha256_tensor(torch.from_numpy(a))
        assert mask_hash == reference_metrics['mask_sha256'][str(rate)]
        report['mask_matches'][str(rate)] = True
        pred = {m: np.concatenate(p) for m,p in predictions.items()}
        eligible = a.sum(-1) == 2
        groups = {'one_missing': eligible, 'AT': eligible & (a[:,2] == 0),
                  'AV': eligible & (a[:,1] == 0), 'TV': eligible & (a[:,0] == 0)}
        for group, selected in groups.items():
            n = int((selected & (y != 0)).sum())
            for mode in MODES:
                metrics = tr._metrics(cfg.dataset, y[selected], pred[mode][selected]) if n else None
                report['rows'].append(dict(seed=seed, rate=rate, group=group, mode=mode,
                    count=int(selected.sum()), nonzero_count=n, metrics=metrics))
        np.savez_compressed(output/f'seed{seed}_rate{rate:.1f}.npz', labels=y, availability=a, **pred)
        print(f'seed={seed} rate={rate:.1f} one_missing={eligible.sum()} normal_replay=exact masks=match', flush=True)
    (output/f'seed{seed}.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(66,71)))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for seed in args.seeds:
        run(seed, args.output)
