"""Evaluation-only removal of the trained Relation residual, never its weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def verify_replay(actual, expected):
    import numpy as np
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0, equal_nan=False)
    np.testing.assert_array_equal(actual > 0, expected > 0)
    return float(np.max(np.abs(actual - expected)))


class ResidualIntervention:
    def __init__(self, disabled):
        self.disabled = disabled
        self.digest = hashlib.sha256()
        self.calls = 0

    def __call__(self, module, args, output):
        import torch
        self.calls += 1
        for x in args:  # Local, Base, Gap, availability, umask, original Flat anchor.
            self.digest.update(str(tuple(x.shape)).encode())
            self.digest.update(x.detach().cpu().contiguous().numpy().tobytes())
        if self.disabled:
            module.last_diagnostics = dict(module.last_diagnostics,
                relation_residual_norm=0., relation_anchor_norm_ratio=0.)
            return torch.zeros_like(output)
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules, evaluate_rate
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model, _roots
    from experiments.osram_paired_history_rho025_20261002.sweep import gpu_check, UUIDS
    from experiments.osram_cfg84_history_query_random_20260928.run import sha, write

    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '5'
    assert gpu_check('5') > 3000
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.source / 'config.json').read_text())
    c = TrainConfig(**config)
    assert c.seed == 66 and c.osram_relation_block and c.osram_relation_mode == 'pairwise'
    assert c.osram_readout_fusion == 'flat' and c.training_objective == 'emotion-only'
    record = dict(status='running', training=False, evaluation_only=True, source=str(args.source),
        code_commit=args.commit, pid=os.getpid(), server='biggpu', gpu=5, gpu_uuid=UUIDS['5'],
        intervention='relation_block forward-output hook returns zeros_like(residual)',
        checkpoint_selection='Reuse original Relation per-rate BEST; never reselect for R_off',
        source_sha256={p: sha(ROOT / p) for p in ('gcnet_missing_m3/model.py',
            'gcnet_missing_m3/osram.py','gcnet_missing_m3/train_gcnet.py',
            str(Path(__file__).relative_to(ROOT)))})
    write(args.output / 'PROVENANCE.json', record)
    try:
        torch.set_num_threads(2)
        _, _, loaders, ad, td, vd = get_loaders(audio_root=_roots()[0], text_root=_roots()[1],
            video_root=_roots()[2], num_folder=1, dataset=c.dataset, batch_size=c.batch_size,
            num_workers=0, seed=c.seed, validation_fraction=c.validation_fraction,
            evaluation_protocol=c.evaluation_protocol)
        dims = (ad, td, vd)
        model = _build_model(c, dims).cuda().eval()
        model.requires_grad_(False)
        schedules = _schedules(c, 'test')
        rows = []
        for rate in [i / 10 for i in range(8)]:
            key = f'{rate:.1f}'
            suffix = key.replace('.', 'p')
            checkpoint = args.source / f'best_miss_{suffix}.pt'
            checkpoint_hash = sha(checkpoint)
            state = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(state['model'], strict=True)
            upstream = None
            with np.load(args.source / f'predictions_miss_{suffix}.npz') as f:
                reference = {k: f[k].copy() for k in f.files}
            for disabled in (False, True):
                intervention = ResidualIntervention(disabled)
                handle = model.osram.relation_block.register_forward_hook(intervention)
                try:
                    with torch.no_grad():
                        metrics, artifacts = evaluate_rate(model, loaders[0], schedules[rate],
                            c.dataset, dims, torch.device('cuda:0'), True, c.mosi_task_mode,
                            c.task_regression_loss, c.task_smooth_l1_beta)
                finally:
                    handle.remove()
                assert intervention.calls > 0
                for k in ('labels', 'availability'):
                    np.testing.assert_array_equal(artifacts[k], reference[k])
                if not disabled:
                    replay_error = verify_replay(artifacts['predictions'], reference['predictions'])
                    upstream = intervention.digest.hexdigest()
                    on_metrics = metrics
                else:
                    assert intervention.digest.hexdigest() == upstream, 'Upstream read/Flat anchor changed'
                    assert metrics['mask_sha256'] == on_metrics['mask_sha256']
                    assert metrics['current_history_relation']['relation_residual_norm'] == 0
                    np.savez_compressed(args.output / f'predictions_miss_{suffix}.npz', **artifacts)
                    rows.append(dict(rate=rate, epoch=state['epoch'], checkpoint=str(checkpoint),
                        checkpoint_sha256=checkpoint_hash, upstream_sha256=upstream,
                        on_replay_max_abs_error=replay_error, on_replay_polarity_exact=True,
                        on_metrics=on_metrics, metrics=metrics))
                    write(args.output / 'rows.json', rows)
                    print(f'rate={key} on={100*on_metrics["weighted_f1"]:.6f} off={100*metrics["weighted_f1"]:.6f}', flush=True)
            for k, v in model.state_dict().items():
                assert torch.equal(v.cpu(), state['model'][k]), f'Model state changed: {k}'
            assert sha(checkpoint) == checkpoint_hash
        record['status'] = 'complete'
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        write(args.output / 'PROVENANCE.json', record)


if __name__ == '__main__':
    main()
