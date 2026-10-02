"""Existing paired-view checkpoints, original fixed-pattern evaluator; no training."""
import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_history_drift_20261002.run import read, write, sha, state_hash, verify_gpu, DEFAULT_DATA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ['GCNET_DATASET_ROOT'] = str(DEFAULT_DATA)
    gpu = verify_gpu()
    import torch
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_missing_m3.train_gcnet import TrainConfig, _attach_history_projector
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model, _evaluate_pattern, _roots
    torch.set_num_threads(2)
    patterns = dict(A=(1,0,0), T=(0,1,0), V=(0,0,1), AT=(1,1,0), AV=(1,0,1), TV=(0,1,1))
    report = dict(status='running', seed=66, gpu=gpu, training=False,
                  checkpoint_selection='inherited per-rate Test-oracle', records=[],
                  source_sha256={str(p.relative_to(REPO)):sha(p) for p in
                                 [Path(__file__), REPO/'gcnet_missing_m3/model.py',
                                  REPO/'gcnet_missing_m3/osram.py',
                                  REPO/'experiments/osram_cfg84_fixed_modality_ablation_20260923/run.py']})
    write(args.output, report)
    for arm in ('control', 'contrastive'):
        source = Path('/data1/yb/remote_experiments/osram_paired_history_views_20261001/runs')/arm
        config = TrainConfig(**read(source/'config.json'))
        assert config.seed == 66 and config.paired_history_views
        _, _, test, *dims = get_loaders(audio_root=_roots()[0], text_root=_roots()[1], video_root=_roots()[2],
            num_folder=1, dataset='CMUMOSI', batch_size=config.batch_size, num_workers=0, seed=66,
            validation_fraction=config.validation_fraction, evaluation_protocol=config.evaluation_protocol)
        model = _build_model(config, tuple(dims)).cuda()
        _attach_history_projector(model, config)
        model.requires_grad_(False).eval()
        calls = []
        hook = model.history_contrast_projector.register_forward_pre_hook(lambda *_: calls.append(1))
        for pattern, bits in patterns.items():
            rate = '.7' if sum(bits)==1 else '.3'
            checkpoint = source/f'best_miss_0p{rate[-1]}.pt'
            digest = sha(checkpoint)
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            before = state_hash(model)
            test[0].sampler.set_epoch(0)
            metrics = _evaluate_pattern(model, test[0], bits, tuple(dims), torch.device('cuda:0'))
            assert metrics['sample_count']==686 and metrics['nonzero_sample_count']==656
            assert before==state_hash(model) and digest==sha(checkpoint) and not calls
            record = dict(arm=arm, pattern=pattern, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                checkpoint_epoch=int(state['epoch']), checkpoint_rate=float(rate), state_sha256=before,
                frozen_hash_check=True, projector_calls=0, config_sha256=sha(source/'config.json'), **metrics)
            report['records'].append(record)
            write(args.output, report)
            print(arm, pattern, metrics['weighted_f1'], flush=True)
        hook.remove()
    report['status']='complete'
    write(args.output, report)


if __name__ == '__main__':
    main()
