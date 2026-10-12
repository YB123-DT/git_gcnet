"""Frozen, one-slot-at-a-time scaling of original Nested corrections on MOSI."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SLOTS = ('Local', 'Base', 'Gap-A', 'Gap-T', 'Gap-V')
SCALES = (0., .5, 1., 1.5, 2.)
GPU_UUIDS = {6: 'GPU-e4cafb17-818e-216a-b94a-7440063a9153'}


def source(seed):
    remote = Path('/data2/yb/remote_experiments')
    if seed == 66:
        return remote/'osram_new40_gpu0123_20261004/attempt2/runs/nested_gnn_rooted_evidence/seed_66'
    return remote/'osram_readout_top3_3seed_20261005/runs/nested_gnn_rooted_evidence'/f'seed_{seed}'


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def wf1(labels, predictions):
    from sklearn.metrics import f1_score
    keep = labels != 0
    return 100.*f1_score(labels[keep] > 0, predictions[keep] > 0,
                         average='weighted', zero_division=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=GPU_UUIDS, default=6)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[66, 67, 68])
    args = parser.parse_args()
    uuid = GPU_UUIDS[args.gpu]
    actual = subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=uuid',
                                      '--format=csv,noheader'], text=True).strip()
    if actual != uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise RuntimeError('Expected the explicit healthy physical GPU UUID')
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                 pid=os.getpid(), server='biggpu', physical_gpu=args.gpu, gpu_uuid=uuid,
                 code_commit=args.code_commit, seeds=args.seeds, training_runs=0,
                 label='INTERNAL DIAGNOSTIC ONLY', completed=[])
    write(args.output/'STATUS.json', state)

    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules, _move_batch, _prepare_view
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    from experiments.osram_nested_delta_off_missing_text_20261010.readout import (
        flat_prediction, scale_correction_slot)
    from experiments.osram_nested_diagnostics_20261007.evaluate import state_hash

    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    dataset = Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset')
    os.environ['GCNET_DATASET_ROOT'] = str(dataset)
    features = [str(dataset/'CMUMOSI/features'/name) for name in
                ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    rows = []
    try:
        for seed in args.seeds:
            src = source(seed)
            config = TrainConfig(**json.loads((src/'config.json').read_text()))
            if (config.seed != seed or config.training_objective != 'emotion-only'
                    or config.osram_meaningful_block != 'nested_gnn_rooted_evidence'
                    or config.osram_readout_fusion != 'flat' or config.osram_bidirectional
                    or config.mosi_task_mode != 'regression'):
                raise ValueError(f'Unexpected source configuration for seed {seed}')
            train, val, test, *dims = get_loaders(
                audio_root=features[0], text_root=features[1], video_root=features[2],
                num_folder=1, dataset='CMUMOSI', batch_size=config.batch_size,
                num_workers=0, seed=seed, validation_fraction=config.validation_fraction,
                evaluation_protocol=config.evaluation_protocol)
            del train, val
            loader = test[0]
            model = _build_model(config, tuple(dims)).to(device).eval().requires_grad_(False)
            reference = json.loads((src/'metrics.json').read_text())
            for rate in [i/10 for i in range(8)]:
                suffix = f'{rate:.1f}'.replace('.', 'p')
                checkpoint = src/f'best_miss_{suffix}.pt'
                checkpoint_hash = sha(checkpoint)
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                model.load_state_dict(saved['model'], strict=True)
                epoch = int(saved['epoch'])
                del saved
                before = state_hash(model)
                labels, availability = [], []
                predictions = {(slot, scale): [] for slot in SLOTS for scale in SCALES}
                replay_errors = []
                loader.sampler.set_epoch(0)
                for raw_batch in loader:
                    view = _prepare_view(_move_batch(raw_batch, device),
                                         _schedules(config, 'test')[rate], 0, tuple(dims))
                    capture = {}
                    def pre(_module, inputs):
                        if capture:
                            raise AssertionError('Multiple Nested scans in one forward')
                        capture['raw'] = tuple(x.detach().clone() for x in inputs)
                    def post(_module, _inputs, outputs):
                        capture['corrected'] = tuple(x.detach().clone() for x in outputs)
                    handles = (model.osram.meaningful_block.register_forward_pre_hook(pre),
                               model.osram.meaningful_block.register_forward_hook(post))
                    try:
                        with torch.no_grad():
                            original = model([view['incomplete']], view['availability'], view['qmask'],
                                             view['umask'], view['lengths'], predict_missing=False)[0].squeeze(-1)
                            valid = view['umask'].T.bool()
                            replay = flat_prediction(model, *capture['raw'], readout=capture['corrected'])
                            batch_valid = view['umask'].bool()
                            baseline_batch = replay.T[batch_valid].cpu().tolist()
                            for key in predictions:
                                if key[1] == 1.:
                                    predictions[key].extend(baseline_batch)
                                    continue
                                readout = scale_correction_slot(capture['raw'], capture['corrected'], *key)
                                pred = flat_prediction(model, *capture['raw'], readout=readout)
                                predictions[key].extend(pred.T[batch_valid].cpu().tolist())
                    finally:
                        for handle in handles:
                            handle.remove()
                    replay_errors.append(float((original[valid]-replay[valid]).abs().max()))
                    labels.extend(view['labels'][batch_valid].cpu().tolist())
                    availability.extend(view['availability'].transpose(0,1)[batch_valid].cpu().tolist())
                labels_array = np.asarray(labels)
                archive = np.load(src/f'predictions_miss_{suffix}.npz')
                if not np.array_equal(labels_array, archive['labels']):
                    raise AssertionError('Label/sample order differs from archived evaluation')
                baseline = np.asarray(predictions[('Local', 1.)])
                archive_error = float(np.abs(baseline-archive['predictions'].reshape(-1)).max())
                if archive_error > 1e-5 or max(replay_errors) > 1e-5:
                    raise AssertionError('Frozen replay differs from original prediction')
                expected = 100.*reference['selected_weighted_f1_by_rate'][str(rate)]
                if abs(wf1(labels_array, baseline)-expected) > 1e-8:
                    raise AssertionError('Metric replay differs from source metrics')
                for (slot, scale), values in predictions.items():
                    pred = np.asarray(values)
                    keep = labels_array != 0
                    base_correct = (baseline[keep] > 0) == (labels_array[keep] > 0)
                    new_correct = (pred[keep] > 0) == (labels_array[keep] > 0)
                    rows.append(dict(seed=seed, rate=rate, slot=slot, scale=scale,
                                     wf1=wf1(labels_array, pred), n=len(pred),
                                     corrections=int((~base_correct & new_correct).sum()),
                                     harms=int((base_correct & ~new_correct).sum()),
                                     mean_abs_shift=float(np.abs(pred-baseline).mean())))
                if before != state_hash(model) or checkpoint_hash != sha(checkpoint):
                    raise AssertionError('Frozen model/checkpoint was mutated')
                state['completed'].append(dict(seed=seed, rate=rate, epoch=epoch,
                    checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                    state_sha256=before, replay_max_abs=max(replay_errors),
                    archive_max_abs=archive_error))
                write(args.output/'STATUS.json', state)
                print(json.dumps(state['completed'][-1]), flush=True)

        with (args.output/'per_rate.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        macro = []
        for seed in args.seeds:
            for slot in SLOTS:
                for scale in SCALES:
                    selected = [r for r in rows if r['seed']==seed and r['slot']==slot and r['scale']==scale]
                    macro.append(dict(seed=seed, slot=slot, scale=scale,
                        mean8=statistics.mean(r['wf1'] for r in selected),
                        high=statistics.mean(r['wf1'] for r in selected if r['rate']>=.5),
                        corrections=sum(r['corrections'] for r in selected),
                        harms=sum(r['harms'] for r in selected),
                        mean_abs_shift=statistics.mean(r['mean_abs_shift'] for r in selected)))
        with (args.output/'macro.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(macro[0]))
            writer.writeheader(); writer.writerows(macro)
        write(args.output/'SUMMARY.json', dict(label=state['label'], scales=SCALES, slots=SLOTS,
              protocol='Each slot scaled separately; other four corrections fixed at 1.0',
              per_rate=rows, macro=macro, records=state['completed']))
        state.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat())
        write(args.output/'STATUS.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        write(args.output/'STATUS.json', state)
        raise


if __name__ == '__main__':
    main()
