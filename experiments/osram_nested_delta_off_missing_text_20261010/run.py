"""Frozen paired Nested correction ON/OFF across eight original test masks."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.osram_nested_delta_off_missing_text_20261010.readout import flat_prediction
from experiments.osram_nested_input_gradient_20261010.compare_missing_text import wf1


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def summary(rows):
    import numpy as np
    y = np.array([r['label'] for r in rows])
    on = np.array([r['on'] for r in rows])
    off = np.array([r['off'] for r in rows])
    keep = y != 0
    a, b = wf1(y[keep], off[keep]), wf1(y[keep], on[keep])
    oc, nc = (off[keep] > 0) == (y[keep] > 0), (on[keep] > 0) == (y[keep] > 0)
    return dict(n=len(rows), nonneutral_n=int(keep.sum()),
                off_wf1=100*a if a is not None else None,
                on_wf1=100*b if b is not None else None,
                delta_pp=100*(b-a) if a is not None else None,
                corrections=int((~oc & nc).sum()), harms=int((oc & ~nc).sum()),
                both_correct=int((oc & nc).sum()), both_wrong=int((~oc & ~nc).sum()),
                flip_percent=100*float((oc != nc).mean()) if keep.any() else None,
                mean_abs_shift=float(abs(on-off).mean()) if len(rows) else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--code-commit', required=True)
    args = p.parse_args()
    uuid = 'GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e'
    actual = subprocess.check_output(['nvidia-smi', '--id=7', '--query-gpu=uuid',
                                     '--format=csv,noheader'], text=True).strip()
    assert actual == uuid == os.environ.get('CUDA_VISIBLE_DEVICES'), 'Wrong physical GPU'
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', pid=os.getpid(), server='biggpu', physical_gpu=7,
                 gpu_uuid=uuid, code_commit=args.code_commit, records=[],
                 label='INTERNAL DIAGNOSTIC ONLY', training_runs=0)
    dump(args.output/'STATUS.json', state)
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules, _move_batch, _prepare_view
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    from experiments.osram_nested_diagnostics_20261007.evaluate import state_hash
    torch.set_num_threads(2)
    config = TrainConfig(**json.loads((args.source/'config.json').read_text()))
    assert config.seed == 66 and config.training_objective == 'emotion-only'
    assert config.osram_meaningful_block == 'nested_gnn_rooted_evidence'
    assert config.osram_readout_fusion == 'flat' and not config.osram_bidirectional
    assert config.mosi_task_mode == 'regression'
    dataset = Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset')
    os.environ['GCNET_DATASET_ROOT'] = str(dataset)
    features = [str(dataset/'CMUMOSI/features'/n) for n in
                ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    train, val, test, *dims = get_loaders(audio_root=features[0], text_root=features[1],
        video_root=features[2], num_folder=1, dataset='CMUMOSI', batch_size=config.batch_size,
        num_workers=0, seed=66, validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol)
    del train, val
    loader = test[0]
    device = torch.device('cuda:0')
    model = _build_model(config, tuple(dims)).to(device).eval().requires_grad_(False)
    reference = json.loads((args.source/'metrics.json').read_text())
    rows, table = [], []
    try:
        for rate in [i/10 for i in range(8)]:
            suffix = f'{rate:.1f}'.replace('.', 'p')
            checkpoint = args.source/f'best_miss_{suffix}.pt'
            file_hash = sha(checkpoint)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model'], strict=True)
            epoch = int(saved['epoch'])
            del saved
            before = state_hash(model)
            current, predictions, labels, masks = [], [], [], []
            max_error = 0.
            loader.sampler.set_epoch(0)
            for raw in loader:
                view = _prepare_view(_move_batch(raw, device), _schedules(config, 'test')[rate], 0, tuple(dims))
                capture = {}
                def pre(module, inputs):
                    assert 'input' not in capture, 'Multiple Memory/readout scans'
                    capture['input'] = tuple(x.detach().clone() for x in inputs)
                def post(module, inputs, outputs):
                    capture['output'] = tuple(x.detach().clone() for x in outputs)
                h1 = model.osram.meaningful_block.register_forward_pre_hook(pre)
                h2 = model.osram.meaningful_block.register_forward_hook(post)
                try:
                    with torch.no_grad():
                        result = model([view['incomplete']], view['availability'], view['qmask'],
                                       view['umask'], view['lengths'], predict_missing=False)
                        on = result[0].squeeze(-1)
                        replay = flat_prediction(model, *capture['input'], readout=capture['output'])
                        off = flat_prediction(model, *capture['input'])
                finally:
                    h1.remove()
                    h2.remove()
                valid = view['umask'].T.bool()
                max_error = max(max_error, float((on[valid]-replay[valid]).abs().max()))
                assert max_error <= 1e-5, 'Full readout replay differs'
                assert torch.isfinite(off).all() and not off[~valid].count_nonzero()
                # Original _collect_predictions archives conversation-major arrays.
                batch_valid = view['umask'].bool()
                predictions.extend(on.T[batch_valid].cpu().tolist())
                labels.extend(view['labels'][batch_valid].cpu().tolist())
                masks.extend(view['availability'].transpose(0,1)[batch_valid].cpu().tolist())
                for t, b in valid.nonzero().cpu().tolist():
                    av = view['availability'][t,b].cpu().tolist()
                    r = dict(seed=66, rate=rate, conversation=str(view['conversation_ids'][b]),
                             utterance=t, pattern=''.join(n for n,a in zip('ATV', av) if a),
                             label=float(view['labels'][b,t]), on=float(on[t,b]), off=float(off[t,b]))
                    current.append(r)
            archive = np.load(args.source/f'predictions_miss_{suffix}.npz')
            assert np.array_equal(np.asarray(labels), archive['labels'])
            assert np.array_equal(np.asarray(masks), archive['availability'])
            archive_error = float(abs(np.asarray(predictions)-archive['predictions'].reshape(-1)).max())
            assert archive_error <= 1e-5, 'Original prediction archive mismatch'
            full = summary(current)
            assert abs(full['on_wf1']/100-reference['selected_weighted_f1_by_rate'][str(rate)]) < 1e-10
            assert before == state_hash(model) and file_hash == sha(checkpoint)
            assert all(not x.requires_grad and x.grad is None for x in model.parameters())
            for group in ('all', 'no_Text', 'A', 'V', 'AV'):
                selected = [r for r in current if group == 'all' or
                            (group == 'no_Text' and 'T' not in r['pattern']) or r['pattern'] == group]
                table.append(dict(rate=rate, group=group, **summary(selected)))
            record = dict(rate=rate, epoch=epoch, checkpoint=str(checkpoint),
                          checkpoint_sha256=file_hash, state_sha256=before,
                          frozen_unchanged=True, on_replay_max_abs=max_error,
                          archive_max_abs=archive_error, mask_sha256=hashlib.sha256(
                              np.asarray(masks, dtype=np.float32).tobytes()).hexdigest())
            state['records'].append(record)
            rows.extend(current)
            dump(args.output/'STATUS.json', state)
            print(json.dumps(dict(record, full=full)), flush=True)
        macro = []
        for group in ('all', 'no_Text', 'A', 'V', 'AV'):
            for rates in ('nonempty_rates', 'high_missing'):
                selected = [r for r in table if r['group'] == group and r['on_wf1'] is not None
                            and (rates != 'high_missing' or r['rate'] >= .5)]
                macro.append(dict(group=group, rates=rates, groups=len(selected),
                    **{k:statistics.mean(r[k] for r in selected) if selected else None
                       for k in ('on_wf1','off_wf1','delta_pp','flip_percent','mean_abs_shift')},
                    corrections=sum(r['corrections'] for r in selected),
                    harms=sum(r['harms'] for r in selected)))
        for name, data in (('utterances', rows), ('per_rate', table), ('macro', macro)):
            with (args.output/f'{name}.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
        dump(args.output/'SUMMARY.json', dict(label=state['label'], per_rate=table, macro=macro,
             records=state['records'], aggregation='Equal mean over nonempty rate groups; same utterance may recur across rates'))
        state['status'] = 'completed'
        dump(args.output/'STATUS.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        dump(args.output/'STATUS.json', state)
        raise


if __name__ == '__main__':
    main()
