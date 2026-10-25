"""Frozen-parent Signature, cyclic TRAIN and one current-rate TEST, finite extension."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import resource
import time


SCOPE = {'CMUMOSI': (66, 67), 'CMUMOSEI': (66, 67, 68), 'MELD': (66,)}


def validate_scope(dataset, seed):
    if seed not in SCOPE.get(dataset, ()):
        raise ValueError('Outside the finite authorized cyclic Signature extension')


def budget(dataset):
    return {'CMUMOSI': 150, 'CMUMOSEI': 60, 'MELD': 50}[dataset]


def learning_rate(dataset, epoch):
    if not 0 <= epoch < budget(dataset):
        raise ValueError('Epoch outside the finite stage2 budget')
    return 3e-5 if dataset == 'CMUMOSI' and epoch >= 100 else 3e-4


def selection_metric(dataset):
    return 'ACC' if dataset == 'MELD' else 'WF1'


def improves(dataset, current, previous):
    metric = selection_metric(dataset)
    return previous is None or current[metric] > previous[metric]


def measured_metrics(dataset, labels, prediction, base):
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
    labels = np.asarray(labels).reshape(-1)
    prediction, base = np.asarray(prediction), np.asarray(base)
    if not np.isfinite(prediction).all():
        raise RuntimeError('Nonfinite Signature prediction')
    if dataset == 'MELD':
        if prediction.shape != (len(labels), 7) or base.shape != prediction.shape:
            raise ValueError('MELD requires seven class logits, never flattened scores')
        truth = labels.astype(np.int64)
        new, old = prediction.argmax(-1), base.argmax(-1)
        shifted = prediction.astype(np.float64) - prediction.max(-1, keepdims=True)
        ce = np.log(np.exp(shifted).sum(-1)) - shifted[np.arange(len(labels)), truth]
        _, recalls, _, supports = precision_recall_fscore_support(
            truth, new, labels=np.arange(7), zero_division=0)
        extra = dict(CE=float(ce.mean()), UA=100*float(recalls.mean()),
                     recall=(100*recalls).tolist(), support=supports.tolist())
    else:
        prediction, base = prediction.reshape(-1), base.reshape(-1)
        if len(prediction) != len(labels) or len(base) != len(labels):
            raise ValueError('Regression prediction count differs')
        valid = labels != 0
        truth, new, old = labels[valid] > 0, prediction[valid] > 0, base[valid] > 0
        extra = dict(MSE=float(np.mean((labels-prediction)**2)))
    return dict(ACC=100*float(accuracy_score(truth, new)),
        WF1=100*float(f1_score(truth, new, average='weighted', zero_division=0)),
        N=len(truth), corrections=int(((old != truth) & (new == truth)).sum()),
        harms=int(((old == truth) & (new != truth)).sum()), **extra)


def evaluate(bundle, reader, block, rows):
    import numpy as np
    import torch
    dataset = bundle['cfg'].dataset
    block.eval()
    pieces = []
    with torch.no_grad():
        for row in rows:
            frame = row['frame'].to(next(block.parameters()).device)
            prediction = reader.decode_delta(frame, block(frame))
            valid = frame.valid.T
            predicted = prediction.transpose(0, 1)[valid].cpu().numpy()
            base = frame.prediction.transpose(0, 1)[valid].cpu().numpy()
            if dataset != 'MELD':
                predicted, base = predicted.reshape(-1), base.reshape(-1)
            pieces.append(dict(row['metadata'], availability=row['availability'],
                               prediction=predicted, base_prediction=base))
    arrays = {key: np.concatenate([row[key] for row in pieces]) for key in pieces[0]}
    score = measured_metrics(dataset, arrays['labels'], arrays['prediction'], arrays['base_prediction'])
    return score, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(SCOPE), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--rate', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    validate_scope(args.dataset, args.seed)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    from experiments.osram_three_block_joint_mosi_20261025.scheduler import gpu_inventory, guard_gpu
    host = gpu_inventory().get(args.gpu_uuid, {}).get('host_index')
    if host not in (0, 1, 2, 3):
        raise ValueError('Only healthy physical GPU0/1/2/3 authorized for this extension')
    guard_gpu(args.gpu_uuid, host)
    import fcntl
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import _atomic_arrays
    from experiments.osram_signature_cyclic68_20261025.train import train_epoch
    from osram_four_block_wiring import SignatureBlock
    from . import prepare
    rate = prepare.validate_rate(args.rate)
    torch.set_num_threads(1)
    repair = repair_uuid_rng(torch)
    if (args.output/'SUMMARY.json').exists():
        previous = json.loads((args.output/'SUMMARY.json').read_text())
        if not args.resume or previous['status'] == 'complete':
            raise FileExistsError('Explicit resume of unfinished run required')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tick, path = time.monotonic(), args.output/'SUMMARY.json'
    summary = dict(status='preflight', source_commit=args.source_commit, mode='signature',
        dataset=args.dataset, seed=args.seed, rate=rate, pid=os.getpid(),
        label='INTERNAL TEST-ORACLE SCREEN')
    try:
        bundle, reader, parent = prepare.setup(args.dataset, args.seed, args.gpu_uuid, args.output, rate)
        count = budget(args.dataset)
        bundle['cfg'] = replace(bundle['cfg'], train_rate_mode='cyclic', fixed_missing_rate=None,
            train_missing_rates=tuple(float(r) for r in prepare.RATES), epochs=count, learning_rate=3e-4)
        rows = prepare.test_cache(bundle, reader, rate)
        with h.isolated_rng(args.seed):
            block = SignatureBlock().to('cuda:0')
        before = h.backbone_digest(reader.parent)
        baseline, initial = evaluate(bundle, reader, block, rows)
        np.testing.assert_array_equal(initial['prediction'], initial['base_prediction'])
        root = Path(__file__).resolve().parents[2]
        files = list(Path(__file__).parent.glob('*.py')) + [
            root/'osram_four_block_wiring/blocks.py', root/'osram_four_block_wiring/frozen_flat.py',
            root/'experiments/osram_signature_cyclic68_20261025/train.py',
            root/'experiments/osram_signature_stage2_extension_20261025/prepare.py',
            root/'experiments/osram_neumiss_fixed_rate_stage2_20261025/prepare.py']
        identity = dict(mode='signature', dataset=args.dataset, seed=args.seed, source_commit=args.source_commit,
            smoke=args.smoke, parent=parent, training_rates=list(prepare.RATES), evaluation_rate=rate,
            cyclic_training=True, initial_block_sha256=h.backbone_digest(block),
            parent_config=parent['config'], effective_stage2_config=asdict(bundle['cfg']),
            source_hashes={str(p.relative_to(root)):h.sha(p) for p in files}, rng_repair=repair,
            mask_protocol='gcnet-random-mask-v1',
            task='original seven-class CE' if args.dataset == 'MELD' else 'original MSE including gold0',
            test_mask_sha256=hashlib.sha256(initial['availability'].tobytes()).hexdigest(),
            selection=f'one current-rate TEST {selection_metric(args.dataset)}; earliest strict tie',
            stage2_epochs=count, algorithm='unchanged SignatureBlock; original cyclic schedule; one-rate evaluation')
        optimizer = torch.optim.Adam(block.parameters(), lr=3e-4, weight_decay=1e-5)
        start, history, best = 0, [], None
        if args.resume:
            saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            if saved['identity'] != identity:
                raise ValueError('Immutable resume identity differs')
            block.load_state_dict(saved['block'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            h.restore_rng(saved['rng'])
            start, history, best = saved['epoch'], saved['history'], saved['best']
            for field, digest in (('checkpoint','checkpoint_sha256'),('predictions','prediction_sha256')):
                if h.sha(args.output/best[field]) != best[digest]:
                    raise ValueError('Saved best artifact integrity differs')
        else:
            reset_training_stochasticity(args.seed, bundle['cfg'].fold)
        summary.update(status='running', identity=identity, parent_id=parent['parent_id'], baseline=baseline,
            initial_real_batch_exact=True, environment=bundle['environment'], gpu=bundle['gpu'],
            total_parameters=sum(p.numel() for p in block.parameters()))
        h.write_json(args.output/'config.json', identity)
        h.write_json(path, summary)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start, 2 if args.smoke else count):
            optimizer.param_groups[0]['lr'] = learning_rate(args.dataset, epoch)
            row = train_epoch(bundle, reader, block, optimizer, epoch, rate, args.smoke)
            row['task_loss'] = row['task_MSE']
            if args.dataset == 'MELD':
                row['task_CE'] = row.pop('task_MSE')
            score, arrays = evaluate(bundle, reader, block, rows)
            if improves(args.dataset, score, None if best is None else best['metrics']):
                ckpt, pred = f'best_epoch_{epoch+1}.pt', f'predictions_epoch_{epoch+1}.npz'
                _atomic_arrays(args.output/pred, arrays)
                h.atomic_torch_save(args.output/ckpt, dict(block=block.state_dict(), identity=identity,
                    epoch=epoch+1, parent_id=parent['parent_id']))
                best = dict(epoch=epoch+1, metrics=score, checkpoint=ckpt, predictions=pred,
                    checkpoint_sha256=h.sha(args.output/ckpt), prediction_sha256=h.sha(args.output/pred))
            row['test'] = score
            history.append(row)
            h.atomic_torch_save(args.output/'last.pt', dict(block=block.state_dict(), optimizer=optimizer.state_dict(),
                identity=identity, epoch=epoch+1, history=history, best=best, rng=h.rng_state()))
            summary.update(epoch=epoch+1, best=best, seconds=time.monotonic()-tick,
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
            h.write_json(args.output/'history.json', history)
            h.write_json(path, summary)
            print(json.dumps(dict(dataset=args.dataset, seed=args.seed, rate=rate, epoch=epoch+1,
                current=score, best=best['metrics'], seconds=summary['seconds'])), flush=True)
        saved = torch.load(args.output/best['checkpoint'], map_location='cpu', weights_only=False)
        block.load_state_dict(saved['block'], strict=True)
        score, replay = evaluate(bundle, reader, block, rows)
        with np.load(args.output/best['predictions']) as saved_arrays:
            np.testing.assert_array_equal(replay['prediction'], saved_arrays['prediction'])
        if score != best['metrics']:
            raise ValueError('Selected checkpoint metrics differ after restoration')
        if not args.smoke and {r for row in history for r in row['training_rates']} != {float(r) for r in prepare.RATES}:
            raise ValueError('Cyclic stage2 did not cover all eight training rates')
        if not any(row['gradients']['nonzero'] for row in history):
            raise RuntimeError('Signature never received nonzero gradient')
        if before != h.backbone_digest(reader.parent) or any(p.grad is not None for p in reader.parent.parameters()):
            raise RuntimeError('Frozen parent changed or received gradient')
        if any(h.sha(root/name) != digest for name, digest in identity['source_hashes'].items()):
            raise RuntimeError('Immutable source changed')
        summary.update(status='smoke_passed' if args.smoke else 'complete', seconds=time.monotonic()-tick,
            checks=dict(parent_unchanged=True, parent_grad_none=True, checkpoint_replay_exact=True,
                        cyclic_train_rates_verified=True, only_requested_test_rate=True),
            peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        h.write_json(args.output/'RESULT.json', summary)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
