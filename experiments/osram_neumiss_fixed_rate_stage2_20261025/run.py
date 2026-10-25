"""One frozen per-rate Flat parent, one branch, one fixed TRAIN/TEST rate."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time


def evaluation(bundle, reader, block, rows):
    import numpy as np
    import torch
    from experiments.osram_four_block_mosi_20261025.run import metrics
    block.eval()
    pieces = []
    with torch.no_grad():
        for row in rows:
            frame = row['frame'].to(next(block.parameters()).device)
            prediction = reader.decode_delta(frame, block(frame))
            valid = frame.valid.T
            pieces.append(dict(row['metadata'], availability=row['availability'],
                prediction=prediction.transpose(0, 1)[valid].cpu().numpy().reshape(-1),
                base_prediction=frame.prediction.transpose(0, 1)[valid].cpu().numpy().reshape(-1)))
    arrays = {key: np.concatenate([row[key] for row in pieces]) for key in pieces[0]}
    if not np.isfinite(arrays['prediction']).all():
        raise RuntimeError('Nonfinite fixed-rate prediction')
    return metrics(bundle['trainer'], arrays['labels'], arrays['prediction'], arrays['base_prediction']), arrays


def train_epoch(bundle, reader, block, optimizer, epoch, rate, smoke):
    import torch
    from .prepare import train_schedule
    trainer, cfg, loader = bundle['trainer'], bundle['cfg'], bundle['train_loader']
    device = next(block.parameters()).device
    loader.sampler.set_epoch(epoch)
    schedule = train_schedule(bundle, rate)  # Never a cyclic-rate selector.
    block.train()
    reader.parent.eval()
    total, count, steps, coverage = 0., 0, 0, []
    trace = hashlib.sha256()
    for index, raw in enumerate(loader):
        if smoke and index >= 1:
            break
        data = trainer._move_batch(raw, device)
        view = trainer._prepare_view(data, schedule, epoch, bundle['dimensions'])
        frame = reader(view['incomplete'], view['availability'], view['qmask'], view['umask'], view['lengths'])
        optimizer.zero_grad(set_to_none=True)
        prediction = reader.decode_delta(frame, block(frame))
        loss = trainer._task_loss(cfg.dataset, prediction, view['labels'], view['umask'],
            cfg.mosi_task_mode, cfg.task_regression_loss, cfg.task_smooth_l1_beta)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite fixed-rate task MSE')
        loss.backward()
        gradients = [p.grad for p in block.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Disconnected or nonfinite branch gradient')
        coverage.append(dict(connected=len(gradients), nonzero=sum(int(g.ne(0).any()) for g in gradients)))
        norm = torch.nn.utils.clip_grad_norm_(block.parameters(), cfg.gradient_clip_norm)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite clipped norm')
        optimizer.step()
        n = int(view['umask'].sum())
        total += float(loss.detach())*n
        count += n
        steps += 1
        trace.update(json.dumps(list(map(str, view['conversation_ids']))).encode())
        trace.update(view['availability'].detach().cpu().numpy().tobytes())
    if not count:
        raise RuntimeError('Empty fixed-rate training epoch')
    return dict(epoch=epoch+1, training_rate=rate, task_MSE=total/count, N=count,
        optimizer_steps=steps, mask_trace_sha256=trace.hexdigest(),
        gradients=coverage[-1], learning_rate=optimizer.param_groups[0]['lr'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('neumiss', 'signature'), required=True)
    parser.add_argument('--rate', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    from experiments.osram_three_block_joint_mosi_20261025.scheduler import gpu_inventory, guard_gpu
    inventory = gpu_inventory()
    host = inventory.get(args.gpu_uuid, {}).get('host_index')
    if host not in (0, 3):
        raise ValueError('This finite stage2 queue uses only healthy physical GPU0/3')
    guard_gpu(args.gpu_uuid, host)
    import fcntl
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import _atomic_arrays
    from osram_four_block_wiring import NeuMissBlock, SignatureBlock
    from . import prepare
    rate = prepare.validate_rate(args.rate)
    torch.set_num_threads(1)
    repair = repair_uuid_rng(torch)
    if (args.output/'SUMMARY.json').exists() and not args.resume:
        raise FileExistsError('Do not overwrite a previous stage2 run')
    if (args.output/'SUMMARY.json').exists() and json.loads((args.output/'SUMMARY.json').read_text()).get('status') == 'complete':
        raise ValueError('Completed stage2 run must not be resumed or repeated silently')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tick, summary_path = time.monotonic(), args.output/'SUMMARY.json'
    summary = dict(status='preflight', source_commit=args.source_commit, mode=args.mode,
                   rate=rate, seed=66, pid=os.getpid(), label='INTERNAL TEST-ORACLE SCREEN')
    try:
        bundle, reader, parent_identity = prepare.setup(args.gpu_uuid, args.output, rate)
        # Parent architecture/config is verified before declaring the separate stage2 protocol.
        bundle['cfg'] = replace(bundle['cfg'], train_rate_mode='fixed', fixed_missing_rate=float(rate),
                                train_missing_rates=(float(rate),))
        rows = prepare.test_cache(bundle, reader, rate)  # Only this TEST rate is read/inferred.
        with h.isolated_rng(66):
            block = (NeuMissBlock() if args.mode == 'neumiss' else SignatureBlock()).to('cuda:0')
        before = h.backbone_digest(reader.parent)
        baseline, initial_arrays = evaluation(bundle, reader, block, rows)
        np.testing.assert_array_equal(initial_arrays['prediction'], initial_arrays['base_prediction'])
        root = Path(__file__).resolve().parents[2]
        files = list(Path(__file__).parent.glob('*.py')) + list((root/'osram_four_block_wiring').glob('*.py'))
        identity = dict(mode=args.mode, seed=66, source_commit=args.source_commit, smoke=args.smoke,
            parent=parent_identity, training_rate=rate, evaluation_rate=rate, cyclic_training=False,
            parent_config=parent_identity['config'], effective_stage2_config=asdict(bundle['cfg']),
            mask_protocol='gcnet-random-mask-v1', source_hashes={str(p.relative_to(root)):h.sha(p) for p in files},
            rng_repair=repair, test_mask_sha256=hashlib.sha256(initial_arrays['availability'].tobytes()).hexdigest(),
            selection='TEST W-F1 at the one requested rate; earliest strict tie', task='original MSE including gold0')
        optimizer = torch.optim.Adam(block.parameters(), lr=3e-4, weight_decay=1e-5)
        summary.update(status='running', identity=identity, parent_id=parent_identity['parent_id'],
            baseline=baseline, initial_real_batch_exact=True, environment=bundle['environment'],
            total_parameters=sum(p.numel() for p in block.parameters()), gpu=bundle['gpu'])
        start, history, best = 0, [], None
        if args.resume:
            saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            if saved['identity'] != identity:
                raise ValueError('Immutable stage2 resume identity changed')
            block.load_state_dict(saved['block'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            h.restore_rng(saved['rng'])
            start, history, best = saved['epoch'], saved['history'], saved['best']
            for field, digest in (('checkpoint','checkpoint_sha256'), ('predictions','prediction_sha256')):
                if h.sha(args.output/best[field]) != best[digest]:
                    raise ValueError('Resume best ledger integrity failure')
        else:
            reset_training_stochasticity(66, bundle['cfg'].fold)
        h.write_json(args.output/'config.json', identity)
        h.write_json(summary_path, summary)
        torch.cuda.reset_peak_memory_stats()
        epochs = 2 if args.smoke else 150
        if not prepare.should_train(rate, args.mode):
            epochs = 0  # NeuMiss is identically zero at ATV: no fake optimizer updates.
        for epoch in range(start, epochs):
            optimizer.param_groups[0]['lr'] = 3e-4 if epoch < 100 else 3e-4/10
            row = train_epoch(bundle, reader, block, optimizer, epoch, rate, args.smoke)
            score, arrays = evaluation(bundle, reader, block, rows)
            if best is None or score['WF1'] > best['metrics']['WF1']:
                checkpoint, predictions = f'best_epoch_{epoch+1}.pt', f'predictions_epoch_{epoch+1}.npz'
                _atomic_arrays(args.output/predictions, arrays)
                h.atomic_torch_save(args.output/checkpoint, dict(block=block.state_dict(), identity=identity,
                    epoch=epoch+1, parent_id=parent_identity['parent_id']))
                best = dict(epoch=epoch+1, metrics=score, checkpoint=checkpoint, predictions=predictions,
                    checkpoint_sha256=h.sha(args.output/checkpoint), prediction_sha256=h.sha(args.output/predictions))
            row['test'] = score
            history.append(row)
            h.atomic_torch_save(args.output/'last.pt', dict(block=block.state_dict(), optimizer=optimizer.state_dict(),
                identity=identity, epoch=epoch+1, history=history, best=best, rng=h.rng_state()))
            summary.update(epoch=epoch+1, best=best, seconds=time.monotonic()-tick,
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            h.write_json(args.output/'history.json', history)
            h.write_json(summary_path, summary)
            print(json.dumps(dict(mode=args.mode, rate=rate, epoch=epoch+1, current=score,
                                  best=best['metrics'], seconds=summary['seconds'])), flush=True)
        if epochs == 0:
            _atomic_arrays(args.output/'predictions_noop.npz', initial_arrays)
            best = dict(epoch=0, metrics=baseline, predictions='predictions_noop.npz',
                        prediction_sha256=h.sha(args.output/'predictions_noop.npz'), no_op=True)
            summary.update(epoch=0, best=best, no_op=True)
            h.write_json(args.output/'history.json', [])
        else:
            saved = torch.load(args.output/best['checkpoint'], map_location='cpu', weights_only=False)
            block.load_state_dict(saved['block'], strict=True)
            score, replay = evaluation(bundle, reader, block, rows)
            with np.load(args.output/best['predictions']) as saved_arrays:
                np.testing.assert_array_equal(replay['prediction'], saved_arrays['prediction'])
            if score != best['metrics']:
                raise ValueError('Checkpoint replay metrics differ')
            if not any(row['gradients']['nonzero'] > 0 for row in history):
                raise RuntimeError('Branch never received nonzero task gradient')
        if before != h.backbone_digest(reader.parent) or any(p.grad is not None for p in reader.parent.parameters()):
            raise RuntimeError('Frozen parent changed or received gradient')
        if any(h.sha(root/name) != digest for name,digest in identity['source_hashes'].items()):
            raise RuntimeError('Source changed during run')
        summary.update(status='smoke_passed' if args.smoke else 'complete',
            checks=dict(parent_unchanged=True, parent_grad_none=True, checkpoint_replay_exact=epochs > 0,
                        baseline_replay_exact=True,
                        only_requested_train_rate=True, only_requested_test_rate=True),
            seconds=time.monotonic()-tick, peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
        h.write_json(args.output/'RESULT.json', summary)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        h.write_json(summary_path, summary)


if __name__ == '__main__':
    main()
