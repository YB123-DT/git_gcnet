"""Signature-only extension of fixed-rate frozen-parent stage-two experiments."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time


def budget(dataset):
    if dataset not in ('CMUMOSI', 'CMUMOSEI'):
        raise ValueError('Only MOSI/MOSEI Signature extension authorized')
    return 150 if dataset == 'CMUMOSI' else 60


def learning_rate(dataset, epoch):
    if not 0 <= epoch < budget(dataset):
        raise ValueError('Epoch outside the finite stage2 budget')
    return 3e-4/10 if dataset == 'CMUMOSI' and epoch >= 100 else 3e-4


def evaluate(bundle, reader, block, rows):
    import numpy as np
    import torch
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
    labels, prediction, base = (arrays[k] for k in ('labels','prediction','base_prediction'))
    if not np.isfinite(prediction).all():
        raise RuntimeError('Nonfinite frozen Signature prediction')
    measured = bundle['trainer']._metrics(bundle['cfg'].dataset, labels, prediction, 'regression')
    valid = labels != 0
    old, new, truth = base[valid] > 0, prediction[valid] > 0, labels[valid] > 0
    return dict(ACC=100*measured['accuracy'], WF1=100*measured['weighted_f1'],
        MSE=float(np.mean((labels-prediction)**2)),
        corrections=int(((old != truth) & (new == truth)).sum()),
        harms=int(((old == truth) & (new != truth)).sum()), N=int(valid.sum())), arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('CMUMOSI','CMUMOSEI'), required=True)
    parser.add_argument('--seed', type=int, choices=(66,67,68), required=True)
    parser.add_argument('--rate', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not ((args.dataset == 'CMUMOSI' and args.seed in (67,68))
            or (args.dataset == 'CMUMOSEI' and args.seed == 66)):
        raise ValueError('Reuse completed MOSI66; no other seeds authorized')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    from experiments.osram_three_block_joint_mosi_20261025.scheduler import gpu_inventory, guard_gpu
    host = gpu_inventory().get(args.gpu_uuid, {}).get('host_index')
    if host not in (0,1,3):
        raise ValueError('Only healthy physical GPU0/1/3 authorized for this queue')
    guard_gpu(args.gpu_uuid, host)
    import fcntl
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import _atomic_arrays
    from experiments.osram_neumiss_fixed_rate_stage2_20261025.run import train_epoch
    from osram_four_block_wiring import SignatureBlock
    from . import prepare
    rate = prepare.validate_rate(args.rate)
    torch.set_num_threads(1)
    repair = repair_uuid_rng(torch)
    if (args.output/'SUMMARY.json').exists():
        previous = json.loads((args.output/'SUMMARY.json').read_text())
        if not args.resume or previous['status'] == 'complete':
            raise FileExistsError('Explicit resume of unfinished run required; no duplicate MOSI66')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tick, path = time.monotonic(), args.output/'SUMMARY.json'
    summary = dict(status='preflight', source_commit=args.source_commit, mode='signature',
                   dataset=args.dataset, seed=args.seed, rate=rate, pid=os.getpid(),
                   label='INTERNAL TEST-ORACLE SCREEN')
    try:
        bundle, reader, parent = prepare.setup(args.dataset,args.seed,args.gpu_uuid,args.output,rate)
        count = budget(args.dataset)
        bundle['cfg'] = replace(bundle['cfg'], train_rate_mode='fixed', fixed_missing_rate=float(rate),
            train_missing_rates=(float(rate),), epochs=count, learning_rate=3e-4)
        rows = prepare.test_cache(bundle,reader,rate)
        with h.isolated_rng(args.seed):
            block = SignatureBlock().to('cuda:0')
        before = h.backbone_digest(reader.parent)
        baseline, initial = evaluate(bundle,reader,block,rows)
        np.testing.assert_array_equal(initial['prediction'],initial['base_prediction'])
        root = Path(__file__).resolve().parents[2]
        files = list(Path(__file__).parent.glob('*.py')) + [root/'osram_four_block_wiring/blocks.py',
            root/'osram_four_block_wiring/frozen_flat.py',
            root/'experiments/osram_neumiss_fixed_rate_stage2_20261025/run.py',
            root/'experiments/osram_neumiss_fixed_rate_stage2_20261025/prepare.py']
        identity = dict(mode='signature', dataset=args.dataset, seed=args.seed, source_commit=args.source_commit,
            smoke=args.smoke, parent=parent, training_rate=rate, evaluation_rate=rate, cyclic_training=False,
            parent_config=parent['config'], effective_stage2_config=asdict(bundle['cfg']),
            source_hashes={str(p.relative_to(root)):h.sha(p) for p in files}, rng_repair=repair,
            mask_protocol='gcnet-random-mask-v1', task='original MSE including gold0',
            test_mask_sha256=hashlib.sha256(initial['availability'].tobytes()).hexdigest(),
            selection='one current-rate TEST W-F1; earliest strict tie', stage2_epochs=count,
            algorithm='exact unchanged SignatureBlock and prior stage2 train_epoch')
        optimizer = torch.optim.Adam(block.parameters(),lr=3e-4,weight_decay=1e-5)
        start,history,best = 0,[],None
        if args.resume:
            saved = torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
            if saved['identity'] != identity:
                raise ValueError('Immutable extension resume identity differs')
            block.load_state_dict(saved['block'],strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            h.restore_rng(saved['rng'])
            start,history,best = saved['epoch'],saved['history'],saved['best']
            for field,digest in (('checkpoint','checkpoint_sha256'),('predictions','prediction_sha256')):
                if h.sha(args.output/best[field]) != best[digest]:
                    raise ValueError('Saved best artifact integrity differs')
        else:
            reset_training_stochasticity(args.seed,bundle['cfg'].fold)
        summary.update(status='running',identity=identity,parent_id=parent['parent_id'],baseline=baseline,
            initial_real_batch_exact=True,environment=bundle['environment'],gpu=bundle['gpu'],
            total_parameters=sum(p.numel() for p in block.parameters()))
        h.write_json(args.output/'config.json',identity)
        h.write_json(path,summary)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start,2 if args.smoke else count):
            optimizer.param_groups[0]['lr'] = learning_rate(args.dataset,epoch)
            row = train_epoch(bundle,reader,block,optimizer,epoch,rate,args.smoke)
            score,arrays = evaluate(bundle,reader,block,rows)
            if best is None or score['WF1'] > best['metrics']['WF1']:
                ckpt,pred = f'best_epoch_{epoch+1}.pt',f'predictions_epoch_{epoch+1}.npz'
                _atomic_arrays(args.output/pred,arrays)
                h.atomic_torch_save(args.output/ckpt,dict(block=block.state_dict(),identity=identity,
                    epoch=epoch+1,parent_id=parent['parent_id']))
                best = dict(epoch=epoch+1,metrics=score,checkpoint=ckpt,predictions=pred,
                            checkpoint_sha256=h.sha(args.output/ckpt),prediction_sha256=h.sha(args.output/pred))
            row['test'] = score
            history.append(row)
            h.atomic_torch_save(args.output/'last.pt',dict(block=block.state_dict(),optimizer=optimizer.state_dict(),
                identity=identity,epoch=epoch+1,history=history,best=best,rng=h.rng_state()))
            summary.update(epoch=epoch+1,best=best,seconds=time.monotonic()-tick,
                           peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            h.write_json(args.output/'history.json',history)
            h.write_json(path,summary)
            print(json.dumps(dict(dataset=args.dataset,seed=args.seed,rate=rate,epoch=epoch+1,
                current=score,best=best['metrics'],seconds=summary['seconds'])),flush=True)
        saved = torch.load(args.output/best['checkpoint'],map_location='cpu',weights_only=False)
        block.load_state_dict(saved['block'],strict=True)
        score,replay = evaluate(bundle,reader,block,rows)
        with np.load(args.output/best['predictions']) as saved_arrays:
            np.testing.assert_array_equal(replay['prediction'],saved_arrays['prediction'])
        if score != best['metrics']:
            raise ValueError('Selected checkpoint metrics differ after restoration')
        if not any(row['gradients']['nonzero'] for row in history):
            raise RuntimeError('Signature never received nonzero gradient')
        if before != h.backbone_digest(reader.parent) or any(p.grad is not None for p in reader.parent.parameters()):
            raise RuntimeError('Frozen parent changed or received gradient')
        if any(h.sha(root/name) != digest for name,digest in identity['source_hashes'].items()):
            raise RuntimeError('Immutable source changed')
        summary.update(status='smoke_passed' if args.smoke else 'complete',seconds=time.monotonic()-tick,
            checks=dict(parent_unchanged=True,parent_grad_none=True,checkpoint_replay_exact=True,
                        only_requested_train_rate=True,only_requested_test_rate=True),
            peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
        h.write_json(args.output/'RESULT.json',summary)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        raise
    finally:
        h.write_json(path,summary)


if __name__ == '__main__':
    main()
