"""Finite, from-scratch one-stage MOSI experiments. INTERNAL TEST-ORACLE SCREEN."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
ACCEPTED = Path('/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2')
RATES = tuple(str(i / 10) for i in range(8))


def forward(model, view, generator=None, query_keys=None):
    import torch
    if hasattr(model, 'mode'):
        return model(view, generator=generator, query_keys=query_keys)
    prediction = model([view['incomplete']], view['availability'], view['qmask'],
                       view['umask'], view['lengths'], predict_missing=False)[0]
    return dict(prediction=prediction, base_prediction=prediction,
                aux_loss=prediction.new_zeros(()))


def train_epoch(bundle, model, optimizer, generator, epoch, smoke):
    import torch
    from gcnet_missing_m3.mixed_rate import BalancedBatchRateSchedule
    trainer, cfg, loader = bundle['trainer'], bundle['cfg'], bundle['train_loader']
    loader.sampler.set_epoch(epoch)
    schedules = trainer._schedules(cfg, 'train')
    cycle = BalancedBatchRateSchedule(rates=cfg.train_missing_rates)
    model.train()
    digest, rows = hashlib.sha256(), []
    for index, raw in enumerate(loader):
        if smoke and index >= 1:
            break
        data = trainer._move_batch(raw, next(model.parameters()).device)
        view = trainer._prepare_view(data, schedules[cycle.rate_for(epoch, index)], epoch,
                                     bundle['dimensions'])
        optimizer.zero_grad(set_to_none=True)
        result = forward(model, view, generator)
        task = trainer._task_loss(cfg.dataset, result['prediction'], view['labels'], view['umask'],
            cfg.mosi_task_mode, cfg.task_regression_loss, cfg.task_smooth_l1_beta)
        loss = task + result['aux_loss']
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite joint objective')
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            group = ('parent' if name.startswith('parent.') or not hasattr(model, 'mode')
                     else name.split('.')[0])
            row = gradients.setdefault(group, dict(connected=0, nonzero=0))
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError('Nonfinite gradient: ' + name)
                row['connected'] += 1
                row['nonzero'] += int(parameter.grad.ne(0).any())
        if not gradients.get('parent', {}).get('nonzero'):
            raise RuntimeError('Original backbone has no task gradient')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite clipping norm')
        optimizer.step()
        n = int(view['umask'].sum())
        rows.append(dict(N=n, task_MSE=float(task.detach()), aux_loss=float(result['aux_loss'].detach()),
                         gradient_norm=float(norm), gradients=gradients))
        digest.update(json.dumps([str(v) for v in data[-1]], separators=(',', ':')).encode())
        digest.update(view['availability'].detach().cpu().numpy().tobytes())
    count = sum(row['N'] for row in rows)
    return dict(epoch=epoch+1, batches=len(rows), optimizer_steps=len(rows), memory_scans=len(rows),
        N=count, task_MSE=sum(r['task_MSE']*r['N'] for r in rows)/count,
        aux_loss=sum(r['aux_loss']*r['N'] for r in rows)/count,
        gradients=rows[-1]['gradients'], gradient_norm_max=max(r['gradient_norm'] for r in rows),
        mask_trace_sha256=digest.hexdigest(), learning_rates=[g['lr'] for g in optimizer.param_groups])


def evaluate(bundle, model, bank, rate, smoke):
    import numpy as np
    import torch
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    from experiments.osram_tasr_20261018 import runner_helpers as h
    loader, trainer = bundle['test_loader'], bundle['trainer']
    loader.sampler.set_epoch(0)
    model.eval()
    pieces = {}
    with h.isolated_rng(99259), torch.no_grad():
        for index, raw in enumerate(loader):
            if smoke and index >= 1:
                break
            data = trainer._move_batch(raw, next(model.parameters()).device)
            meta, record = metadata(data), bank[index]
            for key, value in meta.items():
                np.testing.assert_array_equal(value, record[key])
            a = record['availability'].to(data[7].device)
            view = trainer._prepare_view_from_primary_masks(data, a, a, bundle['dimensions'])
            ids = [str(v) for v in data[-1]]
            keys = [[f'CMUMOSI:{rate}:{ids[b]}:{t}' for b in range(len(ids))]
                    for t in range(a.shape[0])]
            result = forward(model, view, query_keys=keys)
            values = dict(meta, availability=a.transpose(0, 1)[data[7].bool()].cpu().numpy())
            for key in ('prediction', 'base_prediction'):
                values[key] = result[key].transpose(0, 1)[data[7].bool()].cpu().numpy()
            for key, value in values.items():
                pieces.setdefault(key, []).append(value)
    arrays = {key: np.concatenate(value) for key, value in pieces.items()}
    metrics = {}
    for tag, key in (('router', 'prediction'), ('full', 'base_prediction')):
        pred = arrays[key].reshape(-1)
        labels = arrays['labels'].reshape(-1)
        measured = trainer._metrics('CMUMOSI', labels, pred, 'regression')
        metrics[tag] = dict(ACC=100*measured['accuracy'], WF1=100*measured['weighted_f1'],
                            MSE=float(np.mean((pred-labels)**2)))
    valid = arrays['labels'].reshape(-1) != 0
    truth = arrays['labels'].reshape(-1)[valid] > 0
    old = arrays['base_prediction'].reshape(-1)[valid] > 0
    new = arrays['prediction'].reshape(-1)[valid] > 0
    metrics['paired_vs_same_backbone_branch_off'] = dict(
        corrections=int(((old != truth) & (new == truth)).sum()),
        harms=int(((old == truth) & (new != truth)).sum()))
    metrics['mask_sha256'] = hashlib.sha256(arrays['availability'].tobytes()).hexdigest()
    return metrics, arrays


def aggregate(best):
    return {tag: {scope: {key: sum(best[r]['metrics'][tag][key] for r in rates)/len(rates)
                          for key in ('ACC', 'WF1', 'MSE')}
                  for scope, rates in (('mean', RATES), ('high', RATES[5:]))}
            for tag in ('router', 'full')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('flat', 'neumiss', 'signature', 'csdi'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    from .scheduler import guard_gpu
    guard_gpu(args.gpu_uuid, 1 if args.mode == 'csdi' else 0)
    import fcntl
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from experiments.osram_mosi_cmerc_20261023 import worker as control
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from .model import JointBlockModel
    torch.set_num_threads(1)
    repair = repair_uuid_rng(torch)
    # The scheduler creates the directory for its log; never accept existing artifacts silently.
    if (args.output/'SUMMARY.json').exists() and not args.resume:
        raise FileExistsError('Existing experiment; explicit --resume required')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tick, summary_path = time.monotonic(), args.output/'SUMMARY.json'
    summary = dict(status='preflight', mode=args.mode, source_commit=args.source_commit,
                   seed=66, pid=os.getpid(), label='INTERNAL TEST-ORACLE SCREEN')
    try:
        prior = json.loads((control.REFERENCE/'SUMMARY.json').read_text())
        accepted = json.loads((ACCEPTED/'SUMMARY.json').read_text())
        bundle = control.setup(args.gpu_uuid, args.output, prior)
        parent, cfg = bundle['model'], bundle['cfg']
        if h.backbone_digest(parent) != accepted['initial_backbone_sha256']:
            raise ValueError('Fresh original initialization differs')
        projection, projection_proof = None, None
        if args.mode == 'csdi':
            from experiments.osram_four_block_mosi_20261025.prepare import fit_projection
            projection, projection_proof = fit_projection(bundle, SimpleNamespace(parent=parent))
        with h.isolated_rng(66):
            model = (parent if args.mode == 'flat' else JointBlockModel(parent, args.mode, projection)).to('cuda:0')
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
        generator = torch.Generator(device='cuda:0').manual_seed(6601)
        masks = torch.load(ACCEPTED/'fixed_masks.pt', map_location='cpu', weights_only=False)
        hashes = {r: hashlib.sha256(b''.join(row['availability'].numpy().tobytes()
                  for row in masks['test'][r])).hexdigest() for r in RATES}
        if hashes != accepted['identity']['mask_hashes']['test']:
            raise ValueError('Accepted exact TEST masks changed')
        source_files = list(prior['identity']['source_hashes'])
        source_files += [str(p.relative_to(ROOT)) for folder in
            (ROOT/'osram_four_block_wiring', Path(__file__).parent) for p in folder.glob('*.py')]
        identity = dict(mode=args.mode, source_commit=args.source_commit, seed=66, smoke=args.smoke,
            source_hashes={p: h.sha(ROOT/p) for p in source_files}, config=asdict(cfg),
            environment=bundle['environment'], data_manifest=bundle['data_manifest'],
            split=bundle['split'], seed_protocol=bundle['seed_protocol'], mask_hashes=hashes,
            rng_repair=repair, old_weights_loaded=False, initialization=bundle['initial_backbone_sha256'],
            projection=projection_proof, csdi_noise_loss_weight=1.0,
            selection='test-oracle-per-rate; weighted_f1; earliest tie')
        summary.update(identity=identity, total_parameters=sum(p.numel() for p in model.parameters()),
            added_parameters=sum(p.numel() for p in model.parameters())-13509793,
            gpu=bundle['gpu'], environment=bundle['environment'], status='running')
        reset_training_stochasticity(66, cfg.fold)
        start, history, best = 0, [], {}
        if args.resume:
            state = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            if state['identity'] != identity:
                raise ValueError('Resume identity mismatch')
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            h.restore_rng(state['rng'])
            generator.set_state(state['noise_rng'])
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            selected.validate_selected(args.output, best)
        h.write_json(args.output/'config.json', identity)
        h.write_json(summary_path, summary)
        if not args.resume:
            torch.save(masks, args.output/'fixed_masks.pt')
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start, 3 if args.smoke else 150):
            for group in optimizer.param_groups:
                group['lr'] = 3e-4 if epoch < 100 else 3e-4/10
            row = train_epoch(bundle, model, optimizer, generator, epoch, args.smoke)
            evaluated = {r: evaluate(bundle, model, masks['test'][r], r, args.smoke) for r in RATES}
            row['test'] = {r: values[0] for r, values in evaluated.items()}
            history.append(row)
            improved = []
            for rate, (metrics, arrays) in evaluated.items():
                if rate in best and metrics['router']['WF1'] <= best[rate]['metrics']['router']['WF1']:
                    continue
                # Immutable versions keep the previous last.pt ledger recoverable
                # if a process stops between selected artifacts and last.pt.
                stem = 'miss_'+rate.replace('.', 'p')+f'_epoch_{epoch+1}'
                predpath, ckptpath = args.output/('predictions_'+stem+'.npz'), args.output/('best_'+stem+'.pt')
                selected._atomic_arrays(predpath, arrays)
                h.atomic_torch_save(ckptpath, dict(model=model.state_dict(), identity=identity,
                    config=asdict(cfg), epoch=epoch+1, rate=rate))
                best[rate] = dict(epoch=epoch+1, metrics=metrics, checkpoint=ckptpath.name,
                    predictions=predpath.name, checkpoint_sha256=h.sha(ckptpath), prediction_sha256=h.sha(predpath))
                improved.append(rate)
            h.atomic_torch_save(args.output/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                identity=identity, epoch=epoch+1, history=history, best_by_rate=best,
                rng=h.rng_state(), noise_rng=generator.get_state()))
            h.write_json(args.output/'history.json', history)
            summary.update(epoch=epoch+1, best_by_rate=best, aggregate=aggregate(best),
                seconds=time.monotonic()-tick, peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            h.write_json(summary_path, summary)
            print(json.dumps(dict(mode=args.mode, epoch=epoch+1, task_MSE=row['task_MSE'],
                aux_loss=row['aux_loss'], mean=summary['aggregate']['router']['mean'],
                seconds=summary['seconds'], improved_rates=improved)), flush=True)
        selected.validate_selected(args.output, best)
        if any(h.sha(ROOT/p) != digest for p, digest in identity['source_hashes'].items()):
            raise ValueError('Source snapshot changed during training')
        if hasattr(model, 'mode') and not any(r['gradients'].get('block', r['gradients'].get('bridge', {})).get('nonzero')
                                              for r in history):
            raise RuntimeError('New branch never received nonzero gradients')
        summary.update(status='smoke_passed' if args.smoke else 'complete', seconds=time.monotonic()-tick)
        h.write_json(args.output/'RESULT.json', summary)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        h.write_json(summary_path, summary)


if __name__ == '__main__':
    main()
