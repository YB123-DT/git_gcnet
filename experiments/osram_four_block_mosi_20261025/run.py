"""Four independent frozen-parent MOSI experiments; finite, resumable runs."""
import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import time


def prediction(mode, reader, block, frame, view, rate_index, extras):
    from osram_four_block_wiring import wiring
    if mode == 'ridge':
        return wiring.ridge_prediction(block, frame, extras['bank'], view['conversation_ids'],
                                      rate_index, extras['parent_id'])
    if mode == 'csdi':
        return wiring.csdi_prediction(reader, extras['denoiser'], block, extras['projection'],
            frame, generator=extras['generator'], samples=extras.get('samples'))
    return reader.decode_delta(frame, block(frame))


def metrics(trainer, labels, predictions, base):
    import numpy as np
    m = trainer._metrics('CMUMOSI', labels, predictions, 'regression')
    valid = labels != 0
    old, new, truth = base[valid] > 0, predictions[valid] > 0, labels[valid] > 0
    return dict(ACC=100*m['accuracy'], WF1=100*m['weighted_f1'],
        MSE=float(np.mean((labels-predictions)**2)),
        corrections=int(((old != truth) & (new == truth)).sum()),
        harms=int(((old == truth) & (new != truth)).sum()), N=int(valid.sum()))


def aggregate(evaluation):
    from .prepare import RATES
    return {scope: {key: sum(evaluation[r][key] for r in rates)/len(rates)
                    for key in ('ACC','WF1','MSE')}
            for scope, rates in (('mean',RATES),('high',RATES[5:]))}


def evaluate(bundle, reader, block, cache, mode, extras, epoch, output, best):
    import numpy as np
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    device = next(block.parameters()).device
    block.eval()
    scores = {}
    with torch.no_grad(), h.isolated_rng(660029):
        for rate, rows in cache.items():
            pieces = []
            for row in rows:
                frame = row['frame'].to(device)
                view = dict(conversation_ids=row['conversation_ids'])
                aux = dict(extras)
                if mode == 'csdi':
                    aux['samples'] = row['samples'].to(device)
                pred = prediction(mode, reader, block, frame, view, round(float(rate)*10), aux)
                valid = frame.valid.T
                pieces.append(dict(row['metadata'],
                    availability=row['availability'],
                    prediction=pred.transpose(0,1)[valid].cpu().numpy().reshape(-1),
                    base_prediction=frame.prediction.transpose(0,1)[valid].cpu().numpy().reshape(-1)))
            arrays = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
            if not np.isfinite(arrays['prediction']).all():
                raise ValueError('Nonfinite evaluation prediction')
            score = metrics(bundle['trainer'], arrays['labels'], arrays['prediction'], arrays['base_prediction'])
            scores[rate] = score
            if epoch >= 0 and (rate not in best or score['WF1'] > best[rate]['metrics']['WF1']):
                tag = rate.replace('.', 'p')
                checkpoint, predictions = f'best_miss_{tag}.pt', f'predictions_miss_{tag}.npz'
                torch.save(dict(block=block.state_dict(), epoch=epoch, mode=mode,
                    parent_id=extras['parent_id'], protocol='INTERNAL TEST-ORACLE SCREEN'), output/checkpoint)
                np.savez_compressed(output/predictions, **arrays)
                best[rate] = dict(epoch=epoch, metrics=score, checkpoint=checkpoint,
                    checkpoint_sha256=h.sha(output/checkpoint), predictions=predictions,
                    prediction_sha256=h.sha(output/predictions),
                    mask_sha256=hashlib.sha256(arrays['availability'].tobytes()).hexdigest())
    return scores


def cache_hypotheses(reader, denoiser, projection, cache, device, parent_id):
    import torch
    # One deterministic sample bank, reused by every bridge epoch and comparison.
    for rate, rows in cache.items():
        for row in rows:
            frame = row['frame'].to(device)
            keys = [[f'{parent_id}:{rate}:{cid}:{t}' for cid in row['conversation_ids']]
                    for t in range(frame.valid.shape[0])]
            row['samples'] = denoiser.sample(projection(frame.latents,frame.availability),
                frame.availability, frame.context, frame.valid, query_keys=keys).cpu()


def save_last(output, block, optimizer, epoch, history, best, identity, extras):
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    state = dict(block=block.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
        history=history, best=best, identity=identity, rng=h.rng_state(),
        generator=extras['generator'].get_state())
    temporary = output/'last.tmp'
    torch.save(state, temporary)
    temporary.replace(output/'last.pt')


def write_report(output, summary):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    rows = ['# Four independent MOSI frozen-parent experiments', '',
        'INTERNAL TEST-ORACLE SCREEN', '',
        f"Method: {summary['mode']}; status: {summary['status']}; task epoch: {summary.get('epoch',0)}/150.",
        '', 'One shared miss0.3 Flat checkpoint, frozen in eval mode. This is not from-scratch training.',
        'Eight-rate Test-best may combine different epochs. All rows use identical fixed legal test masks.', '']
    if 'best_by_rate' in summary:
        scores = {r: v['metrics'] for r,v in summary['best_by_rate'].items()}
        rows += ['| Rate | Frozen Flat ACC | New ACC | Frozen W-F1 | New W-F1 | MSE | Epoch |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for rate, best in summary['best_by_rate'].items():
            old, new = summary['baseline'][rate], best['metrics']
            rows.append(f"| {rate} | {old['ACC']:.4f} | {new['ACC']:.4f} | {old['WF1']:.4f} | {new['WF1']:.4f} | {new['MSE']:.4f} | {best['epoch']} |")
        if len(scores) == 8:
            summary['aggregate'] = aggregate(scores)
            summary['baseline_aggregate'] = aggregate(summary['baseline'])
            rows += ['', 'Aggregate: '+json.dumps(summary['aggregate']),
                     'Frozen parent aggregate: '+json.dumps(summary['baseline_aggregate'])]
    rows += ['', 'No training/test labels enter missing inputs. TRAIN labels enter only loss/support residuals.',
        'CSDI TRAIN complete features supervise missing-coordinate denoising only, never CHM write.',
        'All checkpoints and exact predictions/masks remain in the remote output.', '']
    (output/'RESULT.md').write_text('\n'.join(rows))
    h.write_json(output/'SUMMARY.json', summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('neumiss','signature','ridge','csdi'), required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import fcntl
    import numpy as np
    import torch
    from osram_four_block_wiring import NeuMissBlock, SignatureBlock, RidgeResidualBlock, LatentCSDI, HypothesisBridge
    from osram_four_block_wiring import wiring
    from gcnet_missing_m3.mixed_rate import BalancedBatchRateSchedule
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from . import prepare
    args.output.mkdir(parents=True,exist_ok=args.resume)
    lock = (args.output/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.resume and (args.output/'SUMMARY.json').exists():
        raise FileExistsError('Existing run cannot be overwritten')
    torch.set_num_threads(1)
    tick = time.monotonic()
    summary = dict(status='preflight',mode=args.mode,source_commit=args.source_commit,pid=os.getpid(),epoch=0)
    h.write_json(args.output/'SUMMARY.json',summary)
    try:
        bundle, reader, identity = prepare.setup(args.gpu_uuid,args.output)
        device = next(reader.parent.parameters()).device
        before = h.backbone_digest(reader.parent)
        cache = prepare.build_test_cache(bundle,reader)
        with h.isolated_rng(66):
            constructors = dict(neumiss=NeuMissBlock,signature=SignatureBlock,ridge=RidgeResidualBlock,csdi=HypothesisBridge)
            block = constructors[args.mode]().to(device)
        generator = torch.Generator(device=device).manual_seed(6601)
        extras = dict(parent_id=identity['parent_id'],generator=generator)
        if args.mode == 'ridge':
            bank, manifest = prepare.build_support_bank(bundle,reader,identity['parent_id'])
            torch.save(bank,args.output/'train_support.pt')
            for field in fields(bank):
                value = getattr(bank,field.name)
                if isinstance(value,torch.Tensor):
                    setattr(bank,field.name,value.to(device))
            extras['bank'] = bank
            h.write_json(args.output/'support_manifest.json',manifest)
        if args.mode == 'csdi':
            projection, manifest = prepare.fit_projection(bundle,reader)
            extras.update(projection=projection.to(device))
            h.write_json(args.output/'pca_manifest.json',manifest)
            with h.isolated_rng(66):
                denoiser = LatentCSDI().to(device)
            extras['denoiser'] = denoiser
            noise_optimizer = torch.optim.Adam(denoiser.parameters(),lr=3e-4,weight_decay=1e-5)
            stage_path = args.output/'denoiser_last.pt'
            start_noise, noise_history = 0, []
            if args.resume and stage_path.exists():
                saved = torch.load(stage_path,map_location=device)
                assert saved['parent_id']==identity['parent_id']
                denoiser.load_state_dict(saved['model'],strict=True)
                noise_optimizer.load_state_dict(saved['optimizer'])
                start_noise, noise_history = saved['epoch'],saved['history']
                h.restore_rng(saved['rng']);generator.set_state(saved['generator'].cpu())
            else:
                reset_training_stochasticity(66,bundle['cfg'].fold)
            for epoch in range(start_noise,3 if args.smoke else 150):
                loss_row, mask_hash = train_epoch(bundle,reader,denoiser,noise_optimizer,epoch,args.mode,extras,
                                                 noise=True,max_batches=1 if args.smoke else None)
                noise_history.append(dict(epoch=epoch+1,noise_loss=loss_row,mask_hash=mask_hash))
                torch.save(dict(parent_id=identity['parent_id'],model=denoiser.state_dict(),
                    optimizer=noise_optimizer.state_dict(),epoch=epoch+1,history=noise_history,
                    rng=h.rng_state(),generator=generator.get_state(),projection=projection.state_dict()),stage_path)
                summary.update(status='denoising',denoising_epoch=epoch+1,noise_loss=loss_row)
                h.write_json(args.output/'SUMMARY.json',summary)
                print(f'csdi denoise {epoch+1}: {loss_row:.6f}',flush=True)
            denoiser.eval().requires_grad_(False)
            cache_hypotheses(reader,denoiser,projection,cache,device,identity['parent_id'])
            torch.save({rate:[row['samples'] for row in rows] for rate,rows in cache.items()},args.output/'test_hypotheses.pt')
        # Compute baseline from cached original parent outputs, not a separately selected baseline.
        baseline = {}
        for rate,rows in cache.items():
            labels = np.concatenate([row['metadata']['labels'] for row in rows])
            old = np.concatenate([row['frame'].prediction.transpose(0,1)[row['umask'].bool()].numpy().reshape(-1) for row in rows])
            baseline[rate] = metrics(bundle['trainer'],labels,old,old)
        optimizer = torch.optim.Adam(block.parameters(),lr=3e-4,weight_decay=1e-5)
        best, history, start_epoch = {}, [], 0
        identity.update(method=args.mode,task_epochs=150,denoise_epochs=150 if args.mode=='csdi' else 0,
            lr_first100=3e-4,lr_last50=3e-5,weight_decay=1e-5,seed=66,batch_size=32,
            source_commit=args.source_commit,mask_protocol='gcnet-random-mask-v1',
            selection='INTERNAL TEST-ORACLE SCREEN; per-rate WF1; same exact test masks',
            module_parameters=sum(p.numel() for p in block.parameters()),
            extra_denoiser_parameters=sum(p.numel() for p in extras['denoiser'].parameters()) if args.mode=='csdi' else 0)
        summary.update(identity=identity,baseline=baseline,checks=dict(frozen_parent=True,
            zero_delta_decode_exact=True,exact_parent_mask_replay=True))
        initial = evaluate(bundle,reader,block,cache,args.mode,extras,-1,args.output,{})
        summary['initial_metrics'] = initial
        if args.mode != 'ridge':
            assert initial == baseline, 'Zero-exit block must match frozen baseline exactly'
        reset_training_stochasticity(66,bundle['cfg'].fold)
        generator.manual_seed(6602)
        if args.resume and (args.output/'last.pt').exists():
            saved = torch.load(args.output/'last.pt',map_location=device)
            assert saved['identity'] == identity
            block.load_state_dict(saved['block'],strict=True);optimizer.load_state_dict(saved['optimizer'])
            start_epoch,history,best = saved['epoch'],saved['history'],saved['best']
            h.restore_rng(saved['rng']);generator.set_state(saved['generator'].cpu())
        for epoch in range(start_epoch,3 if args.smoke else 150):
            loss_row, mask_hash = train_epoch(bundle,reader,block,optimizer,epoch,args.mode,extras,
                                             max_batches=1 if args.smoke else None)
            scores = evaluate(bundle,reader,block,cache,args.mode,extras,epoch+1,args.output,best)
            history.append(dict(epoch=epoch+1,train_MSE=loss_row,mask_hash=mask_hash,
                learning_rate=optimizer.param_groups[0]['lr'],evaluation=scores,aggregate=aggregate(scores)))
            save_last(args.output,block,optimizer,epoch+1,history,best,identity,extras)
            summary.update(status='running',epoch=epoch+1,best_by_rate=best,
                branch_gradient_coverage=extras.get('gradient_coverage'),
                seconds=time.monotonic()-tick,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
            h.write_json(args.output/'history.json',history)
            write_report(args.output,summary)
            print(f'{args.mode} epoch{epoch+1} MSE{loss_row:.6f} meanWF1{aggregate(scores)["mean"]["WF1"]:.4f}',flush=True)
        assert before == h.backbone_digest(reader.parent)
        assert all(p.grad is None for p in reader.parent.parameters())
        # Strict branch checkpoint recovery on the last state.
        saved = torch.load(args.output/'last.pt',map_location=device)
        assert all(torch.equal(v,saved['block'][k]) for k,v in block.state_dict().items())
        summary.update(status='smoke_passed' if args.smoke else 'complete',seconds=time.monotonic()-tick,
            checks=dict(summary['checks'],parent_unchanged=True,parent_grad_none=True,last_state_exact=True))
        write_report(args.output,summary)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        h.write_json(args.output/'SUMMARY.json',summary)
        raise


def train_epoch(bundle,reader,block,optimizer,epoch,mode,extras,noise=False,max_batches=None):
    import torch
    from gcnet_missing_m3.mixed_rate import BalancedBatchRateSchedule
    from osram_four_block_wiring import wiring
    from .prepare import _read
    trainer,cfg,loader = bundle['trainer'],bundle['cfg'],bundle['train_loader']
    device = next(block.parameters()).device
    loader.sampler.set_epoch(epoch)
    schedules = trainer._schedules(cfg,'train')
    cycle = BalancedBatchRateSchedule(rates=cfg.train_missing_rates)
    block.train();reader.parent.eval()
    for group in optimizer.param_groups:
        group['lr'] = 3e-4 if epoch < 100 else 3e-5
    total,n_total = 0.,0
    digest = hashlib.sha256()
    for bi,raw in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        data = trainer._move_batch(raw,device)
        rate = cycle.rate_for(epoch,bi)
        view = trainer._prepare_view(data,schedules[rate],epoch,bundle['dimensions'])
        frame = _read(reader,view)
        optimizer.zero_grad(set_to_none=True)
        if noise:
            loss = wiring.csdi_noise_loss(reader,block,extras['projection'],frame,
                view['complete'],extras['generator'],split='train')
        else:
            pred = prediction(mode,reader,block,frame,view,round(rate*10),extras)
            loss = trainer._task_loss(cfg.dataset,pred,view['labels'],view['umask'],
                cfg.mosi_task_mode,cfg.task_regression_loss,cfg.task_smooth_l1_beta)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        gradients=[p.grad for p in block.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Disconnected/nonfinite trainable branch')
        extras['gradient_coverage'] = dict(connected_tensors=len(gradients),
            nonzero_tensors=sum(bool(g.ne(0).any()) for g in gradients))
        norm = torch.nn.utils.clip_grad_norm_(block.parameters(),cfg.gradient_clip_norm)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite clipped gradient')
        optimizer.step()
        n=int(view['umask'].sum());total+=float(loss.detach())*n;n_total+=n
        digest.update(json.dumps(view['conversation_ids']).encode());digest.update(view['availability'].cpu().numpy().tobytes())
    if not n_total:
        raise RuntimeError('Empty training loader')
    return total/n_total,digest.hexdigest()


if __name__ == '__main__':
    main()
