"""Only the second-stage training-rate selector changes from the fixed screen."""
import hashlib
import json


def train_epoch(bundle, reader, block, optimizer, epoch, rate, smoke):
    import torch
    from gcnet_missing_m3.mixed_rate import BalancedBatchRateSchedule
    trainer, cfg, loader = bundle['trainer'], bundle['cfg'], bundle['train_loader']
    device = next(block.parameters()).device
    loader.sampler.set_epoch(epoch)
    schedules = trainer._schedules(cfg, 'train')
    cycle = BalancedBatchRateSchedule(rates=cfg.train_missing_rates)
    block.train()
    reader.parent.eval()
    total,count,steps,coverage,seen = 0.,0,0,[],[]
    trace = hashlib.sha256()
    for index, raw in enumerate(loader):
        if smoke and index >= 1:
            break
        actual_rate = cycle.rate_for(epoch,index)
        seen.append(float(actual_rate))
        data = trainer._move_batch(raw,device)
        view = trainer._prepare_view(data,schedules[actual_rate],epoch,bundle['dimensions'])
        frame = reader(view['incomplete'],view['availability'],view['qmask'],view['umask'],view['lengths'])
        optimizer.zero_grad(set_to_none=True)
        prediction = reader.decode_delta(frame,block(frame))
        loss = trainer._task_loss(cfg.dataset,prediction,view['labels'],view['umask'],
            cfg.mosi_task_mode,cfg.task_regression_loss,cfg.task_smooth_l1_beta)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite cyclic stage2 task MSE')
        loss.backward()
        grads = [p.grad for p in block.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Disconnected/nonfinite cyclic stage2 gradients')
        coverage.append(dict(connected=len(grads),nonzero=sum(int(g.ne(0).any()) for g in grads)))
        norm = torch.nn.utils.clip_grad_norm_(block.parameters(),cfg.gradient_clip_norm)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient clipping norm')
        optimizer.step()
        n = int(view['umask'].sum())
        total += float(loss.detach())*n
        count += n
        steps += 1
        trace.update(json.dumps(list(map(str,view['conversation_ids']))).encode())
        trace.update(view['availability'].detach().cpu().numpy().tobytes())
    if not count:
        raise RuntimeError('Empty cyclic stage2 training epoch')
    return dict(epoch=epoch+1,training_rates=seen,evaluation_rate=str(rate),task_MSE=total/count,N=count,
        optimizer_steps=steps,mask_trace_sha256=trace.hexdigest(),gradients=coverage[-1],
        learning_rate=optimizer.param_groups[0]['lr'])
