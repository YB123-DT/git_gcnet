"""Bounded IEMOCAP6 fold5 compatibility check before the fivefold queue."""
import argparse
import json
import math
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    target = args.output/'PREFLIGHT.json'
    if target.exists():
        raise FileExistsError(target)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import numpy as np
    import torch
    from . import bundle, iemocap as c, runtime as r
    from .stage2 import restore_parent
    from .train import checkpoint_payload,restore
    torch.set_num_threads(1)
    source = bundle.verify(Path(__file__).resolve().parents[2])
    report = dict(status='running',source_commit=source['source_commit'],smoke_only=True)
    tick = time.monotonic()
    try:
        print('Checking original IEMOCAP6 fold5 config, split, masks and legacy initialization...',flush=True)
        b = c.setup('A3',5,args.output/'preflight',args.gpu_uuid)
        reference = b['selected']['0.3']
        model = b['model']
        parent = restore_parent(model,reference['checkpoint'],expected_sha256=reference['checkpoint_sha256'])
        metrics,arrays = c.evaluate(b,model,'0.3')
        previous = b['selected_reference']['0.3']
        for key in ('labels','speakers','conversation_ids','utterance_indices','availability','logits'):
            np.testing.assert_array_equal(arrays[key],previous[key],err_msg='Original IEMOCAP parity: '+key)
        np.testing.assert_array_equal(arrays['prediction'],previous['predictions'])
        report.update(parent_replay=dict(parent=parent,arrays_bitwise_equal=True,metrics=metrics))
        print('Original complete fold5 test prediction parity passed; checking fresh CE update...',flush=True)
        model = c.build_model(b['cfg'],b['dimensions'],torch.device('cuda:0'))
        b['model'] = model
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        reset_training_stochasticity(66,5)
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
        torch.cuda.reset_peak_memory_stats()
        smoke = r.train_epoch(b,model,optimizer,0,max_batches=3)
        smoke['task_CE'] = smoke.pop('task_MSE')
        if any(not smoke['gradients'].get(group,{}).get('nonzero') for group in
               ('observed_set.projectors','osram.key_projectors','osram.value_projectors','osram.query_projector')):
            raise RuntimeError('Expected CE writer/query path did not receive nonzero startup gradients')
        loader = b['train_loader']
        indices = sorted(loader.sampler.indices,key=lambda i: len(
            loader.dataset.videoLabelsNew[loader.dataset.vids[i]]),reverse=True)[:32]
        longest = loader.collate_fn([loader.dataset[i] for i in indices])
        data = b['trainer']._move_batch(longest,torch.device('cuda:0'))
        schedule = b['trainer']._schedules(b['cfg'],'train')[.7]
        view = b['trainer']._prepare_view(data,schedule,0,b['dimensions'])
        optimizer.zero_grad(set_to_none=True)
        prediction = r.forward(model,view)
        loss = b['trainer']._task_loss('IEMOCAPSix',prediction,view['labels'],view['umask'])
        loss.backward()
        if not torch.isfinite(loss) or any(not torch.isfinite(x.grad).all()
                for x in model.parameters() if x.grad is not None):
            raise RuntimeError('Nonfinite longest IEMOCAP batch loss/gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),b['cfg'].gradient_clip_norm)
        optimizer.step()
        torch.cuda.synchronize()
        identity = json.loads(json.dumps(b['identity']))
        r.atomic_save(args.output/'smoke_full_resume.pt',checkpoint_payload(model,optimizer,
            epoch=1,identity=identity,history=[smoke],best={}))
        digest = r.state_digest(model)
        saved = torch.load(args.output/'smoke_full_resume.pt',map_location='cpu',weights_only=False)
        restore(model,optimizer,saved,identity)
        if r.state_digest(model) != digest or len(r.rng_state()['cuda']) != 1:
            raise RuntimeError('Incomplete model or UUID RNG state restoration')
        memory = torch.cuda.max_memory_reserved()/2**20
        report.update(status='passed',source_commit=source['source_commit'],environment=b['environment'],
            gpu=b['gpu'],identity=b['identity'],smoke=smoke,optimizer_rng_state_restore_exercised=True,
            longest_batch=dict(batch=len(indices),padded_length=int(longest[7].shape[1]),rate=.7,CE=float(loss.detach())),
            original_parameters=sum(x.numel() for x in model.parameters()),
            peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_memory_mib=memory,
            job_memory_budget_mib=int(math.ceil((memory*1.25+1024)/256)*256))
        print(json.dumps({k:report[k] for k in ('status','peak_memory_mib','peak_reserved_memory_mib',
                                               'job_memory_budget_mib')},indent=2),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        report['seconds'] = time.monotonic()-tick
        r.save_json(target,report)


if __name__ == '__main__':
    main()
