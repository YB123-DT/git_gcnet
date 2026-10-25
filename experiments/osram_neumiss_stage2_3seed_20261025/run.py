"""Complete only MOSI NeuMiss fixed-rate stage2 seeds67/68; reuse seed66."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import resource
import time


def validate_seed(seed):
    if type(seed) is not int or seed not in (67,68):
        raise ValueError('Only missing NeuMiss seeds67/68 may be added; reuse completed66')


def epoch_budget(rate,smoke=False):
    from experiments.osram_neumiss_fixed_rate_stage2_20261025.prepare import validate_rate
    return 0 if validate_rate(rate)=='0.0' else (2 if smoke else 150)


def learning_rate(epoch):
    if not 0 <= epoch < 150:
        raise ValueError('Outside the original finite MOSI150 budget')
    return 3e-4 if epoch < 100 else 3e-4/10


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--rate',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    validate_seed(args.seed)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    from experiments.osram_three_block_joint_mosi_20261025.scheduler import gpu_inventory,guard_gpu
    host=gpu_inventory().get(args.gpu_uuid,{}).get('host_index')
    if host not in (0,1,2):
        raise ValueError('Only healthy physical GPU0/1/2 authorized for this completion')
    guard_gpu(args.gpu_uuid,host)
    import fcntl
    import numpy as np
    import torch
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import _atomic_arrays
    from experiments.osram_neumiss_fixed_rate_stage2_20261025.run import evaluation,train_epoch
    from experiments.osram_signature_stage2_extension_20261025 import prepare
    from osram_four_block_wiring import NeuMissBlock
    rate=prepare.validate_rate(args.rate)
    torch.set_num_threads(1)
    repair=repair_uuid_rng(torch)
    if (args.output/'SUMMARY.json').exists():
        previous=json.loads((args.output/'SUMMARY.json').read_text())
        if not args.resume or previous['status']=='complete':
            raise FileExistsError('Explicit resume required and completed run cannot be overwritten')
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'RUN.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tick,path=time.monotonic(),args.output/'SUMMARY.json'
    summary=dict(status='preflight',source_commit=args.source_commit,mode='neumiss',dataset='CMUMOSI',
        rate=rate,seed=args.seed,pid=os.getpid(),label='INTERNAL TEST-ORACLE SCREEN')
    try:
        bundle,reader,parent=prepare.setup('CMUMOSI',args.seed,args.gpu_uuid,args.output,rate)
        bundle['cfg']=replace(bundle['cfg'],train_rate_mode='fixed',fixed_missing_rate=float(rate),
            train_missing_rates=(float(rate),))
        rows=prepare.test_cache(bundle,reader,rate)
        with h.isolated_rng(args.seed):
            block=NeuMissBlock().to('cuda:0')
        before=h.backbone_digest(reader.parent)
        baseline,initial=evaluation(bundle,reader,block,rows)
        np.testing.assert_array_equal(initial['prediction'],initial['base_prediction'])
        root=Path(__file__).resolve().parents[2]
        files=list(Path(__file__).parent.glob('*.py'))+list((root/'osram_four_block_wiring').glob('*.py'))+[
            root/'experiments/osram_neumiss_fixed_rate_stage2_20261025/run.py',
            root/'experiments/osram_neumiss_fixed_rate_stage2_20261025/prepare.py',
            root/'experiments/osram_signature_stage2_extension_20261025/prepare.py']
        identity=dict(mode='neumiss',dataset='CMUMOSI',seed=args.seed,source_commit=args.source_commit,
            smoke=args.smoke,parent=parent,training_rate=rate,evaluation_rate=rate,cyclic_training=False,
            initial_block_sha256=h.backbone_digest(block),parent_config=parent['config'],
            effective_stage2_config=asdict(bundle['cfg']),mask_protocol='gcnet-random-mask-v1',
            source_hashes={str(p.relative_to(root)):h.sha(p) for p in files},rng_repair=repair,
            test_mask_sha256=hashlib.sha256(initial['availability'].tobytes()).hexdigest(),
            selection='TEST W-F1 at the one requested rate; earliest strict tie',
            task='original MSE including gold0',stage2_epochs=epoch_budget(rate))
        optimizer=torch.optim.Adam(block.parameters(),lr=3e-4,weight_decay=1e-5)
        start,history,best=0,[],None
        if args.resume:
            saved=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
            if saved['identity']!=identity:
                raise ValueError('Immutable resume identity changed')
            block.load_state_dict(saved['block'],strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            h.restore_rng(saved['rng'])
            start,history,best=saved['epoch'],saved['history'],saved['best']
            if best:
                for field,digest in [('checkpoint','checkpoint_sha256'),('predictions','prediction_sha256')]:
                    if field in best and h.sha(args.output/best[field])!=best[digest]:
                        raise ValueError('Saved best artifact checksum differs')
        else:
            reset_training_stochasticity(args.seed,bundle['cfg'].fold)
        summary.update(status='running',identity=identity,parent_id=parent['parent_id'],baseline=baseline,
            initial_real_batch_exact=True,environment=bundle['environment'],gpu=bundle['gpu'],
            total_parameters=sum(p.numel() for p in block.parameters()))
        h.write_json(args.output/'config.json',identity)
        h.write_json(path,summary)
        torch.cuda.reset_peak_memory_stats()
        epochs=epoch_budget(rate,args.smoke)
        for epoch in range(start,epochs):
            optimizer.param_groups[0]['lr']=learning_rate(epoch)
            row=train_epoch(bundle,reader,block,optimizer,epoch,rate,args.smoke)
            score,arrays=evaluation(bundle,reader,block,rows)
            if best is None or score['WF1']>best['metrics']['WF1']:
                ckpt,pred=f'best_epoch_{epoch+1}.pt',f'predictions_epoch_{epoch+1}.npz'
                _atomic_arrays(args.output/pred,arrays)
                h.atomic_torch_save(args.output/ckpt,dict(block=block.state_dict(),identity=identity,
                    epoch=epoch+1,parent_id=parent['parent_id']))
                best=dict(epoch=epoch+1,metrics=score,checkpoint=ckpt,predictions=pred,
                    checkpoint_sha256=h.sha(args.output/ckpt),prediction_sha256=h.sha(args.output/pred))
            row['test']=score
            history.append(row)
            h.atomic_torch_save(args.output/'last.pt',dict(block=block.state_dict(),optimizer=optimizer.state_dict(),
                identity=identity,epoch=epoch+1,history=history,best=best,rng=h.rng_state()))
            summary.update(epoch=epoch+1,best=best,seconds=time.monotonic()-tick,
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
            h.write_json(args.output/'history.json',history)
            h.write_json(path,summary)
            print(json.dumps(dict(seed=args.seed,rate=rate,epoch=epoch+1,current=score,
                best=best['metrics'],seconds=summary['seconds'])),flush=True)
        if epochs==0:
            _atomic_arrays(args.output/'predictions_noop.npz',initial)
            best=dict(epoch=0,metrics=baseline,predictions='predictions_noop.npz',
                prediction_sha256=h.sha(args.output/'predictions_noop.npz'),no_op=True)
            summary.update(epoch=0,best=best,no_op=True)
            h.write_json(args.output/'history.json',[])
            h.atomic_torch_save(args.output/'last.pt',dict(block=block.state_dict(),optimizer=optimizer.state_dict(),
                identity=identity,epoch=0,history=[],best=best,rng=h.rng_state()))
        else:
            saved=torch.load(args.output/best['checkpoint'],map_location='cpu',weights_only=False)
            block.load_state_dict(saved['block'],strict=True)
            score,replay=evaluation(bundle,reader,block,rows)
            with np.load(args.output/best['predictions']) as saved_arrays:
                np.testing.assert_array_equal(replay['prediction'],saved_arrays['prediction'])
            if score!=best['metrics']:
                raise ValueError('Selected checkpoint metrics differ after restoration')
            if not any(row['gradients']['nonzero'] for row in history):
                raise RuntimeError('NeuMiss never received nonzero task gradient')
        if before!=h.backbone_digest(reader.parent) or any(p.grad is not None for p in reader.parent.parameters()):
            raise RuntimeError('Frozen parent changed or received gradient')
        if any(h.sha(root/name)!=digest for name,digest in identity['source_hashes'].items()):
            raise RuntimeError('Immutable source changed')
        summary.update(status='smoke_passed' if args.smoke else 'complete',
            checks=dict(parent_unchanged=True,parent_grad_none=True,checkpoint_replay_exact=epochs>0,
                baseline_replay_exact=True,only_requested_train_rate=True,only_requested_test_rate=True),
            seconds=time.monotonic()-tick,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        h.write_json(args.output/'RESULT.json',summary)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        raise
    finally:
        h.write_json(path,summary)


if __name__=='__main__':
    main()
