"""Evaluate existing rho.10 task-only on six whole-conversation patterns."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
PATTERNS={'A':(1,0,0),'T':(0,1,0),'V':(0,0,1),
          'AT':(1,1,0),'AV':(1,0,1),'TV':(0,1,1)}


def checkpoint_rate(bits):
    assert sum(bits) in (1,2)
    return .7 if sum(bits)==1 else .3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runner-commit',required=True)
    a=p.parse_args()
    assert not a.output.exists(), 'Do not overwrite old results'
    from experiments.osram_paired_history_views_20261001.run import read,write,sha
    from experiments.osram_history_drift_20261002.run import state_hash
    prior=read(a.source/'PROVENANCE.json')
    assert prior['status']=='complete' and prior['code_commit']=='6e2a998'
    assert {k:sha(ROOT/k) for k in prior['source_sha256']}==prior['source_sha256']
    uuid='GPU-e4cafb17-818e-216a-b94a-7440063a9153'
    assert subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=uuid',
        '--format=csv,noheader'],text=True).strip()==uuid==os.environ['CUDA_VISIBLE_DEVICES']
    import torch
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_missing_m3.train_gcnet import TrainConfig,_attach_history_projector
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model,_evaluate_pattern,_roots
    torch.set_num_threads(2)
    os.environ['GCNET_DATASET_ROOT']='/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset'
    config=TrainConfig(**read(a.source/'config.json'))
    assert config.seed==66 and config.paired_history_views
    assert config.history_task_view2_weight==.1 and config.history_contrast_weight==0
    _,_,test,*dims=get_loaders(audio_root=_roots()[0],text_root=_roots()[1],video_root=_roots()[2],
        num_folder=1,dataset='CMUMOSI',batch_size=config.batch_size,num_workers=0,seed=66,
        validation_fraction=config.validation_fraction,evaluation_protocol=config.evaluation_protocol)
    model=_build_model(config,tuple(dims)).cuda()
    _attach_history_projector(model,config)
    model.eval().requires_grad_(False)
    calls=[]
    handle=model.history_contrast_projector.register_forward_pre_hook(lambda *_:calls.append(1))
    report=dict(status='running',label='INTERNAL TEST-ORACLE DIAGNOSTIC ONLY',seed=66,
        rho=.1,contrast_weight=0,training=False,source=str(a.source),
        backbone_commit='6e2a998',runner_commit=a.runner_commit,runner_sha256=sha(Path(__file__)),
        server='biggpu',physical_gpu=6,gpu_uuid=uuid,pid=os.getpid(),records=[])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write(a.output,report)
    try:
        for pattern,bits in PATTERNS.items():
            rate=checkpoint_rate(bits)
            checkpoint=a.source/f'best_miss_0p{int(rate*10)}.pt'
            digest=sha(checkpoint)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model'],strict=True)
            before=state_hash(model)
            test[0].sampler.set_epoch(0)
            metrics=_evaluate_pattern(model,test[0],bits,tuple(dims),torch.device('cuda:0'))
            assert metrics['sample_count']==686 and metrics['nonzero_sample_count']==656
            assert before==state_hash(model) and digest==sha(checkpoint) and not calls
            report['records'].append(dict(pattern=pattern,checkpoint=str(checkpoint),
                checkpoint_rate=rate,checkpoint_epoch=int(saved['epoch']),checkpoint_sha256=digest,
                state_sha256=before,frozen_hash_check=True,projector_calls=0,**metrics))
            write(a.output,report)
            print(pattern,metrics['weighted_f1'],flush=True)
        report['status']='complete'
        write(a.output,report)
    finally:
        handle.remove()


if __name__=='__main__':
    main()
