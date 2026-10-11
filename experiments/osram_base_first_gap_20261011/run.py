"""One approved seed66 run, reusing the existing full-state100+50 protocol."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import platform
import sys

from experiments.osram_complete150_20261011 import run as continuation
from experiments.osram_complete150_20261011.run import read, write, sha, now

GPU = 'GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049'
REFERENCE = Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66')
DATA = Path('/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json')


def candidate_config(reference):
    expected = dict(dataset='CMUMOSI', seed=66, epochs=100, learning_rate=.001,
        lr_schedule='constant', backbone_type='osram', training_objective='emotion-only',
        checkpoint_selection='test-oracle-per-rate', train_rate_mode='cyclic',
        osram_readout_fusion='flat', osram_output_dim=1600, osram_num_heads=8,
        osram_key_dim=64, osram_value_dim=64, osram_write_step=.6, osram_bidirectional=False)
    for key,value in expected.items():
        if reference.get(key)!=value: raise ValueError(f'Unexpected cfg84 {key}')
    for key in ('paired_history_views','osram_relation_block','osram_post_grn',
                'classification_completion','osram_base_first_gap','osram_decision_correction',
                'osram_memory_only_adapter','osram_local_evidence_gate'):
        if reference.get(key,False): raise ValueError(f'Unexpected enabled {key}')
    if reference.get('osram_meaningful_block','none')!='none':
        raise ValueError('Reference must not contain Nested or another meaningful block')
    return dict(reference,osram_base_first_gap=True)


def prepare(root,commit):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig
    assert not (root/'MANIFEST.json').exists(), 'Existing experiment, do not overwrite'
    source = Path(__file__).resolve().parents[2]
    names = ['config.py']
    for directory in ('gcnet_missing_m3','gcnet_modality_jepa','gcnet','gcnet_jepa_replacement',
                      'gcnet_plci_jepa','experiments/osram_complete150_20261011',
                      'experiments/osram_base_first_gap_20261011'):
        names.extend(str(p.relative_to(source)) for p in (source/directory).rglob('*.py'))
    hashes = {name:sha(source/name) for name in sorted(set(names))}
    base = asdict(TrainConfig(**read(REFERENCE/'config.json')))
    cfg = asdict(TrainConfig(**candidate_config(base)))
    delta = {k:dict(baseline=base[k],candidate=v) for k,v in cfg.items() if v!=base[k]}
    assert set(delta)=={'osram_base_first_gap'}
    data = read(DATA); verified = {}
    for name,digest in data['files'].items():
        assert sha(name)==digest,name
        st=Path(name).stat(); verified[name]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,sha256=digest)
    task = dict(id='base_first_gap_seed66',dataset='CMUMOSI',model='base_first_gap',seed=66,fold=1,
        reference=str(REFERENCE),original=None,source=str(source),config_path=str(REFERENCE/'config.json'),
        config_sha256=sha(REFERENCE/'config.json'),config=cfg,configuration_delta=delta,
        mode='fresh100-then-full-state50')
    write(root/'MANIFEST.json',dict(label=continuation.LABEL,runner_commit=commit,
        runner_sha256=sha(continuation.__file__),tasks=[task],data={'CMUMOSI':data},
        data_verified=verified,sources={str(source):dict(commit=commit,hashes=hashes)},prepared_at=now(),
        environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,host=platform.node())))


def summarize(root):
    run=root/'runs/base_first_gap_seed66'
    provenance=read(run/'PROVENANCE.json')
    assert provenance['status']=='complete' and provenance['outputs_verified']
    metrics=read(run/'stage150/metrics.json')
    scores={k:100*v['weighted_f1'] for k,v in metrics['test'].items()}
    result=dict(label=continuation.LABEL,status='complete',seed=66,epochs=150,
        schedule='100@0.001+50@0.0001',per_rate_wf1=scores,
        mean8=sum(scores.values())/8,high=sum(scores[str(i/10)] for i in (5,6,7))/3,
        selected_epoch_by_rate=metrics.get('selected_epoch_by_rate'),
        reference_flat150_mean8=81.068095,reference_flat150_high=76.352251)
    write(root/'SUMMARY.json',result)
    print(result,flush=True)


def smoke(root):
    import torch
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_missing_m3.train_mc_fwc import make_loaders, build_model
    from gcnet_missing_m3.mc_fwc import forward_view, _task_loss
    from types import SimpleNamespace
    cfg=trainer.TrainConfig(**candidate_config(read(REFERENCE/'config.json')))
    trainer.set_random_seed(cfg.seed); torch.set_num_threads(1)
    data=read(DATA)
    train,_,_,dims,_=make_loaders(SimpleNamespace(dataset_root=data['dataset_root']),cfg)
    trainer.set_random_seed(trainer.SeedBundle(cfg.seed).derive('missing_m3_model_init:fold:5'))
    model=build_model(cfg,dims).train()
    groups,_=trainer._optimizer_parameter_groups(model,cfg)
    optimizer=torch.optim.Adam(groups,weight_decay=cfg.weight_decay)
    batch=trainer._move_batch(next(iter(train)),torch.device(cfg.device))
    view=trainer._prepare_view(batch,trainer._schedules(cfg,'train')[.5],0,dims)
    losses=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        output=forward_view(model,view)
        logits=output[0] if isinstance(output,tuple) else output
        loss=_task_loss(logits,view,cfg.dataset)
        assert torch.isfinite(loss)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step(); losses.append(dict(step=step+1,loss=float(loss),preclip_norm=float(norm)))
    write(root/'SMOKE.json',dict(label='ENGINEERING ONLY',steps=losses,
        parameters=sum(p.numel() for p in model.parameters()),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),valid_utterances=int(view['umask'].sum())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit');p.add_argument('--prepare',action='store_true')
    p.add_argument('--train',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if sum((a.prepare,a.train,a.smoke))!=1:p.error('Choose exactly one operation')
    if a.prepare:
        if not a.commit:p.error('--commit required')
        prepare(a.root,a.commit)
    else:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU,'Use verified healthy hostGPU1 UUID'
        if a.smoke:smoke(a.root)
        else:continuation.train(a.root,0,GPU);summarize(a.root)


if __name__=='__main__':main()
