"""Frozen Nested representation audit; train lightweight probes, never backbone."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch

from experiments.osram_nested_root_probe_20261011.capture import RootCollector
from experiments.osram_nested_root_probe_20261011.probe import run_probes
from experiments.osram_nested_diagnostics_20261007.evaluate import state_hash
from experiments.osram_nested_delta_off_missing_text_20261010.run import sha,dump
from experiments.osram_frozen_memory_audit_20261009.probe import metrics

GPU='GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e'

def extract(model,loader,schedule,dims,device):
    from gcnet_missing_m3.train_gcnet import _move_batch,_prepare_view
    pieces={k:[] for k in ('flat','rooted','predecoder','role_active','labels','availability',
        'conversation_ids','utterance_indices','prediction','flat_prediction')}
    loader.sampler.set_epoch(0)
    for raw in loader:
        view=_prepare_view(_move_batch(raw,device),schedule,0,dims)
        with torch.no_grad():
            with RootCollector(model.osram.meaningful_block) as capture:
                output=model([view['incomplete']],view['availability'],view['qmask'],view['umask'],
                    view['lengths'],predict_missing=False)
            root,decoder,active=capture.finish()
            local,base,gap,av,umask=capture.raw
            valid=umask.T.bool()
            local=torch.where(valid[...,None],local,0.)
            base=torch.where(valid[...,None],base,0.)
            gap=torch.where((valid[...,None]&~av.bool())[...,None],gap,0.)
            hidden=model.osram.emotion_norm(model.osram.local_skip(local)+
                model.osram.emotion_adapter(torch.cat((local,base,gap.flatten(2)),-1)))
            pred_off=model.smax_fc(hidden).squeeze(-1)
        # Conversation-major is the original prediction archive order.
        batch_valid=view['umask'].bool()
        def select(x):return x.transpose(0,1)[batch_valid].cpu().numpy()
        values={'flat':select(hidden),'rooted':select(root),'predecoder':select(decoder),
            'role_active':select(active),'labels':view['labels'][batch_valid].cpu().numpy(),
            'availability':select(view['availability']),'prediction':select(output[0]).reshape(-1),
            'flat_prediction':select(pred_off)}
        positions=batch_valid.nonzero().cpu().tolist()
        values['conversation_ids']=np.array([str(view['conversation_ids'][b]) for b,t in positions])
        values['utterance_indices']=np.array([t for b,t in positions])
        assert not root[~valid].count_nonzero()
        for key,value in values.items():pieces[key].append(value)
    data={k:np.concatenate(v) for k,v in pieces.items()}
    for k in ('flat','rooted','predecoder','labels','prediction'):assert np.isfinite(data[k]).all()
    assert len(set(zip(data['conversation_ids'],data['utterance_indices'])))==len(data['labels'])
    assert not data['rooted'][data['utterance_indices']==0].any()
    assert not data['rooted'][~np.repeat(data['role_active'],64,axis=1)].any()
    return data

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--code-commit',required=True)
    args=p.parse_args()
    uuid=subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid==GPU==os.environ.get('CUDA_VISIBLE_DEVICES')
    args.output.mkdir(parents=True,exist_ok=False)
    state={'status':'running','pid':os.getpid(),'server':'biggpu','gpu':7,'gpu_uuid':GPU,
        'code_commit':args.code_commit,'backbone_training_runs':0,'probe_seeds':[66,67,68],
        'probe_epochs':100,'probe_selection':'fixed final epoch','records':[],
        'label':'INTERNAL DIAGNOSTIC ONLY; inherited Test-oracle backbone, train-only fixed-epoch probes'}
    dump(args.output/'STATUS.json',state)
    try:
        from gcnet_missing_m3.train_gcnet import TrainConfig,_schedules
        from gcnet_modality_jepa.train_gcnet import get_loaders
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        torch.set_num_threads(1)
        config=TrainConfig(**json.loads((args.source/'config.json').read_text()))
        assert config.seed==66 and config.osram_meaningful_block=='nested_gnn_rooted_evidence'
        assert config.training_objective=='emotion-only' and config.osram_readout_fusion=='flat'
        dataset=Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset')
        os.environ['GCNET_DATASET_ROOT']=str(dataset)
        features=[str(dataset/'CMUMOSI/features'/n) for n in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
        train,val,test,*dims=get_loaders(audio_root=features[0],text_root=features[1],video_root=features[2],
            num_folder=1,dataset='CMUMOSI',batch_size=config.batch_size,num_workers=0,seed=66,
            validation_fraction=config.validation_fraction,evaluation_protocol=config.evaluation_protocol)
        del val
        loaders={'train':train[0],'test':test[0]}
        device=torch.device('cuda:0')
        model=_build_model(config,tuple(dims)).to(device).eval().requires_grad_(False)
        all_results=[]
        for rate in [i/10 for i in range(8)]:
            folder=args.output/f'rate_{rate:.1f}'.replace('.','p');folder.mkdir()
            suffix=f'{rate:.1f}'.replace('.','p')
            checkpoint=args.source/f'best_miss_{suffix}.pt';checkpoint_hash=sha(checkpoint)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model'],strict=True);epoch=int(saved['epoch']);del saved
            before=state_hash(model)
            splits={name:extract(model,loader,_schedules(config,name)[rate],tuple(dims),device)
                for name,loader in loaders.items()}
            assert not set(splits['train']['conversation_ids'])&set(splits['test']['conversation_ids'])
            archive=np.load(args.source/f'predictions_miss_{suffix}.npz')
            assert np.array_equal(splits['test']['labels'],archive['labels'])
            assert np.array_equal(splits['test']['availability'],archive['availability'])
            error=float(np.abs(splits['test']['prediction']-archive['predictions'].reshape(-1)).max())
            assert error<=1e-5
            for name,data in splits.items():np.savez_compressed(folder/f'{name}_features.npz',**data)
            record={'rate':rate,'checkpoint':str(checkpoint),'checkpoint_epoch':epoch,
                'checkpoint_sha256':checkpoint_hash,'state_sha256':before,'archive_max_abs':error,
                'mask_sha256':hashlib.sha256(splits['test']['availability'].astype('float32').tobytes()).hexdigest(),
                'nested_full':metrics(splits['test']['labels'],splits['test']['prediction']),
                'flat_off_original_head':metrics(splits['test']['labels'],splits['test']['flat_prediction'])}
            print(json.dumps({'rate':rate,'phase':'extracted','test_n':len(splits['test']['labels'])}),flush=True)
            fits=run_probes(splits,folder)
            all_results.extend(dict(r,rate=rate) for r in fits)
            assert before==state_hash(model) and checkpoint_hash==sha(checkpoint)
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            record['frozen_unchanged']=True;record['probe_fits']=len(fits)
            state['records'].append(record);dump(args.output/'STATUS.json',state)
            dump(args.output/'PROBE_RESULTS.json',all_results)
            print(json.dumps({'rate':rate,'phase':'complete','fits':len(fits)}),flush=True)
        state['status']='completed';dump(args.output/'STATUS.json',state)
        from experiments.osram_nested_root_probe_20261011.report import render
        render(args.output)
    except Exception as e:
        state.update(status='failed',error=repr(e));dump(args.output/'STATUS.json',state);raise

if __name__=='__main__':main()
