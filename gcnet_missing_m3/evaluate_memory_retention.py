"""Evaluation-only real-checkpoint retention collection; no optimizer exists."""
import argparse
import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path

import torch
from . import train_gcnet as tr
from .memory_retention import MemoryRetentionDiagnostics
from .model import MissingM3GraphModel
from .analyze_osram_memory_retention import summarize, write_reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--feature-root',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--rate',type=float,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=tr.TrainConfig(**checkpoint['config'])
    assert cfg.backbone_type=='osram' and not cfg.osram_bidirectional
    assert not cfg.osram_forward_slot_reuse
    tr.set_random_seed(cfg.seed)
    shape=tr._resolve_task_contract(cfg.dataset,cfg.mosi_task_mode)
    paths=[str(args.feature_root/name) for name in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
    loaders=tr.get_loaders(audio_root=paths[0],text_root=paths[1],video_root=paths[2],
        num_folder=int(shape['num_folds']),dataset=cfg.dataset,batch_size=cfg.batch_size,
        num_workers=0,seed=cfg.seed,validation_fraction=cfg.validation_fraction,
        evaluation_protocol=cfg.evaluation_protocol)
    _,_,test,adim,tdim,vdim=loaders
    settings=asdict(cfg)
    names=inspect.signature(MissingM3GraphModel.__init__).parameters
    kw={k:v for k,v in settings.items() if k in names}
    kw.update(adim=adim,tdim=tdim,vdim=vdim,D_e=cfg.hidden,
        graph_hidden_size=cfg.hidden//2,n_speakers=int(shape['num_speakers']),
        n_classes=int(shape['num_classes']),time_attn=cfg.time_attention,no_cuda=args.device=='cpu')
    model=MissingM3GraphModel(**kw).to(args.device).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    schedule=tr._schedules(cfg,'test')[args.rate]
    records=[];previous={};continuity=[];mask_digest=hashlib.sha256()
    with (args.output_dir/'retention.jsonl').open('w') as stream, torch.no_grad():
        def sink(row):
            stream.write(json.dumps(row)+'\n');records.append(row)
            if row['status']=='retention':
                key=(row['sample_id'],row['target_modality'],row['head'])
                old=previous.get(key)
                if old and old['time_index']+1==row['time_index'] and old['history_distance']+1==row['history_distance']:
                    diff=max(abs(old[f'{m}_post']-row[f'{m}_pre']) for m in ('err','cos','norm_ratio'))
                    continuity.append(diff)
                previous[key]=row
        for batch_index,raw in enumerate(test[cfg.fold-1]):
            data=tr._move_batch(raw,torch.device(args.device))
            view=tr._prepare_view(data,schedule,epoch=0,dimensions=(adim,tdim,vdim))
            mask_digest.update(view['availability'].cpu().numpy().tobytes())
            collector=MemoryRetentionDiagnostics(sink,dataset=cfg.dataset,missing_rate=args.rate,
                sample_ids=[f'seed{cfg.seed}:{x}' for x in data[-1]])
            def inject(module,inputs,kwargs):
                return inputs,dict(kwargs,collect_memory_retention_diagnostics=True,
                                   memory_retention_diagnostics=collector)
            inputs=([view['incomplete']],view['availability'],view['qmask'],view['umask'],view['lengths'])
            reference=model(*inputs,predict_missing=False)[0] if batch_index==0 else None
            handle=model.osram.register_forward_pre_hook(inject,with_kwargs=True)
            try:
                output=model(*inputs,predict_missing=False)[0]
            finally:
                handle.remove()
            if reference is not None:
                assert torch.equal(reference,output),'logits changed with diagnostics'
            stream.flush()
            print(f'batch={batch_index} records={len(records)}',flush=True)
    assert continuity and max(continuity)<1e-6, 'continuous-missing post/pre mismatch'
    write_reports(summarize(records),args.output_dir/'tables')
    metadata=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=tr._sha256_file(args.checkpoint),
        epoch=checkpoint['epoch'],selection_protocol=checkpoint.get('selection_protocol'),
        dataset=cfg.dataset,seed=cfg.seed,rate=args.rate,records=len(records),
        continuity_pairs=len(continuity),continuity_max_metric_difference=max(continuity),
        continuity_scope='same last probe, consecutive missing: error/cosine/norm_ratio post vs pre',
        first_batch_logits_exact=True,mask_sha256=mask_digest.hexdigest(),evaluation_only=True)
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':
    main()
