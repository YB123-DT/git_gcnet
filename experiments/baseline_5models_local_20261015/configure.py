"""Generate an explicit frozen queue. Readiness is granted by verified smoke gates."""
from pathlib import Path
from runtime import ROOT,RUN_ROOT,GPUS,HOST,PYTHON,atomic_json

VERSIONS = [('MMIN','original','Utterance-only Causal'),('MoMKE','causal','Causal'),('GCNet','original','Non-causal'),('SDR-GNN','released_compatible','Non-causal'),('ComP','causal','Causal'),('MoMKE','original','Non-causal'),('ComP','original','Non-causal'),('SDR-GNN','implementation_fixed','Non-causal')]


def config(model,version,dataset,seed,mr,fold):
    if model == 'SDR-GNN' and version not in ('released_compatible', 'implementation_fixed'):
        raise ValueError('SDR-GNN requires an explicit released_compatible or implementation_fixed version')
    c = dict(model=model,model_version=version,dataset=dataset,seed=seed,missing_rate=mr,fold=fold,
             eval_mask_seed=20261015,checkpoint_selection='test_best_acc',evaluation_protocol='test_oracle',
             selection_range='stage2_only' if model in ('MoMKE','ComP') else 'all_epochs',
             optimizer='Adam',weight_decay=0. if model=='MMIN' else 1e-5,
             hidden=256,depth=4,num_heads=2,drop_rate=.5,attn_drop_rate=0.,lr=1e-4,
             epochs=100,stage_epoch=0,batch_size=32,causal=version=='causal',temporal_causal=version=='causal',lbd=.3,
             n_classes={'MELD':7,'IEMOCAPFour':4,'IEMOCAPSix':6}[dataset],
             n_speakers=9 if dataset=='MELD' else 2,teacher_epochs=100 if dataset=='MELD' else 20)
    if model=='ComP': c.update(epochs=300 if dataset=='MELD' else 200,stage_epoch=150 if dataset=='MELD' else 100)
    if model=='MoMKE': c.update(epochs=200,stage_epoch=100,batch_size=16)
    if model in ('GCNet','SDR-GNN'): c.update(lr=.001,hidden=200,window_past=2 if model=='GCNet' else 3,window_future=2 if model=='GCNet' else 3,base_model='LSTM' if model=='GCNet' else 'GRU',recon_weight=1. if model=='GCNet' else .5)
    if model=='MMIN': c.update(epochs=100 if dataset=='MELD' else 60,batch_size=256 if dataset=='MELD' else 128,lr=.001 if dataset=='MELD' else .0002,ae_layers=[256,128,64],n_blocks=5,recon_weight=4.,cycle_weight=2.)
    if model=='MMIN':
        import json
        c.update(json.loads((ROOT/'adapters/mmin/hyperparameters.json').read_text())[dataset])
        c['hidden']=128
    if model == 'SDR-GNN' and version == 'released_compatible':
        c.update(attention_embed_dim=1666 if dataset == 'MELD' else 2560,
                 attention_heads=238 if dataset == 'MELD' else 256,
                 python_hash_seed=seed,
                 relation_mapping='row_major_9speaker' if dataset == 'MELD' else 'upstream_set_with_startup_hash_seed',
                 reconstruction_layout='upstream_time_major_features_batch_major_padding')
    c['implementation_label'] = ('MMIN-UTT-adapted Original' if model=='MMIN' else 'SDR-GNN-Released-code-Compatible' if model=='SDR-GNN' and version=='released_compatible' else 'SDR-GNN-Implementation-fixed' if model=='SDR-GNN' else model+'-'+version) + (' (MELD-adapted)' if dataset=='MELD' and model in ('GCNet','SDR-GNN') else '')
    return c


def main():
    import json
    existing=ROOT/'experiment_registry.json'
    if existing.exists():
        registry=json.loads(existing.read_text())
        if any((Path(t['run_dir'])/'launch.json').exists() for t in registry['tasks']) or (ROOT/'audit/momke_budget_decision.json').exists():
            raise RuntimeError('Execution has started: preserve frozen registry/configs; use dispatcher to resume')
    tasks=[]
    for dataset in ('MELD','IEMOCAPFour','IEMOCAPSix'):
        for index,(model,version,causality) in enumerate(VERSIONS):
            for seed in (66,67,68):
                for mr_i in (3,6,0,1,2,4,5,7):
                    for fold in ([0] if dataset=='MELD' else range(1,6)):
                        mr=mr_i/10
                        c=config(model,version,dataset,seed,mr,fold)
                        ident=f'{dataset}/{model}/{version}/seed_{seed}/mr_{mr:.1f}/fold_{fold}'
                        appendix = (model, version) in {('MoMKE','original'), ('ComP','original'), ('SDR-GNN','implementation_fixed')}
                        priority=(0 if dataset=='MELD' else 20 if dataset=='IEMOCAPFour' else 30)+(40 if appendix else 0)
                        tasks.append(dict(id=ident,config=c,run_dir=str(RUN_ROOT/ident),causality=causality,phase='formal',priority=priority,gate=f'{model}_{version}_{dataset}',status='MISSING'))
    # Budget pilots remain traceable and are explicitly reused when selected and identical.
    for mr in (.3,.6):
        for epochs,stage in ((100,50),(200,100)):
            c=config('MoMKE','causal','MELD',66,mr,0)
            c.update(epochs=epochs,stage_epoch=stage)
            ident=f'pilots/MELD/MoMKE/causal/seed_66/mr_{mr:.1f}/budget_{epochs}_{stage}'
            tasks.append(dict(id=ident,config=c,run_dir=str(RUN_ROOT/ident),causality='Causal',phase='budget_pilot',priority=-2,gate='MoMKE_causal_MELD_pilot',status='MISSING'))
    for task in tasks:
        path=ROOT/'configs'/ (task['id'].replace('/','__')+'.json')
        if path.exists() and __import__('json').loads(path.read_text()) != task['config']:
            raise RuntimeError('Config already exists with different content: '+str(path))
        atomic_json(path,task['config'])
        task['config_path']=str(path)
    atomic_json(ROOT/'experiment_registry.json',dict(host=HOST,gpu_uuids=GPUS,python=PYTHON,max_gpu_jobs=2,formal_seed_configs=576,formal_fold_units=2112,tasks=tasks))
    print(f'{len(tasks)} queued fold-units ({sum(t["phase"]=="formal" for t in tasks)} formal, 4 budget pilots)')
if __name__=='__main__': main()
