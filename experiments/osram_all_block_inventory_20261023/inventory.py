"""Read metadata only: no Torch, checkpoints, inference, training or mutations of runs.

The same stdlib script can scan remote roots over SSH stdin. Generated tables
retain evidence granularity: a configuration, a summary and an Optuna trial are
not silently counted as independent trained models. Unknown fields stay unknown.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import html
import io
import gzip
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

SELF = 'osram_all_block_inventory_20261023'
ROOT = Path(__file__).resolve().parents[2] if '__file__' in globals() and __file__ != '<stdin>' else Path.cwd()
REMOTE_ROOTS = ['/data2/yb/remote_experiments', '/data1/yb/remote_experiments',
    '/home/qinjinghui/remote_experiments', '/data2/yb/paper/remote_runs_20261012/results',
    '/data2/yb/experiment_records',
    '/data2/yb/paper/GCNet_TPAMI/.worktrees/osram-reg-only/experiments']
SKIP = {'dataset','datasets','data','features','cache','feature_cache','envs',
        'node_modules','wandb','training_versions','__pycache__','tests','literature',
        'source','code','snapshot',SELF}


def snapshot_directory(name):
    return (name.startswith(('.', '__','source_', 'snapshot','code_','control_','_code'))
            or '_code_' in name or '_snapshot_' in name or name=='code_snapshots')
CONTEXT = ('dataset','seed','fold','mode','variant','model','method','candidate','id',
           'name','status','phase','epoch','epochs','rate','missing_rate','scope',
           'source_commit','code_commit','label','no_training','inference_only','smoke_only')
METRIC_KEYS = {'accuracy':'ACC','acc':'ACC','weightedf1':'WF1','wf1':'WF1',
    'macrorecall':'UA','uar':'UA','ua':'UA','unweightedaccuracy':'UA',
    'balancedaccuracy':'UA','mse':'MSE','mae':'MAE','ce':'CE'}
MASK_PROTOCOLS={'gcnet-random-mask-v1','conversation-mask-v2','gcnet-legacy-fixed-v1'}
SCOPES={'mean','high','mean8','overall','macro','paired','testbest','persistent','rate',
        'unspecified','eight-rate','fivefold-eight-rate','stored_selected_point'}
CFG_KEYS = ('dataset','seed','fold','epochs','learning_rate','weight_decay','batch_size',
    'optimizer','dropout','projector_dropout','lr_schedule','warmup_ratio','backbone_type',
    'training_objective','task_regression_loss','mosi_task_mode','fusion_type','representation_type',
    'missing_mask_protocol','checkpoint_selection','checkpoint_selection_metric',
    'evaluation_protocol','train_rate_mode','fixed_missing_rate','train_missing_rates',
    'initial_backbone_checkpoint','pretrained_learning_rate','gradient_clip_norm',
    'paired_history_views','history_drop_prob','history_task_view2_weight',
    'history_contrast_weight','history_contrast_temperature','core20_method',
    'core20_aux_weight','num_experts','target_space','readout_type')
TUNABLES = ('learning_rate','weight_decay','batch_size','dropout','lr_schedule',
    'warmup_ratio','pretrained_learning_rate','history_task_view2_weight',
    'history_contrast_weight','history_drop_prob','core20_aux_weight')


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def scalar(v):return v is None or isinstance(v,(str,int,float,bool))


def compact_config(cfg):
    if not isinstance(cfg,dict):return {}
    return {k:v for k,v in cfg.items() if (k in CFG_KEYS or k.startswith('osram_'))
            and (scalar(v) or (isinstance(v,list) and all(scalar(x) for x in v)))}


def normalized_metrics(obj):
    result={}
    for key,value in obj.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):continue
        norm=re.sub('[^a-z0-9]','',key.lower())
        name=METRIC_KEYS.get(norm)
        if name is None:continue
        value=float(value)
        # Uppercase ACC/UA/WF1 in these experiment workers are explicit percent.
        # Legacy lowercase trainer accuracy/weighted_f1 are fractions.
        if name in ('ACC','UA','WF1') and key not in ('ACC','UA','WF1') and 0<=value<=1:
            value*=100
        result[name]=value
    return result


def protocol(cfg, obj, old_index):
    value=cfg.get('missing_mask_protocol')
    if isinstance(value,str) and value in MASK_PROTOCOLS:return str(value),'effective_config'
    for key in ('missing_mask_protocol','mask_protocol','missing_protocol'):
        if isinstance(obj.get(key),str) and obj[key] in MASK_PROTOCOLS:return obj[key],'explicit_metadata:'+key
    p=obj.get('protocol')
    if isinstance(p,str) and p in ('gcnet-random-mask-v1','conversation-mask-v2','gcnet-legacy-fixed-v1'):
        return p,'explicit_metadata:protocol'
    identity=obj.get('identity',{})
    for parent in (obj,identity if isinstance(identity,dict) else {}):
        for key in ('source_hashes','source_sha256'):
            hashes=parent.get(key,{})
            if isinstance(hashes,dict) and hashes.get('gcnet_missing_m3/train_gcnet.py') in old_index:
                return 'conversation-mask-v2','exact_executed_trainer_hash'
    return 'unknown','not_inferred_from_date_or_current_defaults'


def classify_status(obj, has_scores=False):
    text=str(obj.get('status',obj.get('state',''))).lower()
    if obj.get('smoke_only') or obj.get('phase')=='smoke':
        return 'smoke_complete' if text in ('complete','completed','success','finished') else 'smoke_'+(text or 'recorded')
    if obj.get('no_training') or obj.get('inference_only'):
        return 'diagnostic_'+(text or 'results_present' if has_scores else text or 'recorded')
    if text in ('complete','completed','success','finished','done'):return 'complete_declared'
    if text:return text
    return 'results_present_completion_unknown' if has_scores else 'configuration_or_metadata_only'


def embedded_config(obj, fallback=None):
    out=dict(fallback or {})
    identity=obj.get('identity',{})
    for candidate in (identity.get('config') if isinstance(identity,dict) else None,
                      obj.get('effective_config'),obj.get('cfg'),obj.get('config')):
        if isinstance(candidate,dict):out.update(compact_config(candidate))
    if 'dataset' in obj and any(k in obj for k in ('learning_rate','backbone_type','train_missing_rates')):
        out.update(compact_config(obj))
    return out


def extract_document(obj, source_file, fallback, old_index):
    rows=[]
    skip={'identity','config','cfg','effective_config','source_hashes','source_sha256',
        'source_files','data_manifest','data_files','mask_sha256','mask_hashes',
        'optimizer','rng','history','untracked_paths','gradient_rows','samples',
        'per_sample','utterances','predictions','labels','confusion','confusion_matrix'}
    def visit(value, pointer, context, cfg):
        if isinstance(value,list):
            for i,v in enumerate(value):
                if isinstance(v,(dict,list)):visit(v,pointer+'/'+str(i),context,cfg)
            return
        if not isinstance(value,dict):return
        cfg=embedded_config(value,cfg)
        own_cfg=embedded_config(value)
        ctx=dict(context)
        ctx.update({k:v for k,v in value.items() if k in CONTEXT and scalar(v)})
        for k in ('dataset','seed','fold','epochs'):
            if k in cfg and k not in value and (k in own_cfg or k not in ctx):ctx[k]=cfg[k]
        measured=normalized_metrics(value)
        if measured:
            parts=pointer.split('/')
            rate=ctx.get('rate',ctx.get('missing_rate',''))
            rates=[p for p in parts if re.fullmatch(r'0[.][0-7]',p)]
            if rates:rate=rates[-1]
            scope=ctx.get('scope','')
            if not scope:
                scope=next((p for p in reversed(parts) if p in ('mean','high','mean8','overall','macro','paired','testbest','persistent')), '')
            if not scope and rate!='':scope='rate'
            proto,proof=protocol(cfg,value,old_index)
            if proto=='unknown':proto,proof=protocol(cfg,obj if isinstance(obj,dict) else {},old_index)
            row=dict(source_file=source_file,json_pointer=pointer or '/',
                dataset=ctx.get('dataset','unknown'),seed=ctx.get('seed',''),fold=ctx.get('fold',''),
                model=ctx.get('model',ctx.get('method',ctx.get('candidate',ctx.get('variant',ctx.get('mode',''))))),
                variant=ctx.get('variant',ctx.get('mode','')),rate=str(rate),scope=scope or 'unspecified',
                selection_scope='/'.join(p for p in parts if p in ('testbest','test_best','paired','paired_mask','PAIRED_MASK_REPLAY','validation','test')),
                status=classify_status(ctx,True),epoch=ctx.get('epoch',''),
                missing_protocol=proto,protocol_proof=proof,
                checkpoint_selection=cfg.get('checkpoint_selection','unknown'),
                checkpoint_selection_metric=cfg.get('checkpoint_selection_metric','unknown'),
                **measured)
            rows.append(row)
        for key,child in value.items():
            if key in skip:continue
            if isinstance(child,(dict,list)):visit(child,pointer+'/'+str(key),ctx,cfg)
    visit(obj,'',{},fallback)
    return rows


def method_id(cfg, obj, path):
    for key in ('osram_meaningful_block','core20_method','osram_readout_candidate'):
        value=cfg.get(key)
        if value not in (None,'none','flat'):return str(value)
    flags=[key.removeprefix('osram_') for key,value in cfg.items()
           if key.startswith('osram_') and value is True and key not in
           ('osram_query_availability','osram_source_tracking')]
    if flags:return '+'.join(flags)
    for key in ('method','candidate','variant','mode','model'):
        if scalar(obj.get(key)) and obj.get(key) is not None:return str(obj[key])
    return 'Flat/control/unspecified'


def pick_mean(raw):
    if not isinstance(raw,dict):return {}
    selected=selected_metrics(raw)
    if selected:return selected['mean']
    aggregate=raw.get('aggregate',{})
    if isinstance(aggregate,dict):
        for cand in (aggregate.get('router',{}).get('mean',{}) if isinstance(aggregate.get('router'),dict) else {},
                     aggregate.get('mean',{}),aggregate.get('overall',{})):
            m=normalized_metrics(cand) if isinstance(cand,dict) else {}
            if m:return m
    values=raw.get('evaluation',raw.get('test',{}))
    if not isinstance(values,dict) or not all(f'{i/10:.1f}' in values for i in range(8)):return {}
    rows=[]
    for i in range(8):
        value=values[f'{i/10:.1f}']
        if not isinstance(value,dict):return {}
        value=value.get('router',value)
        measured=normalized_metrics(value)
        if not measured:return {}
        rows.append(measured)
    common=set.intersection(*(set(r) for r in rows))
    return {k:sum(r[k] for r in rows)/8 for k in sorted(common)}


def selected_metrics(raw):
    """Legacy final test replay is not the stored winning-epoch prediction."""
    if not isinstance(raw,dict):return {}
    rates=[f'{i/10:.1f}' for i in range(8)]
    rows={r:{} for r in rates}
    for key,name in (('selected_accuracy_by_rate','ACC'),
                     ('selected_weighted_f1_by_rate','WF1')):
        values=raw.get(key,{})
        if not isinstance(values,dict) or not all(r in values for r in rates):continue
        if not all(isinstance(values[r],(int,float)) and math.isfinite(values[r]) and 0<=values[r]<=1 for r in rates):continue
        for r in rates:rows[r][name]=float(values[r])*100
    if not rows[rates[0]]:return {}
    return dict(mean={k:sum(rows[r][k] for r in rates)/8 for k in rows[rates[0]]},
                rates=rows,epochs=raw.get('selected_epoch_by_rate',{}))


def refresh_selected(output,snapshot):
    """Bounded metadata replay: exact metric file list, not another tree walk."""
    scans=[compact_scan(s) for s in snapshot]
    local={r['source_file'] for s in scans if s['origin']=='local' for r in s['runs']
           if Path(r['source_file']).name=='metrics.json'}
    remote={r['source_file'] for s in scans if s['origin']=='biggpu' for r in s['runs']
            if Path(r['source_file']).name in ('metrics.json','scores_original.json')}
    result={}
    for path in sorted(local):
        raw,_=safe_load(Path(path));sel=selected_metrics(raw)
        if sel:result[path]=sel
    if remote:
        script=Path(__file__).read_text().rsplit("if __name__=='__main__':main()",1)[0]
        script+='\nfor path in '+repr(sorted(remote))+':\n'
        script+='    raw,error=safe_load(Path(path));sel=selected_metrics(raw)\n'
        script+='    if sel:print(dumps(dict(path=path,selected=sel)),flush=True)\n'
        script+='    if isinstance(raw,list):\n'
        script+='        for i,item in enumerate(raw):\n'
        script+='            sel=selected_metrics(item.get("metrics",{})) if isinstance(item,dict) else {}\n'
        script+='            if sel:print(dumps(dict(path=path+"#/"+str(i),selected=sel)),flush=True)\n'
        completed=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','biggpu','python -'],
            input=script,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=120)
        for line in completed.stdout.splitlines():
            item=json.loads(line);result[item['path']]=item['selected']
    write_json(output/'SELECTED_METRICS.json',result)
    print('Refreshed stored selected metrics: '+str(len(result)),flush=True)


def environment(obj, extras):
    identity=obj.get('identity',{})
    env=obj.get('environment',identity.get('environment',{}) if isinstance(identity,dict) else {})
    if not isinstance(env,dict):env={}
    for v in extras:
        if isinstance(v,dict) and isinstance(v.get('environment'),dict):env={**v['environment'],**env}
    return '; '.join(str(env[k]) for k in ('python','python_version','torch','numpy','sklearn') if k in env)


def embedded_runs(obj, source_file, project, origin, old_index):
    """Retain archived runs and HPO trials whose leaf folders no longer exist.

    These remain evidence records, not another count of completed trainings.
    Test-best and paired means get separate columns, never spliced together.
    """
    rows=[]
    def visit(value, pointer=''):
        if isinstance(value,list):
            for i,item in enumerate(value):visit(item,pointer+'/'+str(i))
            return
        if not isinstance(value,dict):return
        cfg=embedded_config(value)
        if cfg.get('dataset') and any(k in value for k in
                ('metrics','testbest','paired','original_run','number','trial_number','summary')):
            prov=value.get('provenance',{})
            if not isinstance(prov,dict):prov={}
            proto,proof=protocol(cfg,value,old_index)
            if proto=='unknown':proto,proof=protocol(cfg,prov,old_index)
            metrics=value.get('metrics',{})
            if not isinstance(metrics,dict):metrics={}
            scores=pick_mean(metrics)
            testbest=value.get('testbest',{})
            if isinstance(testbest,dict) and isinstance(testbest.get('mean'),dict):
                scores=normalized_metrics(testbest['mean'])
            status={**prov,**{k:value[k] for k in ('state','status','smoke_only') if k in value}}
            row=dict(project=project,origin=origin,run_directory=value.get('original_run',''),
                source_file=source_file,json_pointer=pointer or '/',
                evidence_kind='embedded_trial_or_archived_run',
                method_id=method_id(cfg,value,source_file),dataset=cfg['dataset'],
                seed=cfg.get('seed',''),fold=cfg.get('fold',''),
                trial=value.get('number',value.get('trial_number','')),
                epochs_planned=cfg.get('epochs',''),epoch_recorded='',
                status=classify_status(status,bool(scores)),missing_protocol=proto,protocol_proof=proof,
                checkpoint_selection=cfg.get('checkpoint_selection',metrics.get('selection_protocol','unknown')),
                checkpoint_selection_metric=cfg.get('checkpoint_selection_metric',metrics.get('selection_metric','unknown')),
                environment=environment(prov,[]),
                source_commit=prov.get('git_sha',prov.get('source_commit','')),
                learning_rate=cfg.get('learning_rate',''),weight_decay=cfg.get('weight_decay',''),
                batch_size=cfg.get('batch_size',''),
                selected_epochs_by_rate=metrics.get('selected_epoch_by_rate',{}),
                parameter_count=metrics.get('parameter_count',value.get('parameters','')),
                config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                scope='eight_rate_mean' if scores else 'no_verified_eight_rate_mean',config=cfg,**scores)
            paired=value.get('paired',{})
            if isinstance(paired,dict):
                on=paired.get('on',paired)
                mean=on.get('mean',{}) if isinstance(on,dict) else {}
                row.update({'PAIRED_'+k:v for k,v in normalized_metrics(mean).items()})
            rows.append(row)
        for key,child in value.items():
            if key in ('config','identity','cfg','effective_config','provenance','metrics',
                       'paired','testbest','parameters','params','summary'):continue
            if isinstance(child,(dict,list)):visit(child,pointer+'/'+key)
    visit(obj)
    return rows


def selected_json(name):
    lower=name.lower()
    return lower.endswith('.json') and 'history' not in lower and any(x in lower for x in
        ('summary','metrics','report','trials','scores','aggregate','result','config','status','queue','provenance','parameters','runner'))


def safe_load(path):
    try:
        if path.stat().st_size>32*1024**2:return None,'over_32MiB_not_loaded'
        return json.loads(path.read_text()),None
    except (OSError,ValueError,UnicodeError) as e:return None,type(e).__name__+':'+str(e)[:140]


def scan_tree(base, origin, old_index):
    base=Path(base)
    result=dict(root=str(base),origin=origin,available=base.is_dir(),projects={},runs=[],metrics=[],tables=[],files=[],issues=[])
    if not base.is_dir():return result
    for dp,dirs,names in os.walk(base,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in SKIP and not snapshot_directory(d))
        here=Path(dp);relative=here.relative_to(base)
        if not relative.parts:continue
        if 'gcnet_missing_m3' in dirs and 'experiments' in dirs:
            # An immutable repository checkout inside an experiment is source,
            # not a second collection of experiments copied from that checkout.
            dirs[:]=[]
            continue
        project=relative.parts[0]
        if snapshot_directory(project) or project==SELF:
            dirs[:]=[];continue
        project_row=result['projects'].setdefault(project,dict(name=project,paths=[],documents=[],headings=[]))
        if str(base/project) not in project_row['paths']:project_row['paths'].append(str(base/project))
        docs={}
        for name in sorted(names):
            path=here/name
            if path.is_symlink() or not path.is_file():continue
            low=name.lower()
            if low.endswith('.md'):
                try:
                    if path.stat().st_size>2*1024**2:continue
                    text=path.read_text();project_row['documents'].append(str(path))
                    project_row['headings'].extend(line.strip('# ').strip() for line in text.splitlines()
                                                  if line.startswith('#') and len(line)<180)
                    header=None
                    for number,line in enumerate(text.splitlines(),1):
                        if line.strip().startswith('|'):
                            if re.fullmatch(r'[| :\-]+',line.strip()):continue
                            if header is None:header=line
                            result['tables'].append(dict(project=project,origin=origin,
                                source_file=str(path),line=number,header=header,row=line))
                        else:header=None
                    result['files'].append(dict(path=str(path),project=project,origin=origin,
                        sha256=hashlib.sha256(text.encode()).hexdigest(),kind='markdown'))
                except (OSError,UnicodeError) as e:result['issues'].append(dict(path=str(path),error=str(e)))
            elif selected_json(name):
                obj,error=safe_load(path)
                if error:
                    result['issues'].append(dict(path=str(path),error=error));continue
                docs[name]=obj
                result['files'].append(dict(path=str(path),project=project,origin=origin,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kind='json'))
        fallback={}
        for parent in [here,*list(here.parents)[:8]]:
            if parent!=base and base not in parent.parents:break
            p=parent/'config.json'
            if p.is_file():
                cfg,_=safe_load(p)
                if isinstance(cfg,dict) and 'dataset' in cfg:fallback=compact_config(cfg);break
        for name,obj in docs.items():
            if not isinstance(obj,(dict,list)):continue
            metrics=extract_document(obj,str(here/name),fallback,old_index)
            for row in metrics:row.update(project=project,origin=origin)
            result['metrics'].extend(metrics)
            result['runs'].extend(embedded_runs(obj,str(here/name),project,origin,old_index))
        primary=next((docs[n] for n in ('SUMMARY.json','summary.json','metrics.json','status.json','STATUS.json')
                      if isinstance(docs.get(n),dict)),None)
        if primary is None:continue
        cfg=embedded_config(primary,fallback)
        if not cfg.get('dataset') and not primary.get('dataset'):continue
        # A project-wide aggregate with no run config is not a trained run.
        if not cfg and not any(k in primary for k in ('epoch','phase','seed','no_training','inference_only')):continue
        proto,proof=protocol(cfg,primary,old_index)
        if proto=='unknown':
            for obj in docs.values():
                if isinstance(obj,dict):
                    proto,proof=protocol(cfg,obj,old_index)
                    if proto!='unknown':break
        scores=pick_mean(primary)
        if not scores and isinstance(docs.get('metrics.json'),dict):scores=pick_mean(docs['metrics.json'])
        raw_state=primary
        # Explicit status.json wins over a result file that has no state.
        if not primary.get('status') and isinstance(docs.get('status.json'),dict):
            raw_state={**primary,**docs['status.json']}
        source_file=next(str(here/n) for n,v in docs.items() if v is primary)
        config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        run=dict(project=project,origin=origin,run_directory=str(here),source_file=source_file,
            evidence_kind='leaf_result_directory',
            method_id=method_id(cfg,primary,str(here)),dataset=cfg.get('dataset',primary.get('dataset','unknown')),
            seed=cfg.get('seed',primary.get('seed','')),fold=cfg.get('fold',primary.get('fold','')),
            epochs_planned=cfg.get('epochs',primary.get('epochs','')),epoch_recorded=primary.get('epoch',''),
            status=classify_status(raw_state,bool(scores)),missing_protocol=proto,protocol_proof=proof,
            checkpoint_selection=cfg.get('checkpoint_selection',primary.get('selection_protocol','unknown')),
            checkpoint_selection_metric=cfg.get('checkpoint_selection_metric',primary.get('selection_metric','unknown')),
            environment=environment(primary,docs.values()),
            source_commit=primary.get('source_commit',primary.get('identity',{}).get('source_commit','')),
            learning_rate=cfg.get('learning_rate',''),weight_decay=cfg.get('weight_decay',''),
            batch_size=cfg.get('batch_size',''),config_sha256=config_hash,
            selected_epochs_by_rate=primary.get('selected_epoch_by_rate',{}),
            parameter_count=primary.get('parameter_count',primary.get('parameters','')),
            scope='eight_rate_mean' if scores else 'no_verified_eight_rate_mean',
            config=cfg,**scores)
        result['runs'].append(run)
    return result


def dumps(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,default=str)


def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')


def write_csv(path,rows,fields=None):
    fields=fields or sorted({k for row in rows for k in row})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for row in rows:w.writerow({k:dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def load_families(output):
    result=[]
    for name in ('readout_families.json','training_input_families.json'):
        path=output/name
        if path.is_file():result.extend(json.loads(path.read_text()).get('families',[]))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan',action='store_true')
    parser.add_argument('--roots',nargs='*')
    parser.add_argument('--origin',default='biggpu')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--include-remote',action='store_true')
    parser.add_argument('--old-index-json',default='')
    parser.add_argument('--rebuild',action='store_true',help='Regenerate tables from saved scan, without SSH or rescanning')
    parser.add_argument('--refresh-selected',action='store_true')
    args=parser.parse_args(argv)
    index_path=ROOT/'experiments/osram_old_missing_cleanup_20261022/OLD_SOURCE_INDEX.json'
    if not index_path.is_file():index_path=ROOT/'experiments'/SELF/'OLD_MASK_SOURCE_PROOFS.json'
    old_index=json.loads(args.old_index_json) if args.old_index_json else (json.loads(index_path.read_text()) if index_path.is_file() else {})
    if args.scan:
        for root in args.roots:print(dumps(scan_tree(root,args.origin,old_index)),flush=True)
        return
    output=args.output or ROOT/'experiments'/SELF
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'OLD_MASK_SOURCE_PROOFS.json',old_index)
    if args.rebuild:
        snapshot=json.loads((output/'SCAN_SNAPSHOT.json').read_text())
        if args.refresh_selected:refresh_selected(output,snapshot)
        build(output,snapshot)
        return
    snapshot=[scan_tree(ROOT/'experiments','local',old_index)]
    if args.include_remote:
        # Checkpoint each root independently: an unavailable archive must not
        # discard results from all preceding successful roots.
        for remote_root in REMOTE_ROOTS:
            command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=10',
                '-o','ServerAliveCountMax=3','biggpu','python - --scan --old-index-json '+shlex.quote(dumps(old_index))+' --roots '+shlex.quote(remote_root)]
            try:
                completed=subprocess.run(command,input=Path(__file__).read_text(),text=True,
                                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=180)
                snapshot.extend(json.loads(line) for line in completed.stdout.splitlines() if line.strip())
                print('Scanned '+remote_root,flush=True)
            except (subprocess.SubprocessError,ValueError) as error:
                scan=dict(root=remote_root,origin='biggpu',available=False,
                          projects={},runs=[],metrics=[],tables=[],files=[],
                          issues=[dict(path=remote_root,error=type(error).__name__,
                                       limitation='Remote root scan incomplete; not evidence of absent results')])
                snapshot.append(scan)
                print('Incomplete remote root: '+remote_root+' ('+type(error).__name__+')',flush=True)
            write_json(output/'SCAN_SNAPSHOT.json',snapshot)
    write_json(output/'SCAN_SNAPSHOT.json',snapshot)
    if args.refresh_selected:refresh_selected(output,snapshot)
    build(output,snapshot)


def fmt(value):return f'{value:.3f}' if isinstance(value,(int,float)) else str(value or '—')


def compact_scan(scan):
    """Also filter copied repository evidence in scans made by older versions."""
    def actual(path):
        try:parts=Path(path).relative_to(scan['root']).parts
        except ValueError:return True
        return not ('experiments' in parts[1:] or
                    any(snapshot_directory(p) or p in ('source','code','snapshot') for p in parts[:-1]))
    result={**scan}
    for key in ('runs','metrics','tables'):
        result[key]=[r for r in scan[key] if actual(r['source_file'])]
    result['files']=[r for r in scan['files'] if actual(r['path'])]
    result['projects']={}
    for name,p in scan['projects'].items():
        docs=[d for d in p['documents'] if actual(d)]
        result['projects'][name]={**p,'documents':docs}
    result['excluded_copied_repository_rows']=sum(len(scan[k])-len(result[k]) for k in ('runs','metrics','tables','files'))
    return result


def build(output,snapshot):
    snapshot=[compact_scan(s) for s in snapshot]
    families=load_families(output);links=collections.defaultdict(list)
    for f in families:
        for project in f.get('experiment_dirs',[]):links[Path(project).name].append(f)
    projects={};runs=[];metrics=[];tables=[];files=[];issues=[]
    for scan in snapshot:
        runs.extend(scan['runs']);metrics.extend(scan['metrics']);tables.extend(scan['tables']);files.extend(scan['files']);issues.extend(scan['issues'])
        for name,row in scan['projects'].items():
            p=projects.setdefault(name,dict(name=name,paths=[],documents=[],headings=[],origins=[]))
            for key in ('paths','documents','headings'):
                p[key].extend(x for x in row[key] if x not in p[key])
            if scan['origin'] not in p['origins']:p['origins'].append(scan['origin'])
    # Preserve source evidence; identical config hashes are not proof of the
    # same optimizer trajectory. Mark possible copies rather than merging them.
    duplicates=collections.Counter((r['dataset'],str(r['seed']),str(r['fold']),r['config_sha256']) for r in runs)
    for row in runs:
        if row['missing_protocol'] not in MASK_PROTOCOLS|{'unknown'}:
            row['declared_missing_protocol']=row['missing_protocol'];row['missing_protocol']='unknown'
        row['families']='; '.join(f['name'] for f in links[row['project']]) or '未归类/见实验目录'
        row['same_config_evidence_count']=duplicates[(row['dataset'],str(row['seed']),str(row['fold']),row['config_sha256'])]
        row['tunable_config']={k:v for k,v in row['config'].items() if (k in TUNABLES or k.startswith('osram_'))
                               and v is not False and v is not None and v!='none'}
    selected_path=output/'SELECTED_METRICS.json'
    selected=json.loads(selected_path.read_text()) if selected_path.is_file() else {}
    added=set()
    for row in runs:
        path=row['source_file'];selection_key=path+'#'+row.get('json_pointer','')
        sel=selected.get(selection_key,selected.get(path))
        if not sel:
            row['score_basis']='source aggregate or test replay; inspect JSON pointer before treating as Test-best'
            continue
        for key in ('ACC','UA','WF1','CE','MSE','MAE'):
            if key in row:
                row['FINAL_RETEST_'+key]=row.pop(key)
        row.update(sel['mean']);row['selected_epochs_by_rate']=sel['epochs']
        row['scope']='eight_rate_mean';row['score_basis']='stored selected_*_by_rate; not final random retest'
        if selection_key in added:continue
        added.add(selection_key)
        for rate,values in sel['rates'].items():
            prefix=(row.get('json_pointer','')+'/metrics' if row.get('json_pointer') else '')
            pointers={k+'_json_pointer':prefix+'/selected_'+('accuracy' if k=='ACC' else 'weighted_f1')+'_by_rate/'+rate for k in values}
            metrics.append(dict(project=row['project'],origin=row['origin'],source_file=path,
                json_pointer=pointers.get('WF1_json_pointer',pointers.get('ACC_json_pointer')),
                scope='stored_selected_point',rate=rate,
                dataset=row['dataset'],seed=row['seed'],fold=row['fold'],
                missing_protocol=row['missing_protocol'],epoch=sel['epochs'].get(rate,''),
                selection_scope='stored_selected_point; see run checkpoint_selection for test/validation',
                model=row['method_id'],**pointers,**values))
    for row in metrics:
        if row.get('missing_protocol') not in MASK_PROTOCOLS|{'unknown',None}:
            row['declared_missing_protocol']=row['missing_protocol'];row['missing_protocol']='unknown'
        if row.get('scope') not in SCOPES:
            row['scope_description']=row.get('scope','')
            parts=row.get('json_pointer','').split('/')
            row['scope']=next((p for p in reversed(parts) if p in SCOPES-{'unspecified','rate'}),
                              'rate' if row.get('rate') else 'unspecified')
    run_by_project=collections.defaultdict(list);metric_by_project=collections.Counter()
    for row in runs:run_by_project[row['project']].append(row)
    for row in metrics:metric_by_project[row['project']]+=1
    project_rows=[]
    for name,p in sorted(projects.items()):
        rr=run_by_project[name];ff=links[name]
        states=collections.Counter(r['status'] for r in rr)
        project_rows.append(dict(experiment=name,
            families='; '.join(f['name'] for f in ff) or '未归类/见原文',
            categories='; '.join(sorted({f.get('category','') for f in ff})),
            tunable_parameters='; '.join(sorted({str(v) for f in ff for v in f.get('tunable_parameters',[])})),
            datasets='; '.join(sorted({str(r['dataset']) for r in rr})),
            seeds='; '.join(sorted({str(r['seed']) for r in rr if r['seed']!=''})),
            missing_protocols='; '.join(sorted({r['missing_protocol'] for r in rr})),
            statuses=dumps(states) if states else '仅文档/汇总/无叶级运行记录，不能推断未训练',
            run_evidence_rows=len(rr),metric_evidence_rows=metric_by_project[name],
            paths=p['paths'],documents=p['documents'],titles=p['headings'][:8],origins=p['origins']))
    write_csv(output/'EXPERIMENTS.csv',project_rows)
    write_csv(output/'RUNS.csv',[{k:v for k,v in r.items() if k!='config'} for r in runs])
    write_csv(output/'METRICS.csv',metrics)
    write_csv(output/'SOURCE_TABLE_ROWS.csv',tables)
    write_json(output/'SOURCE_FILES.json',files)
    write_json(output/'SCAN_ISSUES.json',issues)
    write_json(output/'RUN_RECORDS.json',runs)
    write_json(output/'EXPERIMENTS.json',project_rows)
    write_csv(output/'BLOCK_FAMILIES.csv',families)
    representatives=[{**r,'family_id':f['family_id'],'family':f['name']}
                     for f in families for r in f.get('representative_results',[])]
    write_csv(output/'REPRESENTATIVE_RESULTS.csv',representatives)
    result_lines=['# 各Block家族的已有结果索引','','INTERNAL DIAGNOSTIC ONLY',
        '', '这是带出处的代表结果，不是跨Missing/环境排名。完整逐配置和逐率数据另见RUNS.csv/METRICS.csv。',
        '', '| 家族 | 数据集/范围 | Seed | Epoch | 指标/结果原字段 | 协议/出处 |',
        '|---|---|---|---|---|---|']
    for r in representatives:
        excluded={'family','family_id','dataset','seeds','seed','epochs','source','sources','protocol','warning'}
        value={k:v for k,v in r.items() if k not in excluded}
        cells=[r['family'],r.get('dataset',r.get('scope','见原字段')),
               r.get('seeds',r.get('seed','见原字段')),r.get('epochs','见原字段'),
               dumps(value),str(r.get('protocol',''))+' / '+str(r.get('source',r.get('sources','见家族sources')))]
        result_lines.append('| '+' | '.join(str(x).replace('|','/').replace('\n',' ') for x in cells)+' |')
    (output/'FAMILY_RESULTS.md').write_text('\n'.join(result_lines)+'\n')
    catalog_path=output/'registry_catalog.json'
    catalog=json.loads(catalog_path.read_text()) if catalog_path.is_file() else {}
    write_csv(output/'BLOCK_CATALOG.csv',catalog.get('methods',[]))
    method_counts=collections.Counter(r['method_id'] for r in runs)
    catalog_lines=['# 全部机制/配置入口逐项清单','','INTERNAL DIAGNOSTIC ONLY',
        '', '注册不等于训练；训练证据数量含本地/远程/归档副本，不是独立训练次数。',
        '精确配置、分数和选点见 RUNS.csv、METRICS.csv，原文表格见 SOURCE_TABLE_ROWS.csv。',
        '', '| ID | 名称 | 所属批次 | 配置变体 | 插入位置 | 实现状态 | 同ID运行证据 |',
        '|---|---|---|---|---|---|---:|']
    for m in catalog.get('methods',[]):
        cells=[m['method_id'],m.get('display_name',''),', '.join(m.get('catalog_batches',[])),
               str(m.get('configuration_variant','')),m.get('insertion_point',''),
               m.get('implementation_status',''),str(method_counts[m['method_id']])]
        catalog_lines.append('| '+' | '.join(str(x).replace('|','/').replace('\n',' ') for x in cells)+' |')
    (output/'ALL_BLOCKS.md').write_text('\n'.join(catalog_lines)+'\n')
    counts=dict(experiment_directories=len(project_rows),semantic_family_rows=len(families),
        selector_or_catalog_entries=len(catalog.get('methods',[])),
        run_evidence_rows=len(runs),metric_evidence_rows=len(metrics),
        source_markdown_table_rows=len(tables),source_files=len(files),scan_issues=len(issues),
        run_statuses=dict(collections.Counter(r['status'] for r in runs)),
        missing_protocols=dict(collections.Counter(r['missing_protocol'] for r in runs)))
    manifest=dict(generated_at=utc(),source_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scope='Repository experiment folders plus metadata-only scans of explicitly listed established remote roots',
        roots=[dict(root=s['root'],origin=s['origin'],available=s['available']) for s in snapshot],counts=counts,
        inventory_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        excluded_copied_repository_evidence_rows=sum(s['excluded_copied_repository_rows'] for s in snapshot),
        no_training=True,no_inference=True,no_checkpoint_contents_read=True,
        interpretation='Counts are evidence/configuration granularities, not independent methods or unique training trajectories.')
    write_json(output/'MANIFEST.json',manifest)
    result_lines=['# 盘点结果','','INTERNAL DIAGNOSTIC ONLY','',
        '本轮仅整理记录，没有训练、推理、读取权重或修改正在运行的实验。','',
        '| 内容 | 数量 | 含义 |','|---|---:|---|',
        f"| 实验目录 | {len(project_rows)} | 本地及列明的远端根目录合并同名目录；不是独立训练数 |",
        f"| 语义家族 | {len(families)} | 读出、训练、输入、调参、协议及诊断分类；不是创新数 |",
        f"| 注册/配置入口 | {counts['selector_or_catalog_entries']} | 包含维度/拓扑/放置控制，注册不等于训练 |",
        f"| 运行证据 | {len(runs)} | 包括副本、trial及归档，保留出处，不冒充去重后的训练次数 |",
        f"| 指标证据 | {len(metrics)} | 逐率/逐条件，可能包含复本和诊断子集 |",
        f"| 无法自动读取的文件 | {len(issues)} | 见SCAN_ISSUES.json，不判为模型失败 |",'',
        '完整表：[家族/实验](ALL_EXPERIMENTS.md)、[逐入口](ALL_BLOCKS.md)、[运行配置](RUNS.csv)、[代表结果](FAMILY_RESULTS.md)。','',
        '关键边界：原选点成绩与最终随机重测/共同Mask重放分开；缺UA不补造；未知Missing不按日期猜测。',
        '额外保存了旧源码hash证明和所有Optuna/归档证据。超32MiB汇总未自动载入，其目录及已有文档仍入表。','',
        '剩余风险：源报告本身可能过期；未记录协议/环境/结束状态的旧运行不能补造；不同服务器的副本不能仅凭相同配置完全去重。']
    (output/'RESULT.md').write_text('\n'.join(result_lines)+'\n')
    lines=['# 全部 Block 与调参实验总账','','INTERNAL DIAGNOSTIC ONLY',
        '',f"盘点时间：{manifest['generated_at']}；源码：`{manifest['source_git_sha']}`。",
        '',f"共覆盖 **{len(project_rows)} 个实验目录**、**{len(families)} 条语义家族**、**{counts['selector_or_catalog_entries']} 个注册/配置入口**。",
        f"另有 {len(runs)} 条叶级运行证据、{len(metrics)} 条结构化指标证据；它们不是去重后的独立训练次数。",
        '', '## 使用说明','',
        '- `EXPERIMENTS.csv`：每个实验目录一行；不会遗漏只有文档/诊断的目录。',
        '- `RUNS.csv`：能找到运行配置的叶级记录，保留逐 seed/fold/结构/超参数/环境/协议。',
        '- `METRICS.csv`：逐条件/逐率原始指标行；保留JSON指针，不将Test-best和paired拼成同一成绩。',
        '- `BLOCK_CATALOG.csv`：全部注册机制/配置入口；注册不等于已训练。',
        '- `SOURCE_TABLE_ROWS.csv`：所有发现的Markdown表格行，含来源及行号，覆盖无法统一解析的历史汇总。',
        '- `ALL_BLOCKS.html`：可搜索的本地总表，不需要服务器或网络。',
        '', '## 不可混用的边界','',
        '- 新 `gcnet-random-mask-v1`、旧 `conversation-mask-v2`、legacy-fixed及unknown分别标记，不作统一最高分排名。',
        '- Missing未显式记录且没有精确执行源码证明时标unknown，不能按日期或当前默认值猜。',
        '- ACC/UA/W-F1的百分数单位统一；Macro-F1不冒充UA。缺失UA不补造。',
        '- 八率均值只从已保存八率同名指标求等权平均；单率/子集/三种子汇总不会填进叶级八率列。',
        '- 完成状态来自记录声明；只有结果、没有完成证据时写completion_unknown。仅smoke不列为正式完成。',
        '- 相同配置在本地和远端可能有多份证据，不凭相同config自动认定同一次训练。',
        '- 旧结果保留；早期launch/not-trained声明可能已过期，应查看后续完成报告。',
        '', '## Block/训练家族表','',
        '| 家族 | 类型 | 机制/职责 | 调整项 | 实验目录数 | 已有代表结果（不跨协议比较） |',
        '|---|---|---|---|---:|---|']
    for f in families:
        clean=lambda x:str(x).replace('|','/').replace('\n',' ')[:500]
        lines.append('| '+' | '.join([clean(f['name']),clean(f.get('category','')),
            clean(f.get('mechanism','')),clean(', '.join(map(str,f.get('tunable_parameters',[])))),
            str(len(f.get('experiment_dirs',[]))),clean(dumps(f.get('representative_results',[])))])+' |')
    lines += ['', '## 全部实验目录表','',
        '| 实验目录 | 家族 | 数据集 | Seed | Missing | 运行证据 | 指标证据 | 状态 |',
        '|---|---|---|---|---|---:|---:|---|']
    for p in project_rows:
        clean=lambda x:str(x).replace('|','/').replace('\n',' ')
        lines.append('| '+' | '.join(clean(p[k]) for k in ('experiment','families','datasets','seeds',
            'missing_protocols','run_evidence_rows','metric_evidence_rows','statuses'))+' |')
    (output/'ALL_EXPERIMENTS.md').write_text('\n'.join(lines)+'\n')
    render_html(output,project_rows,runs,families,catalog.get('methods',[]),counts)
    for name in ('METRICS.csv','SOURCE_TABLE_ROWS.csv','RUN_RECORDS.json'):
        with (output/name).open('rb') as src, gzip.GzipFile(filename=str(output/(name+'.gz')),mode='wb',mtime=0) as dst:
            __import__('shutil').copyfileobj(src,dst)
    print(dumps(counts))


def render_html(output,projects,runs,families,methods,counts):
    payload=dumps(dict(projects=projects,runs=[{k:v for k,v in r.items() if k != 'config'}
        for r in runs],families=families,methods=methods)).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><title>OSRAM 全部Block与实验总账</title>
<style>body{font:14px system-ui;margin:24px;color:#17212b}input,select{padding:9px;margin:5px}table{border-collapse:collapse;width:100%}th,td{padding:8px;border:1px solid #dde2e7;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#edf2f7}td{max-width:450px;overflow-wrap:anywhere}small{color:#526171}.wrap{overflow:auto;max-height:77vh}button{padding:9px}</style>
<h1>全部 Block 与调参实验总账</h1><p>INTERNAL DIAGNOSTIC ONLY。新旧Missing/环境/选点分开；空白不是零分；配置入口不是独立方法。</p>
<p id="stats"></p><select id="view"><option value="projects">全部实验目录</option><option value="runs">叶级运行记录</option><option value="families">Block与训练家族</option><option value="methods">注册/配置入口</option></select>
<input id="q" placeholder="搜索 Nested / B6 / seed / gcnet-random-mask-v1 …" size="65"><button id="prev">上一页</button><button id="next">下一页</button><small id="n"></small>
<div class="wrap"><table><thead id="head"></thead><tbody id="body"></tbody></table></div>
<script>const data=PAYLOAD;let page=0;const keys={projects:['experiment','families','datasets','seeds','missing_protocols','statuses','run_evidence_rows','metric_evidence_rows'],runs:['project','families','method_id','dataset','seed','fold','epoch_recorded','epochs_planned','learning_rate','weight_decay','batch_size','tunable_config','missing_protocol','checkpoint_selection','status','ACC','UA','WF1','environment','source_file'],families:['name','category','mechanism','tunable_parameters','experiment_dirs','status','protocol','representative_results'],methods:['method_id','display_name','canonical_family','catalog_batches','insertion_point','implementation_status','training_status','implementation_files']};
const el=id=>document.getElementById(id),s=v=>v==null?'':typeof v==='object'?JSON.stringify(v):String(v);function draw(){let view=el('view').value,q=el('q').value.toLowerCase(),rows=data[view].filter(r=>JSON.stringify(r).toLowerCase().includes(q));let cols=keys[view].filter(k=>rows.some(r=>k in r));if(!cols.length)cols=Object.keys(rows[0]||{}).slice(0,12);page=Math.min(page,Math.max(0,Math.ceil(rows.length/100)-1));el('head').replaceChildren();let tr=document.createElement('tr');cols.forEach(k=>{let th=document.createElement('th');th.textContent=k;tr.append(th)});el('head').append(tr);el('body').replaceChildren();rows.slice(page*100,page*100+100).forEach(r=>{let tr=document.createElement('tr');cols.forEach(k=>{let td=document.createElement('td');td.textContent=s(r[k]);tr.append(td)});el('body').append(tr)});el('n').textContent=`${rows.length}条，${page+1}/${Math.max(1,Math.ceil(rows.length/100))}页`}
el('view').onchange=el('q').oninput=()=>{page=0;draw()};el('prev').onclick=()=>{page=Math.max(0,page-1);draw()};el('next').onclick=()=>{page++;draw()};draw();</script>'''
    page=page.replace('PAYLOAD',payload)
    (output/'ALL_BLOCKS.html').write_text(page)


if __name__=='__main__':main()
