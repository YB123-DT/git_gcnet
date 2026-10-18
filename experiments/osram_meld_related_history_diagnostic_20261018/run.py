"""Three finite frozen B1/B2 diagnostics; no optimizer or mask sampler calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.osram_meld_optimizer_screen_20261018 import run as screen
from experiments.osram_meld_block2_auto_20261014.scheduler import write_json

KINDS = ('B1', 'B2')
RATES = tuple(i/10 for i in range(8))
COMMON_RATE = .1
MATCHED_EPOCH = 13
MATCHED_SOURCE_RATE = .5
SOURCE = Path('/home/qinjinghui/remote_experiments/osram_meld_related_history_20261018/runs_v1')
GPU_UUIDS = {3:'GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'}


def assert_trajectory(summary, last, checkpoint, expected_epoch):
    if (summary['status'] != 'complete' or last['identity'] != summary['identity']
            or last['next_epoch'] != 50 or len(last['history']) != 50
            or [r['epoch'] for r in last['history']] != list(range(1,51))
            or checkpoint['epoch'] != expected_epoch or not 1 <= expected_epoch <= 50
            or checkpoint['config'] != summary['config']
            or checkpoint['related_history'] != summary['branch']):
        raise ValueError('Checkpoint is not from the required complete original trajectory')


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checked_reference(source, reference):
    path = (source / reference['path']).resolve()
    if not path.is_relative_to(source.resolve()) or screen.original.sha(path) != reference['sha256']:
        raise ValueError('Protected source reference integrity mismatch')
    return path


def check_sources():
    from experiments.osram_meld_related_history_20261018 import run as training
    training.check_source()
    manifest = json.loads(Path(__file__).with_name('SOURCE_MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        if screen.original.sha(ROOT / name) != digest:
            raise ValueError('Frozen diagnostic source changed: ' + name)
    return manifest


def aggregate_rates(rows):
    if set(rows) != {str(r) for r in RATES}:
        raise ValueError('Require all eight exact-mask evaluations')
    conditions = tuple(next(iter(rows.values()))['metrics'])
    result = {}
    for scope, rates in (('mean',RATES),('high',RATES[5:])):
        result[scope] = {c:{m:sum(rows[str(r)]['metrics'][c][m] for r in rates)/len(rates)
            for m in ('ACC','UA','WF1','CE')} for c in conditions}
    changes = tuple(next(iter(rows.values()))['changes'])
    result['changes_total'] = {c:{key:sum(rows[str(r)]['changes'][c][key] for r in RATES)
        for key in ('corrections','harms','prediction_changes','both_correct','both_wrong')}
        for c in changes}
    result['norm_means'] = {key:sum(rows[str(r)]['norms'][key]['mean'] for r in RATES)/8
        for key in next(iter(rows.values()))['norms']}
    return result


def render(summary):
    lines = ['# B1/B2 frozen diagnostic', '', 'INTERNAL DIAGNOSTIC ONLY — Test-oracle seed66.', '',
        'No training, new seeds or U. Original GCNet masks replayed exactly; source files read-only.',
        'Off is the SAME trained checkpoint without its new residual, not original Flat weights.', '',
        '## 1. Whole-branch contribution', '',
        '| Model | condition | ACC | UA | W-F1 | CE |', '|---|---|---:|---:|---:|---:|']
    for kind, row in summary['models'].items():
        for condition, metrics in row['mean'].items():
            lines.append(f'| {kind} | {condition} | ' + ' | '.join(
                f'{metrics[m]:.6f}' for m in ('ACC','UA','WF1','CE')) + ' |')
        changes = row['changes_total']['off_to_on']
        lines += ['', f'{kind} off→on: corrections{changes["corrections"]}, harms{changes["harms"]}, '
            f'changed predictions{changes["prediction_changes"]}; counts sum8rate evaluations, not unique utterances.',
            'Mean norm statistics: `' + json.dumps(row['norm_means'],sort_keys=True) + '`.']
    lines += ['', '## 2. Fixed epoch and mask (rate0.1)', '',
        'Common mask: B1 rate0.1 selected epoch13. B2 epoch13 weights recovered read-only from',
        'its rate0.5 BEST and verified against final full-state trajectory references.',
        '| Readout | epoch | ACC | UA | W-F1 | CE |', '|---|---:|---:|---:|---:|---:|']
    for key, row in summary['common']['cases'].items():
        metrics = row['metrics']['on']
        lines.append(f'| {key} | {row["epoch"]} | ' + ' | '.join(
            f'{metrics[m]:.6f}' for m in ('ACC','UA','WF1','CE')) + ' |')
    for name, row in summary['common']['comparisons'].items():
        lines += ['', name + ': `' + json.dumps(row,sort_keys=True) + '`.']
    lines += ['', '## 3. Learned S and S=0.5', '',
        'S metrics use only valid strict-past8 pairs, same-class labels scored OFFLINE.',
        'All relation scores below are fractions0–1. No threshold/calibration fitting.',
        '| rate | pairs | same prevalence | mean S | balanced ACC | AUROC | AUPRC | BCE | Brier | ECE10 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    entries = [(r,row['relation']) for r,row in summary['models']['B2']['rates'].items()]
    entries.append(('pooled-rate-evaluations',summary['relation_pooled']))
    for rate, row in entries:
        values = [row['prevalence'],row['probability']['mean'],row['balanced_accuracy'],
            row['auroc'],row['auprc'],row['bce'],row['brier'],row['ece']]
        lines.append(f'| {rate} | {row["N"]} | ' + ' | '.join(
            'undefined' if v is None else f'{v:.6f}' for v in values) + ' |')
    lines += ['', 'Full distributions, fixed-bin calibration, S=.5 baseline and retrieval-weighted',
        'secondary scores are in RESULTS.json; individual pair/prediction NPZ files stay remote.',
        'S=.5 replay sets D=0,rho=.5 for nonempty history ONLY; E,R,Local,Base,Gap and CHM unchanged.',
        'On-vs-half task metrics and corrections/harms are listed above/in per-rate JSON.', '',
        '## Boundaries', '',
        '- Whole-branch on/off is not an isolation of E or attentionR: its MLP also sees Local/Base/Gap.',
        '- S=.5 changes an internal training-time distribution: sensitivity analysis, NOT a new model score.',
        '- A learned relation classifier and useful task evidence are different claims; do not equate them.',
        '- Pooled pairs repeat utterance identities across rates and share conversations; no independence/significance claim.',
        '- Test labels enter offline metrics only, never an inference rule or a trained Gate.',
        '- No automatic multi-seed, U, threshold search or new training is authorized by this result.']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--diagnostic-commit', required=True)
    args = p.parse_args()
    manifest = check_sources()
    if args.output.exists() or args.output.resolve().is_relative_to(SOURCE.resolve()):
        raise FileExistsError('Require an independent new diagnostic output directory')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,GCNET_MELD_PICKLE=str(screen.original.DATASET),
        GCNET_DATASET_ROOT=str(screen.MODEL_SOURCE/'dataset'),GCNET_CACHE_ROOT=str(args.output/'feature_cache'))
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu, state_digest
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment, repair_uuid_rng
    import torch
    import numpy as np
    gpu = guard_gpu(args.gpu_uuid)
    if GPU_UUIDS.get(gpu['host_index']) != gpu['uuid']:
        raise ValueError('Diagnostic requires healthy physicalGPU3')
    environment = require_replication_environment(torch); repair_uuid_rng(torch)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if screen.original.sha(screen.original.DATASET) != screen.original.DATA_SHA:
        raise ValueError('Original MELD feature hash changed')
    trainer = screen.original.load_historical(screen.MODEL_SOURCE)
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_meld_fixed_rates_20261018.run import selected_result
    from experiments.osram_meld_related_history_20261018.run import load_file
    from experiments.osram_meld_related_history_diagnostic_20261018.evaluate import evaluate_rate
    from experiments.osram_meld_related_history_diagnostic_20261018.metrics import relation_metrics, paired_comparison
    module = load_file('diagnostic_frozen_related_history',ROOT/'gcnet_missing_m3/related_history.py')
    integration = load_file('diagnostic_frozen_related_integration',ROOT/'experiments/osram_meld_related_history_20261018/integration.py')
    args.output.mkdir(parents=True)
    summary = dict(status='running',training=False,new_seeds=False,diagnostic_commit=args.diagnostic_commit,
        manifest=manifest,environment=environment,gpu=gpu,pid=os.getpid(),started_unix=time.time(),
        protocol='INTERNAL DIAGNOSTIC ONLY; exact Test-oracle selected masks; no training',models={})
    input_hashes = {}
    def remember(path):
        input_hashes[str(path)] = screen.original.sha(path)
    models = {}; trajectories = {}; sources = {}
    tick = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    write_json(args.output/'RESULTS.json',summary)
    try:
        for kind in KINDS:
            source = SOURCE/kind
            sources[kind] = json.loads((source/'SUMMARY.json').read_text())
            remember(source/'SUMMARY.json'); remember(source/'last_training.pt')
            trajectories[kind] = torch.load(source/'last_training.pt',map_location='cpu',weights_only=False)
            if sources[kind]['config'] != screen.make_config(screen.source_config(),'O6'):
                raise ValueError('Unexpected source configuration')
        cfg = trainer.TrainConfig(**sources['B1']['config'])
        _,_,test,ad,td,vd = get_loaders(audio_root='MELD-pickle',text_root='MELD-pickle',video_root='MELD-pickle',
            dataset='MELD',num_folder=1,batch_size=cfg.batch_size,num_workers=0,seed=cfg.seed,
            validation_fraction=cfg.validation_fraction,evaluation_protocol=cfg.evaluation_protocol)
        dimensions = (ad,td,vd); loader = test[0]

        def load_checkpoint(kind, rate):
            source = SOURCE/kind; path = source/f'best_miss_{str(rate).replace(".","p")}.pt'
            selected_result(source,rate)
            remember(path)
            checkpoint = torch.load(path,map_location='cpu',weights_only=False)
            last = trajectories[kind]
            assert_trajectory(sources[kind],last,checkpoint,checkpoint['epoch'])
            refs = last['best_references'][str(rate)]
            model_path = checked_reference(source,refs['model'])
            selection_path = checked_reference(source,refs['selection'])
            metadata = json.loads(selection_path.read_text())
            if metadata['checkpoint'] != {k:v for k,v in checkpoint.items() if k!='model'}:
                raise ValueError('Canonical BEST metadata differ from committed training trajectory')
            version = torch.load(model_path,map_location='cpu',weights_only=False)
            if state_hash(version) != state_hash(checkpoint['model']):
                raise ValueError('Canonical BEST weights differ from committed trajectory version')
            artifact_path = checked_reference(source,checkpoint['selection_artifact'])
            for ref in (model_path,selection_path,artifact_path):remember(ref)
            with np.load(artifact_path,allow_pickle=False) as data:
                saved = {k:data[k].copy() for k in data.files}
            proof = dict(epoch=checkpoint['epoch'],source_rate=rate,checkpoint=str(path),
                checkpoint_sha256=input_hashes[str(path)],committed_model_reference=refs['model'],
                selected_artifact=checkpoint['selection_artifact'],same_complete_trajectory=True)
            return checkpoint,saved,proof

        def evaluate(kind, checkpoint, saved, name, verify_saved):
            model = models[kind]
            model.load_state_dict(checkpoint['model'],strict=True)
            before = state_digest(model)
            result,artifacts,pairs = evaluate_rate(model,loader,cfg,dimensions,saved,trainer,verify_saved=verify_saved)
            if state_digest(model) != before or any(p.grad is not None for p in model.parameters()):
                raise ValueError('Frozen diagnostic modified weights or gradients')
            target = args.output/name
            target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target.with_suffix('.npz'),**artifacts)
            if pairs:
                np.savez_compressed(target.with_name(target.name+'_pairs.npz'),**pairs)
                result['relation'] = relation_metrics(pairs['probabilities'],pairs['labels'],pairs['attention'])
            return result,artifacts,pairs

        for kind in KINDS:
            model = _build_model(cfg,dimensions,torch.device('cuda:0'))
            integration.attach_related_history(model,kind=kind,module_class=module.RelatedHistoryResidual)
            models[kind] = model.eval().requires_grad_(False)
            row = dict(rates={})
            pooled = []
            for rate in RATES:
                checkpoint,saved,proof = load_checkpoint(kind,rate)
                result,_,pairs = evaluate(kind,checkpoint,saved,f'{kind}/rate_{str(rate).replace(".","p")}',True)
                result['source'] = proof
                row['rates'][str(rate)] = result
                if pairs:pooled.append(pairs)
                print(kind,'rate',rate,'epoch',checkpoint['epoch'],result['changes'],flush=True)
                summary['models'][kind] = row
                write_json(args.output/'RESULTS.json',summary)
            row.update(aggregate_rates(row['rates']))
            if kind=='B2':
                summary['relation_pooled'] = relation_metrics(
                    np.concatenate([v['probabilities'] for v in pooled]),
                    np.concatenate([v['labels'] for v in pooled]),
                    np.concatenate([v['attention'] for v in pooled]))

        # Same cohort/common mask. B1's on/off predictions already exist above.
        b1_checkpoint,common_saved,b1_proof = load_checkpoint('B1',COMMON_RATE)
        assert_trajectory(sources['B1'],trajectories['B1'],b1_checkpoint,MATCHED_EPOCH)
        with np.load(args.output/'B1/rate_0p1.npz',allow_pickle=False) as file:
            common_artifacts = {'B1_epoch13':{k:file[k].copy() for k in file.files}}
        cases = {'B1_epoch13':dict(summary['models']['B1']['rates']['0.1'],epoch=13)}
        for name, rate in (('B2_epoch10',COMMON_RATE),('B2_epoch13',MATCHED_SOURCE_RATE)):
            checkpoint,_,proof = load_checkpoint('B2',rate)
            expected_epoch = 10 if rate==COMMON_RATE else MATCHED_EPOCH
            assert_trajectory(sources['B2'],trajectories['B2'],checkpoint,expected_epoch)
            result,artifacts,_ = evaluate('B2',checkpoint,common_saved,'common/'+name,False)
            cases[name] = dict(result,epoch=expected_epoch,source=proof)
            common_artifacts[name] = artifacts
        comparisons = {}
        for name,reference,candidate in (
                ('model_at_epoch13','B1_epoch13','B2_epoch13'),
                ('B2_epoch_selection_10_to_13','B2_epoch10','B2_epoch13'),
                ('B1_epoch13_to_B2_selected10','B1_epoch13','B2_epoch10')):
            a,b = common_artifacts[reference],common_artifacts[candidate]
            for field in ('labels','conversation_ids','utterance_indices','speakers','availability'):
                np.testing.assert_array_equal(a[field],b[field])
            delta = paired_comparison(a['labels'],a['logits_on'],b['logits_on'])
            delta['metric_deltas'] = {m:cases[candidate]['metrics']['on'][m]-cases[reference]['metrics']['on'][m]
                for m in ('ACC','UA','WF1','CE')}
            comparisons[name] = delta
        summary['common'] = dict(rate=COMMON_RATE,mask_source=b1_proof,cases=cases,comparisons=comparisons)
        for path,digest in input_hashes.items():
            if screen.original.sha(Path(path)) != digest:
                raise ValueError('Protected input changed during diagnostic: '+path)
        summary.update(status='complete',source_inputs_unchanged=True,source_input_hashes=input_hashes,
            seconds=time.monotonic()-tick,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
            finished_unix=time.time())
        (args.output/'RESULT.md').write_text(render(summary))
    except BaseException as error:
        summary.update(status='failed',error=repr(error)); raise
    finally:
        write_json(args.output/'RESULTS.json',summary)


if __name__ == '__main__':
    main()
