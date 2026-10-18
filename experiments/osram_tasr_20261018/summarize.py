"""Fail-closed paired F0/F1/F2 analysis; reads predictions, never fits a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.osram_tasr_20261018.protocol import PATTERNS, RATES, TRAINING_PROTOCOL, validation_key

METRICS = ('ACC', 'UA', 'WF1', 'CE')
DATASETS = ('MELD', 'IEMOCAPSix')
MODES = {'F1': 'neutral', 'F2': 'real'}
FIELDS = ('labels', 'conversation_ids', 'utterance_indices', 'speakers', 'availability')


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def equal(left, right, label):
    if left != right:
        raise ValueError('Unpaired or invalid ' + label)


def array_equal(left, right, label):
    if not np.array_equal(left, right):
        raise ValueError('Unpaired prediction field ' + label)


def metrics(arrays, classes):
    from experiments.osram_persistent_patterns_20261018.evaluate import classification_metrics
    result = classification_metrics(arrays['labels'], arrays['logits'], classes)
    result['mask_sha256'] = hashlib.sha256(arrays['availability'].tobytes()).hexdigest()
    array_equal(arrays['predictions'], arrays['logits'].argmax(1), 'argmax prediction')
    return result


def check_reported(actual, reported, label):
    for key in (*METRICS, 'N'):
        if not math.isclose(float(actual[key]), float(reported[key]), rel_tol=0., abs_tol=1e-8):
            raise ValueError(f'Reported metric disagrees with predictions: {label}:{key}')
    equal(actual['mask_sha256'], reported['mask_sha256'], label + ':mask hash')


def pair(original, candidate):
    for field in FIELDS:
        array_equal(original[field], candidate[field], field)
    old = original['predictions'] == original['labels']
    new = candidate['predictions'] == candidate['labels']
    return dict(corrections=int((~old & new).sum()), harms=int((old & ~new).sum()),
                both_correct=int((old & new).sum()), both_wrong=int((~old & ~new).sum()), N=len(old))


def load_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        result = {key: source[key].copy() for key in source.files}
    if not set((*FIELDS, 'logits', 'predictions')).issubset(result):
        raise ValueError('Missing required prediction arrays: ' + str(path))
    n = len(result['labels'])
    if any(len(value) != n for value in result.values()):
        raise ValueError('Prediction array lengths differ: ' + str(path))
    return result


def average(conditions, chosen):
    return {tag: {m: math.fsum(conditions[r]['scores'][tag][m] for r in chosen) / len(chosen)
                  for m in METRICS} for tag in ('F0', 'F1', 'F2')}


def summarize_diagnostics(evaluation):
    maxima = {}
    stats = []
    for group in ('random', 'persistent'):
        for condition in evaluation[group]['conditions'].values():
            for batch in condition.get('diagnostics', []):
                for name, value in batch.get('source_tracking', {}).items():
                    if not math.isfinite(float(value)):
                        raise ValueError('Nonfinite source reconstruction diagnostic')
                    maxima[name] = max(maxima.get(name, 0.), float(value))
                stats.append((batch.get('valid', 0), batch.get('tasr', {})))
    total = sum(weight for weight, _ in stats)
    means = {}
    for name in {key for _, row in stats for key in row}:
        entries = [(weight, np.asarray(row[name], dtype=float)) for weight, row in stats if name in row]
        if not entries or any(not np.isfinite(value).all() for _, value in entries):
            raise ValueError('Missing/nonfinite TASR diagnostics: ' + name)
        if name.endswith('_min'):
            value = min(float(value) for _, value in entries)
        elif name.endswith('_max'):
            value = max(float(value) for _, value in entries)
        else:
            denominator = sum(weight for weight, _ in entries)
            if denominator <= 0:
                continue
            value = sum(weight * value for weight, value in entries) / denominator
            value = value.tolist()
        means[name] = value
    required = ('memory_sum_max_abs_error', 'memory_sum_max_relative_error',
                'gap_sum_max_abs_error', 'gap_sum_max_relative_error')
    missing = [name for name in required if name not in maxima]
    missing += [name for name in ('target_source_read_norm', 'target_source_gamma_mean') if name not in means]
    return dict(source_tracking_max=maxima, batch_valid_weighted_statistics=means,
                missing=missing, status='pending' if missing else 'complete',
                diagnostic_batch_valid_sum=total,
                interpretation='Batch valid-count weighted descriptive statistics, not source utility; '
                    'averaged batch quantiles are not global quantiles. Reconstruction errors use maxima.')


def check_job(summary, history, dataset, mode):
    equal(summary['phase'], 'train', 'phase')
    equal(summary['dataset'], dataset, 'dataset')
    equal(summary['mode'], mode, 'mode')
    for flag in ('backbone_unchanged', 'zero_adapter_logits_bit_exact'):
        equal(summary.get(flag), True, flag)
    equal(summary.get('epoch'), 30, '30 completed epochs')
    equal(len(history), 30, '30 recorded epochs')
    equal(summary['identity']['training'], TRAINING_PROTOCOL, 'pre-registered training protocol')
    case = summary['identity']['case']
    equal(case['dataset'], dataset, 'case dataset')
    equal(case['seed'], 66, 'seed')
    equal(case['fold'], 1 if dataset == 'MELD' else 5, 'fold')
    equal(case['anchor_rate'], '0.3', 'single fixed anchor rate')
    equal(case['classes'], 7 if dataset == 'MELD' else 6, 'class count')
    equal(summary['anchor_parity']['checks']['selected_logit_max_abs_error'], 0, 'historical anchor parity')
    equal(set(summary['validation_mask_hashes']), set(RATES), 'validation mask bank')
    if not summary['trainable_parameters'] or any(not name.startswith('osram.tasr.') for name in summary['trainable_parameters']):
        raise ValueError('Unfrozen original parameters or empty trainable parameter list')
    best = None
    selected = None
    for epoch, report in enumerate(history, 1):
        equal(report['epoch'], epoch, 'consecutive epoch order')
        equal(report['batches'], report['optimizer_steps'], 'one optimizer step per batch')
        if report['optimizer_steps'] <= 0:
            raise ValueError('Empty optimizer budget')
        key = validation_key(report['validation'])
        if not np.allclose(key, report['validation_key'], rtol=0, atol=1e-10):
            raise ValueError('Stored validation key differs from metrics')
        if best is None or key > best:
            best, selected = key, epoch
    equal(summary['selected_epoch'], selected, 'validation-only selected epoch')
    if not np.allclose(best, summary['validation_best_key'], rtol=0, atol=1e-10):
        raise ValueError('Validation best key mismatch')


def method_report(summary, history, evaluation):
    counts = {p: sum(row['persistent_pattern_conversations'][p] for row in history) for p in PATTERNS}
    total = sum(counts.values())
    return dict(selected_epoch=summary['selected_epoch'], validation_best_key=summary['validation_best_key'],
        adapter_parameters=summary['adapter_parameters'], total_parameters=summary['total_parameters'],
        seconds=summary['seconds'], peak_memory_mib=summary['peak_memory_mib'],
        optimizer_steps=sum(row['optimizer_steps'] for row in history),
        persistent_pattern_conversations=counts,
        persistent_pattern_fractions={p: counts[p] / max(1, total) for p in PATTERNS},
        random_actual_missing_fraction_epoch_mean=math.fsum(row['random_actual_missing_fraction'] for row in history)/30,
        persistent_actual_missing_fraction_epoch_mean=math.fsum(row['persistent_actual_missing_fraction'] for row in history)/30,
        validation_curve=[dict(epoch=row['epoch'], validation=row['validation'], key=row['validation_key'],
            **{k:row[k] for k in ('loss','random_CE','persistent_CE','gradient_norm') if k in row}) for row in history],
        random_mask_trace=[row['original_random_mask_trace_sha256'] for row in history],
        validation_mask_hashes=summary['validation_mask_hashes'], diagnostics=summarize_diagnostics(evaluation),
        environment=summary['environment'], gpu=summary.get('gpu'), identity=summary['identity'],
        checkpoint_references=summary['checkpoint_references'])


def summarize_dataset(dataset, jobs):
    summaries, histories, evaluations = {}, {}, {}
    for tag, mode in MODES.items():
        path, summary = jobs[f'{dataset}_{mode}']
        history = read(path/'history.json')
        evaluation = read(path/'evaluation'/'RESULTS.json')
        equal(evaluation, summary['evaluation'], 'saved evaluation and summary')
        check_job(summary, history, dataset, mode)
        summaries[tag], histories[tag], evaluations[tag] = summary, history, evaluation
    first, second = summaries['F1'], summaries['F2']
    for field in ('case', 'training', 'source_hashes', 'split'):
        equal(first['identity'][field], second['identity'][field], 'F1/F2 ' + field)
    for field in ('validation_mask_hashes', 'checkpoint_references', 'adapter_parameters',
                  'total_parameters', 'trainable_parameters', 'environment'):
        equal(first[field], second[field], 'F1/F2 ' + field)
    for left, right in zip(histories['F1'], histories['F2']):
        for field in ('epoch','batches','optimizer_steps','original_random_mask_trace_sha256',
                      'persistent_pattern_conversations','random_actual_missing_fraction','persistent_actual_missing_fraction'):
            equal(left[field], right[field], 'F1/F2 training ' + field)
    result = dict(methods={tag:method_report(summaries[tag],histories[tag],evaluations[tag]) for tag in MODES},
                  paired_identity_verified=True)
    classes = first['identity']['case']['classes']
    for group, names in (('random', RATES), ('persistent', (*PATTERNS, 'ATV'))):
        conditions = {}
        for tag in MODES:
            equal(set(evaluations[tag][group]['conditions']), set(names), group + ' complete conditions')
        for name in names:
            scores, predictions = {}, {}
            for tag, mode in MODES.items():
                path, _ = jobs[f'{dataset}_{mode}']
                saved = evaluations[tag][group]['conditions'][name]
                arrays = {which:load_arrays(path/'evaluation'/f'{group}_{name.replace(".","p")}_{which}.npz')
                          for which in ('F0','adapter')}
                pair(arrays['F0'], arrays['adapter'])
                for which in ('F0','adapter'):
                    measured = metrics(arrays[which], classes)
                    check_reported(measured,saved[which],f'{dataset}/{mode}/{group}/{name}/{which}')
                    if which == 'F0' and 'F0' in predictions:
                        for field in (*FIELDS,'logits','predictions'):
                            array_equal(predictions['F0'][field], arrays[which][field], 'F0 across runs:' + field)
                    mapped = 'F0' if which == 'F0' else tag
                    scores[mapped], predictions[mapped] = measured, arrays[which]
                equal(pair(arrays['F0'],arrays['adapter']),saved['paired'],'reported corrections/harms')
            comparisons = {name:pair(predictions[old],predictions[new])
                           for name,old,new in (('F1_vs_F0','F0','F1'),('F2_vs_F0','F0','F2'),('F2_vs_F1','F1','F2'))}
            deltas = {name:{m:scores[new][m]-scores[old][m] for m in METRICS}
                      for name,old,new in (('F1_vs_F0','F0','F1'),('F2_vs_F0','F0','F2'),('F2_vs_F1','F1','F2'))}
            conditions[name] = dict(scores=scores, paired=comparisons, delta=deltas)
        result[group] = dict(conditions=conditions)
        scopes = (('mean', RATES), ('high', RATES[5:])) if group == 'random' else (('six', PATTERNS),('seven', (*PATTERNS,'ATV')))
        result[group].update({scope:average(conditions, chosen) for scope,chosen in scopes})
    mean = result['random']['mean']
    persistent = result['persistent']['six']
    random_delta = {m:mean['F2'][m]-mean['F0'][m] for m in METRICS}
    persistent_delta = {m:persistent['F2'][m]-persistent['F1'][m] for m in METRICS}
    persistent_original_delta = {m:persistent['F2'][m]-persistent['F0'][m] for m in METRICS}
    # A pre-registered engineering screen, not a statistical or semantic claim.
    result['verdict'] = dict(random_F2_minus_F0=random_delta, persistent_F2_minus_F1=persistent_delta,
        persistent_F2_minus_F0=persistent_original_delta,
        random_protection_pass=all(random_delta[m] >= -.3 for m in ('ACC','WF1')),
        persistent_F2_extra_value_over_F1=all(persistent_delta[m] >= 0 for m in ('ACC','UA','WF1'))
            and any(persistent_delta[m] > 0 for m in ('ACC','UA','WF1')),
        note='Six-pattern macro is descriptive; individual conditions remain primary. No multi-seed inference.')
    result['diagnostics_status'] = 'complete' if all(row['diagnostics']['status']=='complete' for row in result['methods'].values()) else 'pending'
    return result


def summarize(runs):
    runs = Path(runs)
    jobs, pending = {}, []
    for dataset in DATASETS:
        for mode in MODES.values():
            identifier = f'{dataset}_{mode}'
            path = runs/identifier
            if not (path/'SUMMARY.json').is_file():
                pending.append(dict(id=identifier,status='not-found'))
                continue
            summary = read(path/'SUMMARY.json')
            if summary.get('status') != 'complete':
                pending.append(dict(id=identifier,status=summary.get('status','unknown'),error=summary.get('error')))
                continue
            required = ['history.json','evaluation/RESULTS.json']
            required += [f'evaluation/{group}_{name.replace(".","p")}_{tag}.npz'
                         for group,names in (('random',RATES),('persistent',(*PATTERNS,'ATV')))
                         for name in names for tag in ('F0','adapter')]
            for relative in required:
                if not (path/relative).is_file():
                    pending.append(dict(id=identifier,status='missing-artifact',artifact=relative))
            jobs[identifier] = path, summary
    if pending:
        return dict(status='pending',pending=pending,protocol='INTERNAL DIAGNOSTIC ONLY',runs=str(runs))
    datasets = {dataset:summarize_dataset(dataset,jobs) for dataset in DATASETS}
    return dict(status='complete',protocol='INTERNAL DIAGNOSTIC ONLY',runs=str(runs),datasets=datasets,
        scope='seed66 MELD and IEMOCAPSix fold5 only; no new full five-fold or multi-seed training',
        limitations=['Frozen original anchor miss=.3 was selected historically by Test-oracle.',
            'One new adapter checkpoint is selected using validation random8 ACC, then CE, and reused for every test condition.',
            'IEMOCAP adapter-only validation holdout was previously seen by the frozen backbone; not independent end-to-end validation.',
            'Source slots are additive joint Value-column contributions, not pure-modality semantic or causal attribution.',
            'No baseline ranking against historical per-rate selected checkpoints is implied.'])


def render_markdown(result):
    lines = ['# TASR finite frozen-backbone screen', '', '**INTERNAL DIAGNOSTIC ONLY**', '']
    if result['status'] != 'complete':
        lines += ['Status: pending. No incomplete run participates in ranking.', '',
                  '```json', json.dumps(result.get('pending',[]),ensure_ascii=False,indent=2), '```']
        return '\n'.join(lines)+'\n'
    lines += [result['scope'], '', *['- '+value for value in result['limitations']], '',
              'F0 = same protected frozen Flat; F1 = source-neutral; F2 = real-source TASR.',
              'Every entry uses the same anchor and condition mask. Cells: ACC / UA / W-F1 / CE.', '']
    for dataset, data in result['datasets'].items():
        lines += ['## '+dataset, '', '| Condition | F0 | F1 | F2 | F2−F1 ACC / UA / W-F1 |',
                  '|---|---:|---:|---:|---:|']
        for group in ('random','persistent'):
            entries = [(f'{group} {name}', row['scores']) for name,row in data[group]['conditions'].items()]
            for scope in (('mean','high') if group=='random' else ('six','seven')):
                entries.append((group+' '+scope,data[group][scope]))
            for name, scores in entries:
                cells = [' / '.join(f'{scores[tag][m]:.3f}' for m in METRICS) for tag in ('F0','F1','F2')]
                delta = ' / '.join(f'{scores["F2"][m]-scores["F1"][m]:+.3f}' for m in ('ACC','UA','WF1'))
                lines.append('| '+name+' | '+' | '.join(cells)+' | '+delta+' |')
        lines += ['', '### Paired corrections / harms', '',
                  '| Condition | F1 vs F0 | F2 vs F0 | F2 vs F1 | N |', '|---|---:|---:|---:|---:|']
        for group in ('random','persistent'):
            for name,row in data[group]['conditions'].items():
                pairs = row['paired']
                cells = [f'{pairs[key]["corrections"]} / {pairs[key]["harms"]}' for key in ('F1_vs_F0','F2_vs_F0','F2_vs_F1')]
                lines.append('| '+group+' '+name+' | '+' | '.join(cells)+f' | {pairs["F2_vs_F1"]["N"]} |')
        lines += ['', '### Optimization and diagnostics', '']
        for tag, method in data['methods'].items():
            lines += [f'- {tag}: selected epoch {method["selected_epoch"]}/30; adapter parameters {method["adapter_parameters"]}; '
                      f'{method["optimizer_steps"]} optimizer steps; peak {method["peak_memory_mib"]:.1f} MiB; {method["seconds"]:.1f}s.',
                      f'- {tag} actual random missing (epoch mean): {method["random_actual_missing_fraction_epoch_mean"]:.6f}; '
                      'Persistent pattern counts: '+json.dumps(method['persistent_pattern_conversations'],sort_keys=True)+'.',
                      f'- {tag} source reconstruction maximum errors: '+json.dumps(method['diagnostics']['source_tracking_max'],sort_keys=True)+'.',
                      f'- {tag} diagnostic status: {method["diagnostics"]["status"]}; full gamma/read matrices, coverage and validation curves are in COMPARISON.json.']
        verdict = data['verdict']
        lines += ['', f'Random protection (F2 vs F0, ACC and W-F1 ≥−0.3pp): {verdict["random_protection_pass"]}.',
                  f'Descriptive Persistent-six F2 extra value over F1 (ACC/UA/W-F1 non-decreasing, at least one positive): '
                  f'{verdict["persistent_F2_extra_value_over_F1"]}.',
                  'This screen is not significance testing. Individual condition failures and trade-offs above must not be hidden.', '']
    return '\n'.join(lines)+'\n'


def write_outputs(output, result):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'COMPARISON.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    (output/'RESULT.md').write_text(render_markdown(result))
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(('dataset','group','condition','method','ACC','UA','WF1','CE','N',
                     'vs_F0_corrections','vs_F0_harms','F2_vs_F1_corrections','F2_vs_F1_harms'))
    if result['status']=='complete':
        for dataset,data in result['datasets'].items():
            for group in ('random','persistent'):
                for name,row in data[group]['conditions'].items():
                    for tag in ('F0','F1','F2'):
                        score = row['scores'][tag]
                        change = row['paired'].get(tag+'_vs_F0',{})
                        last = row['paired']['F2_vs_F1'] if tag=='F2' else {}
                        writer.writerow((dataset,group,name,tag,*[score[m] for m in METRICS],score['N'],
                            change.get('corrections',''),change.get('harms',''),last.get('corrections',''),last.get('harms','')))
    (output/'COMPARISON.csv').write_text(stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = summarize(args.runs)
    write_outputs(args.output,result)
    print(json.dumps(dict(status=result['status'],output=str(args.output)),sort_keys=True))
    if result['status']!='complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
