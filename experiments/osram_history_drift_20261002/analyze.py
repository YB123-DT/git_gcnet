"""Summarize frozen history interventions without fitting or choosing a model."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

MEASURES=('delta_local','absmax_local','base_cos','base_abs','base_rel','gap_cos',
          'gap_abs','gap_rel','gap_a_cos','gap_t_cos','gap_v_cos','hidden_cos',
          'hidden_abs','hidden_rel','prediction_shift','prior_deleted_bit_count')
COUNTS={'anchors','nonzero_labels','flips','correct_to_wrong','wrong_to_correct','all_label_flips'}


def number(row,key):
    try: return float(row.get(key,float('nan')))
    except (TypeError,ValueError): return float('nan')


def summarize_rows(rows):
    result={'anchors':len(rows)}
    for key in MEASURES:
        values=[number(row,key) for row in rows]
        values=[v for v in values if math.isfinite(v)]
        result[key]=statistics.mean(values) if values else None
        result[key+'_n']=len(values)
    result['local_max']=max((number(r,'absmax_local') for r in rows),default=0.)
    result['nonzero_labels']=result['flips']=result['correct_to_wrong']=result['wrong_to_correct']=0
    result['all_label_flips']=0
    for row in rows:
        p1,p2=number(row,'pred1')>0,number(row,'pred2')>0
        result['all_label_flips']+=int(p1!=p2)
        label=number(row,'label')
        if label==0: continue
        result['nonzero_labels']+=1
        target=label>0
        result['flips']+=int(p1!=p2)
        result['correct_to_wrong']+=int(p1==target and p2!=target)
        result['wrong_to_correct']+=int(p1!=target and p2==target)
    result['flip_percent']=100*result['flips']/result['nonzero_labels'] if result['nonzero_labels'] else None
    result['all_label_flip_percent']=100*result['all_label_flips']/len(rows) if rows else None
    assert result['flips']==result['correct_to_wrong']+result['wrong_to_correct']
    return result


def rate_macro(rows):
    result={'rate_count':len(rows)}
    for key in set().union(*(r.keys() for r in rows)):
        values=[r[key] for r in rows if r.get(key) is not None]
        if key in COUNTS or key.endswith('_n'):
            result[key]=sum(values)
        elif key=='local_max':
            result[key]=max(values,default=0.)
        else:
            result[key]=statistics.mean(values) if values else None
            result[key+'_rates']=len(values)
    return result


def summarize_group(rows):
    groups=defaultdict(list)
    for row in rows: groups[row['rate']].append(row)
    rates={rate:summarize_rows(items) for rate,items in sorted(groups.items())}
    return dict(pooled=summarize_rows(rows),rate_macro=rate_macro(list(rates.values())),per_rate=rates)


def write_csv(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def analyze(root,output):
    status=json.loads((root/'status.json').read_text())
    assert status['status']=='complete' and len(status['records'])==16
    provenance=json.loads((root/'provenance.json').read_text())
    assert not provenance['smoke'] and provenance['training'] is False
    rows=[]
    for record in status['records']:
        assert record['model_state_before_sha256']==record['model_state_after_sha256']
        assert record['checkpoint_sha256']==record['checkpoint_after_sha256']
        assert record['identity_repeat_exact'] and record['prefix_allclose']
        assert record['local_absolute_max_error']<=1e-5
        assert record['local_relative_max_error']<=1e-6
        for name,digest in record.get('artifact_sha256',{}).items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
        file=root/record['anchors_file']
        assert hashlib.sha256(file.read_bytes()).hexdigest()==record['anchors_sha256']
        with file.open() as stream: current=list(csv.DictReader(stream))
        assert len(current)==record['anchors']
        for row in current:
            assert int(row['nearest_prior_change_distance'])>=1
            assert int(row['prior_deleted_bit_count'])>=1
            assert number(row,'absmax_local')<=1e-5
            assert number(row,'delta_local')<=1e-6
            for key in ('pred1','pred2','prediction_shift','delta_local','hidden_cos'):
                assert math.isfinite(number(row,key))
            assert abs(abs(number(row,'pred1')-number(row,'pred2'))-number(row,'prediction_shift'))<1e-6
        rows.extend(current)
    assert len({(r['split'],r['rate'],r['mode'],r['conversation_id'],r['utterance_index']) for r in rows})==len(rows)
    output.mkdir(parents=True,exist_ok=True)
    summary={'label':'INTERNAL frozen history diagnostic; inherited Test-oracle checkpoints; seed66',
             'anchor_exposures':len(rows),'splits':{}}
    flatrows=[]; detailed=[]
    for split in ('validation','test'):
        subset=[r for r in rows if r['split']==split]
        result={'by_mode':{},'matched_atv':{},'by_distance':{},'by_current_pattern':{},'mixed_prefix_groups':{}}
        for mode in ('A','T','V','mixed'):
            mode_rows=[r for r in subset if r['mode']==mode]
            result['by_mode'][mode]=summarize_group(mode_rows)
            for rate,stats in result['by_mode'][mode]['per_rate'].items():
                flatrows.append(dict(split=split,mode=mode,rate=rate,**stats))
            for label,lo,hi in [('1',1,1),('2',2,2),('3-4',3,4),('5+',5,100000)]:
                selected=[r for r in mode_rows if lo<=int(r['nearest_prior_change_distance'])<=hi]
                result['by_distance'][mode+':'+label]=summarize_group(selected)
                detailed.append(dict(split=split,mode=mode,group='distance',value=label,**summarize_rows(selected)))
            for pattern in ('A','T','V','AT','AV','TV','ATV'):
                selected=[r for r in mode_rows if r['current_pattern']==pattern]
                result['by_current_pattern'][mode+':'+pattern]=summarize_group(selected)
                detailed.append(dict(split=split,mode=mode,group='current_pattern',value=pattern,**summarize_rows(selected)))
        identity=lambda r:(r['rate'],r['conversation_id'],r['utterance_index'])
        keys=[{identity(r) for r in subset if r['mode']==m} for m in ('A','T','V')]
        common=set.intersection(*keys)
        for mode in ('A','T','V'):
            result['matched_atv'][mode]=summarize_group([r for r in subset if r['mode']==mode and identity(r) in common])
        for group in ('A','T','V','mixed'):
            selected=[r for r in subset if r['mode']=='mixed' and
                      (r['prior_deleted_modalities'] if len(r['prior_deleted_modalities'])==1 else 'mixed')==group]
            result['mixed_prefix_groups'][group]=summarize_group(selected)
        summary['splits'][split]=result
    (output/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    write_csv(output/'by_rate.csv',flatrows)
    write_csv(output/'by_distance_pattern.csv',detailed)
    fmt=lambda x:'—' if x is None else f'{x:.5f}'
    report=['# Frozen history drift — original cfg84 seed66','',
        'No training or model changes. Validation/test separate; inherited per-rate Test-oracle checkpoints.',
        'Only same-current/different-strict-history anchors. Base/Gap effective first512dimensions.',
        'Zero-vector cosine excluded with valid counts, not replaced by0or1.','',
        'Main numbers are unweighted macro means over rates with eligible anchors/nonzero vectors. Gap has no',
        'active slots at rate0; its macro therefore excludes that rate. Per-metric rate counts retained in JSON.',
        'Counts sum rate exposures,',
        'not independent samples. Flip% excludes neutral labels and uses original prediction>0 threshold.',
        'Local max is the largest absolute coordinate difference. Different layers use different scales;',
        'these numbers alone cannot establish excessive amplification or harmful drift.','']
    for split,result in summary['splits'].items():
        report += [f'## {split}: nominal deletion interventions','',
            '|Delete|N anchors|Rates|Local max|Base cosine drift|Gap cosine drift|Hidden cosine drift|Prediction shift|Flip%|Correct→wrong|Wrong→correct|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for mode,item in result['by_mode'].items():
            s=item['rate_macro']
            report.append(f"|{mode}|{s.get('anchors',0)}|{s['rate_count']}|{s.get('local_max',0):.3e}|{fmt(s.get('base_cos'))}|{fmt(s.get('gap_cos'))}|{fmt(s.get('hidden_cos'))}|{fmt(s.get('prediction_shift'))}|{fmt(s.get('flip_percent'))}|{s.get('correct_to_wrong',0)}|{s.get('wrong_to_correct',0)}|")
        report+=['',f'### {split}: matched A/T/V anchors','',
            '|Delete|N anchors|Prediction shift|Flip%|Mean prior deleted bits|',
            '|---|---:|---:|---:|---:|']
        for mode,item in result['matched_atv'].items():
            s=item['rate_macro']; report.append(f"|{mode}|{s.get('anchors',0)}|{fmt(s.get('prediction_shift'))}|{fmt(s.get('flip_percent'))}|{fmt(s.get('prior_deleted_bit_count'))}|")
        report+=['',f'### {split}: nearest deletion distance, mixed intervention','',
            '|Distance|N anchors|Prediction shift|Flip%|Mean prior deleted bits|',
            '|---|---:|---:|---:|---:|']
        for distance in ('1','2','3-4','5+'):
            s=result['by_distance']['mixed:'+distance]['rate_macro']
            report.append(f"|{distance}|{s.get('anchors',0)}|{fmt(s.get('prediction_shift'))}|{fmt(s.get('flip_percent'))}|{fmt(s.get('prior_deleted_bit_count'))}|")
        report.append('')
    report+=['## Limits','',
        'Nominal modes can have different eligible anchors/deletion counts; matched A/T/V table controls',
        'current sample/rate but not number or timing of prior deletions. Mixed prefix group breakdown,',
        'individual Gap drifts, pooled statistics and current-pattern strata are in SUMMARY.json/CSV.',
        'Distance means nearest prior deletion under repeated deletions, not a single isolated perturbation.',
        'Cosine can stay unchanged despite magnitude drift; norm differences are retained in the tables.',
        'Historical information differs genuinely; flips may correct or harm. No labels used to fit or select.',
        'Single seed and Test-oracle origins preclude formal generalization/significance claims.','']
    (output/'RESULT.md').write_text('\n'.join(report))
    print('\n'.join(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); analyze(args.root,args.output)
