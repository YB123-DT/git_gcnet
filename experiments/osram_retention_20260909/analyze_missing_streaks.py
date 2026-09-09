"""Reconstruct last-observation-anchored gaps from existing retention records only."""
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def correlation(x,y):
    if len(x)<3: return None
    mx,my=mean(x),mean(y)
    dx=[v-mx for v in x];dy=[v-my for v in y]
    den=math.sqrt(sum(v*v for v in dx)*sum(v*v for v in dy))
    return sum(a*b for a,b in zip(dx,dy))/den if den else None


def ranks(x):
    result=[0.]*len(x);order=sorted(range(len(x)),key=x.__getitem__)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and x[order[j]]==x[order[i]]: j+=1
        for k in order[i:j]: result[k]=(i+j-1)/2
        i=j
    return result


def spearman(x,y): return correlation(ranks(x),ranks(y))


def bucket(n): return '1' if n==1 else '2-3' if n<=3 else '4-7' if n<=7 else '8+'


def reconstruct(records,metadata):
    tracks=defaultdict(list)
    no_history=set()
    for r in records:
        if r['status']=='NO_HISTORY':
            no_history.add((r['sample_id'],r['target_modality'],r['time_index']))
            continue
        anchor=r['time_index']-r['history_distance']
        tracks[(r['sample_id'],r['target_modality'],anchor,r['head'])].append(r)
    heads=[]
    for (sample,modality,anchor,head),rs in tracks.items():
        rs.sort(key=lambda r:r['time_index'])
        length=len(rs)
        if [r['history_distance'] for r in rs]!=list(range(1,length+1)):
            raise ValueError('Incomplete/duplicate gap records; cannot infer cumulative damage')
        continuity=max([abs(a['err_post']-b['err_pre']) for a,b in zip(rs,rs[1:])]+[0.])
        decay=sum(r['decay_damage'] for r in rs)
        write=sum(r['write_damage'] for r in rs)
        change=rs[-1]['err_post']-rs[0]['err_pre']
        closure=abs(change-decay-write)
        if continuity>1e-6 or closure>1e-5: raise ValueError('Telescoping identity failed')
        overlaps=[r['max_key_overlap'] for r in rs if r.get('max_key_overlap') is not None]
        heads.append(dict(dataset=metadata['dataset'],seed=metadata['seed'],rate=metadata['rate'],
            sample_id=sample,modality=modality,anchor_time=anchor,head=head,
            start_time=rs[0]['time_index'],end_time=rs[-1]['time_index'],gap_length=length,
            distance_bucket=bucket(length),boundary='observed_or_end_unresolved',
            cumulative_write=write,cumulative_decay=decay,error_start=rs[0]['err_pre'],
            error_end=rs[-1]['err_post'],error_change=change,
            write_per_step=write/length,decay_per_step=decay/length,
            overlap_mean=mean(overlaps) if overlaps else None,
            overlap_max=max(overlaps) if overlaps else None,
            overlap_sum=sum(overlaps) if overlaps else None,
            overlap_steps=len(overlaps),closure_error=closure,continuity_error=continuity))
    groups=defaultdict(list)
    for h in heads: groups[(h['sample_id'],h['modality'],h['anchor_time'])].append(h)
    gaps=[]
    expected_heads={r['head'] for r in records if r['status']=='retention'}
    for hs in groups.values():
        if {h['head'] for h in hs}!=expected_heads or len({h['gap_length'] for h in hs})!=1:
            raise ValueError('Unequal head coverage')
        row={k:v for k,v in hs[0].items() if k not in ('head',)}
        for k in ('cumulative_write','cumulative_decay','error_start','error_end','error_change',
                  'write_per_step','decay_per_step','overlap_mean','overlap_sum'):
            vals=[h[k] for h in hs if h[k] is not None];row[k]=mean(vals) if vals else None
        for k in ('overlap_max','closure_error','continuity_error'):
            vals=[h[k] for h in hs if h[k] is not None];row[k]=max(vals) if vals else None
        row['head_count']=len(hs);gaps.append(row)
    return heads,gaps,len(no_history)


def write_csv(path,rows):
    if not rows: return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    root=Path(__file__).resolve().parent
    allheads=[];allgaps=[];trends=[];buckets=[];excluded=0
    for p in sorted((root/'raw').glob('*/metadata.json')):
        meta=json.loads(p.read_text());raw=p.parent/'retention.jsonl'
        with (raw.open() if raw.exists() else gzip.open(str(raw)+'.gz','rt')) as f:
            records=[json.loads(x) for x in f]
        if len(records)!=meta['records']: raise ValueError('Record count mismatch')
        heads,gaps,n=reconstruct(records,meta);allheads+=heads;allgaps+=gaps;excluded+=n
        row={k:meta[k] for k in ('dataset','seed','rate')};row['gaps']=len(gaps)
        length=[g['gap_length'] for g in gaps]
        for metric in ('cumulative_write','cumulative_decay','error_change','write_per_step'):
            values=[g[metric] for g in gaps]
            row[metric+'_pearson']=correlation(length,values)
            row[metric+'_spearman']=spearman(length,values)
        conv=defaultdict(list)
        for g in gaps: conv[g['sample_id']].append(g)
        cs=[spearman([g['gap_length'] for g in gs],[g['cumulative_write'] for g in gs]) for gs in conv.values()]
        cs=[c for c in cs if c is not None]
        row['eligible_conversations']=len(cs);row['positive_conversations']=sum(c>0 for c in cs)
        row['conversation_mean_spearman']=mean(cs) if cs else None
        trends.append(row)
        groups=defaultdict(list)
        for g in gaps: groups[(g['modality'],g['distance_bucket'])].append(g)
        for (mod,b),gs in groups.items():
            r={k:meta[k] for k in ('dataset','seed','rate')}
            r.update(modality=mod,bucket=b,gaps=len(gs),mean_gap_length=mean(g['gap_length'] for g in gs))
            for metric in ('cumulative_write','cumulative_decay','error_change','write_per_step','overlap_mean','overlap_max','overlap_sum'):
                vals=[g[metric] for g in gs if g[metric] is not None]
                r[metric+'_mean']=mean(vals) if vals else None
                r[metric+'_median']=median(vals) if vals else None
            buckets.append(r)
    for name,rows in [('streak_heads',allheads),('streak_gaps',allgaps),('streak_buckets',buckets),('streak_trends',trends)]:
        write_csv(root/(name+'.csv'),rows)
    print(json.dumps(dict(runs=len(trends),head_streaks=len(allheads),gaps=len(allgaps),
        excluded_no_history_queries=excluded,max_closure=max(h['closure_error'] for h in allheads),
        positive_length_write_spearman=sum((t['cumulative_write_spearman'] or 0)>0 for t in trends),
        positive_length_per_step_spearman=sum((t['write_per_step_spearman'] or 0)>0 for t in trends))))


if __name__=='__main__':main()
