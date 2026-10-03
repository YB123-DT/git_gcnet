"""Add only adjacent label-polarity relation to the existing offline Context Audit."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st
from collections import defaultdict
from sklearn.metrics import f1_score


def attach(rows):
    key=lambda r:(int(r['seed']),float(r['rate']),r['conversation_id'],int(r['utterance_index']))
    lookup={key(r):r for r in rows};assert len(lookup)==len(rows)
    output=[];excluded=dict(first=0,neutral_pair=0)
    for r in rows:
        s,rate,c,t=key(r)
        if t==0:excluded['first']+=1;continue
        assert (s,rate,c,t-1) in lookup,'Missing immediate predecessor; do not skip'
        previous=lookup[s,rate,c,t-1]
        y,yp=float(r['label']),float(previous['label'])
        if y==0 or yp==0:excluded['neutral_pair']+=1;continue
        output.append(dict(r,previous_label=yp,
            relation='same' if (y>0)==(yp>0) else 'opposite',
            boundary='near' if abs(float(r['local_prediction']))<=.25 else 'far'))
    return output,excluded


def score(rows):
    n=len(rows)
    y=[float(r['label'])>0 for r in rows]
    a=[float(r['local_prediction'])>0 for r in rows]
    b=[float(r['memory_prediction'])>0 for r in rows]
    l=100*f1_score(y,a,average='weighted',zero_division=0) if n else None
    m=100*f1_score(y,b,average='weighted',zero_division=0) if n else None
    return dict(n=n,local_wf1=l,memory_wf1=m,delta=None if not n else m-l,
        corrections=sum(x!=z and v==z for x,v,z in zip(a,b,y)),
        harms=sum(x==z and v!=z for x,v,z in zip(a,b,y)),
        no_text=sum('T' not in r['pattern'] for r in rows))


def macro(cells):
    valid=[c for c in cells if c['n']]
    seeds=sorted({c['seed'] for c in valid})
    result={k:sum(c[k] for c in valid) for k in ('n','corrections','harms','no_text')}
    result.update(cells=len(valid),seeds=len(seeds))
    for k in ('local_wf1','memory_wf1','delta'):
        values=[st.mean(c[k] for c in valid if c['seed']==s) for s in seeds]
        result[k]=st.mean(values) if values else None
        result[k+'_seed_sd']=st.stdev(values) if len(values)>1 else None
    result['no_text_percent']=100*result['no_text']/result['n'] if result['n'] else None
    return result


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    root=Path(__file__).resolve().parent;source=root/'results/utterances.csv'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==3*8*686
    selected,excluded=attach(rows)
    assert len(selected)+sum(excluded.values())==len(rows)
    groups=defaultdict(list)
    for r in selected:
        for pattern in ('ALL',r['pattern']):
            groups[int(r['seed']),float(r['rate']),r['boundary'],r['relation'],pattern].append(r)
    cells=[]
    for seed in (66,67,68):
        for rate in [i/10 for i in range(8)]:
            for boundary in ('near','far'):
                for relation in ('same','opposite'):
                    for pattern in ('ALL','A','T','V','AT','AV','TV','ATV'):
                        cells.append(dict(seed=seed,rate=rate,boundary=boundary,relation=relation,pattern=pattern,
                            **score(groups[seed,rate,boundary,relation,pattern])))
    aggregates=[]
    for boundary in ('near','far'):
        for pattern in ('ALL','A','T','V','AT','AV','TV','ATV'):
            subset=[c for c in cells if c['boundary']==boundary and c['pattern']==pattern]
            keys=[{(c['seed'],c['rate']) for c in subset if c['relation']==r and c['n']} for r in ('same','opposite')]
            common=keys[0]&keys[1]
            for relation in ('same','opposite'):
                # Main four cells preserve original macro protocol. Availability comparisons
                # restrict both relation groups to identical nonempty seed/rate support.
                chosen=[c for c in subset if c['relation']==relation and
                        (pattern=='ALL' or (c['seed'],c['rate']) in common)]
                aggregates.append(dict(boundary=boundary,relation=relation,pattern=pattern,**macro(chosen)))
    out=root/'adjacent_results';out.mkdir(exist_ok=False)
    write_csv(out/'utterances.csv',selected);write_csv(out/'by_seed_rate.csv',cells)
    write_csv(out/'aggregate.csv',aggregates)
    report=['# 相邻话语同／异极性：Context Audit 补充','',
        '纯离线；同一conversation相邻t−1,t，两端标签均非零。首句和包含中性的相邻对排除，不跳过、不跨对话。',
        'Local-off沿用上次Local-only（关闭Memory读值，Local路径保留），不是关闭Local。',
        '预测阈值>0；边界距离abs(Local预测)<=.25。每seed/rate同一批样本算两种W-F1，再rate平均、seed平均。',
        '计数跨seed/rate累加，非独立样本数。W-F1差值不是用合并样本或平均MSE计算。',
        f"排除首句{excluded['first']}次；中性相邻对{excluded['neutral_pair']}次；保留{len(selected)}次样本出现。",'',
        '|Local边界距离|关系|Local-only W-F1|Memory-on W-F1|Δpp|纠错|致错|N|当前无Text占比|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:
        if r['pattern']!='ALL':continue
        report.append(f"|{r['boundary']}|{r['relation']}|{r['local_wf1']:.3f}|{r['memory_wf1']:.3f}|{r['delta']:+.3f}|{r['corrections']}|{r['harms']}|{r['n']}|{r['no_text_percent']:.2f}%|")
    report+=['','## 相同当前availability：同极性／异极性的Memory增益','',
        '每行两组使用共同非空seed/rate支持；不存在的格子不填0。完整前后W-F1、计数、seed SD见aggregate.csv。','',
        '|边界|当前可见|同极性Δpp|异极性Δpp|异−同|同极性N|异极性N|共同seed-rate数|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for boundary in ('near','far'):
        for pattern in ('A','T','V','AT','AV','TV','ATV'):
            a,b=[next(r for r in aggregates if r['boundary']==boundary and r['pattern']==pattern and r['relation']==rel) for rel in ('same','opposite')]
            if a['delta'] is None:continue
            report.append(f"|{boundary}|{pattern}|{a['delta']:+.3f}|{b['delta']:+.3f}|{b['delta']-a['delta']:+.3f}|{a['n']}|{b['n']}|{a['cells']}|")
    report+=['','near=abs(Local预测)<=.25；far=>.25。same=同极性；opposite=异极性。',
        '只称相邻话语同／异极性，不等同真实情绪延续／转折；上一句标签不代表整个Memory状态。',
        'availability分层仅排查这一混杂，不能证明关系的因果作用。小格和类别构成会影响W-F1。',
        '同一语句跨seed/rate重复，未作独立样本显著性检验。Test-oracle内部标签辅助诊断。','']
    (out/'RESULT.md').write_text('\n'.join(report))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    (out/'AUDIT.json').write_text(json.dumps(dict(source_sha256=digest,source_rows=len(rows),
        selected=len(selected),excluded=excluded,training=False,inference=False),indent=2)+'\n')
    print('\n'.join(report))


if __name__=='__main__':main()
