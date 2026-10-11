"""Summarize all predetermined subsets, without selecting a deployable rule."""
import json
import csv
from pathlib import Path

root=Path(__file__).resolve().parent
s=json.loads((root/'results/SUMMARY.json').read_text())
status=json.loads((root/'results/STATUS.json').read_text())
assert status['status']=='completed' and len(status['records'])==8
assert all(r['frozen_unchanged'] and r['archive_max_abs']<=1e-5 for r in status['records'])
macro={(r['intervention'],r['group'],r['rates']):r for r in s['macro']}
slots=list(dict.fromkeys(r['intervention'] for r in s['macro']))
assert len(slots)==31
prior=root.parent/'osram_nested_slot_delta_off_20261011/results/utterances.csv'
oldall=root.parent/'osram_nested_delta_off_missing_text_20261010/results/utterances.csv'
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))
old={(r['rate'],r['conversation'],r['utterance'],r['intervention']):float(r['off']) for r in rows(prior)}
alloff='Local+Base+Gap-A+Gap-T+Gap-V'
old.update({(r['rate'],r['conversation'],r['utterance'],alloff):float(r['off']) for r in rows(oldall)})
seen=0
for r in rows(root/'results/utterances.csv'):
    key=(r['rate'],r['conversation'],r['utterance'],r['intervention'])
    if key in old:
        assert abs(float(r['off'])-old[key])<=1e-5
        seen+=1
assert seen==len(old)
first=macro[(slots[0],'all','nonempty_rates')]
lines=['# Nested correction subset interventions','','INTERNAL DIAGNOSTIC ONLY','',
       'Completed: seed66, eight original100epoch per-rate Test-oracle BEST checkpoints.',
       'No training; one Memory scan per batch. Full plus31 predetermined OFF subsets.',
       'OFF restores raw evidence, not zero evidence. Original Local Skip/Flat/head retained.',
       'Single-slot and all-OFF predictions match prior diagnostics <=1e-5; Full matches',
       'original archive labels/masks/predictions. Eight model/checkpoint hashes unchanged.',
       'No configuration selection or mechanism claim from test outcomes.','',
       '## Overall W-F1 (%)','',
       '| OFF slots | Mean8 | High .5/.6/.7 | OFF-minus-Full pp | Corrections/harms OFF->Full |',
       '|---|---:|---:|---:|---:|',
       f"|None (Full)|{first['on_wf1']:.6f}|{macro[(slots[0],'all','high_missing')]['on_wf1']:.6f}|0|—|"]
for slot in slots:
    r=macro[(slot,'all','nonempty_rates')];h=macro[(slot,'all','high_missing')]
    lines.append(f"|{slot}|{r['off_wf1']:.6f}|{h['off_wf1']:.6f}|{-r['delta_pp']:+.6f}|{r['corrections']}/{r['harms']}|")
lines+=['','## Current observed pattern macro W-F1 (%)','',
        'Equal mean over nonempty rate groups; not whole-conversation fixed missing.',
        'T=Text; repeated utterances across rates are not independent or unique.','',
        '| OFF slots | A | T | V | AT | AV | TV | ATV |','|---|---:|---:|---:|---:|---:|---:|---:|']
groups=('A','T','V','AT','AV','TV','ATV')
lines.append('|None (Full)|'+'|'.join(f"{macro[(slots[0],g,'nonempty_rates')]['on_wf1']:.6f}" for g in groups)+'|')
for slot in slots:
    lines.append('|'+slot+'|'+'|'.join(f"{macro[(slot,g,'nonempty_rates')]['off_wf1']:.6f}" for g in groups)+'|')
lines+=['','Full per-rate/N/corrections/harms: results/per_rate.csv; per-utterance predictions:',
        'results/utterances.csv. Provenance/checkpoint hashes: results/STATUS.json.',
        'Conditional effects do not have to add; this is not separately retrained ablation.',
        'No additional training/sweeps; unchanged running capacity trials are separate.','']
(root/'RESULT.md').write_text('\n'.join(lines))
print('\n'.join(lines[:48]))
