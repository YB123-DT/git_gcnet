"""Render fixed per-slot intervention outputs; never select checkpoints."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
s = json.loads((root / 'results/SUMMARY.json').read_text())
status = json.loads((root / 'results/STATUS.json').read_text())
assert status['status'] == 'completed' and len(status['records']) == 8
assert all(r['frozen_unchanged'] and r['archive_max_abs'] <= 1e-5 for r in status['records'])
slots = ['Local', 'Base', 'Gap-A', 'Gap-T', 'Gap-V']
macro = {(r['intervention'], r['group'], r['rates']): r for r in s['macro']}
lines = ['# Nested per-slot correction OFF', '', 'INTERNAL DIAGNOSTIC ONLY', '',
    'MOSI seed66, original 100epoch Nested per-rate Test-oracle BEST. Frozen',
    'evaluation only; no training. Memory computed once per batch, identical',
    'weights/query/write/masks/Local Skip/Flat/head. Remove a correction by',
    'restoring its raw evidence, not by zeroing the evidence itself.', '',
    'Source code e31f4a2; server biggpu GPU7. Eight checkpoint/model hashes',
    'unchanged, labels/masks equal original archive, predictions agree <=1e-5.', '',
    '## Overall W-F1 (%)', '',
    '| Disabled delta | Mean8 | High (.5/.6/.7) | OFF-minus-ON pp | Corrections / harms (OFF -> ON) |',
    '|---|---:|---:|---:|---:|']
full = macro[('Local', 'all', 'nonempty_rates')]['on_wf1']
high = macro[('Local', 'all', 'high_missing')]['on_wf1']
lines.append(f'|None (Full)|{full:.6f}|{high:.6f}|0|—|')
for slot in slots:
    r = macro[(slot, 'all', 'nonempty_rates')]
    h = macro[(slot, 'all', 'high_missing')]
    lines.append(f"|{slot}|{r['off_wf1']:.6f}|{h['off_wf1']:.6f}|{-r['delta_pp']:+.6f}|{r['corrections']} / {r['harms']}|")
lines += ['', '## Current observed modality groups', '',
    'Equal mean over nonempty rate groups, NOT whole-conversation fixed missing.',
    'T is Text. Group N and counts available in per_rate.csv. Small low-rate',
    'groups can make macro differences noisy; repeated samples are not unique.', '',
    '| Observed | Full | Local OFF | Base OFF | Gap-A OFF | Gap-T OFF | Gap-V OFF |',
    '|---|---:|---:|---:|---:|---:|---:|']
for group in ['A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV', 'no_Text']:
    r = macro[('Local', group, 'nonempty_rates')]
    values = [r['on_wf1']] + [macro[(slot, group, 'nonempty_rates')]['off_wf1'] for slot in slots]
    lines.append('|'+group+'|'+'|'.join(f'{v:.6f}' for v in values)+'|')
lines += ['', '## Interpretation and limits', '',
    'Each individual correction has a small total effect at this trained working',
    'point. Gap-T has 11 corrections / 8 harms; Local has 0 / 2. This is direct',
    'inference contribution, not separately retrained ablation. Other corrections',
    'remain ON, so effects are conditional and need not sum to all-delta-off.',
    'It does not prove no useful information exists or that training had no effect.',
    'Inactive Gap remains masked. No automatic extra seeds/training.', '',
    'Artifacts: results/{STATUS.json,SUMMARY.json,utterances.csv,per_rate.csv,macro.csv}.',
    'Implementation: prior runner --per-slot and readout.disable_correction_slot.',
    'Launch command and PID in RUNNING.md; checkpoints remain remote.', '']
(root / 'RESULT.md').write_text('\n'.join(lines))
print('\n'.join(lines[:24]))
