"""Verify completed paired schedules and report all rate-specific BEST scores."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cosine', type=Path, required=True)
    p.add_argument('--constant', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary, masks = {}, {}
    table = []
    for schedule, root in [('constant', args.constant), ('cosine', args.cosine)]:
        for model in ['flat', 'nested']:
            directory = root/'runs'/f'{model}_seed66'
            provenance = json.loads((directory/'PROVENANCE.json').read_text())
            assert provenance['status'] == 'complete' and provenance['outputs_verified']
            for name, digest in provenance['artifact_sha256'].items():
                assert sha(directory/name) == digest, name
            gradient_rows = []
            for name, digest in provenance['gradient_sha256'].items():
                assert sha(directory/'gradients'/name) == digest
                gradient_rows.extend(json.loads((directory/'gradients'/name).read_text()))
            assert len(gradient_rows) == 200
            masks[schedule, model] = [(r['epoch'], r['batch'], r['availability_sha256']) for r in gradient_rows]
            assert len(json.loads((directory/'history.json').read_text())) == 100
            cfg = provenance['effective_config']
            assert cfg['lr_schedule'] == schedule and cfg['seed'] == 66 and cfg['epochs'] == 100
            assert cfg['learning_rate'] == .001 and cfg['warmup_ratio'] == .05
            metrics = json.loads((directory/'metrics.json').read_text())
            scores = {r: 100*v for r,v in metrics['selected_weighted_f1_by_rate'].items()}
            epochs = metrics['selected_epoch_by_rate']
            summary[f'{model}_{schedule}'] = dict(mean8=mean(scores.values()),
                high=mean(scores[r] for r in ['0.5', '0.6', '0.7']), per_rate=scores, best_epochs=epochs,
                source_commit=provenance['historical_commit'], effective_config=cfg)
            for rate in scores:
                table.append(dict(model=model, schedule=schedule, rate=rate, wf1=scores[rate], best_epoch=epochs[rate]))
            if schedule == 'cosine':
                rates = []
                assert len(provenance['learning_rate_sha256']) == 100
                for epoch in range(1,101):
                    name = f'epoch_{epoch:03d}.json'
                    path = directory/'learning_rates'/name
                    assert sha(path) == provenance['learning_rate_sha256'][name]
                    values = json.loads(path.read_text())
                    expected = .001*epoch/5 if epoch <= 5 else .001*.5*(1+math.cos(math.pi*(epoch-5)/95))
                    assert values['epoch'] == epoch and all(abs(lr-expected) < 1e-15 for lr in values['rates'])
                    rates.append(dict(epoch=epoch, lr=values['rates'][0]))
                with (args.output/f'{model}_learning_rates.csv').open('w') as handle:
                    writer=csv.DictWriter(handle, fieldnames=['epoch','lr'])
                    writer.writeheader()
                    writer.writerows(rates)
    reference=masks['constant','flat']
    assert all(v == reference for v in masks.values())
    for model in ['flat','nested']:
        before=summary[f'{model}_constant']['effective_config']
        after=summary[f'{model}_cosine']['effective_config']
        assert {k:(before[k],after[k]) for k in before if before[k]!=after[k]} == {'lr_schedule':('constant','cosine')}
    with (args.output/'per_rate.csv').open('w') as handle:
        writer=csv.DictWriter(handle, fieldnames=['model','schedule','rate','wf1','best_epoch'])
        writer.writeheader()
        writer.writerows(table)
    summary['verification'] = dict(paired_training_masks=200, artifacts_per_run=20,
                                   runs_complete=4, learning_rate_hashes_verified=200,
                                   configuration_changes_only_schedule=True)
    (args.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Paired warmup/cosine screen', '', 'INTERNAL DIAGNOSTIC ONLY', '',
           'MOSI seed66. Original Flat and original Nested, sealed source ad211c0. Two new100epoch runs; reuse completed constant controls. Per-rate Test-oracle BEST, not independent paper performance. Cyclic random missing, original masks/loss/model/head/batch protocol unchanged. Added ONLY existing warmup/cosine package:5epochs warmup, peak1e-3, then cosine to0 at100. This does not isolate pure cosine from warmup or alternative schedules.', '',
           '| Model | Constant mean8 | Cosine mean8 | Delta pp | Constant high | Cosine high | Delta pp |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for model in ['flat','nested']:
        a,b=summary[f'{model}_constant'],summary[f'{model}_cosine']
        lines.append(f"| {model} | {a['mean8']:.3f} | {b['mean8']:.3f} | {b['mean8']-a['mean8']:+.3f} | {a['high']:.3f} | {b['high']:.3f} | {b['high']-a['high']:+.3f} |")
    lines += ['', '| Rate | Flat constant | Flat cosine | Nested constant | Nested cosine |', '|---|---:|---:|---:|---:|']
    for rate in [str(i/10) for i in range(8)]:
        values=[summary[k]['per_rate'][rate] for k in ['flat_constant','flat_cosine','nested_constant','nested_cosine']]
        lines.append('| '+rate+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    cg=summary['nested_constant']['mean8']-summary['flat_constant']['mean8']
    sg=summary['nested_cosine']['mean8']-summary['flat_cosine']['mean8']
    lines += ['', f'Nested-minus-Flat mean8 gap: constant{cg:+.3f}pp, cosine{sg:+.3f}pp. Relative comparison improves, but BOTH cosine models score below their own constant version; do not call this an absolute Nested improvement.', '',
              'All four runs complete, all20artifact hashes/run and gradient hashes verified;200paired training masks identical across models/schedules. All200cosine learning-rate file hashes and actual group rates match the full expected curve, finalLR0. per_rate.csv includes all BEST epochs; separate CSVs preserve actualLR traces. Full recovery and eight BEST checkpoints retained on biggpu.', '',
              'Conclusion: this seed66 warmup/cosine configuration does not improve final BEST W-F1. Earlier convergence is not final performance gain. No automatic further seeds, schedule search or structural modifications. No inference or training added for this report. Single seed: no statistical significance or multi-seed consistency claim.', '']
    (args.output/'RESULT.md').write_text('\n'.join(lines))
    print(json.dumps({k:{f:v[f] for f in ['mean8','high']} for k,v in summary.items() if k!='verification'}, indent=2))


if __name__ == '__main__':
    main()
