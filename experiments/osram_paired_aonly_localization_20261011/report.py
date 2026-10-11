"""Render frozen localization results with the limits of each intervention."""
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    status=json.loads((root/'results_v2/STATUS.json').read_text())
    data=json.loads((root/'results_v2/SUMMARY.json').read_text())
    assert status['status']=='complete' and len(data['records'])==3
    lines=['# Persistent A-only: where the positive bias appears', '',
        'INTERNAL DIAGNOSTIC ONLY. MOSI seed66; three frozen miss=.7 BEST checkpoints.', '',
        '## Observed failure location', '',
        'The rho.10 trained Flat readout predicts positive for every nonneutral A-only utterance when Base/Gap inputs are zero. Adding persistent Audio-only history changes only one polarity. On matched current-A utterances, original random-history reads substantially compensate this positive bias. The failure therefore already exists in the no-history-evidence decision path; it is not created solely by adding Gap reads.', '',
        'This localizes behavior to the learned Local-to-task decision path under absent historical reads. It does not separately identify an encoder defect versus a classifier defect, or prove Audio has no sentiment information.', '',
        '## Persistent A-only: all656 nonneutral utterances', '',
        'L is the SAME trained Flat with Base=Gap=0, not a separately trained Local model. LB masks Gap; LG masks Base; LBG is unchanged full output.', '',
        '| Model | Readout | Predicted positive /656 | W-F1 % | Prediction mean* | Prediction std* |',
        '|---|---|---:|---:|---:|---:|']
    for r in data['table']:
        if r['regime']=='persistent' and r['cohort']=='all_A':
            lines.append(f"|{r['model']}|{r['readout']}|{r['pred_positive']}|{100*r['weighted_f1']:.3f}|{r['mean']:.4f}|{r['std']:.4f}|")
    lines+=['', '*Prediction moments include all686 utterances; classification excludes30 neutral labels.', '',
        '## Same current A utterances: paired history comparison', '',
        '161 nonneutral utterances, identical IDs/labels/current raw inputs. Local differences are only float32 packing noise (absolute≤1.91e-6, relative≤1.93e-7). No model parameters changed.', '',
        '| Model | History | Readout | Predicted positive /161 | W-F1 % |',
        '|---|---|---|---:|---:|']
    for r in data['table']:
        if r['cohort']=='matched_A' and r['readout'] in ('L','LBG'):
            lines.append(f"|{r['model']}|{r['regime']}|{r['readout']}|{r['pred_positive']}|{100*r['weighted_f1']:.3f}|")
    lines+=['', '## What is established and what is not', '',
        '- rho.10: all656 positive without read values;655 positive with full A-only history. Base/Gap removal does not rescue this model.',
        '- rho.50:208 positive without read values,154 positive with full A-only history. Its better fixed-pattern output does not require a positive contribution from persistent Memory in this readout intervention.',
        '- Original Flat is also biased positive without reads (631/656); rho.10 intensifies this observed weakness. This was not introduced by InfoNCE: both compared paired-view models are task-only.',
        '- On matched current-A samples, rho.10 Full scores65.424 with random history versus24.556 with A-only history, while Local-input readout is23.223 in both. Random test performance can conceal failure when historical support is persistently restricted.',
        '- First turns provide a supplementary natural no-history check: rho.10 predicts all26 nonneutral first turns positive, and L/Full outputs agree. Small cohort, not a separate broad claim.',
        '- Random-history versus persistent-A changes historical support in both directions: it removes T/V but can restore earlier A. It is not a pure Text-deletion intervention and cannot identify Text alone as the causal source.',
        '- Setting active reads to zero may be out-of-distribution. Zero-readout outcomes cannot establish absence of information in Local, and the first-turn check has limited sample size.',
        '- Checkpoints differ (Flat39, rho.50 epoch43, rho.10 epoch100). No common-epoch A-only trajectories are available, so weaker augmentation weight versus late training/selection cannot be causally separated here.',
        '- Historical code/config audit finds no reversed loss weights, mask difference, modality swap, inference second view or shared Memory. See CODE_AUDIT.md.',
        '- No training, threshold search, learned routing, new block or parameter change. One backbone seed, inherited Test-oracle protocol.', '',
        '## Execution and verification', '',
        f"Historical backbone6e2a998; runner{status['code_commit']}. biggpu GPU6, Python PID{status['pid']}.",
        'Remote root: /data1/yb/remote_experiments/osram_paired_aonly_localization_20261011.',
        'Initial run stopped on overly strict absolute Local tolerance1e-6. Reused the already validated historical diagnostic limits (absolute1e-5 AND relative1e-6), with exact current raw input equality. Original failure/log preserved. No model or input change.',
        'All three original random prediction archives and full replay outputs match exactly; checkpoint/model hashes unchanged. All per-utterance values and all grouped readouts retained in results_v2/.', '',
        '```bash',
        'CUDA_VISIBLE_DEVICES=GPU-e4cafb17-818e-216a-b94a-7440063a9153 python -m experiments.osram_paired_aonly_localization_20261011.run \\',
        '  --output <new-output-directory> --commit 61d011f', '```','']
    (root/'RESULT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
