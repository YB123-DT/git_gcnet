"""Fixed 0.25 near-only splice of existing predictions; no model execution."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.osram_current_history_relation_20261003.analyze import (
    RATES, align_predictions, read_csv, write_csv, score, macro,
)


def splice(local, flat, relation_prediction):
    return np.where(np.abs(local) <= .25, relation_prediction, flat)


def main():
    root = Path(__file__).resolve().parent
    output = root / 'near_splice_results'
    if output.exists():
        raise FileExistsError('Do not overwrite prior analysis')
    hashes = {}

    def source(path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    audit = read_csv(source(ROOT / 'experiments/osram_context_audit_20261003/results/utterances.csv'))
    fixed = read_csv(source(ROOT / 'experiments/osram_context_audit_20261003/adjacent_results/utterances.csv'))
    mapping = sorted((r for r in read_csv(source(ROOT / 'experiments/osram_cfg84_history_scale_fine_20260929/prediction_analysis/reconstructed_sample_ids.csv')) if int(r['seed']) == 66), key=lambda r: int(r['artifact_row']))
    assert [int(r['artifact_row']) for r in mapping] == list(range(686))
    ids = [r['utterance_id'] for r in mapping]
    index = {uid: i for i, uid in enumerate(ids)}
    per_rate, cells, utterances = [], [], []
    for rate in RATES:
        base_path = ROOT / f'experiments/osram_cfg84_history_scale_20260929/results/seed_66_miss_{rate:.1f}_alpha_1.0.npz'
        relation_path = root / 'results/seed_66' / f"predictions_miss_{rate:.1f}.npz".replace('.', 'p', 1)
        with np.load(source(base_path)) as f:
            base = {k: f[k] for k in f.files}
        with np.load(source(relation_path)) as f:
            relation = align_predictions({k: f[k] for k in f.files}, base, ids)
        original = {r['utterance_id']: r for r in audit if int(r['seed']) == 66 and float(r['rate']) == rate}
        assert set(original) == set(ids)
        local = np.array([float(original[uid]['local_prediction']) for uid in ids])
        flat, labels = base['predictions'], base['labels']
        np.testing.assert_array_equal(flat, [float(original[uid]['memory_prediction']) for uid in ids])
        np.testing.assert_array_equal(labels, [float(original[uid]['label']) for uid in ids])
        prediction = splice(local, flat, relation)
        near = np.abs(local) <= .25
        np.testing.assert_array_equal(prediction[~near], flat[~near])
        np.testing.assert_array_equal(prediction[near], relation[near])
        per_rate.append(dict(seed=66, rate=rate, **score(labels, flat, prediction, local)))
        for i, uid in enumerate(ids):
            utterances.append(dict(seed=66, rate=rate, utterance_id=uid, label=float(labels[i]),
                original_local_prediction=float(local[i]), flat_prediction=float(flat[i]),
                relation_prediction=float(relation[i]), use_relation=bool(near[i]),
                splice_prediction=float(prediction[i])))
        selected = [r for r in fixed if int(r['seed']) == 66 and float(r['rate']) == rate]
        for r in selected:
            assert (r['boundary'] == 'near') == bool(near[index[r['utterance_id']]])
        for boundary in ('near', 'far'):
            for rel in ('same', 'opposite'):
                take = [index[r['utterance_id']] for r in selected if r['boundary'] == boundary and r['relation'] == rel]
                cell = dict(seed=66, rate=rate, boundary=boundary, relation=rel,
                            **score(labels[take], flat[take], prediction[take], local[take]))
                if boundary == 'far':
                    assert cell['delta'] == 0 and cell['corrections'] == cell['harms'] == 0
                cells.append(cell)
    aggregate = [dict(boundary=b, relation=r, **macro([c for c in cells if c['boundary'] == b and c['relation'] == r])) for b in ('near', 'far') for r in ('same', 'opposite')]
    overall = macro(per_rate)
    high = macro([r for r in per_rate if r['rate'] >= .5])
    # Rename the generic scoring helper's candidate column explicitly.
    def rename(row):
        return {('splice_wf1' if k == 'relation_wf1' else k): v for k, v in row.items()}
    per_rate, cells, aggregate = [list(map(rename, rows)) for rows in (per_rate, cells, aggregate)]
    overall, high = rename(overall), rename(high)
    report = ['# Flat / Relation Near-only Splice', '', 'INTERNAL DIAGNOSTIC ONLY', '',
        'MOSI seed66；仅使用已保存预测，无训练、无新增推理、无权重修改。',
        '固定abs(original Flat Local-only prediction)<=0.25时使用Relation full，否则保留Flat full。',
        'Local-only是原Audit关闭Memory读值的预测；same/opposite和gold label绝不参与切换。',
        '未扫描threshold。输入沿用每rate BEST Test-oracle，因此不是正式论文成绩或已验证部署方案。',
        '沿用原ID及availability、label!=0过滤、pred>0分类；先按rate算W-F1再平均。', '',
        '|Rate|Flat|Splice|Δpp|纠错|致错|N|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in per_rate + [dict(rate='8-rate mean', **overall), dict(rate='High .5/.6/.7', **high)]:
        report.append(f"|{r['rate']}|{r['baseline_wf1']:.3f}|{r['splice_wf1']:.3f}|{r['delta']:+.3f}|{r['corrections']}|{r['harms']}|{r['n']}|")
    report += ['', '|固定四格|Flat|Splice|Δpp|纠错|致错|N|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:
        report.append(f"|{r['boundary']} + {r['relation']}|{r['baseline_wf1']:.3f}|{r['splice_wf1']:.3f}|{r['delta']:+.3f}|{r['corrections']}|{r['harms']}|{r['n']}|")
    report += ['', '纠错/致错均相对于原Flat full。计数为跨rate重复出现次数，不是独立样本数。',
        '固定四格沿用原分组，排除首句及任一端中性的相邻对；不是全测试集。',
        'near两格保留Relation结果；far两格逐样本严格等于Flat。没有依据分组标签切换。',
        '本结果只是固定阈值的事后拼接诊断，单seed且沿用Test-oracle checkpoint，不证明可训练单模型或多seed收益。',
        '运行：python experiments/osram_current_history_relation_20261003/near_splice.py', '']
    for path, digest in hashes.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    output.mkdir()
    for name, rows in [('per_rate', per_rate), ('four_cells_per_rate', cells), ('four_cells', aggregate), ('utterances', utterances)]:
        write_csv(output / f'{name}.csv', rows)
    (output / 'summary.json').write_text(json.dumps(dict(overall=overall, high_missing=high, four_cells=aggregate), indent=2) + '\n')
    (output / 'AUDIT.json').write_text(json.dumps(dict(threshold=.25, threshold_sweep=False, training=False,
        inference=False, label_conditioned_switch=False, seed=66, source_sha256=hashes), indent=2) + '\n')
    (output / 'RESULT.md').write_text('\n'.join(report))
    print('\n'.join(report))


if __name__ == '__main__':
    main()
