"""Summarize all four arms, retaining failures and incomplete-run status."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .protocol import RATES, VARIANTS, LABEL


def collect(output):
    runs, per_rate = {}, []
    for variant in VARIANTS:
        path = output / 'runs' / variant / 'SUMMARY.json'
        row = json.loads(path.read_text()) if path.exists() else dict(status='not_started', epoch=0)
        runs[variant] = row
        for rate, best in row.get('best_by_rate', {}).items():
            per_rate.append(dict(variant=variant, status=row['status'], rate=rate,
                selected_epoch=best['epoch'], **best['metrics']))
    identities = [r['identity'] for r in runs.values() if 'identity' in r]
    matched = all(all(i[key] == identities[0][key] for key in
        ('initial_backbone_sha256', 'mask_reference', 'split', 'environment', 'source_hashes'))
        for i in identities[1:])
    if not matched:
        raise ValueError('Different initialization, masks, split, environment or source; do not pool runs')
    histories = {}
    for variant in VARIANTS:
        path = output / 'runs' / variant / 'history.json'
        if path.exists():
            histories[variant] = json.loads(path.read_text())
    shared_epochs = min((len(x) for x in histories.values()), default=0)
    for epoch in range(shared_epochs):
        traces = {h[epoch]['mask_trace_sha256'] for h in histories.values()}
        if len(traces) != 1:
            raise ValueError(f'Training missing-mask trajectories differ at epoch{epoch+1}')
    return dict(label=LABEL, status='complete' if all(r['status'] == 'complete' for r in runs.values())
        else 'incomplete', matched_identities=matched, paired_training_epochs_checked=shared_epochs,
        runs=runs, per_rate=per_rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.output)
    with (args.output / 'RESULT.json').open('w') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    fields = ('variant', 'status', 'rate', 'selected_epoch', 'ACC', 'WF1', 'MSE', 'N', 'mask_sha256')
    with (args.output / 'per_rate.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result['per_rate'])
    text = ['# CAM A0–A3 MOSEI seed66', '', LABEL, '',
        f"状态：{result['status']}。已核对共同训练mask轨迹 {result['paired_training_epochs_checked']} 轮。",
        '进行中的数字仅为截至当前的Test-best，不是完成成绩。', '',
        '|组|状态|进度|八率ACC|八率W-F1|高缺失W-F1|', '|---|---|---:|---:|---:|---:|']
    for variant, row in result['runs'].items():
        if 'aggregate' in row:
            mean, high = row['aggregate']['mean'], row['aggregate']['high']
            text.append(f"|{variant}|{row['status']}|{row.get('epoch',0)}/60|"
                        f"{mean['ACC']:.3f}|{mean['WF1']:.3f}|{high['WF1']:.3f}|")
        else:
            text.append(f"|{variant}|{row['status']}|{row.get('epoch',0)}/60|—|—|—|")
        if row.get('error'):
            text.extend(['', f"{variant} failure: `{row['error']}`", ''])
    text.extend(['', '|组|Miss|选点epoch|ACC|W-F1|MSE|', '|---|---:|---:|---:|---:|---:|'])
    for row in result['per_rate']:
        text.append(f"|{row['variant']}|{row['rate']}|{row['selected_epoch']}|{row['ACC']:.3f}|"
                    f"{row['WF1']:.3f}|{row['MSE']:.5f}|")
    text.extend(['', '所有A0–A3从头训练。第二阶段Nested未在本轮训练。',
        '保留相同原Flat参数树；A0/A1只屏蔽任务端槽位。',
        '八率均值可能使用不同epoch的模型，不能称为单模型部署性能。',
        '旧Test-oracle参考和本轮四组不得与正式独立泛化验证混淆。', ''])
    (args.output / 'RESULT.md').write_text('\n'.join(text))
    print('\n'.join(text[:14]))


if __name__ == '__main__':
    main()
