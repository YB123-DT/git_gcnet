"""Audit completed four-arm records without inference or parameter updates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    status = json.loads((root/'STATUS.json').read_text())
    summary = json.loads((root/'SUMMARY.json').read_text())
    assert status['status'] == 'complete' and len(status['records']) == 8
    audit = dict(verified=True, total_pairs=0, nonneutral_pairs=0, code_hashes_verified=0,
                 max_current_local_error=0., max_focal_local_error=0., max_baseline_prediction_error=0.,
                 polarity_rescue_to_harm=0, polarity_harm_to_rescue=0, rates={}, raw_sha256={})
    provenance = json.loads((root/'PROVENANCE.json').read_text())
    for name, digest in provenance['code_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
        audit['code_hashes_verified'] += 1
    csv_rows = []
    for rate in [i/10 for i in range(8)]:
        path = root/f'rate_{rate:.1f}.json'
        record = json.loads(path.read_text())
        assert record['rate'] == rate and record['frozen_unchanged']
        rows = record['rows']
        assert len(rows) == summary['per_rate'][str(rate)]['n']
        assert len({(r['conversation'], r['target']) for r in rows}) == len(rows)
        audit['raw_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        audit['rates'][str(rate)] = dict(n=len(rows), coverage=record['coverage'],
            checkpoint_sha256=record['checkpoint_sha256'], original_full_test=record['original_full_test'])
        for coverage in record['coverage']:
            assert coverage['targets'] == coverage['eligible'] + coverage['skipped']
            assert coverage['eligible'] == coverage['evaluated']
            for key in ['max_current_local_error', 'max_focal_local_error', 'max_baseline_prediction_error']:
                audit[key] = max(audit[key], coverage[key])
                assert coverage[key] <= 3e-6
        for row in rows:
            assert row['text_position'] < row['target'] and row['other_position'] < row['target']
            assert row['text_position'] != row['other_position'] and row['other_modality'] in ['A', 'V']
            y, p = row['label'], row['predictions']
            da = (y-p['A-'])**2-(y-p['A+'])**2
            db = (y-p['B-'])**2-(y-p['B+'])**2
            assert abs(da-row['delta_A']) < 1e-12 and abs(db-row['delta_B']) < 1e-12
            assert abs(db-da-row['interaction']) < 1e-12
            for arm in ['A', 'B']:
                on, off = p[arm+'+'] > 0, p[arm+'-'] > 0
                assert row[arm+'_rescue'] == (y != 0 and on == (y > 0) and off != (y > 0))
                assert row[arm+'_harm'] == (y != 0 and off == (y > 0) and on != (y > 0))
            audit['total_pairs'] += 1
            audit['nonneutral_pairs'] += y != 0
            audit['polarity_rescue_to_harm'] += row['A_rescue'] and row['B_harm']
            audit['polarity_harm_to_rescue'] += row['A_harm'] and row['B_rescue']
            csv_rows.append(dict(rate=rate, **{k:v for k,v in row.items() if k not in ['predictions', 'availability', 'masks_sha256']},
                                 **{f'pred_{k}':v for k,v in p.items()}))
    with (root/'paired_predictions.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (root/'AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
