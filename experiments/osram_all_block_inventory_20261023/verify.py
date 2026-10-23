"""Verify generated inventory joins without importing training code."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

from .inventory import ROOT, SELF, SKIP, snapshot_directory


def main():
    output=ROOT/'experiments'/SELF
    manifest=json.loads((output/'MANIFEST.json').read_text())
    counts=manifest['counts']
    projects=list(csv.DictReader((output/'EXPERIMENTS.csv').open()))
    runs=list(csv.DictReader((output/'RUNS.csv').open()))
    methods=json.loads((output/'registry_catalog.json').read_text())['methods']
    names={r['experiment'] for r in projects}
    local={p.name for p in (ROOT/'experiments').iterdir()
           if p.is_dir() and p.name not in SKIP and not snapshot_directory(p.name)}
    assert local<=names,sorted(local-names)
    assert len(projects)==counts['experiment_directories']
    assert len(runs)==counts['run_evidence_rows']
    assert len(methods)==len({m['method_id'] for m in methods})==counts['selector_or_catalog_entries']
    for r in runs:
        assert r['project'] in names
        for key in ('ACC','UA','WF1'):
            if r.get(key):assert -1e-6<=float(r[key])<=100+1e-6,(key,r)
    path=str(ROOT/'experiments/osram_shift_capacity_20261002/results/D3-W256/metrics.json')
    sample=next(r for r in runs if r['source_file']==path)
    assert abs(float(sample['WF1'])-80.93184090337002)<1e-9,sample
    assert 'stored selected' in sample['score_basis']
    compressed={}
    for name in ('METRICS.csv','SOURCE_TABLE_ROWS.csv','RUN_RECORDS.json'):
        original=hashlib.sha256()
        restored=hashlib.sha256()
        with (output/name).open('rb') as src:
            for chunk in iter(lambda:src.read(1024**2),b''):original.update(chunk)
        with gzip.open(output/(name+'.gz'),'rb') as src:
            for chunk in iter(lambda:src.read(1024**2),b''):restored.update(chunk)
        assert original.digest()==restored.digest(),name
        compressed[name]=original.hexdigest()
    with gzip.open(output/'METRICS.csv.gz','rt') as src:
        metric_count=sum(1 for _ in csv.DictReader(src))
    assert metric_count==counts['metric_evidence_rows']
    print(json.dumps(dict(local_directories_covered=len(local),
        total_directories=len(projects),run_evidence_rows=len(runs),
        catalog_entries=len(methods),metric_rows=metric_count,
        compressed_roundtrip_sha256=compressed,
        known_selected_score_verified=True),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
