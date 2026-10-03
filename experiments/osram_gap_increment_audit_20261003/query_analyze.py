"""Summarize existing OSRAM per-head query/read diagnostics without fitting."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.osram_gap_increment_audit_20261003 import analyze as common

ALIASES = dict(cos_base_gap_query='query_base_gap_cos', cos_gap_residual_query='query_gap_residual_cos',
               cos_base_gap_read='read_base_gap_cos')
METRICS = ('query_base_gap_cos', 'query_gap_residual_cos', 'read_base_gap_cos',
           'rho', 'eta', 'norm_q_base', 'norm_q_gap', 'norm_base_read', 'norm_gap_read',
           'query_base_residual_gap_cos', 'read_minus_raw_query_cos', 'read_minus_actual_query_cos')
STATS = ('mean', 'p10', 'p50', 'p90')


def prepare_rows(rows):
    result, seen = [], set()
    for source in rows:
        r = dict(source)
        r['seed'], r['rate'], r['head'] = int(r['seed']), float(r['rate']), int(r['head'])
        if r['availability'] not in common.PATTERNS or r['head'] not in range(8) or r['modality'] not in ('A', 'T', 'V'):
            raise ValueError('Invalid diagnostic indices')
        if r['modality'] in r['availability']:
            raise ValueError('Expected active (currently missing) Gap modality')
        key = r['seed'], r['rate'], r['utterance_id'], r['modality'], r['head']
        if key in seen:
            raise ValueError('Duplicate query diagnostic')
        seen.add(key)
        for old, new in ALIASES.items():
            if old in r:
                r[new] = r.pop(old)
        for name in METRICS:
            r[name] = common.number(r.get(name))
            if r[name] is not None and name.endswith('_cos') and not name.startswith('read_minus'):
                if not -1.00001 <= r[name] <= 1.00001:
                    raise ValueError('Invalid cosine outside [-1,1]')
        for name, query in (('read_minus_raw_query_cos', 'query_base_gap_cos'),
                            ('read_minus_actual_query_cos', 'query_base_residual_gap_cos')):
            r[name] = r['read_base_gap_cos'] - r[query] if r['read_base_gap_cos'] is not None and r[query] is not None else None
        result.append(r)
    return result


def quantile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (pos - lo) * (values[hi] - values[lo])


def summarize(rows):
    out = dict(n=len(rows))
    for name in METRICS:
        values = [r[name] for r in rows if r[name] is not None]
        out[name + '_n'] = len(values)
        out[name + '_missing'] = len(rows) - len(values)
        out[name + '_mean'] = statistics.mean(values) if values else None
        for suffix, fraction in (('p10', .1), ('p50', .5), ('p90', .9)):
            out[name + '_' + suffix] = quantile(values, fraction) if values else None
    return out


def join_categories(rows, audit):
    lookup = {}
    for r in audit:
        key = int(r['seed']), float(r['rate']), r['utterance_id']
        if key in lookup:
            raise ValueError('Duplicate stage1 audit sample')
        lookup[key] = r
    result = []
    for r in rows:
        old = lookup[(r['seed'], r['rate'], r['utterance_id'])]
        if old['availability'] != r['availability']:
            raise ValueError('Stage1 availability mismatch')
        if 'label' in r and float(old['label']) != float(r['label']):
            raise ValueError('Stage1 label mismatch')
        result.append(dict(r, polarity_category=old['polarity_category']))
    return result


def analyze(rows):
    groups = defaultdict(list)
    for r in rows:
        for modality, head in ((r['modality'], r['head']), (r['modality'], 'ALL'), ('ALL', 'ALL')):
            groups[(r['seed'], r['rate'], modality, head)].append(r)
    per_rate = [dict(seed=seed, rate=rate, modality=modality, head=head, **summarize(values))
                for (seed, rate, modality, head), values in sorted(groups.items(), key=lambda item: str(item[0]))]
    scores = tuple(name + '_' + suffix for name in METRICS for suffix in STATS)
    counts = ('n',) + tuple(name + '_' + suffix for name in METRICS for suffix in ('n', 'missing'))
    macro = []
    for modality, head in sorted({(r['modality'], r['head']) for r in per_rate}, key=str):
        selected = [r for r in per_rate if r['modality'] == modality and r['head'] == head]
        macro.append(dict(modality=modality, head=head, **common.macro(selected, scores, counts)))
    category_per_rate = []
    if rows and 'polarity_category' in rows[0]:
        grouped = defaultdict(list)
        for r in rows:
            if r['polarity_category'] in ('rescue', 'harm'):
                grouped[(r['seed'], r['rate'], r['modality'], r['head'], r['polarity_category'])].append(r)
        for (seed, rate, modality, head, category), values in sorted(grouped.items()):
            category_per_rate.append(dict(seed=seed, rate=rate, modality=modality, head=head, category=category, **summarize(values)))
    return dict(per_rate=per_rate, macro=macro, rescue_harm_per_rate=category_per_rate)


def report(data):
    text = ['# Query / addressing descriptive audit', '', 'INTERNAL DIAGNOSTIC ONLY', '',
            'Original seed66 Full checkpoints, frozen inference; no model/loss/head selection changes. Only valid utterances with an active Gap '
            'are included, eight forward memory heads. Zero-norm undefined cosines are omitted with explicit per-metric valid/missing counts.',
            'Each seed/rate/modal/head cell is summarized first. Macro values equally average nonempty rates per seed, then seeds. '
            'Macro p10/p50/p90 are averages of within-rate quantiles, NOT pooled distribution quantiles. '
            'Counts are repeated utterance × active-Gap × head exposures, not independent samples.', '',
            '|Modality|Head|Raw query cosine mean|Raw query p50|Gap/residual query mean|Read cosine mean|Read p50|Read−raw query mean|Paired N|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['macro']:
        names = ('query_base_gap_cos_mean', 'query_base_gap_cos_p50', 'query_gap_residual_cos_mean',
                 'read_base_gap_cos_mean', 'read_base_gap_cos_p50', 'read_minus_raw_query_cos_mean')
        text.append(f"|{r['modality']}|{r['head']}|" + '|'.join(common.fmt(r[name]) for name in names) + f"|{r['read_minus_raw_query_cos_n']}|")
    text += ['', '## Interpretation limits', '',
             'Raw q_Base/q_Gap cosine is measured BEFORE residual addressing. Gap read uses residualized addressing, '
             'so read−raw-query cosine is a descriptive paired difference, NOT an isolated effect of the Memory map. '
             'Existing cos(q_Gap,q_residual), rho, and eta are separately preserved. The raw-query/read comparison cannot '
             'by itself establish rank collapse, target specificity failure, or that Memory makes distinct actual queries identical.',
             'No arbitrary similar/different threshold, sign fitting, significance test, head routing, or learned classifier is used. '
             'Cosine measures angle, not equality or information content. Near-constant rho/eta may reflect addressing construction '
             'rather than independent evidence. Norms and quantiles are available in CSVs.',
             'Per-rate weights differ; head IDs and descriptive associations do not prove stable specialization across checkpoints or seeds. '
             'Optional rescue/harm join uses stage1 labels only for retrospective reporting; the same utterance label is repeated across active heads. '
             'No further diagnostic stage or mechanism is automatically added.', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    def read(path):
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))
    rows = prepare_rows(read(args.input))
    if args.audit:
        rows = join_categories(rows, read(args.audit))
    grouped = defaultdict(set)
    for r in rows:
        grouped[(r['seed'], r['rate'], r['utterance_id'], r['modality'])].add(r['head'])
    if any(heads != set(range(8)) for heads in grouped.values()):
        raise ValueError('Expected all eight heads for each active Gap')
    data = analyze(rows)
    for source, digest in sources.items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Input changed during analysis')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, values in data.items():
        common.write_csv(args.output / f'{name}.csv', values)
    (args.output / 'RESULT.md').write_text(report(data))
    (args.output / 'provenance.json').write_text(json.dumps(sources, indent=2) + '\n')
    selected = [r for r in data['macro'] if r['head'] == 'ALL']
    compact = [{k: r[k] for k in ('modality', 'n', 'query_base_gap_cos_mean', 'query_gap_residual_cos_mean',
                                  'read_base_gap_cos_mean', 'read_minus_raw_query_cos_mean')} for r in selected]
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
