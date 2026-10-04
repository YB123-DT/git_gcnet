"""Reserve evidenced implemented subsets; 120 is a target, not a completed count."""
import argparse
import fcntl
from pathlib import Path

from experiments.osram_meaningful20_20261003.manifest import read, write, digest_json
from experiments.osram_meaningful20_round2_20261004.manifest import validate_round
from gcnet_missing_m3.meaningful_v3_registry import V3_METHODS


def records(value):
    if isinstance(value, dict):
        if all(key in value for key in ('id', 'title', 'status')):
            yield value
        else:
            for item in value.values():
                yield from records(item)
    elif isinstance(value, list):
        for item in value:
            yield from records(item)


def select(root, names):
    if (not names or len(names) > 20 or len(names) != len(set(names))
            or not set(names) <= set(V3_METHODS)):
        raise ValueError('Select 1..20 unique implemented methods, not survey-only ideas')
    sources = {}
    for path in sorted((root / 'docs/osram_survey80_20261004').glob('*.json')):
        for record in records(read(path)):
            if record['id'] in sources:
                raise ValueError('Duplicate paper card ID: ' + record['id'])
            sources[record['id']] = (path, record)
    cards = []
    for name in names:
        path, source = sources[name]
        if source['status'] != 'admissible':
            raise ValueError('Conditional/rejected sources cannot enter training')
        if not isinstance(source.get('authors'), list) or not source['authors']:
            raise ValueError('Traceable paper authors required')
        card = dict(id=name, title=source['title'], authors=source['authors'],
                    accepted=True, design_file=str(path.relative_to(root)),
                    design_sha256=digest_json(source),
                    source={'paper': source['url'], 'code': source['code_url']},
                    novelty_audit={
                        'closest_prior_ids': source['nearest_prior'],
                        'substantive_difference': source['substantive_difference'],
                        'same_round_overlap': 'Distinct complete algorithms, not width/depth variants.',
                        'source_code_evidence': source['source_evidence']})
        cards.append(card)
    manifest = dict(round_id='v3_120_incremental', target_count=20,
                    batch_target=120, status='source_accepted',
                    label='INTERNAL DIAGNOSTIC ONLY', cards=cards)
    validate_round(manifest)
    return manifest


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=root / 'experiments/osram_method_registry.json')
    args = parser.parse_args()
    manifest = select(root, args.candidate)
    with args.registry.with_suffix('.selection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.output.exists():
            raise FileExistsError('Preserve immutable previously selected subsets')
        registry = read(args.registry)
        rounds = registry.setdefault('rounds', {})
        key = 'v3_120_20261004'
        previous = {item['id'] for r, value in rounds.items() if r != key
                    for item in value.get('methods', [])}
        if previous.intersection(args.candidate):
            raise ValueError('Do not rerun an older method under the new batch')
        batch = rounds.setdefault(key, dict(status='incremental_reserved',
            max_distinct_methods=120, methods=[],
            user_request='2026-10-04: find 120 modules, implement, then launch'))
        known = {item['id'] for item in batch['methods']}
        if len(known | set(args.candidate)) > 120:
            raise ValueError('This batch exceeds its explicit 120-method authorization')
        batch['methods'].extend(card for card in manifest['cards'] if card['id'] not in known)
        # The new batch is additional to prior reservations, not unlimited search.
        budget = len(previous) + 120
        registry['max_distinct_trained_methods'] = budget
        registry['training_budget_override'] = dict(max_distinct_trained_methods=budget,
            user_request=batch['user_request'], prior_reserved=len(previous), additional_batch_limit=120)
        write(args.registry, registry)
        write(args.output, manifest)


if __name__ == '__main__':
    main()
