"""Pin the explicitly requested forty comparisons, without starting a process."""
import argparse
import fcntl
from pathlib import Path

from experiments.osram_meaningful20_20261003.manifest import digest_json, read, sha, write
from experiments.osram_meaningful20_20261003.run import candidate_config
from experiments.osram_meaningful20_round2_20261004.manifest import validate_round
from .configs import placement, HISTORICAL_MECHANISMS
from gcnet_missing_m3.priority40_registry import PRIORITY_FAMILIES, PRIORITY_METHODS


PAPERS = {
    'm01': 'https://proceedings.mlr.press/v97/lee19d.html',
    'm02': 'https://proceedings.mlr.press/v267/altabaa25a.html',
    'm03': 'https://arxiv.org/abs/2105.14491',
    'm04': 'https://arxiv.org/abs/1801.07829',
    'm05': 'https://arxiv.org/abs/2004.05718',
    'm06': 'https://aclanthology.org/D17-1115/',
    'm07': 'https://aclanthology.org/P18-1209/',
    'm08': 'https://arxiv.org/abs/1708.01471',
    'm09': 'https://arxiv.org/abs/1705.06676',
    'm10': 'https://arxiv.org/abs/1902.00038',
    'm11': 'https://arxiv.org/abs/1709.07871',
    'm12': 'https://arxiv.org/abs/1604.03539',
    'm13': 'https://arxiv.org/abs/1911.08670',
    'm14': 'https://arxiv.org/abs/1909.11519',
    'm15': 'https://arxiv.org/abs/2003.10027',
    'm16': 'https://arxiv.org/abs/1703.06114',
    'm17': 'https://arxiv.org/abs/1811.01900',
    'm18': 'https://arxiv.org/abs/1906.02795',
    'm19': 'https://arxiv.org/abs/1511.07247',
    'm20': 'https://arxiv.org/abs/1803.10963',
    'm21': 'https://arxiv.org/abs/1708.05027',
    'm22': 'https://arxiv.org/abs/2008.13535',
    'm23': 'https://arxiv.org/abs/1803.05170',
    'm24': 'https://arxiv.org/abs/1708.04617',
    'm25': 'https://arxiv.org/abs/2404.19756',
    'm26': 'https://arxiv.org/abs/2108.02927',
    'm27': 'https://arxiv.org/abs/1712.01034',
    'm28': 'https://arxiv.org/abs/2106.09681',
    'm29': 'https://arxiv.org/abs/2206.11925',
    'm30': 'https://arxiv.org/abs/2503.10622',
    'm31': 'https://arxiv.org/abs/2105.01601',
    'm32': 'https://arxiv.org/abs/2105.08050',
    'm33': 'https://arxiv.org/abs/2105.03824',
    'm34': 'https://arxiv.org/abs/2301.00808',
    'm35': 'https://arxiv.org/abs/2201.12083',
    'm36': 'https://arxiv.org/abs/2008.02217',
    'm37': 'https://arxiv.org/abs/1905.05702',
    'm38': 'https://arxiv.org/abs/2308.00951',
    'm39': 'https://arxiv.org/abs/1909.06312',
    'm40': 'https://arxiv.org/abs/1710.09829',
}


def manifests(repo):
    directory = repo / 'experiments/osram_priority40_20261004'
    baseline = read(directory / 'BASELINE_CONFIG.json')
    cards = []
    for method in PRIORITY_METHODS:
        config_path = directory / 'configs' / (method + '.json')
        config = read(config_path)
        if config != candidate_config(baseline, method, seed=66):
            raise ValueError('Saved configuration differs from exact cfg84 seed66: ' + method)
        family = next((family for family, ids in PRIORITY_FAMILIES.items() if method in ids), 'common')
        source = 'gcnet_missing_m3/priority40_' + family + '.py'
        design = dict(method=method, placement=placement(method), config_sha256=sha(config_path),
                      implementation=source, implementation_sha256=sha(repo / source))
        prior = list(HISTORICAL_MECHANISMS.get(method, ('cfg84_flat',)))
        cards.append(dict(id=method, accepted=True, design_sha256=digest_json(design),
            design=design, source={'paper': PAPERS[method[:3]], 'code': source},
            source_code_kind='local independently implemented adaptation; upstream evidence in module docstring',
            explicit_user_comparison=True, counted_as_new_method=False if method in HISTORICAL_MECHANISMS else None,
            novelty_audit=dict(closest_prior_ids=prior,
                substantive_difference='User-specified fixed five-role configuration. Known overlaps are intentional controls, not new methods; see design and source module.',
                same_round_overlap='M01/M37 intentionally differ only by attention normalization; all overlap is retained as requested, never counted as separate novelty.',
                source_code_evidence=source + ': source links, formula transfers and limitations recorded in module.')))
    result = []
    for index in range(2):
        batch = dict(round_id=f'priority40_explicit_comparisons_part{index+1}', target_count=20,
            status='source_accepted', label='INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle',
            authorization='User 2026-10-04: 写好了就那个延续启动吧; explicitly supplied M01-M40',
            cards=cards[index*20:(index+1)*20])
        validate_round(batch)
        result.append(batch)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    parts = manifests(repo)
    if args.output.exists():
        raise FileExistsError('Do not overwrite an existing launch authorization')
    registry_path = repo / 'experiments/osram_method_registry.json'
    with registry_path.with_suffix('.selection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = read(registry_path)
        batches = registry.setdefault('explicit_comparison_batches', {})
        key = 'priority40_20261004'
        if key in batches:
            raise ValueError('Already authorized; inspect the existing plan instead of duplicating it')
        batches[key] = dict(status='authorized_not_yet_launched', configuration_count=40,
            seed=66, epochs=100, gpu_indices=[2, 6], max_concurrent_per_gpu=11,
            method_ids=list(PRIORITY_METHODS), new_method_count=None,
            user_request='2026-10-04: 写好了就那个延续启动吧',
            historical_controls_not_counted_as_novel=True,
            manifests=[str(args.output / f'PART{i+1}.json') for i in range(2)])
        args.output.mkdir(parents=True, exist_ok=False)
        for index, part in enumerate(parts, 1):
            write(args.output / f'PART{index}.json', part)
        write(registry_path, registry)


if __name__ == '__main__':
    main()
