"""Three finite frozen workers on healthy GPU3; summarize only complete results."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.osram_persistent_patterns_20261018.analysis import render_report
from experiments.osram_persistent_patterns_20261018.run import write, check_source, sha

DATASETS = ('MELD', 'IEMOCAPFour', 'IEMOCAPSix')


def reuse_meld(source, destination, manifest):
    """Reuse a complete unchanged MELD evaluation after an IEMOCAP-only guard fix."""
    row = json.loads((source/'RESULTS.json').read_text())
    if row['status'] != 'complete' or not row['source_inputs_unchanged'] or row['dataset'] != 'MELD':
        raise ValueError('Require a complete frozen MELD result')
    # Only IEMOCAP source-adapter/config checks may differ, never MELD numerics.
    for name in ('run.py', 'evaluate.py', 'analysis.py'):
        key = 'experiments/osram_persistent_patterns_20261018/'+name
        if row['manifest']['files'][key] != manifest['files'][key]:
            raise ValueError('Cannot reuse MELD after evaluator changes')
    key = 'experiments/osram_meld_wd_seed66_20261018/BASELINE.json'
    if row['manifest']['files'][key] != manifest['files'][key]:
        raise ValueError('Cannot reuse a different MELD baseline')
    if any(sha(path) != digest for path,digest in row['source_hashes'].items()):
        raise ValueError('Reused MELD source input changed')
    destination.mkdir()
    write(destination/'RESULTS.json', dict(row, reused_complete_result=str(source)))


def summarize(root):
    datasets = {}
    for name in DATASETS:
        row = json.loads((root/name/'RESULTS.json').read_text())
        if row['status'] != 'complete' or not row['source_inputs_unchanged']:
            raise ValueError('Require complete verified dataset: '+name)
        datasets[name] = row
    report = dict(status='complete', training=False, datasets=datasets)
    # Include existing original Random0.0–0.7 scores separately, never mixed
    # with Persistent7 scores or used to choose a new checkpoint.
    meld = json.loads((ROOT/'experiments/osram_meld_wd_seed66_20261018/BASELINE.json').read_text())
    iemo = json.loads((ROOT/'experiments/osram_iemocap_flat_3seed_cmerc_20261016/SUMMARY.json').read_text())
    random = {'MELD': {str(i/10): {m: sum(r['testbest']['rates'][str(i/10)][m]
        for r in meld['runs'])/3 for m in ('ACC','UA','WF1')} for i in range(8)}}
    for name in DATASETS[1:]:
        random[name] = {str(i/10): {m: sum(iemo['datasets'][name]['folds'][str(s)][str(f)]['rates'][str(i/10)][m]
            for s in (66,67,68) for f in range(1,6))/15 for m in ('ACC','UA','WF1')} for i in range(8)}
    report['original_random_reference'] = random
    text = render_report(datasets)
    text += '\n## Original random-missing reference (not Persistent)\n\n'
    text += '| Rate | '+' | '.join(f'{d} {m}' for d in DATASETS for m in ('ACC','UA','WF1'))+' |\n'
    text += '|---|'+'---:|'*9+'\n'
    for rate in [str(i/10) for i in range(8)]:
        text += '| '+rate+' | '+' | '.join(f'{random[d][rate][m]:.3f}' for d in DATASETS for m in ('ACC','UA','WF1'))+' |\n'
    text += '| Mean8 | '+' | '.join(f'{sum(random[d][str(i/10)][m] for i in range(8))/8:.3f}' for d in DATASETS for m in ('ACC','UA','WF1'))+' |\n'
    text += '\nMELD remains lr1e-5/wd1e-5/50epochs; IEMOCAP remains original lr1e-3/wd1e-5/100epochs. No protocol unification by retraining.\n'
    text += 'MELD A/T/V reuse the just-completed baseline evaluation; full ATV reuses the verified all-visible selected replay.\n'
    write(root/'RESULTS.json', report)
    (root/'RESULT.md').write_text(text)
    with (root/'per_seed_fold_pattern.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset','seed','fold','pattern','ACC','UA','WF1','CE','N'])
        for name,row in datasets.items():
            for run in row['runs']:
                for p,value in run['patterns'].items():
                    writer.writerow([name,run['seed'],run['fold'],p,*[value['metrics'][m] for m in ('ACC','UA','WF1','CE','N')]])
    with (root/'per_seed_pattern.csv').open('w') as stream:
        writer = csv.writer(stream);writer.writerow(['dataset','seed','pattern','ACC','UA','WF1','CE'])
        for name,row in datasets.items():
            for seed,patterns in row['aggregation']['per_seed'].items():
                for p,score in patterns.items():
                    writer.writerow([name,seed,p,*[score[m] for m in ('ACC','UA','WF1','CE')]])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--reuse-meld', type=Path)
    args = parser.parse_args()
    manifest = check_source()
    if args.output.exists():
        raise FileExistsError('New independent finite evaluation root required')
    if shutil.disk_usage(args.output.parent).free < 2*2**30:
        raise RuntimeError('Need 2GiB output/cache headroom')
    args.output.mkdir(parents=True)
    record = dict(status='running', pid=os.getpid(), source_commit=args.commit,
        started_unix=time.time(), training=False, workers={})
    children = []
    try:
        env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        for name in DATASETS:
            if name == 'MELD' and args.reuse_meld is not None:
                reuse_meld(args.reuse_meld, args.output/name, manifest)
                record['workers'][name] = dict(reused=str(args.reuse_meld), returncode=0)
                write(args.output/'RUNNER.json', record)
                continue
            log = (args.output/f'{name}.log').open('w')
            command = [sys.executable,'-u',str(Path(__file__).with_name('run.py')),
                '--dataset',name,'--output',str(args.output/name),'--commit',args.commit]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append((name,process,log))
            record['workers'][name] = dict(pid=process.pid, command=command)
            write(args.output/'RUNNER.json',record)
        for name,process,log in children:
            record['workers'][name]['returncode'] = process.wait()
            log.close()
            write(args.output/'RUNNER.json',record)
        failures = {name:r['returncode'] for name,r in record['workers'].items() if r['returncode']}
        if failures:
            raise RuntimeError('Frozen worker failures: '+str(failures))
        summarize(args.output)
        record['status'] = 'complete'
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['finished_unix'] = time.time()
        write(args.output/'RUNNER.json',record)


if __name__ == '__main__':
    main()
