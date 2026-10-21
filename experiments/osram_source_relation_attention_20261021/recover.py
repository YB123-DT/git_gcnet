"""Recover only the audited MOSI LR-check failure; never restart MOSEI."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .report import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Existing finite queue root')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--record-name', default='RECOVERY.json')
    args = parser.parse_args(); root = args.output
    backup = root.parent/'recovery_epoch100'
    old = json.loads((backup/'SUMMARY.json').read_text())
    if old['status'] != 'failed' or old['epoch'] != 100 or not (backup/'last.pt').exists():
        raise ValueError('Require preserved failed100-run and full epoch100 backup')
    if Path(args.record_name).name != args.record_name or not args.record_name.endswith('.json'):
        raise ValueError('Recovery record must be a simple JSON filename')
    path = root/args.record_name
    if path.exists(): raise FileExistsError('Do not duplicate a recovery watcher')
    from experiments.osram_tasr_20261018.runner_helpers import sha, write_json
    from .run import gpu_free
    if sha(root/'runs/CMUMOSI/last.pt') != sha(backup/'last.pt'):
        raise ValueError('The original full epoch100 checkpoint has changed')
    _, free = gpu_free(args.gpu_uuid)
    if free < 5000: raise RuntimeError('Insufficient free GPU memory to resume MOSI')
    state = dict(status='resuming', source_commit=args.source_commit,
        original_source_commit=old['source_commit'], original_failure=old['error'],
        preserved_epoch100=str(backup), epoch100_checkpoint_sha256=sha(backup/'last.pt'),
        resume_script_sha256=sha(Path(__file__)), root=str(root),
        recovery='Full model/Adam/Python-NumPy-Torch-CUDA RNG; not weights-only; no protocol change',
        original_queue_failed_exit_is_preserved=True, mosei_restarted=False)
    command = [sys.executable, '-u', '-m', 'experiments.osram_source_relation_attention_20261021.worker',
        '--dataset', 'CMUMOSI', '--phase', 'train', '--seed', '66', '--gpu-uuid', args.gpu_uuid,
        '--source-commit', args.source_commit, '--output', str(root/'runs/CMUMOSI'),
        '--resume', '--allow-audit-only-resume']
    state['command'] = command; write_json(path, state)
    try:
        with (root/'train_CMUMOSI_resume.log').open('w') as stream:
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            state.update(worker_pid=proc.pid); write_json(path, state)
            code = proc.wait()
        model = json.loads((root/'runs/CMUMOSI/SUMMARY.json').read_text())
        state.update(returncode=code, mosi_status=model['status'], mosi_epoch=model.get('epoch'))
        if code or model['status'] != 'complete': raise RuntimeError('MOSI recovery failed')
        state['status'] = 'mosi_complete_waiting_original_mosei'; write_json(path, state)
        original_controller = json.loads((root/'QUEUE.json').read_text())['pid']
        while True:
            other = json.loads((root/'runs/CMUMOSEI/SUMMARY.json').read_text())
            state.update(mosei_status=other['status'], mosei_epoch=other.get('epoch')); write_json(path, state)
            if other['status'] == 'failed': raise RuntimeError('Original MOSEI failed; not restarted')
            if other['status'] == 'complete' and not Path('/proc/'+str(original_controller)).exists(): break
            time.sleep(15)
        summarize(root)
        state.update(status='complete', both_formal_runs_complete=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error)); raise
    finally: write_json(path, state)


if __name__ == '__main__': main()
