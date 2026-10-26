"""Recover only the known final aggregation failure; never start training."""
import argparse
import fcntl
import json
from pathlib import Path

from experiments.osram_three_block_joint_mosi_20261025.scheduler import now, save
from . import summarize as collector


def require_completed_training(queue, expected_source):
    if (queue.get('source_commit') != expected_source or queue.get('expected_train_jobs') != 128
        or len(queue.get('train', [])) != 128
        or any(j.get('status') != 'complete' for j in queue['train'])
        or queue.get('status') != 'failed' or queue.get('error') != "KeyError('branch_off_ACC')"):
        raise ValueError('Only the completed128-job queue with known postprocessing failure may be finalized')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--training-source', required=True)
    parser.add_argument('--collector-source', required=True)
    args = parser.parse_args()
    with (args.root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.root/'QUEUE.json'
        queue = json.loads(path.read_text())
        require_completed_training(queue, args.training_source)
        from .scheduler import validate_state
        validate_state(queue, args.training_source)
        recovery = dict(original_error=queue['error'], failed_at=queue.get('finished_at'),
            training_source=args.training_source, collector_source=args.collector_source,
            collector_sha256=collector.digest(Path(collector.__file__)),
            finalizer_sha256=collector.digest(Path(__file__)), started_at=now(),
            training_repeated=False, model_or_mask_changes=False)
        result = collector.summarize(args.root, args.training_source, queue['train'])
        if result['status'] != 'complete' or result['validated_rate_results'] != 192:
            raise RuntimeError('Final artifact verification failed; do not change failed queue status')
        recovery['completed_at'] = now()
        save(args.root/'POSTPROCESS_RECOVERY.json', recovery)
        queue.update(status='complete', completed_at=recovery['completed_at'],
            updated_at=recovery['completed_at'], postprocess_recovery=recovery)
        queue.pop('error', None)
        # The training-source identity stays intact for consumers and cleanup.
        save(path, queue)
        print(json.dumps(dict(status='complete', validated_rate_results=192, recovery=recovery)), flush=True)


if __name__ == '__main__':
    main()
