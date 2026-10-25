"""MOSI66 frozen training-only audit. No optimizer, reset or new training."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REFERENCE = Path('/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2')
PATTERNS = {'audio': [4, 5, 6, 7], 'text': [2, 3, 6, 7], 'visual': [1, 3, 5, 7]}


@contextmanager
def capture(model, view):
    """Capture real fc2 inputs without replacing any original computation."""
    handles, tensors, ids = [], {}, {}
    availability = view['availability']
    valid = view['umask'].T.bool()
    pattern = (availability.long() * availability.new_tensor([4, 2, 1])).sum(-1).long()
    names = list(model.observed_set.projectors)
    if len(names) != 3:
        raise ValueError('Three modality projectors required')
    for index, name in enumerate(names):
        selected = valid & availability[..., index].bool()
        ids[name] = pattern[selected].detach().cpu().numpy()
        def hook(module, args, name=name):
            if name in tensors:
                raise RuntimeError('Projector called twice in one scan')
            tensors[name] = args[0]
        handles.append(model.observed_set.projectors[name].fc2.register_forward_pre_hook(hook))
    try:
        yield tensors, ids
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if args.batches <= 0:
        raise ValueError('Positive batch count required')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import torch
    from experiments.osram_mosi_cmerc_20261023 import worker as control
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from .stats import Stats
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    start = time.monotonic()
    state = dict(status='running', pid=os.getpid(), train_steps=0, dataset='CMUMOSI', seed=66,
                 source_commit=args.source_commit, python=sys.executable)
    h.write_json(args.output/'STATUS.json', state)
    try:
        original = json.loads((control.REFERENCE/'SUMMARY.json').read_text())
        prior = json.loads((REFERENCE/'SUMMARY.json').read_text())
        bundle = control.setup(args.gpu_uuid, args.output, original)
        model = bundle['model'].eval()
        best = prior['best_by_rate']['0.3']
        checkpoint = REFERENCE/best['checkpoint']
        assert h.sha(checkpoint) == best['checkpoint_sha256']
        saved = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(saved['model'], strict=True)
        del saved
        before = h.backbone_digest(model)
        trainer, loader, cfg = bundle['trainer'], bundle['train_loader'], bundle['cfg']
        assert cfg.missing_mask_protocol == 'gcnet-random-mask-v1'
        names = list(model.observed_set.projectors)
        # Code modality order is A/T/V, regardless of dictionary spelling.
        roles = dict(zip(names, PATTERNS))
        total = {n: Stats(model.observed_set.projectors[n].fc1.out_features) for n in names}
        per_rate, batches, mask_arrays = {}, [], {}
        cohort = None
        for ri in range(8):
            rate = ri / 10
            schedules = trainer._schedules(cfg, 'train')
            stats = {n: Stats(model.observed_set.projectors[n].fc1.out_features) for n in names}
            loader.sampler.set_epoch(0)
            rate_cohort = []
            with h.isolated_rng(660025 + ri):
                for bi, raw in enumerate(loader):
                    if bi >= args.batches:
                        break
                    data = trainer._move_batch(raw, next(model.parameters()).device)
                    view = trainer._prepare_view(data, schedules[rate], 0, bundle['dimensions'])
                    rate_cohort.extend(str(v) for v in view['conversation_ids'])
                    a, valid = view['availability'], view['umask'].T.bool()
                    assert not a[~valid].any() and (a[valid].sum(-1) >= 1).all()
                    if ri == 0:
                        assert (a[valid].sum(-1) == 3).all()
                    if ri == 7:
                        assert (a[valid].sum(-1) == 1).all()
                    key = f'rate{ri}_batch{bi}'
                    mask_arrays[key] = a.detach().cpu().numpy()
                    # No-hook reference on identical inputs; this does not resample masks.
                    with torch.no_grad():
                        expected = h.forward(model, view)
                    rng_before = torch.get_rng_state().clone()
                    cuda_before = torch.cuda.get_rng_state().clone()
                    with capture(model, view) as (hidden, ids):
                        prediction = h.forward(model, view)
                        torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
                        errors = (prediction[..., 0] - view['labels'].T).square()
                        # Sum removes variable batch-size scaling. Full sequence task
                        # gradients include causal future effects of these same latents.
                        loss = errors[valid].sum()
                        keys = list(hidden)
                        gradients = torch.autograd.grad(loss, [hidden[n] for n in keys])
                        for name, grad in zip(keys, gradients):
                            activation = hidden[name].detach().cpu().numpy()
                            gradient = grad.detach().cpu().numpy()
                            stats[name].update(ids[name], activation, gradient)
                            total[name].update(ids[name], activation, gradient)
                    assert torch.equal(rng_before, torch.get_rng_state())
                    assert torch.equal(cuda_before, torch.cuda.get_rng_state())
                    batches.append(dict(rate=rate, batch=bi, conversations=view['conversation_ids'],
                        N=int(valid.sum()), mse=float(loss.detach()/valid.sum()),
                        mask_sha256=hashlib.sha256(mask_arrays[key].tobytes()).hexdigest()))
                    del hidden, gradients, prediction, loss, expected
            if cohort is None:
                cohort = rate_cohort
            assert cohort == rate_cohort, 'Use the same training conversations at every rate'
            per_rate[str(rate)] = {roles[n]: stats[n].report(PATTERNS[roles[n]]) for n in names}
            print(f'completed rate={rate}', flush=True)
        assert before == h.backbone_digest(model)
        assert all(p.grad is None for p in model.parameters())
        result = dict(status='complete', checkpoint=str(checkpoint), checkpoint_sha256=h.sha(checkpoint),
            selected_epoch=best['epoch'], original_selection='Test-oracle miss0.3; training-only diagnostic',
            environment=bundle['environment'], gpu=bundle['gpu'], config=prior['config'],
            source_commit=args.source_commit, data_manifest=bundle['data_manifest'],
            cohort=cohort, batches=batches, per_rate=per_rate,
            pooled={roles[n]: total[n].report(PATTERNS[roles[n]]) for n in names},
            checks=dict(parameters_unchanged=True, no_parameter_grads=True, hooked_prediction_exact=True,
                        hook_rng_unchanged=True, mask_boundaries_valid=True, same_train_cohort=True),
            seconds=time.monotonic()-start, peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
            limitation=f'One checkpoint, up to {args.batches} batches per rate (actual {len(batches)} total), no temporal persistence or causal uselessness inference. Gradients include later losses; conditions refer to source utterance availability.')
        np.savez_compressed(args.output/'train_masks.npz', **mask_arrays)
        h.write_json(args.output/'RESULT.json', result)
        lines = ['# MOSI projector usage audit', '', 'INTERNAL DIAGNOSTIC ONLY', '',
                 'Training-only, frozen weights, eval/dropout-off, sum-MSE gradients. No optimizer.', '',
                 '| Modality | Width | Low at .01 | Low at .05 | Low at .1 |', '|---|---:|---:|---:|---:|']
        for role, row in result['pooled'].items():
            counts = [row['thresholds'][t]['all_conditions_count'] for t in ('0.01','0.05','0.1')]
            lines.append(f'| {role} | 256 | {counts[0]} | {counts[1]} | {counts[2]} |')
        lines += ['', result['limitation'], '', 'Thresholds are descriptive fractions of per-condition channel means, not validated reset rules.',
                  'Pooled sample slots repeat the same conversations under eight masks; not independent samples.',
                  'Exact masks remain remote; JSON retains cohort and per-batch mask hashes.', '']
        (args.output/'RESULT.md').write_text('\n'.join(lines))
        state.update(status='complete', seconds=result['seconds'])
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        h.write_json(args.output/'STATUS.json', state)


if __name__ == '__main__':
    main()
