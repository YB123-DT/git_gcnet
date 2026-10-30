"""One bounded real-MOSEI compatibility and worst-length GPU verification."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'PREFLIGHT.json'
    if target.exists():
        raise FileExistsError('Preflight must not overwrite previous evidence')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import numpy as np
    import torch
    from . import bundle as source_bundle, protocol as p, runtime as r
    from .stage2 import FrozenNestedModel, restore_parent
    from .train import checkpoint_payload, restore

    torch.set_num_threads(1)
    source = source_bundle.verify(Path(__file__).resolve().parents[2])
    result = dict(status='running', source_commit=source['source_commit'],
                  smoke_only=True, seconds=0, started_at=time.time())
    r.save_json(target, result)
    tick = time.monotonic()
    try:
        print('Verifying environment, data, initialization, fixed masks...', flush=True)
        b = r.setup('A3', args.output / 'preflight', args.gpu_uuid)
        model = b['model']
        original = json.loads((p.PARENT / 'SUMMARY.json').read_text())
        selected = original['best_by_rate']['0.3']
        parent = restore_parent(model, p.PARENT / selected['checkpoint'],
                                expected_sha256=selected['checkpoint_sha256'])
        scan_calls = []
        native_scan = model.osram._scan

        def observe(*args, **kwargs):
            scan_calls.append(1)
            return native_scan(*args, **kwargs)

        print('Replaying protected miss0.3 checkpoint on the complete saved test mask...', flush=True)
        with patch.object(model.osram, '_scan', side_effect=observe):
            scores, arrays = r.evaluate(b, model, '0.3')
        if len(scan_calls) != len(b['test_loader']):
            raise RuntimeError('Original Flat did not perform exactly one causal scan per batch')
        prediction_path = p.PARENT / selected['predictions']
        if r.sha(prediction_path) != selected['prediction_sha256']:
            raise ValueError('Protected selected predictions changed')
        with np.load(prediction_path, allow_pickle=False) as previous:
            for key in ('labels', 'speakers', 'conversation_ids', 'utterance_indices', 'availability', 'prediction'):
                np.testing.assert_array_equal(arrays[key], previous[key],
                                              err_msg='Protected original checkpoint mismatch: ' + key)
        expected = selected['metrics']['full']
        for key in ('ACC', 'WF1', 'MSE'):
            if not math.isclose(scores[key], expected[key], abs_tol=1e-7, rel_tol=0):
                raise ValueError('Selected metric mismatch: ' + key)
        result.update(parent_replay=dict(parent=parent, selected_prediction_sha256=r.sha(prediction_path),
            arrays_bitwise_equal=True, metrics=scores, live_scan_calls=len(scan_calls)))

        # Run all requested paths on the same real padded batch, no retraining.
        raw = next(iter(b['test_loader']))
        data = b['trainer']._move_batch(raw, torch.device('cuda:0'))
        view = r.saved_view(b, data, b['test_masks']['0.3'][0])
        control = {}
        for variant, switches in p.VARIANTS.items():
            model.osram.osram_emotion_ablation = switches['osram_emotion_ablation']
            model.osram.osram_gap_read = switches['osram_gap_read']
            inputs = []
            handle = model.osram.emotion_adapter.register_forward_pre_hook(
                lambda _, args: inputs.append(args[0].detach().clone()))
            try:
                with torch.no_grad():
                    prediction = r.forward(model, view)
            finally:
                handle.remove()
            if len(inputs) != 1 or not torch.isfinite(prediction).all():
                raise RuntimeError('Invalid single-pass native task path')
            value = inputs[0]
            if variant == 'A0' and value[..., 256:].count_nonzero():
                raise RuntimeError('A0 received historical task input')
            if variant == 'A1' and value[..., 1280:].count_nonzero():
                raise RuntimeError('A1 received Missing-info task input')
            control[variant] = dict(flat_shape=list(value.shape), finite=True,
                                    base_norm=float(value[..., 256:1280].norm()),
                                    gap_norm=float(value[..., 1280:].norm()))
        model.osram.osram_emotion_ablation = 'full'
        model.osram.osram_gap_read = 'residual'
        model.eval()
        with r.isolated_rng(66):
            nested = FrozenNestedModel(model).to('cuda:0').eval()
        calls = []
        with patch.object(model.osram, '_scan', side_effect=lambda *a, **kw: (calls.append(1), native_scan(*a, **kw))[1]):
            with torch.no_grad():
                nested_prediction = nested.forward_view(view)
                baseline_prediction = r.forward(model, view)
        torch.testing.assert_close(nested_prediction, baseline_prediction, rtol=0, atol=0)
        if len(calls) != 2:
            raise RuntimeError('Nested must scan once, as must the separate control call')
        result.update(real_batch_paths=control, nested_zero_init_exact=True,
                      nested_parameters=sum(x.numel() for x in nested.block.parameters()))
        del nested, model, arrays, prediction, nested_prediction, baseline_prediction

        print('Running fresh-task backward/update and full resume check...', flush=True)
        model = r.build_model(b['cfg'], b['dimensions'], torch.device('cuda:0'))
        b['model'] = model
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        reset_training_stochasticity(66, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        torch.cuda.reset_peak_memory_stats()
        smoke = r.train_epoch(b, model, optimizer, 0, max_batches=3)
        if any(not smoke['gradients'].get(group, {}).get('nonzero') for group in
               ('observed_set.projectors', 'osram.key_projectors', 'osram.value_projectors',
                'osram.query_projector', 'osram.emotion_adapter')):
            raise RuntimeError('An expected original task path has no nonzero gradient after startup')

        # Explicitly include the longest real train conversations at batch32.
        loader = b['train_loader']
        indices = sorted(loader.sampler.indices, key=lambda i: len(
            loader.dataset.videoLabelsNew[loader.dataset.vids[i]]), reverse=True)[:32]
        longest = loader.collate_fn([loader.dataset[i] for i in indices])
        data = b['trainer']._move_batch(longest, torch.device('cuda:0'))
        schedule = b['trainer']._schedules(b['cfg'], 'train')[.7]
        view = b['trainer']._prepare_view(data, schedule, 0, b['dimensions'])
        optimizer.zero_grad(set_to_none=True)
        prediction = r.forward(model, view)
        loss = b['trainer']._task_loss('CMUMOSEI', prediction, view['labels'], view['umask'],
                                      'regression', 'mse', b['cfg'].task_smooth_l1_beta)
        loss.backward()
        if not torch.isfinite(loss) or any(not torch.isfinite(x.grad).all()
                for x in model.parameters() if x.grad is not None):
            raise RuntimeError('Longest real batch produced nonfinite loss/gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), b['cfg'].gradient_clip_norm)
        optimizer.step()
        torch.cuda.synchronize()
        identity = json.loads(json.dumps(b['identity']))
        saved = checkpoint_payload(model, optimizer, epoch=1, identity=identity, history=[smoke], best={})
        r.atomic_save(args.output / 'smoke_full_resume.pt', saved)
        restored = torch.load(args.output / 'smoke_full_resume.pt', map_location='cpu', weights_only=False)
        digest = r.state_digest(model)
        restore(model, optimizer, restored, identity)
        if r.state_digest(model) != digest or len(r.rng_state()['cuda']) != 1:
            raise RuntimeError('Strict recovery or UUID CUDA RNG coverage failed')
        memory = torch.cuda.max_memory_reserved() / 2**20
        # Includes allocator, contexts and a generous allowance for concurrent jobs.
        job_mib = int(math.ceil((memory * 1.25 + 1024) / 256) * 256)
        result.update(status='passed', original_parameters=sum(x.numel() for x in model.parameters()),
            smoke=smoke, longest_batch=dict(batch=len(indices), padded_length=int(longest[7].shape[1]),
                rate=.7, MSE=float(loss.detach())), optimizer_rng_strict_restore=True,
            peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_memory_mib=memory, job_memory_budget_mib=job_mib,
            gpu=b['gpu'], environment=b['environment'], identity=b['identity'],
            source_commit=source['source_commit'], seconds=time.monotonic()-tick)
        print(json.dumps({k: result[k] for k in ('status', 'seconds', 'peak_memory_mib',
                                               'peak_reserved_memory_mib', 'job_memory_budget_mib')}, indent=2), flush=True)
    except BaseException as error:
        result.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        r.save_json(target, result)


if __name__ == '__main__':
    main()
