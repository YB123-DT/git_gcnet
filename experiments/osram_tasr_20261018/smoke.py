"""Real protected-checkpoint GPU verification; NOT a performance experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('MELD', 'IEMOCAPSix'), required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    import numpy as np
    import torch
    from . import runner_helpers as helper
    from .protocol import persistent_availability
    args.output.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic()
    bundle = helper.setup_case(args.dataset, args.gpu_uuid, args.output)
    model, trainer = bundle['model'], bundle['trainer']
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats()
    # Exact original selected mask, without consuming the original mask sampler.
    raw = next(iter(bundle['test_loader']))
    data = trainer._move_batch(raw, device)
    valid = data[7].bool()
    count = int(valid.sum())
    availability = torch.zeros((*valid.shape, 3), device=device)
    availability[valid] = torch.as_tensor(bundle['selected']['0.3']['availability'][:count], device=device)
    availability = availability.transpose(0, 1)
    view = trainer._prepare_view_from_primary_masks(data, availability, availability, bundle['dimensions'])
    with torch.no_grad():
        before_rng = helper.rng_state()
        reference = helper.forward(model, view).detach().clone()
        after_rng = helper.rng_state()
    assert torch.equal(before_rng['torch'], after_rng['torch'])
    assert all(torch.equal(a, b) for a, b in zip(before_rng['cuda'], after_rng['cuda']))
    baseline_digest = helper.backbone_digest(model)
    names = helper.attach_tasr(model, 'real')
    assert helper.backbone_digest(model) == baseline_digest
    assert torch.equal(after_rng['torch'], torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(after_rng['cuda'], torch.cuda.get_rng_state_all()))
    with torch.no_grad():
        zero_on = helper.forward(model, view)
    torch.testing.assert_close(zero_on, reference, rtol=0, atol=0)
    random_source_errors = dict(model.osram.last_source_trace['diagnostics'])
    assert not model.osram.last_source_trace['source_gap'][0].any()
    initial_adapter = {k: v.detach().clone() for k, v in model.osram.tasr.state_dict().items()}

    # Updates use TRAINING data only. Test labels never supervise the smoke.
    bundle['train_loader'].sampler.set_epoch(0)
    train_raw = next(iter(bundle['train_loader']))
    train_data = trainer._move_batch(train_raw, device)
    with helper.isolated_rng(66731):
        schedule = trainer._schedules(bundle['cfg'], 'train')[.3]
        random_view = trainer._prepare_view(train_data, schedule, 0, bundle['dimensions'])
    persistent_mask, pattern_ids = persistent_availability(train_data[7], 66, 0, 0)
    persistent_mask = persistent_mask.to(device)
    persistent_view = trainer._prepare_view_from_primary_masks(
        train_data, persistent_mask, persistent_mask, bundle['dimensions'])
    optimizer = torch.optim.Adam(model.osram.tasr.parameters(), lr=3e-4, weight_decay=1e-5)
    losses, gradients, errors = [], [], []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        pieces = []
        for item in (random_view, persistent_view):
            logits = helper.forward(model, item)
            mask = item['umask'].bool()
            ce = torch.nn.functional.cross_entropy(logits.transpose(0, 1)[mask], item['labels'][mask].long())
            pieces.append(ce)
            errors.append(dict(model.osram.last_source_trace['diagnostics']))
            assert torch.isfinite(logits).all()
        loss = .8 * pieces[0] + .2 * pieces[1]
        loss.backward()
        report = helper.check_adapter_gradients(model)
        if report['nonzero_parameters'] == 0:
            raise RuntimeError('Expected real adapter gradient')
        torch.nn.utils.clip_grad_norm_(model.osram.tasr.parameters(), bundle['cfg'].gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach()))
        gradients.append(report)
    assert helper.backbone_digest(model) == baseline_digest
    if gradients[-1]['nonzero_parameters'] <= gradients[0]['nonzero_parameters']:
        raise RuntimeError('Upstream adapter gradient was not demonstrated after zero-head update')
    with torch.no_grad():
        trained_logits = helper.forward(model, random_view).detach().clone()
    state = helper.checkpoint_payload(model, optimizer, epoch=0,
        identity={'anchor_sha256': bundle['case']['checkpoint_sha256']}, history=[], best_key=(0., 0.))
    helper.atomic_torch_save(args.output / 'smoke_adapter.pt', state)
    model.osram.tasr.load_state_dict(initial_adapter, strict=True)
    helper.restore_checkpoint(model, optimizer, torch.load(args.output / 'smoke_adapter.pt', weights_only=False),
                              identity=state['identity'])
    with torch.no_grad():
        torch.testing.assert_close(helper.forward(model, random_view), trained_logits, rtol=0, atol=0)
    model.osram.tasr.load_state_dict(initial_adapter, strict=True)
    model.osram.tasr.source_mode = 'neutral'
    with torch.no_grad():
        torch.testing.assert_close(helper.forward(model, view), reference, rtol=0, atol=0)
    result = dict(status='complete', smoke_only=True, dataset=args.dataset,
        environment=bundle['environment'], gpu=bundle['gpu'], split=bundle['split'],
        anchor=bundle['case'], anchor_replay=bundle['anchor_parity'],
        actual_model_parameters=sum(p.numel() for p in model.parameters()),
        trainable_adapter_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_names=names, original_backbone_sha256=baseline_digest,
        original_backbone_unchanged=True, default_eval_rng_unchanged=True,
        adapter_initialization_rng_unchanged=True, real_and_neutral_zero_logit_max_abs_error=0.,
        random_source_errors=random_source_errors, training_source_errors=errors,
        train_losses=losses, train_gradient_coverage=gradients, updates=3,
        persistent_pattern_counts=np.bincount(pattern_ids.cpu().numpy(), minlength=6).tolist(),
        strict_save_reload_logit_max_abs_error=0., peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
        seconds=time.monotonic()-tick,
        source_files={str(p.relative_to(ROOT)):helper.sha(p) for p in
                      (ROOT/'gcnet_missing_m3/osram.py', ROOT/'gcnet_missing_m3/tasr.py',
                       Path(__file__), Path(helper.__file__))})
    helper.write_json(args.output / 'SMOKE.json', result)
    print(json.dumps({k:result[k] for k in ('status','dataset','trainable_adapter_parameters',
        'random_source_errors','train_losses','train_gradient_coverage','peak_memory_mib','seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
