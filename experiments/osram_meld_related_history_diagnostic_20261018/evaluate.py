"""Frozen B1/B2 readout interventions using one unchanged CHM scan per batch.

The S=.5 condition changes only B2's D/rho fusion slots, not retrieval or
memory. Labels below annotate saved predictions and relation pairs offline;
they are never passed to either model or the relation predictor.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.osram_meld_gcnet_random_no_skip_evidence_20261014.evaluate import (
    replay_view_from_availability, scatter_saved_availability,
)
from experiments.osram_meld_small_lr_readout_20261018.evaluate import (
    METADATA, _metadata, _numpy, _validate_saved,
)
from experiments.osram_meld_write_tuning_20261017.capture import _metrics


class BranchCapture:
    """Capture actual computation nodes before the integration clears its cache."""

    def __init__(self, osram):
        self.osram = osram
        self.branch = osram.related_history
        self.handles = []
        self.anchor_calls = self.branch_calls = 0

    def __enter__(self):
        def anchor(module, args):
            self.anchor_calls += 1
            self.anchor = args[0].detach().clone()

        def fusion(module, args):
            self.fusion_input = args[0].detach().clone()

        def relation(module, args, output):
            self.relation_logits = output.squeeze(-1).detach().clone()

        def branch(module, args, output):
            self.branch_calls += 1
            self.residual = output.detach().clone()
            self.attention = module.last_attention.detach().clone()
            self.cache = None if module._cache is None else {
                name: value.detach().clone() for name, value in module._cache.items()}

        # prepend captures the raw Flat anchor before the attachment adds delta.
        try:
            handle = self.osram.emotion_norm.register_forward_pre_hook(anchor, prepend=True)
        except TypeError:
            # Torch 1.13 (the fixed experimental environment) lacks prepend.
            # Move only our new hook, preserving every existing hook's order.
            handle = self.osram.emotion_norm.register_forward_pre_hook(anchor)
            self.osram.emotion_norm._forward_pre_hooks.move_to_end(handle.id, last=False)
        self.handles.append(handle)
        self.handles.append(self.branch.fusion[0].register_forward_pre_hook(fusion))
        self.handles.append(self.branch.register_forward_hook(branch))
        if self.branch.kind == 'B2':
            self.handles.append(self.branch.relation.register_forward_hook(relation))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


def _ce(logits, labels):
    shifted = logits.astype(np.float64) - logits.max(1, keepdims=True)
    return np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(labels)), labels]


def _summary(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite diagnostic values')
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p05=float(np.quantile(values, .05)), p95=float(np.quantile(values, .95)),
                max=float(values.max()), N=int(len(values)))


def _pair_artifacts(capture, data, availability):
    cache = capture.cache
    if cache is None:
        raise RuntimeError('B2 did not expose its actual relation cache')
    mask = cache['pair_mask'].bool()
    current, batch, slot = mask.nonzero(as_tuple=True)
    past = cache['indices'][current, slot]
    labels = data[8].transpose(0, 1)
    speakers = data[6].transpose(0, 1)
    if speakers.ndim == 3:
        speakers = speakers.argmax(-1)
    ids = np.asarray([str(value) for value in data[-1]])
    logits = capture.relation_logits[mask]
    return dict(
        labels=_numpy(labels[current, batch] == labels[past, batch]).astype(np.int64),
        logits=_numpy(logits), probabilities=_numpy(logits.sigmoid()),
        attention=_numpy(capture.attention[mask]),
        conversation_ids=ids[_numpy(batch)], current_indices=_numpy(current),
        past_indices=_numpy(past), lag=_numpy(current - past),
        same_speaker=_numpy(speakers[current, batch] == speakers[past, batch]),
        availability=_numpy(availability[current, batch]),
        past_availability=_numpy(availability[past, batch]),
        availability_pattern=_numpy((availability[current, batch].long()
            * availability.new_tensor([1, 2, 4]).long()).sum(-1)))


def evaluate_rate(model, loader, cfg, dimensions, saved, trainer, verify_saved=True):
    """Return metrics, per-utterance arrays and actual B2 relation-pair arrays.

    ``verify_saved=False`` is only for the matched-epoch cross-model replay:
    masks and metadata still match exactly; checkpoint-specific logits need not.
    Every on/off/half comparison inside a model remains strictly paired.
    """
    if cfg.dataset != 'MELD':
        raise ValueError('Only seven-class MELD is supported')
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Provide a fully frozen eval-mode model')
    osram = model.osram
    branch = osram.related_history
    if not osram.related_history_enabled or branch.kind not in {'B1', 'B2'}:
        raise ValueError('Start from an enabled B1/B2 checkpoint')
    saved = _validate_saved(saved)
    conditions = ('on', 'off', 'half') if branch.kind == 'B2' else ('on', 'off')
    predictions = {name: [] for name in conditions}
    parts = {name: [] for name in METADATA}
    norm_parts = {name: [] for name in ('residual_norm', 'anchor_norm', 'residual_anchor_ratio')}
    if branch.kind == 'B2':
        norm_parts['half_residual_norm'] = []
    pairs = []
    scans = cursor = 0
    saved_error = replay_error = 0.
    saved_disagreements = 0
    with torch.no_grad():
        for raw in loader:
            data = trainer._move_batch(raw, next(model.parameters()).device)
            metadata = _metadata(data)
            count = len(metadata['labels'])
            for name, actual in metadata.items():
                np.testing.assert_array_equal(actual, saved[name][cursor:cursor + count],
                    err_msg='Selected mask/loader alignment failed for ' + name)
            availability, stop = scatter_saved_availability(saved['availability'], cursor, data[7])
            view = replay_view_from_availability(data, availability, dimensions)
            valid = view['umask'].bool()
            np.testing.assert_array_equal(_numpy(view['availability'].transpose(0, 1))[valid.cpu().numpy()],
                                          saved['availability'][cursor:stop])
            with BranchCapture(osram) as capture:
                logits, _, _, missing = model([view['incomplete']], view['availability'],
                    view['qmask'], view['umask'], view['lengths'], predict_missing=False)
            if missing is not None or capture.anchor_calls != 1 or capture.branch_calls != 1:
                raise RuntimeError('Expected one original scan, one Flat and one branch call')
            scans += 1
            original = _numpy(logits.transpose(0, 1)[valid])
            expected = saved['logits'][cursor:stop]
            saved_error = max(saved_error, float(np.abs(original - expected).max(initial=0.)))
            saved_disagreements += int((original.argmax(1) != saved['predictions'][cursor:stop]).sum())
            if verify_saved:
                np.testing.assert_allclose(original, expected, rtol=2e-5, atol=2e-5)
                np.testing.assert_array_equal(original.argmax(1), saved['predictions'][cursor:stop])
            active = valid.transpose(0, 1) & (capture.attention.sum(-1) > 0)
            residuals = dict(on=capture.residual, off=torch.zeros_like(capture.residual))
            if branch.kind == 'B2':
                inputs = capture.fusion_input.clone()
                inputs[..., -(branch.attention_dim + 1):-1] = 0.
                inputs[..., -1] = torch.where(active, .5, 0.)
                half = branch.output(branch.fusion(inputs))
                residuals['half'] = torch.where(active[..., None], half, torch.zeros_like(half))
                pairs.append(_pair_artifacts(capture, data, view['availability']))
            previous = osram.related_history_enabled
            osram.related_history_enabled = False
            try:
                for condition, residual in residuals.items():
                    hidden = osram.emotion_norm(capture.anchor + residual)
                    hidden = torch.where(valid.transpose(0, 1)[..., None], hidden, torch.zeros_like(hidden))
                    replay = _numpy(model.smax_fc(hidden).transpose(0, 1)[valid])
                    if not np.isfinite(replay).all():
                        raise ValueError('Nonfinite replay logits')
                    if condition == 'on':
                        np.testing.assert_allclose(replay, original, rtol=2e-5, atol=2e-5)
                        np.testing.assert_array_equal(replay.argmax(1), original.argmax(1))
                        replay_error = max(replay_error, float(np.abs(replay - original).max(initial=0.)))
                    predictions[condition].append(replay)
            finally:
                osram.related_history_enabled = previous
            residual_norm = capture.residual.norm(dim=-1)
            anchor_norm = capture.anchor.norm(dim=-1)
            norms = dict(residual_norm=residual_norm, anchor_norm=anchor_norm,
                         residual_anchor_ratio=residual_norm / anchor_norm.clamp_min(1e-12))
            if branch.kind == 'B2':
                norms['half_residual_norm'] = residuals['half'].norm(dim=-1)
            for name, value in norms.items():
                norm_parts[name].append(_numpy(value.transpose(0, 1)[valid]))
            for name, value in metadata.items():
                parts[name].append(value)
            cursor = stop
    if cursor != len(saved['labels']) or not scans:
        raise ValueError('Loader must consume every selected row exactly once')
    artifacts = {name: np.concatenate(values) for name, values in parts.items()}
    artifacts.update({name: np.concatenate(values) for name, values in norm_parts.items()})
    artifacts['availability'] = saved['availability'].copy()
    labels = artifacts['labels']
    metrics = {}
    for name, values in predictions.items():
        combined = np.concatenate(values)
        artifacts['logits_' + name] = combined
        artifacts['pred_' + name] = combined.argmax(1)
        artifacts['ce_' + name] = _ce(combined, labels)
        metrics[name] = _metrics(dict(labels=labels, logits=combined))
    changes = {}
    on_correct = artifacts['pred_on'] == labels
    for name in conditions[1:]:
        correct = artifacts['pred_' + name] == labels
        delta = artifacts['ce_on'] - artifacts['ce_' + name]
        changes[name + '_to_on'] = dict(corrections=int((~correct & on_correct).sum()),
            harms=int((correct & ~on_correct).sum()), both_correct=int((correct & on_correct).sum()),
            both_wrong=int((~correct & ~on_correct).sum()),
            prediction_changes=int((artifacts['pred_on'] != artifacts['pred_' + name]).sum()),
            mean_ce_on_minus_reference=float(delta.mean()),
            ce_improvements=int((delta < -1e-8).sum()), ce_harms=int((delta > 1e-8).sum()))
    relation_pairs = {} if not pairs else {
        name: np.concatenate([part[name] for part in pairs]) for name in pairs[0]}
    return dict(metrics=metrics, changes=changes,
        norms={name: _summary(artifacts[name]) for name in norm_parts},
        checks=dict(memory_scan_count=scans, saved_availability_rows=cursor,
            metadata_exact=True, saved_prediction_verification_requested=bool(verify_saved),
            saved_predictions_exact=saved_disagreements == 0,
            saved_prediction_disagreements=saved_disagreements,
            saved_logit_max_abs_error=saved_error, on_replay_class_exact=True,
            on_replay_logit_max_abs_error=replay_error)), artifacts, relation_pairs
