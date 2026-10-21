"""MOSEI66 Direct: Local/Base/Gap readout diagnostic without training."""
from pathlib import Path
import numpy as np

from experiments.osram_mosei_source_role_switch_20261021.run import main as frozen_main
from .readout import replay_roles
from .analysis import analyze
from .report import summarize


def collect(model, trainer, loader, dimensions, saved):
    import torch
    from experiments.osram_12_source_evidence_audit_20261019.run import capture_flat, require_same_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    parts = {}; cursor = 0; scans = 0
    loader.sampler.set_epoch(0)
    with torch.no_grad():
        for raw in loader:
            data = trainer._move_batch(raw, next(model.parameters()).device)
            valid = data[7].bool(); meta = metadata(data); n = len(meta['labels'])
            for k, v in meta.items():
                np.testing.assert_array_equal(v, saved[k][cursor:cursor+n])
            av = torch.zeros((*valid.shape, 3), device=valid.device)
            av[valid] = torch.as_tensor(saved['availability'][cursor:cursor+n], device=av.device, dtype=av.dtype)
            av = av.transpose(0, 1)
            view = trainer._prepare_view_from_primary_masks(data, av, av, dimensions)
            if not torch.equal(view['availability'], av):
                raise ValueError('Exact availability changed')
            rng = h.rng_state()
            with capture_flat(model) as cap:
                full = h.forward(model, view)
            scans += 1
            predictions = replay_roles(model.osram, model.smax_fc, cap['local'], cap['emotion_input'], av, data[7], full)
            require_same_rng(rng, h.rng_state())
            flatten = lambda v: v.transpose(0, 1)[valid].detach().cpu().numpy()
            values = dict(meta, availability=flatten(av), **{
                'prediction_' + k: flatten(v).reshape(-1) for k, v in predictions.items()})
            np.testing.assert_array_equal(values['prediction_Full'], saved['prediction'][cursor:cursor+n].reshape(-1))
            for k, v in values.items():
                parts.setdefault(k, []).append(v)
            cursor += n
    if cursor != len(saved['labels']):
        raise ValueError('Incomplete cohort')
    return {k: np.concatenate(v) for k, v in parts.items()}, dict(
        memory_scans=scans, test_batches=scans, selected_predictions_exact=True,
        exact_labels_order_masks=True, no_rng_change=True, no_second_memory_scan=True,
        learned_gamma_retained=True, original_local_skip_retained=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    frozen_main(collect_fn=collect, analyze_fn=analyze, report_fn=summarize,
        extra_source_files=tuple(str(p.relative_to(root)) for p in Path(__file__).parent.glob('*.py')))
