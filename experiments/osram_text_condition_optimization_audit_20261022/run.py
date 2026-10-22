"""One fixed Flat checkpoint: test strata, train-only paired gradients/SGD."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.osram_source_relation_attention_20261021.worker import REFERENCES, RATES
from experiments.osram_v3_source_attention_audit_20261022.run import sha, write


def choose_targets(view, dataset, rate):
    mask = view['availability'].detach().cpu().numpy()
    valid = view['umask'].detach().cpu().numpy().astype(bool)
    result = []
    for b, name in enumerate(view['conversation_ids']):
        eligible = np.flatnonzero(valid[b] & (mask[:, b, [0, 2]].sum(-1) > 0))
        eligible = eligible[eligible > 0]
        if len(eligible):
            seed = hashlib.sha256(f'{dataset}:{rate}:{name}:66'.encode()).digest()
            result.append((b, int(eligible[int.from_bytes(seed[:8], 'big') % len(eligible)])))
    return result


def prefix_view(view, b, t, dimensions, text_visible):
    import torch
    from .core import paired_masks
    availability = view['availability'][:t+1, b:b+1]
    umask = view['umask'][b:b+1, :t+1]
    yes, no = paired_masks(availability, umask, t, 0)
    selected = yes if text_visible else no
    complete = view['complete'][:t+1, b:b+1].detach()
    expanded = torch.repeat_interleave(selected, torch.tensor(dimensions, device=selected.device), dim=-1)
    return dict(complete=complete, incomplete=complete*expanded, availability=selected,
        qmask=view['qmask'][b:b+1, :t+1], umask=umask,
        labels=view['labels'][b:b+1, :t+1], lengths=[int(umask.sum())],
        conversation_ids=[view['conversation_ids'][b]])


@contextmanager
def prewrite_capture(osram, target):
    original = osram.block_write
    result = {'calls': 0}
    def counted(memory, *args, **kwargs):
        if result['calls'] == target:
            result['memory'] = memory.detach().cpu().clone()
        result['calls'] += 1
        return original(memory, *args, **kwargs)
    osram.block_write = counted
    try:
        yield result
    finally:
        osram.block_write = original


@contextmanager
def isolated_current_encoding(model):
    """Keep packed historical encoder GEMMs identical across a paired prefix.

    The original encoder is utterance-local, but packs observed rows. Changing
    the last Text bit otherwise changes the GEMM shape for historical Text too.
    This diagnostic-only execution ordering preserves both history gradients
    and the original encoder parameters. The causal CHM still scans once.
    """
    import torch
    encoder = model.observed_set
    if any(module.training for module in model.modules()):
        raise ValueError('Paired diagnostic requires eval mode, including encoder dropout')
    original = encoder.forward
    had_override = 'forward' in encoder.__dict__
    def segmented(features, availability, umask):
        if features.shape[0] < 2:
            raise ValueError('Paired prefix requires a historical utterance')
        past, past_latents = original(features[:-1], availability[:-1], umask[:, :-1])
        current, current_latents = original(features[-1:], availability[-1:], umask[:, -1:])
        return torch.cat((past, current), dim=0), {
            k: torch.cat((past_latents[k], current_latents[k]), dim=0)
            for k in past_latents}
    encoder.forward = segmented
    try:
        yield
    finally:
        if had_override:
            encoder.forward = original
        else:
            del encoder.forward


def target_loss(model, view, capture=False):
    from experiments.osram_tasr_20261018.runner_helpers import forward
    t = view['availability'].shape[0]-1
    with isolated_current_encoding(model):
        if capture:
            with prewrite_capture(model.osram, t) as memory:
                prediction = forward(model, view)
            memory['past_prediction'] = prediction[:t].detach().cpu().clone()
            memory['target_prediction'] = float(prediction[t,0,0].detach())
        else:
            prediction = forward(model, view)
            memory = None
    loss = (prediction[t, 0, 0]-view['labels'][0, t]).square()
    return loss, memory


def test_strata(bundle, model, banks, output):
    import torch
    from .core import history_text_seen, group_summary
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    from experiments.osram_tasr_20261018.runner_helpers import forward, isolated_rng
    trainer, loader = bundle['trainer'], bundle['test_loader']
    all_arrays, summaries = [], {}
    for rate in RATES:
        loader.sampler.set_epoch(0)
        pieces = {}
        with isolated_rng(66+99193), torch.no_grad():
            for index, raw in enumerate(loader):
                data = trainer._move_batch(raw, next(model.parameters()).device)
                meta = metadata(data)
                record = banks[rate][index]
                for k, value in meta.items():
                    np.testing.assert_array_equal(value, record[k])
                a = record['availability'].to(data[7].device)
                view = trainer._prepare_view_from_primary_masks(data, a, a, bundle['dimensions'])
                captured, captured_skip = [], []
                handle = model.osram.emotion_adapter.register_forward_pre_hook(
                    lambda _, args: captured.append(args[0]))
                skip_handle = model.osram.local_skip.register_forward_hook(
                    lambda _, args, result: captured_skip.append(result))
                try:
                    full = forward(model, view)
                finally:
                    handle.remove()
                    skip_handle.remove()
                if len(captured) != 1 or len(captured_skip) != 1:
                    raise RuntimeError('Require exactly one original Flat/CHM forward')
                features = captured[0]
                ld = model.osram.latent_dim
                # Reuse the ACTUAL original skip. Recomputing it from a
                # noncontiguous concatenation slice can choose another GEMM
                # kernel and differ by ~1e-7 even with identical values.
                skip = captured_skip[0]
                def classify(x):
                    hidden = model.osram.emotion_norm(skip+model.osram.emotion_adapter(x))
                    hidden = hidden*view['umask'].T[..., None]
                    return model.smax_fc(hidden)
                torch.testing.assert_close(classify(features), full, rtol=0, atol=0)
                masked = features.clone(); masked[..., ld:] = 0
                local_only = classify(masked)
                valid = data[7].bool()
                take = lambda v: v.transpose(0, 1)[valid].cpu().numpy()
                values = dict(meta, full=take(full).reshape(-1), local=take(local_only).reshape(-1),
                    availability=take(a), history_seen=take(history_text_seen(a, data[7])))
                for k, value in values.items():
                    pieces.setdefault(k, []).append(value)
        arrays = {k: np.concatenate(v) for k, v in pieces.items()}
        if rate == '0.3':
            selected = bundle['reference_summary']['best_by_rate'][rate]
            with np.load(bundle['reference']/selected['predictions'], allow_pickle=False) as saved:
                for k in ('labels', 'conversation_ids', 'utterance_indices', 'speakers', 'availability'):
                    np.testing.assert_array_equal(arrays[k], saved[k])
                np.testing.assert_allclose(arrays['full'], saved['prediction'].reshape(-1), rtol=1e-6, atol=1e-6)
                np.testing.assert_array_equal(arrays['full'] > 0, saved['prediction'].reshape(-1) > 0)
        summaries[rate] = group_summary(*(arrays[k] for k in
            ('labels','full','local','availability','history_seen')))
        np.savez_compressed(output/f'test_rate_{rate.replace(".","p")}.npz', **arrays)
        all_arrays.append(arrays)
    pooled = {k: np.concatenate([a[k] for a in all_arrays]) for k in all_arrays[0]}
    return dict(per_rate=summaries, pooled_test_slots=group_summary(*(pooled[k] for k in
        ('labels','full','local','availability','history_seen'))))


def train_geometry(bundle, model, per_rate, output):
    import torch
    from .core import gradient_geometry, temporary_sgd, history_text_seen
    from experiments.osram_tasr_20261018.runner_helpers import isolated_rng
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    params = [p for _,p in named]
    trainer, loader, cfg = bundle['trainer'], bundle['train_loader'], bundle['cfg']
    schedules = trainer._schedules(cfg, 'train')
    records = []
    for rate in RATES:
        count = 0
        loader.sampler.set_epoch(0)
        with isolated_rng(66+91917+int(float(rate)*10)):
            for raw in loader:
                data = trainer._move_batch(raw, next(model.parameters()).device)
                background = trainer._prepare_view(data, schedules[float(rate)], 0, bundle['dimensions'])
                for b,t in choose_targets(background, cfg.dataset, rate):
                    yes = prefix_view(background,b,t,bundle['dimensions'],True)
                    no = prefix_view(background,b,t,bundle['dimensions'],False)
                    lv, mv = target_loss(model,yes,True)
                    gv = torch.autograd.grad(lv,params,allow_unused=True)
                    visible_loss = float(lv.detach())
                    del lv
                    lm, mm = target_loss(model,no,True)
                    gm = torch.autograd.grad(lm,params,allow_unused=True)
                    missing_loss = float(lm.detach())
                    del lm
                    torch.testing.assert_close(mv['memory'],mm['memory'],rtol=0,atol=0)
                    torch.testing.assert_close(mv['past_prediction'],mm['past_prediction'],rtol=0,atol=0)
                    if mv['calls'] != t+1 or mm['calls'] != t+1:
                        raise RuntimeError('A pair must execute independent read-before-write scans')
                    geometry = gradient_geometry(named,gv,gm)
                    # For MSE, loss-gradient sign can reverse just because the
                    # two predictions lie on opposite sides of the SAME label.
                    # Divide out 2*(prediction-y) to inspect prediction Jacobians
                    # without an additional backward or a new training target.
                    label = float(yes['labels'][0,t])
                    rv = mv['target_prediction']-label
                    rm = mm['target_prediction']-label
                    jacobian = None
                    if abs(rv)>1e-8 and abs(rm)>1e-8:
                        jacobian = gradient_geometry(named,
                            [g/(2*rv) if g is not None else None for g in gv],
                            [g/(2*rm) if g is not None else None for g in gm])
                        jacobian.pop('partition',None)
                    steps = {}
                    for condition, gradients in (('visible',gv),('missing',gm)):
                        with temporary_sgd(named,gradients,learning_rate=1e-5) as proof, torch.no_grad():
                            after_v,_ = target_loss(model,yes)
                            after_m,_ = target_loss(model,no)
                            steps[condition] = dict(proof,visible_loss=float(after_v),missing_loss=float(after_m),
                                delta_visible=float(after_v)-visible_loss,delta_missing=float(after_m)-missing_loss,
                                linear_cross_prediction=-1e-5*geometry['overall']['dot'])
                    record = dict(rate=rate,conversation_id=str(background['conversation_ids'][b]),
                        target=t,history_seen=bool(history_text_seen(no['availability'],no['umask'])[t,0]),
                        original_target_availability=background['availability'][t,b].detach().cpu().tolist(),
                        label=label,visible_loss=visible_loss,missing_loss=missing_loss,
                        visible_prediction=mv['target_prediction'],missing_prediction=mm['target_prediction'],
                        residual_sign_opposed=rv*rm<0,prediction_jacobian_geometry=jacobian,
                        prewrite_max_error=float((mv['memory']-mm['memory']).abs().max()),
                        past_prediction_max_error=float((mv['past_prediction']-mm['past_prediction']).abs().max()),
                        geometry=geometry,virtual_steps=steps,split='train')
                    records.append(record);count += 1
                    del gv,gm,yes,no
                    write(output/'GRADIENTS.json',records)
                    print(json.dumps(dict(event='paired_gradient',rate=rate,index=count,
                        cosine=geometry['overall']['cosine'],visible_to_missing=steps['visible']['delta_missing'],
                        missing_to_visible=steps['missing']['delta_visible'])),flush=True)
                    if count >= per_rate:
                        break
                if count >= per_rate:
                    break
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=tuple(REFERENCES),required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from .report import report_dataset
    if args.output.exists():
        raise FileExistsError('Fresh diagnostic output required')
    args.output.mkdir(parents=True)
    state = dict(status='preflight',dataset=args.dataset,seed=66,source_commit=args.source_commit,
                 pid=os.getpid(),new_training_steps=0)
    tick = time.monotonic();write(args.output/'STATUS.json',state)
    try:
        torch.set_num_threads(1)
        if args.dataset == 'CMUMOSI':
            from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker as original
        else:
            from experiments.osram_mosei_gcnet_router_20261020 import worker as original
        bundle = original.setup(args.gpu_uuid,args.output,66)
        folder = REFERENCES[args.dataset]
        summary = json.loads((folder/'SUMMARY.json').read_text())
        bundle.update(cfg=TrainConfig(**summary['identity']['config']),reference=folder,reference_summary=summary)
        if bundle['cfg'].osram_source_role_v3 or bundle['cfg'].missing_mask_protocol != 'gcnet-random-mask-v1':
            raise ValueError('Require original new-missing no-JEPA LargeFlat')
        selected = summary['best_by_rate']['0.3']; checkpoint = folder/selected['checkpoint']
        if sha(checkpoint) != selected['checkpoint_sha256']:
            raise ValueError('Protected original checkpoint changed')
        saved = torch.load(checkpoint,map_location='cpu')
        model = bundle['model'].eval();model.load_state_dict(saved['model'],strict=True);del saved
        before = h.backbone_digest(model)
        masks = torch.load(folder/'fixed_masks.pt',map_location='cpu')['test']
        state.update(status='running',reference=str(folder),checkpoint=str(checkpoint),
            checkpoint_sha256=selected['checkpoint_sha256'],selected_epoch=selected['epoch'],
            masks_sha256=sha(folder/'fixed_masks.pt'),original_digest=before,
            config=summary['identity']['config'],environment=bundle['environment'],gpu=bundle['gpu'])
        write(args.output/'STATUS.json',state)
        strata = test_strata(bundle,model,masks,args.output);write(args.output/'TEST_STRATA.json',strata)
        records = train_geometry(bundle,model,8,args.output)
        if h.backbone_digest(model) != before or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Diagnostic must restore every parameter and leave no accumulated gradients')
        state.update(status='complete',paired_targets=len(records),weights_restored_exact=True,
                     seconds=time.monotonic()-tick)
        report_dataset(args.output,strata,records,state)
    except BaseException as error:
        state.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        raise
    finally:
        write(args.output/'STATUS.json',state)


if __name__ == '__main__':
    main()
