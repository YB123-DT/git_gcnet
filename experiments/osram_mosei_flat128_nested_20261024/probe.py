"""Bounded real-batch correctness/resource probe, never a performance result."""
import argparse
from dataclasses import replace
import gc
import json
import math
import os
from pathlib import Path
import time


def choose_probe_batches(records):
    """Cover the largest padded rectangle and the most valid utterances."""
    if not records:
        raise ValueError('No real batches to profile')
    return sorted({max(range(len(records)), key=lambda i: records[i][7].numel()
                       if hasattr(records[i][7], 'numel') else records[i][7].size),
                   max(range(len(records)), key=lambda i: float(records[i][7].sum()))})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from .run import GPU0, validate_gpu_uuid
    validate_gpu_uuid(args.gpu_uuid)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import torch
    from . import config, worker
    from experiments.osram_nested_cmerc_tuning_20261023 import run as common
    from experiments.osram_nested_cmerc_tuning_20261023 import worker as shared
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_20261019 import run as recovery
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    root = Path(__file__).resolve().parents[2]
    manifest = common.verify_source_snapshot(root, args.source_commit)
    extension = config.verify_registry_extension(root)
    if args.output.exists():
        raise FileExistsError('Probe must preserve existing outputs')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    bundle, plain = worker.setup('CMUMOSEI', config.method_name('N01'), args.gpu_uuid,
        args.output/'bootstrap', seed=66, fold=1, variant='N01')
    original = replace(bundle['cfg'], osram_adapter_hidden_dim=0, osram_meaningful_block='none', seed=66)
    trainer = bundle['trainer']
    bundle['train_loader'].sampler.set_epoch(0)
    raw = list(bundle['train_loader'])
    indices = choose_probe_batches(raw)
    first = bundle.pop('model')
    results = {}
    for variant in ('N01', 'N20', 'N21', 'F67'):
        tick = time.monotonic()
        cfg = config.candidate_config(original, variant)
        if variant == 'N01':
            model = first
            first = None
        else:
            model = _build_model(cfg, bundle['dimensions'], torch.device('cuda:0'))
        if variant == 'F67':
            del plain
            plain = _build_model(replace(cfg, osram_meaningful_block='none'),
                                 bundle['dimensions'], torch.device('cuda:0'))
        model.requires_grad_(True)
        schedules = trainer._schedules(cfg, 'train')
        # The two real geometry extremes and rates0/.7 bracket active gap/root counts.
        views = [trainer._prepare_view(trainer._move_batch(raw[i], torch.device('cuda:0')),
                    schedules[rate], 0, bundle['dimensions']) for i in indices for rate in (0., .7)]
        model.eval(); plain.eval()
        with torch.no_grad():
            parity = shared.prediction_parity(h.forward(plain, views[-1]).cpu().numpy(),
                                             h.forward(model, views[-1]).cpu().numpy(), scalar=True)
        optimizer = scalar.make_optimizer(model, cfg, 'flat')
        observed, handles = {}, []
        def capture(name):
            def hook(gradient):
                if not torch.isfinite(gradient).all():
                    raise RuntimeError('Nonfinite task gradient: '+name)
                observed[name] = observed.get(name, False) or bool(gradient.ne(0).any())
            return hook
        if variant.startswith('N'):
            handles = [p.register_hook(capture(name)) for name,p in
                       model.osram.meaningful_block.core.named_parameters()]
        scans = [0]
        handles.append(model.osram.register_forward_hook(lambda *unused: scans.__setitem__(0,scans[0]+1)))
        torch.cuda.reset_peak_memory_stats()
        model.train()
        updates = [scalar.optimize_view(model, optimizer, view, cfg, 'flat', trainer) for view in views]
        if variant.startswith('N'):
            shared.check_nested_gradients(observed)
        if scans[0] != len(updates):
            raise RuntimeError('More than one Memory scan per task update')
        for handle in handles:
            handle.remove()
        identity = dict(variant=variant, config=cfg.__dict__, source_commit=args.source_commit)
        state = recovery.checkpoint_payload(model, optimizer, epoch=1, identity=identity,
                                            history=updates, best_key=None)
        if len(state['rng']['cuda']) != 1:
            raise RuntimeError('UUID CUDA RNG missing from full recovery')
        path = args.output/(variant+'_full_recovery.pt')
        torch.save(state, path)
        model.eval()
        with torch.no_grad():
            expected = h.forward(model, views[-1]).detach().clone()
        saved = torch.load(path, map_location='cpu', weights_only=False)
        recovery.restore(model, optimizer, saved, identity)
        with torch.no_grad():
            actual = h.forward(model, views[-1])
        if not torch.equal(expected, actual):
            raise RuntimeError('Checkpoint/Adam/RNG restore prediction mismatch')
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()/2**20
        # Historical complete-phase peaks are <=3039MiB allocated for larger
        # 8.36M models. Include CUDA context/allocator reserve, with a4GiB floor.
        reserve = max(4096, math.ceil((max(peak, 3039)+1024)/256)*256)
        results[variant] = dict(reserve_mib=reserve, peak_allocated_mib=peak,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            batches=indices, geometry=[dict(shape=list(raw[i][7].shape),
                valid=int(raw[i][7].sum())) for i in indices],
            task_updates=updates, nested_gradient_paths=observed,
            memory_scans=scans[0], zero_init_parity=parity,
            full_model_adam_rng_reload_exact=True, cuda_rng_states=1,
            parameters=sum(p.numel() for p in model.parameters()), seconds=time.monotonic()-tick)
        h.write_json(args.output/'PROGRESS.json', results)
        print(json.dumps(dict(variant=variant, peak_mib=peak, reserve_mib=reserve,
                              seconds=results[variant]['seconds'])), flush=True)
        del model, optimizer, views, state, saved, expected, actual
        gc.collect(); torch.cuda.empty_cache()
    if common.verify_source_snapshot(root, args.source_commit) != manifest:
        raise RuntimeError('Source changed during probe')
    h.write_json(args.output/'RESOURCE_PROOF.json', dict(status='passed',
        source_commit=args.source_commit, source_manifest_sha256=manifest['manifest_sha256'],
        registry_proof=extension, gpu_uuid=args.gpu_uuid, variants=results,
        default_reserve_mib=max(row['reserve_mib'] for row in results.values()),
        scope='Bounded real maximum-rectangle/maximum-valid batches; not training results',
        reserve_basis='Measured batch peaks plus1024MiB;3039MiB historical full-phase upper anchor from larger8.36M model;4GiB minimum'))


if __name__ == '__main__':
    main()
