"""Frozen original Flat with both classification Local paths removed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    local_keeps = (1., 0.)
    import numpy as np
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules, evaluate_rate
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model, _roots

    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    torch.set_num_threads(2)
    source = Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full')
    run_name = 'osram_cfg84_no_local_20260929'
    output = Path('/data1/yb/remote_experiments') / run_name / 'results'
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    save('provenance.json', dict(server='biggpu', host_gpu=0,
        gpu_uuid='GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45', source=str(source),
        code_snapshot=str(ROOT), torch=torch.__version__, seed_list=[66,67,68],
        protocol='random-missing; frozen original per-rate Test-oracle checkpoints',
        intervention='zero emotion_adapter Local slice AND zero local_skip output; preserve Base/masked Gap and Memory',
        local_keep=list(local_keeps), training=False,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT/'gcnet_missing_m3/osram.py',
                      ROOT/'gcnet_missing_m3/model.py', ROOT/'gcnet_missing_m3/train_gcnet.py']}))
    for seed in (66,67,68):
        path = source / f'seed_{seed}'
        cfg_dict = json.loads((path/'config.json').read_text())
        c = TrainConfig(**cfg_dict)
        assert c.osram_readout_fusion == 'flat' and not c.osram_post_grn
        assert c.training_objective == 'emotion-only' and c.completion_path == 'none'
        assert c.train_rate_mode == 'cyclic' and not c.osram_history_query_adapter
        expected = json.loads((path/'metrics.json').read_text())
        save(f'config_{seed}.json', cfg_dict)
        _,_,loaders,ad,td,vd = get_loaders(audio_root=_roots()[0],text_root=_roots()[1],
            video_root=_roots()[2],num_folder=1,dataset=c.dataset,batch_size=c.batch_size,
            num_workers=0,seed=seed,validation_fraction=c.validation_fraction,
            evaluation_protocol=c.evaluation_protocol)
        dims=(ad,td,vd); model=_build_model(c,dims).cuda().eval()
        model.requires_grad_(False)
        schedules=_schedules(c,'test')
        for rate in [i/10 for i in range(8)]:
            key=f'{rate:.1f}'
            checkpoint=path/f'best_miss_{key.replace(".","p")}.pt'
            state=torch.load(checkpoint,map_location='cpu')
            model.load_state_dict(state['model'],strict=True)
            reference=None
            for local_keep in local_keeps:
                digest=hashlib.sha256()
                def scale(module,args):
                    x=args[0]
                    assert x.shape[-1] == c.latent_dim+4*model.osram.context_dim
                    assert torch.isfinite(x).all()
                    digest.update(x.detach().cpu().contiguous().numpy().tobytes())
                    if local_keep == 1.: return args
                    # Keep history exactly; remove only the Local concatenation slice.
                    y=x.clone(); y[...,:c.latent_dim]=0.
                    assert torch.equal(y[...,c.latent_dim:],x[...,c.latent_dim:])
                    assert y[...,:c.latent_dim].count_nonzero()==0
                    return (y,)
                hook=model.osram.emotion_adapter.register_forward_pre_hook(scale)
                skip_hook = model.osram.local_skip.register_forward_hook(
                    lambda module, args, out: torch.zeros_like(out)) if local_keep == 0. else None
                try:
                    with torch.no_grad():
                        metrics, artifacts=evaluate_rate(model,loaders[0],schedules[rate],
                            c.dataset,dims,torch.device('cuda:0'),True,c.mosi_task_mode,
                            c.task_regression_loss,c.task_smooth_l1_beta)
                finally:
                    hook.remove()
                    if skip_hook is not None: skip_hook.remove()
                if local_keep == 1.:
                    assert abs(metrics['weighted_f1']-expected['test'][key]['weighted_f1'])<1e-10
                    reference=(digest.hexdigest(),metrics['mask_sha256'],artifacts['labels'].copy())
                assert digest.hexdigest()==reference[0], 'upstream Local/Memory read values changed'
                assert metrics['mask_sha256']==reference[1]
                assert np.array_equal(artifacts['labels'],reference[2])
                np.savez_compressed(output/f'seed_{seed}_miss_{key}_local_keep_{local_keep}.npz',**artifacts)
                row=dict(seed=seed,rate=rate,local_keep=local_keep,epoch=int(state['epoch']),
                    checkpoint=str(checkpoint),upstream_sha256=digest.hexdigest(),metrics=metrics)
                rows.append(row);save('rows.json',rows)
                print(f'seed={seed} rate={rate} local_keep={local_keep} wf1={100*metrics["weighted_f1"]:.6f}',flush=True)
    summary={}
    for local_keep in sorted(local_keeps):
        seed_means=[statistics.mean(100*r['metrics']['weighted_f1'] for r in rows
            if r['local_keep']==local_keep and r['seed']==seed) for seed in (66,67,68)]
        summary[str(local_keep)]=dict(mean=statistics.mean(seed_means),sample_sd=statistics.stdev(seed_means),
            seed_means=seed_means,per_rate={str(rate):statistics.mean(100*r['metrics']['weighted_f1']
                for r in rows if r['local_keep']==local_keep and r['rate']==rate) for rate in [i/10 for i in range(8)]})
    save('SUMMARY.json',dict(status='complete',label='INTERNAL TEST-ORACLE; EVALUATION ONLY',
        rows=len(rows),summary=summary))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
