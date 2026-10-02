"""Investigate packed-projector floating point sensitivity; never changes models."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.osram_history_drift_20261002.run import DEFAULT_DATA, DEFAULT_REFERENCE, verify_gpu
from experiments.osram_history_drift_20261002.diagnostic import capture, nested_view


def drift(a, b, mask):
    if not mask.any():
        return dict(count=0, absmax=0., relative_l2_max=0.)
    a, b = a[mask].double().flatten(1), b[mask].double().flatten(1)
    return dict(count=len(a), absmax=(a-b).abs().max().item(),
                relative_l2_max=((a-b).norm(dim=-1)/(a.norm(dim=-1)+1e-12)).max().item())


def main():
    import torch
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules, _move_batch, _prepare_view
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    gpu = verify_gpu()
    torch.set_num_threads(2)
    os.environ['GCNET_DATASET_ROOT'] = str(DEFAULT_DATA)
    config = TrainConfig(**json.loads((DEFAULT_REFERENCE/'seed_66/config.json').read_text()))
    features = [str(DEFAULT_DATA/'CMUMOSI/features'/name) for name in
                ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    _, val, _, *dimensions = get_loaders(audio_root=features[0], text_root=features[1],
        video_root=features[2], num_folder=1, dataset='CMUMOSI', batch_size=config.batch_size,
        num_workers=0, seed=66, validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol)
    loader = val[0]
    loader.sampler.set_epoch(0)
    view = _prepare_view(_move_batch(next(iter(loader)), torch.device('cuda:0')),
                         _schedules(config,'validation')[0.], 0, tuple(dimensions))
    model = _build_model(config, tuple(dimensions)).cuda().requires_grad_(False).eval()
    state = torch.load(DEFAULT_REFERENCE/'seed_66/best_miss_0p0.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    result = dict(gpu=gpu, torch=torch.__version__, records=[])
    with torch.no_grad():
        for precision in ('float32', 'float64'):
            current = model if precision == 'float32' else copy.deepcopy(model).double()
            first = {k: (v.double() if precision == 'float64' and isinstance(v,torch.Tensor) and v.is_floating_point() else v)
                     for k,v in view.items()}
            def snap(v):
                return capture(current, lambda: current([v['incomplete']],v['availability'],v['qmask'],
                    v['umask'],v['lengths'],predict_missing=False))
            base = snap(first)
            _, latents1 = current.observed_set(first['incomplete'], first['availability'], first['umask'])
            valid = first['umask'].T.bool()
            for mode in ('A','T','V','mixed'):
                availability2, meta = nested_view(first['availability'], first['umask'], mode, 66000000, .2)
                expanded = torch.repeat_interleave(availability2,
                    torch.tensor(dimensions, device=availability2.device), dim=-1)
                second = dict(first, availability=availability2,
                              incomplete=torch.where(expanded.bool(), first['incomplete'], 0.))
                altered = snap(second)
                _, latents2 = current.observed_set(second['incomplete'], availability2, second['umask'])
                anchor = meta['contrast_mask']
                prefix = valid & (meta['changed'].long().cumsum(0)==0)
                row = dict(precision=precision,mode=mode,
                    anchor_local=drift(base['local'],altered['local'],anchor),
                    prefix={key:drift(base[key],altered[key],prefix) for key in base}, modalities={})
                offset = 0
                for index,(name,width) in enumerate(zip(current.observed_set.projectors, dimensions)):
                    select1 = valid & first['availability'][...,index].bool()
                    select2 = valid & availability2[...,index].bool()
                    mask = anchor & select1
                    info = dict(packed_rows1=int(select1.sum()),packed_rows2=int(select2.sum()),
                                latent_drift=drift(latents1[name],latents2[name],mask))
                    positions = mask.nonzero(as_tuple=False)
                    # Identical geometry for both evaluations removes variable packed-row count.
                    max_diff = 0.
                    exact_input = True
                    for t,b in positions.cpu().tolist():
                        a = first['incomplete'][t,b,offset:offset+width].unsqueeze(0).contiguous()
                        c = second['incomplete'][t,b,offset:offset+width].unsqueeze(0).contiguous()
                        exact_input &= torch.equal(a,c)
                        p = current.observed_set.projectors[name]
                        max_diff = max(max_diff, (p(a)-p(c)).abs().max().item())
                    info.update(singleton_input_exact=exact_input,singleton_output_absmax=max_diff)
                    row['modalities'][name] = info
                    offset += width
                result['records'].append(row)
                print(json.dumps(row),flush=True)
            if precision == 'float64':
                del current
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)


if __name__ == '__main__':
    main()
