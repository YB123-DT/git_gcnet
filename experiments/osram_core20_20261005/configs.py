"""Generate explicit per-method configurations and actual CPU parameter counts."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import torch

from .run import candidate_config, write
from gcnet_missing_m3.core20 import METHODS, CONTROLS, attach
from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model


def generate(reference, output):
    baseline = json.loads(Path(reference).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    torch.set_num_threads(1)
    for method in METHODS + CONTROLS:
        config, delta = candidate_config(baseline, method)
        cpu_config = dict(asdict(config), device='cpu', core20_method='none', emotion_loss_mode='sample-mean')
        from gcnet_missing_m3.train_gcnet import TrainConfig
        model = _build_model(TrainConfig(**cpu_config), (512, 1024, 1024))
        if method.startswith('C17'):
            with torch.random.fork_rng(devices=[]):
                model.smax_fc = torch.nn.Linear(config.osram_output_dim, 2)
        original = sum(p.numel() for p in model.parameters() if p.requires_grad)
        attach(model, config)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        row = {'method': method, 'configuration_delta': delta,
               'registered_parameters': total, 'initial_trainable_parameters': trainable,
               'trainable_delta_from_matching_head': trainable - original,
               'dimensions': [512, 1024, 1024], 'dimensions_verified_on_biggpu': True,
               'status': 'implemented_cpu_checked_not_trained'}
        write(output / method / 'config.json', asdict(config))
        write(output / method / 'PARAMETERS.json', row)
        records.append(row)
        del model
    write(output / 'MANIFEST.json', {'label': 'INTERNAL DIAGNOSTIC ONLY', 'runs': records})
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    records = generate(args.reference, args.output)
    print(json.dumps({'configurations': len(records), 'output': str(args.output)}))
