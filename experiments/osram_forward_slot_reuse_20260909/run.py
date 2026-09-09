"""Single-variable past-slot reuse, inheriting completed causal controls."""
import argparse
import json
from pathlib import Path
import torch
from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    remote = Path('/data2/yb/remote_experiments')
    tasks = [
        ('mosi', remote / 'osram_forward_only_mosi_20260908',
         Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features')),
        *[(f'iemocap{c}', remote / 'osram_forward_only_iemocap_20260908' / f'iemocap{c}',
           Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset/IEMOCAP/features'))
          for c in (4, 6)],
    ]
    for name, reference, features in tasks:
        settings = json.loads((reference / f'seed_{args.seed}/config.json').read_text())
        assert settings['osram_bidirectional'] is False
        settings['osram_forward_slot_reuse'] = True
        output = remote / 'osram_forward_slot_reuse_20260909' / name / f'seed_{args.seed}'
        if output.exists():
            raise FileExistsError(output)
        print(f'START {name} seed={args.seed}', flush=True)
        run_experiment(TrainConfig(**settings),
                       str(features / 'wav2vec-large-c-UTT'),
                       str(features / 'deberta-large-4-UTT'),
                       str(features / 'manet_UTT'), output_dir=str(output))
        print(f'COMPLETE {name} seed={args.seed}', flush=True)


if __name__ == '__main__':
    main()
