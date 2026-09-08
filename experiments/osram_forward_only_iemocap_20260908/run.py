"""Inherit each completed IEMOCAP configuration; disable future memory only."""
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
    features = Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset/IEMOCAP/features')
    for classes in (4, 6):
        reference = remote / f'osram_iemocap{classes}_20260906' / f'seed_{args.seed}'
        settings = json.loads((reference / 'config.json').read_text())
        settings['osram_bidirectional'] = False
        output = remote / 'osram_forward_only_iemocap_20260908' / f'iemocap{classes}' / f'seed_{args.seed}'
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite {output}')
        print(f'START IEMOCAP-{classes} seed={args.seed}', flush=True)
        run_experiment(TrainConfig(**settings),
                       str(features / 'wav2vec-large-c-UTT'),
                       str(features / 'deberta-large-4-UTT'),
                       str(features / 'manet_UTT'), output_dir=str(output))
        print(f'COMPLETE IEMOCAP-{classes} seed={args.seed}', flush=True)


if __name__ == '__main__':
    main()
