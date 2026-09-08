"""Run the approved single-variable memory diagnostic from inherited config."""
import argparse
import json
from pathlib import Path

import torch
from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    reference = Path('/data2/yb/remote_experiments/osram_query_no_availability_mosi_20260907')
    settings = json.loads((reference / f'seed_{args.seed}' / 'config.json').read_text())
    settings.update(osram_query_availability=True, osram_bidirectional=False)
    cfg = TrainConfig(**settings)
    torch.set_num_threads(6)
    root = Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features')
    output = Path('/data2/yb/remote_experiments/osram_forward_only_mosi_20260908')
    run_experiment(cfg, str(root / 'wav2vec-large-c-UTT'),
                   str(root / 'deberta-large-4-UTT'), str(root / 'manet_UTT'),
                   output_dir=str(output / f'seed_{args.seed}'))


if __name__ == '__main__':
    main()
