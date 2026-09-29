"""One-epoch integration check only; not a formal training launcher."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
    from experiments.osram_cfg84_history_query_random_20260928.run import verify_outputs
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('verification is assigned to biggpu host GPU0 only')
    output = Path('/data2/yb/remote_experiments/osram_cfg84_post_grn_20260929/smoke/seed_66')
    reference = Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66')
    original = TrainConfig(**json.loads((reference / 'config.json').read_text()))
    assert original.training_objective == 'emotion-only' and original.train_rate_mode == 'cyclic'
    assert original.osram_readout_fusion == 'flat' and not original.osram_post_grn
    assert not original.osram_history_query_adapter and original.completion_path == 'none'
    assert original.osram_output_dim == 1600 and not original.osram_bidirectional
    config = replace(original, osram_post_grn=True, epochs=1)
    output.mkdir(parents=True, exist_ok=False)
    files = ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/model.py',
             'gcnet_missing_m3/train_gcnet.py',
             'experiments/osram_cfg84_post_grn_20260929/verify.py')
    provenance = dict(status='running', smoke_only=True, epochs=1,
        started_utc=datetime.now(timezone.utc).isoformat(), server='biggpu',
        host_gpu=0, gpu_uuid='GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45',
        source_sha256={name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in files},
        reference_config_sha256=hashlib.sha256((reference / 'config.json').read_bytes()).hexdigest(),
        semantic_delta={'osram_post_grn': True},
        torch_version=torch.__version__, python_version=sys.version,
        label='ONE-EPOCH IMPLEMENTATION CHECK ONLY; NOT A PERFORMANCE RESULT')
    def save():
        (output / 'VERIFICATION.json').write_text(json.dumps(provenance, indent=2)+'\n')
    save()
    roots = tuple(str(Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features') / name)
                  for name in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT'))
    torch.set_num_threads(2)
    try:
        metrics = run_experiment(config, *roots, output_dir=str(output))
        verify_outputs(output, reference)
        history = json.loads((output / 'history.json').read_text())
        assert len(history) == 1 and history[0]['train']['jepa_loss'] == 0
        for row in [history[0]['train'], *metrics['test'].values()]:
            assert row['post_grn']['valid_count'] > 0
            assert 0 <= row['post_grn']['gate_mean'] <= 1
            assert 0 <= row['post_grn']['gate_saturation_fraction'] <= 1
        provenance.update(status='complete', canonical_masks_match=True,
            completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        provenance.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
