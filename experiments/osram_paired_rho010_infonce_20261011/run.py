"""One fixed seed66 run: retain rho.10 task-only protocol, add InfoNCE.1."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
BACKBONE_COMMIT = '6e2a998'
GPU_UUID = 'GPU-e4cafb17-818e-216a-b94a-7440063a9153'


def configuration(control):
    assert control['seed']==66 and control['epochs']==100
    assert control['paired_history_views'] and control['history_contrast_weight']==0
    assert control['history_task_view2_weight']==.1 and control['history_drop_prob']==.2
    assert control['history_contrast_temperature']==.1
    return dict(control,history_contrast_weight=.1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runner-commit',required=True)
    a=p.parse_args()
    from experiments.osram_paired_history_views_20261001.run import common,read,write,sha,aggregate_diagnostics
    from experiments.osram_memory_only_adapter_20261001.run import rate_scores
    actual=subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert actual[0].strip()==GPU_UUID==os.environ.get('CUDA_VISIBLE_DEVICES')
    assert int(actual[1])>6000
    prior=read(a.control/'PROVENANCE.json')
    assert prior['status']=='complete' and prior['code_commit']==BACKBONE_COMMIT
    hashes={k:sha(ROOT/k) for k in prior['source_sha256']}
    assert hashes==prior['source_sha256'], 'Use exact historical implementation, not current modified backbone'
    config=configuration(read(a.control/'config.json'))
    a.output.mkdir(parents=True,exist_ok=False)
    record=dict(status='training',label='INTERNAL TEST-ORACLE DIAGNOSTIC ONLY',
        server='biggpu',physical_gpu=6,gpu_uuid=GPU_UUID,pid=os.getpid(),
        backbone_commit=BACKBONE_COMMIT,runner_commit=a.runner_commit,
        source_sha256=hashes,runner_sha256=sha(Path(__file__)),config=config,
        control=str(a.control),from_scratch=True,
        started_at=datetime.now(timezone.utc).isoformat())
    write(a.output/'PROVENANCE.json',record)
    try:
        import torch
        from gcnet_missing_m3.train_gcnet import TrainConfig,run_experiment
        torch.set_num_threads(2)
        dataset=Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset')
        os.environ['GCNET_DATASET_ROOT']=str(dataset)
        features=[str(dataset/'CMUMOSI/features'/n) for n in
            ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
        run_experiment(TrainConfig(**config),*features,output_dir=str(a.output))
        common.verify_outputs(a.output,a.control)
        history=read(a.output/'history.json')
        control_history=read(a.control/'history.json')
        assert len(history)==len(control_history)==100
        assert all(left['train']['paired_history'][k]==right['train']['paired_history'][k]
            for left,right in zip(history,control_history)
            for k in ('view1_mask_sha256','view2_mask_sha256'))
        scores=rate_scores(read(a.output/'metrics.json'))
        control_scores=rate_scores(read(a.control/'metrics.json'))
        summary=dict(label=record['label'],scores=scores,control_scores=control_scores,
            delta_vs_control={k:scores[k]-control_scores[k] for k in ('mean_8rate','high_missing')},
            training_counts=aggregate_diagnostics([r['train']['paired_history'] for r in history]),
            all_training_mask_hashes_match_control=True)
        write(a.output/'SUMMARY.json',summary)
        record['status']='complete'
        print(json.dumps(summary),flush=True)
    except BaseException as error:
        record.update(status='failed',error=repr(error))
        raise
    finally:
        record['updated_at']=datetime.now(timezone.utc).isoformat()
        write(a.output/'PROVENANCE.json',record)


if __name__=='__main__':
    main()
