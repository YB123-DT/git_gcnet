"""Verify finished scores, shared cohorts/masks and equal training mask traces."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    from gcnet_missing_m3 import train_gcnet as trainer
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from .run import metrics,aggregate
    root=args.root
    queue=json.loads((root/'QUEUE.json').read_text())
    assert queue['status']=='complete'
    summary={m:json.loads((root/('train_'+m)/'SUMMARY.json').read_text())
             for m in ('neumiss','signature','ridge','csdi')}
    baseline=None;training_masks=None;parent=None;cohorts={};details={}
    for mode,run in summary.items():
        assert run['status']=='complete' and run['epoch']==150
        assert all(run['checks'].values())
        assert not run['identity']['smoke'] and run['identity']['max_batches'] is None
        if parent is None:parent=run['identity']['parent_id']
        assert parent==run['identity']['parent_id']
        if baseline is None:baseline=run['baseline']
        assert baseline==run['baseline']
        folder=root/('train_'+mode)
        history=json.loads((folder/'history.json').read_text())
        assert len(history)==150
        trace=[row['mask_hash'] for row in history]
        if training_masks is None:training_masks=trace
        assert training_masks==trace,'Training masks/order differ: '+mode
        actual={}
        for rate,best in run['best_by_rate'].items():
            assert 1<=best['epoch']<=150
            assert h.sha(folder/best['checkpoint'])==best['checkpoint_sha256']
            assert h.sha(folder/best['predictions'])==best['prediction_sha256']
            with np.load(folder/best['predictions'],allow_pickle=False) as arrays:
                scored=metrics(trainer,arrays['labels'],arrays['prediction'],arrays['base_prediction'])
                for key,value in scored.items():
                    np.testing.assert_allclose(value,best['metrics'][key],rtol=0,atol=1e-10)
                keys=('labels','conversation_ids','utterance_indices','speakers','availability','base_prediction')
                if rate not in cohorts:cohorts[rate]={k:arrays[k].copy() for k in keys}
                for key in keys:np.testing.assert_array_equal(arrays[key],cohorts[rate][key])
            actual[rate]=scored
        assert len(actual)==8
        for scope,row in aggregate(actual).items():
            for key,value in row.items():
                np.testing.assert_allclose(value,run['aggregate'][scope][key],rtol=0,atol=1e-10)
        details[mode]=dict(selected_epoch={r:b['epoch'] for r,b in run['best_by_rate'].items()},
                           aggregate=run['aggregate'],parameters=run['identity']['module_parameters'])
    result=dict(status='passed',parent_id=parent,training_masks_equal_all150epochs=True,
        test_masks_labels_cohorts_parent_predictions_exact=True,
        saved_metrics_recomputed=True,selected_artifact_hashes_verified=True,
        baseline=aggregate(baseline),methods=details)
    h.write_json(root/'VERIFICATION.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
