"""Strict reload and same-mask reproduction of each selected rate0.3 branch."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import os


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    args=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import numpy as np
    import torch
    from osram_four_block_wiring import NeuMissBlock,SignatureBlock,RidgeResidualBlock,HypothesisBridge
    from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from . import prepare
    from .run import prediction,metrics,aggregate
    from osram_four_block_wiring.wiring import csdi_prediction
    torch.set_num_threads(1);repair_uuid_rng(torch)
    output=args.root/'reload_verification';output.mkdir(exist_ok=False)
    bundle,reader,identity=prepare.setup(args.gpu_uuid,output)
    device=next(reader.parent.parameters()).device
    before=h.backbone_digest(reader.parent)
    cache=prepare.build_test_cache(bundle,reader)
    checks={}
    latent_control={}
    constructors=dict(neumiss=NeuMissBlock,signature=SignatureBlock,ridge=RidgeResidualBlock,csdi=HypothesisBridge)
    for mode,constructor in constructors.items():
        folder=args.root/('train_'+mode)
        summary=json.loads((folder/'SUMMARY.json').read_text())
        assert summary['status']=='complete'
        selected=summary['best_by_rate']['0.3']
        block=constructor().to(device).eval()
        state=torch.load(folder/selected['checkpoint'],map_location='cpu')
        assert state['parent_id']==identity['parent_id']
        block.load_state_dict(state['block'],strict=True)
        extras=dict(parent_id=identity['parent_id'])
        if mode=='ridge':
            bank=torch.load(folder/'train_support.pt',map_location='cpu')
            for field in fields(bank):
                value=getattr(bank,field.name)
                if isinstance(value,torch.Tensor):setattr(bank,field.name,value.to(device))
            extras['bank']=bank
        if mode=='csdi':
            samples=torch.load(folder/'test_hypotheses.pt',map_location='cpu')['0.3']
            # Cached hypotheses have their own recorded final denoiser state;
            # task replay uses these exact values, without fresh noise draws.
            extras.update(denoiser=None,projection=None,generator=None)
        rows=[]
        with torch.no_grad():
            for bi,row in enumerate(cache['0.3']):
                frame=row['frame'].to(device)
                if mode=='csdi':extras['samples']=samples[bi].to(device)
                pred=prediction(mode,reader,block,frame,dict(conversation_ids=row['conversation_ids']),3,extras)
                rows.append(pred.transpose(0,1)[frame.valid.T].cpu().numpy().reshape(-1))
        actual=np.concatenate(rows)
        with np.load(folder/selected['predictions'],allow_pickle=False) as expected:
            target=expected['prediction'].reshape(-1)
        np.testing.assert_allclose(actual,target,rtol=1e-5,atol=1e-6)
        np.testing.assert_array_equal(actual>0,target>0)
        checks[mode]=dict(strict_load=True,predicted_polarity_exact=True,
            max_absolute_prediction_error=float(np.abs(actual-target).max()),epoch=state['epoch'])
        if mode=='csdi':
            all_samples=torch.load(folder/'test_hypotheses.pt',map_location='cpu')
            for rate,rows in cache.items():
                selected_rate=summary['best_by_rate'][rate]
                branch_state=torch.load(folder/selected_rate['checkpoint'],map_location='cpu')
                block.load_state_dict(branch_state['block'],strict=True)
                mean_predictions=[]
                with torch.no_grad():
                    for bi,row in enumerate(rows):
                        frame=row['frame'].to(device)
                        pred=csdi_prediction(reader,None,block,None,frame,
                            samples=all_samples[rate][bi].to(device),aggregation='latent_mean')
                        mean_predictions.append(pred.transpose(0,1)[frame.valid.T].cpu().numpy().reshape(-1))
                with np.load(folder/selected_rate['predictions'],allow_pickle=False) as original:
                    measured=metrics(bundle['trainer'],original['labels'],np.concatenate(mean_predictions),original['base_prediction'])
                latent_control[rate]=measured
    assert before==h.backbone_digest(reader.parent)
    h.write_json(output/'RESULT.json',dict(status='passed',methods=checks,parent_unchanged=True,train_steps=0,
        csdi_latent_mean_same_weights=latent_control,
        csdi_latent_mean_aggregate=aggregate(latent_control)))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
