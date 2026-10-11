import torch
from experiments.osram_nested_root_probe_20261011.capture import RootCollector
from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter

def test_root_capture_is_before_decoder_and_masks_slots():
    torch.set_num_threads(1)
    block=MeaningfulInputAdapter(256,1024,1600,'nested_gnn_rooted_evidence',8,64)
    local=torch.randn(3,1,256)
    base=torch.cat((torch.randn(3,1,512),torch.zeros(3,1,512)),-1)
    gap=torch.cat((torch.randn(3,1,3,512),torch.zeros(3,1,3,512)),-1)
    availability=torch.tensor([[[1,0,1]]]).expand(3,-1,-1)
    umask=torch.tensor([[1,1,0]])
    with RootCollector(block) as collector:
        block(local,base,gap,availability,umask)
    root,decoder,active=collector.finish()
    assert root.shape==(3,1,2112) and decoder.shape==root.shape
    assert not root[0].count_nonzero() and not root[2].count_nonzero()
    assert root[1].count_nonzero() and decoder[1].count_nonzero()
    slots=root.reshape(3,1,33,64)
    assert not slots[1,0,9:17].count_nonzero() and not slots[1,0,25:33].count_nonzero()
    assert active[1,0].tolist()==[True]+[True]*8+[False]*8+[True]*8+[False]*8
    assert torch.isfinite(root).all()

def test_test_labels_do_not_change_probe_weights(tmp_path):
    import numpy as np
    from experiments.osram_nested_root_probe_20261011.probe import run_probes
    rng=np.random.default_rng(66)
    splits={}
    for name in ('train','test'):
        splits[name]={'flat':rng.normal(size=(4,1600)).astype('float32'),
            'rooted':rng.normal(size=(4,2112)).astype('float32'),
            'predecoder':rng.normal(size=(4,2112)).astype('float32'),
            'role_active':np.ones((4,33),bool),'labels':np.array([1.,-1.,1.,-1.],dtype='float32'),
            'availability':np.tile([0,1,0],(4,1)),
            'conversation_ids':np.array([name+str(i) for i in range(4)]),
            'utterance_indices':np.ones(4,dtype=int)}
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    first=run_probes(splits,a,epochs=2)
    splits['test']['labels']=-splits['test']['labels']
    run_probes(splits,b,epochs=2)
    assert all(r['parameter_count']==118945 for r in first)
    for path in a.glob('seed*/*/final.pt'):
        x=torch.load(path,weights_only=False)['model']
        y=torch.load(b/path.relative_to(a),weights_only=False)['model']
        assert all(torch.equal(v,y[k]) for k,v in x.items())
