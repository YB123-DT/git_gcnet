"""CPU regression checks; run with multimodalerc310 Python, no GPU required."""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parent

def standalone(name, filename):
    folder = ROOT / name
    for mod in ['graph', 'module']:
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location('checked_' + name, folder / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)

class GraphRegression(unittest.TestCase):
    def test_sdr_both_graphs_receive_classification_gradient(self):
        sdr = standalone('sdrgnn', 'model_SDRGNN.py')
        m = sdr.GraphModel('GRU',1024,1024,512,4,2,2,1,1,4,no_cuda=True,dropout=0)
        x = torch.randn(3,2,2560)
        logits, rec, hidden = m([x],torch.tensor([[0,1,0],[1,0,1]]),torch.ones(2,3),[3,3])
        logits.square().mean().backward()
        for graph in [m.graph_net_temporal, m.graph_net_speaker]:
            grad = graph.conv1.weight.grad
            self.assertIsNotNone(grad)
            self.assertGreater(grad.abs().sum().item(),0)
        self.assertTrue(torch.isfinite(rec[0]).all())
        self.assertEqual(m.multihead_attn.embed_dim,2560)
        # Public model input/output convention is T,B,D; attention must keep it.
        seen=[]
        handle=m.multihead_attn.register_forward_pre_hook(lambda _,args:seen.append(tuple(args[0].shape)))
        m([x],torch.tensor([[0,1,0],[1,0,1]]),torch.ones(2,3),[3,3]);handle.remove()
        self.assertEqual(seen,[(3,2,2560)])

    def test_nine_speakers_and_meld_width(self):
        for name, filename in [('gcnet','model.py'),('sdrgnn','model_SDRGNN.py')]:
            with self.subTest(model=name):
                module=standalone(name,filename)
                m=module.GraphModel('LSTM',3,4,2,4,2,9,2,2,7,no_cuda=True,dropout=0)
                logits,reconstruction,hidden=m([torch.randn(3,2,9)],torch.tensor([[0,8,4],[3,1,2]]),torch.ones(2,3),[3,3])
                self.assertEqual(tuple(logits.shape),(3,2,7))
                self.assertEqual(tuple(reconstruction[0].shape),(3,2,9))
                loss=logits.square().mean()+reconstruction[0].square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(m.graph_net_speaker.conv1.num_relations,81)

class WrapperRegression(unittest.TestCase):
    def test_wrapper_training_loss_and_graph_isolation(self):
        sys.path.insert(0,str(ROOT.parent))
        from graph_models import build_model
        for name in ['GCNet','SDR-GNN']:
            with self.subTest(model=name):
                torch.manual_seed(66)
                model=build_model({'model':name,'hidden':4,'dropout':0},[3,4,2],7,9)
                x=torch.randn(4,2,9)
                valid=torch.tensor([[1,1,1,1],[1,1,0,0]],dtype=torch.bool)
                speakers=torch.tensor([[0,8,3,4],[4,1,0,0]])
                availability=torch.ones(4,2,3);availability[0,0,0]=0
                out=model(x,speakers,valid,[4,2])
                total,parts=model.loss(out,torch.zeros(2,4,dtype=torch.long),x,availability,valid)
                total.backward()
                self.assertTrue(torch.isfinite(total))
                self.assertGreater(parts['reconstruction'].item(),0)
                model.eval()
                with torch.no_grad():
                    out=model(x,speakers,valid,[4,2])
                    changed=x.clone();changed[:,1]=1000
                    other=model(changed,speakers,valid,[4,2])
                    torch.testing.assert_close(out['logits'][:,0],other['logits'][:,0])
                    torch.testing.assert_close(out['reconstruction'][:,0],other['reconstruction'][:,0])
                from graph_models.gcnet.graph import batch_graphify
                _,edges,relations,mapping=batch_graphify(torch.randn(4,2,1,8),speakers,[4,2],9,2,2,'speaker',True)
                self.assertEqual(mapping['8:8'],80)
                self.assertEqual(len(mapping),81)
                self.assertTrue(((edges[0]<4)==(edges[1]<4)).all())

if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
