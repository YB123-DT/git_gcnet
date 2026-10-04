"""Source-grounded complete-core residuals, independent of the withdrawn screen."""
import importlib
import torch
from torch import nn
from .meaningful_blocks_common import safe_mask
from .meaningful_input import INPUT_METHODS

FAMILIES = {
    'graph': ('rrn_evidence','egt_evidence','residual_gated_graph_evidence','pna_evidence'),
    'hypergraph': ('allset_transformer','ed_hnn','hyper_sagnn','sheaf_hypergnn_diag'),
    'grouping': ('capsule_dynamic_routing','slot_attention','otke','capsule_variational_bayes'),
    'set_context': ('perceiver_io','dgcnn_dynamic_edgeconv','graph_multiset_transformer'),
    'optimization': ('hamburger_nmf_full','crate_mssa_ista_full','equilibrium_aggregation'),
    'feature_reasoning': ('tabnet','node'),
}
ROUND1_METHODS = tuple(name for names in FAMILIES.values() for name in names)
MEANINGFUL_METHODS = ROUND1_METHODS + INPUT_METHODS


def build_core(method, latent_dim, num_heads, value_dim):
    for family, names in FAMILIES.items():
        if method in names:
            module = importlib.import_module('.meaningful_blocks_'+family, __package__)
            return getattr(module,'build_'+family)(method,latent_dim,num_heads,value_dim)
    raise ValueError('Unknown meaningful block: '+method)


class MeaningfulReadoutResidual(nn.Module):
    def __init__(self, latent_dim, context_dim, output_dim, method, num_heads, value_dim):
        super().__init__()
        if num_heads != 8 or value_dim != 64 or context_dim != 2*num_heads*value_dim:
            raise ValueError('Meaningful screen requires genuine cfg84 eight 64d forward heads')
        self.method, self.forward_dim, self.output_dim = method,num_heads*value_dim,output_dim
        self.core = build_core(method,latent_dim,num_heads,value_dim)
        self.output = nn.Linear(self.core.output_dim,output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_diagnostics = {}

    def forward(self, local,base,gap,availability,umask,flat_anchor):
        valid = umask.T.bool()
        shape = tuple(valid.shape)
        if (tuple(local.shape[:2]) != shape or base.shape != (*shape,2*self.forward_dim)
                or gap.shape != (*shape,3,2*self.forward_dim)
                or availability.shape != (*shape,3) or flat_anchor.shape != (*shape,self.output_dim)):
            raise ValueError('Meaningful readout input shapes mismatch')
        av = availability[valid]
        if not bool(((av==0)|(av==1)).all()) or not bool(av.bool().any(-1).all()):
            raise ValueError('Each valid utterance needs nonempty binary availability')
        history = valid & (valid.long().cumsum(0)>1)
        active = torch.cat((history[...,None],history[...,None] & ~availability.bool()),-1)
        evidence = torch.cat((base[...,:self.forward_dim].unsqueeze(2),gap[...,:self.forward_dim]),2)
        evidence = safe_mask(evidence,active)
        residual = torch.zeros_like(flat_anchor)
        if bool(history.any()):
            features = self.core(safe_mask(local,history)[history],evidence[history],
                                 active[history],availability[history])
            if features.shape != (int(history.sum()),self.core.output_dim):
                raise RuntimeError('Core returned wrong feature shape')
            residual[history] = self.output(features)
        with torch.no_grad():
            n = int(valid.sum())
            norms = residual[valid].norm(dim=-1)
            self.last_diagnostics = {
                'method':self.method,'valid_count':n,'history_count':int(history.sum()),
                'residual_norm':float(norms.mean()) if n else 0.,
                'residual_anchor_norm_ratio':float((norms/flat_anchor[valid].norm(dim=-1).clamp_min(1e-8)).mean()) if n else 0.,
                'active_memory_count_mean':float(active.sum(-1)[valid].float().mean()) if n else 0.,
            }
        return residual
