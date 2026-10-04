"""Second-round complete mechanisms at Flat inputs, never on the Local skip.

The original read/write trajectory and original task head remain untouched.
Source decisions live in experiments/osram_meaningful20_round2_20261004.
"""
import importlib

import torch
from torch import nn

from .meaningful_blocks_common import safe_mask


INPUT_FAMILIES = {
    'attention': ('differential_attention_v1', 'flowformer_conservation',
                  'compositional_search_retrieval', 'dual_attention_symbolic_relations',
                  'dcformer_dynamic_head_composition'),
    'structured': ('conditional_rational_quadratic_spline_coupling',
                   'learned_robust_pca_evidence_decomposition',
                   'contractive_deep_equilibrium_evidence'),
    'representation': ('tokenlearner_fuser_v11', 'mbt_role_bottleneck',
                       'netvlad_residual_encoding', 'tome_merge_reconstruct'),
    'vqa': ('mcan_encoder_decoder', 'dense_coattention', 'mac_control_read_write'),
    'spectral': ('spdnet_bimap_reeig_logeig',),
    'logic': ('neural_logic_machine_evidence',),
    'sorting': ('fspool_fsunpool_evidence',),
    'circuit': ('rat_spn_evidence_circuit',),
    'pair': ('ppgn_pair_composition',),
}
INPUT_METHODS = tuple(name for names in INPUT_FAMILIES.values() for name in names)


class MeaningfulInputAdapter(nn.Module):
    def __init__(self, latent_dim, context_dim, output_dim, method, num_heads, value_dim):
        super().__init__()
        if (num_heads,value_dim,context_dim) != (8,64,1024):
            raise ValueError('Input methods require the actual cfg84 8x64 forward heads')
        self.method, self.forward_dim = method, num_heads*value_dim
        for family, names in INPUT_FAMILIES.items():
            if method in names:
                module = importlib.import_module('.meaningful_input_'+family,__package__)
                self.core = getattr(module,'build_'+family)(method,latent_dim,num_heads,value_dim)
                break
        else:
            raise ValueError('Unknown input mechanism: '+method)
        self.last_diagnostics = {}

    def forward(self, local, base, gap, availability, umask):
        valid = umask.T.bool()
        shape = tuple(valid.shape)
        if (local.shape[:2] != shape or base.shape != (*shape,1024)
                or gap.shape != (*shape,3,1024) or availability.shape != (*shape,3)):
            raise ValueError('Input adapter expects fixed Local/Base/Gap slots')
        av = availability[valid]
        if not bool(((av==0)|(av==1)).all()) or not bool(av.bool().any(-1).all()):
            raise ValueError('Valid utterances require nonempty binary availability')
        history = valid & (valid.long().cumsum(0)>1)
        active = torch.cat((valid[...,None],valid[...,None] & ~availability.bool()),-1)
        local = safe_mask(local,valid)
        base = safe_mask(base,valid)
        gap = safe_mask(gap,active[...,1:])
        evidence = torch.cat((base[...,:512].unsqueeze(2),gap[...,:512]),2)
        # Empty reads (including the first utterance) cannot create history from
        # type embeddings. This also respects an all-history readout ablation.
        history = history & evidence.ne(0).any(-1).any(-1)
        new_local, new_evidence = local.clone(), evidence.clone()
        if bool(history.any()):
            changed_local, changed_evidence = self.core(
                local[history],evidence[history],active[history],availability[history])
            if changed_local.shape != local[history].shape or changed_evidence.shape != evidence[history].shape:
                raise RuntimeError('Input mechanism changed the Flat interface')
            new_local[history] = changed_local
            new_evidence[history] = safe_mask(changed_evidence,active[history])
        new_base = torch.cat((new_evidence[...,0,:],base[...,512:]),-1)
        new_gap = torch.cat((new_evidence[...,1:,:],gap[...,512:]),-1)
        with torch.no_grad():
            delta = torch.cat(((new_local-local), (new_evidence-evidence).flatten(2)),-1)
            norms = delta[valid].norm(dim=-1)
            self.last_diagnostics = {
                'method':self.method,'placement':'flat_adapter_input_only',
                'history_count':int(history.sum()),'valid_count':int(valid.sum()),
                'input_change_norm':float(norms.mean()) if norms.numel() else 0.,
            }
        return new_local.contiguous(),new_base,new_gap
