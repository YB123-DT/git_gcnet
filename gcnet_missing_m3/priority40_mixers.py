"""M31–M35: five fixed current-utterance roles -> one dim-wide feature.

Source checks (2026-10-04), independently implemented without source dependencies:
M31 arXiv:2105.01601 §2; google-research/vision_transformer,
    vit_jax/models_mixer.py::MixerBlock.
M32 arXiv:2105.08050 Figure 1's author gmlp_block/spatial_gating_unit pseudocode.
M33 arXiv:2105.03824; google-research/google-research,
    f_net/layers.py::FourierTransform and EncoderBlock.
M34 arXiv:2301.00808; facebookresearch/ConvNeXt-V2,
    models/convnextv2.py::Block and models/utils.py::GRN.
    Transfers the expansion/GELU/GRN/projection branch, not an image convolution
    or FCMAE objective, and not this repository's unrelated PostGRN gate.
M35 arXiv:2201.12083; ziyuwwang/DynaMixer,
    models/dynamixer.py::DynaMixerOp (not the two-dimensional image block).

Masking is the OSRAM adaptation: absent roles remain fixed zero slots, never
packed into a different sequence. No batch/time mixing or persistent state.
The parent adapter owns the zero-initialized 1600-dimensional output bridge.
"""
import torch
from torch import nn
from torch.nn import functional as F


METHODS = ("m31_mlp_mixer", "m32_gmlp_sgu", "m33_fnet",
           "m34_convnext_grn", "m35_dynamixer")


def masked(x, active):
    return torch.where(active[..., None], x, torch.zeros_like(x))


class RoleCore(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tokens, active):
        if tokens.ndim != 3 or tokens.shape[1:] != (5, self.dim):
            raise ValueError(f"Expected tokens [N,5,{self.dim}]")
        if active.shape != tokens.shape[:2] or active.dtype != torch.bool:
            raise ValueError("Expected boolean active [N,5]")
        output = masked(self.forward_tokens(masked(tokens, active), active), active)
        return output.sum(1) / active.sum(1, keepdim=True).clamp_min(1).to(output.dtype)


class MLPMixer(RoleCore):
    def __init__(self, dim=64):
        super().__init__(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(nn.Linear(5, 10), nn.GELU(), nn.Linear(10, 5))
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward_tokens(self, x, active):
        values = masked(self.token_norm(x), active).transpose(1, 2)
        x = masked(x + self.token_mlp(values).transpose(1, 2), active)
        return masked(x + self.channel_mlp(masked(self.channel_norm(x), active)), active)


class GMLP(RoleCore):
    def __init__(self, dim=64):
        super().__init__(dim)
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 2 * dim)
        self.gate_norm = nn.LayerNorm(dim)
        self.spatial = nn.Linear(5, 5)
        nn.init.normal_(self.spatial.weight, std=1e-4)
        nn.init.ones_(self.spatial.bias)
        self.project = nn.Linear(dim, dim)

    def forward_tokens(self, x, active):
        u, v = masked(F.gelu(self.expand(masked(self.norm(x), active))), active).chunk(2, -1)
        v = masked(self.gate_norm(v), active)
        gate = self.spatial(v.transpose(1, 2)).transpose(1, 2)
        return masked(x + self.project(u * gate), active)


class FNet(RoleCore):
    def __init__(self, dim=64):
        super().__init__(dim)
        self.mix_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.output_norm = nn.LayerNorm(dim)

    @staticmethod
    def fourier(x):
        # FFT is ONLY across five roles and channels, never the batch axis.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return torch.fft.fft2(work, dim=(-2, -1)).real.to(x.dtype)

    def forward_tokens(self, x, active):
        x = masked(self.mix_norm(x + self.fourier(x)), active)
        return masked(self.output_norm(x + self.ffn(x)), active)


class ConvNeXtGRN(RoleCore):
    def __init__(self, dim=64):
        super().__init__(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.expand = nn.Linear(dim, 4 * dim)
        self.gamma = nn.Parameter(torch.zeros(4 * dim))
        self.beta = nn.Parameter(torch.zeros(4 * dim))
        self.project = nn.Linear(4 * dim, dim)

    def response_norm(self, x, active):
        x = masked(x, active)
        energy = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        relative = energy / (energy.mean(-1, keepdim=True) + 1e-6)
        return masked(x + self.gamma * x * relative + self.beta, active)

    def forward_tokens(self, x, active):
        expanded = masked(F.gelu(self.expand(masked(self.norm(x), active))), active)
        return masked(x + self.project(self.response_norm(expanded, active)), active)


class DynaMixer(RoleCore):
    def __init__(self, dim=64):
        super().__init__(dim)
        if dim % 2:
            raise ValueError("DynaMixer requires an even dim for two channel segments")
        self.segments = 2
        self.reduced_dim = 4
        self.compress = nn.Linear(dim, self.segments * self.reduced_dim)
        self.generate = nn.Linear(5 * self.reduced_dim, 5 * 5)
        self.project = nn.Linear(dim, dim)

    def mixing_weights(self, x, active):
        compressed = masked(self.compress(masked(x, active)), active)
        compressed = compressed.reshape(-1, 5, 2, 4).permute(0, 2, 1, 3).flatten(2)
        logits = self.generate(compressed).reshape(-1, 2, 5, 5)
        source = active[:, None, :, None]
        # Finite sentinel keeps all-empty rows defined; zero their weights next.
        logits = logits.masked_fill(~source, torch.finfo(logits.dtype).min)
        weights = torch.where(source, logits.softmax(-2), torch.zeros_like(logits))
        return torch.where(active[:, None, None, :], weights, torch.zeros_like(weights))

    def forward_tokens(self, x, active):
        weights = self.mixing_weights(x, active)
        values = masked(x, active).reshape(-1, 5, 2, self.dim // 2).permute(0, 2, 3, 1)
        mixed = (values @ weights).permute(0, 3, 1, 2).reshape(-1, 5, self.dim)
        return masked(self.project(mixed), active)


def build(method, dim=64):
    constructors = dict(zip(METHODS, (MLPMixer, GMLP, FNet, ConvNeXtGRN, DynaMixer)))
    if method not in constructors:
        raise ValueError("Unknown priority40 mixer: " + method)
    return constructors[method](dim)
