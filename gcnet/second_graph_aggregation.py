"""Second-layer graph aggregators compatible with the locked GCNet stack."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GraphConv
from torch_scatter import scatter_add, scatter_mean


class CompatibleMish(nn.Module):
    """Mish without relying on ``nn.Mish`` from newer Torch releases."""

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class _ScalarMap(nn.Module):
    """Apply an MLP to the final scalar-map dimension."""

    def __init__(self, dimensions):
        super().__init__()
        modules = []
        for position, (input_size, output_size) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            modules.append(nn.Linear(input_size, output_size))
            if position < len(dimensions) - 2:
                modules.append(nn.BatchNorm1d(output_size))
                modules.append(CompatibleMish())
        self.layers = nn.Sequential(*modules)

    def reset_parameters(self):
        for module in self.layers:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                module.reset_parameters()

    def forward(self, values):
        shape = values.shape[:-1]
        if values.numel() == 0:
            output_size = self.layers[-1].out_features
            return values.new_empty(shape + (output_size,))
        mapped = self.layers(values.reshape(-1, values.size(-1)))
        return mapped.reshape(shape + (mapped.size(-1),))


class _LegacyNamedGraphConv(GraphConv):
    """Expose the PyG 2.0.1 ``lin_l``/``lin_r`` parameter contract."""

    def _use_legacy_linear_names(self):
        if "lin_rel" in self._modules:
            self.lin_l = self._modules.pop("lin_rel")
        if "lin_root" in self._modules:
            self.lin_r = self._modules.pop("lin_root")

    def reset_parameters(self):
        # GraphConv.__init__ dispatches here before the compatibility rename.
        if not hasattr(self, "lin_l") or not hasattr(self, "lin_r"):
            GraphConv.reset_parameters(self)
            return
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if hasattr(self, "forward_map"):
            self.forward_map.reset_parameters()
            self.inverse_map.reset_parameters()
            nn.init.zeros_(self.alpha)
            nn.init.zeros_(self.beta)

    @staticmethod
    def _as_pair(x):
        if isinstance(x, torch.Tensor):
            return x, x
        return x


class GenAggGraphConv(_LegacyNamedGraphConv):
    """GraphConv using the paper-era augmented generalised f-mean."""

    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__(in_channels, out_channels, bias=bias, **kwargs)
        self._use_legacy_linear_names()

        rng_state = torch.get_rng_state()
        try:
            self.forward_map = _ScalarMap((1, 2, 2, 4))
            self.inverse_map = _ScalarMap((4, 2, 2, 1))
            self.alpha = nn.Parameter(torch.zeros(()))
            self.beta = nn.Parameter(torch.zeros(()))
            self.forward_map.reset_parameters()
            self.inverse_map.reset_parameters()
        finally:
            torch.set_rng_state(rng_state)
        self._current_inverse_loss = None

    def forward(self, x, edge_index, edge_weight=None, size=None):
        x = self._as_pair(x)
        out = self.propagate(
            edge_index, x=x, edge_weight=edge_weight, size=size
        )
        out = self.lin_l(out)
        if x[1] is not None:
            out = out + self.lin_r(x[1])
        return out

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        mean = scatter_mean(inputs, index, dim=0, dim_size=dim_size)
        centered = inputs - self.beta * mean.index_select(0, index)
        encoded = self.forward_map(centered.unsqueeze(-1))
        encoded_mean = scatter_mean(encoded, index, dim=0, dim_size=dim_size)
        degree = scatter_add(
            inputs.new_ones((inputs.size(0), 1)),
            index,
            dim=0,
            dim_size=dim_size,
        )
        decoded = self.inverse_map(
            encoded_mean * degree.clamp_min(1).unsqueeze(-1).pow(self.alpha)
        ).squeeze(-1)
        self._current_inverse_loss = self._inverse_consistency_loss(
            centered, encoded
        )
        return decoded * degree.gt(0).to(decoded.dtype)

    def inverse_consistency_loss(self, inputs=None):
        """Return the source squared absolute-value inverse error explicitly."""
        if inputs is None:
            if self._current_inverse_loss is not None:
                return self._current_inverse_loss
            return self.alpha.sum() * 0.0
        if inputs.numel() == 0:
            return inputs.sum() * 0.0
        encoded = self.forward_map(inputs.unsqueeze(-1))
        return self._inverse_consistency_loss(inputs, encoded)

    def _inverse_consistency_loss(self, centered, encoded):
        reconstructed = self.inverse_map(encoded).squeeze(-1)
        return (reconstructed.abs() - centered.abs()).square().mean()


class SoftMedoidGraphConv(_LegacyNamedGraphConv):
    """GraphConv using scaled Soft Medoid over transformed messages."""

    def __init__(
        self, in_channels, out_channels, temperature=1.0, bias=True, **kwargs
    ):
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError("temperature must be finite and positive")
        super().__init__(in_channels, out_channels, bias=bias, **kwargs)
        self._use_legacy_linear_names()
        self.temperature = float(temperature)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        x_l, x_r = self._as_pair(x)
        source, target = edge_index[0], edge_index[1]
        messages = F.linear(x_l.index_select(0, source), self.lin_l.weight)
        if edge_weight is not None:
            messages = messages * edge_weight.view(-1, 1)

        if x_r is not None:
            target_count = x_r.size(0)
        elif size is not None and size[1] is not None:
            target_count = size[1]
        elif target.numel() > 0:
            target_count = int(target.max().item()) + 1
        else:
            target_count = 0
        neighbor = self._soft_medoid(messages, target, target_count)
        if self.lin_l.bias is not None:
            neighbor = neighbor + self.lin_l.bias
        if x_r is not None:
            neighbor = neighbor + self.lin_r(x_r)
        return neighbor

    def _soft_medoid(self, messages, target, target_count):
        output = messages.new_zeros((target_count, self.out_channels))
        if messages.size(0) == 0:
            return output

        degree = scatter_add(
            messages.new_ones((messages.size(0),)),
            target,
            dim=0,
            dim_size=target_count,
        ).long()
        max_degree = int(degree.max().item())
        permutation = torch.argsort(target)
        sorted_target = target.index_select(0, permutation)
        counts = degree[degree.gt(0)]
        starts = torch.cumsum(counts, dim=0) - counts
        slots = torch.arange(
            messages.size(0), device=messages.device, dtype=torch.long
        ) - torch.repeat_interleave(starts, counts)

        packed = messages.new_zeros(
            (target_count, max_degree, self.out_channels)
        )
        packed[sorted_target, slots] = messages.index_select(0, permutation)
        valid = (
            torch.arange(max_degree, device=messages.device).unsqueeze(0)
            < degree.unsqueeze(1)
        )
        pairwise = torch.norm(
            packed.unsqueeze(2) - packed.unsqueeze(1), p=2, dim=-1
        )
        distances = (pairwise * valid.unsqueeze(1).to(pairwise.dtype)).sum(-1)
        logits = -distances / self.temperature
        logits = logits.masked_fill(~valid, float("-inf"))
        logits = torch.where(
            degree.gt(0).unsqueeze(1), logits, torch.zeros_like(logits)
        )
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        weighted = (weights.unsqueeze(-1) * packed).sum(1)
        return degree.to(weighted.dtype).unsqueeze(1) * weighted


class SSMAGraphConv(_LegacyNamedGraphConv):
    """GraphConv with fixed-signal sequential signal-mixing aggregation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kappa=5,
        epsilon=1e-6,
        bias=True,
        **kwargs
    ):
        if int(kappa) < 1:
            raise ValueError("kappa must be positive")
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if not isinstance(in_channels, int):
            raise TypeError("SSMA requires an integer input dimension")

        super().__init__(in_channels, out_channels, bias=bias, **kwargs)
        self._use_legacy_linear_names()
        self.kappa = int(kappa)
        self.epsilon = float(epsilon)
        self.signal_height = self.kappa + 1
        self.signal_width = self.kappa * (in_channels - 1) + 1
        self.signal_size = self.signal_height * self.signal_width

        rng_state = torch.get_rng_state()
        try:
            self.compressor = nn.Linear(self.signal_size, in_channels)
        finally:
            torch.set_rng_state(rng_state)

    def reset_parameters(self):
        if not hasattr(self, "lin_l") or not hasattr(self, "lin_r"):
            GraphConv.reset_parameters(self)
            return
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if hasattr(self, "compressor"):
            self.compressor.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, size=None):
        x = self._as_pair(x)
        out = self.propagate(
            edge_index, x=x, edge_weight=edge_weight, size=size
        )
        out = self.lin_l(out)
        if x[1] is not None:
            out = out + self.lin_r(x[1])
        return out

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() else 0
        if inputs.size(0) == 0:
            return inputs.new_zeros((dim_size, inputs.size(-1)))
        degree = scatter_add(
            inputs.new_ones((inputs.size(0),)),
            index,
            dim=0,
            dim_size=dim_size,
        )
        if degree.numel() and int(degree.max().item()) > self.kappa:
            raise ValueError(
                "neighborhood degree exceeds kappa={}".format(self.kappa)
            )

        signal = inputs.new_zeros(
            (inputs.size(0), self.signal_height, self.signal_width)
        )
        signal[:, 0, : inputs.size(-1)] = -inputs
        signal[:, 1, 0] = 1.0
        spectrum = torch.fft.fft2(signal)
        log_magnitude = scatter_mean(
            torch.log(spectrum.abs() + self.epsilon),
            index,
            dim=0,
            dim_size=dim_size,
        )
        phase = scatter_add(
            torch.angle(spectrum),
            index,
            dim=0,
            dim_size=dim_size,
        )
        magnitude = torch.exp(log_magnitude)
        mixed = torch.complex(
            magnitude * torch.cos(phase),
            magnitude * torch.sin(phase),
        )
        decoded = torch.fft.ifft2(mixed).real.reshape(dim_size, -1)
        compressed = self.compressor(decoded)
        return compressed * degree.gt(0).to(compressed.dtype).unsqueeze(1)


def build_second_graph_conv(selector, in_channels, out_channels):
    """Construct a second graph convolution by experiment selector."""
    if selector == "add":
        return GraphConv(in_channels, out_channels)
    if selector == "genagg":
        return GenAggGraphConv(in_channels, out_channels)
    if selector == "soft_medoid":
        return SoftMedoidGraphConv(in_channels, out_channels)
    if selector == "ssma":
        return SSMAGraphConv(in_channels, out_channels)
    raise ValueError("unknown second graph aggregation: {}".format(selector))
