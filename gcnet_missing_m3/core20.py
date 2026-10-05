"""Explicit integration contracts for twenty distinct research mechanisms.

No universal residual template: memory cores, representation objectives,
inference cores, heads and optimizers have different attachment points.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

METHODS = tuple(f'C{i:02d}' for i in range(1, 21))
CONTROLS = ('C17-binary-control', 'C20-mean-control')
TRANSFER_METHODS = ('R02', 'R03', 'R12', 'R18')
MODALITIES = ('audio', 'text', 'visual')


def validate_config(config):
    method = config.core20_method
    if method == 'none':
        return
    if method not in METHODS + CONTROLS + TRANSFER_METHODS:
        raise ValueError('unknown core20_method')
    forbidden = ('paired_history_views', 'osram_relation_block', 'osram_relation_dual_readout',
                 'osram_decision_correction', 'osram_gap_increment_filter', 'osram_post_grn',
                 'osram_history_query_adapter', 'classification_completion', 'osram_local_skip_gate',
                 'osram_memory_only_adapter', 'osram_history_input_gate', 'osram_local_evidence_gate',
                 'osram_hierarchical_evidence_gate', 'completion_write_to_memory',
                 'local_context_residual', 'node_interaction_residual', 'text_core')
    if any(getattr(config, key) for key in forbidden):
        raise ValueError('C01–C20 are independent experiments, not combinations with prior modules')
    if (config.backbone_type != 'osram' or config.osram_bidirectional
            or config.osram_forward_slot_reuse or config.osram_readout_fusion != 'flat'
            or config.osram_meaningful_block != 'none' or config.osram_readout_candidate != 'none'
            or config.training_objective != 'emotion-only' or config.completion_path != 'none'
            or config.train_rate_mode != 'cyclic' or config.readout_type != 'shared'
            or config.osram_ablation != 'full' or config.osram_emotion_ablation != 'full'
            or not config.disable_unused_aux_modules):
        raise ValueError('core20 requires independent causal single-view full cfg84 task training')
    expected_mode = 'pattern-groupdro-author' if method == 'C18' else 'sample-mean'
    if config.emotion_loss_mode != expected_mode:
        raise ValueError(f'{method} requires emotion_loss_mode={expected_mode}')
    if config.core20_aux_weight < 0 or not torch.isfinite(torch.tensor(config.core20_aux_weight)):
        raise ValueError('core20 auxiliary weight must be finite/nonnegative')
    if method.startswith('C17') and config.mosi_task_mode != 'binary':
        raise ValueError('C17 MOSI needs matched binary classification, not unchanged regression')
    if method in ('C12', 'C19') and (config.dataset not in ('CMUMOSI', 'CMUMOSEI') or config.mosi_task_mode != 'regression'):
        raise ValueError(f'{method} first transfer supports MOSI/MOSEI regression only')


class Core20(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.method = config.core20_method
        self.aux_weight = config.core20_aux_weight
        self.auxiliary_loss = torch.tensor(0.)
        self.metadata = None
        self.multilevel_losses = None
        self.task_parameters = None
        self.prototype_distances = None
        self.gradient_diagnostics = {}
        dim, out = config.latent_dim, config.osram_output_dim
        if self.method == 'R18':
            from .r18_lupi import PrivilegedNoise
            self.privileged_noise = PrivilegedNoise(sum(model.observed_set.dimensions), out)
            self.complete_features = None
        if self.method == 'C03':
            from .core20_representation import VQVAEValues
            # Register on OSRAM where projected Values become available.
            model.osram.core20_value = VQVAEValues(config.osram_value_dim, num_heads=config.osram_num_heads,
                reconstruction_dims=dict(zip(MODALITIES, model.observed_set.dimensions)))
        elif self.method == 'C11':
            from .core20_probability import MMVAE
            self.representation = MMVAE(dim, reconstruction_dims=dict(zip(MODALITIES, model.observed_set.dimensions)))
        elif self.method == 'C12':
            from .core20_probability import NaturalPosteriorNetwork
            self.head = NaturalPosteriorNetwork(out)
            model.smax_fc.requires_grad_(False)
        elif self.method == 'C15':
            from .core20_representation import FactorCL
            self.representation = FactorCL(dim)
        elif self.method == 'C17':
            from .core20_inference import PrototypeClassifier
            self.head = PrototypeClassifier(out, num_classes=model.smax_fc.out_features)
            model.smax_fc.requires_grad_(False)
        elif self.method == 'C19':
            from .core20_representation import FishrPenalty
            self.fishr = FishrPenalty()
        common = dict(num_heads=config.osram_num_heads, key_dim=config.osram_key_dim,
                      value_dim=config.osram_value_dim, latent_dim=dim)
        if self.method in ('C01', 'C02', 'C04'):
            from .core20_storage import build
            model.osram.core20_memory = build(self.method, **common)
        elif self.method == 'R02':
            from .r02_delta_product import DeltaProductStorage
            model.osram.core20_memory = DeltaProductStorage(config, model.osram)
        elif self.method == 'R03':
            from .r03_mesa import MesaStorage
            model.osram.core20_memory = MesaStorage(config, model.osram)
        elif self.method in ('C05', 'C06', 'C07', 'C08'):
            from .core20_dynamics import build
            model.osram.core20_memory = build(self.method, **common)
        elif self.method in ('C09', 'C10'):
            from .core20_probability import build_memory
            model.osram.core20_memory = build_memory(self.method, **common)
        if hasattr(model.osram, 'core20_memory'):
            model.osram.alpha_logits.requires_grad_(False)
            model.osram.beta_logits.requires_grad_(False)
        if self.method in ('C13', 'C14', 'C16'):
            from .core20_inference import build_readout
            model.osram.core20_readout = build_readout(self.method, dim, model.osram.context_dim, out)
            # Replaced readout is explicit; original weights remain loadable but
            # are not counted as trainable/optimized capacity.
            for module in (model.osram.emotion_adapter, model.osram.local_skip, model.osram.emotion_norm):
                module.requires_grad_(False)
        model.osram.core20_collect_multilevel = self.method in ('C20', 'C20-mean-control')

    def prepare(self, model, encoded, latents, availability, umask, features):
        self.auxiliary_loss = encoded.new_zeros(())
        if self.method == 'C11':
            targets = dict(zip(MODALITIES, features.split(model.observed_set.dimensions, -1)))
            encoded, self.auxiliary_loss, self.metadata = self.representation(
                latents, availability, umask, reconstruction_targets=targets)
        elif self.method == 'C15':
            latents, self.metadata = self.representation(latents, availability, umask)
            encoder = model.observed_set
            if encoder.fusion_type != 'mean':
                raise ValueError('C15 first transfer requires cfg84 mean observed-set fusion')
            valid = umask.T.bool()
            slots = [torch.where((valid & availability[..., i].bool())[..., None],
                                 latents[m] + encoder.modality_embedding.weight[i], 0.)
                     for i, m in enumerate(MODALITIES)]
            count = availability.sum(-1, keepdim=True).clamp_min(1)
            pattern = (availability.long() * availability.new_tensor([4, 2, 1]).long()).sum(-1)
            fused = sum(slots) / count + encoder.pattern_embedding(pattern)
            encoded = torch.zeros_like(encoded)
            encoded[valid] = encoder.fusion(fused[valid])
        return encoded, latents

    def predict(self, hidden, original_logits, umask, model=None):
        if self.method == 'R12':
            self.ranking_hidden = hidden
        if self.method == 'R18':
            noisy, self.auxiliary_loss = self.privileged_noise(
                hidden, self.complete_features, umask.T.bool())
            self.complete_features = None
            return model.smax_fc(noisy) if self.training else original_logits
        if self.method == 'C12':
            logits, self.task_parameters = self.head(hidden, umask)
            return logits
        if self.method == 'C17':
            valid = umask.T.bool()
            logits = hidden.new_zeros(*hidden.shape[:2], self.head.last_layer.out_features)
            selected, self.prototype_distances = self.head(hidden[valid])
            logits[valid] = selected
            return logits
        return original_logits

    def loss(self, model, config, view, logits, full_loss, task_loss):
        valid = view['umask'].T.bool()
        if self.method == 'C12':
            full_loss = self.head.loss(view['labels'], view['umask'], self.task_parameters)
        elif self.method == 'C17':
            labels = view['labels'].T[valid]
            if config.dataset in ('CMUMOSI', 'CMUMOSEI'):
                # Match original classification criterion: zero targets excluded.
                selected = labels.ne(0)
                full_loss = (self.head.loss(logits[valid][selected], self.prototype_distances[selected],
                                           labels[selected].gt(0).long()) if bool(selected.any()) else logits.sum() * 0)
            else:
                full_loss = self.head.loss(logits[valid], self.prototype_distances, labels.long())
        aux = self.auxiliary_loss
        if self.method == 'R12':
            from .r12_rnc import rnc_loss
            features = self.ranking_hidden[valid]
            labels = view['labels'].T[valid]
            if features.shape[0] > 128:
                indices = torch.randperm(features.shape[0], device=features.device)[:128]
                features, labels = features[indices], labels[indices]
            aux = aux + rnc_loss(features, labels, temperature=2.0)
        for name in ('core20_memory', 'core20_value', 'core20_readout'):
            module = getattr(model.osram, name, None)
            if module is not None:
                aux = aux + getattr(module, 'auxiliary_loss', logits.new_zeros(()))
                aux = aux + getattr(module, 'ponder_cost', logits.new_zeros(()))
        if self.method == 'C15':
            complete_availability = view['umask'].T[..., None].expand(-1, -1, 3).to(view['availability'])
            _, complete = model.observed_set(view['complete'], complete_availability, view['umask'])
            aux = aux + self.representation.training_loss(self.metadata, complete, view['labels'],
                                                        view['umask'], include_critic_fit=False)
        if self.method == 'C19':
            if config.mosi_task_mode != 'regression' or config.dataset not in ('CMUMOSI', 'CMUMOSEI'):
                raise ValueError('C19 first transfer currently supports MOSI/MOSEI original regression')
            targets = view['labels'].T[valid]
            losses = (logits[..., 0][valid] - targets).square()
            if config.task_regression_loss == 'smooth-l1':
                losses = F.smooth_l1_loss(logits[..., 0][valid], targets, reduction='none',
                                          beta=config.task_smooth_l1_beta)
            groups = self.fishr.availability_groups(view['availability'])[valid]
            aux = aux + self.fishr.penalty(losses, model.smax_fc.parameters(), groups)
        if self.method in ('C20', 'C20-mean-control'):
            exits = model.osram.core20_multilevel_hidden
            self.multilevel_losses = [task_loss(model.smax_fc(exits[name])) for name in ('local', 'base')]
            self.multilevel_losses.append(full_loss)
            full_loss = sum(self.multilevel_losses) / 3
        self.last_auxiliary_loss = aux.detach()
        return full_loss + self.aux_weight * aux


def attach(model, config):
    if config.core20_method == 'none':
        return
    validate_config(config)
    with torch.random.fork_rng(devices=[]):
        model.core20 = Core20(config, model)
    model.core20.to(next(model.parameters()).device)
    # Modules attached to OSRAM during Core20 init must follow model device too.
    model.to(next(model.parameters()).device)
    model.core20.initial_trainable_names = frozenset(name for name, p in model.named_parameters() if p.requires_grad)


def cagrad_backward(losses, model):
    from .core20_cagrad import conflict_averse_direction
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradients = [torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True) for loss in losses]
    shared = [i for i in range(len(parameters)) if all(g[i] is not None for g in gradients)]
    if not shared:
        raise ValueError('CAGrad tasks have no common trainable parameters')
    matrix = torch.stack([torch.cat([g[i].reshape(-1) for i in shared]) for g in gradients], -1)
    direction, model.core20.gradient_diagnostics = conflict_averse_direction(matrix)
    cursor = 0
    for i, parameter in enumerate(parameters):
        if i in shared:
            parameter.grad = direction[cursor:cursor + parameter.numel()].reshape_as(parameter).detach()
            cursor += parameter.numel()
        else:
            terms = [g[i] for g in gradients if g[i] is not None]
            parameter.grad = sum(terms).detach() / len(losses) if terms else None


def prepare_optimizers(model, config, groups):
    if config.core20_method != 'C15':
        return groups
    critic = list(model.core20.representation.upper_critic_parameters())
    excluded = {id(p) for p in critic}
    groups = [dict(group, params=[p for p in group['params'] if id(p) not in excluded]) for group in groups]
    groups = [group for group in groups if group['params']]
    model.core20_critic_optimizer = torch.optim.Adam(critic, lr=config.learning_rate,
                                                    weight_decay=config.weight_decay)
    return groups


def post_optimizer_step(model, view):
    if not hasattr(model, 'core20_critic_optimizer'):
        return None
    # Author FactorCL order: representation optimizer first, then recompute
    # complete TRAIN features and fit CLUB critics with encoder detached.
    with torch.no_grad():
        av = view['umask'].T[..., None].expand(-1, -1, 3).to(view['availability'])
        _, latents = model.observed_set(view['complete'], av, view['umask'])
    optimizer = model.core20_critic_optimizer
    optimizer.zero_grad(set_to_none=True)
    fitting = model.core20.representation.critic_learning_loss(latents, view['labels'], view['umask'])
    if not bool(torch.isfinite(fitting)):
        raise ValueError('nonfinite FactorCL critic fitting loss')
    fitting.backward()
    optimizer.step()
    return float(fitting.detach())


def auxiliary_state(model, group_weights):
    result = {'group_dro_weights': group_weights} if group_weights is not None else {}
    if hasattr(model, 'core20_critic_optimizer'):
        result['critic_optimizer'] = model.core20_critic_optimizer.state_dict()
    if hasattr(model, 'core20') and model.core20.method == 'C17':
        result['prototype_projection'] = model.core20.head.projection_records
    return result or None


def restore_auxiliary(model, state):
    if hasattr(model, 'core20_critic_optimizer'):
        if 'critic_optimizer' not in state:
            raise ValueError('FactorCL continuation lacks critic optimizer state')
        model.core20_critic_optimizer.load_state_dict(state['critic_optimizer'])
    if hasattr(model, 'core20') and model.core20.method == 'C17':
        model.core20.head.projection_records = state.get('prototype_projection', [])


def prototype_phase(model, epoch):
    if not hasattr(model, 'core20') or model.core20.method != 'C17':
        return None
    # Fixed first transfer schedule: 10 warm epochs, joint until each 10th
    # epoch push, two following head-only epochs, then joint. No test tuning.
    phase = 'warm' if epoch < 10 else 'head' if epoch % 10 < 2 else 'joint'
    for name, p in model.named_parameters():
        if name not in model.core20.initial_trainable_names:
            p.requires_grad_(False)
        elif phase == 'head':
            p.requires_grad_(name.startswith('core20.head.last_layer.'))
        elif name.startswith('core20.head.last_layer.'):
            p.requires_grad_(False)
        elif name.startswith('core20.head.'):
            p.requires_grad_(True)
        else:
            p.requires_grad_(phase == 'joint')
    model.core20.phase = phase
    return phase
