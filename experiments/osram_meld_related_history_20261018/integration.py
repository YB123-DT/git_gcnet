"""Opt-in B1/B2 pre-LN residuals for the unchanged archived Flat model.

The historical trainer/model remain unedited. Hooks reuse actual graph nodes
after readout ablation, never rerun CHM, and keep the original Local Skip input.
Reinstall the attachment before strictly restoring a new-branch checkpoint.
"""
import weakref

import torch


def attach_related_history(model, *, kind='B1', enabled=True, module_class=None,
                           window=8, attention_dim=128):
    if not enabled:
        return None
    if kind not in {'B1', 'B2'}:
        raise ValueError('kind must be B1 or B2')
    osram = model.osram
    if hasattr(osram, 'related_history'):
        raise ValueError('Related history is already attached')
    if (osram.osram_readout_fusion != 'flat' or osram.bidirectional
            or osram.osram_meaningful_block != 'none'
            or getattr(osram, 'osram_flat_exclude_local', False)):
        raise ValueError('Related history requires causal original full-width Flat')
    if module_class is None:
        from gcnet_missing_m3.related_history import RelatedHistoryResidual
        module_class = RelatedHistoryResidual
    # Fork only initialization: branch forwards are deterministic, preserving
    # the original train/evaluation dropout and missing-sampler RNG streams.
    with torch.random.fork_rng(devices=[]):
        branch = module_class(kind=kind, local_dim=osram.latent_dim,
                              context_dim=osram.context_dim // 2,
                              output_dim=osram.output_dim, window=window,
                              attention_dim=attention_dim)
    parameter = next(osram.parameters())
    osram.related_history = branch.to(device=parameter.device, dtype=parameter.dtype)
    osram.related_history_enabled = True
    osram._related_history_inputs = None
    osram._related_history_aux_pending = False
    osram.related_history_last_aux = None

    def before_scan(module, args):
        branch.clear_cache()
        module._related_history_aux_pending = False
        module._related_history_inputs = {
            'availability': args[2], 'speakers': args[3].transpose(0, 1),
            'umask': args[4].transpose(0, 1)}

    def before_adapter(module, args):
        inputs = osram._related_history_inputs
        if inputs is None:
            raise RuntimeError('Flat adapter called outside the current CHM forward')
        flat = args[0]
        local_dim, full_context = osram.latent_dim, osram.context_dim
        expected = local_dim + 4 * full_context
        if flat.shape[-1] != expected:
            raise RuntimeError(f'Expected unchanged Flat width {expected}, got {flat.shape[-1]}')
        valid = inputs['umask'].bool()
        active = valid.unsqueeze(-1) & ~inputs['availability'].bool()
        inputs['local'] = torch.where(valid.unsqueeze(-1), flat[..., :local_dim], 0.)
        # Use masked classification inputs, never the unablated CHM outputs.
        base = flat[..., local_dim:local_dim + full_context]
        gap = flat[..., local_dim + full_context:].reshape(*flat.shape[:2], 3, full_context)
        inputs['base'] = torch.where(valid.unsqueeze(-1), base[..., :full_context // 2], 0.)
        inputs['gap'] = torch.where(active.unsqueeze(-1), gap[..., :full_context // 2], 0.)

    def before_norm(module, args):
        if not osram.related_history_enabled:
            return None
        inputs = osram._related_history_inputs
        if inputs is None or 'base' not in inputs:
            raise RuntimeError('Original Flat inputs were not captured')
        residual = branch(**inputs)
        osram._related_history_aux_pending = (
            kind == 'B2' and model.training and torch.is_grad_enabled())
        return (args[0] + residual,)

    def after_scan(module, args, output):
        module._related_history_inputs = None
        if not module._related_history_aux_pending:
            branch.clear_cache()

    osram._related_history_handles = [
        osram.register_forward_pre_hook(before_scan),
        osram.emotion_adapter.register_forward_pre_hook(before_adapter),
        osram.emotion_norm.register_forward_pre_hook(before_norm),
        osram.register_forward_hook(after_scan),
    ]
    return branch


def set_enabled(model, enabled):
    """Bypass only the second block for matched frozen readout evaluation."""
    model.osram.related_history_enabled = bool(enabled)
    if not enabled:
        model.osram._related_history_aux_pending = False
        model.osram.related_history.clear_cache()


def install_task_loss(trainer, model_provider, *, relation_weight=.05):
    """Add B2 supervision once per training forward, never to evaluation CE.

    Historical labels are [batch,time]; the branch explicitly consumes
    [time,batch]. Its auxiliary forward detaches backbone features internally.
    The task path remains the original, fully differentiable model forward.
    """
    if not 0 <= relation_weight < float('inf'):
        raise ValueError('relation_weight must be finite and nonnegative')
    original = trainer._task_loss

    def task_loss(dataset, logits, labels, umask, *args, **kwargs):
        task = original(dataset, logits, labels, umask, *args, **kwargs)
        model = model_provider()
        if model is None:
            return task
        osram = model.osram
        if not (model.training and torch.is_grad_enabled()
                and getattr(osram, 'related_history_enabled', False)
                and getattr(osram, '_related_history_aux_pending', False)):
            return task
        # Consume before computing to protect against repeated loss calls.
        osram._related_history_aux_pending = False
        try:
            auxiliary = osram.related_history.auxiliary_relation_loss(
                labels.transpose(0, 1), dataset=dataset)
            osram.related_history_last_aux = float(auxiliary.detach())
            return task + relation_weight * auxiliary
        finally:
            osram.related_history.clear_cache()

    trainer._task_loss = task_loss
    return original


def install(trainer, *, kind, module_class=None, relation_weight=.05, **settings):
    """Attach after original initialization and before optimizer construction."""
    original = trainer.MissingM3GraphModel
    current = [None]

    def construct(*args, **kwargs):
        model = original(*args, **kwargs)
        attach_related_history(model, kind=kind, module_class=module_class, **settings)
        current[0] = weakref.ref(model)
        return model

    trainer.MissingM3GraphModel = construct
    if kind == 'B2':
        install_task_loss(trainer, lambda: None if current[0] is None else current[0](),
                          relation_weight=relation_weight)
    return original
