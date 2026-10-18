"""Opt-in sidecar for the unmodified, archived cfg84 Flat implementation.

Hooks expose actual graph nodes, not detached copies. They do not rerun CHM or
replace its inputs. Reinstall these hooks before strict checkpoint loading.
"""
import torch


def attach_recent_history(model, *, enabled=True, module_class=None, window=4,
                          attention_dim=128, num_heads=4):
    if not enabled:
        return None
    osram = model.osram
    if hasattr(osram, 'recent_history'):
        raise ValueError('Recent history is already attached')
    if (osram.osram_readout_fusion != 'flat' or osram.bidirectional
            or osram.osram_meaningful_block != 'none'
            or osram.osram_emotion_ablation != 'full'):
        raise ValueError('Recent history requires the original causal full Flat')
    if module_class is None:
        from gcnet_missing_m3.recent_history import RecentHistoryAttentionResidual
        module_class = RecentHistoryAttentionResidual
    # Construct after the complete original model, without advancing its RNG.
    with torch.random.fork_rng(devices=[]):
        branch = module_class(local_dim=osram.latent_dim, latent_dim=osram.latent_dim,
            context_dim=osram.context_dim // 2, output_dim=osram.output_dim,
            window=window, attention_dim=attention_dim, num_heads=num_heads, dropout=0.)
    parameter = next(osram.parameters())
    osram.recent_history = branch.to(device=parameter.device, dtype=parameter.dtype)
    osram.recent_history_enabled = True
    osram._recent_history_inputs = None

    def before_scan(module, args):
        # OSRAM positional API: node, actual observed latents, availability,
        # speaker mask, utterance mask, lengths. No source tensor is modified.
        module._recent_history_inputs = dict(latents=args[1], availability=args[2], umask=args[4].T)

    def before_adapter(module, args):
        inputs = osram._recent_history_inputs
        if inputs is None:
            raise RuntimeError('Adapter called without the current OSRAM trajectory')
        inputs['local'] = args[0][..., :osram.latent_dim]
        inputs['base'] = args[0][..., osram.latent_dim:osram.latent_dim + osram.context_dim]

    def before_norm(module, args):
        if not osram.recent_history_enabled:
            return None
        inputs = osram._recent_history_inputs
        if inputs is None or 'base' not in inputs:
            raise RuntimeError('Flat anchor was not captured before normalization')
        residual = osram.recent_history(**inputs)
        return (args[0] + residual,)

    def after_scan(module, args, output):
        # Never retain the previous conversation/batch autograd graph.
        module._recent_history_inputs = None

    osram._recent_history_handles = [
        osram.register_forward_pre_hook(before_scan),
        osram.emotion_adapter.register_forward_pre_hook(before_adapter),
        osram.emotion_norm.register_forward_pre_hook(before_norm),
        osram.register_forward_hook(after_scan),
    ]
    return branch


def set_enabled(model, enabled):
    """Frozen paired bypass; never changes original inputs or CHM parameters."""
    model.osram.recent_history_enabled = bool(enabled)


def install(trainer, module_class, **settings):
    """Attach before optimizer construction, while reusing the original trainer.

    In the historical protocol, every osram.* parameter joins the backbone
    optimizer group; the extra branch is therefore trained jointly from step1.
    """
    original = trainer.MissingM3GraphModel
    def construct(*args, **kwargs):
        model = original(*args, **kwargs)
        attach_recent_history(model, module_class=module_class, **settings)
        return model
    trainer.MissingM3GraphModel = construct
    return original
