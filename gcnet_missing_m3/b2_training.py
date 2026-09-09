"""B2 fixed-space pretraining and explicit, provenance-checked transfer."""
import hashlib
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .b2 import SourceOnlyM3Predictor
from .loss import missing_m3_loss
from .model import MODALITIES, ModalityProjector, EMATeacherProjectors, _validate_observed_inputs

AGGREGATION = "mean-real-observed-sources-per-missing-target-v1"
FORMAT = "source-only-b2-pretrain-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda: handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def subset(state,prefix):
    return {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}


def state_sha256(state):
    digest = hashlib.sha256()
    for k,v in sorted(state.items()):
        value=v.detach().cpu().contiguous()
        digest.update(k.encode())
        digest.update(str((value.dtype,tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_base(config):
    if (config.backbone_type != "osram" or config.osram_bidirectional
            or config.osram_forward_slot_reuse or config.osram_write_step != .6
            or config.classification_completion or config.completion_path != "none"
            or config.train_rate_mode != "cyclic" or config.fusion_type != "mean"
            or config.osram_ablation != "full" or config.node_interaction_residual
            or config.target_private_rank != 0):
        raise ValueError("B2 initialization requires the unchanged causal .6 mean/cyclic baseline")


class SourceOnlyPretrainer(nn.Module):
    """Owns only frozen modality projectors/targets and a new source-only MMoE."""

    def __init__(self,config,dimensions):
        super().__init__()
        self.config=config
        self.dimensions=tuple(dimensions)
        self.projectors=nn.ModuleDict({m:ModalityProjector(w,config.latent_dim,config.projector_dropout)
                                      for m,w in zip(MODALITIES,dimensions)})
        self.teacher=EMATeacherProjectors(self.projectors)
        self.projectors.requires_grad_(False)
        self.predictor=SourceOnlyM3Predictor(config.latent_dim,config.num_experts,config.top_k,
            config.predictor_dropout,config.mmoe_variant,config.target_private_rank)
        self.source_path=None
        self.source_hash=None
        self.projectors.eval()

    @classmethod
    def from_checkpoint(cls,path):
        from .train_gcnet import TrainConfig, set_random_seed
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        config=TrainConfig(**checkpoint["config"])
        validate_base(config)
        set_random_seed(config.seed)
        state=checkpoint["model"]
        dimensions=[state[f"observed_set.projectors.{m}.fc1.weight"].shape[1] for m in MODALITIES]
        result=cls(config,dimensions)
        result.projectors.load_state_dict(subset(state,"observed_set.projectors."),strict=True)
        result.teacher.load_state_dict(subset(state,"teacher."),strict=True)
        result.source_path=str(Path(path).resolve())
        result.source_hash=file_sha256(path)
        result.frozen_hash=state_sha256({**{"projectors."+k:v for k,v in result.projectors.state_dict().items()},
                                       **{"teacher."+k:v for k,v in result.teacher.state_dict().items()}})
        return result

    def train(self,mode=True):
        super().train(mode)
        self.projectors.eval()
        self.teacher.eval()
        return self

    @torch.no_grad()
    def source_latents(self,features,availability,umask):
        valid=_validate_observed_inputs(features,availability,umask,self.dimensions)
        result={}
        for i,(m,part) in enumerate(zip(MODALITIES,torch.split(features,self.dimensions,dim=-1))):
            selected=valid & availability[...,i].bool()
            z=features.new_zeros(*features.shape[:2],self.config.latent_dim)
            if bool(selected.any()):
                z[selected]=self.projectors[m](part[selected])
            result[m]=z
        return result

    def forward(self,incomplete,availability,umask):
        return self.predictor(self.source_latents(incomplete,availability,umask),availability,umask)

    @torch.no_grad()
    def targets(self,complete):
        return {m:self.teacher[m](part) for m,part in
                zip(MODALITIES,torch.split(complete,self.dimensions,dim=-1))}

    def save(self,path,epoch,optimizer=None):
        frozen={**{"projectors."+k:v for k,v in self.projectors.state_dict().items()},
                **{"teacher."+k:v for k,v in self.teacher.state_dict().items()}}
        assert state_sha256(frozen)==self.frozen_hash, "Stage1 frozen space changed"
        torch.save(dict(format=FORMAT,predictor={k:v.detach().cpu() for k,v in self.predictor.state_dict().items()},
            projectors={k:v.detach().cpu() for k,v in self.projectors.state_dict().items()},
            teacher={k:v.detach().cpu() for k,v in self.teacher.state_dict().items()},
            source_checkpoint=self.source_path,source_checkpoint_sha256=self.source_hash,
            frozen_space_sha256=self.frozen_hash,config=asdict(self.config),latent_dim=self.config.latent_dim,
            dimensions=self.dimensions,aggregation=AGGREGATION,feedback_branch="reg_predictions",
            selection_protocol="fixed-final",epoch=epoch,
            optimizer=optimizer.state_dict() if optimizer is not None else None,
            torch_rng_state=torch.get_rng_state()),path)


def pretrain_step(stage,view,optimizer,config):
    stage.train()
    optimizer.zero_grad(set_to_none=True)
    pred=stage(view["incomplete"],view["availability"],view["umask"])
    loss=missing_m3_loss(pred,stage.targets(view["complete"]),temperature=config.temperature,
        regression_aggregation=config.jepa_regression_aggregation,
        contrastive_prediction_source=config.jepa_contrastive_source)
    if not torch.isfinite(loss.total):
        raise ValueError("nonfinite Stage1 completion loss")
    grad=0.
    if loss.target_count:
        loss.total.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(stage.predictor.parameters(),
            config.gradient_clip_norm if config.gradient_clip_norm>0 else float("inf")))
        if not torch.isfinite(torch.tensor(grad)):
            raise ValueError("nonfinite Stage1 gradient")
        optimizer.step()
    return dict(loss=float(loss.total.detach()),regression=float(loss.regression.detach()),
                contrastive=float(loss.contrastive.detach()),target_count=loss.target_count,
                skipped=loss.target_count==0,gradient_norm=grad)


def load_b2_initialization(model,base_path,pretrain_path):
    """Do not silently drop shared keys or accept another projector coordinate space."""
    if model.completion_path != "pre_osram_b2":
        raise ValueError("B2 transfer requires pre_osram_b2 model")
    from .train_gcnet import TrainConfig
    base=torch.load(base_path,map_location="cpu",weights_only=False)
    source=TrainConfig(**base["config"])
    validate_base(source)
    pre=torch.load(pretrain_path,map_location="cpu",weights_only=False)
    if asdict(TrainConfig(**pre.get("config",{}))) != asdict(source):
        raise ValueError("Stage1/base config mismatch")
    aliases={"D_e":"hidden","time_attn":"time_attention"}
    for key,value in model.b2_model_settings.items():
        source_key=aliases.get(key,key)
        if key not in {"completion_path","no_cuda"} and hasattr(source,source_key) and value != getattr(source,source_key):
            raise ValueError(f"B2 model/source config mismatch: {key}")
    if pre.get("format") != FORMAT or pre.get("aggregation") != AGGREGATION or pre.get("feedback_branch") != "reg_predictions":
        raise ValueError("invalid source-only checkpoint format/semantics")
    if pre.get("source_checkpoint_sha256") != file_sha256(base_path):
        raise ValueError("Stage1/base checkpoint provenance mismatch")
    if model.latent_dim != pre["latent_dim"] or tuple(model.dimensions) != tuple(pre["dimensions"]):
        raise ValueError("Stage1/model dimensions mismatch")
    for prefix,name in (("observed_set.projectors.","projectors"),("teacher.","teacher")):
        if state_sha256(subset(base["model"],prefix)) != state_sha256(pre[name]):
            raise ValueError("Stage1 frozen projector/teacher provenance mismatch")
    expected={k for k in model.state_dict() if k.startswith(("source_only_predictor.","completed_read_fusion."))}
    missing,unexpected=model.load_state_dict(base["model"],strict=False)
    if set(missing)!=expected or unexpected:
        raise ValueError(f"Unexpected backbone transfer keys: missing={missing}, unexpected={unexpected}")
    model.source_only_predictor.load_state_dict(pre["predictor"],strict=True)
    model.ema_step=int(base.get("ema_step",0))
    return dict(source_checkpoint=str(base_path),source_checkpoint_sha256=pre["source_checkpoint_sha256"],
                pretrain_checkpoint=str(pretrain_path),pretrain_checkpoint_sha256=file_sha256(pretrain_path),
                frozen_space_sha256=pre["frozen_space_sha256"],pretrain_epoch=pre["epoch"],
                aggregation=AGGREGATION,source_selection_protocol=base.get("selection_protocol"))


def validate_stage2_config(config):
    """Allow only stage metadata, device and budget overrides, not scientific changes."""
    from .train_gcnet import TrainConfig
    cp=torch.load(config.b2_base_checkpoint,map_location="cpu",weights_only=False)
    source=TrainConfig(**cp["config"])
    validate_base(source)
    allowed={"completion_path","b2_base_checkpoint","b2_pretrain_checkpoint","device","epochs",
             "initial_backbone_checkpoint","pretrained_learning_rate","training_objective"}
    changed=[key for key,value in asdict(config).items() if key not in allowed and value!=getattr(source,key)]
    if changed:
        raise ValueError(f"B2 Stage2 config changed from source: {changed}")
