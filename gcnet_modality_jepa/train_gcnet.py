import os
import time
import glob
import math
import pickle
import random
import argparse
import json
import hashlib
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from numpy.random import randint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder

import sys
sys.path.append('../')
import config

from .model import ModalityJEPAGraphModel
from .dataloader_iemocap import load_iemocap_dataset
from .dataloader_cmumosi import load_cmumosi_dataset
from .loss import (
    AllModalReconLoss,
    MaskedCELoss,
    MaskedMSELoss,
    MaskedReconLoss,
    masked_centered_cosine_loss,
)
from .metrics import (
    compute_epoch_collapse_diagnostics,
    compute_modality_diagnostics,
)
from .parity import miss0_jepa_loss
from .mask_schedule import ConversationMaskSchedule
from .protocol import EpochSeededSubsetSampler, SeedBundle
from .run_manifest import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    collect_environment,
    collect_provenance,
    feature_metadata_hash,
    sampler_signature,
    write_manifest_atomic,
)
from .shared_state import load_shared_checkpoint, shared_state_hash
from .splits import build_iemocap_loso_split, build_official_split
from .targets import ModalityMeans, compute_modality_means
from gcnet_jepa_replacement.model import ReplacementJEPAGraphModel


def set_random_seed(seed, strict_deterministic=False):
    if strict_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict_deterministic:
        torch.use_deterministic_algorithms(True)


def reset_training_stochasticity(
    master_seed,
    fold,
    strict_deterministic=False,
):
    """Reset global training RNGs after variant-specific setup has completed."""
    training_seed = SeedBundle(master_seed).derive(
        "training_stochasticity:fold:{}".format(fold)
    )
    set_random_seed(
        training_seed,
        strict_deterministic=strict_deterministic,
    )
    return training_seed


def _build_protocol_loader(
    dataset,
    indices,
    dataset_name,
    fold,
    split,
    batch_size,
    num_workers,
    seed_bundle,
    split_hash,
    evaluation_protocol="official",
):
    sampler_seed = seed_bundle.derive(
        "data_order:{}:fold:{}:{}".format(dataset_name, fold, split)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=EpochSeededSubsetSampler(indices, seed=sampler_seed),
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )
    loader.protocol_metadata = {
        "split": split,
        "fold": int(fold),
        "indices": [int(index) for index in indices],
        "split_hash": split_hash,
        "order_seed": sampler_seed,
        "order_signature": sampler_signature(indices, sampler_seed),
        "evaluation_protocol": evaluation_protocol,
    }
    return loader


def _build_iemocap_official_fold(vids, test_session):
    """Return the original GCNet LOSO topology: held-out data is val and test."""
    prefix = "Ses0{}".format(test_session)
    test_indices = tuple(
        index for index, vid in enumerate(vids) if str(vid).startswith(prefix)
    )
    train_indices = tuple(
        index for index, vid in enumerate(vids) if index not in set(test_indices)
    )
    if not train_indices or not test_indices:
        raise ValueError(
            "IEMOCAP official fold {} must have nonempty train and test data".format(
                test_session
            )
        )
    payload = {
        "evaluation_protocol": "official",
        "test": list(test_indices),
        "train": list(train_indices),
        "validation": list(test_indices),
    }
    split_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return train_indices, test_indices, split_hash


def get_loaders(
    audio_root,
    text_root,
    video_root,
    num_folder,
    dataset,
    batch_size,
    num_workers,
    seed,
    validation_fraction=0.1,
    evaluation_protocol="official",
):
    if evaluation_protocol not in ("official", "strict"):
        raise ValueError("evaluation_protocol must be official or strict")
    dataset_name = dataset
    seed_bundle = SeedBundle(master_seed=seed)

    ###########################################################################
    ###########################################################################
    if dataset_name in ['CMUMOSI', 'CMUMOSEI']:

        dataset = load_cmumosi_dataset(
            label_path=config.PATH_TO_LABEL[dataset_name],
            audio_root=audio_root,
            text_root=text_root,
            video_root=video_root,
            dataset_name=dataset_name,
        )
        split_indices = build_official_split(
            dataset.vids,
            train_vids=dataset.trainVids,
            validation_vids=dataset.valVids,
            test_vids=dataset.testVids,
        )
        train_loader = _build_protocol_loader(
            dataset, split_indices.train, dataset_name, 1, "train",
            batch_size, num_workers, seed_bundle, split_indices.split_hash,
            evaluation_protocol,
        )
        val_loader = _build_protocol_loader(
            dataset, split_indices.validation, dataset_name, 1, "validation",
            batch_size, num_workers, seed_bundle, split_indices.split_hash,
            evaluation_protocol,
        )
        test_loader = _build_protocol_loader(
            dataset, split_indices.test, dataset_name, 1, "test",
            batch_size, num_workers, seed_bundle, split_indices.split_hash,
            evaluation_protocol,
        )

        train_loaders = [train_loader]
        val_loaders = [val_loader]
        test_loaders = [test_loader]

        ## return loaders
        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, val_loaders, test_loaders, adim, tdim, vdim


    ###########################################################################
    ###########################################################################
    if dataset_name in ['IEMOCAPFour', 'IEMOCAPSix']:

        dataset = load_iemocap_dataset(
            label_path=config.PATH_TO_LABEL[dataset_name],
            audio_root=audio_root,
            text_root=text_root,
            video_root=video_root,
        )
        if num_folder != 5:
            raise ValueError("IEMOCAP requires exactly five LOSO folds")
        labels_by_vid = getattr(dataset, "videoLabelsNew", None)
        if labels_by_vid is None:
            labels_by_vid = getattr(dataset, "videoLabels", None)
        if labels_by_vid is None:
            raise ValueError("IEMOCAP dataset does not expose conversation labels")

        train_loaders = []
        val_loaders = []
        test_loaders = []
        split_seed = seed_bundle.derive("split")
        for fold in range(1, num_folder + 1):
            if evaluation_protocol == "official":
                train_indices, test_indices, split_hash = (
                    _build_iemocap_official_fold(dataset.vids, fold)
                )
                validation_indices = test_indices
            else:
                split_indices = build_iemocap_loso_split(
                    dataset.vids,
                    labels_by_vid,
                    test_session=fold,
                    validation_fraction=validation_fraction,
                    seed=split_seed,
                )
                train_indices = split_indices.train
                validation_indices = split_indices.validation
                test_indices = split_indices.test
                split_hash = split_indices.split_hash
            train_loaders.append(_build_protocol_loader(
                dataset, train_indices, dataset_name, fold, "train",
                batch_size, num_workers, seed_bundle, split_hash,
                evaluation_protocol,
            ))
            val_loaders.append(_build_protocol_loader(
                dataset, validation_indices, dataset_name, fold,
                "validation", batch_size, num_workers, seed_bundle,
                split_hash, evaluation_protocol,
            ))
            test_loaders.append(_build_protocol_loader(
                dataset, test_indices, dataset_name, fold, "test",
                batch_size, num_workers, seed_bundle, split_hash,
                evaluation_protocol,
            ))

        ## return loaders
        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, val_loaders, test_loaders, adim, tdim, vdim

    raise ValueError("unsupported dataset: {}".format(dataset_name))


def build_model(args, adim, tdim, vdim):
    D_e = args.hidden
    graph_h = args.hidden // 2
    model_class = (
        ReplacementJEPAGraphModel
        if getattr(args, "model_variant", "addon") == "replacement"
        else ModalityJEPAGraphModel
    )
    model = model_class(args.base_model,
                       adim, tdim, vdim, D_e, graph_h,
                       n_speakers=args.n_speakers,
                       window_past=args.windowp,
                       window_future=args.windowf,
                       n_classes=args.n_classes,
                       dropout=args.dropout,
                       time_attn=args.time_attn,
                       no_cuda=args.no_cuda,
                       predictor_dropout=args.predictor_dropout,
                       enable_stability_reconstruction=(
                           getattr(args, "stability_recon_weight", 0.0) > 0.0
                       ))
    print("Model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print ('Graph NN with', args.base_model, 'as base model.')
    return model


## gain input features: ?*[seqlen, batch, dim]
def generate_inputs(audio_host, text_host, visual_host, audio_guest, text_guest, visual_guest, qmask):
    input_features = [] 
    feat1 = torch.cat([audio_host, text_host, visual_host], dim=2) # [seqlen, batch, featdim=adim+tdim+vdim]
    feat2 = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    featdim = feat1.size(-1)
    tmask = qmask.transpose(0, 1) # [batch, seqlen] -> [seqlen, batch]
    tmask = tmask.unsqueeze(2).repeat(1,1,featdim) # -> [seqlen, batch, featdim]
    select_feat = torch.where(tmask==0, feat1, feat2) # -> [seqlen, batch, featdim]
    input_features.append(select_feat) # 1 * [seqlen, batch, dim]
    return input_features


def build_mask_schedule(args, split, fold, mask_rate):
    """Build the primary missing-modality schedule for one fold and split."""
    return ConversationMaskSchedule(
        dataset=args.dataset,
        split=split,
        fold=fold,
        requested_missing_rate=mask_rate,
        mask_seed=SeedBundle(master_seed=args.seed).derive("missing_mask"),
        freeze_evaluation=(
            getattr(args, "evaluation_protocol", "official") == "strict"
        ),
    )


def build_primary_mask_tensors(
    mask_schedule,
    conversation_ids,
    umask,
    epoch,
):
    """Materialize host/guest ``[sequence, batch, 3]`` availability tensors."""
    if not isinstance(mask_schedule, ConversationMaskSchedule):
        raise TypeError("mask_schedule must be a ConversationMaskSchedule")
    if umask.ndim != 2:
        raise ValueError("umask must have shape [batch, sequence]")
    conversation_ids = list(conversation_ids)
    batch_size, sequence_length = umask.shape
    if len(conversation_ids) != batch_size:
        raise ValueError("conversation IDs must match the umask batch size")

    side_tensors = []
    for side in ("host", "guest"):
        conversations = []
        for batch_index, conversation_id in enumerate(conversation_ids):
            valid_length = int(umask[batch_index].sum().item())
            if valid_length < 1:
                raise ValueError(
                    "conversation {!r} has no real utterances".format(
                        conversation_id
                    )
                )
            generated = mask_schedule.generate(
                str(conversation_id),
                length=sequence_length,
                valid_length=valid_length,
                side=side,
                epoch=epoch,
            )
            conversations.append(torch.as_tensor(generated.availability))
        side_tensors.append(
            torch.stack(conversations, dim=1).to(device=umask.device)
        )
    return tuple(side_tensors)


def primary_mask_audit(host_availability, guest_availability, umask):
    """Count missing modality elements over real utterances only."""
    valid = umask.transpose(0, 1).bool().unsqueeze(-1)
    missing_elements = 0
    total_elements = 0
    for availability in (host_availability, guest_availability):
        expanded_valid = valid.expand_as(availability)
        total_elements += int(expanded_valid.sum().item())
        missing_elements += int(
            ((availability == 0) & expanded_valid).sum().item()
        )
    realized = (
        float(missing_elements) / float(total_elements)
        if total_elements
        else 0.0
    )
    return {
        "missing_elements": missing_elements,
        "total_elements": total_elements,
        "realized_missing_rate": realized,
    }


## follow cpm-net's masking manner
def random_mask(view_num, input_len, missing_rate, rng=None):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    """

    assert missing_rate is not None
    if rng is None:
        random_integers = randint
    elif hasattr(rng, "integers"):
        random_integers = rng.integers
    else:
        random_integers = rng.randint
    one_rate = 1 - missing_rate      # missing_rate: 0.8; one_rate: 0.2

    if one_rate <= (1 / view_num): # 
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(random_integers(0, view_num, size=(input_len, 1))).toarray() # only select one view [avoid all zero input]
        return view_preserve # [samplenum, viewnum] => one value set=1, others=0

    if one_rate == 1:
        matrix = random_integers(1, 2, size=(input_len, view_num)) # [samplenum, viewnum] => all ones
        return matrix

    ## for one_rate between [1 / view_num, 1] => can have multi view input
    ## ensure at least one of them is avaliable 
    ## since some sample is overlapped, which increase difficulties
    if input_len < 32:
        alldata_len = 32
    else:
        alldata_len = input_len
    error = 1
    while error >= 0.005:

        ## gain initial view_preserve
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(random_integers(0, view_num, size=(alldata_len, 1))).toarray() # [samplenum, viewnum=2] => one value set=1, others=0

        ## further generate one_num samples
        one_num = view_num * alldata_len * one_rate - alldata_len  # left one_num after previous step
        ratio = one_num / (view_num * alldata_len)                 # now processed ratio
        matrix_iter = (random_integers(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int64) # based on ratio => matrix_iter
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int64)) # a: overlap number
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (random_integers(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int64)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int64)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    
    matrix = matrix[:input_len, :]
    return matrix


def build_masked_auxiliary_view(input_tensor, missing_rate, dimensions, rng=None):
    """Create a training-only masked view while preserving the full target.

    Args:
        input_tensor: Full fused features [seq, batch, A+T+V].
        missing_rate: Probability-like rate used by GCNet's random_mask.
        dimensions: Tuple of audio, text, and visual dimensions.

    Returns:
        masked_tensor: Auxiliary input [seq, batch, A+T+V].
        availability: Modality availability mask [seq, batch, 3].
    """
    if input_tensor.ndim != 3 or len(dimensions) != 3:
        raise ValueError("expected input [seq, batch, D] and three dimensions")
    if sum(dimensions) != input_tensor.size(-1):
        raise ValueError("modality dimensions do not match input feature size")
    sequence_length, batch_size, _ = input_tensor.shape
    matrix = random_mask(
        3, sequence_length * batch_size, missing_rate, rng=rng
    )
    availability = torch.as_tensor(
        matrix,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    ).reshape(sequence_length, batch_size, 3)
    expanded = torch.cat(
        [
            availability[..., index:index + 1].expand(
                sequence_length, batch_size, dimension
            )
            for index, dimension in enumerate(dimensions)
        ],
        dim=-1,
    )
    return input_tensor * expanded, availability


def create_stability_mask_rng(seed):
    component_seed = SeedBundle(master_seed=seed).derive("stability_mask")
    return np.random.RandomState(component_seed)


@contextmanager
def preserve_torch_rng_state():
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def compute_stability_reconstruction_loss(
    args,
    model,
    rec_loss,
    input_features,
    qmask,
    umask,
    lengths,
    dimensions,
    train,
    stability_mask_rng,
):
    if not train or getattr(args, "stability_recon_weight", 0.0) <= 0.0:
        return None
    auxiliary_tensor, auxiliary_availability = build_masked_auxiliary_view(
        input_features[0],
        missing_rate=args.stability_aux_mask_rate,
        dimensions=dimensions,
        rng=stability_mask_rng,
    )
    with preserve_torch_rng_state():
        _, _, auxiliary_hidden, _ = model(
            [auxiliary_tensor],
            qmask,
            umask,
            lengths,
            predict_modalities=False,
            detach_predictor_input=False,
        )
        auxiliary_reconstruction = [
            model.reconstruct_stability(auxiliary_hidden)
        ]
    adim, tdim, vdim = dimensions
    return rec_loss(
        auxiliary_reconstruction,
        input_features,
        [auxiliary_availability],
        umask,
        adim,
        tdim,
        vdim,
    )


def validate_training_args(args):
    if args.model_variant == "replacement" and args.loss_recon:
        raise ValueError(
            "--model-variant replacement cannot be combined with --loss-recon"
        )
    if args.model_variant == "replacement" and args.all_modal_recon_weight:
        raise ValueError(
            "--all-modal-recon-weight requires the addon reconstruction head"
        )
    if not 0.0 <= args.stability_aux_mask_rate <= 0.7:
        raise ValueError(
            "--stability-aux-mask-rate must be between 0.0 and 0.7"
        )
    if args.stability_recon_weight < 0.0:
        raise ValueError("--stability-recon-weight must be non-negative")
    validation_fraction = getattr(args, "validation_fraction", 0.1)
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be strictly between 0 and 1")


def compose_total_loss(
    args,
    regression_loss,
    missing_reconstruction_loss,
    jepa_loss,
    all_modal_reconstruction_loss,
    enable_prediction,
    stability_reconstruction_loss=None,
):
    loss = regression_loss
    if args.loss_recon:
        loss = loss + missing_reconstruction_loss
    if enable_prediction:
        loss = loss + args.jepa_weight * jepa_loss
    loss = loss + args.all_modal_recon_weight * all_modal_reconstruction_loss
    if stability_reconstruction_loss is not None:
        loss = (
            loss
            + args.stability_recon_weight * stability_reconstruction_loss
        )
    return loss


def build_loss_vector(
    total,
    primary,
    missing_reconstruction,
    jepa,
    all_modal_reconstruction,
    stability_reconstruction,
):
    return [
        total,
        primary,
        missing_reconstruction,
        jepa,
        all_modal_reconstruction,
        stability_reconstruction,
    ]


def train_or_eval_model(
    args,
    model,
    reg_loss,
    cls_loss,
    rec_loss,
    dataloader,
    modality_means,
    mask_rate=None,
    optimizer=None,
    train=False,
    compute_diagnostics=False,
    *,
    all_modal_rec_loss=None,
    stability_mask_rng=None,
    split=None,
    fold=1,
    epoch=0,
    mask_schedule=None,
    collect_artifacts=True,
):
    if all_modal_rec_loss is None:
        all_modal_rec_loss = AllModalReconLoss()
    if stability_mask_rng is None:
        stability_mask_rng = create_stability_mask_rng(args.seed)
    if split is None:
        split = "train" if train else "validation"
    if mask_schedule is None and mask_rate is not None:
        mask_schedule = build_mask_schedule(args, split, fold, mask_rate)
    return _train_or_eval_model_impl(
        args,
        model,
        reg_loss,
        cls_loss,
        rec_loss,
        dataloader,
        modality_means,
        mask_rate=mask_rate,
        optimizer=optimizer,
        train=train,
        compute_diagnostics=compute_diagnostics,
        all_modal_rec_loss=all_modal_rec_loss,
        stability_mask_rng=stability_mask_rng,
        split=split,
        fold=fold,
        epoch=epoch,
        mask_schedule=mask_schedule,
        collect_artifacts=collect_artifacts,
    )


def _train_or_eval_model_impl(
    args,
    model,
    reg_loss,
    cls_loss,
    rec_loss,
    dataloader,
    modality_means,
    mask_rate=None,
    optimizer=None,
    train=False,
    compute_diagnostics=False,
    *,
    all_modal_rec_loss,
    stability_mask_rng,
    split,
    fold,
    epoch,
    mask_schedule,
    collect_artifacts,
):
    preds, masks, labels, vidnames = [], [], [], []
    savepreds, savelabels, savespeakers, savehiddens, savefmask = [], [], [], [], []
    losses, losses1, losses2, losses3, losses4, losses5 = [], [], [], [], [], []
    diagnostic_predictions = {"audio": [], "text": [], "visual": []}
    diagnostic_targets = {"audio": [], "text": [], "visual": []}
    collapse_tensors = {
        "temporal_pre": [],
        "temporal_hidden": [],
        "speaker_pre": [],
        "speaker_hidden": [],
        "final_hidden": [],
        "predictions": [],
        "labels": [],
    }
    mask_missing_elements = 0
    mask_total_elements = 0

    dataset = args.dataset
    reccls_flag = args.reccls_flag
    lower_bound = args.lower_bound
    cuda = torch.cuda.is_available() and not args.no_cuda

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    enable_prediction = bool(mask_rate and args.jepa_weight)
    record_epoch_collapse = bool(
        train
        and getattr(args, "epoch_collapse_diagnostics", False)
        and dataset in ["CMUMOSI", "CMUMOSEI"]
    )
    model.graph_net_temporal.record_activation_diagnostics = record_epoch_collapse
    model.graph_net_speaker.record_activation_diagnostics = record_epoch_collapse

    for data in dataloader:
        if train: optimizer.zero_grad()
        
        ## read dataloader
        """
        audio_host, text_host, visual_host: [seqlen, batch, dim]
        audio_guest, text_guest, visual_guest: [seqlen, batch, dim]
        qmask: speakers, [batch, seqlen]
        umask: has utt, [batch, seqlen]
        label: [batch, seqlen]
        """
        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        qmask, umask, label = data[6], data[7], data[8]
        if collect_artifacts:
            vidnames += data[-1]
        adim = audio_host.size(2)
        tdim = text_host.size(2)
        vdim = visual_host.size(2)

        ## using cmp-net masking manner [at least one view exists]
        """
        ?_?_mask: [seqlen, batch, dim]   => gain mask
        masked_?_?: [seqlen, batch, dim] => masked features

        # if audio_feature is None: audio_feature = text_feature
        # if text_feature is None: text_feature = audio_feature
        # if video_feature is None: video_feature = text_feature
        # mask sure, same mask for same features [include padded features]
        """
        if mask_schedule is None:
            raise ValueError("primary masking requires an explicit mask schedule")
        host_availability, guest_availability = build_primary_mask_tensors(
            mask_schedule,
            conversation_ids=data[-1],
            umask=umask,
            epoch=epoch,
        )
        batch_mask_audit = primary_mask_audit(
            host_availability, guest_availability, umask
        )
        mask_missing_elements += batch_mask_audit["missing_elements"]
        mask_total_elements += batch_mask_audit["total_elements"]
        audio_host_mask = host_availability[..., 0:1]
        text_host_mask = host_availability[..., 1:2]
        visual_host_mask = host_availability[..., 2:3]
        audio_guest_mask = guest_availability[..., 0:1]
        text_guest_mask = guest_availability[..., 1:2]
        visual_guest_mask = guest_availability[..., 2:3]

        ## lower bound==True => remove missing data
        if not lower_bound:
            masked_audio_host = audio_host * audio_host_mask
            masked_audio_guest = audio_guest * audio_guest_mask
            masked_text_host = text_host * text_host_mask
            masked_text_guest = text_guest * text_guest_mask
            masked_visual_host = visual_host * visual_host_mask
            masked_visual_guest = visual_guest * visual_guest_mask
        else:
            host_mask = torch.logical_and(torch.logical_and(audio_host_mask, text_host_mask), visual_host_mask).int() # [seqlen, bacth, 1]
            masked_audio_host = audio_host * host_mask
            masked_text_host = text_host * host_mask
            masked_visual_host = visual_host * host_mask
            audio_host_mask = host_mask
            text_host_mask = host_mask
            visual_host_mask = host_mask
            guest_mask = torch.logical_and(torch.logical_and(audio_guest_mask, text_guest_mask), visual_guest_mask).int() # [seqlen, bacth, 1]
            masked_audio_guest = audio_guest * guest_mask
            masked_text_guest = text_guest * guest_mask
            masked_visual_guest = visual_guest * guest_mask
            audio_guest_mask = guest_mask
            text_guest_mask = guest_mask
            visual_guest_mask = guest_mask

        ## add cuda for tensor
        if cuda:
            audio_host = audio_host.cuda()
            text_host = text_host.cuda()
            visual_host = visual_host.cuda()
            audio_guest = audio_guest.cuda()
            text_guest = text_guest.cuda()
            visual_guest = visual_guest.cuda()

            masked_audio_host, audio_host_mask = masked_audio_host.cuda(), audio_host_mask.cuda()
            masked_text_host, text_host_mask = masked_text_host.cuda(), text_host_mask.cuda()
            masked_visual_host, visual_host_mask = masked_visual_host.cuda(), visual_host_mask.cuda()
            masked_audio_guest, audio_guest_mask = masked_audio_guest.cuda(), audio_guest_mask.cuda()
            masked_text_guest, text_guest_mask = masked_text_guest.cuda(), text_guest_mask.cuda()
            masked_visual_guest, visual_guest_mask = masked_visual_guest.cuda(), visual_guest_mask.cuda()

            qmask = qmask.cuda()
            umask = umask.cuda()
            label = label.cuda()

        ## [conversation_len1, conversation_len2, ..., conversation_lenN]
        lengths = []
        for j in range(len(umask)):
            length = (umask[j] == 1).nonzero().tolist()[-1][0] + 1 
            lengths.append(length)

        ## generate input_features: ? * [seqlen, batch, dim]
        input_features = generate_inputs(audio_host, text_host, visual_host, \
                                         audio_guest, text_guest, visual_guest, qmask)
        masked_input_features = generate_inputs(masked_audio_host, masked_text_host, masked_visual_host, \
                                                masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)
        input_features_mask = generate_inputs(audio_host_mask, text_host_mask, visual_host_mask, \
                                                audio_guest_mask, text_guest_mask, visual_guest_mask, qmask)

        '''
        # input_features, masked_input_features, input_features_mask: ?*[seqlen, batch, dim]
        # qmask: speakers, [batch, seqlen]
        # umask: has utt, [batch, seqlen]
        # label: [batch, seqlen]
        # log_prob: [seqlen, batch, num_classes]
        # input_features_recon # padded, ?*[seqlen, batch, dim]
        '''
        if reccls_flag: # whether use reconstruction features for classification
            _, recon_input_features, _, _ = model(
                masked_input_features, qmask, umask, lengths,
                predict_modalities=enable_prediction,
                detach_predictor_input=args.detach_predictor_input,
            )
            log_prob, _, hidden, modality_predictions = model(
                recon_input_features, qmask, umask, lengths,
                predict_modalities=enable_prediction,
                detach_predictor_input=args.detach_predictor_input,
            )
        else:
            log_prob, recon_input_features, hidden, modality_predictions = model(
                masked_input_features, qmask, umask, lengths,
                predict_modalities=enable_prediction,
                detach_predictor_input=args.detach_predictor_input,
            )

        if record_epoch_collapse:
            valid = umask.transpose(0, 1).bool()
            temporal_pre = model.graph_net_temporal.last_pre_activation
            temporal_hidden = model.graph_net_temporal.last_hidden
            speaker_pre = model.graph_net_speaker.last_pre_activation
            speaker_hidden = model.graph_net_speaker.last_hidden
            if any(value is None for value in (
                temporal_pre, temporal_hidden, speaker_pre, speaker_hidden
            )):
                raise RuntimeError("activation diagnostics were not captured")
            collapse_tensors["temporal_pre"].append(
                temporal_pre[valid].detach().cpu()
            )
            collapse_tensors["temporal_hidden"].append(
                temporal_hidden[valid].detach().cpu()
            )
            collapse_tensors["speaker_pre"].append(
                speaker_pre[valid].detach().cpu()
            )
            collapse_tensors["speaker_hidden"].append(
                speaker_hidden[valid].detach().cpu()
            )
            collapse_tensors["final_hidden"].append(hidden[valid].detach().cpu())
            collapse_tensors["predictions"].append(
                log_prob[valid].detach().cpu()
            )
            collapse_tensors["labels"].append(
                label[umask.bool()].detach().cpu()
            )

        if collect_artifacts:
            ## gain saved results [utterance-level]
            tempseqlen = np.sum(umask.cpu().data.numpy(), 1) # [batch]
            temphidden = hidden.transpose(0,1).cpu().data.numpy() # [batch, seqlen, featdim]
            temppred = log_prob.transpose(0,1).cpu().data.numpy() # [batch, seqlen, num_classes]
            templabel = label.cpu().data.numpy() # [batch, seqlen]
            tempqmask = qmask.cpu().data.numpy() # [batch, seqlen]
            tempfmask = input_features_mask[0].transpose(0,1).cpu().data.numpy() # [seqlen, batch, 3] -> [batch, seqlen, 3]
            for ii in range(len(tempseqlen)): # utt_number for each conversation
                itemhidden = temphidden[ii][:int(tempseqlen[ii]), :] # [seqlen, featdim]
                itempred   = temppred[ii][:int(tempseqlen[ii]), :]   # [seqlen, num_classes]
                itemfmask  = tempfmask[ii][:int(tempseqlen[ii]), :]  # [seqlen, 3]
                itemlabel  = templabel[ii][:int(tempseqlen[ii])]     # [len, ]
                itemspks   = tempqmask[ii][:int(tempseqlen[ii])]     # [len, ]
                savehiddens.append(itemhidden)
                savepreds.append(itempred)
                savefmask.append(itemfmask)
                savelabels.append(itemlabel)
                savespeakers.append(itemspks)

        ## calculate loss
        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size(2)) # [batch*seq_len, n_classes]
        labels_ = label.view(-1) # [batch*seq_len]
        if dataset in ['IEMOCAPFour', 'IEMOCAPSix']: loss1 = cls_loss(lp_, labels_, umask)
        if dataset in ['CMUMOSI', 'CMUMOSEI']  : loss1 = reg_loss(lp_, labels_, umask)
        if recon_input_features:
            loss2 = rec_loss(recon_input_features, input_features, input_features_mask, umask, adim, tdim, vdim)
        else:
            loss2 = log_prob.new_zeros(())
        if args.all_modal_recon_weight and recon_input_features:
            loss4 = all_modal_rec_loss(
                recon_input_features, input_features, umask, adim, tdim, vdim
            )
        else:
            loss4 = log_prob.new_zeros(())
        if modality_predictions is None:
            loss3, _ = miss0_jepa_loss(model)
            missing_counts = {"audio": 0, "text": 0, "visual": 0}
        else:
            loss3, missing_counts = masked_centered_cosine_loss(
                modality_predictions,
                input_features[0],
                input_features_mask[0],
                umask,
                modality_means,
            )
        stability_reconstruction_loss = compute_stability_reconstruction_loss(
            args=args,
            model=model,
            rec_loss=rec_loss,
            input_features=input_features,
            qmask=qmask,
            umask=umask,
            lengths=lengths,
            dimensions=(adim, tdim, vdim),
            train=train,
            stability_mask_rng=stability_mask_rng,
        )
        loss5 = (
            log_prob.new_zeros(())
            if stability_reconstruction_loss is None
            else stability_reconstruction_loss
        )
        loss = compose_total_loss(
            args,
            loss1,
            loss2,
            loss3,
            loss4,
            enable_prediction,
            stability_reconstruction_loss=loss5,
        )
        named_losses = (
            ("total", loss),
            ("primary", loss1),
            ("missing reconstruction", loss2),
            ("JEPA", loss3),
            ("all-modal reconstruction", loss4),
            ("stability reconstruction", loss5),
        )
        for loss_name, loss_value in named_losses:
            if not bool(torch.isfinite(loss_value.detach()).all().item()):
                raise ValueError("{} loss must be finite".format(loss_name))

        if not train and compute_diagnostics and modality_predictions is not None:
            valid = umask.transpose(0, 1).bool()
            dimensions = (adim, tdim, vdim)
            target_parts = torch.split(input_features[0], dimensions, dim=-1)
            pred_parts = (
                modality_predictions.audio,
                modality_predictions.text,
                modality_predictions.visual,
            )
            mean_parts = (
                modality_means.audio,
                modality_means.text,
                modality_means.visual,
            )
            for modality_index, modality_name in enumerate(("audio", "text", "visual")):
                selected = valid & (input_features_mask[0][..., modality_index] == 0)
                if selected.any():
                    diagnostic_predictions[modality_name].append(
                        pred_parts[modality_index][selected].detach().cpu()
                    )
                    diagnostic_targets[modality_name].append(
                        (target_parts[modality_index][selected] - mean_parts[modality_index])
                        .detach().cpu()
                    )
        
        ## save batch results
        # pred_ = torch.argmax(lp_,1) # [batch*seq_len]
        preds.append(lp_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())
        losses1.append(loss1.item()*masks[-1].sum())
        losses2.append(loss2.item()*masks[-1].sum())
        losses3.append(loss3.item()*masks[-1].sum())
        losses4.append(loss4.item()*masks[-1].sum())
        losses5.append(loss5.item()*masks[-1].sum())

        if train:
            loss.backward()
            optimizer.step()

    assert preds!=[], f'Error: no dataset in dataloader'
    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)

    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        preds = np.argmax(preds, 1)
        avg_loss = round(np.sum(losses)/np.sum(masks), 4)
        avg_loss1 = round(np.sum(losses1)/np.sum(masks), 4)
        avg_loss2 = round(np.sum(losses2)/np.sum(masks), 4)
        avg_accuracy = accuracy_score(labels, preds, sample_weight=masks)
        avg_fscore = f1_score(labels, preds, sample_weight=masks, average='weighted')
    elif dataset in ['CMUMOSI', 'CMUMOSEI']:
        non_zeros = np.array([i for i, e in enumerate(labels) if e != 0]) # remove 0, and remove mask
        avg_loss = round(np.sum(losses)/np.sum(masks), 4)
        avg_loss1 = round(np.sum(losses1)/np.sum(masks), 4)
        avg_loss2 = round(np.sum(losses2)/np.sum(masks), 4)
        avg_accuracy = accuracy_score((labels[non_zeros] > 0), (preds[non_zeros] > 0))
        avg_fscore = f1_score((labels[non_zeros] > 0), (preds[non_zeros] > 0), average='weighted')
        
    avg_loss3 = round(np.sum(losses3)/np.sum(masks), 4)
    avg_loss4 = round(np.sum(losses4)/np.sum(masks), 4)
    avg_loss5 = round(np.sum(losses5)/np.sum(masks), 4)
    diagnostics = {}
    if record_epoch_collapse:
        diagnostics = compute_epoch_collapse_diagnostics(
            temporal_pre=torch.cat(collapse_tensors["temporal_pre"], dim=0),
            temporal_hidden=torch.cat(collapse_tensors["temporal_hidden"], dim=0),
            speaker_pre=torch.cat(collapse_tensors["speaker_pre"], dim=0),
            speaker_hidden=torch.cat(collapse_tensors["speaker_hidden"], dim=0),
            final_hidden=torch.cat(collapse_tensors["final_hidden"], dim=0),
            predictions=torch.cat(collapse_tensors["predictions"], dim=0),
            labels=torch.cat(collapse_tensors["labels"], dim=0),
            regression_head=model.smax_fc,
        )
    model.graph_net_temporal.record_activation_diagnostics = False
    model.graph_net_speaker.record_activation_diagnostics = False
    if not train and compute_diagnostics:
        for modality_name in ("audio", "text", "visual"):
            if diagnostic_predictions[modality_name]:
                diagnostics[modality_name] = compute_modality_diagnostics(
                    torch.cat(diagnostic_predictions[modality_name], dim=0),
                    torch.cat(diagnostic_targets[modality_name], dim=0),
                    shuffle_seed=args.seed,
                )
            else:
                diagnostics[modality_name] = compute_modality_diagnostics(
                    torch.empty(0, 1), torch.empty(0, 1), shuffle_seed=args.seed
                )
    diagnostics["primary_mask"] = {
        "missing_elements": int(mask_missing_elements),
        "total_elements": int(mask_total_elements),
        "realized_missing_rate": (
            float(mask_missing_elements) / float(mask_total_elements)
            if mask_total_elements
            else 0.0
        ),
    }
    print (f'sample number: {np.sum(masks)}')
    loss_vector = build_loss_vector(
        total=avg_loss,
        primary=avg_loss1,
        missing_reconstruction=avg_loss2,
        jepa=avg_loss3,
        all_modal_reconstruction=avg_loss4,
        stability_reconstruction=avg_loss5,
    )
    return avg_accuracy, avg_fscore, vidnames, loss_vector, [savepreds, savelabels, savespeakers, savehiddens, savefmask], diagnostics


def _require_finite_result(result, split):
    try:
        losses = result[3]
    except (IndexError, TypeError) as error:
        raise ValueError("{} result does not contain a loss vector".format(split)) from error
    for index, value in enumerate(losses):
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(
                "{} loss at index {} must be finite".format(split, index)
            )
    try:
        fscore = float(result[1])
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("{} weighted F1 must be finite".format(split)) from error
    if not math.isfinite(fscore):
        raise ValueError("{} weighted F1 must be finite".format(split))


def _snapshot_cpu_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _compact_epoch_result(result):
    diagnostics = result[5]
    if isinstance(diagnostics, dict):
        diagnostics = dict(diagnostics)
    return {
        "accuracy": float(result[0]),
        "weighted_f1": float(result[1]),
        "loss": [float(value) for value in result[3]],
        "diagnostics": diagnostics,
    }


def build_fold_archive_entry(best_epoch, final_test_payload):
    """Keep the legacy zero-based checkpoint index plus one final payload."""
    return [int(best_epoch) - 1, final_test_payload]


def lifecycle_manifest_evidence(lifecycle):
    """Extract compact mask/checkpoint evidence without retaining artifacts."""
    train_rates = []
    validation_rates = []
    test_rates = []
    for record in lifecycle["epoch_records"]:
        train_rates.append(
            float(record["train"]["diagnostics"]["primary_mask"]["realized_missing_rate"])
        )
        validation_rates.append(
            float(record["validation"]["diagnostics"]["primary_mask"]["realized_missing_rate"])
        )
        if "test" in record:
            test_rates.append(
                float(record["test"]["diagnostics"]["primary_mask"]["realized_missing_rate"])
            )
    if not validation_rates or not train_rates:
        raise ValueError("manifest requires nonempty train and validation mask evidence")
    evaluation_protocol = lifecycle.get("evaluation_protocol", "official")
    if evaluation_protocol == "strict":
        if not all(np.isclose(rate, validation_rates[0]) for rate in validation_rates):
            raise ValueError("fixed validation mask produced inconsistent realized rates")
        test_diagnostics = lifecycle["test_result"][5]
        validation_evidence = validation_rates[0]
        test_evidence = float(
            test_diagnostics["primary_mask"]["realized_missing_rate"]
        )
    else:
        if len(test_rates) != len(train_rates):
            raise ValueError("official protocol requires one test result per epoch")
        validation_evidence = validation_rates
        test_evidence = test_rates
    return {
        "evaluation_protocol": evaluation_protocol,
        "epochs_completed": len(lifecycle["epoch_records"]),
        "best_epoch": int(lifecycle["best_epoch"]),
        "best_validation_f1": float(lifecycle["best_validation_f1"]),
        "test_call_count": int(lifecycle["test_call_count"]),
        "mask_schedule_hashes": dict(lifecycle["mask_schedule_hashes"]),
        "realized_missing_rates": {
            "train": train_rates,
            "validation": validation_evidence,
            "test": test_evidence,
        },
    }


def build_fold_run_manifest(
    *,
    args,
    fold,
    loader_metadata,
    lifecycle_evidence,
    fold_record,
    feature_evidence,
    environment,
    provenance,
    shared_init_hash,
    training_seed,
    mask_rate,
    output_paths,
):
    """Assemble one complete, validation-ready fold manifest."""
    split_hashes = {
        metadata["split_hash"] for metadata in loader_metadata.values()
    }
    if len(split_hashes) != 1:
        raise ValueError("train/validation/test loaders have different split hashes")
    bundle = SeedBundle(args.seed)
    return {
        "schema": {"name": MANIFEST_NAME, "version": MANIFEST_VERSION},
        "run": {
            "dataset": args.dataset,
            "fold": int(fold),
            "master_seed": int(args.seed),
        },
        "environment": environment,
        "provenance": provenance,
        "features": feature_evidence,
        "split": {
            "indices": {
                split: list(loader_metadata[split]["indices"])
                for split in ("train", "validation", "test")
            },
            "hash": next(iter(split_hashes)),
        },
        "samplers": {
            split: {
                "seed": int(loader_metadata[split]["order_seed"]),
                "signature": loader_metadata[split]["order_signature"],
            }
            for split in ("train", "validation", "test")
        },
        "masks": {
            "requested_missing_rate": float(mask_rate),
            "config_hashes": dict(lifecycle_evidence["mask_schedule_hashes"]),
            "realized_missing_rates": dict(
                lifecycle_evidence["realized_missing_rates"]
            ),
        },
        "seeds": {
            "model_init": bundle.derive("model_init:fold:{}".format(fold)),
            "training_stochasticity": int(training_seed),
            "split": bundle.derive("split"),
            "data_order": {
                split: int(loader_metadata[split]["order_seed"])
                for split in ("train", "validation", "test")
            },
            "missing_mask": bundle.derive("missing_mask"),
            "stability_mask": bundle.derive(
                "stability_mask:fold:{}".format(fold)
            ),
        },
        "initialization": {"shared_hash": shared_init_hash},
        "stability": {
            "enabled": bool(getattr(args, "stability_recon_weight", 0.0) > 0.0),
            "mask_rate": float(getattr(args, "stability_aux_mask_rate", 0.0)),
            "weight": float(getattr(args, "stability_recon_weight", 0.0)),
        },
        "method": {
            "model_variant": getattr(args, "model_variant", "addon"),
            "jepa_weight": float(getattr(args, "jepa_weight", 0.0)),
            "loss_reconstruction": bool(getattr(args, "loss_recon", False)),
            "all_modal_reconstruction_weight": float(
                getattr(args, "all_modal_recon_weight", 0.0)
            ),
        },
        "lifecycle": {
            "evaluation_protocol": lifecycle_evidence["evaluation_protocol"],
            "checkpoint_metric": "validation_weighted_f1",
            "best_epoch": int(lifecycle_evidence["best_epoch"]),
            "best_validation_f1": float(
                lifecycle_evidence["best_validation_f1"]
            ),
            "test_call_count": int(lifecycle_evidence["test_call_count"]),
            "epochs_completed": int(lifecycle_evidence["epochs_completed"]),
        },
        "metrics": {
            "weighted_f1": float(fold_record["weighted_f1"]),
            "accuracy": float(fold_record["accuracy"]),
        },
        "outputs": dict(output_paths),
    }


def run_training_fold(
    *,
    args,
    model,
    reg_loss,
    cls_loss,
    rec_loss,
    train_loader,
    val_loader,
    test_loader,
    modality_means,
    mask_rate,
    optimizer,
    fold,
    all_modal_rec_loss=None,
    stability_mask_rng=None,
    evaluation_fn=None,
):
    """Run either the official every-epoch test or strict test-once lifecycle."""
    if evaluation_fn is None:
        evaluation_fn = train_or_eval_model
    schedules = {
        split: build_mask_schedule(args, split, fold, mask_rate)
        for split in ("train", "validation", "test")
    }
    epoch_records = []
    best_validation_f1 = -float("inf")
    best_epoch = None
    best_model_state = None
    best_test_result = None
    test_call_count = 0
    evaluation_protocol = getattr(args, "evaluation_protocol", "official")
    if evaluation_protocol not in ("official", "strict"):
        raise ValueError("evaluation_protocol must be official or strict")

    for epoch in range(args.epochs):
        epoch_loaders = (
            (train_loader, val_loader, test_loader)
            if evaluation_protocol == "official"
            else (train_loader,)
        )
        for loader in epoch_loaders:
            sampler = getattr(loader, "sampler", None)
            if sampler is None or not hasattr(sampler, "set_epoch"):
                raise TypeError("protocol loader sampler must implement set_epoch")
            sampler.set_epoch(epoch)

        train_result = evaluation_fn(
            args,
            model,
            reg_loss,
            cls_loss,
            rec_loss,
            train_loader,
            modality_means,
            mask_rate=mask_rate,
            optimizer=optimizer,
            train=True,
            compute_diagnostics=False,
            all_modal_rec_loss=all_modal_rec_loss,
            stability_mask_rng=stability_mask_rng,
            split="train",
            fold=fold,
            epoch=epoch,
            mask_schedule=schedules["train"],
            collect_artifacts=False,
        )
        _require_finite_result(train_result, "train")
        validation_result = evaluation_fn(
            args,
            model,
            reg_loss,
            cls_loss,
            rec_loss,
            val_loader,
            modality_means,
            mask_rate=mask_rate,
            optimizer=None,
            train=False,
            compute_diagnostics=False,
            all_modal_rec_loss=all_modal_rec_loss,
            stability_mask_rng=stability_mask_rng,
            split="validation",
            fold=fold,
            epoch=epoch if evaluation_protocol == "official" else 0,
            mask_schedule=schedules["validation"],
            collect_artifacts=False,
        )
        _require_finite_result(validation_result, "validation")
        validation_f1 = float(validation_result[1])
        is_best = validation_f1 > best_validation_f1
        if is_best:
            best_validation_f1 = validation_f1
            best_epoch = epoch + 1
            if evaluation_protocol == "strict":
                best_model_state = _snapshot_cpu_state(model)
        epoch_record = {
            "epoch": epoch + 1,
            "train": _compact_epoch_result(train_result),
            "validation": _compact_epoch_result(validation_result),
        }
        if evaluation_protocol == "official":
            test_result = evaluation_fn(
                args,
                model,
                reg_loss,
                cls_loss,
                rec_loss,
                test_loader,
                modality_means,
                mask_rate=mask_rate,
                optimizer=None,
                train=False,
                compute_diagnostics=is_best,
                all_modal_rec_loss=all_modal_rec_loss,
                stability_mask_rng=stability_mask_rng,
                split="test",
                fold=fold,
                epoch=epoch,
                mask_schedule=schedules["test"],
                collect_artifacts=is_best,
            )
            test_call_count += 1
            _require_finite_result(test_result, "test")
            epoch_record["test"] = _compact_epoch_result(test_result)
            if is_best:
                best_test_result = test_result
        epoch_records.append(epoch_record)

    if evaluation_protocol == "official":
        if best_test_result is None or best_epoch is None:
            raise RuntimeError("no finite validation epoch was selected")
        return {
            "evaluation_protocol": evaluation_protocol,
            "best_epoch": best_epoch,
            "best_validation_f1": best_validation_f1,
            "epoch_records": epoch_records,
            "test_result": best_test_result,
            "test_call_count": test_call_count,
            "mask_schedule_hashes": {
                split: schedule.config_hash for split, schedule in schedules.items()
            },
        }

    if best_model_state is None or best_epoch is None:
        raise RuntimeError("no finite validation checkpoint was selected")
    model.load_state_dict(best_model_state, strict=True)
    test_result = evaluation_fn(
        args,
        model,
        reg_loss,
        cls_loss,
        rec_loss,
        test_loader,
        modality_means,
        mask_rate=mask_rate,
        optimizer=None,
        train=False,
        compute_diagnostics=True,
        all_modal_rec_loss=all_modal_rec_loss,
        stability_mask_rng=stability_mask_rng,
        split="test",
        fold=fold,
        epoch=0,
        mask_schedule=schedules["test"],
        collect_artifacts=True,
    )
    _require_finite_result(test_result, "test")
    return {
        "evaluation_protocol": evaluation_protocol,
        "best_epoch": best_epoch,
        "best_validation_f1": best_validation_f1,
        "epoch_records": epoch_records,
        "test_result": test_result,
        "test_call_count": test_call_count + 1,
        "mask_schedule_hashes": {
            split: schedule.config_hash for split, schedule in schedules.items()
        },
    }


def prepare_shared_initialization(
    model,
    checkpoint_path=None,
    required_hash=None,
):
    """Load/validate shared tensors without touching variant-specific heads."""
    if checkpoint_path is not None:
        return load_shared_checkpoint(
            checkpoint_path,
            model,
            expected_hash=required_hash,
        )
    actual_hash = shared_state_hash(model)
    if required_hash is not None and actual_hash != required_hash:
        raise ValueError(
            "required shared initialization hash does not match: {} != {}".format(
                required_hash, actual_hash
            )
        )
    return actual_hash


def build_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio-feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text-feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video-feature', type=str, default=None, help='video feature name')
    parser.add_argument('--dataset', type=str, default='IEMOCAPFour', help='dataset type')

    parser.add_argument('--base-model', type=str, choices=['LSTM', 'GRU'], help='base recurrent model, must be one of LSTM/GRU')
    parser.add_argument('--time-attn', action='store_true', default=False, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')
    parser.add_argument('--windowp', type=int, default=6, help='context window size for constructing edges in graph model for past utterances, -1: fully connect')
    parser.add_argument('--windowf', type=int, default=6, help='context window size for constructing edges in graph model for future utterances, -1: fully connect')
    parser.add_argument('--hidden', type=int, default=100, help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes [defined by args.dataset]')
    parser.add_argument('--n_speakers', type=int, default=2, help='number of speakers [defined by args.dataset]')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--num-threads', type=int, default=6, help='Torch CPU threads per process')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--num-folder', type=int, default=5, help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--validation-fraction', type=float, default=0.1, help='IEMOCAP validation conversation fraction from non-test sessions')
    parser.add_argument('--evaluation-protocol', choices=['official', 'strict'], default='official', help='official evaluates test every epoch; strict uses internal validation and tests once')
    parser.add_argument('--mask-type', type=str, default='constant-0.1', help='mask rate [0~1] for input argumentation: constant-float; linear; convex; concave')
    parser.add_argument('--loss-recon', action='store_true', default=False, help='whether to use reconstrctuion loss')
    parser.add_argument('--reccls-flag', action='store_true', default=False, help='whether to use reconstrctuion features for classification')
    parser.add_argument('--lower-bound', action='store_true', default=False, help='whether remove missing modality in the training process')
    parser.add_argument('--jepa-weight', type=float, default=0.1, help='centered modality prediction loss weight')
    parser.add_argument('--all-modal-recon-weight', type=float, default=0.0, help='diagnostic reconstruction weight over complete observed modalities')
    parser.add_argument('--detach-predictor-input', action='store_true', default=False, help='train only the Predictor with JEPA gradients')
    parser.add_argument('--predictor-dropout', type=float, default=0.1, help='dropout inside modality predictors')
    parser.add_argument('--model-variant', choices=['addon', 'replacement'], default='addon')
    parser.add_argument('--fold', type=int, choices=range(1, 6), default=None, help='run only one 1-based IEMOCAP fold')
    parser.add_argument('--output-dir', type=str, default=None, help='isolated result directory')
    parser.add_argument('--allow-short-run', action='store_true', default=False, help='allow fewer than 60 epochs for smoke tests')
    parser.add_argument('--epoch-collapse-diagnostics', action='store_true', default=False, help='record per-epoch GCNet activation and regression-collapse metrics')
    parser.add_argument('--strict-deterministic', action='store_true', default=False, help='error if CUDA/PyTorch selects a nondeterministic operation')
    parser.add_argument('--stability-aux-mask-rate', type=float, default=0.1, help='training-only missing rate for auxiliary reconstruction')
    parser.add_argument('--stability-recon-weight', type=float, default=0.0, help='weight for training-only masked reconstruction stability loss')
    parser.add_argument('--shared-init-checkpoint', type=str, default=None, help='load shared encoder/classifier initialization from this checkpoint')
    parser.add_argument('--require-shared-init-hash', type=str, default=None, help='require this shared initialization SHA-256 hash')
    return parser


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)
    try:
        validate_training_args(args)
    except ValueError as error:
        parser.error(str(error))
    set_random_seed(args.seed, strict_deterministic=args.strict_deterministic)

    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        args.num_folder = 1
        args.n_classes = 1
        args.n_speakers = 1
    elif args.dataset == 'IEMOCAPFour':
        args.num_folder = 5
        args.n_classes = 4
        args.n_speakers = 2
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder = 5
        args.n_classes = 6
        args.n_speakers = 2
    cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)


    print (f'====== Reading Data =======')
    audio_feature = args.audio_feature
    text_feature = args.text_feature
    video_feature = args.video_feature
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), f'features not exist!'
    run_environment = collect_environment()
    run_provenance = collect_provenance()
    feature_evidence = {
        "audio": {
            "path": str(Path(audio_root).resolve()),
            "metadata_sha256": feature_metadata_hash(audio_root),
        },
        "text": {
            "path": str(Path(text_root).resolve()),
            "metadata_sha256": feature_metadata_hash(text_root),
        },
        "visual": {
            "path": str(Path(video_root).resolve()),
            "metadata_sha256": feature_metadata_hash(video_root),
        },
    }
    train_loaders, val_loaders, test_loaders, adim, tdim, vdim = get_loaders( audio_root = audio_root,
                                                                              text_root  = text_root,
                                                                              video_root = video_root,
                                                                              num_folder = args.num_folder,
                                                                              batch_size = args.batch_size,
                                                                              dataset = args.dataset,
                                                                              num_workers = 0,
                                                                              seed = args.seed,
                                                                              validation_fraction = args.validation_fraction,
                                                                              evaluation_protocol = args.evaluation_protocol)
    assert len(train_loaders) == args.num_folder, f'Error: folder number'

    
    print (f'====== Training and Evaluation =======')
    folder_acc = []       # save best epoch
    folder_f1 = []        # save best epoch
    folder_recon = []     # save best epoch
    folder_save = []      # save best epoch
    folder_losswhole = [] # save whole epoch
    folder_savewhole = [] # save whole epoch
    fold_indices = [args.fold - 1] if args.fold is not None else list(range(args.num_folder))
    fold_records = []
    fold_manifest_contexts = []
    epoch_collapse_records = []
    for ii in fold_indices:
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        val_loader = val_loaders[ii]
        test_loader = test_loaders[ii]
        start_time = time.time()

        print (f'Step1: build model (each folder has its own model)')
        model_seed = SeedBundle(args.seed).derive(
            "model_init:fold:{}".format(ii + 1)
        )
        set_random_seed(
            model_seed,
            strict_deterministic=args.strict_deterministic,
        )
        model = build_model(args, adim, tdim, vdim)
        shared_init_hash = prepare_shared_initialization(
            model,
            checkpoint_path=args.shared_init_checkpoint,
            required_hash=args.require_shared_init_hash,
        )
        print('shared initialization hash: {}'.format(shared_init_hash))
        torch_rng_state = torch.get_rng_state()
        numpy_rng_state = np.random.get_state()
        python_rng_state = random.getstate()
        modality_means = compute_modality_means(train_loader)
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)
        random.setstate(python_rng_state)
        reg_loss = MaskedMSELoss()
        cls_loss = MaskedCELoss()
        rec_loss = MaskedReconLoss()
        all_modal_rec_loss = AllModalReconLoss()
        if cuda:
            model.cuda()
            cls_loss.cuda()
            rec_loss.cuda()
            all_modal_rec_loss.cuda()
            modality_means = modality_means.to("cuda")
            torch.cuda.reset_peak_memory_stats()
        # Shared tensors must be loaded and validated before optimizer state exists.
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        print (f'Step2: training (multiple epoches)')
        if not args.mask_type.startswith('constant'):
            raise ValueError('mask_type must be constant-x.x')
        mask_rate = float(args.mask_type.split('-')[-1])
        stability_mask_rng = np.random.RandomState(
            SeedBundle(args.seed).derive(
                "stability_mask:fold:{}".format(ii + 1)
            )
        )
        training_seed = reset_training_stochasticity(
            args.seed,
            ii + 1,
            strict_deterministic=args.strict_deterministic,
        )
        print('training stochasticity seed: {}'.format(training_seed))
        lifecycle = run_training_fold(
            args=args,
            model=model,
            reg_loss=reg_loss,
            cls_loss=cls_loss,
            rec_loss=rec_loss,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            modality_means=modality_means,
            mask_rate=mask_rate,
            optimizer=optimizer,
            fold=ii + 1,
            all_modal_rec_loss=all_modal_rec_loss,
            stability_mask_rng=stability_mask_rng,
        )
        all_losses = []
        for epoch_record in lifecycle["epoch_records"]:
            train_result = epoch_record["train"]
            validation_result = epoch_record["validation"]
            train_fscore = train_result["weighted_f1"]
            train_loss = train_result["loss"]
            val_fscore = validation_result["weighted_f1"]
            val_loss = validation_result["loss"]
            train_diagnostics = train_result["diagnostics"]
            all_losses.append({
                'train_loss': train_loss,
                'val_loss': val_loss,
            })
            if args.epoch_collapse_diagnostics:
                epoch_collapse_records.append({
                    "fold": ii + 1,
                    "epoch": epoch_record["epoch"],
                    "train_weighted_f1": float(train_fscore),
                    "val_weighted_f1": float(val_fscore),
                    "train_total_loss": float(train_loss[0]),
                    "train_regression_loss": float(train_loss[1]),
                    "train_stability_reconstruction_loss": float(train_loss[5]),
                    **train_diagnostics,
                })
            print(f'epoch:{epoch_record["epoch"]}; train_fscore:{train_fscore:2.2%}; train_loss:{train_loss[0]}; train_loss1:{train_loss[1]}; train_loss2:{train_loss[2]}; train_loss3:{train_loss[3]}; train_loss4:{train_loss[4]}; train_loss5:{train_loss[5]}')

        print (f'Step3: saving and testing on the {ii+1} folder')
        bestacc, bestf1, test_names, test_loss, testsave, bestdiagnostics = (
            lifecycle["test_result"]
        )
        bestrecon = test_loss[2]
        bestjepa = test_loss[3]
        bestallrecon = test_loss[4]
        bestsave = {
            'test_labels': testsave[1],
            'test_preds': testsave[0],
            'test_hiddens': testsave[3],
            'test_names': test_names,
            'test_fmask': testsave[4],
        }
        folder_f1.append(bestf1)
        folder_acc.append(bestacc)
        folder_recon.append(bestrecon)
        folder_save.append(bestsave)
        folder_losswhole.append(all_losses)
        if not args.allow_short_run:
            assert args.epochs >= 60, f'epoch number should large then 60'
        folder_savewhole.append(build_fold_archive_entry(
            lifecycle["best_epoch"], bestsave
        ))
        peak_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2) if cuda else 0.0
        )
        fold_record = {
            "fold": ii + 1,
            "seed": args.seed,
            "strict_deterministic": bool(args.strict_deterministic),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "missing_rate": mask_rate,
            "best_epoch": int(lifecycle["best_epoch"]),
            "best_validation_f1": float(lifecycle["best_validation_f1"]),
            "weighted_f1": float(bestf1),
            "accuracy": float(bestacc),
            "reconstruction_loss": float(bestrecon),
            "jepa_loss": float(bestjepa),
            "all_modal_reconstruction_loss": float(bestallrecon),
            "stability_aux_mask_rate": float(args.stability_aux_mask_rate),
            "stability_recon_weight": float(args.stability_recon_weight),
            "shared_init_hash": shared_init_hash,
            "mask_schedule_hashes": lifecycle["mask_schedule_hashes"],
            "test_call_count": lifecycle["test_call_count"],
            "peak_memory_mb": float(peak_memory_mb),
            "diagnostics": bestdiagnostics,
        }
        fold_records.append(fold_record)
        fold_manifest_contexts.append({
            "fold": ii + 1,
            "archive_fold_index": len(fold_manifest_contexts),
            "loader_metadata": {
                "train": dict(train_loader.protocol_metadata),
                "validation": dict(val_loader.protocol_metadata),
                "test": dict(test_loader.protocol_metadata),
            },
            "lifecycle_evidence": lifecycle_manifest_evidence(lifecycle),
            "fold_record": dict(fold_record),
            "shared_init_hash": shared_init_hash,
            "training_seed": training_seed,
            "mask_rate": mask_rate,
        })
        end_time = time.time()
        print (f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>')


    print (f'====== Saving =======')
    save_root = args.output_dir or config.MODEL_DIR
    if not os.path.exists(save_root): os.makedirs(save_root)
    ## gain suffix_name
    mask_rate = args.mask_type.split('-')[-1]
    suffix_name = f'{args.dataset.lower()}_Graph{args.base_model}_mask:{mask_rate}'
    ## gain feature_name and cls_name
    feature_name = f'{audio_feature};{text_feature};{video_feature}'
    cls_name = f'lossrecon:{args.loss_recon}+lower:{args.lower_bound}+reccls:{args.reccls_flag}'
    ## gain res_name
    mean_f1 = np.mean(np.array(folder_f1))
    mean_acc = np.mean(np.array(folder_acc))
    mean_recon = np.mean(np.array(folder_recon))
    res_name = f'f1:{mean_f1:2.2%}_acc:{mean_acc:2.2%}_reconloss:{mean_recon:.4f}'

    save_path = f'{save_root}/{suffix_name}_features:{feature_name}_classifier:{cls_name}_{res_name}_{time.time()}.npz'
    print (f'save results in {save_path}')
    np.savez_compressed(save_path,
                        args=np.array(args, dtype=object),
                        folder_losswhole=np.array(folder_losswhole, dtype=object),
                        folder_savewhole=np.array(folder_savewhole, dtype=object)
                        )
    metrics_path = Path(save_root) / "fold_metrics.json"
    metrics_path.write_text(json.dumps(fold_records, indent=2), encoding="utf-8")
    print(f'save fold metrics in {metrics_path}')
    run_id = str(int(time.time() * 1000000))
    run_record_root = Path(save_root) / "run_records" / run_id
    run_record_root.mkdir(parents=True, exist_ok=False)
    immutable_metrics_path = run_record_root / "fold_metrics.json"
    immutable_metrics_path.write_text(
        json.dumps(fold_records, indent=2), encoding="utf-8"
    )
    for context in fold_manifest_contexts:
        manifest = build_fold_run_manifest(
            args=args,
            fold=context["fold"],
            loader_metadata=context["loader_metadata"],
            lifecycle_evidence=context["lifecycle_evidence"],
            fold_record=context["fold_record"],
            feature_evidence=feature_evidence,
            environment=run_environment,
            provenance=run_provenance,
            shared_init_hash=context["shared_init_hash"],
            training_seed=context["training_seed"],
            mask_rate=context["mask_rate"],
            output_paths={
                "result_archive": str(Path(save_path).resolve()),
                "fold_metrics": str(immutable_metrics_path.resolve()),
                "archive_fold_index": int(context["archive_fold_index"]),
            },
        )
        manifest_path = run_record_root / "run_manifest_fold_{}.json".format(
            context["fold"]
        )
        write_manifest_atomic(manifest_path, manifest)
        print('save run manifest in {}'.format(manifest_path))
    if args.epoch_collapse_diagnostics:
        epoch_diagnostics_path = Path(save_root) / "epoch_collapse_diagnostics.json"
        epoch_diagnostics_path.write_text(
            json.dumps(epoch_collapse_records, indent=2), encoding="utf-8"
        )
        print(f'save epoch collapse diagnostics in {epoch_diagnostics_path}')
