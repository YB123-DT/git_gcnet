"""Replace one observed historical modality slice; selection never sees labels."""

from __future__ import annotations

import numpy as np
import torch

from experiments.osram_history_effect_stratification_20261013.evaluate import per_utterance_effects
from experiments.osram_meld_flat_source_combinations_20261013.evaluate import FlatReadoutCapture

MODALITIES = ("A", "T", "V")
ARMS = ("same_speaker", "other_speaker")
ROW_FIELDS = (
    "conversation_id", "position", "modality", "arm", "source_position", "donor_position",
    "source_lag", "donor_lag", "lag_mismatch", "target_speaker", "source_speaker", "donor_speaker",
    "source_same_speaker", "donor_same_speaker", "label", "logits_original", "logits_replaced",
    "ce_original", "ce_replaced", "delta_ce", "true_probability_original", "true_probability_replaced",
    "true_probability_delta", "correction", "harm", "current_local_exact", "current_local_within_tolerance",
    "current_local_max_abs_error", "current_availability_exact", "availability_trajectory_exact", "current_input_exact",
)


def select_replacements(availability, speakers, target):
    """Anchor latest observed same-target-speaker source; match distinct donor lag.

    Both donors are strictly before target. Same-speaker donors share target's
    speaker, other-speaker donors differ. Minimize absolute source/donor lag
    mismatch, breaking ties by most recent position. Missing arms are reported,
    rather than changing the anchor to gain coverage.
    """
    mask, speaker = np.asarray(availability), np.asarray(speakers)
    if mask.ndim != 2 or mask.shape[1] != 3 or not np.isin(mask, (0, 1)).all():
        raise ValueError("expected binary availability [time,3]")
    if speaker.shape != (len(mask),) or not np.isfinite(speaker).all() or not np.equal(speaker, np.floor(speaker)).all():
        raise ValueError("expected finite integer speaker IDs [time]")
    if not 0 <= target < len(mask):
        raise ValueError("target outside conversation")
    plans, skipped = [], []
    for modality in range(3):
        observed = np.flatnonzero(mask[:target, modality]).tolist()
        same = [position for position in observed if speaker[position] == speaker[target]]
        source = same[-1] if same else None
        for arm in ARMS:
            if source is None:
                skipped.append(dict(modality=MODALITIES[modality], arm=arm, reason="no_same_speaker_source"))
                continue
            donors = [position for position in observed if position != source and
                      ((speaker[position] == speaker[target]) == (arm == "same_speaker"))]
            if not donors:
                skipped.append(dict(modality=MODALITIES[modality], arm=arm, reason="no_observed_donor"))
                continue
            donor = min(donors, key=lambda position: (abs(position - source), -position))
            plans.append(dict(target=target, modality=MODALITIES[modality], modality_index=modality,
                              arm=arm, source_position=source, donor_position=donor,
                              source_lag=target-source, donor_lag=target-donor,
                              lag_mismatch=abs(source-donor), target_speaker=int(speaker[target]),
                              source_speaker=int(speaker[source]), donor_speaker=int(speaker[donor]),
                              source_same_speaker=True, donor_same_speaker=arm == "same_speaker"))
    return plans, skipped


def replacement_batch(view, jobs, dimensions):
    """Duplicate conversations and edit exactly one selected modality slice."""
    if len(dimensions) != 3 or any(int(dimension) <= 0 for dimension in dimensions):
        raise ValueError("three positive modality dimensions required")
    if sum(dimensions) != view["incomplete"].shape[-1]:
        raise ValueError("feature width does not match modality dimensions")
    indices = torch.tensor([job["batch_index"] for job in jobs], device=view["incomplete"].device)
    changed = dict(view)
    changed["incomplete"] = view["incomplete"].index_select(1, indices).clone()
    changed["availability"] = view["availability"].index_select(1, indices).clone()
    for name in ("qmask", "umask", "labels"):
        changed[name] = view[name].index_select(0, indices).clone()
    changed["lengths"] = [int(view["lengths"][job["batch_index"]]) for job in jobs]
    changed["conversation_ids"] = [view["conversation_ids"][job["batch_index"]] for job in jobs]
    offsets = np.cumsum((0, *dimensions))
    for index, job in enumerate(jobs):
        source, donor, modality = job["source_position"], job["donor_position"], job["modality_index"]
        if not (0 <= source < job["target"] < changed["lengths"][index] and
                0 <= donor < job["target"] and source != donor and 0 <= modality < 3):
            raise ValueError("source/donor must be distinct strict-past valid positions")
        if changed["availability"][source, index, modality] != 1 or changed["availability"][donor, index, modality] != 1:
            raise ValueError("source and donor modality must both be observed")
        start, end = offsets[modality:modality+2]
        changed["incomplete"][source, index, start:end] = view["incomplete"][donor, job["batch_index"], start:end]
        if not torch.equal(changed["incomplete"][job["target"]:, index], view["incomplete"][job["target"]:, job["batch_index"]]):
            raise AssertionError("current/future inputs changed")
    return changed


def paired_effects(labels, original, replaced):
    values = per_utterance_effects(np.asarray(labels), {"L": original, "LB": replaced, "LBG": replaced})
    return {
        "ce_original": values["ce_L"], "ce_replaced": values["ce_LB"], "delta_ce": -values["Delta_Base"],
        "true_probability_original": values["true_probability_L"], "true_probability_replaced": values["true_probability_LB"],
        "true_probability_delta": values["probability_change_Base"],
        "correction": values["correction_Base"], "harm": values["harm_Base"],
    }


def evaluate_view(model, view, dimensions, *, chunk_size=16, local_rtol=2e-5, local_atol=2e-5):
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("P1 requires eval mode and frozen parameters")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    plans, skipped = [], []
    mask = view["availability"].detach().cpu().numpy()
    speakers = view["qmask"].detach().cpu().numpy()
    for batch, length in enumerate(view["lengths"]):
        for target in range(int(length)):
            selected, absent = select_replacements(mask[:int(length), batch], speakers[batch, :int(length)], target)
            plans.extend(dict(plan, batch_index=batch) for plan in selected)
            skipped.extend(dict(row, conversation_id=str(view["conversation_ids"][batch]), position=target) for row in absent)
    requested = {}
    for plan in plans:
        key = (plan["batch_index"], plan["source_position"], plan["donor_position"], plan["modality_index"])
        requested.setdefault(key, []).append(plan)
    jobs = [values[0] for values in requested.values()]
    rows = []
    checks = {"original_scans": 1, "replacement_scans": 0, "unique_replacements": len(jobs),
              "paired_rows": len(plans), "avoided_repeated_replacements": len(plans)-len(jobs)}
    with torch.no_grad():
        with FlatReadoutCapture(model.osram) as baseline_capture:
            original, _, _, missing = model([view["incomplete"]], view["availability"], view["qmask"], view["umask"], view["lengths"], predict_missing=False)
        if missing is not None or baseline_capture.local_calls != 1 or baseline_capture.adapter_calls != 1:
            raise RuntimeError("expected one unchanged Flat readout")
        for start in range(0, len(jobs), chunk_size):
            chunk = jobs[start:start+chunk_size]
            changed = replacement_batch(view, chunk, dimensions)
            with FlatReadoutCapture(model.osram) as capture:
                replaced, _, _, missing = model([changed["incomplete"]], changed["availability"], changed["qmask"], changed["umask"], changed["lengths"], predict_missing=False)
            if missing is not None or capture.local_calls != 1 or capture.adapter_calls != 1:
                raise RuntimeError("expected one replaced Flat readout")
            checks["replacement_scans"] += 1
            for index, job in enumerate(chunk):
                batch = job["batch_index"]
                key = (batch, job["source_position"], job["donor_position"], job["modality_index"])
                trajectory_exact = torch.equal(changed["availability"][:, index], view["availability"][:, batch])
                if not trajectory_exact:
                    raise AssertionError("availability trajectory changed")
                for plan in requested[key]:
                    target = plan["target"]
                    before, after = baseline_capture.local[target, batch], capture.local[target, index]
                    torch.testing.assert_close(after, before, rtol=local_rtol, atol=local_atol)
                    row = {name: value for name, value in plan.items() if name not in {"batch_index", "target", "modality_index"}}
                    row.update(conversation_id=str(view["conversation_ids"][batch]), position=target,
                               label=int(view["labels"][batch, target]), logits_original=original[target, batch].cpu().numpy(),
                               logits_replaced=replaced[target, index].cpu().numpy(), current_local_exact=torch.equal(before, after),
                               current_local_within_tolerance=True, current_local_max_abs_error=float((before-after).abs().max()),
                               current_availability_exact=torch.equal(changed["availability"][target, index], view["availability"][target, batch]),
                               availability_trajectory_exact=trajectory_exact,
                               current_input_exact=torch.equal(changed["incomplete"][target, index], view["incomplete"][target, batch]))
                    effects = paired_effects(np.asarray([row["label"]]), row["logits_original"][None], row["logits_replaced"][None])
                    row.update({name: values[0] for name, values in effects.items()})
                    rows.append(row)
    return rows, skipped, checks, original.transpose(0, 1)[view["umask"].bool()].cpu().numpy()


def summarize_rows(rows, skipped):
    result = {}
    for modality in MODALITIES:
        result[modality] = {}
        for arm in ARMS:
            selected = [row for row in rows if row["modality"] == modality and row["arm"] == arm]
            absent = [row for row in skipped if row["modality"] == modality and row["arm"] == arm]
            reasons = {reason: sum(row["reason"] == reason for row in absent) for reason in sorted({row["reason"] for row in absent})}
            count, total = len(selected), len(selected)+len(absent)
            item = dict(count=count, targets=total, coverage=count/total if total else None, skipped=len(absent), skipped_reasons=reasons)
            for name in ("delta_ce", "true_probability_delta", "correction", "harm", "lag_mismatch", "current_local_max_abs_error", "current_local_exact"):
                item[f"mean_{name}"] = float(np.mean([row[name] for row in selected])) if selected else None
            item["corrections"] = sum(bool(row["correction"]) for row in selected)
            item["harms"] = sum(bool(row["harm"]) for row in selected)
            result[modality][arm] = item
    return result
