"""CPU-only raw-feature audit and identity-aligned selected-prediction analysis.

Never samples masks, loads model weights, trains, or changes source features.
Only load the project's trusted pickle. Output contains statistics, not features.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


DATA_SHA = "a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80"
CLASSES = ["Neutral", "Surprise", "Fear", "Sadness", "Joy", "Disgust", "Anger"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def counts(flags):
    return {"N": len(flags), "zero_N": int(np.sum(flags)),
            "zero_pct": float(100 * np.mean(flags)) if len(flags) else None}


def align_zero_flags(saved, lookup):
    """Fail closed on identity, label, speaker, cohort, or duplicate mismatch."""
    keys = [(str(c), int(t)) for c, t in zip(saved["conversation_ids"], saved["utterance_indices"])]
    if len(keys) != len(set(keys)) or set(keys) != set(lookup):
        raise ValueError("selected prediction identities do not match complete raw test split")
    for i, key in enumerate(keys):
        raw = lookup[key]
        if raw["label"] != int(saved["labels"][i]) or raw["speaker"] != int(saved["speakers"][i]):
            raise ValueError(f"label/speaker mismatch at {key}")
    return np.array([lookup[key]["visual_zero"] for key in keys])


def visual_groups(zero, availability):
    a = np.asarray(availability)
    if a.shape != (len(zero), 3) or not np.isin(a, [0, 1]).all() or (a.sum(1) == 0).any():
        raise ValueError("invalid saved availability")
    observed = a[:, 2].astype(bool)
    return {"zero_V_observed": zero & observed,
            "nonzero_V_observed": ~zero & observed,
            "V_missing": ~observed}


def metrics(labels, logits):
    n = len(labels)
    if logits.shape != (n, 7) or not np.isfinite(logits).all():
        raise ValueError("invalid logits")
    if n == 0:
        return dict(N=0, ACC=None, UA_7_zero_filled=None, UA_supported=None, WF1=None,
                    CE=None, support=[0] * 7, recall=[None] * 7)
    pred = logits.argmax(1)
    matrix = np.zeros((7, 7), dtype=np.int64)
    np.add.at(matrix, (labels, pred), 1)
    support, predicted = matrix.sum(1), matrix.sum(0)
    correct = matrix.diagonal()
    recall = np.divide(correct, support, out=np.zeros(7), where=support > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(7), where=support + predicted > 0)
    stable = logits.astype(np.float64) - logits.max(1, keepdims=True)
    ce = np.log(np.exp(stable).sum(1)) - stable[np.arange(n), labels]
    return dict(N=n, ACC=100 * float(correct.sum() / n),
                UA_7_zero_filled=100 * float(recall.mean()),
                UA_supported=100 * float(recall[support > 0].mean()),
                WF1=100 * float((f1 * support).sum() / n), CE=float(ce.mean()),
                support=support.tolist(),
                recall=[float(100 * r) if s else None for r, s in zip(recall, support)])


def raw_audit(path):
    if sha(path) != DATA_SHA:
        raise ValueError("not the accepted ECERC feature pickle")
    with Path(path).open("rb") as stream:
        values = pickle.load(stream, encoding="latin1")
    if len(values) != 17:
        raise ValueError("expected 17-field ECERC pickle")
    speakers, labels, _sentiment, text, *_middle, audio, visual, _sentences, train, test, dev = values
    split_ids = {"train": train, "dev": dev, "test": test}
    result, lookup = {}, {}
    if sum(map(len, split_ids.values())) != len(set(train) | set(test) | set(dev)):
        raise ValueError("split overlap")
    for split, ids in split_ids.items():
        by_class, by_speaker, by_idbin, by_position = {}, {}, {}, {}
        convs, all_flags, runs, dtypes, dimensions = [], [], Counter(), set(), set()
        finite = True
        float32_changed = 0
        other_zeros = Counter()
        for cid in ids:
            v = np.asarray(visual[cid])
            a, l = np.asarray(audio[cid]), np.asarray(text[cid])
            y, sp = np.asarray(labels[cid]), np.asarray(speakers[cid])
            if v.ndim != 2 or len(v) != len(y) or sp.shape != (len(y), 9):
                raise ValueError(f"bad shape at {cid}")
            zero = np.all(v == 0, axis=1)
            finite = finite and all(np.isfinite(x).all() for x in (v, a, l))
            dtypes.add(str(v.dtype)); dimensions.add(v.shape[1])
            float32_changed += int(np.sum(zero != np.all(v.astype(np.float32) == 0, axis=1)))
            other_zeros["audio"] += int(np.all(a == 0, axis=1).sum())
            other_zeros["text"] += int(np.all(l == 0, axis=1).sum())
            convs.append(dict(conversation_id=str(cid), **counts(zero)))
            run = 0
            for t, flag in enumerate(zero):
                speaker = int(sp[t].argmax())
                for table, key in ((by_class, str(int(y[t]))), (by_speaker, str(speaker)),
                                   (by_idbin, f"{int(cid) // 100 * 100:04d}-{int(cid) // 100 * 100 + 99:04d}"),
                                   (by_position, str(min(t // 5 * 5, 30)))):
                    table.setdefault(key, []).append(bool(flag))
                if split == "test":
                    lookup[(str(cid), t)] = dict(label=int(y[t]), speaker=speaker, visual_zero=bool(flag))
                if flag:
                    run += 1
                elif run:
                    runs[run] += 1; run = 0
            if run:
                runs[run] += 1
            all_flags.extend(zero.tolist())
        result[split] = dict(**counts(all_flags), dialogue_N=len(ids),
                             all_features_finite=bool(finite), visual_dtypes=sorted(dtypes),
                             visual_dimensions=sorted(dimensions), float32_zero_flag_changes=float32_changed,
                             other_zero_N=dict(other_zeros),
                             by_class={k: counts(x) for k, x in sorted(by_class.items())},
                             by_speaker_slot={k: counts(x) for k, x in sorted(by_speaker.items())},
                             by_global_id_bin={k: counts(x) for k, x in sorted(by_idbin.items())},
                             by_utterance_position_5bin={k: counts(x) for k, x in sorted(by_position.items())},
                             zero_run_length_histogram=dict(sorted(runs.items())),
                             dialogues_all_zero=sum(c["zero_N"] == c["N"] for c in convs),
                             dialogues_any_zero=sum(c["zero_N"] > 0 for c in convs),
                             dialogues=convs)
    return result, lookup


def prediction_audit(root, lookup):
    summary_path = root / "SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    if summary["config"]["missing_mask_protocol"] != "gcnet-random-mask-v1":
        raise ValueError("not the required GCNet random mask protocol")
    if summary["provenance"]["data_sha256"] != DATA_SHA:
        raise ValueError("prediction dataset mismatch")
    output = {}
    for rate, selected in summary["testbest"]["rates"].items():
        path = root / selected["path"]
        if sha(path) != selected["sha256"]:
            raise ValueError("prediction file hash mismatch")
        with np.load(path, allow_pickle=False) as f:
            saved = dict(f)
        zero = align_zero_flags(saved, lookup)
        a, y, logits = saved["availability"], saved["labels"], saved["logits"]
        if not np.array_equal(logits.argmax(1), saved["predictions"]):
            raise ValueError("saved prediction/logit mismatch")
        groups = visual_groups(zero, a)
        if int(saved["selected_epoch"]) != selected["epoch"]:
            raise ValueError("selected epoch mismatch")
        if hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() != selected["availability_sha256"]:
            raise ValueError("availability hash mismatch")
        output[rate] = dict(selected_epoch=selected["epoch"], input_path=str(path),
                            input_sha256=sha(path), availability_sha256=selected["availability_sha256"],
                            overall=metrics(y, logits),
                            groups={k: metrics(y[m], logits[m]) for k, m in groups.items()},
                            group_availability_counts={k: dict(Counter("".join(map(str, row.astype(int)))
                                                                       for row in a[m]))
                                                       for k, m in groups.items()})
        if sum(x["N"] for x in output[rate]["groups"].values()) != len(y):
            raise ValueError("groups do not partition complete test split")
        for field, selected_field in (("ACC", "ACC"), ("UA_7_zero_filled", "UA"), ("WF1", "WF1")):
            if not np.isclose(output[rate]["overall"][field], selected[selected_field], atol=1e-8, rtol=0):
                raise ValueError(f"overall selected metric mismatch: {field}")
    return dict(input_summary=str(summary_path), input_summary_sha256=sha(summary_path),
                numerical_commit=summary["provenance"]["numerical_commit"], rates=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw, lookup = raw_audit(args.pickle)
    result = dict(label="INTERNAL DIAGNOSTIC ONLY", training=False, new_inference=False,
                  dataset_path=str(args.pickle), dataset_sha256=DATA_SHA, class_order=CLASSES,
                  raw=raw, selected_full_prediction=prediction_audit(args.predictions, lookup),
                  limitations=["No Local/Base/Gap gain inferred from Full-only predictions.",
                               "No raw video failure cause inferred from zero features.",
                               "Subgroup class/availability mixtures differ; no causal attribution.",
                               "Speaker slot index is not asserted to be global character identity."])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("N", "zero_N", "zero_pct", "dialogues_all_zero")}
                      for k, v in raw.items()}))


if __name__ == "__main__":
    main()
