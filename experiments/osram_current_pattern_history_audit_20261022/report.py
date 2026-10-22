"""Offline summaries of frozen, same-history current-pattern interventions.

No model loading, training, inference, or third-party statistics dependencies.
The input streams must be closed and explicitly marked collection_complete.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import mean, median


PATTERNS = ("A", "T", "V", "AT", "AV", "TV", "ATV")
GROUPS = ("Writer", "WriteKey", "WriteValue")
CONDITIONS = ("correct", "empty", "replaced")


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _distribution(values):
    present = [v for v in values if v is not None]
    return {
        "defined_count": len(present), "undefined_count": len(values) - len(present),
        "mean": mean(present) if present else None,
        "median": median(present) if present else None,
        "negative_fraction": sum(v < 0 for v in present) / len(present) if present else None,
    }


def _geometry(rows):
    return {
        "pair_count": len(rows),
        "loss_cosine": _distribution([r["loss_cosine"] for r in rows]),
        "jacobian_cosine": _distribution([r["jacobian_cosine"] for r in rows]),
        "residual_sign_opposed_fraction": mean(r["residual_sign_opposed"] for r in rows) if rows else None,
    }


def _gradient_report(rows):
    seen, targets, patterns = set(), {}, {}
    for row in rows:
        if row["split"] != "train" or row["group"] not in GROUPS:
            raise ValueError("gradient rows must be train split and a known Writer group")
        a, b = row["pattern_a"], row["pattern_b"]
        if a not in PATTERNS or b not in PATTERNS or a == b:
            raise ValueError("gradient rows require two distinct valid current patterns")
        convo = row["conversation_id"]
        if convo in targets and targets[convo] != row["target_t"]:
            raise ValueError("gradient collection must use at most one target per conversation")
        targets[convo] = row["target_t"]
        key = (convo, row["target_t"], row["group"], frozenset((a, b)))
        if key in seen:
            raise ValueError("duplicate gradient pair")
        seen.add(key)
        for field in ("label", "residual_a", "residual_b", "loss_dot", "grad_norm_a", "grad_norm_b"):
            _finite(row[field], field)
        if row["grad_norm_a"] < 0 or row["grad_norm_b"] < 0:
            raise ValueError("gradient norms cannot be negative")
        for field in ("loss_cosine", "jacobian_cosine"):
            if row[field] is not None:
                value = _finite(row[field], field)
                if abs(value) > 1 + 1e-5:
                    raise ValueError(f"invalid {field}")
        if (row["grad_norm_a"] == 0 or row["grad_norm_b"] == 0) and row["loss_cosine"] is not None:
            raise ValueError("zero-norm loss gradient cosine must be null, not zero")
        expected_opposed = row["residual_a"] * row["residual_b"] < 0
        if bool(row["residual_sign_opposed"]) != expected_opposed:
            raise ValueError("residual_sign_opposed is inconsistent")
        for pattern, residual in ((a, row["residual_a"]), (b, row["residual_b"])):
            pkey = (convo, row["target_t"], pattern)
            if pkey in patterns and patterns[pkey] != residual:
                raise ValueError("repeated pattern residual differs within the same anchor")
            patterns[pkey] = residual
    groups = {group: _geometry([r for r in rows if r["group"] == group]) for group in GROUPS}
    by_pair = {}
    matrices = {field: {a: {b: None for b in PATTERNS} for a in PATTERNS}
                for field in ("loss_cosine", "jacobian_cosine")}
    csv_rows = []
    for group in GROUPS:
        for index, a in enumerate(PATTERNS):
            for b in PATTERNS[index + 1:]:
                matching = [r for r in rows if r["group"] == group and {r["pattern_a"], r["pattern_b"]} == {a, b}]
                stats = _geometry(matching)
                by_pair[f"{group}:{a}:{b}"] = stats
                record = {"group": group, "pattern_a": a, "pattern_b": b, "pair_count": len(matching),
                          "residual_sign_opposed_fraction": stats["residual_sign_opposed_fraction"]}
                for field in ("loss_cosine", "jacobian_cosine"):
                    for name, value in stats[field].items():
                        record[f"{field}_{name}"] = value
                    if group == "Writer":
                        matrices[field][a][b] = matrices[field][b][a] = stats[field]["mean"]
                csv_rows.append(record)
    return {
        "row_count": len(rows), "anchor_count": len(targets), "groups": groups,
        "pattern_pairs": by_pair, "writer_matrices": matrices,
        "writer_by_residual_sign": {
            "opposed": _geometry([r for r in rows if r["group"] == "Writer" and r["residual_sign_opposed"]]),
            "not_opposed": _geometry([r for r in rows if r["group"] == "Writer" and not r["residual_sign_opposed"]]),
        },
        "pattern_train_MSE": {p: mean(v * v for key, v in patterns.items() if key[-1] == p)
                              if any(key[-1] == p for key in patterns) else None for p in PATTERNS},
    }, csv_rows


def _prediction_metrics(rows, condition):
    available = [r for r in rows if r[f"pred_{condition}"] is not None]
    polarity = [r for r in available if r["label"] != 0]
    if polarity:
        true = [r["label"] > 0 for r in polarity]
        predicted = [r[f"pred_{condition}"] > 0 for r in polarity]
        accuracy = 100 * sum(a == b for a, b in zip(true, predicted)) / len(true)
        weighted_f1 = 0.0
        for cls in (False, True):
            support = sum(v == cls for v in true)
            tp = sum(y == cls and p == cls for y, p in zip(true, predicted))
            fp = sum(y != cls and p == cls for y, p in zip(true, predicted))
            fn = sum(y == cls and p != cls for y, p in zip(true, predicted))
            denominator = 2 * tp + fp + fn
            weighted_f1 += support * (2 * tp / denominator if denominator else 0)
        weighted_f1 *= 100 / len(true)
    else:
        accuracy = weighted_f1 = None
    return {
        "N": len(available), "polarity_N": len(polarity), "unavailable": len(rows) - len(available),
        "MSE": mean((r["label"] - r[f"pred_{condition}"]) ** 2 for r in available) if available else None,
        "ACC": accuracy, "W_F1": weighted_f1,
    }


def _paired(rows, control):
    available = [r for r in rows if r[f"pred_{control}"] is not None]
    polarity = [r for r in available if r["label"] != 0]
    correction = harm = 0
    for row in polarity:
        real_correct = (row["pred_correct"] > 0) == (row["label"] > 0)
        control_correct = (row[f"pred_{control}"] > 0) == (row["label"] > 0)
        correction += real_correct and not control_correct
        harm += control_correct and not real_correct
    real, other = _prediction_metrics(available, "correct"), _prediction_metrics(available, control)
    return {
        "N": len(available), "polarity_N": len(polarity), "corrections": correction, "harms": harm,
        "delta_MSE": other["MSE"] - real["MSE"] if available else None,
        "delta_ACC_pp": real["ACC"] - other["ACC"] if polarity else None,
        "delta_W_F1_pp": real["W_F1"] - other["W_F1"] if polarity else None,
        "definition": "positive delta_MSE/control minus real; positive classification delta/real minus control",
    }


def _history_report(rows):
    seen = set()
    for row in rows:
        if row["split"] != "test" or row["pattern"] not in PATTERNS:
            raise ValueError("history rows must be test split with a valid pattern")
        key = (row["conversation_id"], row["target_t"], row["pattern"])
        if key in seen:
            raise ValueError("duplicate history observation")
        seen.add(key)
        length = row["history_length"]
        if not isinstance(length, int) or isinstance(length, bool) or length < 0:
            raise ValueError("history_length must be a nonnegative integer")
        _finite(row["label"], "label")
        for condition in CONDITIONS:
            if row[f"pred_{condition}"] is not None:
                _finite(row[f"pred_{condition}"], f"pred_{condition}")
            elif condition != "replaced":
                raise ValueError("real and empty predictions must be available")
        if (row["donor_id"] is None) != (row["pred_replaced"] is None):
            raise ValueError("replacement availability must match donor availability")
        if row["donor_id"] == row["conversation_id"]:
            raise ValueError("donor must come from a different conversation")
        if length == 0:
            for condition in ("empty", "replaced"):
                value = row[f"pred_{condition}"]
                if value is not None and not math.isclose(row["pred_correct"], value, rel_tol=0, abs_tol=1e-6):
                    raise ValueError("no-history predictions must agree")
    report, csv_rows = {}, []
    for subset in ("all", "history_present"):
        filtered = rows if subset == "all" else [r for r in rows if r["history_length"] > 0]
        report[subset] = {}
        for pattern in PATTERNS:
            selected = [r for r in filtered if r["pattern"] == pattern]
            metrics = {condition: _prediction_metrics(selected, condition) for condition in CONDITIONS}
            metrics["paired"] = {control: _paired(selected, control) for control in ("empty", "replaced")}
            report[subset][pattern] = metrics
            for condition in CONDITIONS:
                record = {"subset": subset, "pattern": pattern, "condition": condition, **metrics[condition]}
                if condition != "correct":
                    record.update({key: value for key, value in metrics["paired"][condition].items()
                                   if key not in ("N", "polarity_N", "definition")})
                csv_rows.append(record)
    report["row_count"] = len(rows)
    report["target_count"] = len({(r["conversation_id"], r["target_t"]) for r in rows})
    report["no_history_row_count"] = sum(r["history_length"] == 0 for r in rows)
    return report, csv_rows


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def _markdown(report):
    lines = ["# Same-history current-pattern and history-dependence audit", "", "INTERNAL DIAGNOSTIC ONLY", "",
             "Frozen MOSI seed66 CHM; no training or optimizer updates. One original checkpoint, not per-rate BEST scores.",
             "The source checkpoint was originally selected with Test-oracle; these are exploratory model-behavior results.", "",
             "## Experiment 1: historical-write gradient geometry", "",
             "| Group | Pairs | Defined loss cosines | Mean loss cosine | Negative fraction | Mean prediction-Jacobian cosine |", 
             "|---|---:|---:|---:|---:|---:|"]
    for group, stats in report["gradient"]["groups"].items():
        loss, jacobian = stats["loss_cosine"], stats["jacobian_cosine"]
        lines.append(f"| {group} | {stats['pair_count']} | {loss['defined_count']} | {_number(loss['mean'])} | {_number(loss['negative_fraction'])} | {_number(jacobian['mean'])} |")
    lines += ["", "For scalar MSE, grad(loss)=2*(prediction-label)*grad(prediction). Opposite residual signs can invert loss-gradient cosine even with aligned prediction Jacobians. Zero-norm cosine is undefined, not zero.",
              "Negative cosine alone does not establish a CHM defect or harmful optimization. These are paired hypothetical current-pattern losses on the same frozen history, not seven independent optimizer updates; ordinary BPTT already reaches historical writes.", "",
              "Only historical K/V graph nodes are credited here; parameter-sharing paths through current observations are excluded. The collection implementation and STATUS provenance determine the precise graph intervention.", ""]
    for field in ("loss_cosine", "jacobian_cosine"):
        lines += [f"### Writer {field}", "", "| Pattern | " + " | ".join(PATTERNS) + " |",
                  "|---|" + "---:|" * len(PATTERNS)]
        for pattern, values in report["gradient"]["writer_matrices"][field].items():
            lines.append("| " + pattern + " | " + " | ".join(_number(values[p]) for p in PATTERNS) + " |")
        lines.append("")
    lines += ["## Experiment 2: history dependence", "",
              "MSE includes neutral labels; polarity ACC/W-F1 exclude y=0 and use prediction>0. All conditions keep current input fixed. Replacements without a valid donor are unavailable, not zero-valued predictions.", ""]
    for subset in ("all", "history_present"):
        lines += [f"### {subset}", "",
                  "| Pattern | History | N | Polarity N | ACC (%) | W-F1 (%) | MSE |", "|---|---|---:|---:|---:|---:|---:|"]
        for pattern in PATTERNS:
            for condition in CONDITIONS:
                m = report["history"][subset][pattern][condition]
                lines.append(f"| {pattern} | {condition} | {m['N']} | {m['polarity_N']} | {_number(m['ACC'], 3)} | {_number(m['W_F1'], 3)} | {_number(m['MSE'])} |")
        lines += ["", "| Pattern | Control → Real history | Paired N | ΔMSE (control−real) | ΔACC (pp) | ΔW-F1 (pp) | Corrections | Harms |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for pattern in PATTERNS:
            for control in ("empty", "replaced"):
                p = report["history"][subset][pattern]["paired"][control]
                lines.append(f"| {pattern} | {control} → correct | {p['N']} | {_number(p['delta_MSE'])} | {_number(p['delta_ACC_pp'], 3)} | {_number(p['delta_W_F1_pp'], 3)} | {p['corrections']} | {p['harms']} |")
        lines.append("")
    lines += ["Correct means original history, not a guarantee of a correct prediction. Positive ΔMSE means real history lowers MSE. Corrections/harms use the same paired nonneutral samples, not independently evaluated supports.",
              "Empty/replaced states can be out of distribution. This test does not identify pure historical semantics or establish a natural-language causal effect. No new method, gain, or deployment result is established by running the audit.", ""]
    return "\n".join(lines)


def generate(output):
    """Validate completed streams and write small, reproducible offline summaries."""
    output = Path(output)
    status = json.loads((output / "STATUS.json").read_text())
    if status.get("collection_complete") is not True:
        raise ValueError("STATUS.collection_complete must be true before report generation")
    gradients = _rows(output / "GRADIENT_PAIRS.jsonl.gz")
    history = _rows(output / "HISTORY_ROWS.jsonl.gz")
    gradient_report, gradient_csv = _gradient_report(gradients)
    history_report, history_csv = _history_report(history)
    report = {
        "label": "INTERNAL DIAGNOSTIC ONLY", "collection_complete": True,
        "protocol": status.get("protocol", {}),
        "gradient": gradient_report, "history": history_report,
        "source_records": ["STATUS.json", "GRADIENT_PAIRS.jsonl.gz", "HISTORY_ROWS.jsonl.gz"],
    }
    (output / "REPORT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _csv(output / "GRADIENT_MATRIX.csv", gradient_csv)
    _csv(output / "HISTORY_METRICS.csv", history_csv)
    (output / "RESULT.md").write_text(_markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
