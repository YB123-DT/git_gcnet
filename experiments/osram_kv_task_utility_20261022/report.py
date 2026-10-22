"""Descriptive association of joint-write KV improvement and future task utility.

Pair slots repeat sources/targets/rates. They are not independent observations;
bootstrap resampling therefore keeps complete conversations together.
"""
import argparse
from bisect import bisect_left
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import random
import statistics


UTILITY_TOLERANCE = 1e-7
NUMERIC_FIELDS = ("q_gain", "e_before", "e_after", "value_energy",
                  "write_update_norm", "label", "prediction_keep", "prediction_omit",
                  "loss_keep", "loss_omit", "utility")
LIMITATIONS = [
    "INTERNAL DIAGNOSTIC ONLY: one frozen, previously Test-oracle-selected checkpoint; no independent generalization claim.",
    "No new training, no parameter selection, and no change to GCNet Missing or original write equations.",
    "Utility measures a joint coupled write intervention plus all subsequent state evolution; it is not semantic source attribution.",
    "Repeated writes, targets, rates and distances are not independent samples; no iid p-values are reported.",
    "KV reconstruction quality is an internal association objective, not a direct measure of emotion relevance.",
    "Correlation does not establish mechanism or causality beyond the specified frozen-model write intervention.",
    "Median/quartile boundaries describe the observed data; they are not deployable thresholds or optimized settings.",
    "Within-quartile correlations are descriptive range-restricted associations, not independent confirmatory tests.",
    "MSE utility includes all valid labeled targets, including neutral labels; label=0 is excluded only from polarity counts.",
    "Conversation-cluster bootstrap intervals describe this sampled audit, not checkpoint/seed uncertainty.",
]


def _ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        for index in ordered[start:stop]:
            ranks[index] = (start + stop - 1) / 2
        start = stop
    return ranks


def spearman(x, y):
    """Tie-aware rank correlation, undefined for n<2 or constant arguments."""
    if len(x) != len(y):
        raise ValueError("Spearman inputs must have the same length")
    if len(x) < 2:
        return None
    a, b = _ranks(x), _ranks(y)
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = sum((v-ma)**2 for v in a)
    vb = sum((v-mb)**2 for v in b)
    if not va or not vb:
        return None
    return sum((v-ma)*(w-mb) for v, w in zip(a, b)) / math.sqrt(va*vb)


def _mean(values):
    return statistics.mean(values) if values else None


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    offset = fraction * (len(values)-1)
    low, high = math.floor(offset), math.ceil(offset)
    return values[low] + (offset-low)*(values[high]-values[low])


def _summary(rows, median_q):
    q = [r["q_gain"] for r in rows]
    utility = [r["utility"] for r in rows]
    by_conversation = defaultdict(list)
    quadrants = {f"{q_group}_{u_group}": 0 for q_group in ("q_low", "q_high")
                 for u_group in ("utility_positive", "utility_nonpositive")}
    for row in rows:
        by_conversation[row["conversation_id"]].append(row["utility"])
        q_group = "q_high" if row["q_gain"] > median_q else "q_low"
        u_group = "utility_positive" if row["utility"] > UTILITY_TOLERANCE else "utility_nonpositive"
        quadrants[f"{q_group}_{u_group}"] += 1
    return dict(
        N=len(rows), unique_conversations=len(by_conversation),
        unique_writes=len({(r["conversation_id"], r["rate"], r["source_t"]) for r in rows}),
        unique_targets=len({(r["conversation_id"], r["rate"], r["target_t"]) for r in rows}),
        mean_q_gain=_mean(q), mean_utility=_mean(utility),
        median_utility=statistics.median(utility) if utility else None,
        conversation_macro_mean_utility=_mean([statistics.mean(v) for v in by_conversation.values()]),
        spearman_q_utility=spearman(q, utility),
        mean_e_before=_mean([r["e_before"] for r in rows]),
        mean_e_after=_mean([r["e_after"] for r in rows]),
        mean_loss_keep=_mean([r["loss_keep"] for r in rows]),
        mean_loss_omit=_mean([r["loss_omit"] for r in rows]),
        improve_count=sum(u > UTILITY_TOLERANCE for u in utility),
        harm_count=sum(u < -UTILITY_TOLERANCE for u in utility),
        neutral_count=sum(abs(u) <= UTILITY_TOLERANCE for u in utility),
        polarity_eligible_count=sum(bool(r["polarity_eligible"]) for r in rows),
        corrections=sum(bool(r["correction"]) and bool(r["polarity_eligible"]) for r in rows),
        harms=sum(bool(r["harm"]) and bool(r["polarity_eligible"]) for r in rows),
        quadrants=quadrants,
    )


def _cluster_bootstrap(rows, replicates=200, seed=66):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["conversation_id"]].append(row)
    keys = sorted(clusters)
    result = dict(clusters=len(keys), replicates=replicates, seed=seed,
                  mean_utility_ci95=None, spearman_ci95=None,
                  valid_spearman_replicates=0)
    if len(keys) < 2:
        return result
    rng = random.Random(seed)
    means, correlations = [], []
    for _ in range(replicates):
        sampled = [r for key in rng.choices(keys, k=len(keys)) for r in clusters[key]]
        means.append(statistics.mean(r["utility"] for r in sampled))
        corr = spearman([r["q_gain"] for r in sampled], [r["utility"] for r in sampled])
        if corr is not None:
            correlations.append(corr)
    result["mean_utility_ci95"] = [_percentile(means, .025), _percentile(means, .975)]
    result["valid_spearman_replicates"] = len(correlations)
    if len(correlations) >= 2:
        result["spearman_ci95"] = [_percentile(correlations, .025), _percentile(correlations, .975)]
    return result


def build_report(rows):
    """Build a JSON-safe report from one record per joint write/target pair."""
    rows = list(rows)
    for row in rows:
        for name in NUMERIC_FIELDS:
            if not math.isfinite(float(row[name])):
                raise ValueError(f"{name} must be finite")
    median_q = statistics.median([r["q_gain"] for r in rows]) if rows else 0.
    def grouped(key):
        groups = defaultdict(list)
        for row in rows:
            groups[str(key(row))].append(row)
        return {name: _summary(values, median_q) for name, values in sorted(groups.items())}
    cuts = [_percentile([r["q_gain"] for r in rows], fraction) for fraction in (.25, .5, .75)]
    quartiles = {f"Q{i}": [] for i in range(1, 5)}
    for row in rows:
        quartiles[f"Q{bisect_left(cuts, row['q_gain'])+1}"].append(row)
    return dict(
        protocol="INTERNAL DIAGNOSTIC ONLY", utility_tolerance=UTILITY_TOLERANCE,
        utility_definition="loss_omit - loss_keep; positive means keeping the joint write helps",
        q_definition="normalized E_before - normalized E_after; positive means KV fit improves",
        overall=_summary(rows, median_q),
        by_rate=grouped(lambda r: r["rate"]),
        by_distance=grouped(lambda r: r["distance"]),
        by_target_pattern=grouped(lambda r: r["target_pattern"]),
        by_distance_target_pattern=grouped(lambda r: f"{r['distance']}|{r['target_pattern']}"),
        q_median=median_q if rows else None,
        q_median_split={
            "low_or_equal": _summary([r for r in rows if r["q_gain"] <= median_q], median_q),
            "high": _summary([r for r in rows if r["q_gain"] > median_q], median_q)},
        q_quartile_boundaries=cuts, q_quartiles={k: _summary(v, median_q) for k, v in quartiles.items()},
        cluster_bootstrap=_cluster_bootstrap(rows), limitations=LIMITATIONS,
    )


def write_report(output, rows):
    """Write report products only; never alter frozen predictions or model files."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    result = build_report(rows)
    if (output/"STATUS.json").is_file():
        result["run_status"] = json.loads((output/"STATUS.json").read_text())
    (output/"REPORT.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    table = []
    for group in ("overall", "by_rate", "by_distance", "by_target_pattern",
                  "by_distance_target_pattern", "q_median_split", "q_quartiles"):
        items = {"all": result[group]} if group == "overall" else result[group]
        for name, values in items.items():
            table.append(dict(group=group, condition=name,
                              **{k: v for k, v in values.items() if k != "quadrants"},
                              **values["quadrants"]))
    with (output/"AGGREGATE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    fmt = lambda x: "N/A" if x is None else f"{x:.6f}"
    lines = ["# Joint KV Fit and Future Task Utility", "", "INTERNAL DIAGNOSTIC ONLY", "",
             "No new training. Same protected frozen Flat checkpoint and saved GCNet masks.", "",
             "Q = normalized KV error before write minus after write; U = future MSE(omit) - MSE(keep).",
             "Omit preserves decay and skips only the selected joint write; future writes resume normally.", "",
             "| Group | Condition | Pair slots | Conversations | Mean Q | Mean U | Spearman | Helps | Harms | Neutral |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['group']} | {row['condition']} | {row['N']} | {row['unique_conversations']} | "
                     f"{fmt(row['mean_q_gain'])} | {fmt(row['mean_utility'])} | {fmt(row['spearman_q_utility'])} | "
                     f"{row['improve_count']} | {row['harm_count']} | {row['neutral_count']} |")
    lines += ["", "## Conversation-cluster uncertainty", "", json.dumps(result["cluster_bootstrap"]),
              "", "## Limits", ""] + ["- "+item for item in LIMITATIONS]
    (output/"RESULT.md").write_text("\n".join(lines)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    opener = gzip.open if args.rows.suffix == ".gz" else open
    with opener(args.rows, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    write_report(args.output, rows)


if __name__ == "__main__":
    main()
