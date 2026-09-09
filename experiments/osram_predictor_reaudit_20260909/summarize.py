"""Summarize per-seed metrics without pooling teacher spaces."""
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "results/per_seed.csv").open()))
assert len(rows) == 100
metrics = list(rows[0])[7:]
summary = []
paired = []
for branch in ("regression", "contrastive"):
    for pattern, target in (("A", "text"), ("V", "text"), ("AV", "text"),
                            ("A", "visual"), ("V", "audio")):
        groups = {}
        for step in ("1.0", "0.6"):
            group = [r for r in rows if (r["branch"], r["pattern"], r["target"], r["write_step"])
                     == (branch, pattern, target, step)]
            group.sort(key=lambda r: int(r["seed"]))
            assert [int(r["seed"]) for r in group] == list(range(66, 71))
            groups[step] = group
            out = dict(branch=branch, pattern=pattern, target=target, write_step=step)
            for metric in metrics:
                values = [float(r[metric]) for r in group]
                out[metric + "_mean"] = st.mean(values)
                out[metric + "_std"] = st.stdev(values)
            summary.append(out)
        for metric in metrics:
            delta = [float(b[metric]) - float(a[metric])
                     for a, b in zip(groups["1.0"], groups["0.6"])]
            paired.append(dict(branch=branch, pattern=pattern, target=target, metric=metric,
                               mean_delta=st.mean(delta), std_delta=st.stdev(delta),
                               positive_seeds=sum(x > 0 for x in delta),
                               negative_seeds=sum(x < 0 for x in delta)))
for name, data in (("summary.csv", summary), ("paired_delta.csv", paired)):
    with (ROOT / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(data)
lines = []
for branch in ("regression", "contrastive"):
    lines += [f"## {branch}", "", "Five-seed mean ± sample SD; retrieval/chance are percentages.", "",
              "| Pattern→target | η | Centered cos | Real−Shuffle cos | Retrieval / chance | Pred / teacher rank | Std ratio |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        if r["branch"] != branch:
            continue
        def ms(key, scale=1):
            return f"{scale*r[key+'_mean']:.4f} ± {scale*r[key+'_std']:.4f}"
        lines.append(f"| {r['pattern']}→{r['target']} | {r['write_step']} | {ms('centered_cosine')} | "
                     f"{ms('real_minus_shuffle_cosine')} | {ms('retrieval_top1',100)} / "
                     f"{ms('chance',100)} | {ms('effective_rank')} / {ms('target_effective_rank')} | {ms('std_ratio')} |")
    lines.append("")
(ROOT / "TABLES.md").write_text("\n".join(lines))
for seed in range(66, 71):
    a = json.loads((ROOT / f"results/eta1_seed_{seed}.json").read_text())
    b = json.loads((ROOT / f"results/eta06_seed_{seed}.json").read_text())
    assert a["mask_sha256"] == b["mask_sha256"]
    assert a["conversation_ids"] == b["conversation_ids"]
    assert a["state_unchanged"] and b["state_unchanged"]
print("Validated 100 groups, 10 checkpoints, 5 paired masks; summary complete.")
