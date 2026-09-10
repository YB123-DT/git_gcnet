"""Re-extract existing test histories; never train or overwrite checkpoints."""
import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []
provenance = []
for seed in range(66, 71):
    metrics = [json.loads((ROOT / f"results/seed_{seed}/{m}_metrics.json").read_text())
               for m in ("p0", "b2")]
    assert metrics[0]["mask_sha256"] == metrics[1]["mask_sha256"]
    for model in ("p0", "b2"):
        path = ROOT / f"results/seed_{seed}/{model}_history.json"
        history = json.loads(path.read_text())
        assert sorted(r["epoch"] for r in history) == list(range(1, 101))
        provenance.append({"path": str(path.relative_to(ROOT)),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "epochs": len(history)})
        for rate in (f"{i / 10:.1f}" for i in range(8)):
            assert all(math.isfinite(r["test_oracle"][rate]["weighted_f1"]) for r in history)
            best = max(history, key=lambda r: (r["test_oracle"][rate]["weighted_f1"], -r["epoch"]))
            rows.append({"model": model, "seed": seed, "rate": rate,
                         "selected_epoch": best["epoch"],
                         **{k: best["test_oracle"][rate][k] for k in
                            ("weighted_f1", "macro_f1", "accuracy", "mae", "correlation")}})

def scores(model, rate=None, seed=None):
    return [r["weighted_f1"] * 100 for r in rows if r["model"] == model
            and (rate is None or r["rate"] == rate) and (seed is None or r["seed"] == seed)]

with (ROOT / "per_rate_oracle.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
(ROOT / "per_rate_oracle_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
lines = ["# Per-rate Test-oracle re-extraction", "",
         "INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.", "",
         "Each seed and each missing rate independently selects the highest test weighted-F1 among epochs 1–100. Ties select the earliest epoch. Other metrics come from that same selected epoch. This is an optimistic multi-epoch diagnostic, not one deployable checkpoint's eight-rate performance.", "",
         "No retraining or model evaluation was performed. Existing best.pt files are unchanged; these are logged metrics, not a claim that all selected epoch checkpoints were retained. P0 and B2 evaluation mask hashes match for all five seeds. B2 has extra pretraining/fine-tuning budget; this is not an equal-budget ablation.", "",
         "## Five-seed mean W-F1 (%)", "",
         "| Rate | P0 | B2 | Delta (pp) | Positive seeds |",
         "|---|---:|---:|---:|---:|"]
for rate in (f"{i / 10:.1f}" for i in range(8)):
    a, b = scores("p0", rate), scores("b2", rate)
    lines.append(f"| {rate} | {st.mean(a):.4f} | {st.mean(b):.4f} | {st.mean(b)-st.mean(a):+.4f} | {sum(y>x for x,y in zip(a,b))}/5 |")
a, b = st.mean(scores("p0")), st.mean(scores("b2"))
lines += [f"| Eight-rate mean | {a:.4f} | {b:.4f} | {b-a:+.4f} | — |", "",
          "## Per-seed eight-rate mean and selected epochs", "",
          "Epoch lists follow rates 0.0 through 0.7.", "",
          "| Seed | P0 mean | B2 mean | Delta (pp) | P0 epochs | B2 epochs |",
          "|---|---:|---:|---:|---|---|"]
for seed in range(66, 71):
    a, b = st.mean(scores("p0", seed=seed)), st.mean(scores("b2", seed=seed))
    epochs = [[r["selected_epoch"] for r in rows if r["model"] == m and r["seed"] == seed] for m in ("p0", "b2")]
    lines.append(f"| {seed} | {a:.4f} | {b:.4f} | {b-a:+.4f} | {epochs[0]} | {epochs[1]} |")
for model in ("p0", "b2"):
    high = [r["weighted_f1"] * 100 for r in rows if r["model"] == model and float(r["rate"]) >= .5]
    lines += ["", f"{model.upper()} high-missing (0.5/0.6/0.7) mean: {st.mean(high):.4f}%."]
lines += ["", "Full per-seed/rate metrics (fractions for F1/accuracy): `per_rate_oracle.csv`. Original histories and hashes are retained for reproducibility. The previous `RESULT.md` remains the explicitly labeled eight-rate-mean selection report; do not mix the two protocols.", ""]
(ROOT / "RESULT_PER_RATE_ORACLE.md").write_text("\n".join(lines))
print("\n".join(lines))
