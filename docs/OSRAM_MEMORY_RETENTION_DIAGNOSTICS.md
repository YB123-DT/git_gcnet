# Causal OSRAM association retention instrumentation

Diagnostic only. No training or model redesign. No real-data root-cause conclusion yet.

## Insertion and invariants

`OSRAMBackbone.forward` accepts keyword-only
`collect_memory_retention_diagnostics=False` and `memory_retention_diagnostics=None`.
The enabled path requires a forward-only backbone and a fresh external collector.
No model parameter, buffer, checkpoint key or trainer setting is added.
Disabled path allocates no probes and executes the same numerical operations.
Existing diagnostics and the misleading `memory_frobenius_norm` field are untouched.

In `_scan`, capture detached M_prev and raw stacked keys/values before masking;
run the existing decay, Base/Gap reads and block_write unchanged; pass actual
M_prev, M_read and M_post to the observer after write. The observer uses no_grad
and detached inputs exclusively; only its external probe state is updated.
It never mutates model memory or feeds outputs back into the computation.

**err_decay measures the state actually read for this utterance. err_post measures
the resulting state available to subsequent utterances, not current classification.**
These local signed changes do not by themselves establish the cause of F1 loss.

## State and streaming

last_key: [B,H,3,dk]; last_value: [B,H,3,dv].
last_seen_time and has_history: [B,3]. Store actual pre-mask projected key/value
only when observed AND valid. A new collector per batch/forward resets history.
Extra retained probe state is O(B H 3 (dk+dv)); one O(L) scan, no pairwise history.
The sink streams records to disk; passing list.append is useful only for tiny tests.

## Raw schema

All records: dataset, missing_rate, sample_id, time_index (zero-based),
target_modality, missing_pattern (A/T/V availability bits), status.
sample_id must identify conversation uniquely across batches for the analyzed rate;
include seed/run identity in it when pooling multiple runs.

status=NO_HISTORY: exactly one row per missing query without past real evidence,
no fake error, head or distance fields.

status=retention: one row per head, with head, history_distance,
err_pre/decay/post, cos_pre/decay/post, norm_ratio_pre/decay/post,
decay_damage, write_damage, max_key_overlap, mean_key_overlap.
Only currently missing modalities with historical observations are measured.
No current observed keys means null overlap, not zero. Padding emits nothing.
Signed damage may be negative. Head-level raw values allow variability analysis.

## Evaluation-only collection example

Use the existing loaded/eval checkpoint and identical masked input batch. At the
OSRAM call site the inputs already exist (see model.py's self.osram call).
Call that same backbone directly as follows; no optimizer or training is needed:

```python
import json
import torch
from gcnet_missing_m3.memory_retention import MemoryRetentionDiagnostics

# model already restored using its original config/checkpoint and put in eval().
# node, latents, availability, qmask, umask, lengths are the actual OSRAM inputs.
with open("retention.jsonl", "w") as stream:
    collector = MemoryRetentionDiagnostics(
        lambda row: stream.write(json.dumps(row) + "\n"),
        dataset="CMUMOSI", missing_rate=0.5,
        sample_ids=conversation_ids,
    )
    with torch.no_grad():
        h, contexts = model.osram(
            node, latents, availability, qmask, umask, lengths,
            collect_memory_retention_diagnostics=True,
            memory_retention_diagnostics=collector,
        )
```

The example assumes the actual batch has been obtained; it is not a standalone
checkpoint loader. Never replace masked input with full target features.
No changes to model.py or train_gcnet.py are required for this instrumentation.

## Analysis

```bash
python -m gcnet_missing_m3.analyze_osram_memory_retention retention.jsonl --output-dir retention_tables
```

Inputs: JSON list, JSONL or CSV. Outputs: coverage, distance, damage and overlap
tables, each CSV and Markdown. Coverage/distance count unique queries, not heads;
damage/overlap count head records. Dataset and missing rate remain separate.
Distance buckets: 1, 2-3, 4-7, 8+. Overlap buckets: [0,.1), [.1,.2),
[.2,.4), [.4,.6), [.6,1]. Overlap association is correlation, not proof of overwrite.

## Checkpoint availability and scope

No local checkpoint was found in the forward-only MOSI experiment folder.
Remote existing checkpoint confirmed:
`biggpu:/data2/yb/remote_experiments/osram_forward_only_mosi_20260908/seed_66/best.pt`.
This task has not run real-data evaluation or generated real retention records.
All verification records are explicitly synthetic. Do not present them as findings.
The existing checkpoint was selected by eight-rate mean Test-oracle and should be
identified as such if evaluated later. No new training has been launched.

Single-key mathematical check: normalized k and beta=.5, lambda_w=.001 give
beta/(beta+lambda_w)=0.998003992..., not a write coefficient of .5.
This test verifies the current formula and does not modify its parameters.
