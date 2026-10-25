# MOSEI Nested: confirmed issues versus unproven causes

INTERNAL DIAGNOSTIC ONLY. Read-only review, no model changes, task-data forward,
new training run, loss, threshold or LR search. The proposed four global-LR
runs were NOT started. Existing weights, masks, predictions and active jobs
were not modified.

## Direct answer

There is a reproducible **conditional rooted-pool degeneracy**, not evidence
that the entire Nested module fails. With current ATV, no Gap is active:only
Local and Base-head/group tokens remain. Their graph is complete, and every
one-hop rooted subgraph contains the same nodes/edges. At epsilon0 the first
GIN update erases root-marker location. Real trained N10/N15 checkpoints have
small nonzero epsilons; fixed synthetic CPU inputs still yield almost identical
ATV rooted-pool vectors. Six missing patterns do not show this equality and
the final core explicitly concatenates original x, so final node outputs differ.

This does NOT establish why rescues and harms cancel. The prominent measured
A-only harm is a NON-ATV condition and cannot be directly explained by ATV
degeneracy without further evidence. A local structural weakness and the task
failure must not be conflated.

## 1. What is actually implemented

- `gcnet_missing_m3/nested_sweep.py:38`:same-role edges, Local-to-all edges,
  optional same-head/group edges.
- `:48`:GIN update `norm(MLP((1+eps)*h + adjacency@h))`.
- `:62`:each node's one-hop induced subgraph; additive root/distance labels.
- `:69`:mean of node states per layer, followed by shared pool MLP.
- `:72`:final readout sees `[x,rooted,mean(rooted)]`.
- `meaningful_input_new40.py:33`:real token packing and per-head/group decoders.
- `meaningful_input.py:83`:strict history/nonzero-read guard and Gap masks.
- `osram.py:2268`:corrected Local/Base/Gap enter original Flat;
  LocalSkip retains uncorrected Local.

Graph groups concatenate contiguous actual CHM heads; they do not alter the
CHM8x64 state, queries or joint write. No evidence of wrong head slicing,
inactive Gap leakage, wrong LocalSkip routing or a disconnected classifier path
was found.

### Complete-graph calculation

For adjacency A=11^T-I:

```text
(1+epsilon)H + AH = 11^T H + epsilon H.
epsilon=0 -> every node receives the same sum.
```

Each root subgraph has one root label and n-1 nonroot labels, hence the same
label sum. Mean pooling cannot recover the erased label-feature association.
This is exact at initialization, not a proof of equality for every learned
nonzero epsilon or every real input. The two additive binary root/distance
embedding tables also only occur in two combined forms; their separate
parameters are redundant, not a proven performance cause.

### Checks using real trained weights

Read16 protected BEST files on CPU, no GPU visibility, single thread.
Synthetic Local/evidence seed6677, batch2, float64, dimensions256 and4x512.
These are NOT task-data representation distribution measurements.

| Check | N10 | N15 |
|---|---:|---:|
| Absolute trained epsilon range | 2.79e-7–3.06e-5 | 1.57e-5–9.02e-5 |
| ATV max root-pool difference, range over8 BEST | 1.73e-18–1.39e-12 | 2.39e-15–7.11e-12 |
| Final core node difference, miss.7 BEST | 0.274197 | 0.161362 |

At miss.7 BEST, pooled-root maximum differences:

| Current availability | N10 | N15 |
|---|---:|---:|
| A | 0.000721514 | 0.002566203 |
| T | 0.000669533 | 0.002566556 |
| V | 0.000706732 | 0.002497581 |
| AT | 0.000596717 | 0.002011355 |
| AV | 0.000673189 | 0.001861824 |
| TV | 0.000565299 | 0.001946720 |
| ATV | 1.73e-18 | 2.39e-15 |

Nonzero differences elsewhere do not prove useful relations; near-equality in
ATV does not imply the full module output or prediction is identical.
ROOT_CPU_CHECK.json records checkpoint/source hashes and exact measurements.
The separate reviewer's same CPU checks agree up to floating-point roundoff.

Minimal CPU reproduction (use the immutable snapshot as PYTHONPATH, with
CUDA_VISIBLE_DEVICES empty; no data loader or original model forward):

```python
import sys
from pathlib import Path
import torch
root = Path('/data2/yb/remote_experiments/osram_mosei_flat128_nested_20261024')
sys.path.insert(0, str(root / 'source_v1'))
from gcnet_missing_m3.meaningful_input_new40 import build_new40
torch.set_num_threads(1)
checkpoint = torch.load(root / 'execution_v1/runs/n10/best_miss_0p7.pt',
                        map_location='cpu', weights_only=False)
prefix = 'osram.meaningful_block.core.'
adapter = build_new40('mosei_f128_n10').double().eval().requires_grad_(False)
adapter.load_state_dict({k[len(prefix):]: v for k, v in checkpoint['model'].items()
                         if k.startswith(prefix)}, strict=True)
torch.manual_seed(6677)
local = torch.randn(2, 256, dtype=torch.double)
evidence = torch.randn(2, 4, 512, dtype=torch.double)
active = torch.tensor([[1, 0, 0, 0]], dtype=torch.bool).expand(2, -1)  # ATV
tokens, mask = adapter.tokenizer(local, evidence, active)
columns = mask[0].nonzero(as_tuple=True)[0]
pooled = []
hook = adapter.core.pool.register_forward_hook(
    lambda module, args, out: pooled.append(out.detach()))
with torch.no_grad():
    output = adapter.core(tokens[:, columns], columns, adapter.num_heads)
hook.remove()
rooted = torch.stack(pooled, 1)
print((rooted - rooted[:, :1]).abs().max())
print((output - output[:, :1]).abs().max())
```

## 2. Why rescues and harms nearly cancel: what the data actually says

Previously verified N10 versus independently-trained same-width Flat128,
same labels/masks/IDs/utterance order, each model's accepted per-rate BEST:

| Current availability | Corrections | Harms | Net correct |
|---|---:|---:|---:|
| A |112|174|-62|
| T |74|85|-11|
| V |167|137|+30|
| AT |85|78|+7|
| AV |132|115|+17|
| TV |70|62|+8|
| ATV |202|195|+7|
| Total |842|846|-4|

These are29072 repeated rate-utterance records, not unique samples and not a
weighted-F1 decomposition. N10 changes5.81% of polarity decisions; it is not an
unchanged model. In A-only positives,43rescues versus121harms gives net-78;
A-only negatives give69/53,net+16. Positive-to-negative changes total190 and
negative-to-positive changes96:94net fewer positive predictions in A-only.
This is observed conditional decision redistribution, not an identified causal
reason for it.

The ATV-local rooted weakness cannot by itself explain A-only losses. Lack of
past Text alone is also insufficient:the earlier A-only positive analysis found
nearly equal harm proportions among originally correct examples with/without
past Text (5.926%/5.932%). Existing freeze/scale results do not reveal a large
uniform correction-strength fix either.

Sources:`osram_mosei_flat128_nested_20261024/HARM_BREAKDOWN.md`,
`A_ONLY_POSITIVE_DETAIL.md`, and the preserved
`osram_mosei_n10_scale_20261024/{SLOT_RESULT,FINE_SLOT_RESULT}.md`.
These comparisons include jointly-trained backbone changes and different
selected epochs, not just direct opening/closing of one branch.

## 3. Common implementation explanations checked and unsupported

All21 actual runs were independently checked against Flat128:
60complete epochs,60/60identical training mask traces, identical per-epoch
optimizer steps, constant1e-4 LR, same exact test-mask reference/environment
and original-parameter initialization digest. First-epoch finite nonzero
Nested gradients cover39–69parameter tensors and all expected components;
epoch60 also records nonzero paths.

The worker checks all model parameter IDs enter Adam, and every task step
performs one Memory forward. Update order is zero_grad -> masked original MSE
-> backward -> finite-gradient checks -> original global clip -> Adam step.
Selection saves the arrays and ACC/W-F1/MSE from the same evaluation call;
publishing copies saved predictions rather than sampling another mask.

Therefore no present evidence supports permanent zero-init disconnection,
missing optimizer parameters, extra Memory trajectories or mismatched masks
as the common explanation. Nonzero gradients do not prove good optimization
or useful learned features.

## 4. Training trajectories: learning occurs, later task scores can worsen

The following checkpoints are each model's best COMMON-epoch eight-rate WF1
point versus epoch60, used ONLY to inspect trajectories. They do not replace
the accepted per-rate Test-best scores.

| Model | Reference epoch | Train MSE then / epoch60 | Test mean MSE then / epoch60 |
|---|---:|---:|---:|
| LargeFlat66 | 12 | 0.634024 / 0.395881 | 0.645484 / 0.668417 |
| Flat128_66 | 16 | 0.600594 / 0.396012 | 0.612051 / 0.648275 |
| N10 | 24 | 0.532766 / 0.396605 | 0.611863 / 0.683341 |
| N15 | 19 | 0.585643 / 0.401630 | 0.640201 / 0.646124 |

Train loss decreases while test loss later increases in these comparisons.
This is not a universal optimization-failure diagnosis and does not prove
why the added block fails to improve its BEST result. The accepted protocol
already retains earlier BEST checkpoints, so simply stopping earlier does
not manufacture a better reported Test-best score. TRAINING_CURVES.json keeps
all60epochs, including oscillations rather than a cherry-picked monotonic curve.

## 5. A concrete correction to the preceding advice: local LR was already tested

The suggestion that module-specific lower/higher LR was a wholly unanswered
question was insufficiently grounded. Prior complete cmerc MOSEI tests used
LargeFlat1600+Nested32, backbone1e-4, while changing only Nested LR:

| Configuration | Mean8 WF1 | High WF1 |
|---|---:|---:|
| Large Flat1600 |83.477635|79.898427|
| Nested32, local1e-4 |83.549017|80.132940|
| Nested32, local5e-5 |83.490984|80.036477|
| Nested32, local2e-4 |83.369614|79.971689|

This is not the identical Flat128+N15/global-LR comparison, but it refutes
calling generic local-LR mismatch an unexplored explanation. No current
training-fault evidence makes the proposed new four-run sweep urgent.

Also N15=83.400889 is only the best in the Flat12821-candidate batch, NOT the
highest historical MOSEI Nested result. The prior same-protocol Nested32
83.549017 is higher. None of these seed66 test-selected rankings establish
stable multi-seed improvement. PREVIOUS_LOCAL_LR.json gives complete paths/hash.

## 6. Do not recycle already-completed diagnostics as a new idea

- Frozen rooted/PreDecoder probes already exist:
  `osram_nested_root_probe_20261011/completed/RESULT.md`. Those are old MOSI
  checkpoints, not current MOSEI evidence; Flat+Rooted did not beat Flat under
  that bounded probe protocol. It is not valid to propose that exact probe
  again as if never done.
- Text-present/missing gradient and temporary-update diagnostics already exist:
  `osram_text_condition_optimization_audit_20261022/completed/CMUMOSEI/RESULT.md`.
  They are s0/fixed LargeFlat checkpoint, not this cmerc N10. Overall mean
  gradient cosine0.584,17.19%negative; temporary cross-condition steps improved
 53/64and harmed11/64in both directions. This does not support declaring
  universal Text-condition optimization conflict; it also cannot resolve the
  current Nested mechanism by transfer of scores across environments.
- Current all-rate A/T/V strict-past exposure and readout removals were completed
  in `osram_mosei_missing_history_audit_20261025/`. Never-seen target modality
  is not empty shared Memory, and final-slot removal in N10 retains earlier
  mixing into corrected Local/other slots.

## Bottom line

Confirmed:ATV rooted pooling has a concrete identity-discrimination limitation;
the actual decision changes cancel differently across modality/label conditions;
several previous optimization/probe suggestions were already tested.
Not confirmed:a single causal bottleneck explaining all failures, that fixing
ATV rooting improves missing-mode predictions, or that more LR trials are now
the justified next step. No model fix or new experiment is authorized/launched
by this diagnosis. The four proposed LR runs remain unstarted.

Only this audit's small text/JSON artifacts are added. No production code,
weights, masks, old results, running jobs or experiment protocol were changed.
