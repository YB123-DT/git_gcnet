# Equal-Budget Stratified Mixed-Rate Training

## Research question

Can one train a single missing-modality checkpoint across rates 0.0--0.7
without multiplying each training batch into eight views?

The existing `all` mode performs eight forward/backward views for every source
batch and averages their gradients before one optimizer step. It does not add
new utterances, but it supplies approximately eight times as many masked
training views and approximately eight times the trunk compute. Results from
that mode are useful diagnostics, not an equal-budget comparison with a
single-view baseline.

## Alternatives considered

1. **Independent rate per batch.** This preserves one view per batch, but it is
   poorly balanced on dialogue datasets with few batches. With eight equally
   likely rates, a specified rate is absent with probability `(7/8)^B`; this is
   76.56% for MOSI's two training batches per epoch and 58.62% for IEMOCAP-6's
   four.
2. **Deterministic cyclic rate per batch.** The existing `cyclic` mode guarantees
   eventual coverage and is an equal-budget control. It nevertheless exposes
   every conversation in a batch to the same rate and may require several
   epochs to cover all rates.
3. **Conversation-level batch-balanced random rates.** This is the selected
   formal protocol. It gives each conversation one masked view, represents all
   rates within a normal full batch, and randomizes which conversation receives
   each rate. It is stratified sampling without replacement, not IID sampling.

Training one independent checkpoint per rate remains a valid rate-specific
protocol but is not selected: it changes the research claim from one unified
model to eight models and multiplies total model-training cost.

## Sampling unit and balance

The sampling unit is a conversation, because GCNet batches conversations and
constructs temporal/speaker graphs within each conversation. It is not an
utterance and it is not an entire batch.

For a training batch containing `B` conversations and `K=8` rates, define:

```text
q, r = divmod(B, K)
```

Each rate is assigned to `q` conversations. The remaining `r` conversations
receive distinct extra rates. Let `stream_offset` be the number of training
conversations already assigned before this batch, including preceding epochs.
The extras go to consecutive entries beginning at
`stream_offset mod K`, then the complete assignment is shuffled by a local
deterministic generator. After the batch, advance `stream_offset` by `B`.
Equivalently, the epoch starts at `(epoch * N_train) mod K` and advances by the
actual size of every batch. Thus:

- every conversation appears exactly once and receives exactly one rate;
- rate counts within a batch differ by at most one;
- a full `B=32` batch contains exactly four conversations at each rate;
- every batch with `B>=8` contains all eight rates;
- a tail batch with `B<8` does not duplicate conversations to force coverage;
- cumulative counts after every completed batch differ by at most one;
- tail-batch extras continue across batch and epoch boundaries, so no rate is
  permanently omitted.

Because conversation lengths differ, equality by conversation does not imply
exact equality by utterance or loss weight. The allocation must be independent
of labels, conversation length, speaker identity, and batch position; per-rate
utterance and target counts are reported rather than silently assumed equal.

Only the mapping from this balanced rate multiset to conversations is shuffled.
The generator seed is stably derived with SHA-256 from an algorithm version,
the formal seed, dataset, fold, epoch, batch index, batch size, and conversation
identifiers. It must not use Python's process-randomized `hash()` or consume
Python, NumPy, or Torch global RNG state. Equal inputs reproduce the assignment;
changing the formal seed or epoch changes the mapping.

Pure IID sampling is not used. In a full `B=32` batch, IID sampling would omit
a specified rate with probability 1.394% and would omit at least one of the
eight rates with probability 10.872%. Stratification removes both events while
preserving a `1/8` marginal allocation.

The four `eta=0` assignments in a full batch guarantee complete conversations
in that batch, but they do not guarantee that each particular conversation
sees `eta=0` during a finite training run. Under an independent marginal model,
the probability that one conversation never receives `eta=0` in `E` epochs is
`(7/8)^E`; the realized assignment history is therefore retained for audit.

## Missing-mask construction

For conversation `c` assigned rate `eta_c`, reuse the existing
`ConversationMaskSchedule` associated with that rate. Generate host and guest
availability independently with the existing conversation-, side-, epoch-,
and seed-keyed schedule, then assemble the per-conversation tensors into one
batched availability tensor.

The existing mask semantics remain unchanged:

- before nonempty repair, every modality is independently observed with
  probability `1-eta_c`;
- an all-missing valid utterance is repaired by retaining one random modality;
- padding availability is zero;
- `eta_c=0` produces ATV for every valid utterance in that conversation;
- validation and test masks remain frozen, rate-specific, and unchanged.

Because repair restores one modality with probability `eta_c^3`, the expected
realized missing-element fraction is

```text
E[realized_eta_c] = eta_c - eta_c^3 / 3,
```

not `eta_c`. For requested `eta=0.7`, the expectation is approximately
0.585667. Reports must distinguish requested rate from realized missing-element
fraction and record both.

No complete pattern needs to be injected after mask generation. For `B>=8`,
the stratified batch already contains at least one `eta=0` conversation, which
guarantees valid ATV utterances. For smaller tail batches, forcing ATV would
bias the requested rate mixture and is therefore forbidden.

## Training data flow

```text
B original conversations
        -> one stratified rate per conversation
        -> one availability tensor containing mixed rates
        -> one incomplete feature batch
        -> one model forward
        -> one classification loss + one JEPA loss
        -> one backward + one optimizer step
```

The model already supports different availability patterns at different nodes,
so the graph topology, relation types, model architecture, and inference path
do not change. The default formal JEPA rate weighting remains uniform. Any
future non-uniform rate weighting requires a separately specified
per-conversation loss and is outside this change. The implementation must reject
`stratified` together with a non-uniform `jepa_rate_weighting` rather than apply
one incorrect scalar weight to a mixed-rate batch.

## Interface and compatibility

Add a new explicit training mode named `stratified`. Preserve `fixed`,
`cyclic`, and `all` byte-for-byte in behavior so every historical experiment
remains reproducible. Do not silently change the global CLI default; formal
runners must opt into `--train-rate-mode stratified`.

Training logs and result metadata must record:

- the rate assigned to every conversation through a stable assignment hash;
- per-rate conversation counts for each epoch;
- source conversation count, masked-view count, forward count, and optimizer
  step count;
- the sampling algorithm version.

The expected source-conversation count and masked-view count are equal. In
particular, `stratified` must not report the eightfold prediction count produced
by `all`.

The first implementation targets the repository's current single-process,
single-GPU jobs. If distributed data parallelism is later introduced, balance
must be defined per local batch and audited globally; this specification does
not silently assume global balance across devices.

## Fair experimental comparison

The final claim requires the same stratified sampler, epoch count, batch order,
optimizer-step budget, feature bank, and fixed evaluation masks for every
trainable comparison arm. Existing `all` results are retained only as an
extra-view-budget ablation.

The minimal staged experiment is:

1. train Slot Missing-M3 With-JEPA and JEPA-gradient-off under `stratified`;
2. compare seeds 66--70 on IEMOCAP-6 as the first mechanism gate;
3. only if the JEPA contribution is stable, run a stratified Original GCNet
   control before making a superiority claim;
4. do not inherit an Original result trained under a different view budget as
   the formal matched baseline.

Validation selects one checkpoint from the mean metric across the same eight
fixed validation rates. Testing that checkpoint across eight rates is repeated
evaluation, not additional training data.

## Verification requirements

1. `B=32` assigns each rate exactly four times.
2. Arbitrary `B` gives counts differing by at most one; `B>=8` covers all rates;
   cumulative counts at completed batch boundaries remain within one across
   variable-size and tail batches.
3. The same inputs reproduce assignments and leave all global RNG states
   unchanged; seed/epoch changes alter assignments.
4. Mixed-rate masks align with conversation order, speaker selection, valid
   utterances, and padding.
5. Every valid utterance retains at least one modality and every `eta=0`
   conversation is fully ATV.
6. One source batch triggers one model forward, one backward aggregation, and
   one optimizer step.
7. Stored assignment hashes, requested-rate counts, and realized missing rates
   recompute exactly.
8. Existing `fixed`, `cyclic`, and `all` tests remain green.

## Claim boundary

This protocol supports the statement that a single model learns from a
balanced mixture of missing rates without increasing the number of training
examples or masked views **per epoch**. Across epochs it deliberately exposes a
conversation to different masks, which is stochastic data augmentation and
must be described as such. It does not make the amount of information inside
different masks equal, and it does not by itself prove superiority over
rate-specific models. Training compute, per-rate exposure, number of masked
views, number of checkpoints, and evaluation protocol must all be reported
separately.
