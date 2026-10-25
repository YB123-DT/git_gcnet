# What this experiment permits us to say

## Actual matched result

MOSI seed66,new GCNet Missing,cmerc-repro,eight own-parent rate-specific
Signature checkpoints. All normal predictions replayed bit-exactly. Frozen
inference took43.675seconds,peak allocated GPU memory317.314MiB. Local and remote
17-test suites passed. Independent recomputation verified all48 rate/control
metrics and no-history invariance. No training or new checkpoint selection.

| Frozen evaluation | Eight-rate ACC | Eight-rate WF1 | Eight-rate MSE |
| --- | ---: | ---: | ---: |
| Original Flat |79.535061|79.460447|1.496185|
| Normal Signature |79.858994|79.847119|1.464113|
| Interior shuffle,mean of3 inference draws |79.852642|79.840088|1.462995|
| Remove second-level antisymmetric area |79.611280|79.588720|1.454506|

The3 shuffled rows are not3 training seeds or an averaged-prediction ensemble.
Their separate results remain in RESULT.md/JSON/CSV. VERIFICATION.json checks
48 actual conditions,eight rates times six predictions.

## Conclusions,including contrary evidence

1. The Signature branch improves WF1 by0.386672pp against its own frozen Flat
   parent on this internal seed66 comparison. This is additional budget and
   Test-oracle exploration,not an independent generalization claim.
2. Preserving endpoints and shuffling observed interior tokens barely changes
   mean WF1:79.847119→79.840088. Even in shuffle-eligible targets,mean WF1 is
   81.054320 versus81.042886 across the3shuffles. This does not support a strong
   explanation based on precise interior observation ordering. Endpoint trend
   and associated temporal information are retained by this intervention,so it
   also does not establish that all temporal information is irrelevant.
3. Removing the antisymmetric second level lowers WF1 by0.258399pp. Effects are
   conditional: miss0.4 loses1.094448pp,miss0.5 loses0.647276pp,while miss0.1,
   0.3 and0.6 have no polarity-classification change. Thus the oriented path
   features do affect some learned classification decisions.
4. That removal IMPROVES MSE:1.464113→1.454506. Do not claim that all metrics
   improve through ordered dynamics. The intervention changes the learned input
   distribution and may also change downstream normalization; an effect is not
   proof that path order is the unique cause of the original gain.
5. Normal Signature still beats Flat in MSE,but the remaining benefit after
   area removal may come from endpoints,nonlinear endpoint products,past counts,
   added current-Local processing or their interactions. Those sources are not
   separated by this audit. We have not established that equally trained
   alternatives cannot use them.

## Defensible original research motivation

The original task path uses the current representation together with
query-conditioned reads from a compressed causal associative state. These reads
do not explicitly expose a modality-wise description of the recent observed
path. Under missing modalities,the actual points observed in that path differ
between modalities. We therefore investigate a complementary,strict-past
observed-path descriptor: encode each modality's recent projected observations
together with utterance-position coordinates,and supply low-order path features
to the task path without changing CHM writes or inventing missing observations.

This is a prospective design hypothesis. It is compatible with CHM already being
order-sensitive and does not assert that CHM discards all trends,that Signature
recovers unavailable semantics,or that its finite truncation uniquely identifies
the history. Current evidence supports limited use of second-level geometry in
classification,but NOT precise temporal-order recovery as a verified bottleneck.

If a stronger mechanism claim is needed,the missing decisive comparison is an
equally trained branch using the same projected observations,time coordinates,
window,Local/counts,head and budget but omitting oriented-area information. That
new training has NOT been launched or included in these results.

## Reproducibility and limits

Numerical source `e0ca94c4c33de0efda2c2f4786fd289067e1ab2c`;
audit source `9130a9e57ac29e6067fae0bab44c85ff90d3433e`.
Remote outputs:
`/data2/yb/remote_experiments/osram_signature_order_audit_20261025/execution_v1`.
LAUNCH.json records GPU2 UUID,environment and source hashes. VERIFICATION.json
records prediction hashes,exact replay and independent metric checks. Existing
training jobs and model files were not changed. New predictions remain remote.

Bootstrap intervals are exploratory per-rate conversation-cluster MSE intervals,
not multiple-comparison-corrected significance tests or training-seed uncertainty.
This audit does not validate MOSEI,IEMOCAP,MELD or additional learned seeds.
