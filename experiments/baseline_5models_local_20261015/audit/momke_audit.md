# MoMKE source and causal adaptation audit

Source: https://github.com/wxxv/MoMKE, branch `main`, commit
`2dceb57f9e6f4bc84c2fc8f4eb7c761ea64c9efc` (verify branch in repository registry).
Upstream checkout: `sources/MoMKE/`; local adaptation: `adapters/momke/`.
The upstream checkout is untouched by this adapter. It does not use network logging.

## Preserved method and version distinction

The local `model.py` and `attention.py` are copies of upstream
`MoMKE/model_expert_softmoe.py` and `MoMKE/modules/Attention_softmoe.py`.
Parameter names/shapes are unchanged, verified by strict upstream state-dict loading.
The modality projections, modality experts, depth, shared expert application,
per-utterance routers, softmax routing, expert aggregation, residual connections,
cross-modal fusion, auxiliary heads and emotion classifier are retained.
Speakers are not an architectural input to official MoMKE; the adapter accepts
`nspeakers` for the shared factory interface and does not collapse MELD speakers.
Seven-class MELD is implemented through the existing classifier dimension argument.

Only attention visibility differs between Original and Causal (`config.causal`).
Causal adds `key_time <= query_time` to the score mask. Both versions apply the same
modality availability mask and padding mask, and zero unavailable raw inputs before
projection. No expert routing operation reduces across time: routers are MLPs of
individual projected utterances followed by softmax over three experts. Therefore
causal masking of each temporal attention layer suffices; no routing redesign is
necessary. Original intentionally retains future temporal access.

The existing all-masked-key case applied softmax to all `-inf`, then replaced NaN
probabilities. That can retain NaN gradients. Both versions now use finite dummy
scores exclusively for empty attention rows and set their attention probabilities
to zero. Nonempty rows retain the upstream softmax exactly. This is a numerical
compatibility fix, not an added model block. An absent modality's residual MLP bias
can still exist inside the official experts; final availability gating, as upstream,
prevents it from becoming a visible modality representation.

`__init__.py` is a batch-first and random-MR adapter; it does not add parameters.
It masks all unavailable features, so routing and residual paths cannot read hidden
raw inputs. Test masks are supplied by the shared fixed-mask data layer rather than
generated inside MoMKE. Both model versions therefore share identical input masks.
Upstream's diagnostic Stage 2 auxiliary outputs are random placeholder tensors and
must not be used for loss or evaluation; Stage 2 uses only `output['logits']`.

## Shared runner contract

```python
from adapters.momke import build_model, stage_loss
model = build_model(config, (audio_dim, text_dim, video_dim), nclasses, nspeakers)
output = model(a, t, v, availability, padding, first_stage=first_stage)
loss = stage_loss(output, labels, padding, first_stage)
```

Features: `B x L x D`; availability: `B x L x 3` in A/T/V order; padding:
`B x L`, with True denoting a real utterance (the upstream `umask` convention).
`output['logits']`: `B x L x C`;
`output['auxiliary_logits']`: three `B x L x C` heads (valid in Stage 1).
Use `config.causal=False` for Original, True for Causal. All outputs are raw logits.

## Official training configuration and stages

`run_MoMKE_iemocap4.sh`: epochs=200, stage_epoch=100, batch_size=16,
lr=1e-4, hidden=256, depth=4, heads=2, dropout=0.5, attention_dropout=0,
seed=66; official features wav2vec-large-c-UTT, deberta-large-4-UTT, manet_UTT.
`train_MoMKE.py` uses Adam with weight_decay=l2=1e-5 and a single optimizer
through both stages; there is no scheduler, gradient clipping, or expert freezing.
The repository has no dedicated IEMOCAPSix shell script. The same trainer explicitly
supports IEMOCAPSix with six classifier outputs and five folds; using the IEMOCAPFour
configuration for Six should be marked as the shared official IEMOCAP setting.

Stage 1 (`epoch < stage_epoch`, zero-based): all three training modalities are
present. Run each input through its own modality expert stack and minimize
CE(audio head)+CE(text head)+CE(video head), averaged over real utterances.
The original performs three backward calls before one optimizer step; summing these
three independent losses before backward gives the same gradient. Routers and fusion
classifier are not optimized by Stage 1 loss. Stage 2 runs each projected modality
through all three experts, weights outputs with per-utterance routing, concatenates
visible representations and minimizes fused emotion CE. Training random MR replaces
the official fixed `test_condition` for both versions; Stage 1 training remains
complete-modal. Test evaluation always accepts the fixed common availability mask.

The source's intended Stage 1 handoff collects `models.append(model)`, which stores
multiple references to one changing object, and calls `model.state_dict().update(...)`,
which does not reload parameters. Thus its actual effective behavior is continuing
from the final Stage 1 weights. The local runner should preserve that actual stage
transition; it must use real serialized/copied best checkpoints for final selection.
The source also never appends losses to the printed loss list, making its printed
train loss uninformative; the shared runner must record actual `stage_loss` values.

Formal selection must be **Stage 2 only, Test ACC maximum, earliest epoch on ties**.
Save all metrics and predictions from that same real checkpoint. This is
`checkpoint_selection=test_best_acc`, `evaluation_protocol=test_oracle` and has
optimistic test-selection bias. It is not an unbiased independent test evaluation.
For MELD budget comparison use training convergence/stability diagnostics only;
no Test score comparison is authorized to choose 100/50 versus 200/100.

## Verification

Executed on CPU in `comp-repro` using:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/yangbin/miniconda3/envs/comp-repro/bin/python \
  experiments/baseline_5models_local_20261015/tests/test_momke_causality.py
```

9 tests passed on 2026-10-15, including both Stage 1 and Stage 2:
future feature perturbation, future missing-mask perturbation, prefix equivalence,
padding invariance, batch independence, unavailable feature perturbation,
finite forward/backward and actual Adam parameter update, an Original noncausal
negative control, and Original numerical agreement with official logits/auxiliary
heads after strict upstream state-dict loading. Tests use eval mode for causality;
training mode only for backward/update smoke checks. All-missing modality and padded
utterances are included in gradient checks. Tolerances are atol=2e-6, rtol=2e-5.
This verifies semantics and training viability, not predictive performance.

A real MELD-feature CPU smoke check also passed with the full suggested model
(hidden=256, depth=4, heads=2, dropout=0.5): dialogue 0 has 14 utterances,
A300/T1024/V342, seven classes, and fixed test MR=0.6. Both versions completed
Stage 1 and Stage 2 forward, finite backward gradients, an Adam update, and
evaluation producing `[1,14,7]` finite logits. Each version has 5,175,845
parameters. This is a single-batch viability check, not a completed training run.

```json
{
  "kind": "real_feature_cpu_one_batch_smoke_not_training_result",
  "dialogue": "0",
  "seconds": 0.7241824269294739,
  "checks": [
    {
      "causal": true,
      "first_stage": true,
      "loss": 5.473258972167969,
      "shape": [
        1,
        14,
        7
      ],
      "parameters": 5175845
    },
    {
      "causal": true,
      "first_stage": false,
      "loss": 2.0152297019958496,
      "shape": [
        1,
        14,
        7
      ],
      "parameters": 5175845
    },
    {
      "causal": false,
      "first_stage": true,
      "loss": 6.260372638702393,
      "shape": [
        1,
        14,
        7
      ],
      "parameters": 5175845
    },
    {
      "causal": false,
      "first_stage": false,
      "loss": 1.971985936164856,
      "shape": [
        1,
        14,
        7
      ],
      "parameters": 5175845
    }
  ]
}
```
