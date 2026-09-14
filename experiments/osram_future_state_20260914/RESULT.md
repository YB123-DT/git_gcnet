# Future-State JEPA: implementation and smoke evidence

INTERNAL ENGINEERING DIAGNOSTIC ONLY — no full training and no new F1 result.

Branch: `feature/osram-complete`. Reference `2358937`; implemented on its
descendant `abbd71c`, preserving the intervening matched-smoke diagnostic.

## Implemented path

```
real observed input → unchanged causal OSRAM (.6, mean/Flat)
                         ├→ unchanged emotion readout/head
                         └→ post-write M_t → normalized state query per head
                                           → c_t (256) → MLP → predicted next Local
full x_(t+1) → EMA full encoder → node + EMA Local path → target (256)
```

Only adjacent valid utterances within the same conversation are paired. Target
count is independent of missing rate. No teacher memory, labels, B2, WSC,
InfoNCE, old MMoE forward, or prediction write-back. Loss is emotion + .1 times
SmoothL1 of parameter-free LayerNorm outputs. EMA is updated after optimizer.

The observer receives a differentiable clone of the actual post-write memory;
its return value is ignored. Even an in-place-mutating callback cannot alter
the persistent scan. The head retains compressed states, not an explicit list
of full memory matrices. Autograd still retains the intermediates required for
backpropagation through the scan; this is not a constant-memory training claim.

In eval the observer is absent: no state query, predictor, or Teacher is called.
Legacy MMoE/teacher parameters remain loadable but unused/frozen in this mode.
Old configs default to the original path, with no new parameter keys.
Future checkpoints include the new head, EMA teacher, and EMA update counter.

## Parameter accounting

Actual configuration: H=8, dk=dv=32, latent=256, OSRAM output=700.

| Quantity | Count |
|---|---:|
| Complete-State online budget | 250,872 |
| Future online query + predictor | 250,855 |
| Difference | −17 |
| Automatically chosen hidden width | 487 |
| New frozen EMA encoder + Local parameters | 1,061,376 |
| Total added stored parameters | 1,312,231 |
| Full model stored parameters (including legacy components) | 8,493,448 |

The nearest-width solution is derived from an affine parameter formula and
tested against the actual Complete-State online head at multiple dimensions.
Teacher parameters are not part of the matched online-head budget.

## Real MOSI smoke, GPU 5

Seed 66, 32 conversations, 843 valid utterances, original features and optimizer.
Exactly one Adam update plus EMA update. The inherited config contains
`epochs=100` but the smoke does **not** call the full experiment runner.
Follow-up count checks are no-gradient forwards, not extra optimizer updates.

| Check | Result |
|---|---:|
| Valid adjacent transitions at smoke rate .5 | 811 |
| Transitions at rate 0 | 811 |
| Transitions at rate .7 | 811 |
| Emotion loss | 2.844248 |
| Future loss | 0.689168 |
| Total loss | 2.913164 |
| EMA updates | 1 |
| Shared initial state tensors, exactly equal | 144 |

Isolated **unweighted Future loss** gradient L2 (not total-loss gradients):

| Parameter group | L2 |
|---|---:|
| State query | 0.015671 |
| Predictor | 1.213015 |
| OSRAM key projectors | 0.938471 |
| OSRAM value projectors | 0.685829 |
| Observed encoder | 1.658852 |

All gradients and losses finite; Teacher has no gradients. Eval succeeds while
query observer, predictor and Teacher methods are patched to raise errors.
Exact values and inherited config: [SMOKE.json](SMOKE.json).

## Verification

- TDD red: missing `future_state_jepa` constructor argument; trainer rejected
  unsupported `future-state` before implementation.
- 298 relevant tests passed: 17 Future model tests, 24 Future trainer tests,
  plus existing Missing-M3, OSRAM, B2, write-step, retention, Complete-State,
  WSC and per-rate checkpoint-selection tests.
- After strengthening the training compatibility test with nonzero context
  readout weights and matched dropout RNG, all 17 Future model tests passed
  again. Shared emotion logits and gradients are exactly identical.
- Historical OSRAM forward/backward/state/RNG tests received their original
  source fixtures via existing base64 environment hooks (remote mirror has no git).
- Python compilation and `git diff --check` passed.

Run the bounded smoke from the remote repository root:

```bash
CUDA_VISIBLE_DEVICES=5 /data2/yb/reproduction_envs/s0/bin/python3.10 \
  experiments/osram_future_state_20260914/smoke.py
```

## Selection protocol and unresolved questions

`training_objective="future-state"` is supported by config, CLI and trainer.
The experiment entry rejects selection other than `test-oracle-per-rate`:
each seed/rate selects its own best Test epoch. An eight-rate mean never selects
a checkpoint for this mode. Any eventual Test-selected score remains exploratory.

This task proves connectivity, timing, compatibility and finite one-step
operation only. It does not establish long-run stability, target utility,
sample-specific future predictability, absence of representation collapse, or
an F1 improvement. Unlike missing-only Local-state JEPA, complete-condition
training now has a nonzero future objective; trained miss=0 scores need not
match the baseline despite identical inference computation.

No architecture or loss fixes and no full training were launched after smoke.
