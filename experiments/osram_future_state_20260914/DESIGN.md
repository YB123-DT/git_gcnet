# Future-State JEPA implementation plan

Goal: predict the next complete-view Local state from the current post-write
causal memory; do not complete modalities or change emotion inference.

User-approved architecture: causal OSRAM, write step .6, mean/Flat. Add only an
optional post-write observer. Pass a non-detached clone to protect persistent
memory from callback mutation, ignore its return value. Store compressed states
only, not a history of full memory tensors. Default observer is None.

## Execution checklist

- [x] Write failing `tests/test_future_state.py`: default compatibility, shared
  RNG/initialization, causal post-write capture, time shift, padding, gradients,
  EMA and eval bypass. Run on existing remote Python (not local Python).
- [x] Add `gcnet_missing_m3/future_state.py`: H learned normalized queries,
  LN/Linear/GELU/Linear, automatically nearest parameter-matched hidden width,
  frozen EMA copies of full encoder and Local path, adjacent-transition loss.
- [x] Add optional observer in `osram.py`, dispatch/EMA/loss in `model.py`.
  Reject all incompatible completion/ablation modes. Preserve old state keys.
- [x] Integrate `future-state` config/CLI/training metrics in `train_gcnet.py`
  and test lifecycle in `tests/test_future_state_trainer.py` (independent lane).
- [x] Run new and existing OSRAM/state/retention tests. Run one real MOSI batch
  smoke with isolated future-loss gradients and one optimizer/EMA update.
- [x] Document parameter budget, evidence, limitations; diff check and commit.

For dimension d, prior hidden h, the Complete-State online-head budget is
`2*h + 8*16 + (h+16+1)*d + (d+1)*d` (no EMA parameters).
For c=H*dv, new width w has `H*dk + 2*c + d + w*(c+d+1)` parameters.
Choose the nearest positive integer width analytically using floor/ceil, ties
to smaller width. No hardcoded 250872 or default-only width.

Supervision mask is `valid[:-1] & valid[1:]` within each batch column.
Teacher sees full current utterance only, targets indexed `[1:]`; state indexed
`[:-1]`. Last utterance and padding have no target; complete and incomplete
conditions have identical transition counts for identical lengths.

Inference disables observer, query, predictor and Teacher entirely. Training
updates the shared backbone normally; "unchanged backbone" does not mean frozen.
EMA updates only after optimizer. New objective weight .1, SmoothL1 over
parameter-free LayerNorm outputs. No extra objectives or automated training.

Experiment policy: no full run in this task. Future results retain independent
best Test epoch per seed/rate, internal diagnostic only; never select by an
8-rate average. Existing results remain untouched.
