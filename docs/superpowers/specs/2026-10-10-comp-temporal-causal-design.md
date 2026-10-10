# ComP Utterance-Level Temporal-Causal Variant

## Objective

Create an isolated causal variant of the official ComP implementation and reproduce CMU-MOSI with seed 66 at missing rates 0.0 through 0.7.

The causal definition is utterance-level: prediction at utterance index `t` may use the complete audio, text, and visual features of the current utterance and all earlier utterances, but it must not depend on utterances with index greater than `t`.

## Source and Isolation

- Upstream implementation: `/data2/yb/paper/05_reproduction/external_repos/ComP`
- Upstream commit: `28192d3a5683543d7383e40898f9a98d1f114a08`
- New implementation directory: `/data2/yb/paper/05_reproduction/external_repos/ComP_causal`
- New result directory: `/data2/yb/paper/05_reproduction/runs/ComP_causal/seed66/CMUMOSI`
- The original ComP checkout and its completed results remain untouched.

The causal directory will be created as an isolated Git worktree or equivalent repository checkout rooted at the exact upstream commit. All causal changes will live only in that directory.

## Preserved Behavior

To isolate temporal causality as the experimental variable, preserve the official implementation's:

- feature files and train/test splits;
- 10 prototypes per modality;
- 256-dimensional hidden/prototype representation;
- 16-dimensional prompts;
- two prompt/knowledge-propagation stages;
- `lambda=0.3` prompt interpolation;
- cross-sample prototype sharing inside a mini-batch;
- two-stage 150/150 epoch training schedule;
- coordinator, losses, optimizer, batch size, and all other hyperparameters;
- random missing-mask generation and official test-epoch selection protocol.

The last item is retained only for direct comparability. It remains a known test-selection limitation and will be stated in the result report.

## Causal Self-Attention

Every modality-specific attention matrix will combine two masks:

1. the existing modality-availability/key-padding mask; and
2. a lower-triangular temporal mask.

For query position `t`, valid keys satisfy both `tau <= t` and modality availability at `tau`. No query may attend to a future key.

## Causal Prototype Generation

The original `proj_n` MLP compresses the complete sequence dimension into prototypes, so it exposes every time step to future utterances even if self-attention is made causal.

The causal implementation will preserve the same `proj_n.fc1` and `proj_n.fc2` parameters while evaluating the first linear projection as a prefix cumulative sum. For modality feature `z` and first-layer weight `W`:

\[
h_{t,c,d}=b_c+\sum_{\tau=0}^{t}W_{c,\tau}z_{\tau,d}.
\]

GELU, dropout, and the second prototype projection are then applied independently at each `t`. This produces a prefix prototype bank:

\[
P_t\in\mathbb{R}^{C\times D},\qquad C=10,
\]

where `P_t` depends only on positions `0..t`.

This construction is mathematically equivalent to applying the original prototype MLP to an input whose positions after `t` are zero, while avoiding an `O(S)` sequence of separate forward calls.

At each time index, the implementation will retain the original batch sharing behavior: queries at `t` may read the prefix prototype banks at `t` from other samples in the same mini-batch. This is batch-dependent/transductive but does not violate the agreed temporal-causality definition because no bank contains positions after `t`.

## Progressive Prompting

Both prompt-generation blocks will consume the time-indexed prefix prototype bank. The existing cross-modal prompt propagation and block-to-block interpolation remain unchanged:

\[
p^{(2)}=\lambda p_{\mathrm{delivered}}^{(1)}+(1-\lambda)p_{\mathrm{regenerated}}^{(1)}.
\]

No new recurrent prompt state is introduced. The change is limited to the temporal visibility of the prototypes that produce each prompt.

## Verification

Before formal training, the implementation must pass:

1. **Attention causality test:** changing future positions must not change attention outputs at or before `t`.
2. **Prompt causality test:** changing future positions must not change prototypes or prompts at or before `t`.
3. **End-to-end causality test:** in evaluation mode, perturbing all future modality features must leave model predictions at or before `t` unchanged within numerical tolerance.
4. **Final-prefix equivalence test:** at the final valid position, causal prototype generation must match the original full-sequence prototype generation in evaluation mode.
5. **Missing-mask test:** future perturbation invariance must also hold with incomplete modalities.
6. **Training smoke test:** one short CMU-MOSI run must complete forward, backward, optimizer update, and evaluation without NaN/OOM.

## Formal Reproduction

After verification, train eight independent CMU-MOSI models with:

- seed 66;
- missing rates `0.0, 0.1, ..., 0.7`;
- 300 epochs per missing rate;
- the same environment, features, and official hyperparameters used by the completed non-causal reproduction.

Each run receives an isolated log/result directory. Completed artifacts must include the effective command/configuration, full log, final official-protocol metrics, and a comparison against the original seed-66 ComP results.

## Non-Goals

This experiment will not simultaneously:

- remove cross-batch prototype sharing;
- repair the gradient-modulator name mismatch;
- repair historical checkpoint restoration;
- change the official test-set checkpoint-selection protocol;
- fix unrelated private-branch or ablation-switch issues;
- implement frame-level streaming inside an utterance.

Those changes would create additional experimental variables and require separate ablations.

## Acceptance Criteria

- The original repository and result files are unchanged.
- All five causal tests and the training smoke test pass.
- All eight seed-66 CMU-MOSI missing-rate runs reach epoch 299 without runtime errors.
- The final report lists ACC/F1 for every missing rate, their eight-rate mean, and deltas from the original non-causal seed-66 results.
- The report explicitly labels results as utterance-level causal and notes that official random-mask/test-peak evaluation behavior remains unchanged.
