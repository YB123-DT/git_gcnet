# MC-FWC implementation and verification

Engineering verification, not evidence of performance improvement. Baseline:
`feature/osram-uniform-forced-text`, `c838e14`. See `AUDIT.md` for the pre-edit
review of the five implementation files and eight prior experiment families.

## Scope and data flow

No new trainable parameters: every mode has 13,509,793 parameters. Original
ObservedSetEncoder, Local/Skip, queries/residualization, simultaneous Block Delta
Write, retention, Flat, normalization and task head remain intact. Default
forward arguments take the original path; old checkpoints load strictly.

1. Generate a training trajectory using private, label-independent RNG.
2. Sample legal source/target pairs within the same valid conversation, horizon4.
3. Keep: original forward, capturing write-exclusive K/V clones **consumed by
   block_write**, not a disconnected stack constructed after the forward.
4. Omit: replay the same features, input availability and CPU/CUDA random states
   under no_grad; exclude exactly one observed source column from its joint
   write. All later writes follow the original availability. Current Local and
   Query still see that observed modality. Restore post-Keep RNG afterward.
5. For each pair, compute `u=loss_omit-loss_keep` and
   `tau*softplus((loss_keep-stopgrad(loss_omit))/tau)`. The first derivative is
   `sigmoid(-u/tau)*grad(loss_keep)`. This emphasizes harmful/ineffective writes;
   it is not a positive-utility reward, new label supervision, or missing-feature
   reconstruction. Joint-solve coupling and subsequent state changes are included.
6. E4: take gradients at real source write nodes, mask to the selected source
   time/conversation/modality, detach these cotangents, and VJP only into original
   K/V projector parameters. Sum with ordinary task gradients, clip once, step
   once. Shared encoder/query/memory/readout parameters receive only task gradients.

Pairs are averaged within eligible conversation and then across conversations.
Independent conversations can share one Omit replay; two pairs from the same
conversation occupy different rounds. Source masks are applied before gradient
aggregation, avoiding union-of-sources credit leakage.

## Files and interfaces

- `gcnet_missing_m3/osram.py`, `model.py`: optional `write_mask_override` and
  `write_node_collector`; no change to block_write mathematics or state_dict.
- `fwc_sampling.py`: trajectory and legal pair sampling, coverage diagnostics.
- `mc_fwc.py`: paired RNG replay, pair loss, source VJP, E0–E4/shuffled backward.
- `fwc_evaluation.py`: frozen write-utility assessment and grouped statistics.
- `train_mc_fwc.py`: isolated runner reusing original model/loss/mask/loader,
  validation selection, full recovery and four-family evaluation.
- `tests/test_mc_fwc*.py`, `test_fwc_sampling.py`, `test_fwc_evaluation.py`:
  invariance, mask, causal, gradient, replay, recovery and control tests.
- This directory: 18 fixed config files (15 main plus 3 optional shuffled),
  `verify_real_batch.py`, bounded `run_suite.py`, audit and verification artifacts.

Original `train_gcnet.py`, `mixed_rate.py`, `paired_views.py` are reused unchanged.
Unrelated pre-existing `gcnet/model.py` edits are not part of this work or deployment.

## Verification actually performed

69 targeted tests passed on biggpu, including existing OSRAM/write-step tests;
one upstream PyG deprecation warning. No targeted failures remain. This is not a
claim that every historical test in the repository was executed.

Verified default-off forward/RNG/task-gradient/update equivalence; Block Write
column permutation invariance; unchanged Local/Query and past outputs; subset
and nonempty write masks; detached Omit; finite/nonzero warmed source gradients;
writer-only extra gradients; ordinary full-model gradients; legal pair sampling;
matched E2/E3/E4 sampling; empty-pair fallback; unchanged inference; strict legacy
checkpoint restore; complete optimizer/RNG recovery; add/clip/step ordering;
honest zero-initialized first-step coverage. Warmed grouped VJP agrees with
individual pair VJPs. Legacy seed66 miss=.7 checkpoint loaded with all keys matched.

Actual MOSI batch: 32 conversations, 843 valid utterances. Three training steps
plus recovery/equivalence checks passed. For E4, extra writer gradient norms were
0, 13.692, 85.171: step1 has a connected but zero-gradient path because the
original adapter starts at zero; subsequent steps have nonzero credit. All
non-writer gradients exactly match the normal-task reference, and the next
optimizer update after full save/restore matches exactly. See
`REAL_BATCH_VERIFY.json`; recovery weights stay remote, outside Git.

Five-mode smoke (3 optimizer steps each, GPU1 V10032GB):

| Mode | Seconds | Peak allocated GiB |
|---|---:|---:|
| E0 | 3.538 | 1.201 |
| E1 | 5.315 | 1.202 |
| E2 | 6.770 | 1.320 |
| E3 | 7.771 | 1.322 |
| E4 | 7.043 | 1.320 |

All initial parameter hashes match; E1–E4 mask/pair sequences match.
Early MSE spikes occur in E0 as well, so finite smoke loss is not a convergence
or improvement claim. These timings exclude full experiment validation/test cost.
E2/E3/E4 match Omit forward counts, not exact backward time or gradient norm.

## Fixed formal protocol

MOSI seeds66/67/68, 150 epochs from scratch, batch32, Adam, weight decay1e-5,
clip1, 100 epochs at1e-3 then50 at1e-4. Two optimizer steps/epoch, 300/run.
Official splits: 52/10/31 conversations and 1284/229/686 utterances.
Features: wav2vec-large-c-UTT / deberta-large-4-UTT / manet_UTT, dimensions
512/1024/1024. Label/features hashes and split identities are saved per run.

One checkpoint per run, chosen by eight-rate mean validation W-F1. Test is
evaluated only after training at this checkpoint. No per-rate Test-oracle
selection and no comparison treating historical oracle means as deployable scores.

| Mode | Trajectory | Extra gradient |
|---|---|---|
| E0 | Original cyclic random | None |
| E1 | .5 random/.25 transition/.25 persistent | None |
| E2 | Same mixed masks/pairs | Uniform .5 future loss, source-writer VJP |
| E3 | Same mixed masks/pairs | Pair utility softplus, all related parameters |
| E4 | Same mixed masks/pairs | Pair utility softplus, source-writer VJP |

Fixed H4, one pair/conversation, tau1, lambda.1. None is claimed optimal.
E2's constant .5 matches the neutral-utility weight. Optional shuffled mode is
implemented but not included in the initial 15-run suite; matched singleton
shuffle groups stay unchanged and are reported.

Run from the immutable code snapshot on biggpu, with the existing Python env:

```bash
python -m gcnet_missing_m3.train_mc_fwc --mode E0 --seed 66 --epochs 150 --device cuda --dataset-root /data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset --output /absolute/new/E0_seed66
python -m gcnet_missing_m3.train_mc_fwc --mode E1 --seed 66 --epochs 150 --device cuda --dataset-root /data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset --output /absolute/new/E1_seed66
python -m gcnet_missing_m3.train_mc_fwc --mode E2 --seed 66 --epochs 150 --device cuda --dataset-root /data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset --output /absolute/new/E2_seed66
python -m gcnet_missing_m3.train_mc_fwc --mode E3 --seed 66 --epochs 150 --device cuda --dataset-root /data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset --output /absolute/new/E3_seed66
python -m gcnet_missing_m3.train_mc_fwc --mode E4 --seed 66 --epochs 150 --device cuda --dataset-root /data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset --output /absolute/new/E4_seed66
```

Repeat seeds67/68, or use `run_suite.py` for the finite 15-job queue. The launcher
checks host GPU UUID/free memory, limits concurrency, prohibits GPU4/7, refuses
duplicate submission, records PIDs/commands and preserves failures. `--resume`
restores the runner's complete last.pt; it is not weight-only continuation.

Outputs: config/provenance, last.pt, best.pt, history.json; after completion,
metrics.json and random/persistent/transition prediction records including write
utility. Report random ACC/W-F1 by rate, 8-rate/high means, six persistent
patterns, six transition destinations with post-change lags1–4, utility by source,
missing target and lag, and A-only polarity ratio/negative recall/distribution.
Suite SUMMARY/RESULT are generated only once all15 finish and pairing checks pass.

## Remaining scientific risks and success criteria

Persistent A/T/V-only conversations cannot provide legal omission pairs while
retaining one observed source; their ordinary task loss remains active. Persistent
same-source/target modality pairs also cannot exist. This coverage limitation is
explicit, not silently repaired with synthetic observations.

Potential shortcuts remain: ignoring history, boundary-only bias correction,
ordinary future reweighting, extra-gradient magnitude, and uninformative history.
Monitor task performance and signed write utility together. Three-seed E4 gains
over E0 alone are insufficient: compare E1 (trajectory), E2 (weights), E3 (gradient
restriction), and eventually shuffled utility (sample specificity). E4 should
show reproducible task benefits without A-only collapse and consistent paired
write-utility/recall evidence. Tiny validation size, fixed tau saturation,
gradient clipping and unequal backward costs require reporting. No claim that
history must recover future Text, or that implementation success establishes a
second scientific contribution.
