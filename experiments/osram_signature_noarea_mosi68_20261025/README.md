# MOSI seed68: matched retrained Signature without directional area

Finite user-requested transfer of the completed IEMOCAP68 ablation. Only eight
MOSI68 rate-specific frozen-parent branches are newly trained. Reuse the already
completed full Signature and Flat; no new baseline, other seeds, datasets or HPO.

The shared `NoAreaSignatureBlock` retains first level S1, symmetric second level
`0.5*S1 outer S1`, endpoint/time coordinates, counts, Local and availability.
Only antisymmetric second-level oriented area is removed during TRAIN and TEST.
Network dimensions, state keys, zero-initialized outlet, random initialization
and 281856 trainable parameters remain the same. Equal nominal parameter counts
do not imply equal effective degrees of freedom. This is retraining, not a frozen
feature intervention or a test of whether all recent history is useful.

## Matched protocol

- biggpu, cmerc-repro, seed68, original official split and features.
- Original same-rate Flat BEST/checkpoint, exact test masks and predictions.
- Encoder/CHM/Flat/LocalSkip/Norm/task head frozen; only Signature trains.
- Original GCNet-random-mask-v1, cyclic TRAIN across 0.0–0.7, own-rate-only TEST.
- 150 epochs, batch32, Adam, WD1e-5, original gradient clipping and MSE including
  gold0. Original LR expression: 3e-4 for100, 3e-4/10 for50.
- Earliest strict per-rate Test-best WF1; ACC from the same prediction/mask.
- Per-epoch training mask hashes, samples, update counts, rates and LR must equal
  the completed full Signature run. Initial block SHA and parent/mask must match.
- Full block/Adam/RNG/epoch/history/best state for explicit resume; saved selected
  prediction and frozen-parent replay checks. No original artifacts overwritten.

This remains INTERNAL TEST-ORACLE exploration. The eight-rate mean combines eight
parent/branch pairs; it is not one deployable checkpoint or three-seed evidence.
Report eight-rate and high-missing0.5–0.7 ACC/WF1. MSE is retained for provenance,
not used to justify the directional mechanism.

## Immutable numerical source and finite execution

Reuse numerical source `4d5aa09d999b49d7194f72f6bf75c8ab75edbef1` at
`/data2/yb/remote_experiments/osram_signature_cyclic68_20261025/source_4d5aa09`.
Original completed full Signature reference:
`/data2/yb/remote_experiments/osram_signature_cyclic68_20261025/execution_v1/train/RATE`.
New experiment lives in an independent namespace overlay, never edited into that
original source. Overlay includes the shared NoArea block and read-only scheduling
and summarization helpers from later experiments; it does not replace numerical
model/training files. All deployed file hashes are recorded in the launch receipt.

One real two-update smoke precedes eight formal jobs. Smoke replays original
two-update training under the same RNG, rather than comparing truncated smoke
masks to full-epoch historical hashes. Original CPU-backend gradient-test caveat
remains scoped to the inherited unit test; no CUDA backend setting is modified.

Use healthy physical GPU0/1/2 only, UUID-bound, at most3workers/card by default,
8total. Stagger startup and gate on smoke VRAM/RAM plus disk/system reserves. Do
not signal unrelated tasks. Persistent tmux/controller and detached children
record PIDs/logs/status, support full-state recovery and automatically summarize
only after all eight jobs validate. Failures are retained and stop new admission.

```bash
PYTHONPATH=/path/to/immutable/overlay \
 /data2/yb/conda_envs/cmerc-repro/bin/python -u \
 -m experiments.osram_signature_noarea_mosi68_20261025.scheduler \
 --root /path/to/new/execution_v1 --source-commit ACTUAL_IMPLEMENTATION_SHA \
 --gpu-map 0=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45,1=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049,2=GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
 --per-gpu 3
```

Interpret results conservatively: no-area matching full would weaken a claim
that directional second-level geometry explains the gain. A full advantage on
one seed would support this particular control, not demonstrate novelty or prove
that all Signature benefits arise from directional history.
