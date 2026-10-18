# Frozen readout diagnosis: actual launch and first completed seed

INTERNAL DIAGNOSTIC ONLY. No training or model changes.

- Implementation commit `0a968e4`, pushed to `YB123-DT/git_gcnet`,
  branch `feature/ecerc-features`.
- biggpu original cmerc-repro environment, unchanged b4a3138 numerical source
  and original accepted ECERC feature hash.
- GPU1 UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`; diagnostic runs alongside
  existing seed67/68 training on that healthy card. No jobs killed or moved.
- tmux `meld_small_lr_readout_1018`, controller PID2671427.
- Seed66 inference worker PID2671701 has completed successfully.
- Immutable source:
  `/home/qinjinghui/remote_experiments/osram_meld_small_lr_readout_20261018/source_0a968e4`.
- Outputs:
  `/home/qinjinghui/remote_experiments/osram_meld_small_lr_readout_20261018/runs_v1`.
- Logs: experiment root `controller_v1.log` and `runs_v1/queue/logs/D66.attempt1.log`.
- D67/D68 remain pending their existing training's completion. The controller
  waits without holding an inference GPU process, then evaluates each once.

17 new/legacy readout and mask replay tests passed before launch. Actual seed66
replay completed8 rates ×2610 utterances. All8 original Full logit tensors
matched the saved selected logits **exactly** (maximum absolute error0),
including sample IDs, utterance indices, speakers, labels and predictions.
Full readout-only replay versus original Full forward had maximum absolute
error1.90735e-6 and identical classes. Each rate used9 batches/CHM scans;
only the adapter/LayerNorm/head were repeated for the three conditions.
Parameters remained unchanged and fully frozen. No random masks were generated.

## Seed66 eight-rate means

| Readout | ACC | UA | W-F1 | CE |
|---|---:|---:|---:|---:|
| Local-only |60.593870|35.745757|56.604077|1.248916|
| Memory-only (Base+Gap) |41.503831|26.000037|41.419709|6.228478|
| Full |61.786398|38.434973|58.917567|1.194124|

Full minus Local: ACC +1.192529pp, UA +2.689216pp, W-F1 +2.313490pp.
Across8 rate-specific evaluation settings: Local→Full corrections781, harms532,
net249. These counts include repeated utterances across rates and do not mean
249 unique utterances. See per-rate counts and fixed masks in SEED66_SUMMARY.json.

These are input-zeroing responses of the **same trained classifier**, not
separately trained probes. Removing both Local paths can create out-of-training
inputs; Memory-only's low score alone cannot establish poor Memory quality.
Memory-only also preserves original current-input-conditioned queries.
Do not interpret these results as an intrinsic history-information upper bound.

The original Full means61.786398/38.434973/58.917567 reproduce the screened O6
values. Seed67/68 diagnostics have not finished at this record; they were each
17/50 training epochs when seed66 inference was verified. The finite queue will
write combined RESULTS.json/RESULT.md after those source models finish.
