# Signature extension: verified launch, not final scores

Status: **formal training running**. INTERNAL TEST-ORACLE SCREEN.

Source: `c796c48974d4e40c4777264347f3bead0a14b20d`.
Server: biggpu; Python `/data2/yb/conda_envs/cmerc-repro/bin/python`.
Snapshot: `/data2/yb/remote_experiments/osram_signature_stage2_extension_20261025/source_c796c48`.
Outputs: `/data2/yb/remote_experiments/osram_signature_stage2_extension_20261025/execution_v2`.
tmux: `signature_extension_v2_20261025`; controller PID3949611.

- MOSI: reuse all eight completed Signature seed66 results; add67/68×8rates.
- MOSEI: seed66×8rates only, latest repaired-RNG Flat1600 parent (mean W-F1
  83.477635). Never substitute the older unrepaired83.288796 parent score.
- Per-run parent is the accepted same-rate BEST, frozen in eval mode; only
  unchanged SignatureBlock trains. Each run trains/tests only its own rate.
- Stage2 LR3e-4,Adam WD1e-5,batch32. MOSI150epochs with LR/10 after100;
  MOSEI60epochs constant branch LR. Original task MSE includes gold0.
- Original GCNet random-mask-v1,exact accepted test masks,per-rate Test-best W-F1.

37 tests passed in real remote cmerc-repro. Separate real MOSI67.7 and
MOSEI66.7 full-batch two-update smokes passed; both original saved parent
predictions replayed with maximum absolute error0. Frozen-parent integrity,
finite/nonzero branch gradients and selected correction checkpoint replay passed.
Measured smoke peaks: MOSI371.82MiB; MOSEI285.25MiB (allocated, not total process).

First12 formal processes observed alive: MOSEI rates0–0.3 PID3950894–3950897
on GPU0; MOSEI0.4–0.7 PID3950898–3950901 on GPU1; MOSI67 rates0–0.3
PID3950902–3950905 on GPU3. Remaining12 MOSI jobs are persistent queued work,
admitted on freed slots. Max4new processes/GPU,max12global; GPU4 forbidden.

The first source e47c60f/execution_v1 failed BEFORE formal training on an
unreviewed registry guard. It is preserved, not mixed into results. Exact
hash/AST review identified only27 inactive Nested registry additions; the
corrected version accepts only that reviewed hash pair with meaningful_block
none and retains mandatory prediction replay. No model/loss/mask change.

The finite controller will collect all32 conditions,including reused MOSI66,
and write per-seed/per-rate means,high-missing results and sample SD across
three MOSI seeds. One MOSEI seed has no seed SD. Final scores are pending;
smoke scores are not performance evidence. No extra models,seeds or HPO queued.
