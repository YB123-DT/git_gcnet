# Direct XCA and Neural Production input transformations

INTERNAL DIAGNOSTIC ONLY. Status: **running**, not completed; no final scores yet.

Two approved MOSI seed66,100epoch from-scratch runs. Original cfg84 causal
no-JEPA, cyclic0.0–0.7, original Flat/output1600, original task head/MSE,
Adam1e-3 constant, original batch32 and per-rate Test-oracle BEST retained.
No Memory/query/write change, auxiliary loss, completion, Gate or warmup.

| Variant | Evidence input | Added module parameters | Historical residual reference mean8/high W-F1% |
|---|---|---:|---:|
| m28_xcit_xca_direct | replaces adapter Local/Base/Gap with decoded role outputs |115972|81.041519 /76.257088|
| neural_production_direct | replaces Base/Gap only, retains Local |175168|80.981287 /76.475237|

Original Flat reference:81.068095/76.352251. Original XCA module170052params,
so new per-role interface is not a parameter-matched removal-only ablation.
Original NPS module175168params, matched to direct variant; output bridges use
ordinary Linear initialization instead of zero initialization.

Both use `head(LN(skip(original L)+adapter([L',B',G'])))`. There is no external
`L+decoded_L` or `C+decoded_C` bypass. XCA retains five typed64d role updates
before pooling and decodes Local64→256/shared Memory64→512. NPS preserves four
rules, two internal production-update steps, tokenizer and eight head bridges.
Its internal additive rule-state update is intentionally unchanged.
First/empty-history rows retain original inputs via existing adapter wrapper.
Backward halves untouched; inactive Gap and padding hidden safely zero.
Original task head bias means padded logits need not be zero; original umask
continues to exclude padding from loss/metrics. Do not modify the head for this.

## Verification

- Tests written/run RED before implementation; direct variants were unregistered.
- Local12passed/1skipped +9subtests: direct semantics, inactive NaN masking,
  finite gradients and XCA core update, NPS default RNG/parameter parity,
  old XCA arithmetic and original40-method config accounting.
- Remote CPU5passed including full model: original shared parameter initialization
  and Local Skip unchanged; finite predictions and strictly zero padding hidden.
- Initial full-model assertion incorrectly demanded zero padded task logits;
  corrected assertion checks hidden zero and logits equal the unchanged head bias.
  No model code changed for this test correction, and sealed training source was
  not edited. Corrected test executed separately under remote `checks` directory.
- Python compilation passed. No extra GPU smoke and no baseline rerun.

## Launch

Server biggpu physicalGPU7, UUID `GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`.
GPU4 forbidden. Prelaunch free15767MiB; after two starts12215MiB,100%utilization.
Source commit `2fe67e7`, full tracked archive and SHA256 source snapshot sealed.
Root `/data1/yb/remote_experiments/osram_xca_direct_20261010`.
Dedicated tmux socket `gcnet_xca_nps_direct`, session `direct_seed66`.
Dispatcher PID3716697; XCA PID3716782; NPS PID3716783.
Both processes verified live and feature loading recorded. Commands in LAUNCH.json.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
experiments.osram_xca_direct_20261010.dispatch \
--root /data1/yb/remote_experiments/osram_xca_direct_20261010 \
--reference /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/flat_seed66/config.json \
--data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
```

Runs/logs: `runs/{m28_xcit_xca_direct,neural_production_direct}/seed_66/train.log`.
Each retains final config, source/data/version provenance, full recovery checkpoint,
eight BEST checkpoints and original-mask predictions. Dispatcher records failures
without automatically retrying. Completion requires100epochs and artifact/hash
verification. No additional seeds or sweeps are authorized by this report.
