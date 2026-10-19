# IEMOCAP-6 same-cohort Base←T / Gap-T←T audit

INTERNAL DIAGNOSTIC ONLY. No training, no Gate, no model edits. This applies the
completed MELD four-condition analysis to the original IEMOCAP-6 seed66/fold5
anchor from the completed 12-source evidence audit. Not IEMOCAP-4 or five-fold
aggregate. Do not replace the checkpoint per rate.

- Original anchor miss0.3, epoch69, SHA256
  `6a7968ade42513ee868e6dc82f306274a2e7bd53d1943c05f00c1a75af530fd0`.
- Reuse original `gcnet-random-mask-v1` exact saved masks at random0.0--0.7.
- Select current Text missing AND strictly-past Text ever seen, without labels.
- Replay original / remove Base←T / remove Gap-T←T / remove both on identical
  targets within each rate. All other evidence, Local Skip, Query, Memory and
  classifier remain unchanged. Only forward512 source contributions are removed.
- Require original and both single-removal logits to match the old audit cache
  exactly on selected targets. Nonselected predictions and RNG remain unchanged.
- Report ACC, six-class UA, W-F1, CE, corrections/harms and conditional effects.
  Empty cohorts are N/A. A rate subset score is not a whole-test benchmark.

Reuse the existing implementation with the explicit dataset switch:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_meld_base_text_gap_text_20261019.run \
  --dataset IEMOCAPSix \
  --gpu-uuid GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 \
  --source-commit IMPLEMENTATION_SHA \
  --output /home/qinjinghui/remote_experiments/osram_iemocap_base_text_gap_text_20261019/runs_v1/IEMOCAPSix_seed66_fold5
```

Only the audit's class count, case lookup and report title become configurable;
MELD remains the default. No changes to OSRAM, data sampling or model parameters.
Finite-difference effects describe frozen readout behavior and may be off-manifold;
they do not establish semantic causal attribution or imply a useful learned Gate.
