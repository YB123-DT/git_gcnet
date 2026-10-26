# Frozen Memory history span

Scope: IEMOCAP-6 seed66, held-out fold5 only, inference only. One original Flat
miss0.3 checkpoint is reused for K=0,1,2,4,8,16,32,full at all eight original
GCNet saved-mask rates. No training, tuning, new masks, graph/module changes or
deletion. Existing flat/nested checkpoints and evidence are read-only.

For each target, reconstruct zero-initialized Memory on the last K **valid prior
turns**, using cached original projected observed K/V and the original learned
alpha/beta and joint block_write. Apply target decay, then the original Base/Gap
read using the original target Query and currently observed keys. No current
write enters its own prediction. Padding never counts as history or decay.
This is not a periodically reset full scan or a readout-zeroing approximation.

The original encoder/Local is run once per saved-mask batch. The cached original
Local and Query are unchanged across windows. Classifier replay includes the
original LocalSkip, Flat1600, final Norm and head. All backward slots remain zero.

Unit checks compare to independently trimmed original scans, including interior
padding, conversation isolation, future edits, inactive gaps and RNG. Real-data
full replay verifies Base/Gap/logits/classes; miss0.3 original forward must also
exactly reproduce its protected selected predictions. No optimizer is created.

Report every window on all targets and the same past≥32 cohort, plus every
current modality pattern. ACC/UA/W-F1/CE and corrections/harms relative to full
are retained, not used to choose a new checkpoint. A plateau does not establish
a hard Memory capacity, nor does truncation degradation prove semantic retention.

Runtime: ssh biggpu, cmerc-repro Python, UUID-bound healthy physical GPU0.
Numerical cwd pinned to the accepted source_21e46f2; only this independent audit
package is overlaid. The original miss0.3 checkpoint was Test-oracle selected;
this remains internal frozen inference, not independent test generalization.

```sh
cd /data2/yb/remote_experiments/osram_signature_iemocap_cyclic_20261025/source_21e46f2
PYTHONPATH=ACTUAL_IMMUTABLE_OVERLAY:$PWD \
  /data2/yb/conda_envs/cmerc-repro/bin/python \
  -m experiments.osram_iemocap_history_span_20261026.run \
  --source-commit ACTUAL_OVERLAY_SHA --output NEW_EMPTY_RUN_DIRECTORY
```

Raw per-sample logits/masks remain remote. Only code, small metrics and reports
are committed. Existing historical dataset-length statistics are a separate
descriptive audit and are not substituted for these task-level interventions.
