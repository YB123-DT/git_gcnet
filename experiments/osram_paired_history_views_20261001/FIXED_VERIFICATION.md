# Six-pattern evaluation verification

Completed on biggpu hostGPU6 (UUID recorded in fixed_results.json), using existing
paired-view checkpoints and the original `_evaluate_pattern` implementation.
No training; no new checkpoint selection. Source checkpoints loaded strictly,
including training-only projector; its inference call counter stayed0.
All twelve A/B pattern evaluations used686 utterances,656 nonzero sentiment labels.
Model state and checkpoint hashes stayed unchanged. Flat uses existing seed66
fixed-pattern scores and corresponding checkpoint rate/epoch, not the three- or
five-seed aggregate. Raw full metric outputs and hashes are in fixed_results.json.

Remote command (CUDA_VISIBLE_DEVICES=6, OMP_NUM_THREADS=2) from isolated code at
`/data1/yb/remote_experiments/osram_paired_text_drift_20261002/code`:

```
/data2/yb/reproduction_workspace/envs/s0/bin/python experiments/osram_paired_history_views_20261001/fixed_patterns.py --output /data1/yb/remote_experiments/osram_paired_text_drift_20261002/fixed_patterns.json
```

Process exited0 and output status complete with12 unique records. The standalone
runner never invokes the legacy multi-GPU launcher (which contains outdated GPU
defaults). GPU4 was not used. Analysis checks all pattern/rate mappings against
the original Flat CSV; full evaluation metrics are not anchor-subset scores.

Interpretation: task-only A improves A-only/V-only/AV but decreases Text-present
scores. B has a pronounced Audio-only decline; it does not uniformly hurt every
combination (T-only and V-only remain above Flat). B's lower history-deletion flip
rate therefore cannot be treated as evidence of improved persistent-missing
performance. Single seed and different selected epochs limit causal conclusions;
no claim that InfoNCE universally fails or that larger scalar loss caused this.
