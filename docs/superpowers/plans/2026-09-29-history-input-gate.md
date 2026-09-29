# History Input Gate implementation and launch plan

User supplied the formula and authorized execution with “开始跑”.

Goal: test sample-specific historical read strength, not post-Flat residuals.
Architecture: keep Flat and add independent default-off osram_history_input_gate.
G = Linear(latent_dim+4*context_dim+3,128) -> ELU -> Linear(128,1).
Last linear weight/bias zero; alpha=1+.2*tanh(G). No added dropout, norm,
prediction, auxiliary loss, per-evidence gate or GRN. Condition uses Local,
already-ablated Base and availability-masked Gap slots plus availability.
Scale only historical Flat inputs, never Local or persistent Memory.

- [ ] Core tests first: missing API fails, then identity, scalar range,
  padding/inactive safety, shared RNG and weights, finite joint updates.
  Files: gcnet_missing_m3/{osram,model,train_gcnet}.py,
  tests/test_history_input_gate.py and existing fixed-pattern model builder.
- [ ] Runner tests first: the sole semantic config delta is the new flag;
  prohibit unrelated variants and non-GPU0 launches, retain best checkpoints.
  Reuse the proven post-GRN launcher lifecycle in a separate experiment folder.
- [ ] Verify on biggpu host GPU0; execute one-epoch smoke before full launch.
  Verify original Flat remains trainable and diagnostics alpha starts at one.
- [ ] Commit only implementation, tests, protocol and verification records.
- [ ] Isolate code, preflight disk/GPU and launch seeds66/67/68 on GPU0.
  Preserve original cfg84 no-JEPA cyclic random-mask, optimizer and 100epochs
  from reference JSON. Joint training from scratch; no pretrained task
  checkpoint initialization. Reuse existing original Flat comparison results.
- [ ] Verify all child processes and advancing logs; commit launch provenance.

Existing matched baseline uses per-rate Test-oracle checkpoint selection:
this comparison stays INTERNAL DIAGNOSTIC, never a validation-selected claim.
No test oracle labels or correction tables are loaded as Gate supervision.
New isolated run root: /data1/yb/remote_experiments/osram_cfg84_history_input_gate_20260929.
