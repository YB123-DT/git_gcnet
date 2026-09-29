# Optional post-Flat GRN: implementation and verification

User-approved architecture: retain osram_readout_fusion=flat, introduce an
independent osram_post_grn flag (default False). x is the original normalized
Flat hidden. c is Local, ablated Base, three fixed masked/ablated Gap slots,
and availability. Sanitize inactive/padded values with torch.where before
affine operations. Only c gets LayerNorm. Hidden width128; Linear2 maps128
to128; use the existing backbone dropout rate. Gate/value output the actual
OSRAM output_dim. Value weight/bias zero-init, gate bias zero, all other
layers normal initialization. No post-add normalization or parameter freeze.

- [x] Add core, flag and tests in osram.py and tests/test_osram_post_grn.py.
  Check off-path unchanged, initial equality, mask and ablation isolation,
  finite multi-step gradients and parameter updates.
- [x] Add config/model/CLI propagation and fixed-pattern model builder flag.
  Reject accidental non-Flat/JEPA/completion/persistent/query combinations;
  keep default and historical checkpoint behavior unchanged.
- [x] Record valid-token-weighted gate mean, saturation fraction (<=.05 or
  >=.95), mean per-token gated residual / flat hidden L2 norm ratio.
- [x] Run remote isolated regression suite and small CUDA verification on
  existing model server biggpu, host GPU0 only (GPU4 forbidden).
- [x] Record source snapshot, config, environment and verification outcomes.

This is module implementation, not authorization for full100-epoch training
or a new Test-oracle search. Preserve existing checkpoints and dirty edits.
Actual worktree branch remains feature/osram-uniform-forced-text, not its
historical directory label. Do not switch branches.
