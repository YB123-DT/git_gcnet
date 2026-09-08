# Forward-only OSRAM memory diagnostic

User-approved comparison: inherited bidirectional H8/32/32/output700 versus
forward-only memory, with reverse context slots zero-filled. Same parameter
shapes, local/query current input, read-before-write, and no prefix rescans.
No other architecture, loss, optimizer, feature or mask changes.

CMU-MOSI, seeds 66–70, 100 epochs, cyclic rates 0.0–0.7, batch32, LR1e-3.
Reference: experiments/osram_heads8_out700_20260906 (no baseline retraining).
Configuration inherited from the completed query experiment, restoring explicit
query availability=True and disabling only bidirectional memory.

Report per-rate Test-oracle maxima independently, as requested. Retain full
epoch histories and eight-rate-mean-selected checkpoint metadata too.
INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.

Verification: future perturbations must not affect earlier hidden/base/gap;
reverse slots zero; historical forward read nonzero; related regression tests.
Parameter count is unchanged, although zero slots remove effective context capacity.
This is not a parameter-capacity-matched replacement architecture.

Remote output: /data2/yb/remote_experiments/osram_forward_only_mosi_20260908
Launch distribution: seeds66/67/68 GPU0, seeds69/70 GPU1 (3+2 concurrent).
