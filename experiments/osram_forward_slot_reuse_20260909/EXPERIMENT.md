# Forward-slot reuse / Past-context duplication

Approved minimal causal control: replace [past; zero] with [past; past],
for both Base and each hard-masked target Gap. Same single forward memory scan,
no backward scan, no future input, no detach, no new parameters or projections.
Independent information/rank does not increase. Fusion input distribution,
amplitude and gradient paths change; improvement would not prove capacity restored.

MOSI and IEMOCAP4/6, seeds66–70, cyclic eight rates, 100 epochs. Inherit each
completed forward-only config, changing only osram_forward_slot_reuse=True.
Bidirectional and zero-slot controls are inherited, never retrained.
Five concurrent lanes: GPU0 seeds66/67/68, GPU1 seeds69/70. Each lane runs
MOSI then IEMOCAP4 then IEMOCAP6 (15 new runs total).

Report independent per-rate Test-oracle maxima with all per-epoch histories;
INTERNAL DIAGNOSTIC ONLY, NOT A FORMAL PAPER RESULT. IEMOCAP Session5 is
both validation and test under inherited official loader.

Required verification: exact duplicate Base/Gap slots, unchanged past half,
future perturbation invariance and existing related regression tests.
No new decay, gate, loss, feature extractor, MMoE, or memory update changes.
Diagnostic memory_frobenius_norm is an existing readout-norm proxy; do not
interpret it as actual memory Frobenius norm or collapse evidence.

Remote output: /data2/yb/remote_experiments/osram_forward_slot_reuse_20260909
