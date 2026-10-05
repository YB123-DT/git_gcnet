# Four-source audit and transfer boundaries

Scope: original PDFs and public implementation audit only. No module implementation
or training launch; existing experiment queues remain unchanged. This is not
performance evidence. The baseline is the current OSRAM block_write/_scan and Flat.

| Method | Mechanism retained | Proposed OSRAM boundary | Source finding |
|---|---|---|---|
| R02 DeltaProduct | Multiple noncommuting delta/Householder transformations per observation | Replace state transition; derive internal substeps from the observed set, not an arbitrary A/T/V temporal order; preserve read-before-write | Author LM writes before reading; nh/forget defaults differ between entry points |
| R03 MesaNet | Regularized regression on cumulative historical key/value sufficient statistics | Replace historical statistics, not merely the solve call; read past real observations before writing the current event | Public FLA kernels differ from the paper's adaptive CG initialization/stopping |
| R12 Rank-N-Contrast | Representation ranking by continuous TRAIN-label distances | Preserve a single causal history; paper supports one-view/no-augmentation; retain Flat and task head | Default author recipe is two-view and two-stage; joint one-stage MSE+RNC is a transfer, not the full original recipe |
| R18 LUPI Dropout | Training-only privileged features parameterize multiplicative noise | Privileged inputs enter only the noise condition, never Local content or persistent Memory; no privileged features at test | Paper and author CNN regularizers differ; a shared-conv gradient stop does not freeze the whole privileged tower |

R02/R03 change the memory algorithm, not a lightweight readout. R12 changes the
representation objective and potentially training stages; SupCR is not merged
with RNC. R18 changes training noise and access paths, not missing-latent completion.

Correction: the default RNC scripts use two views, but PDF Appendix G.3 explicitly
examines a single-view/no-augmentation variant. The default recipe is not a hard
requirement of every paper-supported variant. Label the proposed one-stage transfer
separately from either recipe.

If implementation is later requested, keep four separate switches/configs and
start with seed66 independently, not a combined model. Preserve datasets, features,
splits, availability and task head; disclose necessary objective/stage changes.
Minimum checks: R02 set permutation and causal prefixes; R03 exact small-system
solve reference; R12 ties, diagonal removal and valid utterances; R18 test-time
privileged-input exclusion. No inferred missing features may become memory writes.
Parameter counts, GPU peaks and throughput have not been measured.

Detailed formula-to-code mappings, pinned commits, PDF versions/hashes and limits
are in the four individual reports. paper_bank.json records canonical metadata.
