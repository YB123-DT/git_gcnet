# Unified cmerc Flat baselines: running

INTERNAL TEST-ORACLE SCREEN. No completed multi-seed performance claim yet.

User's latest correction applies constant1e-4 ONLY to MOSEI. MOSI keeps the
accepted3e-4 for100epochs and3e-4/10 for50epochs. MOSI66's completed cmerc result
is reused; exactly five new runs were actually launched,not six.

All5processes verified running on physicalGPU7 at launch+65seconds. MOSI67/68
reached epoch4/150; all5passed data/split/seed/initial-parameter/exact-test-mask
checks and recorded cmerc-repro as the real executable. MOSEI66/67/68 were inside
their first full training epoch at that check,not still in unverified preflight.
Full epoch completion for MOSEI is not yet claimed by this launch record.

Both MOSI first epochs:LR[.0003],2optimizer steps,72connected/nonzero gradient
tensors,matching same-seed mask/update/LR traces. First-epoch mask hashes:

- seed67:`c69809d3c8f78f824f6847fd9c776a829cb91a10431bc39efc281629da9dd329`.
- seed68:`a1820547710adea8388a6ebf1f608b979ad3f31872e10e2b079ad247e045d0dd`.

No dedicated extra training smoke:the identical original Flat in cmerc already
passed real MOSI forward/backward,then formal first-epoch checks validated the
two new MOSI seeds. Three scope/schedule/schema unit tests pass locally and in
remote cmerc; independent specification and safety/code reviews passed. Allfive
reference configs and canonical model/sampler byte hashes passed CPU preflight.

## Provenance limits

Runtime numerical snapshot0da38b3 has identical `gcnet_missing_m3` and
`gcnet_modality_jepa` trees to5877678 used by the reused cmerc MOSI66. The new
runner overlay is b0a712e;both identities are recorded,not confused.

The s0 MOSEI66 reference was trained later with default-disabled V3/re-read/schema
extensions. Only an explicit eleven-field default whitelist is compatible with
the older canonical config. This difference is disclosed in each run's
`identity.reference.disabled_schema_fields` and `s0_same_seed_core_exact` instead
of falsely claiming all s0/source snapshots are byte-identical. Same-seed
initial parameters,data,split,masks and protocol/update-count checks still apply.
Cross-environment task losses and numerical parameter updates are not required
or claimed to be identical.

## Persistence and remaining work

LAUNCH.json gives source,environment,GPUUUID,tmux,controller/workerPIDs and paths.
Each new run keeps eight per-rate BEST checkpoints,matching selected predictions,
fixed masks,history,and complete Adam/model/RNG last state. Original/historical
checkpoints are not deleted or overwritten. Saving every epoch's weights was
not requested;per-rate BEST follows the existing protocol.

The controller automatically produces complete-only three-seed reports for both
datasets after five new runs finish and reused MOSI66 is validated. No further
seeds,modules,HPO,mask changes or datasets are authorized by this queue.
Existing optional torch-sparse warning is logged;no dependencies were changed.

The four conditional second-read jobs were stopped separately by user request;
their artifacts remain and they are not restarted by this queue. No other
processes were signalled and prohibited physicalGPU4 was not used.
