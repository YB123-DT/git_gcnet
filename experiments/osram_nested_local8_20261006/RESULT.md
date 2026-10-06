# Nested: Local1 versus Local8

INTERNAL DIAGNOSTIC ONLY; per-rate BEST Test-oracle screening.
No validation-selected formal paper claim.

Reference code022aa2c. Old Nested seed66 mean8=80.992147%, high=76.077229%;
Flat seed66 mean8=81.068095%, high=76.352251%. Both existing results reused.

## Locked change

Local256 splits contiguously into8x32. Eight separately learned Linear32->64
produce eight graph nodes with Local role and group-index embeddings. These
are GRAPH LOCAL GROUPS, not existing OSRAM memory/query heads. Base/active
Gap still use actual8x64 forward memory heads, projected exactly as before.

Token order:Local0..7, Base8..15, Gap-A16..23, Gap-T24..31, Gap-V32..39.
Only valid roles are packed. ATV16active nodes, pair24, single32.
Fixed same-role/same-index edges and every Local node connects all active
nodes. Index alignment Local-group/Memory-head is a design prior, not a
claim of learned semantic correspondence. Each Local-root subgraph is full
active graph; they differ in root markers/content, not vertex membership.

Old mean-pooled3layer GIN/root/distance encoding remains. Each Local output
node zero-decodes64->32; concatenation restores256. Base/Gap zero-decoders
restore original512 forward slots. Input residuals feed original Flat adapter;
original Local Skip remains unchanged. First utterance/zero-history bypass,
inactive Gap/padding masks, trailing zero backward512 preserved.

No Memory read/write/query changes, no new history, Gate, completion, JEPA,
auxiliary loss, persistent mix, paired views or optimizer change. One-stage
joint task training, not frozen-memory training. Partitioning changes topology
and parameterization; not a pure test of OSRAM head count.

## Protocol

One MOSI seed66 from scratch100epochs, cfg84 original random cyclic missing
0.0–0.7, Adam/lr/batch32/task MSE unchanged. Original ordered mask hashes
verified on completion. Preserve8BEST/8predictions/last_training.pt.
Server biggpu healthyGPU6 only after live capacity check, GPU4 forbidden.
Runner:experiments.osram_core20_20261005.run --method nested_local8_evidence
with same-seed Flat reference config and audited DATA.json. Source git archive
sealed using existing experiments.osram_nested_sweep_20261005/seal.py.

## Verification

25 focused CPU tests passed17.76s in biggpu s0, one existing torch-geometric
deprecation warning. Tests-first reds covered absent method/registration and
common initialization mismatch; final parity regression verifies identical
common GIN, memory projectors, role/head embeddings and post-factory RNG.
New Local projections initialize under isolated fork_rng; discarded legacy
initialization draw is not an unused registered layer. All zero decoders,
contiguous split/decode ordering,16/24/32 active nodes, finite gradients and
actual updates checked. Public first-utterance/padding/upper512 invariants and
old Nested regressions pass. Real full-model task training3 CPU steps updates
Local projections with jepa_loss=0. Separate spec and quality reviews pass.
Compilation/diff checks pass. No GPU smoke or new dependency.

Parameter counts: Flat13,509,793; old Nested13,669,028; Local8=13,669,476.
Adapter159,683 versus old159,235 (+448). Shared GIN core58,627 unchanged.
More graph nodes may increase compute despite similar parameter counts.

Status: implementation/checks complete; preparing launch. No new score claimed.
