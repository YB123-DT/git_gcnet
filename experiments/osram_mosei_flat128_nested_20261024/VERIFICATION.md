# Pre-training verification

INTERNAL TEST-ORACLE SCREEN. No formal accuracy improvement is claimed.

Implementation source: `e65fd6b4edab93944f8739532aacfe9fe2b6c691`.
Server: biggpu; interpreter `/data2/yb/conda_envs/cmerc-repro/bin/python`.
Python3.10.13, Torch1.13.1+cu117, CUDA11.7, NumPy1.26.4, sklearn1.3.2.

Local and remote focused regression suite:72passed on each. Compileall and
diff checks passed. Independent read-only worker/probe/queue review found no
blocking correctness or admission issue.

The independent snapshot copies the accepted capacity source and overlays only
the committed runner/tests and21 existing Nested configuration registrations.
All130 other numerical model/sampler files are byte-identical. Its canonical
source manifest hash is:
`50419bca46c696a2a7a8540ea5669379bcd612d879f3ef5d73d36d088417e6c5`.

Reused F66 passed strict configuration/environment/data/split/mask/RNG checks,
all eight selected prediction/checkpoint hashes, selection and aggregate checks.
Its eight-rate ACC83.606219/W-F1 83.387044; high-missing W-F1 80.011427.

## Real GPU probe

Read-only data selection covers the largest real training rectangle (32x98,
331valid utterances) and most-valid batch (32x45,355valid), each at rates0/.7.
Four task updates per probe, using original MSE/Adam/global clipping. This is
bounded numerical/resource verification, not a shortened performance run.

| Slot | Parameters | Peak allocated MiB | Peak reserved MiB | Reservation MiB | Seconds |
|---|---:|---:|---:|---:|---:|
| N01 | 4,843,235 | 1878.637 | 2104 | 4096 | 7.742 |
| N20 | 4,906,212 | 2152.983 | 2376 | 4096 | 9.083 |
| N21 | 5,409,380 | 2605.353 | 2788 | 4096 | 10.027 |
| F67 | 4,746,977 | 1752.103 | 1864 | 4096 | 7.451 |

Every initial comparison was bitwise equal, max absolute prediction error0,
with unchanged polarity. All four probes used exactly four Memory forwards for
four optimizer steps. Nested tokenizer/layers/pooling/readout/decoders received
finite nonzero task gradients after decoder startup (39/69/69 parameter tensors
observed nonzero for N01/N20/N21). Model+Adam+CPU/CUDA RNG were saved and restored,
with exact repeated eval predictions and one saved UUID-bound CUDA RNG state.
Probe weights are preserved remotely and are NOT used to initialize formal runs.

Reservation4096MiB exceeds every measured allocator-reserved peak plus1024MiB.
It also incorporates the3039MiB allocated full-phase anchor from the preceding
larger8.36M capacity run. Six admission slots therefore reserve24GiB plus2GiB
card margin. Live allocator/process memory and foreign occupancy can further
lower concurrency. This does not claim that four batches measure all60epochs.

Full evidence:
`/data2/yb/remote_experiments/osram_mosei_flat128_nested_20261024/probe_v1/RESOURCE_PROOF.json`

Evidence SHA256:
`89b4f149539b76d7d9997226c1be5e9e3f0db527c93ebbda648b60785fcc30fa`.

At16:05UTC actual fresh workers already passed configuration and initialization
checks: same accepted split/masks/data/environment/RNG; Nested66 original
parameters match F66 exactly, while fresh Flat67/68 initialization differs as
required. Training mask-trace equality is checked when complete; no completed
Nested screen result yet exists. The queue preserves all failures and stops
new admission on a failed worker rather than silently retrying.
