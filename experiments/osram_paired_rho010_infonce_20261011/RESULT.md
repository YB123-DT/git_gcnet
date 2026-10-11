# rho.10 + InfoNCE: completed

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY. MOSI seed66,100epochs, all100paired-view
training mask hashes match existing task-only C02; original evaluation mask
check passed. Historical backbone6e2a998 retained; InfoNCE weight0→.1 only.

| Rate | rho.10 task-only | rho.10 + InfoNCE |
|---|---:|---:|
|.0|88.552|86.791|
|.1|86.408|85.702|
|.2|83.520|82.655|
|.3|81.263|80.533|
|.4|81.493|79.972|
|.5|76.806|76.336|
|.6|75.246|74.366|
|.7|76.556|75.177|
|Eight-rate mean|81.231|80.191|
|High missing mean|76.203|75.293|

Differences:-1.039pp overall,-.910pp high missing. All eight rate scores lower.
One seed; no causal/generalization claim. No expansion or parameter search.
Training anchor exposures72304; actual observed-bit deletion16.1745%; no batches
without cross-conversation negatives. Single-view inference.

Machine-readable summaries, effective configuration/source hashes, provenance
and per-rate metrics are in completed/. BEST weights remain on biggpu under
/data1/yb/remote_experiments/osram_paired_rho010_infonce_20261011/run.
Continuous-missing results in FIXED_RESULT.md refer to task-only, NOT this run.
