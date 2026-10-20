# Completed MOSI direct-source weighting: interpretation

INTERNAL TEST-ORACLE SCREEN

Three seeds66/67/68 all completed150epochs with unchanged numerical model,
original GCNet missing and original MSE/Flat training. Each comparison uses
the corresponding seed's completed Flat,not an older missing protocol.

|Seed|Flat8-rate W-F1|Direct8-rate W-F1|Delta pp|Flat high|Direct high|Delta pp|
|---|---:|---:|---:|---:|---:|---:|
|66|79.128|79.450|+.322|73.338|73.478|+.140|
|67|78.751|78.578|-.172|72.659|72.320|-.339|
|68|78.794|78.471|-.322|72.727|72.062|-.665|
|Mean|78.891|78.833|-.058|72.908|72.620|-.288|

The seed66 gain did not replicate in67/68. This finite screen therefore does
not support a stable improvement over Flat from this direct-source weighting
configuration. It does not prove that all source-aware methods are ineffective.
Across-seed8-rate std increased from.206 to.536; this is descriptive only,
not a significance claim from three seeds. High-missing mean was lower in
two seeds and on average. Per-rate averaged results improved at.0/.2/.4/.5,
but declined at.1/.3/.6/.7; .7 lost.511pp on average.

This direct form also changes the coefficient range versus the earlier
residual form(.75–1.25 to0–2),so it does not isolate removing a residual
connection. Role/source labels remain joint Value-slot decomposition labels,
not proof of pure modality semantics or semantic causal attribution.

Verification used existing artifacts only:24selected NPZs,exact matched
cohorts/masks,metrics/earliest selection and450training trace comparisons.
No new training,inference,threshold search,feature/mask change or extra seeds.
All checkpoints and exact selected predictions remain preserved remotely.
No further automatic method expansion is authorized by this completed queue.
