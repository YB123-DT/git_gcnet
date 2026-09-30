# Feature-only Gate completed

All three GPU6 runs completed100 epochs, returncode0. Source3d372aa.
Runner verified canonical evaluation masks and eight best checkpoints per seed
before completion. Archived summary, configs, metrics and provenance in results.
No new inference or independent prediction rescoring performed for this status.

Internal per-rate Test-oracle best-checkpoint diagnosis, not formal paper results.
W-F1 percentages; high missing averages .5/.6/.7. Flat references reused.

|Seed|Flat8-rate|Feature-only8-rate|Delta pp|Flat high|Feature-only high|Delta pp|
|---|---:|---:|---:|---:|---:|---:|
|66|81.068|80.243|−0.825|76.352|75.181|−1.172|
|67|80.556|79.671|−0.885|75.990|75.434|−0.555|
|68|80.053|79.956|−0.097|74.440|74.859|+0.419|
|Mean|80.559|79.957|−0.603|75.594|75.158|−0.436|

Previous two-level Gate mean:80.018/high75.267. Removing competition did not
recover the Flat baseline; feature-only mean is lower by.061/.109 points than
the two-level version. This small difference is not a significance claim and
does not establish a general causal explanation. It does not support attributing
the entire prior degradation solely to evidence softmax competition.
Formal feature-only runs used GPU6, whereas prior two-level runs used GPU0 on
the same server; no timing or strict hardware-controlled claim is made.
