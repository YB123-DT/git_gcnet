# Base/Gap representation magnitude audit

**INTERNAL DIAGNOSTIC ONLY.** Same five frozen MOSI Flat causal eta=.6 checkpoints.
AT/AV/TV, exactly one missing and nonzero labels; rates .1–.7. Rate0 has no eligible samples.
One normal eval forward captures Base and the unique active Gap before emotion fusion; no training or memory/readout alteration.
Means weight rates equally within seed then seeds equally. Context has 512 dimensions, including the unchanged zero backward half.

| Pattern | norm B | norm G | norm(B-G) | norm(B+G) | cos(B,G) | centered cosine | relative difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| AT | 10.289923 | 7.023065 | 7.624691 | 15.869146 | 0.645964 | 0.551936 | 0.449434 |
| AV | 10.501300 | 6.950177 | 7.282360 | 16.190058 | 0.649570 | 0.754476 | 0.453887 |
| TV | 10.390477 | 3.101600 | 9.795467 | 11.891799 | 0.314376 | 0.234057 | 0.730807 |

Relative difference = ||B-G||/(||B||+||G||), computed per sample before averaging.
Centered cosine: independently subtract B and G mean vectors WITHIN seed/rate/pattern, then paired cosine; not CKA and not pooled centering.
Zero-norm cosine is undefined, never filled with 0 or 1; counts and zero fractions are explicitly retained.

| Pattern | Base zero fraction | Gap zero fraction |
|---|---:|---:|
| AT | 0.048093 | 0.048093 |
| AV | 0.038091 | 0.038091 |
| TV | 0.038433 | 0.038433 |

Per-sample values, group mean/median/P90, valid counts and across-seed SD are saved in CSV.
Checkpoint selection remains the historical eight-rate-mean Test oracle. No reselection, Jacobian, random-direction control or training.
Norm/cosine evidence alone does not establish whether a difference encodes semantic information or why the downstream classifier is insensitive.
