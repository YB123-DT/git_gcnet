# Second-Graph Mechanism Paired Results

## Initial preregistered discrimination gate

PASS means every initial advancement condition was satisfied; FAIL means at least one condition was not. These decisions remain the historical gate record.

| candidate | gate | seed-macro F1 delta |
|---|---:|---:|
| genagg | FAIL | -0.187831724 |
| soft_medoid | FAIL | -0.004304363 |
| ssma | FAIL | -0.007173215 |
| rtdr | FAIL | +0.002541466 |

## Post-gate RTDR stability audit

The requested follow-up did not revise the initial FAIL. The 15-pair extension met its narrower post-gate extension criterion (`+0.008510981`, 3/5 positive seed macros). After completing all 8 missing rates x 5 seeds, RTDR had an overall paired macro delta of `-0.002810103`, only 3/8 positive rate means and 3/5 positive seed macros. All runs were finite and non-collapsed, but the predefined `stable_positive` field is `false`.

| audit scope | paired cells | overall delta | positive rates | positive seeds | status |
|---|---:|---:|---:|---:|---|
| extension (rates 0.0, 0.5, 0.7) | 15 | +0.008510981 | 3/3 | 3/5 | extension criterion PASS |
| full (rates 0.0-0.7) | 40 | -0.002810103 | 3/8 | 3/5 | `stable_positive=false` |

Detailed task-level evidence is preserved in [extension results](rtdr_extension/RESULTS.en.md) and [full-grid results](rtdr_full/RESULTS.en.md).

## Uniform three-rate, five-seed comparison

To put every implemented mechanism on the same minimum evidence grid, the final comparison uses missing rates `{0.0,0.5,0.7}` and seeds `{66,67,68,69,70}`: 15 paired cells per arm. The `uniform_stable` descriptor requires a positive overall macro delta, 3/3 positive rate means, at least 3/5 positive seed macros, and finite, non-collapsed runs. It is a separate descriptive layer and does not revise the initial gate or RTDR's full-rate `stable_positive=false` result.

| candidate | paired cells | overall delta | positive rates | positive seeds | collapse-free | `uniform_stable` |
|---|---:|---:|---:|---:|---:|---:|
| genagg | 15 | -0.204847963 | 0/3 | 0/5 | no | `false` |
| soft_medoid | 15 | +0.004706753 | 2/3 | 4/5 | yes | `false` |
| ssma | 15 | -0.001153174 | 1/3 | 2/5 | yes | `false` |
| rtdr | 15 | +0.008510981 | 3/3 | 3/5 | yes | `true` |

Soft Medoid is mildly positive in aggregate but misses the all-rate condition because its missing-`0.7` mean delta is `-0.002089281`. RTDR is the only arm meeting this bounded three-rate descriptor, while its broader 40-pair audit remains negative overall. Task-level uniform evidence is in [uniform three-rate results](uniform_three_rate/RESULTS.en.md) and [summary.json](uniform_three_rate/summary.json).
