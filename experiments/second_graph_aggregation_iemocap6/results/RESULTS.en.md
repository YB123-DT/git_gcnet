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
