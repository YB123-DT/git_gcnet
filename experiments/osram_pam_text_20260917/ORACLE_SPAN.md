# PAM-T Oracle Memory-Span diagnostic

**Internal diagnostic only; not a formal paper result.**

Same PAM-T checkpoints and test masks.  For every T-missing utterance:

```text
V_t = previous real observed Text latents in the same conversation
z_t* = current real Text latent
z_t^span = V_t a*,  a* = argmin_a ||V_t a - z_t*||^2 + lambda ||a||^2
```

`z_t^span` is passed through the same `CompletedReadFusion` interface.
`lambda = 1e-3`.

## Results (nonzero rates 0.1–0.7)

| Mode | macro W-F1 |
|---|---:|
| Normal | 79.0871 |
| Zero | 77.9831 |
| Shuffle | 78.0395 |
| Oracle-Fusion (real Text) | **81.4504** |
| **Oracle-Span (historical Text span)** | **81.2023** |

Key differences:

- Oracle-Span − Oracle-Fusion = **−0.2481 pp**
- Oracle-Span − Normal = **+2.1152 pp**

## Per-rate results (%)

| Rate | Normal | Oracle-Fusion | Oracle-Span | Span − Fusion | Span − Normal |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.3198 | 87.3198 | 87.3198 | +0.0000 | +0.0000 |
| 0.1 | 85.0395 | 85.6438 | 85.6750 | +0.0311 | +0.6354 |
| 0.2 | 82.4686 | 84.1888 | 83.8489 | −0.3399 | +1.3804 |
| 0.3 | 81.2851 | 82.8526 | 82.8840 | +0.0315 | +1.5989 |
| 0.4 | 78.4480 | 80.4146 | 80.2411 | −0.1735 | +1.7931 |
| 0.5 | 76.5535 | 80.0288 | 79.7358 | −0.2930 | +3.1823 |
| 0.6 | 75.5602 | 79.1374 | 78.7237 | −0.4138 | +3.1635 |
| 0.7 | 74.2550 | 77.8871 | 77.3078 | −0.5792 | +3.0529 |

## Coverage

- span available ratio: **92.57%** (8,120 / 8,772 T-missing samples have at least one previous real observed Text latent).
- prior-write coverage from the earlier four-mode diagnostic: 88.34%.

## Direct conclusion

Oracle-Span is almost identical to Oracle-Fusion:

$$ \text{Oracle-Span} \approx \text{Oracle-Fusion} \quad (81.20 \text{ vs } 81.45) $$

and both are far above Normal:

$$ \text{Oracle-Span} - \text{Normal} = +2.12 \text{ pp}. $$

Therefore:

$$ \boxed{\text{历史 Text 信息足够，主要瓶颈是 query / address 学不好}} $$

The static span of previous real Text latents already recovers most of the oracle gap.
The remaining problem is that PAM's current query

```text
q_t = f_src(A_t, V_t, mask_AV)
```

cannot select the right historical Text basis from `M_T`.

## Where to spend effort next

The Oracle-Span result says the next stage should target the address/query side:

1. **query encoder capacity**
   - more layers / wider MLP;
   - include causal A/V history context `c_t`.

2. **key learning**
   - current write key and read query share one projection;
   - try separate key/query projections;
   - normalize + temperature/similarity learning.

3. **attention / multi-head / retrieval**
   - replace single linear associative memory with attention over stored
     `(key_j, value_j)` pairs;
   - multi-head reads;
   - learned similarity metric.

4. **better read combination**
   - instead of `M_T q`, use attention weights over historical Text values;
   - this directly mimics the Oracle-Span linear combination, but with a
     learned, history-conditioned addressing function.

Do not start with memory capacity, target geometry or loss weight; the Oracle-Span
span is already sufficient, so the first experiment should improve the query /
addressing function.
