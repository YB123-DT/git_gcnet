# V3 source attention audit

INTERNAL TEST-ORACLE CHECKPOINT DIAGNOSTIC

Original standard V3 seed66 only: width64,LargeFlat1600,residual mode.
MOSI/MOSEI8rates; IEMOCAP4/6 five folds x8rates. No training or model updates.
Every replay uses its own saved BEST checkpoint and exact original mask.
MOSI/MOSEI use original s0; IEMOCAP original cmerc-repro. Saved predictions
must match within1e-6 and predicted classes exactly, before any summary.

## Text currently missing, all historical A/T/V seen

Rows below pool TEST SLOTS across rates/folds descriptively. The same utterance
can appear at multiple rates, from different BEST weights; these are NOT
independent samples, seeds, or a single deployed model. Per-rate tables follow.

| Dataset | Role | N | A | T | V | Normalized entropy(avg heads) | Mean head entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| CMUMOSI | Base | 1632 | 0.3444 | 0.3152 | 0.3405 | 0.9987 | 0.9963 |
| CMUMOSI | Gap-T | 1632 | 0.3172 | 0.3366 | 0.3462 | 0.9989 | 0.9904 |
| CMUMOSEI | Base | 7737 | 0.3333 | 0.3333 | 0.3334 | 1.0000 | 1.0000 |
| CMUMOSEI | Gap-T | 7737 | 0.3333 | 0.3333 | 0.3334 | 1.0000 | 1.0000 |
| IEMOCAPFour | Base | 13708 | 0.3091 | 0.3413 | 0.3496 | 0.9635 | 0.9111 |
| IEMOCAPFour | Gap-T | 13708 | 0.3221 | 0.3121 | 0.3658 | 0.9779 | 0.9461 |
| IEMOCAPSix | Base | 18892 | 0.2924 | 0.4054 | 0.3023 | 0.9792 | 0.9555 |
| IEMOCAPSix | Gap-T | 18892 | 0.3167 | 0.3444 | 0.3389 | 0.9909 | 0.9832 |

## Paired role difference and Local-query sensitivity

| Dataset | Base-v-GapT L1 | Same argmax fraction | Base query-swap L1 | GapT query-swap L1 |
|---|---:|---:|---:|---:|
| CMUMOSI | 0.065920 | 0.226103 | 0.020309 | 0.009475 |
| CMUMOSEI | 0.000056 | 0.793869 | 0.000020 | 0.000010 |
| IEMOCAPFour | 0.230969 | 0.392326 | 0.069924 | 0.051704 |
| IEMOCAPSix | 0.132103 | 0.668907 | 0.045341 | 0.015552 |

Query swap uses another conversation within the same test batch, matched on
role,current availability and seen-source set, retaining identical original
keys/values. Only extra attention is computed; predictions keep original outputs.
This measures query dependence, NOT whether selected evidence improves the task.

## Interpretation limits

- These are Local-to-role output CrossAttention weights, after Layer2 source mixing.
- High weights are NOT original-source causal usefulness, reliability, or sentiment shift.
- No attention is defined for V3-disabled ATV/no-history/padding/inactive Gap rows.
  Base remains available to original Flat even if V3 Base correction is disabled.
- Source counts0/1 are separate; automatic weight1 is not learned confidence.
- Normalized entropy1 means uniform across valid sources;0 means concentrated.
- Per-sample std is descriptive; no bootstrap/significance or three-seed claim.
- Correct/error labels only stratify offline; they never form Query or select weights.
- All96 per-fold/per-rate reports and detailed arrays remain remote; no weights/data
  or large activation caches enter Git.

## Per-rate Text-missing focus

| Dataset | Rate | Role | N | A | T | V | Entropy |
|---|---:|---|---:|---:|---:|---:|---:|
| CMUMOSI | 0.0 | Base | 0 | N/A | N/A | N/A | N/A |
| CMUMOSI | 0.0 | Gap-T | 0 | N/A | N/A | N/A | N/A |
| CMUMOSI | 0.1 | Base | 63 | 0.3247 | 0.3249 | 0.3504 | 0.9992 |
| CMUMOSI | 0.1 | Gap-T | 63 | 0.3291 | 0.3400 | 0.3309 | 0.9999 |
| CMUMOSI | 0.2 | Base | 135 | 0.3383 | 0.3133 | 0.3484 | 0.9988 |
| CMUMOSI | 0.2 | Gap-T | 135 | 0.3234 | 0.3354 | 0.3412 | 0.9997 |
| CMUMOSI | 0.3 | Base | 201 | 0.3391 | 0.3138 | 0.3471 | 0.9987 |
| CMUMOSI | 0.3 | Gap-T | 201 | 0.3167 | 0.3366 | 0.3467 | 0.9992 |
| CMUMOSI | 0.4 | Base | 238 | 0.3435 | 0.3112 | 0.3453 | 0.9985 |
| CMUMOSI | 0.4 | Gap-T | 238 | 0.3222 | 0.3357 | 0.3421 | 0.9996 |
| CMUMOSI | 0.5 | Base | 302 | 0.3375 | 0.3195 | 0.3430 | 0.9992 |
| CMUMOSI | 0.5 | Gap-T | 302 | 0.3291 | 0.3396 | 0.3313 | 0.9998 |
| CMUMOSI | 0.6 | Base | 333 | 0.3516 | 0.3209 | 0.3275 | 0.9988 |
| CMUMOSI | 0.6 | Gap-T | 333 | 0.3017 | 0.3329 | 0.3654 | 0.9968 |
| CMUMOSI | 0.7 | Base | 360 | 0.3527 | 0.3086 | 0.3386 | 0.9982 |
| CMUMOSI | 0.7 | Gap-T | 360 | 0.3140 | 0.3382 | 0.3478 | 0.9990 |
| CMUMOSEI | 0.0 | Base | 0 | N/A | N/A | N/A | N/A |
| CMUMOSEI | 0.0 | Gap-T | 0 | N/A | N/A | N/A | N/A |
| CMUMOSEI | 0.1 | Base | 384 | 0.3333 | 0.3333 | 0.3333 | 1.0000 |
| CMUMOSEI | 0.1 | Gap-T | 384 | 0.3333 | 0.3333 | 0.3333 | 1.0000 |
| CMUMOSEI | 0.2 | Base | 733 | 0.3333 | 0.3333 | 0.3333 | 1.0000 |
| CMUMOSEI | 0.2 | Gap-T | 733 | 0.3333 | 0.3333 | 0.3333 | 1.0000 |
| CMUMOSEI | 0.3 | Base | 991 | 0.3333 | 0.3333 | 0.3333 | 1.0000 |
| CMUMOSEI | 0.3 | Gap-T | 991 | 0.3333 | 0.3333 | 0.3333 | 1.0000 |
| CMUMOSEI | 0.4 | Base | 1298 | 0.3332 | 0.3331 | 0.3337 | 1.0000 |
| CMUMOSEI | 0.4 | Gap-T | 1298 | 0.3331 | 0.3330 | 0.3339 | 1.0000 |
| CMUMOSEI | 0.5 | Base | 1425 | 0.3333 | 0.3333 | 0.3333 | 1.0000 |
| CMUMOSEI | 0.5 | Gap-T | 1425 | 0.3333 | 0.3333 | 0.3333 | 1.0000 |
| CMUMOSEI | 0.6 | Base | 1479 | 0.3333 | 0.3333 | 0.3333 | 1.0000 |
| CMUMOSEI | 0.6 | Gap-T | 1479 | 0.3333 | 0.3333 | 0.3333 | 1.0000 |
| CMUMOSEI | 0.7 | Base | 1427 | 0.3333 | 0.3333 | 0.3333 | 1.0000 |
| CMUMOSEI | 0.7 | Gap-T | 1427 | 0.3333 | 0.3333 | 0.3333 | 1.0000 |
| IEMOCAPFour | 0.0 | Base | 0 | N/A | N/A | N/A | N/A |
| IEMOCAPFour | 0.0 | Gap-T | 0 | N/A | N/A | N/A | N/A |
| IEMOCAPFour | 0.1 | Base | 494 | 0.3005 | 0.3464 | 0.3531 | 0.9446 |
| IEMOCAPFour | 0.1 | Gap-T | 494 | 0.3307 | 0.3402 | 0.3291 | 0.9770 |
| IEMOCAPFour | 0.2 | Base | 1074 | 0.3173 | 0.3482 | 0.3345 | 0.9869 |
| IEMOCAPFour | 0.2 | Gap-T | 1074 | 0.3377 | 0.3282 | 0.3341 | 0.9956 |
| IEMOCAPFour | 0.3 | Base | 1561 | 0.3261 | 0.3568 | 0.3171 | 0.9849 |
| IEMOCAPFour | 0.3 | Gap-T | 1561 | 0.3211 | 0.3201 | 0.3589 | 0.9822 |
| IEMOCAPFour | 0.4 | Base | 2029 | 0.3101 | 0.3706 | 0.3194 | 0.9680 |
| IEMOCAPFour | 0.4 | Gap-T | 2029 | 0.3344 | 0.3063 | 0.3593 | 0.9898 |
| IEMOCAPFour | 0.5 | Base | 2569 | 0.3096 | 0.3153 | 0.3752 | 0.9602 |
| IEMOCAPFour | 0.5 | Gap-T | 2569 | 0.3070 | 0.2867 | 0.4063 | 0.9681 |
| IEMOCAPFour | 0.6 | Base | 2803 | 0.3076 | 0.3148 | 0.3776 | 0.9553 |
| IEMOCAPFour | 0.6 | Gap-T | 2803 | 0.3157 | 0.2912 | 0.3931 | 0.9672 |
| IEMOCAPFour | 0.7 | Base | 3178 | 0.2998 | 0.3562 | 0.3440 | 0.9551 |
| IEMOCAPFour | 0.7 | Gap-T | 3178 | 0.3258 | 0.3411 | 0.3331 | 0.9796 |
| IEMOCAPSix | 0.0 | Base | 0 | N/A | N/A | N/A | N/A |
| IEMOCAPSix | 0.0 | Gap-T | 0 | N/A | N/A | N/A | N/A |
| IEMOCAPSix | 0.1 | Base | 728 | 0.2748 | 0.4434 | 0.2817 | 0.9601 |
| IEMOCAPSix | 0.1 | Gap-T | 728 | 0.3091 | 0.3544 | 0.3365 | 0.9856 |
| IEMOCAPSix | 0.2 | Base | 1422 | 0.2895 | 0.4003 | 0.3102 | 0.9843 |
| IEMOCAPSix | 0.2 | Gap-T | 1422 | 0.3147 | 0.3348 | 0.3505 | 0.9903 |
| IEMOCAPSix | 0.3 | Base | 2088 | 0.2813 | 0.4239 | 0.2948 | 0.9680 |
| IEMOCAPSix | 0.3 | Gap-T | 2088 | 0.3173 | 0.3362 | 0.3465 | 0.9890 |
| IEMOCAPSix | 0.4 | Base | 2752 | 0.2944 | 0.4030 | 0.3025 | 0.9806 |
| IEMOCAPSix | 0.4 | Gap-T | 2752 | 0.3088 | 0.3716 | 0.3195 | 0.9861 |
| IEMOCAPSix | 0.5 | Base | 3461 | 0.2924 | 0.4078 | 0.2998 | 0.9760 |
| IEMOCAPSix | 0.5 | Gap-T | 3461 | 0.3170 | 0.3615 | 0.3215 | 0.9925 |
| IEMOCAPSix | 0.6 | Base | 4096 | 0.2947 | 0.4016 | 0.3037 | 0.9814 |
| IEMOCAPSix | 0.6 | Gap-T | 4096 | 0.3171 | 0.3351 | 0.3478 | 0.9920 |
| IEMOCAPSix | 0.7 | Base | 4345 | 0.2980 | 0.3948 | 0.3072 | 0.9858 |
| IEMOCAPSix | 0.7 | Gap-T | 4345 | 0.3227 | 0.3277 | 0.3497 | 0.9935 |
