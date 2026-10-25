# History exposure proportions and conditional error rates

INTERNAL DIAGNOSTIC ONLY. Derived from the unchanged cmerc LargeFlat seed66 error-source ledger.
Three different quantities are recorded; they must not be substituted for each other:
- Sample proportion: N_never / (N_never + N_seen), and its seen complement.
- Conditional error rate: errors_never / N_never, or errors_seen / N_seen.
- Error composition: errors_never / (errors_never + errors_seen), and its seen complement.
All denominators exclude gold0, as in the original binary metric. History itself includes gold0 turns.
First turns remain in never-seen. Missing-target marginals overlap across modalities; do not sum columns.

## MOSI

### Sample proportions (never / seen)

| Rate | Currently missing A | Currently missing T | Currently missing V |
|---|---|---|---|
|0.0|N/A / N/A|N/A / N/A|N/A / N/A|
|0.1|2.50% / 97.50%|3.23% / 96.77%|3.85% / 96.15%|
|0.2|3.15% / 96.85%|8.97% / 91.03%|9.09% / 90.91%|
|0.3|7.58% / 92.42%|6.07% / 93.93%|7.14% / 92.86%|
|0.4|7.97% / 92.03%|7.92% / 92.08%|7.54% / 92.46%|
|0.5|6.80% / 93.20%|7.03% / 92.97%|7.55% / 92.45%|
|0.6|8.72% / 91.28%|11.75% / 88.25%|8.52% / 91.48%|
|0.7|12.50% / 87.50%|12.30% / 87.70%|11.55% / 88.45%|

### Conditional error rates (never / seen)

| Rate | Currently missing A | Currently missing T | Currently missing V |
|---|---|---|---|
|0.0|N/A / N/A|N/A / N/A|N/A / N/A|
|0.1|50.00% / 20.51%|50.00% / 35.00%|0.00% / 16.00%|
|0.2|25.00% / 13.82%|30.77% / 34.09%|16.67% / 17.50%|
|0.3|13.33% / 16.39%|46.15% / 31.84%|15.38% / 12.43%|
|0.4|35.00% / 17.75%|28.57% / 34.02%|31.58% / 21.89%|
|0.5|30.43% / 21.59%|52.17% / 33.88%|28.00% / 19.28%|
|0.6|41.18% / 21.91%|40.43% / 33.43%|26.47% / 22.74%|
|0.7|31.48% / 25.13%|40.00% / 36.73%|40.00% / 24.54%|

### Share among errors (never / seen)

| Rate | Currently missing A | Currently missing T | Currently missing V |
|---|---|---|---|
|0.0|N/A / N/A|N/A / N/A|N/A / N/A|
|0.1|5.88% / 94.12%|4.55% / 95.45%|0.00% / 100.00%|
|0.2|5.56% / 94.44%|8.16% / 91.84%|8.70% / 91.30%|
|0.3|6.25% / 93.75%|8.57% / 91.43%|8.70% / 91.30%|
|0.4|14.58% / 85.42%|6.74% / 93.26%|10.53% / 89.47%|
|0.5|9.33% / 90.67%|10.43% / 89.57%|10.61% / 89.39%|
|0.6|15.22% / 84.78%|13.87% / 86.13%|9.78% / 90.22%|
|0.7|15.18% / 84.82%|13.25% / 86.75%|17.54% / 82.46%|

## MOSEI

### Sample proportions (never / seen)

| Rate | Currently missing A | Currently missing T | Currently missing V |
|---|---|---|---|
|0.0|N/A / N/A|N/A / N/A|N/A / N/A|
|0.1|13.48% / 86.52%|14.09% / 85.91%|15.05% / 84.95%|
|0.2|13.60% / 86.40%|16.03% / 83.97%|16.41% / 83.59%|
|0.3|21.78% / 78.22%|17.91% / 82.09%|19.58% / 80.42%|
|0.4|19.90% / 80.10%|19.74% / 80.26%|21.79% / 78.21%|
|0.5|24.05% / 75.95%|24.37% / 75.63%|24.88% / 75.12%|
|0.6|27.18% / 72.82%|27.52% / 72.48%|28.20% / 71.80%|
|0.7|29.57% / 70.43%|33.88% / 66.12%|31.83% / 68.17%|

### Conditional error rates (never / seen)

| Rate | Currently missing A | Currently missing T | Currently missing V |
|---|---|---|---|
|0.0|N/A / N/A|N/A / N/A|N/A / N/A|
|0.1|16.00% / 12.15%|26.92% / 23.97%|19.64% / 12.03%|
|0.2|9.28% / 13.80%|22.12% / 19.59%|19.13% / 14.33%|
|0.3|18.88% / 13.65%|25.79% / 20.21%|15.96% / 12.91%|
|0.4|17.13% / 15.64%|26.83% / 22.02%|19.44% / 14.93%|
|0.5|18.79% / 14.03%|25.06% / 21.57%|19.00% / 16.99%|
|0.6|21.85% / 17.00%|27.35% / 24.27%|21.99% / 16.44%|
|0.7|22.05% / 18.75%|28.95% / 23.24%|21.74% / 19.33%|

### Share among errors (never / seen)

| Rate | Currently missing A | Currently missing T | Currently missing V |
|---|---|---|---|
|0.0|N/A / N/A|N/A / N/A|N/A / N/A|
|0.1|17.02% / 82.98%|15.56% / 84.44%|22.45% / 77.55%|
|0.2|9.57% / 90.43%|17.73% / 82.27%|20.75% / 79.25%|
|0.3|27.81% / 72.19%|21.78% / 78.22%|23.13% / 76.87%|
|0.4|21.40% / 78.60%|23.05% / 76.95%|26.61% / 73.39%|
|0.5|29.78% / 70.22%|27.23% / 72.77%|27.02% / 72.98%|
|0.6|32.43% / 67.57%|29.96% / 70.04%|34.44% / 65.56%|
|0.7|33.05% / 66.95%|38.96% / 61.04%|34.43% / 65.57%|

## Interpretation

At rate0.7 with current Text missing, MOSI has87.70%seen-history samples;
their error rate is36.73%, versus40.00%for never-seen samples. MOSEI has66.12%seen-history
samples; their error rate is23.24%, versus28.95%for never-seen samples. More absolute
errors in the seen group therefore do not imply a higher probability of error there.
These are descriptive comparisons:groups differ in position, content and label distribution,
so lower error rates do not establish a causal Memory benefit. MOSI also has rates where
seen error rates are not lower; do not claim the direction holds for all rates/modalities.
No new inference/training, changing model/masks, or additional model selection occurred.
