# Forward-read head intervention

INTERNAL DIAGNOSTIC ONLY

Same original Full checkpoint per rate. Only a selected 64-d forward read slice is zeroed before the classification input. Base-head intervention masks that Base slice; Gap-head intervention masks that slice in all three Gap slots (inactive slots remain zero). This is not a per-modality Gap-head intervention, and does not modify query, write, scan, Local, or task head.
Contribution = W-F1(Full) − W-F1(masked), in percentage points. Positive means masking reduces performance. Head indices are zero-based, 0–7. Current availability conditions label columns; they are not missing-pattern-specific retrained models.
Nonzero MOSI labels only; polarity uses prediction > 0. Per-rate metrics precede equal-rate macro averaging. High missing means rates 0.5/0.6/0.7. Counts sum repeated rate exposures, not independent observations.

## Base head contribution matrix (8-rate macro)

|Head|Overall|A|T|V|AT|AV|TV|ATV|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|0.013|-0.035|0.118|-1.765|-0.337|-0.312|-0.333|0.257|
|1|-0.036|0.000|-0.074|-0.084|0.000|0.000|0.000|0.000|
|2|-0.009|0.000|0.000|-0.107|-0.149|-0.163|-0.160|0.096|
|3|0.051|-0.130|0.118|-0.032|-0.149|0.004|-0.160|0.171|
|4|-0.069|-0.095|0.118|-0.235|-0.337|-0.421|-0.488|0.163|
|5|0.006|0.160|0.118|0.106|-0.337|-0.421|-0.641|0.163|
|6|0.101|0.010|0.118|0.269|-0.337|-0.154|-0.315|0.216|
|7|0.348|0.216|0.037|0.615|0.185|-0.575|0.302|0.397|
## Gap head contribution matrix (8-rate macro)

|Head|Overall|A|T|V|AT|AV|TV|ATV|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|0.081|0.617|0.118|0.050|-0.149|-0.163|0.000|0.000|
|1|-0.018|0.000|-0.074|-0.084|0.000|0.143|0.000|0.000|
|2|0.002|0.133|-0.074|0.034|-0.149|-0.258|-0.160|0.000|
|3|0.009|-0.035|0.118|0.121|-0.149|-0.254|0.000|0.000|
|4|0.066|0.335|0.118|0.324|-0.149|-0.387|-0.160|0.000|
|5|0.115|0.347|0.118|0.350|-0.149|-0.133|0.000|0.000|
|6|0.057|0.083|0.118|-0.074|-0.149|0.143|-0.160|0.000|
|7|0.160|0.760|0.118|0.486|-0.149|-0.884|0.000|0.000|

## Aggregate paired intervention statistics

|Intervention|Head|Scope|Full W-F1|Masked W-F1|Contribution|Mask corrections|Mask harms|N|
|---|---:|---|---:|---:|---:|---:|---:|---:|
|base|0|overall|81.068|81.055|0.013|17|19|5248|
|base|0|high_missing|76.352|76.278|0.074|4|6|1968|
|base|1|overall|81.068|81.104|-0.036|2|0|5248|
|base|1|high_missing|76.352|76.449|-0.097|2|0|1968|
|base|2|overall|81.068|81.077|-0.009|6|6|5248|
|base|2|high_missing|76.352|76.419|-0.066|4|3|1968|
|base|3|overall|81.068|81.017|0.051|14|18|5248|
|base|3|high_missing|76.352|76.391|-0.039|7|7|1968|
|base|4|overall|81.068|81.137|-0.069|25|23|5248|
|base|4|high_missing|76.352|76.386|-0.034|7|7|1968|
|base|5|overall|81.068|81.062|0.006|24|26|5248|
|base|5|high_missing|76.352|76.182|0.171|5|9|1968|
|base|6|overall|81.068|80.967|0.101|17|24|5248|
|base|6|high_missing|76.352|76.252|0.100|8|11|1968|
|base|7|overall|81.068|80.720|0.348|30|50|5248|
|base|7|high_missing|76.352|76.127|0.225|9|14|1968|
|gap|0|overall|81.068|80.987|0.081|8|13|5248|
|gap|0|high_missing|76.352|76.128|0.225|6|11|1968|
|gap|1|overall|81.068|81.086|-0.018|2|1|5248|
|gap|1|high_missing|76.352|76.449|-0.097|2|0|1968|
|gap|2|overall|81.068|81.066|0.002|14|15|5248|
|gap|2|high_missing|76.352|76.279|0.073|7|9|1968|
|gap|3|overall|81.068|81.059|0.009|5|6|5248|
|gap|3|high_missing|76.352|76.224|0.128|3|6|1968|
|gap|4|overall|81.068|81.002|0.066|19|24|5248|
|gap|4|high_missing|76.352|76.051|0.302|9|16|1968|
|gap|5|overall|81.068|80.953|0.115|20|28|5248|
|gap|5|high_missing|76.352|75.862|0.491|10|21|1968|
|gap|6|overall|81.068|81.011|0.057|8|12|5248|
|gap|6|high_missing|76.352|76.144|0.208|6|11|1968|
|gap|7|overall|81.068|80.908|0.160|45|56|5248|
|gap|7|high_missing|76.352|75.712|0.640|21|35|1968|

Mask corrections / harms compare Full → masked, not the reverse. Raw per-rate and per-availability metrics and sample counts are in per_rate.csv; high-missing and all stratified macro results are in macro.csv.
Checkpoint weights differ across rates. A one-seed leave-one-head-out response is not proof of head specialization or conditional routing; marginal effects can overlap and need not add. No significance claim, gate fitting, test-label-based head selection, or Query/Addressing third-stage intervention was performed.
