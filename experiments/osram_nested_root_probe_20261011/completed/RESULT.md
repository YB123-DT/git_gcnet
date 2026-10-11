# Frozen Nested rooted representation probes

INTERNAL DIAGNOSTIC ONLY

Original frozen Nested seed66, original per-rate Test-oracle BEST100epoch checkpoints.
Probe seeds66/67/68 do NOT represent three backbone seeds. Fixed100epochs, train-only
MSE optimization, no test-based probe selection. Test W-F1 excludes zero labels; threshold0.
Rooted=exact33x64 subgraph pooled states; PreDecoder=33x64 graph readout output.
No head pooling or random projection. Flat=R_off hidden from SAME Nested checkpoint,
still includes raw Local/Base/Gap. Not a separately trained Flat checkpoint.
All probes3715->32->GELU->1,118945parameters. Missing input blocks zero. Train-only
normalization; inactive slots re-zeroed. Donor is train-only, same availability/history,
different conversation, no label matching; common eligible cohorts across arms.
All model/checkpoint hashes unchanged; Full predictions match original labels/masks/archive.
Zero-init refers to decoder initialization, NOT necessarily its trained weight values.

## W-F1 %, equal-rate mean (mean ± sample std over probe seeds)

| Probe | Mean8 | High .5/.6/.7 |
|---|---:|---:|
|Flat|79.654480 ± 0.125549|74.392530 ± 0.298910|
|Rooted|78.180585 ± 0.093181|73.079172 ± 0.703110|
|Flat+Rooted|79.058842 ± 0.250791|73.674034 ± 0.082016|
|Flat+DonorRooted|78.873492 ± 0.090401|73.070309 ± 0.511440|
|PreDecoder|77.440502 ± 0.495313|71.861442 ± 0.751620|
|Flat+PreDecoder|78.396285 ± 0.243168|72.390188 ± 0.838812|

## Per probe seed

| Arm | Seed | Mean8 | High |
|---|---:|---:|---:|
|Flat|66|79.592756|74.604085|
|Flat|67|79.798942|74.522932|
|Flat|68|79.571741|74.050574|
|Rooted|66|78.137437|72.441805|
|Rooted|67|78.116798|72.962326|
|Rooted|68|78.287519|73.833384|
|Flat+Rooted|66|79.007696|73.656006|
|Flat+Rooted|67|79.331263|73.763564|
|Flat+Rooted|68|78.837566|73.602532|
|Flat+DonorRooted|66|78.777698|73.592868|
|Flat+DonorRooted|67|78.885473|72.570765|
|Flat+DonorRooted|68|78.957305|73.047294|
|PreDecoder|66|77.926506|72.711091|
|PreDecoder|67|76.936376|71.589929|
|PreDecoder|68|77.458623|71.283306|
|Flat+PreDecoder|66|78.412390|72.860860|
|Flat+PreDecoder|67|78.631000|72.887965|
|Flat+PreDecoder|68|78.145464|71.421738|

## Paired changes vs Flat probe (pp)

| Seed | Flat+Rooted minus Flat | Flat+Rooted minus donor | Flat+PreDecoder minus Flat |
|---|---:|---:|---:|
|66|-0.585060|+0.229998|-1.180365|
|67|-0.467679|+0.445790|-1.167942|
|68|-0.734175|-0.119739|-1.426277|

## Limits

Positive joint gains above both Flat and donor support extra decodable information under
this bounded probe class, not a deployment gain or identified emotion mechanism.
Negative results do not prove information absence; capacity, optimization and the fixed
budget constrain inference. Equal total parameters are not equal effective capacity.
Inherited backbone test selection means all findings are INTERNAL DIAGNOSTIC ONLY.
Per-rate outcomes in per_rate.csv; all observed-pattern metrics, coverage and train curves
in PROBE_RESULTS.json. Frozen original-head baseline metrics/checkpoint hashes in STATUS.json.
Caches, probe weights and predictions remain remote. No automatic new model training.
