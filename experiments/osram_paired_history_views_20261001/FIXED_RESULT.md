# Fixed whole-conversation modality comparison

Seed66; existing per-rate Test-oracle checkpoints. Evaluation-only, no retraining.
A=paired task-only; B=paired task+InfoNCE. T=Text.
Full MOSI test:686 utterances; original nonzero-label binary W-F1 uses656.
Each conversation keeps only the named modalities throughout. Single-modality uses miss.7 best;
pairs use miss.3 best. Flat reuses the original seed66 fixed-pattern record, not a multiseed mean.
Same original evaluator; original task head, no projector calls, frozen checkpoint/state hashes.

|Visible|Flat|A|B|A−Flat|B−Flat|B−A|
|---|---:|---:|---:|---:|---:|---:|
|A|35.643|56.986|25.073|21.344|-10.570|-31.914|
|T|85.620|84.540|86.728|-1.080|1.107|2.188|
|V|53.381|60.264|57.779|6.883|4.398|-2.485|
|AT|86.478|84.074|84.077|-2.403|-2.400|0.003|
|AV|57.958|58.661|56.374|0.703|-1.584|-2.287|
|TV|86.754|84.974|83.709|-1.780|-3.045|-1.265|
|No Text|48.994|58.637|46.408|9.643|-2.585|-12.229|
|Text present|86.284|84.529|84.838|-1.755|-1.446|0.309|
|Six-pattern mean|67.639|71.583|65.623|3.944|-2.016|-5.960|

Units:% W-F1, differences in percentage points. Single seed; no significance claim.
A smaller history-deletion flip rate is a different property from these persistent-missing scores.
This comparison does not establish a causal training mechanism. No extra tuning or trials.
