# Query / addressing descriptive audit

INTERNAL DIAGNOSTIC ONLY

Original seed66 Full checkpoints, frozen inference; no model/loss/head selection changes. Only valid utterances with an active Gap are included, eight forward memory heads. Zero-norm undefined cosines are omitted with explicit per-metric valid/missing counts.
Each seed/rate/modal/head cell is summarized first. Macro values equally average nonempty rates per seed, then seeds. Macro p10/p50/p90 are averages of within-rate quantiles, NOT pooled distribution quantiles. Counts are repeated utterance × active-Gap × head exposures, not independent samples.

|Modality|Head|Raw query cosine mean|Raw query p50|Gap/residual query mean|Read cosine mean|Read p50|Read−raw query mean|Paired N|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|A|ALL|0.958|0.976|0.692|0.593|0.741|-0.365|13768|
|A|0|0.983|0.984|0.503|0.481|0.493|-0.502|1721|
|A|1|0.968|0.970|0.959|0.653|0.713|-0.315|1721|
|A|2|0.928|0.931|0.729|0.752|0.864|-0.176|1721|
|A|3|0.977|0.977|0.568|0.405|0.431|-0.571|1721|
|A|4|0.873|0.878|0.716|0.515|0.683|-0.357|1721|
|A|5|0.978|0.979|0.606|0.557|0.647|-0.421|1721|
|A|6|0.979|0.980|0.947|0.899|0.911|-0.081|1721|
|A|7|0.977|0.977|0.507|0.479|0.622|-0.498|1721|
|ALL|ALL|0.967|0.976|0.760|0.633|0.787|-0.333|40048|
|T|ALL|0.972|0.973|0.800|0.662|0.854|-0.309|13376|
|T|0|0.986|0.986|0.524|0.376|0.338|-0.610|1672|
|T|1|0.959|0.960|0.967|0.476|0.572|-0.484|1672|
|T|2|0.963|0.963|0.930|0.846|0.911|-0.117|1672|
|T|3|0.983|0.984|0.579|0.423|0.418|-0.561|1672|
|T|4|0.958|0.958|0.883|0.803|0.865|-0.155|1672|
|T|5|0.982|0.983|0.917|0.932|0.961|-0.050|1672|
|T|6|0.982|0.982|0.640|0.543|0.577|-0.439|1672|
|T|7|0.959|0.960|0.960|0.898|0.943|-0.061|1672|
|V|ALL|0.971|0.976|0.789|0.645|0.769|-0.325|12904|
|V|0|0.977|0.977|0.912|0.897|0.920|-0.079|1613|
|V|1|0.989|0.989|0.988|0.449|0.523|-0.540|1613|
|V|2|0.943|0.946|0.721|0.509|0.645|-0.434|1613|
|V|3|0.984|0.984|0.823|0.757|0.822|-0.227|1613|
|V|4|0.976|0.977|0.670|0.740|0.809|-0.236|1613|
|V|5|0.952|0.952|0.704|0.561|0.624|-0.390|1613|
|V|6|0.968|0.968|0.841|0.742|0.751|-0.225|1613|
|V|7|0.978|0.978|0.655|0.508|0.589|-0.470|1613|

## Interpretation limits

Raw q_Base/q_Gap cosine is measured BEFORE residual addressing. Gap read uses residualized addressing, so read−raw-query cosine is a descriptive paired difference, NOT an isolated effect of the Memory map. Existing cos(q_Gap,q_residual), rho, and eta are separately preserved. The raw-query/read comparison cannot by itself establish rank collapse, target specificity failure, or that Memory makes distinct actual queries identical.
No arbitrary similar/different threshold, sign fitting, significance test, head routing, or learned classifier is used. Cosine measures angle, not equality or information content. Near-constant rho/eta may reflect addressing construction rather than independent evidence. Norms and quantiles are available in CSVs.
Per-rate weights differ; head IDs and descriptive associations do not prove stable specialization across checkpoints or seeds. Optional rescue/harm join uses stage1 labels only for retrospective reporting; the same utterance label is repeated across active heads. No further diagnostic stage or mechanism is automatically added.
