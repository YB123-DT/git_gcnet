# Paired-view training loss audit

Saved seed66 training logs only. A weight0, B weight0.1. No new inference/training.
Task = .5*(View1 MSE + View2 MSE); total = task + weight*InfoNCE.
All 200 epoch records pass reconstruction checks (absolute tolerance1e-4).

|Arm|Epochs|View1 MSE|View2 MSE|Mean task|Raw InfoNCE|Weighted InfoNCE|Total|Weighted InfoNCE / task %|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|A|1–1|97.3156|81.5110|89.4133|5.1455|0.0000|89.4133|0.00|
|A|1–20|18.1091|18.8859|18.4975|5.5222|0.0000|18.4975|0.00|
|A|21–40|2.0983|2.1121|2.1052|5.7650|0.0000|2.1052|0.00|
|A|41–60|1.2394|1.3703|1.3049|5.6081|0.0000|1.3049|0.00|
|A|61–80|1.0703|1.2056|1.1380|5.5807|0.0000|1.1380|0.00|
|A|81–100|0.8104|0.9692|0.8898|5.4989|0.0000|0.8898|0.00|
|A|100–100|0.7278|0.8898|0.8088|5.5526|0.0000|0.8088|0.00|
|B|1–1|97.3298|81.4959|89.4129|5.1402|0.5140|89.9269|0.57|
|B|1–20|19.7658|20.7127|20.2393|5.0453|0.5045|20.7438|2.49|
|B|21–40|2.2129|2.2208|2.2168|4.8864|0.4886|2.7055|22.04|
|B|41–60|1.4725|1.5707|1.5216|4.0787|0.4079|1.9295|26.81|
|B|61–80|1.0847|1.2042|1.1445|3.5769|0.3577|1.5022|31.25|
|B|81–100|0.8277|0.9835|0.9056|3.4549|0.3455|1.2511|38.15|
|B|100–100|0.7675|0.9230|0.8453|2.9069|0.2907|1.1360|34.39|

Window statistics average the logged epoch means; ratio is ratio of window means.
These are training losses, not loss at each rate-specific best checkpoint.
Cyclic missing rates vary across epochs; one epoch is not an eight-rate average.
A computes raw InfoNCE for logging/equal-compute but multiplies it by0; its projector is not trained by InfoNCE.
Therefore raw A/B InfoNCE is not a comparison of equally trained projectors.
Task losses cover all valid utterances; InfoNCE covers only eligible history anchors.
Scalar loss ratios do not measure gradient magnitude or gradient conflict.
This analysis alone does not establish an oversized contrastive weight as the cause of W-F1 loss.
