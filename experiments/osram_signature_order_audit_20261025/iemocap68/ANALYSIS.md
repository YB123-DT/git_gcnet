# IEMOCAP-6 seed68: mechanism evidence must cover ACC and UA

## Completed frozen intervention

All5folds×8rates (40selected checkpoints) completed. No new training,selection,
mask samples,Local/Base/Gap changes or model-weight writes. Each normal replay
and parent replay was bit-exact. The original numerical source stays21e46f2;
audit implementation326d3c9. Local and remote27-test suites passed. Independent
recomputation checked240 six-class ACC/UA conditions,first-utterance invariance
and equal-fold/rate aggregation. Full predictions remain remote.

Runtime100.288seconds,peak allocated GPU memory694.959MiB,peak RSS2.868GiB,
cmerc-repro on healthy physicalGPU2. This is inference time under shared server
conditions,not a controlled speed benchmark.

| Condition | ACC | UA | ΔACC vs normal | ΔUA vs normal |
| --- | ---: | ---: | ---: | ---: |
| Original Flat |61.507197|60.307075|−0.766216|−0.929139|
| Normal Signature |62.273414|61.236214|0|0|
| Endpoint-preserving interior shuffle,mean of3 draws |62.155781|61.109669|−0.117633|−0.126545|
| Remove antisymmetric second-level area |62.147398|61.115662|−0.126016|−0.120552|

The three shuffle draws are inference controls,not independent training seeds,
and the table averages their metrics rather than ensembles their predictions.
UA is macro recall over all six classes including label0. All scores are equal
arithmetic means across the40fold/rate cells; no sample-size pooling.

## What is supported

Both metrics decrease a little under each complete-dataset control. Thus the
learned predictor uses some interior path/time correspondence and oriented
second-level information. High-missing0.5–0.7 normal ACC/UA are59.938714/58.867712;
area removal gives59.765955/58.700668,changes−0.172759/−0.167045pp.

The full Signature branch has a larger measured advantage over the original
frozen Flat: +0.766216ACC / +0.929139UA. That advantage is present across all
eight rate means and all five fold means. The existing SEED68_ANALYSIS.md
reports the per-rate/per-fold and six-class recall breakdown separately.

## What is not supported

The geometry controls remove only about0.12pp in the full mean,while most of
the original branch gain remains. It is therefore not justified to explain the
whole gain by detailed temporal order or oriented area. The retained endpoint
information,nonlinear endpoint products,counts,current Local processing and
other branch capacity effects remain unseparated.

Effects are not uniform over40conditions: area removal lowers ACC in16cells,
UA in15,both in15. Each shuffle lowers both in13–16cells. This is not evidence
that every fold/missing condition needs the same path component. See all cells
in RESULT.json/CSV rather than only the strongest rate.

CHM is already recurrent and order-sensitive. These controls do not establish
that the original Flat loses all temporal information. The same-checkpoint
interventions may also create out-of-distribution branch features. Seed68 was
chosen for analysis after observing its performance,so this is exploratory
mechanism evidence rather than independent confirmation or significance proof.

## Defensible next question,not an extra launched experiment

The useful distinction is between direct recent-observation content/statistics
and higher-order path structure. A matched trained branch with the same window,
observed latents,time coordinates,Local/counts and learning budget,but without
oriented second-level terms,would address that distinction more directly.
No such new training,additional seeds or broader search was started here.

Remote output:
`/data2/yb/remote_experiments/osram_signature_order_audit_20261025/execution_iemocap68_v1`.
LAUNCH_IEMOCAP68.json records the immutable overlay,numerical source,command,
device and resource checks. VERIFICATION.json preserves all prediction hashes
and independent metric checks. This audit retains the user's Test-best ACC
selection rule,with UA from the same selected weights and same saved mask.
