# Lower-LR continuation after constant epoch100

INTERNAL DIAGNOSTIC ONLY. MOSI seed66 Flat/Nested pair, each50additional epochs.
Resume last_training.pt at exactly100, NOT any rate-specific BEST checkpoint.
Source runs: completed constant monitored runs under
`/data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs`.
These reproduced original seed66 scores; sealed source_ad211c0 unchanged.

Only config changes: totalepochs100→150, learning_rate1e-3→1e-4. Constant schedule,
no warmup/cosine, same Adam/batch/loss/model/missing masks/Test-oracle selection.
Restore model, optimizer moments and steps, RNG, trajectory progress, per-rate
selection and BEST references. Change optimizer group LR IN THE RESTORED STATE,
not only config (otherwise restore would overwrite the requested new rate).
Record every actual optimizer LR at epochs101–150. No optimizer reset.

Copy versions independently into separate /data1 output directories. Preserve
original checkpoints/hashes, all cumulativeBEST and complete recovery files.
Report epoch101–150-only BEST alongside cumulative1–150 and original1–100.
Reused Nested sameLR150 experiment is available as budget-only context, not a
replacement for this lowerLR run. No automatic multi-seed/search expansion.

Verification: config differs only in epochs/LR; optimizer moments/RNG/model
preserved by state transformation; live real restored first epochLR1e-4; final
history prefix matches original100, exactly150epochs,50LRrecords and all artifacts.
Launch two persistent jobs on healthy biggpu GPU7 after resource check.
