# Frozen Nested rooted representation probes

INTERNAL DIAGNOSTIC ONLY. One frozen original Nested backbone seed66; use its
existing100epoch per-rate Test-oracle BEST checkpoints and eight missing rates.
No OSRAM/Nested/Flat/head parameter updates. Probe seeds66/67/68 are NOT three
independently trained backbones. No new loss or module on the deployed model.

Primary Flat representation: SAME Nested checkpoint's emotion_norm hidden after
original raw Local/Base/Gap Flat adapter, before any Nested correction. Base and
Gap retained. This is R_off hidden, not separately trained original Flat hidden.
Full Nested predictions and original head R_off predictions also recorded.

Two exact graph taps: actual rooted tensor is the middle64 coordinates of
core.readout's [original token; rooted; rooted global mean] input; predecoder is
core output AFTER its readout MLP, before normal trained decoder. Decoder was
zero-initialized at training start; do not assume its trained weights are zero.
Pack into original33 slots (Local1, Base8,Gap-A8,Gap-T8,Gap-V8), flatten2112.
No role/head pooling or random projection. First/empty-history/padding zero,
inactive Gap/root slots zero; retain actual runtime packed root identities.

Six arms: Flat; Rooted; Flat+Rooted; Flat+DonorRooted; PreDecoder;
Flat+PreDecoder. Each uses3715->32->GELU->1 (1600+2112+availability3),118945params;
unprovided blocks zero. Inputs train-only standardized, inactive root coordinates
re-zeroed. Donor sampled TRAIN only, same availability/has-history and different
conversation, without labels; common eligible samples for all arms. Save donor
indices/coverage and train/test IDs. Equal parameter counts do not imply equal
effective capacity; donor is an additional-capacity negative control.

Probe protocol: Adam.001,weight_decay1e-5,batch128, taskMSE, fixed100epochs;
train labels only for updates, final epoch only, no test-based probe selection.
No threshold/hyperparameter sweep. Evaluate original nonzero-label W-F1/ACC
with threshold0; MSE secondary. Equal-rate mean and .5/.6/.7 high mean, each
probe seed separately then mean/std. Also seven observed availability groups.
No validation request or test-gradient training.

Evidence: joint probe above Flat-alone AND donor supports extra easily decodable
information under this probe class. Failure does NOT prove information absent:
probe optimization/class and fixed budget limit the conclusion. Positive result
does not identify semantics or establish which deployment mechanism will work.

Implementation: read-only hooks/cached features, strict checkpoint load and hash
checks, Full archive label/mask/prediction parity. CPU probes on remote, GPU7
for frozen extraction. No new dependencies; GPU4 forbidden. Separate sealed
source/output, immutable existing capacity trainings.
