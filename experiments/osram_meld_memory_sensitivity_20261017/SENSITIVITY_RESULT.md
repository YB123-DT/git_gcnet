# Frozen MELD alpha/beta/head sensitivity

INTERNAL DIAGNOSTIC ONLY. COMPLETE. No retraining, optimizer, HPO or new mask.

## Scope and verification

Original ECERC Flat1600 with Local Skip; seed66; rates0.0/0.4/0.7 only.
Each rate has all2,610test utterances from280conversations. These are repeated
observations of the same test set under three masks/checkpoint states, not
7,830independent examples or an eight-rate mean. ACC-selected source epochs
are11/5/10 respectively. All three checkpoint SHA256s match ALPHA_BETA.json.

Numerical source:b4a313841f098487a6482d2712bdac94aa8e319e, restored strictly.
Frozen driver:801f4a29bc02ac29528d428d5b4f78738a59c85e. Exact winning-epoch
masks come from osram_meld_testbest_ua_20261014/exact/source_0, NOT the final
evaluation masks in the checkpoint root. Features retain the accepted SHA256.
Joint A/T/V Block Delta Write, embedded beta, ridge0.001, step0.6, Queries,
Local, Flat, Skip, task head and causal read-before-write are unchanged.

For every condition, current Local/Query/K/V/availability were exactly equal
to the original. Each parameter condition independently scanned from zero.
All saved baseline labels/classes matched exactly; captured readout replay had
maximum logit error1.431e-6 and unchanged classes. First-turn history remained
zero, parameters were restored bit-for-bit, no gradients accumulated and CUDA
RNG was unchanged. Generated IDs/labels agreed across rates; historical NPZs
have no IDs, so archived ID equality is not claimed.

Remote biggpu GPU3(UUID GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a), logical cuda:0;
cmerc-repro Python3.10.13/Torch1.13.1+cu117/CUDA11.7/NumPy1.26.4/sklearn1.3.2.
The frozen run took328.86seconds,2026-10-17T11:17:01Z–11:22:30Z.
Observed process GPU allocation was2,016MiB; this is a sampled reading, not a
recorded lifetime peak. No other process was stopped. See README.md for command.

## 1. Task sensitivity of every preregistered setting

Entries are ACC / UA(percent). UA uses all seven classes. For other metrics,
all groups and per-class recall/support see FROZEN_METRICS.csv and FROZEN_SUMMARY.json.
No setting is selected as a new model or an optimal Memory hyperparameter.

| Setting | miss0.0 | miss0.4 | miss0.7 |
|---|---:|---:|---:|
|Original learned alpha/beta|66.667 /44.454|60.307 /32.850|55.057 /25.161|
|beta0.05|66.667 /44.454|60.345 /32.861|55.096 /25.197|
|beta0.50|66.667 /44.454|60.307 /32.850|55.057 /25.161|
|beta0.95|66.667 /44.454|60.307 /32.850|55.057 /25.161|
|alpha0.950|66.437 /44.061|60.307 /32.820|55.057 /25.155|
|alpha0.980|66.667 /44.454|60.307 /32.850|55.057 /25.161|
|alpha0.995|66.705 /44.466|60.307 /32.850|55.096 /25.203|

| Setting | miss0.0 CE | miss0.4 CE | miss0.7 CE | Corrections/harms by rate |
|---|---:|---:|---:|---|
|Original|1.05526674|1.20381916|1.45282030|0/0;0/0;0/0|
|beta0.05|1.05538058|1.20398307|1.45284140|0/0;1/0;1/0|
|beta0.50|1.05526686|1.20381904|1.45282030|0/0;0/0;0/0|
|beta0.95|1.05526066|1.20381021|1.45281911|0/0;0/0;0/0|
|alpha0.950|1.05574942|1.20450056|1.45249200|0/6;1/1;1/1|
|alpha0.980|1.05521154|1.20378602|1.45287454|0/0;0/0;0/0|
|alpha0.995|1.05495179|1.20340300|1.45320809|1/0;0/0;1/0|

Class changes can also be wrong-to-different-wrong; corrections+harms need not
equal all prediction changes. Unchanged ACC/UA does not imply unchanged logits,
CE or W-F1. Tiny CE differences near1e-7 can include readout rounding noise.

## 2. Beta: direct write sensitivity versus propagation

At each actual original joint write, reuse exactly the SAME original pre-write
state,K,V and availability. Only beta changes in the unchanged block_write.
The relative quantity is norm(deltaM_beta-deltaM_original)/norm(deltaM_original),
computed per head then averaged over head/valid utterance. Absolute change is
the mean per-head Frobenius norm, NOT the norm of the concatenated eight heads.

| Rate | beta | Relative update change mean /p95(percent) | Absolute post-state difference |
|---:|---:|---:|---:|
|0.0|0.05|2.1328 /2.4085|0.097785|
|0.0|0.50|0.00049 /0.00058|0.000025|
|0.0|0.95|0.1156 /0.1312|0.005306|
|0.4|0.05|1.9491 /2.2742|0.062134|
|0.4|0.50|0.00052 /0.00080|0.000021|
|0.4|0.95|0.1054 /0.1237|0.003366|
|0.7|0.05|1.7645 /1.7647|0.046886|
|0.7|0.50|0.00022 /0.00045|0.000008|
|0.7|0.95|0.0948 /0.0951|0.002523|

For full-trajectory beta0.05, mean pre-read state changes were1.483/1.439/1.424%;
Base changes1.559/1.462/1.474%; active-Gap changesN/A/1.622/1.526%, by rate.
Beta0.95 caused Base changes0.0845/0.0791/0.0793% and active-Gap changes
N/A/0.0878/0.0821%. First/no-history rows are excluded. These are per-head
relative-norm means, not source-retrieval accuracy or a semantic information rate.

### Active Gram spectra and effective ridge

Use only observed columns and valid rows. No fake zero eigenvalues from missing
columns are counted. The equivalent unscaled solve has diagonal ridge/beta;
actual learned effective ridge entries were0.001998–0.002013.

| Rate | Observed columns | Writes per head | Range across heads of median min-eigenvalue/ridge |
|---:|---:|---:|---:|
|0.0|3|2610|236.82–420.86|
|0.4|1|910|499.11–499.87|
|0.4|2|1271|302.09–470.20|
|0.4|3|429|227.60–394.21|
|0.7|1|2610|499.22–500.11|

These are weighted Gram eigenvalues divided by original ridge0.001. Across all
62,640valid write/head spectra, no active eigenvalue was below ridge. Even the
smallest observed ratio was90.77(0.0),132.12(0.4),496.89(0.7).

Interpretation:around the trained beta approximately0.5,the embedded scaling of
both K and V nearly cancels through the joint solve when ridge is small relative
to the active Gram. The measured beta intervention is consequently much weaker
than its numeric range suggests. This is NOT evidence of a detached writer,
sigmoid saturation,or that write_step would have the same sensitivity. It also
does not prove that another trained Memory configuration cannot help.

## 3. Alpha: read differences are real, task benefits are not uniform

Full-trajectory alpha0.95 changed mean Base reads by6.724/7.066/7.902% and active
Gap byN/A/8.187/8.398%, by rate. Alpha0.995 changed Base by5.244/4.479/6.018%
and active Gap byN/A/5.263/6.429%. This did not translate into a consistent
direction of CE/ACC/UA improvement. There is no basis here to select a global
new retention constant.

For the explicit lag test,start from each SAME original post-write source state,
change alpha only afterward,and replay original future joint writes. Read the
target BEFORE its write with its captured original Query/Local. Learned-alpha
normal forks reproduce original pre-write states/logits/classes. At each fixed
lag a target has one source; targets repeat across lags/settings, not as
independent observations. Different lags also have different eligible cohorts.

Mean CE change versus matched learned-alpha fork(negative is lower CE):

| Rate | alpha | lag1(N2330) | lag2(N2062) | lag4(N1577) | lag8(N837) |
|---:|---:|---:|---:|---:|---:|
|0.0|0.950|+.000297|+.000379|+.000599|+.000756|
|0.0|0.980|−.000035|−.000043|−.000069|−.000092|
|0.0|0.995|−.000195|−.000244|−.000401|−.000522|
|0.4|0.950|+.000337|+.000541|+.000785|+.000888|
|0.4|0.980|−.000016|−.000025|−.000038|−.000044|
|0.4|0.995|−.000191|−.000311|−.000462|−.000539|
|0.7|0.950|−.000044|−.000225|−.000318|−.000785|
|0.7|0.980|+.000006|+.000029|+.000049|+.000132|
|0.7|0.995|+.000039|+.000183|+.000325|+.000913|

Read changes at actual lags can also be recovered without new inference for
the **first source only**:its post-write state is alpha-independent because
initial Memory is zero. Thus saved full-trajectory rows at utterance_index=d
are exactly source0-anchored lag-d forks. A dedicated CPU identity test verifies
this. Filter state_changes and utterances by alpha condition and index1/2/4/8;
join conversation/index/condition,verify prior_valid_utterances=index,then average
the saved per-head read ratios. This is NOT the all-sources cohort above.

At lag8,this first-source cohort has144targets; active Gap covers0/120/144targets
for rates0.0/0.4/0.7. Mean relative read changes(percent):

| Rate | alpha | Base | Active Gap | Target deltaCE |
|---:|---:|---:|---:|---:|
|0.0|0.950|7.743|N/A|−.000647|
|0.0|0.980|1.111|N/A|+.000086|
|0.0|0.995|5.982|N/A|+.000491|
|0.4|0.950|8.079|9.715|−.000193|
|0.4|0.980|0.415|0.500|+.000013|
|0.4|0.995|5.043|6.140|+.000158|
|0.7|0.950|9.607|10.037|−.001017|
|0.7|0.980|1.237|1.292|+.000132|
|0.7|0.995|7.223|7.560|+.000743|

This provides read and task sensitivity at a genuine lag,not just a histogram
of conversation length. It does not isolate survival of a single source
association:intervening writes remain present. The all-sources and first-source
cohorts differ,so opposite average CE directions between those tables are not
a numerical inconsistency.

## 4. Head use is conditional, with small marginal class effects

Use one unchanged baseline scan. Zero only the head's64forward dimensions in
Base OR all active Gap slots before the original Flat. Local/Skip and all other
read values remain unchanged. Head numbers below are1-based. Positive numbers
mean Full-minus-head-removal is positive:removing that head reduces ACC.

| Head | Base deltaACC .0 | Base .4 | Base .7 | Gap .4 | Gap .7 |
|---:|---:|---:|---:|---:|---:|
|1|+.0766|+.0383|.0000|+.0383|−.0766|
|2|.0000|.0000|+.0383|−.0383|−.0766|
|3|+.2299|.0000|−.0383|−.0383|+.0766|
|4|+.1916|+.0383|−.0766|.0000|−.1149|
|5|+.0766|−.0383|.0000|.0000|+.0766|
|6|−.1533|.0000|−.0383|.0000|−.0766|
|7|+.0766|.0000|.0000|.0000|.0000|
|8|−.0383|.0000|−.0383|−.0383|.0000|

Units:percentage points. At0.0 all Gap slots are inactive by definition:all
Gap-head class effects are zero,not evidence of dead heads. Complete UA/W-F1,
CE,recall,corrections/harms and availability-stratified effects are in
HEAD_ABLATION.csv/FROZEN_METRICS.csv. For example at0.0,Base Head3 contributes
+0.2299ACC/+0.4510UA pp; at0.7 its ACC contribution is−0.0383pp. Removing
Gap Head4 at0.7 yields5corrections/2harms,but that is only3net examples.

Head CE-effect profile correlations compare each utterance's
CE(remove-head)−CE(Full),not raw vectors in differently learned head bases.
Mean off-diagonal Pearson correlations for Base are0.135/0.244/0.194 by rate;
for active Gap conditions they are0.441(0.4),0.350(0.7). Individual Base pairs
range from−0.654to+0.876 across these conditions. Thus their task effects are
not universally identical,though this does not identify semantic roles.

Numerical safeguard:all eight Gap replays at0.0 are exactly identical to one
another,but differ from original forward logits by up to1.431e-6. Naively
correlating those shared tiny CE errors produced a spurious1.0. Postprocessing
now nulls profiles whose entire CE effect lies within twice the measured logit
parity error(a valid CE noise bound). The corrected0.0 Gap correlation isN/A.
The initial remote analysis_724 correlation output is superseded by
analysis_noiseaware; raw inference and performance scores were not changed.
Other heads with no class changes can still change logits/CE; no head is
declared unused,prunable or specialized solely from this one-seed audit.

## Conclusion and limits

- The trained beta neighborhood has very weak functional sensitivity in this
  original embedded-ridge configuration. Beta saturation/large ridge dominance
  is not supported; broader numeric beta changes need not be large write changes.
- Alpha genuinely changes Memory/reads,but these interventions do not give
  a uniform task-level remedy. Stronger retention is not universally better.
- Heads have nonidentical and condition-dependent task effects,but most
  individual class effects are small. This is insufficient support for adding
  a head Gate,assigning semantic roles,or claiming a new second Block.
- This does NOT establish whether useful history was lost in writing,missed
  by queries,or underused by Flat. It narrows specific parameter explanations;
  it does not completely localize the scientific bottleneck.

All presets and negative/mixed effects are retained. No hypothesis-test p-value,
independent-seed uncertainty or optimized deployment score is claimed. Labels
are used only to evaluate frozen predictions,never as model inputs or to train
a routing rule. Source selection remains Test-oracle,so none of these numbers
are independent-test paper claims. No additional training is launched.

## Artifacts and checks

Remote raw output:frozen_801 under
/data2/yb/remote_experiments/osram_meld_memory_sensitivity_20261017/.
It contains saved logits/IDs and utterance/state/spectral/direct-beta/lag streams.
Local compact artifacts:FROZEN_SUMMARY.json,FROZEN_METRICS.csv,ANALYSIS.json,
HEAD_ABLATION.csv,LAGGED_ALPHA.csv. ANALYSIS.json records its own analyzer SHA256.
The running model driver stayed801f4a2; analysis-only changes do not alter it.

Tests cover parameters restored on exceptions,current inputs/queries/KV equal,
first-turn/read-before-write,padding/inactive slots,active-column spectra,
embedded-beta effective-ridge algebra,joint-column permutation,head slicing,
normal-alpha lag parity,source0 fork identity,seven-class UA,artifact alignment,
and rejection of shared readout roundoff as head evidence. See test files and
README.md for the complete command and tests-first implementation history.
