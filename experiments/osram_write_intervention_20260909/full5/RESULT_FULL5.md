# Four-mode frozen-checkpoint evaluation: five seeds

INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.

Seeds: 66–70. Missing rates: 0, 0.1, 0.3, 0.5, 0.7. Dynamic = `global` (norm-matched); fixed comparator = `fixed0.9`.

All 100 evaluations are present. Each seed uses the single checkpoint/config declared below for its 20 evaluations. Weight-unchanged flags and both mask flags and per-rate hashes passed validation. This checks metadata consistency, not an independent rehash of checkpoint contents.

| Seed | Epoch | Checkpoint | SHA256 |
|---|---|---|---|
| 66 | 33 | `/data2/yb/remote_experiments/osram_forward_only_iemocap_20260908/iemocap4/seed_66/best.pt` | `79aef3dab7d61aa41dabe5bd9f5cf8044f8c5fbf9c556684986e485f5a7be461` |
| 67 | 54 | `/data2/yb/remote_experiments/osram_forward_only_iemocap_20260908/iemocap4/seed_67/best.pt` | `5ce1e0e17d91b58476e8fa4fc68575e8f99c0a8a0f54749d20851484510c0a16` |
| 68 | 38 | `/data2/yb/remote_experiments/osram_forward_only_iemocap_20260908/iemocap4/seed_68/best.pt` | `c6458ac89b04de47b0377cb7bd7c82b8ba28676193ad6af4e649a175f38be840` |
| 69 | 30 | `/data2/yb/remote_experiments/osram_forward_only_iemocap_20260908/iemocap4/seed_69/best.pt` | `98f355a5d434c973eb6a6d5c4d48df137d25f14f7f28d44509127e899ee71391` |
| 70 | 64 | `/data2/yb/remote_experiments/osram_forward_only_iemocap_20260908/iemocap4/seed_70/best.pt` | `5a9d795cd796890fbaaf15631c0557855b20881ff0ec826fea7a45cd5167d1a0` |

## Task metrics

Raw metric units; mean ± sample SD across five seeds. `all` first averages the five rates within each seed. Positive/zero/negative counts apply to paired dynamic minus fixed differences, not unpaired pooled evaluations.

| Rate | Mode | Metric | Mean ± SD | + / 0 / − |
|---|---|---|---|---|
| 0 | fixed0.9 | accuracy | 0.844803 ± 0.004580 | — |
| 0 | fixed0.9 | macro_f1 | 0.841667 ± 0.006648 | — |
| 0 | fixed0.9 | weighted_f1 | 0.844714 ± 0.004700 | — |
| 0 | global | accuracy | 0.843513 ± 0.004465 | — |
| 0 | global | macro_f1 | 0.840072 ± 0.006085 | — |
| 0 | global | weighted_f1 | 0.843344 ± 0.004727 | — |
| 0 | global-minus-fixed0.9 | accuracy | -0.001289 ± 0.002390 | 2 / 0 / 3 |
| 0 | global-minus-fixed0.9 | macro_f1 | -0.001595 ± 0.002725 | 2 / 0 / 3 |
| 0 | global-minus-fixed0.9 | weighted_f1 | -0.001370 ± 0.002412 | 2 / 0 / 3 |
| 0 | protected | accuracy | 0.843513 ± 0.004465 | — |
| 0 | protected | macro_f1 | 0.840072 ± 0.006085 | — |
| 0 | protected | weighted_f1 | 0.843344 ± 0.004727 | — |
| 0 | reference | accuracy | 0.843513 ± 0.004465 | — |
| 0 | reference | macro_f1 | 0.840072 ± 0.006085 | — |
| 0 | reference | weighted_f1 | 0.843344 ± 0.004727 | — |
| 0.1 | fixed0.9 | accuracy | 0.842224 ± 0.004781 | — |
| 0.1 | fixed0.9 | macro_f1 | 0.839350 ± 0.007218 | — |
| 0.1 | fixed0.9 | weighted_f1 | 0.842094 ± 0.004657 | — |
| 0.1 | global | accuracy | 0.839645 ± 0.006814 | — |
| 0.1 | global | macro_f1 | 0.836530 ± 0.010370 | — |
| 0.1 | global | weighted_f1 | 0.839552 ± 0.006817 | — |
| 0.1 | global-minus-fixed0.9 | accuracy | -0.002579 ± 0.002636 | 1 / 0 / 4 |
| 0.1 | global-minus-fixed0.9 | macro_f1 | -0.002821 ± 0.003210 | 1 / 0 / 4 |
| 0.1 | global-minus-fixed0.9 | weighted_f1 | -0.002542 ± 0.002602 | 1 / 0 / 4 |
| 0.1 | protected | accuracy | 0.839323 ± 0.005302 | — |
| 0.1 | protected | macro_f1 | 0.836568 ± 0.008058 | — |
| 0.1 | protected | weighted_f1 | 0.839243 ± 0.005313 | — |
| 0.1 | reference | accuracy | 0.839323 ± 0.006257 | — |
| 0.1 | reference | macro_f1 | 0.836496 ± 0.009022 | — |
| 0.1 | reference | weighted_f1 | 0.839241 ± 0.006192 | — |
| 0.3 | fixed0.9 | accuracy | 0.822079 ± 0.006988 | — |
| 0.3 | fixed0.9 | macro_f1 | 0.814368 ± 0.007608 | — |
| 0.3 | fixed0.9 | weighted_f1 | 0.821925 ± 0.006164 | — |
| 0.3 | global | accuracy | 0.821112 ± 0.006932 | — |
| 0.3 | global | macro_f1 | 0.813307 ± 0.007951 | — |
| 0.3 | global | weighted_f1 | 0.820897 ± 0.006196 | — |
| 0.3 | global-minus-fixed0.9 | accuracy | -0.000967 ± 0.001924 | 1 / 1 / 3 |
| 0.3 | global-minus-fixed0.9 | macro_f1 | -0.001061 ± 0.001601 | 2 / 0 / 3 |
| 0.3 | global-minus-fixed0.9 | weighted_f1 | -0.001028 ± 0.001940 | 1 / 0 / 4 |
| 0.3 | protected | accuracy | 0.819339 ± 0.007416 | — |
| 0.3 | protected | macro_f1 | 0.811972 ± 0.007050 | — |
| 0.3 | protected | weighted_f1 | 0.819091 ± 0.006653 | — |
| 0.3 | reference | accuracy | 0.820145 ± 0.006679 | — |
| 0.3 | reference | macro_f1 | 0.812021 ± 0.008110 | — |
| 0.3 | reference | weighted_f1 | 0.819933 ± 0.005976 | — |
| 0.5 | fixed0.9 | accuracy | 0.803223 ± 0.011780 | — |
| 0.5 | fixed0.9 | macro_f1 | 0.793092 ± 0.013105 | — |
| 0.5 | fixed0.9 | weighted_f1 | 0.803385 ± 0.011135 | — |
| 0.5 | global | accuracy | 0.801612 ± 0.014609 | — |
| 0.5 | global | macro_f1 | 0.791542 ± 0.016618 | — |
| 0.5 | global | weighted_f1 | 0.801910 ± 0.013742 | — |
| 0.5 | global-minus-fixed0.9 | accuracy | -0.001612 ± 0.003648 | 2 / 0 / 3 |
| 0.5 | global-minus-fixed0.9 | macro_f1 | -0.001550 ± 0.004313 | 2 / 0 / 3 |
| 0.5 | global-minus-fixed0.9 | weighted_f1 | -0.001476 ± 0.003548 | 2 / 0 / 3 |
| 0.5 | protected | accuracy | 0.803062 ± 0.012400 | — |
| 0.5 | protected | macro_f1 | 0.793328 ± 0.015898 | — |
| 0.5 | protected | weighted_f1 | 0.803108 ± 0.011722 | — |
| 0.5 | reference | accuracy | 0.800967 ± 0.013639 | — |
| 0.5 | reference | macro_f1 | 0.790472 ± 0.015559 | — |
| 0.5 | reference | weighted_f1 | 0.801181 ± 0.012851 | — |
| 0.7 | fixed0.9 | accuracy | 0.780661 ± 0.017556 | — |
| 0.7 | fixed0.9 | macro_f1 | 0.772950 ± 0.016476 | — |
| 0.7 | fixed0.9 | weighted_f1 | 0.781223 ± 0.016229 | — |
| 0.7 | global | accuracy | 0.780016 ± 0.020089 | — |
| 0.7 | global | macro_f1 | 0.772648 ± 0.019424 | — |
| 0.7 | global | weighted_f1 | 0.780661 ± 0.018546 | — |
| 0.7 | global-minus-fixed0.9 | accuracy | -0.000645 ± 0.003923 | 2 / 1 / 2 |
| 0.7 | global-minus-fixed0.9 | macro_f1 | -0.000302 ± 0.004103 | 3 / 0 / 2 |
| 0.7 | global-minus-fixed0.9 | weighted_f1 | -0.000562 ± 0.003882 | 2 / 0 / 3 |
| 0.7 | protected | accuracy | 0.780016 ± 0.020974 | — |
| 0.7 | protected | macro_f1 | 0.772494 ± 0.020459 | — |
| 0.7 | protected | weighted_f1 | 0.780364 ± 0.019595 | — |
| 0.7 | reference | accuracy | 0.776954 ± 0.018210 | — |
| 0.7 | reference | macro_f1 | 0.769258 ± 0.016705 | — |
| 0.7 | reference | weighted_f1 | 0.777502 ± 0.016819 | — |
| all | fixed0.9 | accuracy | 0.818598 ± 0.005929 | — |
| all | fixed0.9 | macro_f1 | 0.812285 ± 0.006595 | — |
| all | fixed0.9 | weighted_f1 | 0.818668 ± 0.005197 | — |
| all | global | accuracy | 0.817180 ± 0.007255 | — |
| all | global | macro_f1 | 0.810820 ± 0.008358 | — |
| all | global | weighted_f1 | 0.817273 ± 0.006538 | — |
| all | global-minus-fixed0.9 | accuracy | -0.001418 ± 0.001965 | 1 / 0 / 4 |
| all | global-minus-fixed0.9 | macro_f1 | -0.001466 ± 0.002006 | 1 / 0 / 4 |
| all | global-minus-fixed0.9 | weighted_f1 | -0.001396 ± 0.001970 | 1 / 0 / 4 |
| all | protected | accuracy | 0.817051 ± 0.006832 | — |
| all | protected | macro_f1 | 0.810887 ± 0.007687 | — |
| all | protected | weighted_f1 | 0.817030 ± 0.006140 | — |
| all | reference | accuracy | 0.816180 ± 0.006071 | — |
| all | reference | macro_f1 | 0.809664 ± 0.006875 | — |
| all | reference | weighted_f1 | 0.816240 ± 0.005384 | — |

## Per-seed five-rate weighted F1

| Seed | Reference | Protected | Dynamic | Fixed 0.9 | Dynamic − fixed |
|---|---|---|---|---|---|
| 66 | 0.817843 | 0.819028 | 0.819131 | 0.820108 | -0.000977 |
| 67 | 0.814469 | 0.815001 | 0.815827 | 0.815460 | 0.000367 |
| 68 | 0.816469 | 0.817090 | 0.816780 | 0.818342 | -0.001562 |
| 69 | 0.808788 | 0.808570 | 0.808224 | 0.812883 | -0.004659 |
| 70 | 0.823632 | 0.825461 | 0.826401 | 0.826548 | -0.000147 |

## Retention and current-write fit

Head/record observations are averaged inside each run by the evaluator; the run means are then averaged with equal seed weight. Counts are not independent replicates. Modalities remain separate. Missing retention at rate zero is omitted, not imputed as zero. Full write-audit means are in CSV.

| Diagnostic | Rate | Mode | Modality | Metric | Mean ± SD |
|---|---|---|---|---|---|
| retention | 0.1 | fixed0.9 | audio | decay_damage | 0.0090446 ± 0.000313853 |
| retention | 0.1 | fixed0.9 | audio | err_decay | 0.102941 ± 0.00857481 |
| retention | 0.1 | fixed0.9 | audio | err_post | 0.319138 ± 0.0152478 |
| retention | 0.1 | fixed0.9 | audio | write_damage | 0.216197 ± 0.0140325 |
| retention | 0.1 | fixed0.9 | text | decay_damage | 0.0107795 ± 0.000466983 |
| retention | 0.1 | fixed0.9 | text | err_decay | 0.139954 ± 0.00578354 |
| retention | 0.1 | fixed0.9 | text | err_post | 0.319507 ± 0.0112279 |
| retention | 0.1 | fixed0.9 | text | write_damage | 0.179553 ± 0.0145824 |
| retention | 0.1 | fixed0.9 | visual | decay_damage | 0.00959174 ± 0.000247488 |
| retention | 0.1 | fixed0.9 | visual | err_decay | 0.10752 ± 0.00638608 |
| retention | 0.1 | fixed0.9 | visual | err_post | 0.315596 ± 0.0218428 |
| retention | 0.1 | fixed0.9 | visual | write_damage | 0.208075 ± 0.0172075 |
| retention | 0.1 | global | audio | decay_damage | 0.0162387 ± 0.000449135 |
| retention | 0.1 | global | audio | err_decay | 0.0520966 ± 0.00643448 |
| retention | 0.1 | global | audio | err_post | 0.308861 ± 0.0161384 |
| retention | 0.1 | global | audio | write_damage | 0.256765 ± 0.0139409 |
| retention | 0.1 | global | text | decay_damage | 0.0164504 ± 0.000593412 |
| retention | 0.1 | global | text | err_decay | 0.0581939 ± 0.00806172 |
| retention | 0.1 | global | text | err_post | 0.281946 ± 0.0137539 |
| retention | 0.1 | global | text | write_damage | 0.223752 ± 0.0151268 |
| retention | 0.1 | global | visual | decay_damage | 0.016049 ± 0.000605186 |
| retention | 0.1 | global | visual | err_decay | 0.059824 ± 0.00730337 |
| retention | 0.1 | global | visual | err_post | 0.305491 ± 0.0206856 |
| retention | 0.1 | global | visual | write_damage | 0.245667 ± 0.0168858 |
| retention | 0.1 | global-minus-fixed0.9 | audio | decay_damage | 0.00719411 ± 0.00040455 |
| retention | 0.1 | global-minus-fixed0.9 | audio | err_decay | -0.0508446 ± 0.00574143 |
| retention | 0.1 | global-minus-fixed0.9 | audio | err_post | -0.0102767 ± 0.00454984 |
| retention | 0.1 | global-minus-fixed0.9 | audio | write_damage | 0.0405679 ± 0.00775529 |
| retention | 0.1 | global-minus-fixed0.9 | text | decay_damage | 0.0056709 ± 0.000313516 |
| retention | 0.1 | global-minus-fixed0.9 | text | err_decay | -0.0817604 ± 0.00578886 |
| retention | 0.1 | global-minus-fixed0.9 | text | err_post | -0.0375611 ± 0.00418669 |
| retention | 0.1 | global-minus-fixed0.9 | text | write_damage | 0.0441993 ± 0.00287009 |
| retention | 0.1 | global-minus-fixed0.9 | visual | decay_damage | 0.00645722 ± 0.000603412 |
| retention | 0.1 | global-minus-fixed0.9 | visual | err_decay | -0.0476965 ± 0.00296982 |
| retention | 0.1 | global-minus-fixed0.9 | visual | err_post | -0.0101052 ± 0.00312856 |
| retention | 0.1 | global-minus-fixed0.9 | visual | write_damage | 0.0375913 ± 0.00555924 |
| retention | 0.1 | protected | audio | decay_damage | 0.0171019 ± 0.000301462 |
| retention | 0.1 | protected | audio | err_decay | 0.0368969 ± 0.00295232 |
| retention | 0.1 | protected | audio | err_post | 0.0369577 ± 0.00295695 |
| retention | 0.1 | protected | audio | write_damage | 6.08946e-05 ± 7.27398e-06 |
| retention | 0.1 | protected | text | decay_damage | 0.017464 ± 0.000464478 |
| retention | 0.1 | protected | text | err_decay | 0.0441587 ± 0.006658 |
| retention | 0.1 | protected | text | err_post | 0.044234 ± 0.0066485 |
| retention | 0.1 | protected | text | write_damage | 7.5367e-05 ± 1.11192e-05 |
| retention | 0.1 | protected | visual | decay_damage | 0.0170385 ± 0.000529933 |
| retention | 0.1 | protected | visual | err_decay | 0.0398812 ± 0.00470925 |
| retention | 0.1 | protected | visual | err_post | 0.0399376 ± 0.00470886 |
| retention | 0.1 | protected | visual | write_damage | 5.64138e-05 ± 1.12947e-05 |
| retention | 0.1 | reference | audio | decay_damage | 0.0173442 ± 0.00051005 |
| retention | 0.1 | reference | audio | err_decay | 0.0486262 ± 0.00714248 |
| retention | 0.1 | reference | audio | err_post | 0.333583 ± 0.0162634 |
| retention | 0.1 | reference | audio | write_damage | 0.284957 ± 0.0158716 |
| retention | 0.1 | reference | text | decay_damage | 0.0171984 ± 0.000516278 |
| retention | 0.1 | reference | text | err_decay | 0.052608 ± 0.00691185 |
| retention | 0.1 | reference | text | err_post | 0.305027 ± 0.0161445 |
| retention | 0.1 | reference | text | write_damage | 0.252419 ± 0.0181042 |
| retention | 0.1 | reference | visual | decay_damage | 0.0168971 ± 0.000435604 |
| retention | 0.1 | reference | visual | err_decay | 0.0571487 ± 0.00612599 |
| retention | 0.1 | reference | visual | err_post | 0.332519 ± 0.0234055 |
| retention | 0.1 | reference | visual | write_damage | 0.27537 ± 0.0206722 |
| retention | 0.3 | fixed0.9 | audio | decay_damage | 0.00809984 ± 0.00036264 |
| retention | 0.3 | fixed0.9 | audio | err_decay | 0.150897 ± 0.00872386 |
| retention | 0.3 | fixed0.9 | audio | err_post | 0.321302 ± 0.0174642 |
| retention | 0.3 | fixed0.9 | audio | write_damage | 0.170405 ± 0.0108456 |
| retention | 0.3 | fixed0.9 | text | decay_damage | 0.00981543 ± 0.000428408 |
| retention | 0.3 | fixed0.9 | text | err_decay | 0.177599 ± 0.0121039 |
| retention | 0.3 | fixed0.9 | text | err_post | 0.328162 ± 0.0169991 |
| retention | 0.3 | fixed0.9 | text | write_damage | 0.150563 ± 0.00788666 |
| retention | 0.3 | fixed0.9 | visual | decay_damage | 0.00884652 ± 0.000212815 |
| retention | 0.3 | fixed0.9 | visual | err_decay | 0.150182 ± 0.00843066 |
| retention | 0.3 | fixed0.9 | visual | err_post | 0.315796 ± 0.0152906 |
| retention | 0.3 | fixed0.9 | visual | write_damage | 0.165613 ± 0.0119798 |
| retention | 0.3 | global | audio | decay_damage | 0.0121438 ± 0.00051749 |
| retention | 0.3 | global | audio | err_decay | 0.112866 ± 0.00787703 |
| retention | 0.3 | global | audio | err_post | 0.305793 ± 0.0152004 |
| retention | 0.3 | global | audio | write_damage | 0.192927 ± 0.00803291 |
| retention | 0.3 | global | text | decay_damage | 0.0129861 ± 0.000525622 |
| retention | 0.3 | global | text | err_decay | 0.117243 ± 0.0121259 |
| retention | 0.3 | global | text | err_post | 0.294163 ± 0.0157376 |
| retention | 0.3 | global | text | write_damage | 0.17692 ± 0.00577519 |
| retention | 0.3 | global | visual | decay_damage | 0.0122826 ± 0.000450936 |
| retention | 0.3 | global | visual | err_decay | 0.115175 ± 0.00888207 |
| retention | 0.3 | global | visual | err_post | 0.302014 ± 0.0138623 |
| retention | 0.3 | global | visual | write_damage | 0.186839 ± 0.0148407 |
| retention | 0.3 | global-minus-fixed0.9 | audio | decay_damage | 0.00404396 ± 0.000164998 |
| retention | 0.3 | global-minus-fixed0.9 | audio | err_decay | -0.0380306 ± 0.00347805 |
| retention | 0.3 | global-minus-fixed0.9 | audio | err_post | -0.0155084 ± 0.00369374 |
| retention | 0.3 | global-minus-fixed0.9 | audio | write_damage | 0.0225222 ± 0.00513831 |
| retention | 0.3 | global-minus-fixed0.9 | text | decay_damage | 0.00317062 ± 0.000172214 |
| retention | 0.3 | global-minus-fixed0.9 | text | err_decay | -0.0603563 ± 0.00434925 |
| retention | 0.3 | global-minus-fixed0.9 | text | err_post | -0.0339992 ± 0.0045056 |
| retention | 0.3 | global-minus-fixed0.9 | text | write_damage | 0.0263571 ± 0.00282069 |
| retention | 0.3 | global-minus-fixed0.9 | visual | decay_damage | 0.00343608 ± 0.00036013 |
| retention | 0.3 | global-minus-fixed0.9 | visual | err_decay | -0.0350075 ± 0.00605779 |
| retention | 0.3 | global-minus-fixed0.9 | visual | err_post | -0.0137818 ± 0.00394531 |
| retention | 0.3 | global-minus-fixed0.9 | visual | write_damage | 0.0212257 ± 0.00456178 |
| retention | 0.3 | protected | audio | decay_damage | 0.0145122 ± 0.000318799 |
| retention | 0.3 | protected | audio | err_decay | 0.0670517 ± 0.00322191 |
| retention | 0.3 | protected | audio | err_post | 0.0670947 ± 0.00322349 |
| retention | 0.3 | protected | audio | write_damage | 4.29972e-05 ± 1.06412e-05 |
| retention | 0.3 | protected | text | decay_damage | 0.0151938 ± 0.000365756 |
| retention | 0.3 | protected | text | err_decay | 0.083989 ± 0.0041972 |
| retention | 0.3 | protected | text | err_post | 0.0840386 ± 0.00419737 |
| retention | 0.3 | protected | text | write_damage | 4.95886e-05 ± 5.90814e-06 |
| retention | 0.3 | protected | visual | decay_damage | 0.0146659 ± 0.00051632 |
| retention | 0.3 | protected | visual | err_decay | 0.0700984 ± 0.00725246 |
| retention | 0.3 | protected | visual | err_post | 0.0701344 ± 0.00724466 |
| retention | 0.3 | protected | visual | write_damage | 3.6025e-05 ± 1.00313e-05 |
| retention | 0.3 | reference | audio | decay_damage | 0.0140966 ± 0.000588243 |
| retention | 0.3 | reference | audio | err_decay | 0.105612 ± 0.00988667 |
| retention | 0.3 | reference | audio | err_post | 0.33033 ± 0.0183626 |
| retention | 0.3 | reference | audio | write_damage | 0.224717 ± 0.0100296 |
| retention | 0.3 | reference | text | decay_damage | 0.014593 ± 0.000661775 |
| retention | 0.3 | reference | text | err_decay | 0.102267 ± 0.01508 |
| retention | 0.3 | reference | text | err_post | 0.311171 ± 0.0196445 |
| retention | 0.3 | reference | text | write_damage | 0.208904 ± 0.00812729 |
| retention | 0.3 | reference | visual | decay_damage | 0.0139534 ± 0.000569626 |
| retention | 0.3 | reference | visual | err_decay | 0.108285 ± 0.00864944 |
| retention | 0.3 | reference | visual | err_post | 0.326343 ± 0.0147961 |
| retention | 0.3 | reference | visual | write_damage | 0.218059 ± 0.016047 |
| retention | 0.5 | fixed0.9 | audio | decay_damage | 0.0074803 ± 0.000360181 |
| retention | 0.5 | fixed0.9 | audio | err_decay | 0.191059 ± 0.0123056 |
| retention | 0.5 | fixed0.9 | audio | err_post | 0.326292 ± 0.0133451 |
| retention | 0.5 | fixed0.9 | audio | write_damage | 0.135233 ± 0.00525247 |
| retention | 0.5 | fixed0.9 | text | decay_damage | 0.00909009 ± 0.000232458 |
| retention | 0.5 | fixed0.9 | text | err_decay | 0.225135 ± 0.00558501 |
| retention | 0.5 | fixed0.9 | text | err_post | 0.35141 ± 0.0124634 |
| retention | 0.5 | fixed0.9 | text | write_damage | 0.126275 ± 0.0108281 |
| retention | 0.5 | fixed0.9 | visual | decay_damage | 0.00829536 ± 0.000396638 |
| retention | 0.5 | fixed0.9 | visual | err_decay | 0.189534 ± 0.0076245 |
| retention | 0.5 | fixed0.9 | visual | err_post | 0.31743 ± 0.0137394 |
| retention | 0.5 | fixed0.9 | visual | write_damage | 0.127897 ± 0.00742679 |
| retention | 0.5 | global | audio | decay_damage | 0.00964868 ± 0.000533132 |
| retention | 0.5 | global | audio | err_decay | 0.163103 ± 0.0140294 |
| retention | 0.5 | global | audio | err_post | 0.308103 ± 0.0132259 |
| retention | 0.5 | global | audio | write_damage | 0.145 ± 0.00486007 |
| retention | 0.5 | global | text | decay_damage | 0.0105079 ± 0.000198371 |
| retention | 0.5 | global | text | err_decay | 0.185774 ± 0.00748828 |
| retention | 0.5 | global | text | err_post | 0.322624 ± 0.0159626 |
| retention | 0.5 | global | text | write_damage | 0.13685 ± 0.00882381 |
| retention | 0.5 | global | visual | decay_damage | 0.00996644 ± 0.000356222 |
| retention | 0.5 | global | visual | err_decay | 0.163713 ± 0.00738629 |
| retention | 0.5 | global | visual | err_post | 0.299515 ± 0.0142909 |
| retention | 0.5 | global | visual | write_damage | 0.135802 ± 0.010067 |
| retention | 0.5 | global-minus-fixed0.9 | audio | decay_damage | 0.00216839 ± 0.000190178 |
| retention | 0.5 | global-minus-fixed0.9 | audio | err_decay | -0.0279561 ± 0.00278527 |
| retention | 0.5 | global-minus-fixed0.9 | audio | err_post | -0.0181891 ± 0.00277215 |
| retention | 0.5 | global-minus-fixed0.9 | audio | write_damage | 0.00976706 ± 0.00438824 |
| retention | 0.5 | global-minus-fixed0.9 | text | decay_damage | 0.00141785 ± 0.000119214 |
| retention | 0.5 | global-minus-fixed0.9 | text | err_decay | -0.0393606 ± 0.00652882 |
| retention | 0.5 | global-minus-fixed0.9 | text | err_post | -0.0287854 ± 0.00487518 |
| retention | 0.5 | global-minus-fixed0.9 | text | write_damage | 0.0105752 ± 0.00269134 |
| retention | 0.5 | global-minus-fixed0.9 | visual | decay_damage | 0.00167108 ± 0.000123459 |
| retention | 0.5 | global-minus-fixed0.9 | visual | err_decay | -0.0258209 ± 0.00342889 |
| retention | 0.5 | global-minus-fixed0.9 | visual | err_post | -0.0179154 ± 0.00253313 |
| retention | 0.5 | global-minus-fixed0.9 | visual | write_damage | 0.00790548 ± 0.00336132 |
| retention | 0.5 | protected | audio | decay_damage | 0.0129517 ± 0.000466179 |
| retention | 0.5 | protected | audio | err_decay | 0.0956001 ± 0.0081046 |
| retention | 0.5 | protected | audio | err_post | 0.0956308 ± 0.00810265 |
| retention | 0.5 | protected | audio | write_damage | 3.06823e-05 ± 3.87352e-06 |
| retention | 0.5 | protected | text | decay_damage | 0.013674 ± 0.000349234 |
| retention | 0.5 | protected | text | err_decay | 0.13582 ± 0.0102941 |
| retention | 0.5 | protected | text | err_post | 0.135857 ± 0.0102932 |
| retention | 0.5 | protected | text | write_damage | 3.74361e-05 ± 3.2425e-06 |
| retention | 0.5 | protected | visual | decay_damage | 0.0134042 ± 0.000294932 |
| retention | 0.5 | protected | visual | err_decay | 0.100418 ± 0.00606515 |
| retention | 0.5 | protected | visual | err_post | 0.100443 ± 0.00606344 |
| retention | 0.5 | protected | visual | write_damage | 2.55664e-05 ± 3.24197e-06 |
| retention | 0.5 | reference | audio | decay_damage | 0.0116286 ± 0.000551803 |
| retention | 0.5 | reference | audio | err_decay | 0.155689 ± 0.014281 |
| retention | 0.5 | reference | audio | err_post | 0.332235 ± 0.0141414 |
| retention | 0.5 | reference | audio | write_damage | 0.176546 ± 0.00544284 |
| retention | 0.5 | reference | text | decay_damage | 0.0120969 ± 0.000110832 |
| retention | 0.5 | reference | text | err_decay | 0.160484 ± 0.0057241 |
| retention | 0.5 | reference | text | err_post | 0.331598 ± 0.0165094 |
| retention | 0.5 | reference | text | write_damage | 0.171115 ± 0.0125802 |
| retention | 0.5 | reference | visual | decay_damage | 0.0115807 ± 0.000264131 |
| retention | 0.5 | reference | visual | err_decay | 0.154966 ± 0.00802124 |
| retention | 0.5 | reference | visual | err_post | 0.322795 ± 0.0153127 |
| retention | 0.5 | reference | visual | write_damage | 0.167829 ± 0.0104652 |
| retention | 0.7 | fixed0.9 | audio | decay_damage | 0.00697516 ± 0.000334809 |
| retention | 0.7 | fixed0.9 | audio | err_decay | 0.239548 ± 0.00904669 |
| retention | 0.7 | fixed0.9 | audio | err_post | 0.352272 ± 0.00972438 |
| retention | 0.7 | fixed0.9 | audio | write_damage | 0.112724 ± 0.00437074 |
| retention | 0.7 | fixed0.9 | text | decay_damage | 0.00872482 ± 0.00025774 |
| retention | 0.7 | fixed0.9 | text | err_decay | 0.257514 ± 0.00773415 |
| retention | 0.7 | fixed0.9 | text | err_post | 0.363757 ± 0.0147226 |
| retention | 0.7 | fixed0.9 | text | write_damage | 0.106244 ± 0.00908498 |
| retention | 0.7 | fixed0.9 | visual | decay_damage | 0.00762871 ± 0.000373823 |
| retention | 0.7 | fixed0.9 | visual | err_decay | 0.236306 ± 0.0156593 |
| retention | 0.7 | fixed0.9 | visual | err_post | 0.34077 ± 0.0192138 |
| retention | 0.7 | fixed0.9 | visual | write_damage | 0.104464 ± 0.00582897 |
| retention | 0.7 | global | audio | decay_damage | 0.00795879 ± 0.000325915 |
| retention | 0.7 | global | audio | err_decay | 0.220199 ± 0.010396 |
| retention | 0.7 | global | audio | err_post | 0.334227 ± 0.00797617 |
| retention | 0.7 | global | audio | write_damage | 0.114029 ± 0.00539509 |
| retention | 0.7 | global | text | decay_damage | 0.00925564 ± 0.000344003 |
| retention | 0.7 | global | text | err_decay | 0.233376 ± 0.014632 |
| retention | 0.7 | global | text | err_post | 0.34114 ± 0.0184065 |
| retention | 0.7 | global | text | write_damage | 0.107763 ± 0.00677187 |
| retention | 0.7 | global | visual | decay_damage | 0.00836071 ± 0.000442884 |
| retention | 0.7 | global | visual | err_decay | 0.218905 ± 0.016602 |
| retention | 0.7 | global | visual | err_post | 0.323698 ± 0.0199949 |
| retention | 0.7 | global | visual | write_damage | 0.104792 ± 0.00619941 |
| retention | 0.7 | global-minus-fixed0.9 | audio | decay_damage | 0.000983626 ± 7.80438e-05 |
| retention | 0.7 | global-minus-fixed0.9 | audio | err_decay | -0.0193493 ± 0.00209093 |
| retention | 0.7 | global-minus-fixed0.9 | audio | err_post | -0.0180443 ± 0.00320056 |
| retention | 0.7 | global-minus-fixed0.9 | audio | write_damage | 0.00130499 ± 0.00311627 |
| retention | 0.7 | global-minus-fixed0.9 | text | decay_damage | 0.000530818 ± 0.000118294 |
| retention | 0.7 | global-minus-fixed0.9 | text | err_decay | -0.0241373 ± 0.00760938 |
| retention | 0.7 | global-minus-fixed0.9 | text | err_post | -0.0226174 ± 0.00405206 |
| retention | 0.7 | global-minus-fixed0.9 | text | write_damage | 0.0015199 ± 0.00386707 |
| retention | 0.7 | global-minus-fixed0.9 | visual | decay_damage | 0.000731999 ± 0.00015493 |
| retention | 0.7 | global-minus-fixed0.9 | visual | err_decay | -0.0174011 ± 0.00163469 |
| retention | 0.7 | global-minus-fixed0.9 | visual | err_post | -0.0170723 ± 0.00250558 |
| retention | 0.7 | global-minus-fixed0.9 | visual | write_damage | 0.000328817 ± 0.00161813 |
| retention | 0.7 | protected | audio | decay_damage | 0.0126317 ± 0.000225351 |
| retention | 0.7 | protected | audio | err_decay | 0.128206 ± 0.00551471 |
| retention | 0.7 | protected | audio | err_post | 0.128233 ± 0.005512 |
| retention | 0.7 | protected | audio | write_damage | 2.70347e-05 ± 3.51973e-06 |
| retention | 0.7 | protected | text | decay_damage | 0.0128183 ± 0.00030573 |
| retention | 0.7 | protected | text | err_decay | 0.175974 ± 0.0165005 |
| retention | 0.7 | protected | text | err_post | 0.176001 ± 0.0165014 |
| retention | 0.7 | protected | text | write_damage | 2.77107e-05 ± 1.83581e-06 |
| retention | 0.7 | protected | visual | decay_damage | 0.013271 ± 0.000141517 |
| retention | 0.7 | protected | visual | err_decay | 0.131386 ± 0.0101075 |
| retention | 0.7 | protected | visual | err_post | 0.131408 ± 0.0101073 |
| retention | 0.7 | protected | visual | write_damage | 2.23701e-05 ± 3.66611e-06 |
| retention | 0.7 | reference | audio | decay_damage | 0.00946597 ± 0.00033757 |
| retention | 0.7 | reference | audio | err_decay | 0.211337 ± 0.012172 |
| retention | 0.7 | reference | audio | err_post | 0.356814 ± 0.0114599 |
| retention | 0.7 | reference | audio | write_damage | 0.145477 ± 0.00554487 |
| retention | 0.7 | reference | text | decay_damage | 0.0105834 ± 0.000233298 |
| retention | 0.7 | reference | text | err_decay | 0.200956 ± 0.00888991 |
| retention | 0.7 | reference | text | err_post | 0.342953 ± 0.0173608 |
| retention | 0.7 | reference | text | write_damage | 0.141997 ± 0.0106244 |
| retention | 0.7 | reference | visual | decay_damage | 0.00944093 ± 0.00050286 |
| retention | 0.7 | reference | visual | err_decay | 0.209545 ± 0.0164014 |
| retention | 0.7 | reference | visual | err_post | 0.344378 ± 0.0207067 |
| retention | 0.7 | reference | visual | write_damage | 0.134834 ± 0.00702467 |
| write_fit | 0 | fixed0.9 | audio | err_after | 0.0707107 ± 0.00426932 |
| write_fit | 0 | fixed0.9 | audio | err_before | 0.69354 ± 0.0418963 |
| write_fit | 0 | fixed0.9 | audio | err_original_after | 0.0016603 ± 7.49435e-05 |
| write_fit | 0 | fixed0.9 | audio | fit_gain | 0.622829 ± 0.0376271 |
| write_fit | 0 | fixed0.9 | text | err_after | 0.106549 ± 0.00242235 |
| write_fit | 0 | fixed0.9 | text | err_before | 1.04402 ± 0.0238517 |
| write_fit | 0 | fixed0.9 | text | err_original_after | 0.00247515 ± 5.68799e-05 |
| write_fit | 0 | fixed0.9 | text | fit_gain | 0.937475 ± 0.0214295 |
| write_fit | 0 | fixed0.9 | visual | err_after | 0.0669382 ± 0.00358426 |
| write_fit | 0 | fixed0.9 | visual | err_before | 0.656386 ± 0.0352114 |
| write_fit | 0 | fixed0.9 | visual | err_original_after | 0.0016274 ± 8.4911e-05 |
| write_fit | 0 | fixed0.9 | visual | fit_gain | 0.589448 ± 0.0316272 |
| write_fit | 0 | global | audio | err_after | 0.00171622 ± 7.70519e-05 |
| write_fit | 0 | global | audio | err_before | 0.720397 ± 0.0435906 |
| write_fit | 0 | global | audio | err_original_after | 0.00171622 ± 7.70519e-05 |
| write_fit | 0 | global | audio | fit_gain | 0.71868 ± 0.0435322 |
| write_fit | 0 | global | text | err_after | 0.00256227 ± 6.01979e-05 |
| write_fit | 0 | global | text | err_before | 1.08454 ± 0.0249842 |
| write_fit | 0 | global | text | err_original_after | 0.00256227 ± 6.01979e-05 |
| write_fit | 0 | global | text | fit_gain | 1.08198 ± 0.0249508 |
| write_fit | 0 | global | visual | err_after | 0.00167965 ± 8.80841e-05 |
| write_fit | 0 | global | visual | err_before | 0.680703 ± 0.0373319 |
| write_fit | 0 | global | visual | err_original_after | 0.00167965 ± 8.80841e-05 |
| write_fit | 0 | global | visual | fit_gain | 0.679023 ± 0.0372547 |
| write_fit | 0 | global-minus-fixed0.9 | audio | err_after | -0.0689945 ± 0.00421057 |
| write_fit | 0 | global-minus-fixed0.9 | audio | err_before | 0.0268567 ± 0.00171361 |
| write_fit | 0 | global-minus-fixed0.9 | audio | err_original_after | 5.59165e-05 ± 3.11616e-06 |
| write_fit | 0 | global-minus-fixed0.9 | audio | fit_gain | 0.0958512 ± 0.00591094 |
| write_fit | 0 | global-minus-fixed0.9 | text | err_after | -0.103987 ± 0.00239131 |
| write_fit | 0 | global-minus-fixed0.9 | text | err_before | 0.0405137 ± 0.00172903 |
| write_fit | 0 | global-minus-fixed0.9 | text | err_original_after | 8.71165e-05 ± 3.89507e-06 |
| write_fit | 0 | global-minus-fixed0.9 | text | fit_gain | 0.1445 ± 0.00372731 |
| write_fit | 0 | global-minus-fixed0.9 | visual | err_after | -0.0652585 ± 0.0035061 |
| write_fit | 0 | global-minus-fixed0.9 | visual | err_before | 0.0243167 ± 0.00236353 |
| write_fit | 0 | global-minus-fixed0.9 | visual | err_original_after | 5.22527e-05 ± 3.91057e-06 |
| write_fit | 0 | global-minus-fixed0.9 | visual | fit_gain | 0.0895752 ± 0.00571773 |
| write_fit | 0 | protected | audio | err_after | 0.00171622 ± 7.70519e-05 |
| write_fit | 0 | protected | audio | err_before | 0.720397 ± 0.0435906 |
| write_fit | 0 | protected | audio | err_original_after | 0.00171622 ± 7.70519e-05 |
| write_fit | 0 | protected | audio | fit_gain | 0.71868 ± 0.0435322 |
| write_fit | 0 | protected | text | err_after | 0.00256227 ± 6.01979e-05 |
| write_fit | 0 | protected | text | err_before | 1.08454 ± 0.0249842 |
| write_fit | 0 | protected | text | err_original_after | 0.00256227 ± 6.01979e-05 |
| write_fit | 0 | protected | text | fit_gain | 1.08198 ± 0.0249508 |
| write_fit | 0 | protected | visual | err_after | 0.00167965 ± 8.80841e-05 |
| write_fit | 0 | protected | visual | err_before | 0.680703 ± 0.0373319 |
| write_fit | 0 | protected | visual | err_original_after | 0.00167965 ± 8.80841e-05 |
| write_fit | 0 | protected | visual | fit_gain | 0.679023 ± 0.0372547 |
| write_fit | 0 | reference | audio | err_after | 0.00171622 ± 7.70519e-05 |
| write_fit | 0 | reference | audio | err_before | 0.720397 ± 0.0435906 |
| write_fit | 0 | reference | audio | err_original_after | 0.00171622 ± 7.70519e-05 |
| write_fit | 0 | reference | audio | fit_gain | 0.71868 ± 0.0435322 |
| write_fit | 0 | reference | text | err_after | 0.00256227 ± 6.01979e-05 |
| write_fit | 0 | reference | text | err_before | 1.08454 ± 0.0249842 |
| write_fit | 0 | reference | text | err_original_after | 0.00256227 ± 6.01979e-05 |
| write_fit | 0 | reference | text | fit_gain | 1.08198 ± 0.0249508 |
| write_fit | 0 | reference | visual | err_after | 0.00167965 ± 8.80841e-05 |
| write_fit | 0 | reference | visual | err_before | 0.680703 ± 0.0373319 |
| write_fit | 0 | reference | visual | err_original_after | 0.00167965 ± 8.80841e-05 |
| write_fit | 0 | reference | visual | fit_gain | 0.679023 ± 0.0372547 |
| write_fit | 0.1 | fixed0.9 | audio | err_after | 0.0715459 ± 0.00416127 |
| write_fit | 0.1 | fixed0.9 | audio | err_before | 0.701812 ± 0.0408169 |
| write_fit | 0.1 | fixed0.9 | audio | err_original_after | 0.00165221 ± 7.70657e-05 |
| write_fit | 0.1 | fixed0.9 | audio | fit_gain | 0.630266 ± 0.0366556 |
| write_fit | 0.1 | fixed0.9 | text | err_after | 0.106645 ± 0.00230066 |
| write_fit | 0.1 | fixed0.9 | text | err_before | 1.0452 ± 0.0226361 |
| write_fit | 0.1 | fixed0.9 | text | err_original_after | 0.00244463 ± 5.27237e-05 |
| write_fit | 0.1 | fixed0.9 | text | fit_gain | 0.938556 ± 0.0203355 |
| write_fit | 0.1 | fixed0.9 | visual | err_after | 0.0680282 ± 0.00315184 |
| write_fit | 0.1 | fixed0.9 | visual | err_before | 0.667173 ± 0.0309679 |
| write_fit | 0.1 | fixed0.9 | visual | err_original_after | 0.00161933 ± 7.3352e-05 |
| write_fit | 0.1 | fixed0.9 | visual | fit_gain | 0.599145 ± 0.0278161 |
| write_fit | 0.1 | global | audio | err_after | 0.00882974 ± 0.000697237 |
| write_fit | 0.1 | global | audio | err_before | 0.725938 ± 0.0425603 |
| write_fit | 0.1 | global | audio | err_original_after | 0.00170247 ± 7.9395e-05 |
| write_fit | 0.1 | global | audio | fit_gain | 0.717109 ± 0.0428335 |
| write_fit | 0.1 | global | text | err_after | 0.0122816 ± 0.00124017 |
| write_fit | 0.1 | global | text | err_before | 1.08169 ± 0.0236757 |
| write_fit | 0.1 | global | text | err_original_after | 0.00252307 ± 5.57353e-05 |
| write_fit | 0.1 | global | text | fit_gain | 1.0694 ± 0.0242596 |
| write_fit | 0.1 | global | visual | err_after | 0.0087048 ± 0.00130855 |
| write_fit | 0.1 | global | visual | err_before | 0.688734 ± 0.0329496 |
| write_fit | 0.1 | global | visual | err_original_after | 0.0016656 ± 7.64155e-05 |
| write_fit | 0.1 | global | visual | fit_gain | 0.68003 ± 0.0324315 |
| write_fit | 0.1 | global-minus-fixed0.9 | audio | err_after | -0.0627162 ± 0.00447093 |
| write_fit | 0.1 | global-minus-fixed0.9 | audio | err_before | 0.0241267 ± 0.00176132 |
| write_fit | 0.1 | global-minus-fixed0.9 | audio | err_original_after | 5.02578e-05 ± 3.17319e-06 |
| write_fit | 0.1 | global-minus-fixed0.9 | audio | fit_gain | 0.0868429 ± 0.00622757 |
| write_fit | 0.1 | global-minus-fixed0.9 | text | err_after | -0.0943634 ± 0.00308068 |
| write_fit | 0.1 | global-minus-fixed0.9 | text | err_before | 0.0364853 ± 0.00150102 |
| write_fit | 0.1 | global-minus-fixed0.9 | text | err_original_after | 7.84407e-05 ± 3.38852e-06 |
| write_fit | 0.1 | global-minus-fixed0.9 | text | fit_gain | 0.130849 ± 0.00405357 |
| write_fit | 0.1 | global-minus-fixed0.9 | visual | err_after | -0.0593234 ± 0.00283783 |
| write_fit | 0.1 | global-minus-fixed0.9 | visual | err_before | 0.0215616 ± 0.00231094 |
| write_fit | 0.1 | global-minus-fixed0.9 | visual | err_original_after | 4.62705e-05 ± 4.18051e-06 |
| write_fit | 0.1 | global-minus-fixed0.9 | visual | fit_gain | 0.080885 ± 0.00503656 |
| write_fit | 0.1 | protected | audio | err_after | 0.0179781 ± 0.00250462 |
| write_fit | 0.1 | protected | audio | err_before | 0.722751 ± 0.0432485 |
| write_fit | 0.1 | protected | audio | err_original_after | 0.00169962 ± 7.95455e-05 |
| write_fit | 0.1 | protected | audio | fit_gain | 0.704773 ± 0.0438588 |
| write_fit | 0.1 | protected | text | err_after | 0.0237528 ± 0.00209613 |
| write_fit | 0.1 | protected | text | err_before | 1.08152 ± 0.0242898 |
| write_fit | 0.1 | protected | text | err_original_after | 0.00252674 ± 5.5469e-05 |
| write_fit | 0.1 | protected | text | fit_gain | 1.05777 ± 0.025247 |
| write_fit | 0.1 | protected | visual | err_after | 0.0181829 ± 0.00257419 |
| write_fit | 0.1 | protected | visual | err_before | 0.68364 ± 0.0339802 |
| write_fit | 0.1 | protected | visual | err_original_after | 0.00166026 ± 7.77275e-05 |
| write_fit | 0.1 | protected | visual | fit_gain | 0.665457 ± 0.0323829 |
| write_fit | 0.1 | reference | audio | err_after | 0.00170806 ± 7.90948e-05 |
| write_fit | 0.1 | reference | audio | err_before | 0.729221 ± 0.0421844 |
| write_fit | 0.1 | reference | audio | err_original_after | 0.00170806 ± 7.90948e-05 |
| write_fit | 0.1 | reference | audio | fit_gain | 0.727513 ± 0.0421192 |
| write_fit | 0.1 | reference | text | err_after | 0.00252873 ± 5.57751e-05 |
| write_fit | 0.1 | reference | text | err_before | 1.08504 ± 0.0235174 |
| write_fit | 0.1 | reference | text | err_original_after | 0.00252873 ± 5.57751e-05 |
| write_fit | 0.1 | reference | text | fit_gain | 1.08251 ± 0.0234832 |
| write_fit | 0.1 | reference | visual | err_after | 0.00167039 ± 7.61694e-05 |
| write_fit | 0.1 | reference | visual | err_before | 0.691739 ± 0.0327185 |
| write_fit | 0.1 | reference | visual | err_original_after | 0.00167039 ± 7.61694e-05 |
| write_fit | 0.1 | reference | visual | fit_gain | 0.690068 ± 0.0326505 |
| write_fit | 0.3 | fixed0.9 | audio | err_after | 0.0740802 ± 0.00359123 |
| write_fit | 0.3 | fixed0.9 | audio | err_before | 0.726857 ± 0.0352155 |
| write_fit | 0.3 | fixed0.9 | audio | err_original_after | 0.00165335 ± 7.08184e-05 |
| write_fit | 0.3 | fixed0.9 | audio | fit_gain | 0.652777 ± 0.0316243 |
| write_fit | 0.3 | fixed0.9 | text | err_after | 0.106882 ± 0.00201127 |
| write_fit | 0.3 | fixed0.9 | text | err_before | 1.04801 ± 0.0197538 |
| write_fit | 0.3 | fixed0.9 | text | err_original_after | 0.00237984 ± 4.09675e-05 |
| write_fit | 0.3 | fixed0.9 | text | fit_gain | 0.941124 ± 0.0177427 |
| write_fit | 0.3 | fixed0.9 | visual | err_after | 0.0704831 ± 0.003573 |
| write_fit | 0.3 | fixed0.9 | visual | err_before | 0.691431 ± 0.0350545 |
| write_fit | 0.3 | fixed0.9 | visual | err_original_after | 0.00161308 ± 8.53989e-05 |
| write_fit | 0.3 | fixed0.9 | visual | fit_gain | 0.620948 ± 0.0314816 |
| write_fit | 0.3 | global | audio | err_after | 0.0224464 ± 0.00114819 |
| write_fit | 0.3 | global | audio | err_before | 0.745902 ± 0.0376519 |
| write_fit | 0.3 | global | audio | err_original_after | 0.00169283 ± 7.48099e-05 |
| write_fit | 0.3 | global | audio | fit_gain | 0.723455 ± 0.0380224 |
| write_fit | 0.3 | global | text | err_after | 0.0328044 ± 0.00387059 |
| write_fit | 0.3 | global | text | err_before | 1.07681 ± 0.021263 |
| write_fit | 0.3 | global | text | err_original_after | 0.00244113 ± 4.35526e-05 |
| write_fit | 0.3 | global | text | fit_gain | 1.044 ± 0.0243644 |
| write_fit | 0.3 | global | visual | err_after | 0.0226048 ± 0.0024435 |
| write_fit | 0.3 | global | visual | err_before | 0.708491 ± 0.0369166 |
| write_fit | 0.3 | global | visual | err_original_after | 0.00164924 ± 8.83065e-05 |
| write_fit | 0.3 | global | visual | fit_gain | 0.685886 ± 0.0383258 |
| write_fit | 0.3 | global-minus-fixed0.9 | audio | err_after | -0.0516338 ± 0.00408827 |
| write_fit | 0.3 | global-minus-fixed0.9 | audio | err_before | 0.0190442 ± 0.00263931 |
| write_fit | 0.3 | global-minus-fixed0.9 | audio | err_original_after | 3.94712e-05 ± 4.99258e-06 |
| write_fit | 0.3 | global-minus-fixed0.9 | audio | fit_gain | 0.0706781 ± 0.00659094 |
| write_fit | 0.3 | global-minus-fixed0.9 | text | err_after | -0.0740778 ± 0.00554888 |
| write_fit | 0.3 | global-minus-fixed0.9 | text | err_before | 0.0288027 ± 0.0016745 |
| write_fit | 0.3 | global-minus-fixed0.9 | text | err_original_after | 6.12919e-05 ± 3.42113e-06 |
| write_fit | 0.3 | global-minus-fixed0.9 | text | fit_gain | 0.102881 ± 0.0070761 |
| write_fit | 0.3 | global-minus-fixed0.9 | visual | err_after | -0.0478783 ± 0.00529079 |
| write_fit | 0.3 | global-minus-fixed0.9 | visual | err_before | 0.0170599 ± 0.00264199 |
| write_fit | 0.3 | global-minus-fixed0.9 | visual | err_original_after | 3.61543e-05 ± 4.67351e-06 |
| write_fit | 0.3 | global-minus-fixed0.9 | visual | fit_gain | 0.0649382 ± 0.00755727 |
| write_fit | 0.3 | protected | audio | err_after | 0.0458295 ± 0.00343827 |
| write_fit | 0.3 | protected | audio | err_before | 0.737044 ± 0.0395028 |
| write_fit | 0.3 | protected | audio | err_original_after | 0.00168303 ± 7.66824e-05 |
| write_fit | 0.3 | protected | audio | fit_gain | 0.691214 ± 0.0403667 |
| write_fit | 0.3 | protected | text | err_after | 0.0657008 ± 0.0075083 |
| write_fit | 0.3 | protected | text | err_before | 1.07606 ± 0.0229661 |
| write_fit | 0.3 | protected | text | err_original_after | 0.00244777 ± 4.49422e-05 |
| write_fit | 0.3 | protected | text | fit_gain | 1.01036 ± 0.0286398 |
| write_fit | 0.3 | protected | visual | err_after | 0.0470605 ± 0.00364192 |
| write_fit | 0.3 | protected | visual | err_before | 0.69622 ± 0.0379571 |
| write_fit | 0.3 | protected | visual | err_original_after | 0.00163285 ± 8.96253e-05 |
| write_fit | 0.3 | protected | visual | fit_gain | 0.64916 ± 0.0392997 |
| write_fit | 0.3 | reference | audio | err_after | 0.00170951 ± 7.29734e-05 |
| write_fit | 0.3 | reference | audio | err_before | 0.755252 ± 0.0363039 |
| write_fit | 0.3 | reference | audio | err_original_after | 0.00170951 ± 7.29734e-05 |
| write_fit | 0.3 | reference | audio | fit_gain | 0.753542 ± 0.0362396 |
| write_fit | 0.3 | reference | text | err_after | 0.00245759 ± 4.27781e-05 |
| write_fit | 0.3 | reference | text | err_before | 1.08614 ± 0.0206608 |
| write_fit | 0.3 | reference | text | err_original_after | 0.00245759 ± 4.27781e-05 |
| write_fit | 0.3 | reference | text | fit_gain | 1.08368 ± 0.0206286 |
| write_fit | 0.3 | reference | visual | err_after | 0.00166222 ± 8.91829e-05 |
| write_fit | 0.3 | reference | visual | err_before | 0.716156 ± 0.0370164 |
| write_fit | 0.3 | reference | visual | err_original_after | 0.00166222 ± 8.91829e-05 |
| write_fit | 0.3 | reference | visual | fit_gain | 0.714494 ± 0.0369305 |
| write_fit | 0.5 | fixed0.9 | audio | err_after | 0.0756793 ± 0.00358759 |
| write_fit | 0.5 | fixed0.9 | audio | err_before | 0.742759 ± 0.0352106 |
| write_fit | 0.5 | fixed0.9 | audio | err_original_after | 0.00162951 ± 6.58266e-05 |
| write_fit | 0.5 | fixed0.9 | audio | fit_gain | 0.66708 ± 0.031623 |
| write_fit | 0.5 | fixed0.9 | text | err_after | 0.107501 ± 0.00250565 |
| write_fit | 0.5 | fixed0.9 | text | err_before | 1.05458 ± 0.0245214 |
| write_fit | 0.5 | fixed0.9 | text | err_original_after | 0.00231733 ± 6.12225e-05 |
| write_fit | 0.5 | fixed0.9 | text | fit_gain | 0.947082 ± 0.0220158 |
| write_fit | 0.5 | fixed0.9 | visual | err_after | 0.073249 ± 0.00387714 |
| write_fit | 0.5 | fixed0.9 | visual | err_before | 0.718824 ± 0.0380027 |
| write_fit | 0.5 | fixed0.9 | visual | err_original_after | 0.00159971 ± 9.5845e-05 |
| write_fit | 0.5 | fixed0.9 | visual | fit_gain | 0.645575 ± 0.0341256 |
| write_fit | 0.5 | global | audio | err_after | 0.0381528 ± 0.0022173 |
| write_fit | 0.5 | global | audio | err_before | 0.755628 ± 0.0375843 |
| write_fit | 0.5 | global | audio | err_original_after | 0.00165647 ± 7.00127e-05 |
| write_fit | 0.5 | global | audio | fit_gain | 0.717475 ± 0.0390374 |
| write_fit | 0.5 | global | text | err_after | 0.0568722 ± 0.00539537 |
| write_fit | 0.5 | global | text | err_before | 1.075 ± 0.0262894 |
| write_fit | 0.5 | global | text | err_original_after | 0.00236074 ± 6.47324e-05 |
| write_fit | 0.5 | global | text | fit_gain | 1.01812 ± 0.029474 |
| write_fit | 0.5 | global | visual | err_after | 0.0390271 ± 0.00114915 |
| write_fit | 0.5 | global | visual | err_before | 0.7301 ± 0.0404096 |
| write_fit | 0.5 | global | visual | err_original_after | 0.00162401 ± 0.000100326 |
| write_fit | 0.5 | global | visual | fit_gain | 0.691073 ± 0.0411846 |
| write_fit | 0.5 | global-minus-fixed0.9 | audio | err_after | -0.0375265 ± 0.00525166 |
| write_fit | 0.5 | global-minus-fixed0.9 | audio | err_before | 0.0128687 ± 0.00267417 |
| write_fit | 0.5 | global-minus-fixed0.9 | audio | err_original_after | 2.69513e-05 ± 4.84771e-06 |
| write_fit | 0.5 | global-minus-fixed0.9 | audio | fit_gain | 0.0503952 ± 0.00787611 |
| write_fit | 0.5 | global-minus-fixed0.9 | text | err_after | -0.050629 ± 0.00692629 |
| write_fit | 0.5 | global-minus-fixed0.9 | text | err_before | 0.0204123 ± 0.00250391 |
| write_fit | 0.5 | global-minus-fixed0.9 | text | err_original_after | 4.34078e-05 ± 5.13201e-06 |
| write_fit | 0.5 | global-minus-fixed0.9 | text | fit_gain | 0.0710413 ± 0.00936874 |
| write_fit | 0.5 | global-minus-fixed0.9 | visual | err_after | -0.0342219 ± 0.00470703 |
| write_fit | 0.5 | global-minus-fixed0.9 | visual | err_before | 0.0112755 ± 0.00256891 |
| write_fit | 0.5 | global-minus-fixed0.9 | visual | err_original_after | 2.42976e-05 ± 4.87806e-06 |
| write_fit | 0.5 | global-minus-fixed0.9 | visual | fit_gain | 0.0454974 ± 0.0072341 |
| write_fit | 0.5 | protected | audio | err_after | 0.0727484 ± 0.00346009 |
| write_fit | 0.5 | protected | audio | err_before | 0.740839 ± 0.0412229 |
| write_fit | 0.5 | protected | audio | err_original_after | 0.00163284 ± 7.4267e-05 |
| write_fit | 0.5 | protected | audio | fit_gain | 0.668091 ± 0.0433894 |
| write_fit | 0.5 | protected | text | err_after | 0.109173 ± 0.00932161 |
| write_fit | 0.5 | protected | text | err_before | 1.07226 ± 0.0290257 |
| write_fit | 0.5 | protected | text | err_original_after | 0.00236195 ± 6.68529e-05 |
| write_fit | 0.5 | protected | text | fit_gain | 0.963083 ± 0.0350653 |
| write_fit | 0.5 | protected | visual | err_after | 0.0764507 ± 0.00171849 |
| write_fit | 0.5 | protected | visual | err_before | 0.713066 ± 0.0399058 |
| write_fit | 0.5 | protected | visual | err_original_after | 0.00159728 ± 0.000100546 |
| write_fit | 0.5 | protected | visual | fit_gain | 0.636615 ± 0.0391659 |
| write_fit | 0.5 | reference | audio | err_after | 0.00168408 ± 6.90055e-05 |
| write_fit | 0.5 | reference | audio | err_before | 0.770635 ± 0.0368228 |
| write_fit | 0.5 | reference | audio | err_original_after | 0.00168408 ± 6.90055e-05 |
| write_fit | 0.5 | reference | audio | fit_gain | 0.76895 ± 0.0367568 |
| write_fit | 0.5 | reference | text | err_after | 0.00239179 ± 6.46826e-05 |
| write_fit | 0.5 | reference | text | err_before | 1.0915 ± 0.0255167 |
| write_fit | 0.5 | reference | text | err_original_after | 0.00239179 ± 6.46826e-05 |
| write_fit | 0.5 | reference | text | fit_gain | 1.08911 ± 0.025458 |
| write_fit | 0.5 | reference | visual | err_after | 0.00164696 ± 0.000101443 |
| write_fit | 0.5 | reference | visual | err_before | 0.742863 ± 0.0408904 |
| write_fit | 0.5 | reference | visual | err_original_after | 0.00164696 ± 0.000101443 |
| write_fit | 0.5 | reference | visual | fit_gain | 0.741216 ± 0.0407896 |
| write_fit | 0.7 | fixed0.9 | audio | err_after | 0.0778311 ± 0.00323483 |
| write_fit | 0.7 | fixed0.9 | audio | err_before | 0.764094 ± 0.0317375 |
| write_fit | 0.7 | fixed0.9 | audio | err_original_after | 0.00161458 ± 6.27042e-05 |
| write_fit | 0.7 | fixed0.9 | audio | fit_gain | 0.686263 ± 0.0285027 |
| write_fit | 0.7 | fixed0.9 | text | err_after | 0.107703 ± 0.00162679 |
| write_fit | 0.7 | fixed0.9 | text | err_before | 1.05706 ± 0.0158897 |
| write_fit | 0.7 | fixed0.9 | text | err_original_after | 0.00224307 ± 4.09209e-05 |
| write_fit | 0.7 | fixed0.9 | text | fit_gain | 0.949362 ± 0.014263 |
| write_fit | 0.7 | fixed0.9 | visual | err_after | 0.074216 ± 0.00274151 |
| write_fit | 0.7 | fixed0.9 | visual | err_before | 0.728539 ± 0.0269153 |
| write_fit | 0.7 | fixed0.9 | visual | err_original_after | 0.00155448 ± 6.1358e-05 |
| write_fit | 0.7 | fixed0.9 | visual | fit_gain | 0.654323 ± 0.0241738 |
| write_fit | 0.7 | global | audio | err_after | 0.0548638 ± 0.00305249 |
| write_fit | 0.7 | global | audio | err_before | 0.771194 ± 0.0340172 |
| write_fit | 0.7 | global | audio | err_original_after | 0.0016296 ± 6.66551e-05 |
| write_fit | 0.7 | global | audio | fit_gain | 0.71633 ± 0.0351612 |
| write_fit | 0.7 | global | text | err_after | 0.0824076 ± 0.010433 |
| write_fit | 0.7 | global | text | err_before | 1.06959 ± 0.017758 |
| write_fit | 0.7 | global | text | err_original_after | 0.00226945 ± 4.43649e-05 |
| write_fit | 0.7 | global | text | fit_gain | 0.987185 ± 0.0253019 |
| write_fit | 0.7 | global | visual | err_after | 0.0544497 ± 0.0024761 |
| write_fit | 0.7 | global | visual | err_before | 0.733759 ± 0.0271755 |
| write_fit | 0.7 | global | visual | err_original_after | 0.00156599 ± 6.1734e-05 |
| write_fit | 0.7 | global | visual | fit_gain | 0.679309 ± 0.0251687 |
| write_fit | 0.7 | global-minus-fixed0.9 | audio | err_after | -0.0229672 ± 0.00513098 |
| write_fit | 0.7 | global-minus-fixed0.9 | audio | err_before | 0.00709986 ± 0.00234544 |
| write_fit | 0.7 | global-minus-fixed0.9 | audio | err_original_after | 1.50214e-05 ± 4.29523e-06 |
| write_fit | 0.7 | global-minus-fixed0.9 | audio | fit_gain | 0.0300671 ± 0.00724077 |
| write_fit | 0.7 | global-minus-fixed0.9 | text | err_after | -0.0252953 ± 0.0113398 |
| write_fit | 0.7 | global-minus-fixed0.9 | text | err_before | 0.0125285 ± 0.00263517 |
| write_fit | 0.7 | global-minus-fixed0.9 | text | err_original_after | 2.63827e-05 ± 5.12744e-06 |
| write_fit | 0.7 | global-minus-fixed0.9 | text | fit_gain | 0.0378238 ± 0.0137667 |
| write_fit | 0.7 | global-minus-fixed0.9 | visual | err_after | -0.0197663 ± 0.00145323 |
| write_fit | 0.7 | global-minus-fixed0.9 | visual | err_before | 0.00521979 ± 0.00143726 |
| write_fit | 0.7 | global-minus-fixed0.9 | visual | err_original_after | 1.15101e-05 ± 2.30619e-06 |
| write_fit | 0.7 | global-minus-fixed0.9 | visual | fit_gain | 0.0249861 ± 0.00275023 |
| write_fit | 0.7 | protected | audio | err_after | 0.0977352 ± 0.00361339 |
| write_fit | 0.7 | protected | audio | err_before | 0.749505 ± 0.0376996 |
| write_fit | 0.7 | protected | audio | err_original_after | 0.00158945 ± 7.21516e-05 |
| write_fit | 0.7 | protected | audio | fit_gain | 0.651769 ± 0.0374464 |
| write_fit | 0.7 | protected | text | err_after | 0.151282 ± 0.0179867 |
| write_fit | 0.7 | protected | text | err_before | 1.06636 ± 0.0218721 |
| write_fit | 0.7 | protected | text | err_original_after | 0.00226683 ± 5.02225e-05 |
| write_fit | 0.7 | protected | text | fit_gain | 0.915076 ± 0.0350258 |
| write_fit | 0.7 | protected | visual | err_after | 0.0979306 ± 0.00452782 |
| write_fit | 0.7 | protected | visual | err_before | 0.708883 ± 0.0269329 |
| write_fit | 0.7 | protected | visual | err_original_after | 0.00152063 ± 6.13366e-05 |
| write_fit | 0.7 | protected | visual | fit_gain | 0.610953 ± 0.0227467 |
| write_fit | 0.7 | reference | audio | err_after | 0.0016693 ± 6.27138e-05 |
| write_fit | 0.7 | reference | audio | err_before | 0.791935 ± 0.0316755 |
| write_fit | 0.7 | reference | audio | err_original_after | 0.0016693 ± 6.27138e-05 |
| write_fit | 0.7 | reference | audio | fit_gain | 0.790265 ± 0.0316144 |
| write_fit | 0.7 | reference | text | err_after | 0.00231336 ± 4.28198e-05 |
| write_fit | 0.7 | reference | text | err_before | 1.09208 ± 0.016321 |
| write_fit | 0.7 | reference | text | err_original_after | 0.00231336 ± 4.28198e-05 |
| write_fit | 0.7 | reference | text | fit_gain | 1.08976 ± 0.0162802 |
| write_fit | 0.7 | reference | visual | err_after | 0.00159837 ± 6.43679e-05 |
| write_fit | 0.7 | reference | visual | err_before | 0.750972 ± 0.0284134 |
| write_fit | 0.7 | reference | visual | err_original_after | 0.00159837 ± 6.43679e-05 |
| write_fit | 0.7 | reference | visual | fit_gain | 0.749374 ± 0.0283498 |
| write_fit | all | fixed0.9 | audio | err_after | 0.0739694 ± 0.00372042 |
| write_fit | all | fixed0.9 | audio | err_before | 0.725813 ± 0.0364979 |
| write_fit | all | fixed0.9 | audio | err_original_after | 0.00164199 ± 6.82501e-05 |
| write_fit | all | fixed0.9 | audio | fit_gain | 0.651843 ± 0.0327776 |
| write_fit | all | fixed0.9 | text | err_after | 0.107056 ± 0.00213586 |
| write_fit | all | fixed0.9 | text | err_before | 1.04978 ± 0.0209624 |
| write_fit | all | fixed0.9 | text | err_original_after | 0.002372 ± 4.73593e-05 |
| write_fit | all | fixed0.9 | text | fit_gain | 0.94272 ± 0.0188267 |
| write_fit | all | fixed0.9 | visual | err_after | 0.0705829 ± 0.00331343 |
| write_fit | all | fixed0.9 | visual | err_before | 0.692471 ± 0.0325179 |
| write_fit | all | fixed0.9 | visual | err_original_after | 0.0016028 ± 7.83469e-05 |
| write_fit | all | fixed0.9 | visual | fit_gain | 0.621888 ± 0.0292045 |
| write_fit | all | global | audio | err_after | 0.0252018 ± 0.000865324 |
| write_fit | all | global | audio | err_before | 0.743812 ± 0.0386491 |
| write_fit | all | global | audio | err_original_after | 0.00167952 ± 7.1513e-05 |
| write_fit | all | global | audio | fit_gain | 0.71861 ± 0.0394321 |
| write_fit | all | global | text | err_after | 0.0373856 ± 0.00394774 |
| write_fit | all | global | text | err_before | 1.07752 ± 0.0222782 |
| write_fit | all | global | text | err_original_after | 0.00243133 ± 4.96782e-05 |
| write_fit | all | global | text | fit_gain | 1.04014 ± 0.0244738 |
| write_fit | all | global | visual | err_after | 0.0252932 ± 0.00114999 |
| write_fit | all | global | visual | err_before | 0.708357 ± 0.0342033 |
| write_fit | all | global | visual | err_original_after | 0.0016369 ± 8.10058e-05 |
| write_fit | all | global | visual | fit_gain | 0.683064 ± 0.0341852 |
| write_fit | all | global-minus-fixed0.9 | audio | err_after | -0.0487677 ± 0.00451875 |
| write_fit | all | global-minus-fixed0.9 | audio | err_before | 0.0179992 ± 0.00219348 |
| write_fit | all | global-minus-fixed0.9 | audio | err_original_after | 3.75237e-05 ± 3.95534e-06 |
| write_fit | all | global-minus-fixed0.9 | audio | fit_gain | 0.0667669 ± 0.00666995 |
| write_fit | all | global-minus-fixed0.9 | text | err_after | -0.0696704 ± 0.00531317 |
| write_fit | all | global-minus-fixed0.9 | text | err_before | 0.0277485 ± 0.00149425 |
| write_fit | all | global-minus-fixed0.9 | text | err_original_after | 5.93279e-05 ± 2.91863e-06 |
| write_fit | all | global-minus-fixed0.9 | text | fit_gain | 0.0974189 ± 0.00655298 |
| write_fit | all | global-minus-fixed0.9 | visual | err_after | -0.0452897 ± 0.00343584 |
| write_fit | all | global-minus-fixed0.9 | visual | err_before | 0.0158867 ± 0.00210015 |
| write_fit | all | global-minus-fixed0.9 | visual | err_original_after | 3.4097e-05 ± 3.56486e-06 |
| write_fit | all | global-minus-fixed0.9 | visual | fit_gain | 0.0611764 ± 0.00542887 |
| write_fit | all | protected | audio | err_after | 0.0472015 ± 0.00131726 |
| write_fit | all | protected | audio | err_before | 0.734107 ± 0.0407391 |
| write_fit | all | protected | audio | err_original_after | 0.00166423 ± 7.33476e-05 |
| write_fit | all | protected | audio | fit_gain | 0.686906 ± 0.0415425 |
| write_fit | all | protected | text | err_after | 0.0704942 ± 0.00700515 |
| write_fit | all | protected | text | err_before | 1.07615 ± 0.0240046 |
| write_fit | all | protected | text | err_original_after | 0.00243311 ± 5.09934e-05 |
| write_fit | all | protected | text | fit_gain | 1.00565 ± 0.0279262 |
| write_fit | all | protected | visual | err_after | 0.0482609 ± 0.00208175 |
| write_fit | all | protected | visual | err_before | 0.696502 ± 0.0343432 |
| write_fit | all | protected | visual | err_original_after | 0.00161814 ± 8.12796e-05 |
| write_fit | all | protected | visual | fit_gain | 0.648242 ± 0.0333826 |
| write_fit | all | reference | audio | err_after | 0.00169743 ± 7.01652e-05 |
| write_fit | all | reference | audio | err_before | 0.753488 ± 0.0376194 |
| write_fit | all | reference | audio | err_original_after | 0.00169743 ± 7.01652e-05 |
| write_fit | all | reference | audio | fit_gain | 0.75179 ± 0.0375571 |
| write_fit | all | reference | text | err_after | 0.00245075 ± 4.98899e-05 |
| write_fit | all | reference | text | err_before | 1.08786 ± 0.0218112 |
| write_fit | all | reference | text | err_original_after | 0.00245075 ± 4.98899e-05 |
| write_fit | all | reference | text | fit_gain | 1.08541 ± 0.0217723 |
| write_fit | all | reference | visual | err_after | 0.00165152 ± 8.19501e-05 |
| write_fit | all | reference | visual | err_before | 0.716487 ± 0.0345091 |
| write_fit | all | reference | visual | err_original_after | 0.00165152 ± 8.19501e-05 |
| write_fit | all | reference | visual | fit_gain | 0.714835 ± 0.0344306 |

## Interpretation and limitations

Dynamic − fixed five-rate weighted-F1 difference: -0.001396 ± 0.001970; positive in 1/5 seeds. Descriptive exact two-sided sign-flip p = 0.1875.

With five paired seeds there are only 32 sign assignments (minimum two-sided p = 0.0625); this is descriptive, not a significance claim. The test assumes sign exchangeability and does not correct multiple exploratory comparisons. Rates and heads are not independent samples. Retention and current-write fit measure different behaviors and neither alone establishes downstream causality.

Fixed 0.9 scales all writes, including complete-input/no-history writes; complete-input identity is therefore not expected for that comparator. This is frozen-checkpoint evaluation only: no retraining, new checkpoint selection, or automatic module changes are justified by these results. Existing checkpoint-selection bias remains; see per-run selection protocol.
