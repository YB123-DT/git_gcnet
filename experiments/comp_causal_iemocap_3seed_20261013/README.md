# Causal ComP IEMOCAP reproduction

This run covers IEMOCAP-4 and IEMOCAP-6, seeds 66/67/68, and missing rates
0.0--0.7.  Each rate is trained independently for 200 epochs with the second
stage beginning at epoch 100.  This budget follows the training curves in the
official ComP paper Figure 5(b-c); it is not inherited from the 300/150 MOSI
example.

The cancelled uniform-budget run remains archived at
`comp_causal_all_3seed_300e_20261013` and must not be mixed with this result.
