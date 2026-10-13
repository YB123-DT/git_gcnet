# Causal ComP CMU-MOSEI reproduction

This queued follow-up covers seeds 66/67/68 and missing rates 0.0--0.7.
Each rate uses 100 epochs and switches to the second stage at epoch 50, matching
the official ComP paper Figure 5(a).  It is queued behind the IEMOCAP-4/6
200/100 run on the same assigned physical GPU3 and does not include MOSI.
Each worker is capped at four CPU compute threads.
