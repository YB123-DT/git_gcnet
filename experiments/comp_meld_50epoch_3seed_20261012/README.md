# Causal ComP MELD 50-epoch three-seed screen

`INTERNAL DIAGNOSTIC ONLY`. Seeds 66/67/68 run concurrently on physical
biggpu GPU3. Each seed processes missing rates 0.0 through 0.7 sequentially.
Every rate uses 50 total epochs with the original two-stage boundary scaled to
25 epochs. Other ComP hyperparameters remain identical to the completed
300/150-epoch causal MELD run.

The shortened budget is a screening comparison and must not be mixed with the
completed 300-epoch results without an explicit protocol label.
