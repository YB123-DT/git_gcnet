# Nested correction subset OFF diagnostic

INTERNAL DIAGNOSTIC ONLY. Frozen MOSI seed66 original100epoch Nested, original
per-rate Test-oracle BEST. No training or new checkpoint selection.

Five correction slots: Local, Base, Gap-A, Gap-T, Gap-V. Test all31 nonempty
OFF subsets (5 singles,10 pairs,10 triples,5 quadruples,1 all), with shared Full
ON baseline. Only restore original raw evidence in the selected slots; never
delete evidence, skip or Memory. Reuse existing one-scan readout replay. The
32 conditions include Full, not32 independent Memory scans or training runs.

Same eight test masks, labels/filter/threshold. Report overall/high and seven
current observed patterns, equal macro over nonempty rate groups. Not persistent
whole-modality missing. Include OFF->ON corrections/harms and sample counts;
do not choose a deployable switch using test labels. Effects need not add.

Server biggpu healthyGPU7 only, UUID GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e.
GPU4 forbidden. Independent source/archive/output path. Prior singleton and
all-OFF diagnostics remain intact. Verify all-OFF and singletons reproduce prior
predictions, Full matches archived masks/labels/predictions, hashes unchanged.
