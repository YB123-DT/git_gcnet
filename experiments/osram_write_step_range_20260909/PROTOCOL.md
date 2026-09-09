# Coarse lower-range extension after the eta=.8 boundary result

User explicitly requested continuing to locate the useful region after the
initial four-point grid. This extension is chosen AFTER observing that grid;
do not present all eight points as originally preregistered.

New points locked before execution: eta=.6/.4/.2/0. No .81/.82-style fine
search. Inherit all four prior points (1/.95/.9/.8); do not rerun them.
Two datasets: IEMOCAPFour and CMUMOSI; five seeds66–70; same five missing rates
0/.1/.3/.5/.7; same already-trained forward-only best.pt per seed, masks,
config, task metric and raw diagnostic definitions. No training or modules.

Combine eight points to locate a sampled interior peak and bracket it by its
adjacent tested values, or identify the eta0/eta1 endpoint as the best sampled
boundary. Report seed/rate consistency and dataset differences. This is coarse
localization, not a continuous/global optimality claim. Do not automatically
add more points or retrain a new strength after looking at results.

Eta0 has zero memory write from the initial zero state. Base/Gap memory values
are therefore zero, but the existing learned local/fusion/classifier parameters
remain unchanged. This is a frozen off-memory intervention, NOT a retrained
Local-only model. The official missing input, student latents and local path
still respond to availability/content. JEPA/MMoE are not invoked at inference.

Keep E_old=historical missing-probe err_decay at actual read (NO_HISTORY excluded)
and E_new=current observed err_after. At eta0, nonzero-value probes have relative
error approximately1; no information has been stored. Do not label this as an
accidental representation collapse or use it to claim memory is generally useless.

200 new evaluation cells plus 200 inherited grid cells. CPU only, two PyTorch
threads per process, five parallel seeds per dataset. Existing eight-rate-mean
Test-oracle checkpoint selection; report FIVE-rate means and diagnostics only.
No parameter changes, checkpoint selection, or optimization performed.

Git keeps summaries, code, provenance and raw SHA256 manifests; full compressed
records stay under the same experiment folder locally and on biggpu.
INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.
