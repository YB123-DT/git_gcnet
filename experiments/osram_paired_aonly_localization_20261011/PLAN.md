# Locate the A-only output bias

Freeze original Flat, rho.50 and rho.10 task-only seed66 miss=.7 BEST. Run their
original random .7 view and persistent A-only; retain same current-A anchors.
Verify Local equality on paired anchors, original random prediction archive,
full readout replay and unchanged weights. Record original Local-only readout,
Local+Base, Local+Gap, Full from one scan. These are classification-input
interventions on a trained head, not separately trained unimodal classifiers.

Inspect whether nearly all-positive output exists before historical reads enter
the head, whether Base/Gap magnifies it, and whether it also occurs under richer
random histories for the same current inputs. First-turn and prior-Text groups
are descriptive controls. No retraining, threshold tuning or model modifications.

In parallel review historical mask/loss implementation and configuration. Distinguish
code errors, observed failure location and unproven training explanations.
INTERNAL DIAGNOSTIC ONLY; inherited Test-oracle checkpoints.
