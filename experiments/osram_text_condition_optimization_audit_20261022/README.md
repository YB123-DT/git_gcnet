# Flat Text-condition optimization audit

One bounded diagnostic, no new module, training run or HPO. Original LargeFlat
seed66 MOSI/MOSEI, s0 environment, gcnet-random-mask-v1. Freeze the original
miss0.3 BEST checkpoint (MOSI epoch50,MOSEI epoch19) as the SAME weight anchor for
all conditions. These checkpoints were Test-oracle selected: INTERNAL DIAGNOSTIC.
No result here is the accepted per-rate Test-best baseline or deployment score.

1. Replay exact original eight-rate TEST banks. One original CHM scan per batch,
   then readout-only history zeroing for Local-only using original LocalSkip,
   Flat,LN/head. Report currentTextvisible/currentTextmissing+historyseen/never
   groups with MSE,ACC,WF1 and Local-to-Full corrections/harms. Previous labels
   are never model inputs. Readout zeroing can be out-of-distribution.
2. TRAIN only: original host/guest GCNet sampler generates background masks on
   batch32 padded rectangles,seed66/epoch0,each rate0..0.7. Pick one valid target
   per conversation,at most8 per rate,using IDs/masks,not labels. Require earlier
   valid history and at least one current observedA/V. Two diagnostic prefixes
   differ ONLY in target's currentText (forceon vs forceoff); all earlier raw
   features,availability,speakers and parameters are identical. Texton can reveal
   actual current rawText originally masked; this is explicitly a diagnostic
   intervention,not a change to the GCNet sampler or an ordinary sampled score.
   Isolate one conversation/prefix only AFTER generating the original masks;
   never resample a batch1 mask. Inspect same-target MSE gradients of the shared
   model,overall and disjoint parameter groups. Past predictions and target
   decayed pre-write Memory must agree between views. Source/target labels are
   not used to choose target positions. Zero-norm gradients have no cosine.
   For paired TRAIN diagnostics only,execute the same utterance-local observed
   encoder separately on the past prefix and last/current utterance,then join
   its outputs before the single original CHM scan. This avoids packed-GEMM
   rounding changes in the identical history when current Text changes the
   number of observed rows. Both pieces retain their full autograd graphs.
   Original test replay and all production/training code remain untouched.
   Because MSE gradients include2*(prediction-y),also inspect residual signs and
   prediction-Jacobian cosines obtained by dividing out this scalar (except
   numerical near-zero residuals). Opposite loss directions alone do not prove
   competing representation learning.
3. Starting from the same fixed weight each time,temporarily apply one plain
   SGD1e-5 step from either condition,then measure BOTH paired losses. Restore
   all original parameters immediately and verify final model digest unchanged.
   No optimizer state is constructed or saved,no persistent training checkpoint.
   Log actual update/parameter norm ratio,self/cross loss deltas and first-order
   prediction -lr*gradient_dot. This checks finite local sensitivity,NOT Adam
   dynamics,independent held-out transfer or a long-term training-conflict claim.

Gradients use eval-mode/dropout-off to separate modality condition from random
dropout. Original train loss remains scalar MSE. Test labels are offline strata
only; virtual updates use train labels only. Repeated conversations across rates
are not independent observations. No threshold/step-size search or best-step
selection, no new seed, no persistent mixed training or test-label optimization.

Separate immutable source/output from every existing experiment. Commands:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
 -m experiments.osram_text_condition_optimization_audit_20261022.run \
 --dataset CMUMOSI --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
 --source-commit SOURCE_SHA --output OUTPUT_ROOT/CMUMOSI
# Same command with --dataset CMUMOSEI and independent OUTPUT_ROOT/CMUMOSEI.
```

Run at most2 diagnostic processes on healthyGPU7 after memory checks. NeverGPU4,
never signal other tasks. Full paired details/predictions stay remote; small
reports/config/hash provenance and CSV summaries go to Git. Existing weights,
predictions and masks are never overwritten or deleted.
