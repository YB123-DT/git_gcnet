# Current status — not training results

INTERNAL TEST-ORACLE FEATURE-SOURCE DIAGNOSTIC.

| Feature source | Data state | Training state | ACC/UA |
|---|---|---|---|
| MAGTKD |Complete; canonical13,708utterances and17field conversion verified|Corrected v2running onGPU3;1/50epochs and full-state commit verified|Pending|
| MM-Align |dev/test/embedding verified; train incomplete due Drive quota; dev110_7absent|Not started; strict full-coverage guard|Unavailable|

No performance claim. No local training or environment change. The intended
only-data comparison is seed66/50epochs,original GCNet random missing and Flat.
User clarification is required before dropping any missing MM-Align utterance or
introducing a reduced-sample control; current code refuses both silently.

Prepared MAGTKD hash and provenance are in `MAGTKD_PREPARED.json`. Upstream
acquisition integrity/source notes are separate artifacts. The helper submitted
the job after SSH recovered. Its three-batch smoke completed,but the first
formal epoch hit a Torch1.13UUID RNG enumeration issue. The failed directory is
preserved. A process-local compatibility fix passed real GPU save/restore and
was deployed as a fresh v2run,not an incomplete-state continuation.
