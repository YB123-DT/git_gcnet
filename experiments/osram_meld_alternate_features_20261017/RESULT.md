# Current status — not training results

INTERNAL TEST-ORACLE FEATURE-SOURCE DIAGNOSTIC.

| Feature source | Data state | Training state | ACC/UA |
|---|---|---|---|
| MAGTKD |Complete; canonical13,708utterances and17field conversion verified|Blocked by biggpu SSH banner timeouts/resets; not yet started|Unavailable|
| MM-Align |dev/test/embedding verified; train incomplete due Drive quota; dev110_7absent|Not started; strict full-coverage guard|Unavailable|

No performance claim. No local training or environment change. The intended
only-data comparison is seed66/50epochs,original GCNet random missing and Flat.
User clarification is required before dropping any missing MM-Align utterance or
introducing a reduced-sample control; current code refuses both silently.

Prepared MAGTKD hash and provenance are in `MAGTKD_PREPARED.json`. Upstream
acquisition integrity/source notes are separate artifacts. The submission helper
can automatically start the approved MAGTKD smoke/training on the same server
when connectivity returns; a waiting helper is not evidence of running training.
