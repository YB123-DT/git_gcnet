# Frozen Nested per-slot correction intervention

INTERNAL DIAGNOSTIC ONLY. MOSI seed66, original 100epoch Nested per-rate
Test-oracle BEST checkpoint. No training, no checkpoint reselection.

Keep the same Memory scan, query, write, raw Local Skip, Flat and head. Replay
five readouts, independently restoring raw Local / Base / Gap-A / Gap-T / Gap-V
in place of that slot's corrected value. Never remove the original evidence.
Not leave-one-head-out: each intervention removes the full slot correction.

Use original cyclic test masks for eight rates; split by current observed
A/T/V/AT/AV/TV/ATV, not whole-conversation fixed missing. Equal mean over
nonempty per-rate groups; high missing .5/.6/.7. Report full/on and slot-off
W-F1, on-minus-off pp, corrections/harms (off -> on) and sample counts.
No-text group also retained. Repeated samples across rates are not unique IDs.

Reuse runner with --per-slot; defaults preserve the previous all-delta-off
diagnostic. Verify exact archive labels/masks, ON prediction agreement <=1e-5,
unchanged model/checkpoint hashes and frozen parameters. Server biggpu GPU7
UUID GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e; GPU4 forbidden.
