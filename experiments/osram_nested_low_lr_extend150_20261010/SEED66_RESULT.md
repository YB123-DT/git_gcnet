# Seed66 completed; multi-seed continuation pending

INTERNAL DIAGNOSTIC ONLY. Per-rate Test-oracle BEST, MOSI.
Full-state constant100 continuation at1e-4 for epochs101–150, no warmup/cosine.

| Model | Original mean8 | New101–150-only mean8 | Cumulative150 mean8 | Original high | New-only high | Cumulative high |
|---|---:|---:|---:|---:|---:|---:|
| Flat | 81.068 | 79.143 | 81.068 | 76.352 | 74.085 | 76.352 |
| Nested | 80.992 | 80.509 | 81.163 | 76.077 | 75.515 | 76.077 |

Only Nested miss.3 updates:80.522617→81.888389 (+1.365772pp), yielding
mean8+.170722pp. Other seven rates unchanged; Flat eight rates all unchanged.
Cumulative scores include inherited BEST; do not describe new-only scores as
uniform improvement. Seed66 alone does not establish stable benefit.

Both final provenance records saycomplete/outputs_verified with150history and
50actualLR files; actual epoch150 group rates all1e-4. Full artifacts retained.
Original100 histories and checkpoint hashes preserved by runner final checks.
67/68 original Nested recovery states are available and authorized for the
same lowerLR continuation. Flat67/68 have no full recovery states and are not
included in this multi-seed continuation expansion.
