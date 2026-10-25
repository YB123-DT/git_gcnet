# MOSI/MOSEI Flat error sources

INTERNAL DIAGNOSTIC ONLY
Original LargeFlat seed66, gcnet-random-mask-v1, cmerc-repro for both datasets.
Saved predictions only: no new model inference/training or changed masks.
Error means wrong binary polarity, excluding gold0; prediction0 is negative.
Seen means observed strictly earlier in the same conversation, not necessarily retained/retrieved.
Neutral-label turns remain in the availability history. First turns count as never-seen.
Cells are errors / nonneutral sample count (error%). -- means no samples, not zero error.
Tables are per rate. Do not add overlapping modality marginals or treat repeated rates as independent.

## MOSI

Original saved mean8 ACC=79.535061, W-F1=79.460447.
Source: `/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2`; LR=0.0003, epochs=150.

### Current observed-set split

| Rate | ALL | A (missing TV) | T (missing AV) | V (missing AT) | AT (missing V) | AV (missing T) | TV (missing A) | ATV |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 81/656 (12.3%) | -- | -- | -- | -- | -- | -- | 81/656 (12.3%) |
| 0.1 | 99/656 (15.1%) | 2/5 (40.0%) | 0/5 (0.0%) | 4/6 (66.7%) | 6/42 (14.3%) | 16/51 (31.4%) | 13/69 (18.8%) | 58/478 (12.1%) |
| 0.2 | 112/656 (17.1%) | 8/19 (42.1%) | 3/20 (15.0%) | 8/22 (36.4%) | 12/93 (12.9%) | 33/104 (31.7%) | 7/85 (8.2%) | 41/313 (13.1%) |
| 0.3 | 120/656 (18.3%) | 10/41 (24.4%) | 2/37 (5.4%) | 17/48 (35.4%) | 11/104 (10.6%) | 43/125 (34.4%) | 13/113 (11.5%) | 24/188 (12.8%) |
| 0.4 | 140/656 (21.3%) | 29/75 (38.7%) | 12/77 (15.6%) | 24/71 (33.8%) | 16/100 (16.0%) | 36/119 (30.3%) | 12/103 (11.7%) | 11/111 (9.9%) |
| 0.5 | 153/656 (23.3%) | 41/124 (33.1%) | 14/129 (10.9%) | 53/120 (44.2%) | 11/78 (14.1%) | 21/83 (25.3%) | 8/89 (9.0%) | 5/33 (15.2%) |
| 0.6 | 173/656 (26.4%) | 66/194 (34.0%) | 24/174 (13.8%) | 59/172 (34.3%) | 2/31 (6.5%) | 12/34 (35.3%) | 9/44 (20.5%) | 1/7 (14.3%) |
| 0.7 | 196/656 (29.9%) | 84/224 (37.5%) | 30/209 (14.4%) | 82/223 (36.8%) | -- | -- | -- | -- |

### Per-missing-modality history split (overlapping marginals)

For example, an A-only utterance enters both missing-T and missing-V columns. Do not sum them.

| Rate | Missing A, never | Missing A, seen | Missing T, never | Missing T, seen | Missing V, never | Missing V, seen |
|---|---|---|---|---|---|---|
| 0.0 | -- | -- | -- | -- | -- | -- |
| 0.1 | 1/2 (50.0%) | 16/78 (20.5%) | 1/2 (50.0%) | 21/60 (35.0%) | 0/2 (0.0%) | 8/50 (16.0%) |
| 0.2 | 1/4 (25.0%) | 17/123 (13.8%) | 4/13 (30.8%) | 45/132 (34.1%) | 2/12 (16.7%) | 21/120 (17.5%) |
| 0.3 | 2/15 (13.3%) | 30/183 (16.4%) | 6/13 (46.2%) | 64/201 (31.8%) | 2/13 (15.4%) | 21/169 (12.4%) |
| 0.4 | 7/20 (35.0%) | 41/231 (17.7%) | 6/21 (28.6%) | 83/244 (34.0%) | 6/19 (31.6%) | 51/233 (21.9%) |
| 0.5 | 7/23 (30.4%) | 68/315 (21.6%) | 12/23 (52.2%) | 103/304 (33.9%) | 7/25 (28.0%) | 59/306 (19.3%) |
| 0.6 | 14/34 (41.2%) | 78/356 (21.9%) | 19/47 (40.4%) | 118/353 (33.4%) | 9/34 (26.5%) | 83/365 (22.7%) |
| 0.7 | 17/54 (31.5%) | 95/378 (25.1%) | 22/55 (40.0%) | 144/392 (36.7%) | 20/50 (40.0%) | 94/383 (24.5%) |

### Exact joint history states inside each current pattern

These rows are mutually exclusive within a current pattern; counts reconcile to the table above.
For current A, T:never,V:seen means missing T has never appeared, while missing V has appeared.
ATV has no missing target and its history-state entry is N/A.

#### Rate 0.0

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|ATV|none|N/A|81/656 (12.3%)|4/26|
#### Rate 0.1

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:seen,V:seen|2/5 (40.0%)|0/0|
|T|AV|A:seen,V:seen|0/5 (0.0%)|0/0|
|V|AT|A:seen,T:seen|4/6 (66.7%)|0/0|
|AT|V|V:never|0/2 (0.0%)|0/2|
|AT|V|V:seen|6/40 (15.0%)|0/0|
|AV|T|T:never|1/2 (50.0%)|1/2|
|AV|T|T:seen|15/49 (30.6%)|0/0|
|TV|A|A:never|1/2 (50.0%)|1/2|
|TV|A|A:seen|12/67 (17.9%)|0/0|
|ATV|none|N/A|58/478 (12.1%)|2/20|
#### Rate 0.2

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:never,V:seen|1/1 (100.0%)|0/0|
|A|TV|T:seen,V:seen|7/18 (38.9%)|0/0|
|T|AV|A:seen,V:never|0/1 (0.0%)|0/0|
|T|AV|A:seen,V:seen|3/19 (15.8%)|0/0|
|V|AT|A:never,T:never|0/1 (0.0%)|0/1|
|V|AT|A:seen,T:seen|8/21 (38.1%)|0/0|
|AT|V|V:never|2/11 (18.2%)|2/6|
|AT|V|V:seen|10/82 (12.2%)|0/0|
|AV|T|T:never|3/11 (27.3%)|2/7|
|AV|T|T:seen|30/93 (32.3%)|0/0|
|TV|A|A:never|1/3 (33.3%)|0/1|
|TV|A|A:seen|6/82 (7.3%)|0/0|
|ATV|none|N/A|41/313 (13.1%)|1/11|
#### Rate 0.3

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:never,V:never|0/1 (0.0%)|0/1|
|A|TV|T:seen,V:seen|10/40 (25.0%)|0/0|
|T|AV|A:never,V:never|0/2 (0.0%)|0/1|
|T|AV|A:never,V:seen|0/1 (0.0%)|0/0|
|T|AV|A:seen,V:seen|2/34 (5.9%)|0/0|
|V|AT|A:never,T:never|0/3 (0.0%)|0/2|
|V|AT|A:seen,T:never|1/2 (50.0%)|0/0|
|V|AT|A:seen,T:seen|16/43 (37.2%)|0/0|
|AT|V|V:never|2/10 (20.0%)|2/5|
|AT|V|V:seen|9/94 (9.6%)|0/0|
|AV|T|T:never|5/7 (71.4%)|1/3|
|AV|T|T:seen|38/118 (32.2%)|0/0|
|TV|A|A:never|2/9 (22.2%)|0/5|
|TV|A|A:seen|11/104 (10.6%)|0/0|
|ATV|none|N/A|24/188 (12.8%)|0/9|
#### Rate 0.4

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:never,V:never|2/5 (40.0%)|1/4|
|A|TV|T:never,V:seen|0/3 (0.0%)|0/0|
|A|TV|T:seen,V:never|1/3 (33.3%)|0/0|
|A|TV|T:seen,V:seen|26/64 (40.6%)|0/0|
|T|AV|A:never,V:never|0/4 (0.0%)|0/4|
|T|AV|A:never,V:seen|2/4 (50.0%)|0/0|
|T|AV|A:seen,V:never|2/2 (100.0%)|0/0|
|T|AV|A:seen,V:seen|8/67 (11.9%)|0/0|
|V|AT|A:never,T:never|1/1 (100.0%)|1/1|
|V|AT|A:never,T:seen|3/5 (60.0%)|0/0|
|V|AT|A:seen,T:never|2/3 (66.7%)|0/0|
|V|AT|A:seen,T:seen|18/62 (29.0%)|0/0|
|AT|V|V:never|1/5 (20.0%)|1/4|
|AT|V|V:seen|15/95 (15.8%)|0/0|
|AV|T|T:never|1/9 (11.1%)|0/5|
|AV|T|T:seen|35/110 (31.8%)|0/0|
|TV|A|A:never|1/6 (16.7%)|1/2|
|TV|A|A:seen|11/97 (11.3%)|0/0|
|ATV|none|N/A|11/111 (9.9%)|2/6|
#### Rate 0.5

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:never,V:never|3/7 (42.9%)|2/6|
|A|TV|T:never,V:seen|0/1 (0.0%)|0/0|
|A|TV|T:seen,V:never|0/2 (0.0%)|0/0|
|A|TV|T:seen,V:seen|38/114 (33.3%)|0/0|
|T|AV|A:never,V:never|1/4 (25.0%)|1/4|
|T|AV|A:never,V:seen|0/3 (0.0%)|0/0|
|T|AV|A:seen,V:never|2/4 (50.0%)|0/0|
|T|AV|A:seen,V:seen|11/118 (9.3%)|0/0|
|V|AT|A:never,T:never|2/3 (66.7%)|2/2|
|V|AT|A:never,T:seen|2/3 (66.7%)|0/0|
|V|AT|A:seen,T:never|4/4 (100.0%)|0/0|
|V|AT|A:seen,T:seen|45/110 (40.9%)|0/0|
|AT|V|V:never|1/8 (12.5%)|0/4|
|AT|V|V:seen|10/70 (14.3%)|0/0|
|AV|T|T:never|3/8 (37.5%)|1/4|
|AV|T|T:seen|18/75 (24.0%)|0/0|
|TV|A|A:never|2/10 (20.0%)|2/5|
|TV|A|A:seen|6/79 (7.6%)|0/0|
|ATV|none|N/A|5/33 (15.2%)|0/1|
#### Rate 0.6

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:never,V:never|3/14 (21.4%)|2/9|
|A|TV|T:never,V:seen|8/11 (72.7%)|0/0|
|A|TV|T:seen,V:never|1/4 (25.0%)|0/0|
|A|TV|T:seen,V:seen|54/165 (32.7%)|0/0|
|T|AV|A:never,V:never|1/5 (20.0%)|1/5|
|T|AV|A:never,V:seen|2/6 (33.3%)|0/0|
|T|AV|A:seen,V:never|3/10 (30.0%)|0/0|
|T|AV|A:seen,V:seen|18/153 (11.8%)|0/0|
|V|AT|A:never,T:never|2/10 (20.0%)|1/8|
|V|AT|A:never,T:seen|9/13 (69.2%)|0/0|
|V|AT|A:seen,T:never|3/8 (37.5%)|0/0|
|V|AT|A:seen,T:seen|45/141 (31.9%)|0/0|
|AT|V|V:never|1/1 (100.0%)|1/1|
|AT|V|V:seen|1/30 (3.3%)|0/0|
|AV|T|T:never|3/4 (75.0%)|2/3|
|AV|T|T:seen|9/30 (30.0%)|0/0|
|TV|A|A:seen|9/44 (20.5%)|0/0|
|ATV|none|N/A|1/7 (14.3%)|0/0|
#### Rate 0.7

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:never,V:never|10/21 (47.6%)|6/13|
|A|TV|T:never,V:seen|3/7 (42.9%)|0/0|
|A|TV|T:seen,V:never|6/9 (66.7%)|0/0|
|A|TV|T:seen,V:seen|65/187 (34.8%)|0/0|
|T|AV|A:never,V:never|3/11 (27.3%)|2/8|
|T|AV|A:never,V:seen|4/17 (23.5%)|0/0|
|T|AV|A:seen,V:never|1/9 (11.1%)|0/0|
|T|AV|A:seen,V:seen|22/172 (12.8%)|0/0|
|V|AT|A:never,T:never|4/10 (40.0%)|3/5|
|V|AT|A:never,T:seen|6/16 (37.5%)|0/0|
|V|AT|A:seen,T:never|5/17 (29.4%)|0/0|
|V|AT|A:seen,T:seen|67/180 (37.2%)|0/0|
## MOSEI

Original saved mean8 ACC=83.492708, W-F1=83.288796.
Source: `/data2/yb/remote_experiments/osram_cmerc_sentiment_3seed_20261023/execution_v1/runs/CMUMOSEI/seed_66/flat`; LR=0.0001, epochs=60.

### Current observed-set split

| Rate | ALL | A (missing TV) | T (missing AV) | V (missing AT) | AT (missing V) | AV (missing T) | TV (missing A) | ATV |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 455/3634 (12.5%) | -- | -- | -- | -- | -- | -- | 455/3634 (12.5%) |
| 0.1 | 495/3634 (13.6%) | 6/28 (21.4%) | 0/29 (0.0%) | 8/28 (28.6%) | 43/315 (13.7%) | 76/313 (24.3%) | 39/314 (12.4%) | 323/2607 (12.4%) |
| 0.2 | 527/3634 (14.5%) | 19/110 (17.3%) | 21/105 (20.0%) | 15/90 (16.7%) | 66/486 (13.6%) | 107/505 (21.2%) | 58/518 (11.2%) | 241/1820 (13.2%) |
| 0.3 | 550/3634 (15.1%) | 44/232 (19.0%) | 36/275 (13.1%) | 55/253 (21.7%) | 67/581 (11.5%) | 126/576 (21.9%) | 78/615 (12.7%) | 144/1102 (13.1%) |
| 0.4 | 620/3634 (17.1%) | 99/438 (22.6%) | 55/423 (13.0%) | 84/425 (19.8%) | 79/603 (13.1%) | 151/591 (25.5%) | 90/589 (15.3%) | 62/565 (11.0%) |
| 0.5 | 636/3634 (17.5%) | 179/729 (24.6%) | 87/675 (12.9%) | 133/651 (20.4%) | 56/437 (12.8%) | 103/471 (21.9%) | 52/466 (11.2%) | 26/205 (12.7%) |
| 0.6 | 728/3634 (20.0%) | 234/929 (25.2%) | 127/1023 (12.4%) | 247/977 (25.3%) | 31/225 (13.8%) | 53/220 (24.1%) | 33/222 (14.9%) | 3/38 (7.9%) |
| 0.7 | 788/3634 (21.7%) | 313/1226 (25.5%) | 172/1187 (14.5%) | 303/1221 (24.8%) | -- | -- | -- | -- |

### Per-missing-modality history split (overlapping marginals)

For example, an A-only utterance enters both missing-T and missing-V columns. Do not sum them.

| Rate | Missing A, never | Missing A, seen | Missing T, never | Missing T, seen | Missing V, never | Missing V, seen |
|---|---|---|---|---|---|---|
| 0.0 | -- | -- | -- | -- | -- | -- |
| 0.1 | 8/50 (16.0%) | 39/321 (12.1%) | 14/52 (26.9%) | 76/317 (24.0%) | 11/56 (19.6%) | 38/316 (12.0%) |
| 0.2 | 9/97 (9.3%) | 85/616 (13.8%) | 25/113 (22.1%) | 116/592 (19.6%) | 22/115 (19.1%) | 84/586 (14.3%) |
| 0.3 | 47/249 (18.9%) | 122/894 (13.6%) | 49/190 (25.8%) | 176/871 (20.2%) | 34/213 (16.0%) | 113/875 (12.9%) |
| 0.4 | 49/286 (17.1%) | 180/1151 (15.6%) | 77/287 (26.8%) | 257/1167 (22.0%) | 62/319 (19.4%) | 171/1145 (14.9%) |
| 0.5 | 81/431 (18.8%) | 191/1361 (14.0%) | 113/451 (25.1%) | 302/1400 (21.6%) | 87/458 (19.0%) | 235/1383 (17.0%) |
| 0.6 | 132/604 (21.9%) | 275/1618 (17.0%) | 160/585 (27.4%) | 374/1541 (24.3%) | 135/614 (22.0%) | 257/1563 (16.4%) |
| 0.7 | 157/712 (22.1%) | 318/1696 (18.8%) | 240/829 (29.0%) | 376/1618 (23.2%) | 167/768 (21.7%) | 318/1645 (19.3%) |

### Exact joint history states inside each current pattern

These rows are mutually exclusive within a current pattern; counts reconcile to the table above.
For current A, T:never,V:seen means missing T has never appeared, while missing V has appeared.
ATV has no missing target and its history-state entry is N/A.

#### Rate 0.0

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|ATV|none|N/A|455/3634 (12.5%)|75/501|
#### Rate 0.1

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:never,V:never|3/7 (42.9%)|3/7|
|A|TV|T:seen,V:seen|3/21 (14.3%)|0/0|
|T|AV|A:never,V:never|0/1 (0.0%)|0/1|
|T|AV|A:never,V:seen|0/2 (0.0%)|0/0|
|T|AV|A:seen,V:never|0/1 (0.0%)|0/0|
|T|AV|A:seen,V:seen|0/25 (0.0%)|0/0|
|V|AT|A:never,T:never|1/4 (25.0%)|1/4|
|V|AT|A:never,T:seen|0/1 (0.0%)|0/0|
|V|AT|A:seen,T:seen|7/23 (30.4%)|0/0|
|AT|V|V:never|8/47 (17.0%)|7/42|
|AT|V|V:seen|35/268 (13.1%)|0/0|
|AV|T|T:never|10/41 (24.4%)|10/39|
|AV|T|T:seen|66/272 (24.3%)|0/0|
|TV|A|A:never|7/42 (16.7%)|7/38|
|TV|A|A:seen|32/272 (11.8%)|0/0|
|ATV|none|N/A|323/2607 (12.4%)|50/370|
#### Rate 0.2

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:never,V:never|2/19 (10.5%)|2/17|
|A|TV|T:never,V:seen|0/2 (0.0%)|0/0|
|A|TV|T:seen,V:never|0/1 (0.0%)|0/0|
|A|TV|T:seen,V:seen|17/88 (19.3%)|0/0|
|T|AV|A:never,V:never|3/8 (37.5%)|3/8|
|T|AV|A:never,V:seen|0/3 (0.0%)|0/0|
|T|AV|A:seen,V:never|4/7 (57.1%)|0/0|
|T|AV|A:seen,V:seen|14/87 (16.1%)|0/0|
|V|AT|A:never,T:never|2/8 (25.0%)|2/8|
|V|AT|A:seen,T:never|1/2 (50.0%)|0/0|
|V|AT|A:seen,T:seen|12/80 (15.0%)|0/0|
|AT|V|V:never|13/80 (16.2%)|11/71|
|AT|V|V:seen|53/406 (13.1%)|0/0|
|AV|T|T:never|20/82 (24.4%)|17/69|
|AV|T|T:seen|87/423 (20.6%)|0/0|
|TV|A|A:never|4/78 (5.1%)|4/67|
|TV|A|A:seen|54/440 (12.3%)|0/0|
|ATV|none|N/A|241/1820 (13.2%)|45/261|
#### Rate 0.3

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:never,V:never|10/37 (27.0%)|10/36|
|A|TV|T:never,V:seen|2/10 (20.0%)|0/0|
|A|TV|T:seen,V:never|1/12 (8.3%)|0/0|
|A|TV|T:seen,V:seen|31/173 (17.9%)|0/0|
|T|AV|A:never,V:never|9/49 (18.4%)|9/44|
|T|AV|A:never,V:seen|1/12 (8.3%)|0/0|
|T|AV|A:seen,V:never|1/8 (12.5%)|0/0|
|T|AV|A:seen,V:seen|25/206 (12.1%)|0/0|
|V|AT|A:never,T:never|13/51 (25.5%)|10/46|
|V|AT|A:never,T:seen|1/8 (12.5%)|0/0|
|V|AT|A:seen,T:never|2/8 (25.0%)|0/0|
|V|AT|A:seen,T:seen|39/186 (21.0%)|0/0|
|AT|V|V:never|13/107 (12.1%)|12/84|
|AT|V|V:seen|54/474 (11.4%)|0/0|
|AV|T|T:never|22/84 (26.2%)|16/66|
|AV|T|T:seen|104/492 (21.1%)|0/0|
|TV|A|A:never|23/129 (17.8%)|18/100|
|TV|A|A:seen|55/486 (11.3%)|0/0|
|ATV|none|N/A|144/1102 (13.1%)|19/125|
#### Rate 0.4

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:never,V:never|21/75 (28.0%)|20/69|
|A|TV|T:never,V:seen|6/20 (30.0%)|0/0|
|A|TV|T:seen,V:never|2/14 (14.3%)|0/0|
|A|TV|T:seen,V:seen|70/329 (21.3%)|0/0|
|T|AV|A:never,V:never|18/81 (22.2%)|16/70|
|T|AV|A:never,V:seen|1/18 (5.6%)|0/0|
|T|AV|A:seen,V:never|2/25 (8.0%)|0/0|
|T|AV|A:seen,V:seen|34/299 (11.4%)|0/0|
|V|AT|A:never,T:never|8/47 (17.0%)|8/45|
|V|AT|A:never,T:seen|6/26 (23.1%)|0/0|
|V|AT|A:seen,T:never|7/26 (26.9%)|0/0|
|V|AT|A:seen,T:seen|63/326 (19.3%)|0/0|
|AT|V|V:never|19/124 (15.3%)|17/78|
|AT|V|V:seen|60/479 (12.5%)|0/0|
|AV|T|T:never|35/119 (29.4%)|18/77|
|AV|T|T:seen|116/472 (24.6%)|0/0|
|TV|A|A:never|16/114 (14.0%)|10/76|
|TV|A|A:seen|74/475 (15.6%)|0/0|
|ATV|none|N/A|62/565 (11.0%)|11/86|
#### Rate 0.5

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:never,V:never|29/117 (24.8%)|27/102|
|A|TV|T:never,V:seen|18/58 (31.0%)|0/0|
|A|TV|T:seen,V:never|19/68 (27.9%)|0/0|
|A|TV|T:seen,V:seen|113/486 (23.3%)|0/0|
|T|AV|A:never,V:never|20/116 (17.2%)|13/95|
|T|AV|A:never,V:seen|8/52 (15.4%)|0/0|
|T|AV|A:seen,V:never|5/44 (11.4%)|0/0|
|T|AV|A:seen,V:seen|54/463 (11.7%)|0/0|
|V|AT|A:never,T:never|24/101 (23.8%)|17/79|
|V|AT|A:never,T:seen|12/48 (25.0%)|0/0|
|V|AT|A:seen,T:never|9/49 (18.4%)|0/0|
|V|AT|A:seen,T:seen|88/453 (19.4%)|0/0|
|AT|V|V:never|14/113 (12.4%)|10/64|
|AT|V|V:seen|42/324 (13.0%)|0/0|
|AV|T|T:never|33/126 (26.2%)|20/71|
|AV|T|T:seen|70/345 (20.3%)|0/0|
|TV|A|A:never|17/114 (14.9%)|7/62|
|TV|A|A:seen|35/352 (9.9%)|0/0|
|ATV|none|N/A|26/205 (12.7%)|5/28|
#### Rate 0.6

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:never,V:never|42/158 (26.6%)|32/130|
|A|TV|T:never,V:seen|20/85 (23.5%)|0/0|
|A|TV|T:seen,V:never|31/103 (30.1%)|0/0|
|A|TV|T:seen,V:seen|141/583 (24.2%)|0/0|
|T|AV|A:never,V:never|35/184 (19.0%)|25/143|
|T|AV|A:never,V:seen|11/94 (11.7%)|0/0|
|T|AV|A:seen,V:never|16/108 (14.8%)|0/0|
|T|AV|A:seen,V:seen|65/637 (10.2%)|0/0|
|V|AT|A:never,T:never|52/178 (29.2%)|43/140|
|V|AT|A:never,T:seen|22/92 (23.9%)|0/0|
|V|AT|A:seen,T:never|30/97 (30.9%)|0/0|
|V|AT|A:seen,T:seen|143/610 (23.4%)|0/0|
|AT|V|V:never|11/61 (18.0%)|5/27|
|AT|V|V:seen|20/164 (12.2%)|0/0|
|AV|T|T:never|16/67 (23.9%)|7/36|
|AV|T|T:seen|37/153 (24.2%)|0/0|
|TV|A|A:never|12/56 (21.4%)|5/20|
|TV|A|A:seen|21/166 (12.7%)|0/0|
|ATV|none|N/A|3/38 (7.9%)|0/5|
#### Rate 0.7

| Observed | Missing | Strict-past state | Errors/N (error%) | First-turn errors/N |
|---|---|---|---|---|
|A|TV|T:never,V:never|60/232 (25.9%)|42/173|
|A|TV|T:never,V:seen|57/184 (31.0%)|0/0|
|A|TV|T:seen,V:never|42/147 (28.6%)|0/0|
|A|TV|T:seen,V:seen|154/663 (23.2%)|0/0|
|T|AV|A:never,V:never|34/208 (16.3%)|26/157|
|T|AV|A:never,V:seen|22/138 (15.9%)|0/0|
|T|AV|A:seen,V:never|31/181 (17.1%)|0/0|
|T|AV|A:seen,V:seen|85/660 (12.9%)|0/0|
|V|AT|A:never,T:never|72/236 (30.5%)|48/171|
|V|AT|A:never,T:seen|29/130 (22.3%)|0/0|
|V|AT|A:seen,T:never|51/177 (28.8%)|0/0|
|V|AT|A:seen,T:seen|151/678 (22.3%)|0/0|

## Limits

Descriptive counts from one seed and per-rate Test-best checkpoints, not independent generalization.
Never-seen contains first turns; CSV additionally reports never_with_history and first-turn counts.
Seen/never groups may differ in conversation position, labels and content; these are not causal effects.
Availability indicates a real observation allowed by the saved mask, not a guarantee of useful content.
No inference that Memory is empty, retained the modality, or should be disabled follows from this split.
