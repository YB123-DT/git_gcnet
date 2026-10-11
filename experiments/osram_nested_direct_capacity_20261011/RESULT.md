# Direct random Nested capacity sweep: complete

INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle; not a formal paper result.

Both authorized batches completed:8/8 MOSI seed66 runs,100epochs each, exit0,
all eight BEST checkpoints present. No capacity sweep training processes remain.
Original batch ran on biggpu GPU7; additional batches on GPU0/GPU1. Bad GPU4
excluded. Code snapshots c08682c and7939317 remain isolated per original launch.

Original Flat1600, Local Skip, OSRAM and task loss retained. Direct refers to
replacing adapter input evidence with Nested outputs WITHOUT the external raw
evidence addition; original Flat is NOT removed. Decoder uses random initialization.
Node projection dimension and shared GIN MLP width vary; original rooted topology,
eight memory heads, three GIN layers and training protocol otherwise retained.

W-F1%, high missing=.5/.6/.7; module parameters are total minus Flat13509793.

| Node dim | GIN MLP width | Module params | Eight-rate | High missing | Eight-rate−Flat pp |
|---:|---:|---:|---:|---:|---:|
| Flat reference | — | — |81.068|76.352|0.000|
|64|128|216899|80.122|74.825|-0.946|
|64|256|332227|79.501|74.573|-1.567|
|64|512|562883|78.584|73.156|-2.484|
|128|128|432387|78.533|72.882|-2.535|
|128|256|662403|78.124|72.596|-2.944|
|128|512|1122435|79.158|73.891|-1.910|
|256|256|1322755|77.980|72.454|-3.088|
|256|512|2241539|78.736|73.923|-2.332|

No tested capacity increase surpasses Flat; scores are not monotonically related
to parameter count. This is a single-seed screen, not evidence all direct methods
or bigger GNNs universally fail. Earlier direct random64/64 scored80.177/high74.800,
so even the best new eight-rate result does not exceed that earlier configuration.

Full per-rate scores and dispatch/exit records:
- Initial4: this folder/completed/{SUMMARY,DISPATCH}.json.
- Additional4: ../osram_nested_direct_capacity_more_20261011/completed/lane{0,1}/.

Verified on biggpu:100 history records for every run, all eight BEST files,
complete status and exit0. Checkpoints stay remote; none uploaded to Git.
No reruns, additional tuning or new training were performed in this completion check.
