# Width sweep selected epochs

INTERNAL DIAGNOSTIC ONLY. MOSI seed66, all runs completed100 epochs. Read-only recheck of COMPLETED_RESULT.json; no new training/inference. Epochs are one-based, ordered by missing rate .0,.1,.2,.3,.4,.5,.6,.7.

| Width | Model | Per-rate BEST epochs | Median | Latest selected epoch |
|---|---|---|---:|---:|
|384|Flat|80,80,77,80,77,77,88,77|78.5|88|
|384|Nested|51,51,71,89,71,71,59,59|65|89|
|512|Flat|75,75,75,76,79,76,78,78|76|79|
|512|Nested|49,68,70,91,70,69,70,70|70|91|
|768|Flat|56,60,60,85,86,86,60,92|72.5|92|
|768|Nested|78,71,74,87,86,73,98,87|82|98|
|1024|Flat|64,64,63,94,93,63,93,64|64|94|
|1024|Nested|85,85,85,97,83,85,87,87|85|97|
|1280|Flat|83,83,82,100,46,82,80,84|82.5|100|
|1280|Nested|75,72,70,85,72,85,72,93|73.5|93|

This series selects substantially later than the MOSEI results. Nested768 selects miss.6 at98; Nested1024 selects miss.3 at97; Flat1280 selects miss.3 at100. Endpoint proximity suggests those specific conditions could be budget-sensitive, but does not establish that extension will improve aggregate performance or beat original large Flat. There is no uniform Nested convergence delay across widths. The latest selected epoch refers to the last member of the eight-checkpoint collection, not a shared global-best checkpoint.
