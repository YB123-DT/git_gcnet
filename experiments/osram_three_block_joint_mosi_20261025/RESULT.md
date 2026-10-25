# Four-arm MOSI joint-block result

Source commit: `96aae75cc9aecc6b9f16cc344f1feafd538740df`

Internal per-rate Test-oracle exploration; four matched seed66 arms.

Verified 150 epochs per arm, identical epoch optimizer counts and training mask traces, and identical eight-rate test mask hashes.

| Method | Mean ACC | Mean WF1 | High WF1 | Mean MSE | Δ WF1 vs Flat |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat | 79.2873 | 79.2583 | 73.0613 | 1.5328 | 0.0000 |
| neumiss | 79.3826 | 79.3469 | 73.3200 | 1.4546 | 0.0885 |
| signature | 78.6585 | 78.6010 | 72.9602 | 1.5151 | -0.6573 |
| csdi | 79.3636 | 79.2619 | 73.2547 | 1.4961 | 0.0036 |

| Method | Missing rate | ACC | WF1 | MSE | Δ WF1 vs Flat |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat | 0.0 | 87.9573 | 87.9285 | 1.0534 | 0.0000 |
| flat | 0.1 | 84.6037 | 84.6884 | 1.2360 | 0.0000 |
| flat | 0.2 | 83.0793 | 83.1060 | 1.4225 | 0.0000 |
| flat | 0.3 | 81.0976 | 80.9994 | 1.3589 | 0.0000 |
| flat | 0.4 | 78.0488 | 78.1603 | 1.6032 | 0.0000 |
| flat | 0.5 | 77.1341 | 76.6597 | 1.5878 | 0.0000 |
| flat | 0.6 | 73.6280 | 73.7819 | 1.9973 | 0.0000 |
| flat | 0.7 | 68.7500 | 68.7423 | 2.0035 | 0.0000 |
| neumiss | 0.0 | 87.3476 | 87.3565 | 1.0402 | -0.5720 |
| neumiss | 0.1 | 84.6037 | 84.5388 | 1.1897 | -0.1496 |
| neumiss | 0.2 | 83.0793 | 83.0388 | 1.3726 | -0.0671 |
| neumiss | 0.3 | 81.8598 | 81.7716 | 1.3074 | 0.7722 |
| neumiss | 0.4 | 78.2012 | 78.1094 | 1.4742 | -0.0509 |
| neumiss | 0.5 | 76.9817 | 76.8385 | 1.4935 | 0.1788 |
| neumiss | 0.6 | 73.9329 | 74.0446 | 1.8141 | 0.2627 |
| neumiss | 0.7 | 69.0549 | 69.0767 | 1.9452 | 0.3344 |
| signature | 0.0 | 87.0427 | 87.0329 | 1.0456 | -0.8956 |
| signature | 0.1 | 83.3841 | 83.3538 | 1.1565 | -1.3346 |
| signature | 0.2 | 82.1646 | 82.2410 | 1.4044 | -0.8649 |
| signature | 0.3 | 80.3354 | 80.1546 | 1.4094 | -0.8448 |
| signature | 0.4 | 77.2866 | 77.1453 | 1.5434 | -1.0150 |
| signature | 0.5 | 75.7622 | 75.6753 | 1.6389 | -0.9844 |
| signature | 0.6 | 73.9329 | 73.7331 | 1.8292 | -0.0488 |
| signature | 0.7 | 69.3598 | 69.4721 | 2.0935 | 0.7298 |
| csdi | 0.0 | 88.1098 | 88.1310 | 1.0026 | 0.2024 |
| csdi | 0.1 | 85.0610 | 85.0536 | 1.3146 | 0.3651 |
| csdi | 0.2 | 82.1646 | 82.2197 | 1.3793 | -0.8863 |
| csdi | 0.3 | 81.2500 | 81.1063 | 1.3031 | 0.1069 |
| csdi | 0.4 | 78.0488 | 77.8205 | 1.4926 | -0.3398 |
| csdi | 0.5 | 77.1341 | 76.6838 | 1.5883 | 0.0241 |
| csdi | 0.6 | 74.2378 | 74.2989 | 1.8655 | 0.5170 |
| csdi | 0.7 | 68.9024 | 68.7813 | 2.0226 | 0.0390 |

## Scope and provenance

Remote artifacts: `/data2/yb/remote_experiments/osram_three_block_joint_mosi_20261025/execution_v1` on `ssh biggpu`; cmerc-repro, seed66. Results were recomputed from saved selected predictions, not newly sampled test masks. Per-rate Test-oracle selection is internal exploration, not independent generalization. One seed does not establish statistical significance; missing rates are not independent repeated seeds.

All four arms completed 150 epochs with matching optimizer counts, training mask traces and exact test-mask hashes. The comparison uses the newly trained RNG-repaired Flat control, not the historical parent used by the separate frozen stage2 experiment.
