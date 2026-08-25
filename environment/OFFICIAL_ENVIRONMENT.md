# Shared official GCNet environment

All four published versions use the same environment and trainer.

Verified environment on `ssh biggpu`:

```text
Python          3.8.20
PyTorch         1.8.0
PyG             2.0.1
torch-scatter   2.0.8
NumPy           1.21.6
SciPy           1.7.3
scikit-learn    1.0.2
pandas          2.0.3
tqdm            4.70.0
```

The verified interpreter is:

```text
/data2/yb/reproduction_envs/gcnet-official/bin/python
```

Feature extraction is outside this compact repository. Point `--data-root` at
an IEMOCAP directory containing the label pickle and the selected audio, text
and visual feature directories. Formal paired runs must also provide the
locked `--mask-bank-root`.

Do not replace this environment with the unrelated `mcv` environment when
checking historical result compatibility.
