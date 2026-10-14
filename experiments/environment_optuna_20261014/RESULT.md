# Optuna installed in the active remote experiment environment

Server: `ssh biggpu`. Environment Python:
`/data2/yb/conda_envs/cmerc-repro/bin/python` (Python3.10).

Requested environment installation only; no model search was authorized or
started by this change. Existing width512 training and frozen task diagnostic
continued, without restart, configuration change or interruption.

## Command

```bash
ssh biggpu '/data2/yb/conda_envs/cmerc-repro/bin/python -m pip install --no-cache-dir optuna==5.0.0'
```

Official installation reference:
https://optuna.readthedocs.io/en/stable/installation.html

The dry-run resolver indicated additions only; no existing core training
package required an upgrade. Actual added packages:

| Package | Version |
|---|---|
| optuna | 5.0.0 |
| alembic | 1.20.0 |
| SQLAlchemy | 2.0.54 |
| colorlog | 6.12.0 |
| PyYAML | 6.0.3 |
| Mako | 1.4.3 |
| greenlet | 3.5.6 |

## Verification

- Optuna import from the exact environment succeeded.
- Three in-memory CPU toy optimization trials completed; no GPU/model used.
- `pip check` before and after: `No broken requirements found.`
- Core versions before and after matched: Torch1.13.1+cu117,
  NumPy1.26.4, scikit-learn1.3.2. Explicit assertions passed.
- Existing training processes and frozen replay continued producing progress.

No project-wide requirements were silently changed. This records an explicit
user-requested environment addition, not a new dependency of the original
CHM inference path and not evidence of model performance improvement.
