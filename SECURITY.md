# Security Policy

## Model file trust (`outputs/model.joblib`)

`train_regression.py` saves the fitted model with `joblib.dump`, which uses
Python's `pickle` format under the hood. `score_new_days.py` loads it with
`joblib.load`, which — like any `pickle`-based loader — **executes arbitrary
code if pointed at an untrusted file**.

Only load `model.joblib` files you trained yourself, or that came from a
source you trust. Don't load one downloaded from an unknown origin.

As a lightweight safeguard, `train_regression.py` writes a
`model.joblib.sha256` checksum sidecar next to the model, and
`score_new_days.py` checks the model against it before loading (see
`utils.write_sha256_sidecar` / `utils.verify_sha256_sidecar`). This catches
accidental corruption or a swapped file. **It is not a substitute for only
running models from sources you trust** — a malicious actor can regenerate a
matching checksum for a malicious file just as easily as a legitimate one.

## Reporting a vulnerability

If you find a security issue in this project (beyond the pickle-trust
tradeoff documented above, which is a known and accepted design constraint
of using `joblib`/`pickle` for `sklearn` Pipelines), please open a GitHub
issue or contact the maintainer directly rather than filing a public issue
with exploit details.
