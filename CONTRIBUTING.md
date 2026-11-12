# Contributing

Use Python 3.10–3.12 and an isolated environment. Install `requirements-dev.txt`.
Run `python -m pytest -q` and `python review_results.py` before submitting changes.

Keep the original authors and distinguish recorded experiment metrics from newly measured results.
For model changes, document the split, seed, preprocessing, checkpoint, hardware, timing protocol
and metric definitions. Never evaluate a development set as an independent test set.

Do not commit raw datasets, credentials, local environments or generated inference outputs.
Preserve the notebook as the original training workflow. Reusable inference lives in
`recognition/model.py`, `recognition/inference.py` and `predict.py`; changes must preserve
checkpoint compatibility and the recorded preprocessing protocol unless explicitly versioned.
