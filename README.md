# Model serving and monitoring

Train a binary classifier from a numeric CSV, track an experiment with MLflow, serve it using FastAPI, check standardized feature mean shifts, and measure HTTP prediction latency. A weekly GitHub Actions workflow retrains on the included **synthetic** example data; put this project at a repository root for GitHub Actions to discover the workflow. It does not deploy a model or retrain on new production data by itself.

```sh
pip install -r requirements.txt
python train.py data/example.csv model.joblib
python drift.py model.joblib data/example.csv
MODEL_PATH=model.joblib uvicorn app:app --port 8000
python benchmark.py --features 1,2 --n 100
```

Input columns are numeric features followed by binary `label`; the endpoint expects `{"features":[number,...]}` in CSV order. The demo data is synthetic, not MLB or Blinker data. The artifact uses joblib and must come from a trusted source. The simple drift rule is a heuristic, not a full monitoring system; its threshold needs calibration. Keep the API local or add authentication and production hardening.

## Verified locally (Nov 24, 2026)

- Trained on `data/mlb.csv` (2,838 MLB player-game rows built by `make_mlb_data.py`; 2,270 used for training): accuracy 0.625, ROC-AUC 0.52. The features carry almost no signal; this project demonstrates the serving and monitoring workflow, not a useful predictor.
- Served with uvicorn and benchmarked 1,000 sequential local requests: p50 1.17 ms, p95 1.78 ms, max 16 ms (the first, cold request took about 1.9 s).
- `drift.py` on the training data flagged no features (standardized mean shifts about 0.01).
