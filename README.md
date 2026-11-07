# VoxSense

VoxSense is a local research and educational dashboard for numerical acoustic measurements in two supplied Parkinson’s voice datasets. It does **not** analyze raw audio, recognize speech, or diagnose disease. Its outputs are research estimates and must not guide care.

## What is in this repository

- `backend/`: FastAPI service, dataset profiling, grouped model training and API tests.
- `frontend/`: responsive Next.js dashboard connected to the local API.
- `data/raw/`: source files, preserved unchanged.
- `backend/ml/explore.py`: repeatable quality profile and feature distribution plots.
- `reports/`: methods, measured model evaluation, and documented dataset findings.
- `docs/ARCHITECTURE.md`: service boundaries and data flow.

## Dataset inspection

The provided `parkinsons.names` describes 197 instances / 23 attributes, while the actual CSV has **195 rows and 24 columns**: 22 numerical acoustic features, `status`, and `name`. There are 48 control rows and 147 Parkinson’s rows. `status` is the binary target (0 control, 1 Parkinson’s). No blank or exact duplicate rows were found. The file’s name field groups recordings under 32 unique prefixes, although the supplied documentation states 31 people; the discrepancy is retained for verification. The model split uses these prefixes as groups, preventing recordings from one prefix crossing training and test sets. The raw name is never a predictor.

`parkinsons_updrs.data` has 5,875 rows and 22 columns: 16 acoustic measures, `subject#`, `age`, `sex`, `test_time`, and the confirmed targets `motor_UPDRS` and `total_UPDRS`. It contains 42 subjects, no blank rows or exact duplicates, and 101–168 observations per subject. Recorded time ranges from −4.2625 to 215.49 days; negative time values are retained as supplied. Subjects are all in an early-stage Parkinson’s study, so this file cannot distinguish disease from control. Its interpolated scores and repeated measures support descriptive within-study trajectories, not proven future prediction.

Severity models use only the 16 voice measures. Subject, time, age, sex, and both score columns are excluded from predictors to avoid patient identity, temporal, demographic, and target leakage. A patient-disjoint 25% holdout is used. Classification likewise uses a patient-group-disjoint holdout and grouped five-fold cross-validation on training groups. Grid search and model selection operate only on training patients; classification uses grouped-CV balanced accuracy with ROC-AUC tie-breaking, and regression uses grouped-CV MAE. Class balancing is confined to the logistic regression and SVM training fits. Small patient counts make all reported scores uncertain and non-clinical.

Numeric missing values, if encountered, are median-imputed within each training pipeline. IQR outliers are flagged for investigation in the generated report and plots; none are silently removed or winsorized. Scaling is fitted within pipelines. Seeds are fixed at 42. No processed copy overwrites source data.

## Run locally

Use Python 3.11 (recommended) and Node.js 20 or newer.

Place the four supplied source files (`parkinsons.data`, `parkinsons.names`, `parkinsons_updrs.data`, and `parkinsons_updrs.names`) in `data/raw/`. They are intentionally excluded from Git.

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.services.training
python -m uvicorn app.main:app --reload
```

In a second terminal:

```powershell
cd frontend
npm install
$env:NEXT_PUBLIC_API_URL="http://localhost:8000/api"
npm run dev
```

Open `http://localhost:3000`; API docs are at `http://localhost:8000/docs`. Raw file paths resolve relative to the repository, not the working directory. Run all backend commands from `backend/`.

To create data quality outputs and histograms, from `backend/` run `python -m ml.explore`. To retrain, run `python -m app.services.training`; model files and measured metrics are written to ignored `backend/artifacts/` and are not committed. Install `requirements.txt` before invoking these commands. XGBoost comparison is added automatically when the optional `xgboost` package is installed; it is omitted by default because its Windows wheel is very large for this small cohort.

## API

- `GET /api/health`, `/api/overview`, `/api/datasets`, `/api/features/schema`, `/api/samples`
- `GET /api/models/metrics`, `/api/models/importance`, `/api/models/shap-summary`
- `GET /api/records/{classification|severity}?limit=100&offset=0`
- `GET /api/progression/{patient_id}`, `/api/progression-summary`
- `POST /api/predict/classification`, `/api/predict/severity`, `/api/explain` with `{"values": {"feature": number}}`
- `POST /api/predict/batch` with `{"csv_text": "..."}` or `{"rows": [...]}` for validated CSV batch classification and downloadable results

Prediction requests require the exact feature names expected by each model. Errors use standard FastAPI status codes and a readable detail. Inference returns an unavailable response until real models have been trained. Explanation endpoint reports its actual method; feature contributions are associations, not causation.

## Known limits

The individual explanation endpoint uses SHAP permutation values against a training background; the global importance endpoint reports mean absolute SHAP values on supplied records. Contributions are associations, not causes. Dataset IQR flags are univariate and do not establish erroneous records. There is no external validation cohort, prospective evaluation, calibration study, fairness analysis, or medical-device validation. The interface has no user accounts or persistent patient uploads; supplied records remain local.

## Verification

The data profile and distribution figures were generated from the supplied CSVs. The grouped, tuned training pipeline ran and wrote real model artifacts and metrics locally. Backend tests passed (`11 passed`). The Next.js production build passed; the environment fell back from its invalid Windows native SWC module to the WASM compiler. SHAP individual and global importance endpoints returned real values, and the summary image endpoint returned a PNG. `reports/MODEL_EVALUATION.md` documents the measured holdout results, including poor classification specificity and negative severity R². Artifacts remain ignored and are reproducible with the documented commands; no performance metrics are fabricated.
