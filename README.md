# VoxSense 🧠
### Explainable AI for Parkinson's Voice Biomarker Analysis

VoxSense is an interactive AI-powered research application that analyses voice-derived acoustic measurements to explore patterns associated with Parkinson's disease.

It combines machine learning, longitudinal symptom analysis and Explainable AI (XAI) to provide transparent predictions and insights through an intuitive web application.

Users can explore acoustic biomarkers, generate predictions using trained ML models, analyse patient-level measurements and understand the factors influencing model outputs.

> **Disclaimer:** VoxSense is intended strictly for research and educational purposes. It is not a medical diagnostic tool and must not be used to make clinical decisions. The supplied datasets contain numerical acoustic measurements extracted from voice recordings; VoxSense does not process raw audio or perform speech recognition.

---

## ✨ Key Features

### 1. Interactive Parkinson's Classification
- Predicts whether an acoustic feature profile belongs to a healthy control or Parkinson's disease group.
- Supports manual feature entry.
- Provides sample patient analysis using records from the supplied dataset.
- Supports CSV-based batch predictions.
- Displays model predictions and probabilities where supported.

### 2. Explainable AI
- Individual prediction explanations using SHAP.
- Global feature importance.
- SHAP summary visualisations.
- Positive and negative feature contributions.
- Interpretable insights into model behaviour.

### 3. UPDRS Severity Analysis
- Predicts Motor UPDRS scores.
- Predicts Total UPDRS scores.
- Compares regression models.
- Evaluates predictions using MAE, RMSE and R².

### 4. Longitudinal Patient Insights
- Explores repeated measurements across participants.
- Visualises Motor UPDRS and Total UPDRS trends.
- Compares patient trajectories.
- Provides descriptive analysis of symptom measurements over time.

### 5. Model Laboratory
- Model performance comparison.
- Classification metrics and confusion matrices.
- ROC and precision-recall analysis.
- Regression evaluation.
- Feature importance visualisations.

### 6. Data Explorer
- Dataset previews.
- Feature descriptions.
- Data quality summaries.
- Missing-value analysis.
- Feature distributions and statistical insights.

---

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Frontend | Next.js, React, TypeScript |
| Styling | Tailwind CSS |
| Backend | Python, FastAPI |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost (optional) |
| Explainable AI | SHAP |
| Visualisation | Recharts, Matplotlib, Plotly |
| Validation | Pydantic |
| Testing | Pytest |

---

## 📊 Datasets

VoxSense uses two publicly available Parkinson's voice measurement datasets.

### Dataset 1: Parkinson's Classification Dataset

**Source:** [UCI Machine Learning Repository — Parkinson's Disease](https://archive.ics.uci.edu/dataset/174/parkinsons)

| Attribute | Details |
|---|---|
| Records | 195 |
| Features | 22 numerical acoustic features |
| Healthy controls | 48 |
| Parkinson's samples | 147 |
| Target | `status` |
| Unique recording-name prefixes | 32 |

The `status` column represents the binary classification target:

- `0` — Healthy control
- `1` — Parkinson's disease group

The dataset includes acoustic measurements such as:

- MDVP:Fo
- MDVP:Fhi
- MDVP:Flo
- Jitter
- Shimmer
- NHR
- HNR
- RPDE
- DFA
- PPE

The supplied documentation describes 197 instances and 23 attributes, while the actual CSV contains 195 rows and 24 columns, including the `name` identifier. This discrepancy is retained and documented for transparency.

The recording-name prefixes are used to create patient/recording-group-aware splits. The original `name` field is excluded from model predictors.

### Dataset 2: Parkinson's Telemonitoring Dataset

**Source:** [UCI Machine Learning Repository — Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)

| Attribute | Details |
|---|---|
| Records | 5,875 |
| Columns | 22 |
| Participants | 42 |
| Acoustic features | 16 |
| Target variables | `motor_UPDRS`, `total_UPDRS` |
| Measurements per participant | 101–168 |

This dataset contains repeated voice measurements from individuals with early-stage Parkinson's disease.

It is used for:

- Motor UPDRS regression.
- Total UPDRS regression.
- Patient-level longitudinal exploration.
- Descriptive symptom trend analysis.

The dataset does not contain healthy control participants and is therefore not used for disease classification.

### Dataset files

Place the following files inside `data/raw/`:

```text
data/
└── raw/
    ├── parkinsons.data
    ├── parkinsons.names
    ├── parkinsons_updrs.data
    └── parkinsons_updrs.names
```

**Data privacy:** The original datasets are intentionally excluded from Git. Users must obtain the datasets from their respective sources and place them in the specified directory.

---

## 🧪 Machine Learning Methodology

### Classification

The classification pipeline uses the Parkinson's classification dataset.

Models compared:

- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosting
- XGBoost (optional)

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The classification pipeline uses recording-group-aware splitting and grouped cross-validation to reduce the risk of recordings from the same group appearing in both training and test sets.

Model selection uses grouped cross-validation balanced accuracy, with ROC-AUC as a tie-breaker.

### Severity Regression

Separate regression models are trained for:

- Motor UPDRS
- Total UPDRS

The regression pipeline uses the 16 acoustic measurements as predictors.

Patient identifiers, time, demographic attributes and target variables are excluded from predictors.

Evaluation metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

A patient-disjoint holdout is used to evaluate generalisation to unseen participants.

### Preprocessing and Leakage Prevention

The implementation includes:

- Missing-value detection and median imputation.
- IQR-based outlier investigation.
- Feature scaling within training pipelines.
- Categorical and numerical preprocessing where applicable.
- Group-aware data splitting.
- Cross-validation and hyperparameter tuning.
- Fixed random seeds for reproducibility.

Preprocessing transformations are fitted only on training data. Raw datasets remain unchanged.

Class balancing is restricted to applicable training fits.

---

## 🔍 Explainable AI

VoxSense uses SHAP to make model predictions more transparent.

The explainability module provides:

| Explanation | Purpose |
|---|---|
| Global feature importance | Identifies features influential across predictions |
| SHAP summary plot | Shows the distribution of feature contributions |
| Individual explanations | Explains a particular prediction |
| Feature contribution ranking | Highlights influential measurements |
| Prediction interpretation | Communicates model behaviour in understandable language |

Feature contributions represent model associations and should not be interpreted as evidence of clinical causation.

---

## 🏗️ Project Architecture

```text
VoxSense/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── schemas/
│   │   └── services/
│   │
│   ├── ml/
│   │   ├── preprocessing/
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── explainability/
│   │
│   ├── artifacts/
│   ├── tests/
│   └── requirements.txt
│
├── frontend/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── ARCHITECTURE.md
│
├── reports/
│
├── .gitignore
└── README.md
```

The backend handles data processing, model inference, explainability and API operations. The Next.js frontend provides the interactive user experience.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 (recommended)
- Node.js 20 or newer
- npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/RagulRishwanth/VoxSense.git
cd VoxSense
```

### 2. Add the datasets

Download the two datasets from the UCI Machine Learning Repository and place the four source files inside:

```text
data/raw/
```

### 3. Set up the backend

**Windows PowerShell:**

```powershell
cd backend

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**macOS/Linux:**

```bash
cd backend

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Train the models

Run the training pipeline from the `backend/` directory:

```bash
python -m app.services.training
```

Trained model artifacts and measured evaluation results are written to:

```text
backend/artifacts/
```

The artifacts are excluded from Git and can be reproduced using the training pipeline.

### 5. Start the backend

From the `backend/` directory:

```bash
python -m uvicorn app.main:app --reload
```

Backend:

`http://localhost:8000`

Interactive API documentation:

`http://localhost:8000/docs`

### 6. Start the frontend

Open a second terminal:

```bash
cd frontend

npm install
```

Set the backend API URL.

**Windows PowerShell:**

```powershell
$env:NEXT_PUBLIC_API_URL="http://localhost:8000/api"

npm run dev
```

**macOS/Linux:**

```bash
export NEXT_PUBLIC_API_URL="http://localhost:8000/api"

npm run dev
```

Open:

**http://localhost:3000**

---

## 🔌 API Reference

The FastAPI backend exposes the following endpoints.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend health check |
| GET | `/api/overview` | Dataset overview |
| GET | `/api/datasets` | Dataset information |
| GET | `/api/features/schema` | Model feature schema |
| GET | `/api/samples` | Sample records |
| GET | `/api/models/metrics` | Model evaluation metrics |
| GET | `/api/models/importance` | Global feature importance |
| GET | `/api/models/shap-summary` | SHAP summary |
| GET | `/api/records/classification` | Classification records |
| GET | `/api/records/severity` | Severity records |
| GET | `/api/progression/{patient_id}` | Patient progression data |
| GET | `/api/progression-summary` | Longitudinal summary |
| POST | `/api/predict/classification` | Parkinson's classification |
| POST | `/api/predict/severity` | UPDRS severity prediction |
| POST | `/api/explain` | Individual prediction explanation |
| POST | `/api/predict/batch` | Batch CSV classification |

Prediction endpoints require the exact feature names expected by the corresponding model.

Batch classification accepts CSV text or structured rows and returns validated predictions.

---

## ✅ Testing and Verification

The project has been tested locally.

| Verification | Result |
|---|---|
| Backend unit tests | 11 passed |
| Next.js production build | Passed |
| Grouped model training pipeline | Executed successfully |
| Model artifacts | Generated locally |
| SHAP individual explanations | Returned real values |
| SHAP global importance | Returned real values |
| SHAP summary endpoint | Returned PNG |
| Data exploration | Generated profiles and distribution plots |

The Windows environment initially encountered an invalid native SWC module and successfully fell back to the WASM compiler during the Next.js production build.

Detailed evaluation findings are documented in:

`reports/MODEL_EVALUATION.md`

### Model evaluation transparency

The measured holdout results include poor classification specificity and negative R² values for severity regression. These limitations are documented rather than concealed.

All metrics are generated from actual model evaluation. No performance results are fabricated.

---

## ⚠️ Limitations and Responsible Use

VoxSense has several important limitations:

1. The classification dataset is small and imbalanced.
2. The longitudinal dataset contains only 42 participants.
3. The longitudinal cohort consists of individuals with early-stage Parkinson's disease.
4. The supplied datasets contain extracted acoustic measurements, not original audio recordings.
5. There is no independent external validation cohort.
6. Prospective evaluation, calibration validation and comprehensive fairness analysis have not been established.
7. The application has not undergone medical-device validation.

Additional considerations:

- IQR outlier flags are statistical indicators and do not establish erroneous records.
- SHAP contributions indicate model associations, not causal relationships.
- Longitudinal visualisations are descriptive and do not establish future progression forecasting.
- Predictions must not be used as a substitute for professional medical assessment.

---

## 🔮 Future Enhancements

Potential extensions include:

- Integration of raw voice recording analysis using an appropriate open-source audio dataset.
- Acoustic feature extraction from uploaded `.wav` recordings.
- Real-time voice biomarker analysis.
- Additional pathological voice datasets.
- Improved external validation and model calibration.
- Multimodal analysis combining acoustic, clinical and demographic information where appropriate.

These are future directions and are not represented as currently implemented functionality.

---

## 👨‍💻 Author

**Ragul Rishwanth P**

GitHub: [@RagulRishwanth](https://github.com/RagulRishwanth)

LinkedIn: [Ragul Rishwanth](https://www.linkedin.com/in/ragul-rishwanth-5330ba210/)

---

*VoxSense — Making voice biomarker analysis more transparent through explainable AI.*
