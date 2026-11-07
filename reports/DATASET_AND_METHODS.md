# Dataset and modeling notes

## Inspected source files

| File | Actual rows | Actual columns | Target / structure |
|---|---:|---:|---|
| `parkinsons.data` | 195 | 24 | `status` (0/1), repeated coded recording groups in `name` |
| `parkinsons_updrs.data` | 5,875 | 22 | `motor_UPDRS`, `total_UPDRS`; 42 `subject#` IDs, ordered `test_time` |

The classification class counts are 48 controls and 147 Parkinson’s recordings (75.4% Parkinson’s). No missing fields or exact duplicate rows were found in either CSV. The classification names file claims 197 rows and 23 attributes, which does not match the actual file. It claims 31 subjects; the actual recording prefix creates 32 groups. Those discrepancies should be checked against the source rather than silently corrected.

## Leakage controls and evaluation

Classification excludes `name`; grouped holdout and grouped CV keep coded recording groups together. Severity predictors are the 16 acoustic columns only. Subject, time, age, sex, and both UPDRS targets are withheld. One 25% subject-disjoint test partition with seed 42 is used. Grid search uses grouped five-fold CV for classification and grouped three-fold CV for severity, always within training subjects. Classifier choice uses CV balanced accuracy and ROC-AUC tie break; regressor choice uses CV MAE. Test subjects are not used for tuning or selection. Do not interpret these internal results as clinical performance.

## Quality and outlier method

`backend/ml/explore.py` reports types, summary distributions, missingness, duplicates, group sizes and flags univariate observations beyond 1.5 IQR. Flags are not removals. Histograms help investigate skew and measurement tails. Median imputation and standardization are performed inside train-fitted pipelines. Source files are read-only inputs.

## Progression scope

UPDRS telemonitoring includes repeated recordings per person and linearly interpolated scores over about seven months, so individual and aggregate descriptions are supportable. Observed trends are not a future progression model. The study has only 42 participants, all diagnosed with early-stage Parkinson’s disease.
