# Model evaluation report

Generated from the supplied datasets with seed 42. One patient-group-disjoint 25% test split is held out. Grid search and model selection run only inside grouped training cross-validation (five folds for classification, three for severity); balanced accuracy with ROC-AUC tie breaking selects the classifier and grouped-CV MAE selects regressors. These are internal research results only.

## Parkinson’s classification

The holdout has 49 recording rows from patient groups not used in model fitting: 12 controls and 37 Parkinson’s recordings. Results show a substantial specificity trade-off: all four models label almost every held-out record as Parkinson’s. High recall here does not mean useful screening performance.

| Model | CV balanced accuracy | CV ROC-AUC | Accuracy | Precision | Recall | Specificity | F1 | Test ROC-AUC | Confusion matrix (control row, PD row) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.761 | 0.881 | 0.612 | 0.765 | 0.703 | 0.333 | 0.732 | 0.649 | `[[4,8],[11,26]]` |
| SVM (selected by CV) | 0.774 | 0.784 | 0.653 | 0.727 | 0.865 | 0.000 | 0.790 | 0.385 | `[[0,12],[5,32]]` |
| Random Forest | 0.761 | 0.850 | 0.735 | 0.761 | 0.946 | 0.083 | 0.843 | 0.703 | `[[1,11],[2,35]]` |
| Gradient Boosting | 0.716 | 0.766 | 0.776 | 0.771 | 1.000 | 0.083 | 0.871 | `[[1,11],[0,37]]` |

The selected SVM is the strongest grouped-CV balanced-accuracy model (0.774). On held-out patients it has poor discrimination (ROC-AUC 0.385), misses 5 of 37 PD recordings, and incorrectly flags all 12 controls. The other models also have specificity at or below 0.333. The classifier is not suitable for diagnosis or screening; results are unstable on this small and imbalanced cohort.

## UPDRS severity regression

The same patient-disjoint split contains repeated test records. Predictors are the 16 voice measures only; age, sex, subject ID, time, and targets are excluded. Grid search and regressor selection use grouped CV MAE on training participants. Gradient Boosting has the lowest training-CV MAE for both scores. Every model has negative test R², so each is worse than predicting the test-set mean under this split.

| Target | Model | CV MAE | Test MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Motor UPDRS | Linear Regression | 7.83 | 6.83 | 7.92 | -0.10 |
| Motor UPDRS | Random Forest | 8.01 | 7.22 | 8.44 | -0.25 |
| Motor UPDRS | Gradient Boosting (selected) | 7.60 | 6.88 | 7.71 | -0.05 |
| Total UPDRS | Linear Regression | 10.42 | 7.96 | 9.29 | -0.13 |
| Total UPDRS | Random Forest | 10.22 | 8.61 | 10.38 | -0.41 |
| Total UPDRS | Gradient Boosting (selected) | 9.75 | 7.87 | 9.18 | -0.10 |

The severity models remain weak for patient-disjoint estimates despite the lower MAE from Gradient Boosting; negative R² for each target is a clear limitation. No accuracy claim should be inferred from the apparent training fit or repeated measurements. XGBoost remains an optional model comparison and was not installed because its Windows wheel exceeded available system-drive space during setup; install it separately to include it on retraining.

## Explainability and data quality

Individual explanations are computed with SHAP’s permutation explainer using the classification model and a 20-row training background. Global importance is mean absolute SHAP contribution over 12 supplied classification records. These contributions describe model behavior, not causal biomarkers.

Run `python -m ml.explore` from `backend/` to regenerate `reports/generated/data_quality.json` and the distribution figures `parkinsons_distributions.png` and `parkinsons_updrs_distributions.png`. IQR flags are exploratory; no records were automatically removed.
