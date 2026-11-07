"""Leakage-aware grouped model training. Run: python -m app.services.training"""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
from app.core.config import ARTIFACTS, RANDOM_STATE
from app.services.data import dataset

def grouped_holdout(X, y, groups, test_size=.25):
    """Return patient-disjoint train/test row indices with the project seed."""
    splitter=GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=RANDOM_STATE)
    return next(splitter.split(X,y,groups))

def severity_features(columns):
    excluded={"subject#","test_time","age","sex","motor_UPDRS","total_UPDRS"}
    return [column for column in columns if column not in excluded]

def train():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    results = {"classification": {}, "severity": {}}
    c = dataset("classification")
    X = c.drop(columns=["name", "status"]); y = c.status
    groups = c.name.str.rsplit("_", n=1).str[0]
    tr, te = grouped_holdout(X,y,groups)
    candidates = {"Logistic Regression": (make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)), {"logisticregression__C":[.1,1,10]}),
                  "SVM": (make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(probability=False, class_weight="balanced", random_state=RANDOM_STATE)), {"svc__C":[.1,1,10]}),
                  "Random Forest": (make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=40, class_weight="balanced", random_state=RANDOM_STATE)), {"randomforestclassifier__min_samples_leaf":[1,3]}),
                  "Gradient Boosting": (make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(random_state=RANDOM_STATE)), {"gradientboostingclassifier__n_estimators":[50,100],"gradientboostingclassifier__learning_rate":[.05,.1]})}
    try:
        from xgboost import XGBClassifier
        candidates["XGBoost"] = (make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(n_estimators=150,max_depth=2,learning_rate=.04,subsample=.8,colsample_bytree=.8,eval_metric="logloss",random_state=RANDOM_STATE)), {"xgbclassifier__max_depth":[2,3]})
    except ImportError: pass
    cv = GroupKFold(n_splits=5)
    fitted = {}
    for name, (model, grid) in candidates.items():
        search=GridSearchCV(model,grid,scoring={"balanced_accuracy":"balanced_accuracy","recall":"recall","roc_auc":"roc_auc","f1":"f1"},refit="balanced_accuracy",cv=cv,n_jobs=1)
        search.fit(X.iloc[tr],y.iloc[tr],groups=groups.iloc[tr]); model=search.best_estimator_; ix=search.best_index_
        if name=="SVM":
            model.set_params(svc__probability=True).fit(X.iloc[tr],y.iloc[tr])
        fitted[name]=model
        pred = model.predict(X.iloc[te]); prob = model.predict_proba(X.iloc[te])[:,1]; fpr,tpr,_=roc_curve(y.iloc[te],prob)
        results["classification"][name] = {"cv_recall":float(search.cv_results_["mean_test_recall"][ix]), "cv_balanced_accuracy":float(search.best_score_), "cv_roc_auc":float(search.cv_results_["mean_test_roc_auc"][ix]),
          "cv_f1":float(search.cv_results_["mean_test_f1"][ix]), "best_params":search.best_params_, "accuracy":float(accuracy_score(y.iloc[te],pred)), "precision":float(precision_score(y.iloc[te],pred,zero_division=0)),
          "recall":float(recall_score(y.iloc[te],pred,zero_division=0)), "specificity":float(recall_score(y.iloc[te],pred,pos_label=0,zero_division=0)), "f1":float(f1_score(y.iloc[te],pred,zero_division=0)),
          "roc_auc":float(roc_auc_score(y.iloc[te],prob)), "roc_curve":[{"fpr":float(a),"tpr":float(b)} for a,b in zip(fpr,tpr)], "confusion_matrix":confusion_matrix(y.iloc[te],pred).tolist(),
          "report":classification_report(y.iloc[te],pred,output_dict=True,zero_division=0)}
    # Select on grouped-CV balanced accuracy with ROC-AUC tie-break; no test score selects the model.
    best = max(results["classification"], key=lambda k:(results["classification"][k]["cv_balanced_accuracy"],results["classification"][k]["cv_roc_auc"]))
    model = fitted[best]
    joblib.dump({"model":model,"features":list(X.columns),"name":best,"baseline":X.iloc[tr].median().to_dict(),"background":X.iloc[tr].sample(min(20,len(tr)),random_state=RANDOM_STATE).to_dict(orient="records")}, ARTIFACTS/"classification.joblib")
    results["classification_selected"] = best
    s = dataset("severity")
    features = severity_features(s.columns)
    sg = s["subject#"]
    strn, ste = grouped_holdout(s[features],s.motor_UPDRS,sg)
    regression_cv=GroupKFold(n_splits=3)
    regs = {"Linear Regression":(make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LinearRegression()),{}),
            "Random Forest":(make_pipeline(SimpleImputer(strategy="median"),RandomForestRegressor(n_estimators=20,random_state=RANDOM_STATE)),{"randomforestregressor__min_samples_leaf":[2,5]}),
            "Gradient Boosting":(make_pipeline(SimpleImputer(strategy="median"),GradientBoostingRegressor(random_state=RANDOM_STATE)),{"gradientboostingregressor__n_estimators":[30,60],"gradientboostingregressor__learning_rate":[.05,.1]})}
    try:
        from xgboost import XGBRegressor
        regs["XGBoost"]=(make_pipeline(SimpleImputer(strategy="median"),XGBRegressor(n_estimators=150,max_depth=2,learning_rate=.04,subsample=.8,colsample_bytree=.8,objective="reg:squarederror",random_state=RANDOM_STATE)),{"xgbregressor__max_depth":[2,3]})
    except ImportError: pass
    for target in ("motor_UPDRS","total_UPDRS"):
        results["severity"][target]={}; reg_scores={}; reg_fitted={}
        for name, (reg, grid) in regs.items():
            search=GridSearchCV(reg,grid,scoring="neg_mean_absolute_error",cv=regression_cv,n_jobs=1)
            search.fit(s.iloc[strn][features],s.iloc[strn][target],groups=sg.iloc[strn]); chosen_model=search.best_estimator_; reg_fitted[name]=chosen_model; reg_scores[name]=-float(search.best_score_)
            pred=chosen_model.predict(s.iloc[ste][features]); actual=s.iloc[ste][target]
            results["severity"][target][name]={"cv_mae":reg_scores[name],"best_params":search.best_params_,"mae":float(mean_absolute_error(actual,pred)),"rmse":float(mean_squared_error(actual,pred)**.5),"r2":float(r2_score(actual,pred))}
            if name=="Linear Regression": regs[name]=(chosen_model,grid)
        selected=min(reg_scores,key=reg_scores.get)
        chosen=reg_fitted[selected]
        joblib.dump({"model":chosen,"features":features,"name":selected,"target":target,"baseline":s.iloc[strn][features].median().to_dict()},ARTIFACTS/f"{target}.joblib")
        results["severity"][target+"_selected"]=selected
    results["protocol"]="One patient-group-held-out test split (25% subjects); grouped 5-fold classification and 3-fold severity GridSearchCV run only on training patients. Classifier selection uses grouped-CV balanced accuracy with ROC-AUC tie-break; severity model selection uses grouped-CV MAE. Test subjects are not used for tuning or model selection. Severity uses only the 16 voice measures. Metrics are patient-disjoint, not a prospective clinical validation."
    (ARTIFACTS/"metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    return results

if __name__ == "__main__": print(json.dumps(train(),indent=2))
