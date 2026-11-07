from functools import lru_cache
import numpy as np
import pandas as pd
from app.core.config import ARTIFACTS, RAW, RANDOM_STATE
from sklearn.model_selection import GroupShuffleSplit

FEATURE_METADATA = {
    "MDVP:Fo(Hz)": {"label": "Average Fundamental Frequency (Fo)", "unit": "Hz", "group": "Fundamental Frequency (Pitch)", "description": "Average vocal fundamental frequency across the sustained vowel phonation."},
    "MDVP:Fhi(Hz)": {"label": "Maximum Fundamental Frequency (Fhi)", "unit": "Hz", "group": "Fundamental Frequency (Pitch)", "description": "Maximum vocal fundamental frequency recorded during phonation."},
    "MDVP:Flo(Hz)": {"label": "Minimum Fundamental Frequency (Flo)", "unit": "Hz", "group": "Fundamental Frequency (Pitch)", "description": "Minimum vocal fundamental frequency recorded during phonation."},
    "MDVP:Jitter(%)": {"label": "MDVP Jitter (%)", "unit": "%", "group": "Frequency Perturbation (Jitter)", "description": "Cycle-to-cycle variation in fundamental frequency expressed as a percentage."},
    "MDVP:Jitter(Abs)": {"label": "MDVP Absolute Jitter", "unit": "s", "group": "Frequency Perturbation (Jitter)", "description": "Absolute cycle-to-cycle period perturbation in seconds."},
    "MDVP:RAP": {"label": "Relative Average Perturbation (RAP)", "unit": "ratio", "group": "Frequency Perturbation (Jitter)", "description": "Relative average perturbation smoothed over 3-period windows."},
    "MDVP:PPQ": {"label": "Five-Point Period Perturbation Quotient (PPQ)", "unit": "ratio", "group": "Frequency Perturbation (Jitter)", "description": "Period perturbation quotient smoothed over 5-period windows."},
    "Jitter:DDP": {"label": "Difference of Differences of Periods (DDP)", "unit": "ratio", "group": "Frequency Perturbation (Jitter)", "description": "Average absolute difference between consecutive differences between pitch periods (3 × RAP)."},
    "MDVP:Shimmer": {"label": "MDVP Local Shimmer", "unit": "ratio", "group": "Amplitude Perturbation (Shimmer)", "description": "Cycle-to-cycle variation in vocal peak-to-peak amplitude."},
    "MDVP:Shimmer(dB)": {"label": "MDVP Shimmer (dB)", "unit": "dB", "group": "Amplitude Perturbation (Shimmer)", "description": "Cycle-to-cycle peak amplitude variation measured in decibels."},
    "Shimmer:APQ3": {"label": "Three-Point Amplitude Perturbation Quotient (APQ3)", "unit": "ratio", "group": "Amplitude Perturbation (Shimmer)", "description": "Three-point amplitude perturbation quotient across glottal cycles."},
    "Shimmer:APQ5": {"label": "Five-Point Amplitude Perturbation Quotient (APQ5)", "unit": "ratio", "group": "Amplitude Perturbation (Shimmer)", "description": "Five-point amplitude perturbation quotient across glottal cycles."},
    "MDVP:APQ": {"label": "MDVP 11-Point Amplitude Perturbation Quotient (APQ)", "unit": "ratio", "group": "Amplitude Perturbation (Shimmer)", "description": "11-point amplitude perturbation quotient capturing broader amplitude instability."},
    "Shimmer:DDA": {"label": "Average Absolute Differences of Amplitude (DDA)", "unit": "ratio", "group": "Amplitude Perturbation (Shimmer)", "description": "Average absolute difference between consecutive differences of cycle amplitudes (3 × APQ3)."},
    "NHR": {"label": "Noise-to-Harmonics Ratio (NHR)", "unit": "ratio", "group": "Harmonicity & Glottal Noise", "description": "Ratio of inharmonic noise energy to harmonic tonal energy in the voice signal."},
    "HNR": {"label": "Harmonics-to-Noise Ratio (HNR)", "unit": "dB", "group": "Harmonicity & Glottal Noise", "description": "Ratio of harmonic periodic energy to acoustic noise in decibels (higher indicates clearer phonation)."},
    "RPDE": {"label": "Recurrence Period Density Entropy (RPDE)", "unit": "norm [0–1]", "group": "Nonlinear Dynamical Complexity", "description": "Uncertainty in glottal pitch period periodicity measured via phase-space recurrence."},
    "DFA": {"label": "Detrended Fluctuation Analysis (DFA)", "unit": "exponent", "group": "Nonlinear Dynamical Complexity", "description": "Signal fractal scaling exponent characterizing turbulent airflow self-similarity."},
    "spread1": {"label": "Nonlinear Pitch Frequency Variation (spread1)", "unit": "log scale", "group": "Nonlinear Dynamical Complexity", "description": "First nonlinear measure of fundamental frequency variation."},
    "spread2": {"label": "Nonlinear Pitch Frequency Variation (spread2)", "unit": "ratio", "group": "Nonlinear Dynamical Complexity", "description": "Second nonlinear measure of fundamental frequency variation."},
    "D2": {"label": "Correlation Dimension (D2)", "unit": "dimension", "group": "Nonlinear Dynamical Complexity", "description": "Nonlinear dynamical complexity estimate representing degrees of freedom in vocal fold oscillation."},
    "PPE": {"label": "Pitch Period Entropy (PPE)", "unit": "entropy", "group": "Nonlinear Dynamical Complexity", "description": "Logarithmic entropy measure of impaired pitch control unaffected by smooth vibrato."}
}

@lru_cache(maxsize=2)
def dataset(name: str) -> pd.DataFrame:
    files = {"classification": "parkinsons.data", "severity": "parkinsons_updrs.data"}
    if name not in files:
        raise KeyError(name)
    path = RAW / files[name]
    if not path.exists():
        raise FileNotFoundError(f"Required dataset is missing: {path.name}")
    frame = pd.read_csv(path)
    frame.columns = [c.strip() for c in frame.columns]
    return frame

@lru_cache(maxsize=1)
def classification_split_indices() -> tuple[set[int], set[int]]:
    c = dataset("classification")
    X = c.drop(columns=["name", "status"])
    groups = c.name.str.rsplit("_", n=1).str[0]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr, te = next(splitter.split(X, c.status, groups))
    return set(int(i) for i in tr), set(int(i) for i in te)

@lru_cache(maxsize=1)
def load_inference_metadata() -> dict:
    meta_path = ARTIFACTS / "inference_metadata.json"
    if meta_path.exists():
        import json
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}

def profile(name: str) -> dict:
    try:
        df = dataset(name)
        numeric = df.select_dtypes(include="number")
        return {"name": name, "rows": len(df), "columns": len(df.columns),
                "missing_cells": int(df.isna().sum().sum()), "duplicate_rows": int(df.duplicated().sum()),
                "column_info": [{"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()),
                                 "unique": int(df[c].nunique()), "min": float(df[c].min()) if c in numeric else None,
                                 "max": float(df[c].max()) if c in numeric else None} for c in df.columns],
                "sample": df.head(8).where(pd.notna(df.head(8)), None).to_dict(orient="records")}
    except Exception:
        meta = load_inference_metadata()
        if "datasets_profile" in meta:
            for item in meta["datasets_profile"]:
                if item.get("name") == name:
                    return item
        raise

def feature_schema() -> list[dict]:
    try:
        c = dataset("classification")
        features = [x for x in c.columns if x not in ("name", "status")]
        healthy = c[c.status == 0]
        pd_group = c[c.status == 1]
        out = []
        for f in features:
            meta = FEATURE_METADATA.get(f, {"label": f, "unit": "value", "group": "Acoustic Features", "description": f})
            col = c[f]
            q1, q3 = float(col.quantile(0.25)), float(col.quantile(0.75))
            span = max(float(col.max()) - float(col.min()), 1e-6)
            out.append({
                "name": f,
                "label": meta["label"],
                "unit": meta["unit"],
                "group": meta["group"],
                "description": meta["description"],
                "min": float(col.min()),
                "max": float(col.max()),
                "median": float(col.median()),
                "mean": float(col.mean()),
                "std": float(col.std()),
                "q1": q1,
                "q3": q3,
                "healthy_median": float(healthy[f].median()),
                "parkinsons_median": float(pd_group[f].median()),
                "healthy_mean": float(healthy[f].mean()),
                "parkinsons_mean": float(pd_group[f].mean()),
                "valid_min": float(max(0.0, col.min() - 1.5 * span)) if col.min() >= 0 else float(col.min() - 1.5 * span),
                "valid_max": float(col.max() + 1.5 * span),
            })
        return out
    except Exception:
        meta = load_inference_metadata()
        if "features_schema" in meta:
            return meta["features_schema"]
        raise

def sample_patients() -> list[dict]:
    try:
        c = dataset("classification")
        tr_idx, te_idx = classification_split_indices()
        features = [x for x in c.columns if x not in ("name", "status")]
        curated = [
            ("phon_R01_S16_1", "Held-out test recording from a participant with Parkinson's disease showing elevated pitch period entropy (PPE) and nonlinear frequency spread."),
            ("phon_R01_S22_1", "Held-out test recording from a participant with Parkinson's disease with lower fundamental frequency and moderate amplitude perturbation."),
            ("phon_R01_S44_1", "Held-out test recording from a participant with Parkinson's disease exhibiting high harmonics-to-noise ratio but subtle nonlinear complexity shifts."),
            ("phon_R01_S25_1", "Held-out test recording from a participant with Parkinson's disease whose mild jitter/shimmer profile leads to a false-negative model estimate."),
            ("phon_R01_S13_1", "Held-out test recording from a healthy control participant that triggers a false-positive estimate, illustrating limited unseen-subject specificity."),
            ("phon_R01_S10_1", "Training-set reference recording from a healthy control participant with low jitter, low PPE, and stable vocal harmonics."),
            ("phon_R01_S07_1", "Training-set reference recording from a healthy control participant demonstrating high HNR and low frequency perturbation."),
            ("phon_R01_S01_1", "Training-set reference recording from a participant with Parkinson's disease exhibiting classic vocal tremor, elevated jitter, and reduced HNR."),
        ]
        samples = []
        for rec_name, note in curated:
            matches = c[c["name"] == rec_name]
            if matches.empty:
                continue
            idx = int(matches.index[0])
            row = matches.iloc[0]
            is_test = idx in te_idx
            status = int(row["status"])
            subject_group = str(row["name"]).rsplit("_", 1)[0]
            samples.append({
                "id": str(row["name"]),
                "subject_group": subject_group,
                "actual_status": status,
                "actual_label": "Parkinson's Disease (status = 1)" if status == 1 else "Healthy Control (status = 0)",
                "partition": "test" if is_test else "train",
                "partition_label": "Held-Out Test Set (Unseen Patient Group)" if is_test else "Training Set (In-Sample Patient Group)",
                "is_independent_test": is_test,
                "partition_notice": (
                    "This subject group was completely held out from model training and cross-validation tuning."
                    if is_test else
                    "This subject group was part of the model training split. Model output on this record reflects in-sample fit, NOT independent test performance."
                ),
                "description": note,
                "values": {f: float(row[f]) for f in features}
            })
        return samples
    except Exception:
        meta = load_inference_metadata()
        if "samples" in meta:
            return meta["samples"]
        raise

def overview() -> dict:
    try:
        c, s = dataset("classification"), dataset("severity")
        vc = c.status.value_counts().sort_index()
        neg_rows = s[s.test_time < 0]
        post_rows = s[s.test_time >= 0]
        subj_spans = s.groupby("subject#")["test_time"].agg(lambda x: float(x.max() - x.min()))
        healthy = c[c.status == 0]
        pd_group = c[c.status == 1]
        key_biomarkers = []
        for feat, label, unit in [
            ("PPE", "Pitch Period Entropy (PPE)", "entropy"),
            ("spread1", "Nonlinear Frequency Spread (spread1)", "log"),
            ("MDVP:Jitter(%)", "Fundamental Frequency Jitter", "%"),
            ("MDVP:Shimmer", "Amplitude Shimmer", "ratio"),
            ("HNR", "Harmonics-to-Noise Ratio (HNR)", "dB"),
            ("RPDE", "Recurrence Period Density Entropy", "norm"),
        ]:
            key_biomarkers.append({
                "feature": feat,
                "label": label,
                "unit": unit,
                "healthy_mean": float(healthy[feat].mean()),
                "healthy_median": float(healthy[feat].median()),
                "parkinsons_mean": float(pd_group[feat].mean()),
                "parkinsons_median": float(pd_group[feat].median()),
            })
        return {
            "classification_records": len(c),
            "healthy_records": int(vc.get(0, 0)),
            "parkinsons_records": int(vc.get(1, 0)),
            "classification_subjects": int(c.name.str.rsplit("_", n=1).str[0].nunique()),
            "severity_records": len(s),
            "severity_subjects": int(s["subject#"].nunique()),
            "severity_days": [float(s.test_time.min()), float(s.test_time.max())],
            "severity_post_recruitment_days": [float(post_rows.test_time.min()), float(post_rows.test_time.max())],
            "severity_pre_baseline_records": int(len(neg_rows)),
            "severity_pre_baseline_subjects": [int(x) for x in sorted(neg_rows["subject#"].unique())],
            "severity_pre_baseline_range": [float(neg_rows.test_time.min()), float(neg_rows.test_time.max())] if not neg_rows.empty else [0.0, 0.0],
            "severity_subject_span_days": [round(float(subj_spans.min()), 2), round(float(subj_spans.max()), 2)],
            "severity_time_explanation": (
                "In parkinsons_updrs.data, test_time measures days relative to each participant's formal trial recruitment baseline (Day 0). "
                "Exactly 12 records across 2 participants (Subject 34 at -3.31d and Subject 42 at -4.26d) were captured 3 to 4 days prior to their formal baseline recruitment date. "
                "Post-recruitment measurements span 0.0 to 215.49 days, and individual participant follow-up spans 153.0 to 204.0 elapsed days."
            ),
            "motor_mean": float(s.motor_UPDRS.mean()),
            "motor_std": float(s.motor_UPDRS.std()),
            "motor_min": float(s.motor_UPDRS.min()),
            "motor_max": float(s.motor_UPDRS.max()),
            "total_mean": float(s.total_UPDRS.mean()),
            "total_std": float(s.total_UPDRS.std()),
            "total_min": float(s.total_UPDRS.min()),
            "total_max": float(s.total_UPDRS.max()),
            "classification_features": [x for x in c.columns if x not in ("name", "status")],
            "severity_features": [x for x in s.columns if x not in ("subject#", "test_time", "motor_UPDRS", "total_UPDRS")],
            "key_biomarkers": key_biomarkers,
        }
    except Exception:
        meta = load_inference_metadata()
        if "overview" in meta:
            return meta["overview"]
        raise

