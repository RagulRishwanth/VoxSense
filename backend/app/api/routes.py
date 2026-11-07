from functools import lru_cache
import csv
import io
import math
from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from app.core.config import ARTIFACTS
from app.services.data import dataset, feature_schema, overview, profile, sample_patients

router = APIRouter(prefix="/api")

class Features(BaseModel):
    values: dict[str, float] = Field(..., min_length=1)

class BatchRequest(BaseModel):
    csv_text: Optional[str] = None
    rows: Optional[list[dict[str, Any]]] = None

@lru_cache(maxsize=3)
def artifact(name):
    p = ARTIFACTS / f"{name}.joblib"
    if not p.exists():
        raise HTTPException(503, "Models are not trained yet. Run python -m app.services.training from backend.")
    return joblib.load(p)

def _validate_feature_values(required_features: list[str], values: dict[str, float]) -> dict[str, float]:
    absent = set(required_features) - set(values)
    if absent:
        raise HTTPException(422, f"Missing features: {', '.join(sorted(absent))}")
    cleaned: dict[str, float] = {}
    invalid: list[str] = []
    for f in required_features:
        val = values.get(f)
        try:
            num = float(val)
            if not math.isfinite(num):
                invalid.append(f)
            else:
                cleaned[f] = num
        except (TypeError, ValueError):
            invalid.append(f)
    if invalid:
        raise HTTPException(422, f"Invalid or non-finite numeric values for: {', '.join(sorted(invalid))}")
    return cleaned

@router.get("/health")
def health():
    return {"status": "ok", "models_ready": (ARTIFACTS / "classification.joblib").exists()}

@router.get("/overview")
def get_overview():
    return overview()

@router.get("/datasets")
def datasets():
    return {"items": [profile("classification"), profile("severity")]}

@router.get("/features/schema")
def get_feature_schema():
    return {"features": feature_schema()}

@router.get("/samples")
def get_samples():
    return {
        "samples": sample_patients(),
        "notice": "Sample records are drawn directly from parkinsons.data. Partition badges indicate whether a subject belonged to the held-out test set or the training set."
    }

@router.get("/models/metrics")
def metrics():
    p = ARTIFACTS / "metrics.json"
    if not p.exists():
        return {"trained": False, "message": "Train the models to generate real evaluation metrics."}
    import json
    return {"trained": True, **json.loads(p.read_text(encoding="utf-8"))}

@router.get("/records/{dataset_name}")
def records(dataset_name: str, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    try:
        df = dataset(dataset_name)
    except KeyError:
        raise HTTPException(404, "Dataset not found")
    return {
        "total": len(df),
        "items": df.iloc[offset:offset + limit].where(pd.notna(df.iloc[offset:offset + limit]), None).to_dict(orient="records")
    }

@router.post("/predict/classification")
def classify(body: Features):
    art = artifact("classification")
    cleaned = _validate_feature_values(art["features"], body.values)
    row = pd.DataFrame([cleaned])
    probs = art["model"].predict_proba(row)[0]
    prob_pd = float(probs[1])
    prob_healthy = float(probs[0])
    pred_class = int(prob_pd >= 0.5)
    return {
        "class": pred_class,
        "label": "Parkinson's group estimate" if pred_class == 1 else "Healthy control group estimate",
        "probability": prob_pd,
        "probabilities": {
            "parkinsons": prob_pd,
            "healthy": prob_healthy
        },
        "model": art["name"],
        "research_notice": "Research estimate only; not a diagnosis."
    }

@router.post("/predict/batch")
def predict_batch(body: BatchRequest):
    art = artifact("classification")
    req_features: list[str] = list(art["features"])

    raw_rows: list[dict[str, Any]] = []
    columns: list[str] = []

    if body.csv_text is not None:
        text = body.csv_text.strip()
        if not text:
            raise HTTPException(422, "Uploaded CSV content is empty.")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise HTTPException(422, "CSV header row is missing or unreadable.")
        columns = [c.strip() for c in reader.fieldnames if c is not None]
        for r in reader:
            cleaned_row = {k.strip(): v for k, v in r.items() if k is not None}
            raw_rows.append(cleaned_row)
    elif body.rows is not None:
        if not body.rows:
            raise HTTPException(422, "No rows provided for batch analysis.")
        col_set: set[str] = set()
        for r in body.rows:
            if isinstance(r, dict):
                col_set.update(str(k).strip() for k in r.keys())
                raw_rows.append({str(k).strip(): v for k, v in r.items()})
        columns = sorted(col_set)
    else:
        raise HTTPException(422, "Provide either 'csv_text' or 'rows' for batch prediction.")

    missing_cols = sorted(set(req_features) - set(columns))
    if missing_cols:
        raise HTTPException(
            422,
            f"Missing required acoustic feature columns ({len(missing_cols)}): {', '.join(missing_cols)}"
        )

    if not raw_rows:
        raise HTTPException(422, "CSV contains column headers but no data rows.")
    if len(raw_rows) > 2000:
        raise HTTPException(422, "Batch size exceeds maximum limit of 2,000 rows.")

    valid_indices: list[int] = []
    valid_feature_rows: list[dict[str, float]] = []
    row_outcomes: list[dict[str, Any]] = []

    for idx, r in enumerate(raw_rows):
        rec_id = str(r.get("name") or r.get("id") or r.get("record_id") or f"Row_{idx + 1}").strip()
        actual_status = None
        if "status" in r and r["status"] not in (None, ""):
            try:
                st_val = int(float(str(r["status"]).strip()))
                if st_val in (0, 1):
                    actual_status = st_val
            except (TypeError, ValueError):
                actual_status = None

        feat_dict: dict[str, float] = {}
        bad_fields: list[str] = []
        for f in req_features:
            raw_val = r.get(f)
            if raw_val is None or str(raw_val).strip() == "":
                bad_fields.append(f"{f} (empty)")
                continue
            try:
                num = float(str(raw_val).strip())
                if not math.isfinite(num):
                    bad_fields.append(f"{f} (non-finite)")
                else:
                    feat_dict[f] = num
            except (TypeError, ValueError):
                bad_fields.append(f"{f} ('{raw_val}')")

        if bad_fields:
            row_outcomes.append({
                "row_number": idx + 1,
                "record_id": rec_id,
                "valid": False,
                "error": f"Invalid numeric values in: {', '.join(bad_fields[:5])}" + (f" (+{len(bad_fields)-5} more)" if len(bad_fields) > 5 else ""),
                "actual_status": actual_status,
                "predicted_class": None,
                "predicted_label": None,
                "probability_parkinsons": None,
                "probability_healthy": None,
                "values": {}
            })
        else:
            valid_indices.append(idx)
            valid_feature_rows.append(feat_dict)
            row_outcomes.append({
                "row_number": idx + 1,
                "record_id": rec_id,
                "valid": True,
                "error": None,
                "actual_status": actual_status,
                "predicted_class": None,
                "predicted_label": None,
                "probability_parkinsons": None,
                "probability_healthy": None,
                "values": feat_dict
            })

    if valid_feature_rows:
        df_valid = pd.DataFrame(valid_feature_rows)[req_features]
        probs = art["model"].predict_proba(df_valid)
        for valid_pos, original_idx in enumerate(valid_indices):
            p_healthy = float(probs[valid_pos, 0])
            p_pd = float(probs[valid_pos, 1])
            pred_cls = int(p_pd >= 0.5)
            row_outcomes[original_idx]["predicted_class"] = pred_cls
            row_outcomes[original_idx]["predicted_label"] = (
                "Parkinson's group estimate" if pred_cls == 1 else "Healthy control group estimate"
            )
            row_outcomes[original_idx]["probability_parkinsons"] = p_pd
            row_outcomes[original_idx]["probability_healthy"] = p_healthy

    # Build downloadable CSV string
    out_buf = io.StringIO()
    csv_cols = [
        "row_number", "record_id", "valid", "predicted_class", "predicted_label",
        "probability_parkinsons", "probability_healthy", "actual_status", "error"
    ] + req_features
    writer = csv.DictWriter(out_buf, fieldnames=csv_cols)
    writer.writeheader()
    for item in row_outcomes:
        out_row = {
            "row_number": item["row_number"],
            "record_id": item["record_id"],
            "valid": item["valid"],
            "predicted_class": item["predicted_class"] if item["predicted_class"] is not None else "",
            "predicted_label": item["predicted_label"] or "",
            "probability_parkinsons": f"{item['probability_parkinsons']:.6f}" if item["probability_parkinsons"] is not None else "",
            "probability_healthy": f"{item['probability_healthy']:.6f}" if item["probability_healthy"] is not None else "",
            "actual_status": item["actual_status"] if item["actual_status"] is not None else "",
            "error": item["error"] or "",
        }
        for f in req_features:
            out_row[f] = item["values"].get(f, "")
        writer.writerow(out_row)

    pred_pd_count = sum(1 for x in row_outcomes if x["valid"] and x["predicted_class"] == 1)
    pred_hc_count = sum(1 for x in row_outcomes if x["valid"] and x["predicted_class"] == 0)

    return {
        "total_rows": len(row_outcomes),
        "valid_rows": len(valid_feature_rows),
        "invalid_rows": len(row_outcomes) - len(valid_feature_rows),
        "predicted_parkinsons": pred_pd_count,
        "predicted_healthy": pred_hc_count,
        "model": art["name"],
        "research_notice": "Research estimates only; not a clinical diagnosis.",
        "results": row_outcomes,
        "csv_download": out_buf.getvalue()
    }

@router.post("/predict/severity")
def severity(body: Features):
    out = {}
    for target in ("motor_UPDRS", "total_UPDRS"):
        art = artifact(target)
        cleaned = _validate_feature_values(art["features"], body.values)
        out[target] = {
            "estimate": float(art["model"].predict(pd.DataFrame([cleaned]))[0]),
            "model": art["name"]
        }
    return {"estimates": out, "research_notice": "Research estimate only; not a clinical assessment."}

@router.get("/progression/{patient_id}")
def progression(patient_id: int):
    df = dataset("severity")
    rows = df[df["subject#"] == patient_id].sort_values("test_time").copy()
    if rows.empty:
        raise HTTPException(404, "Patient identifier not found")
    min_t = float(rows["test_time"].min())
    max_t = float(rows["test_time"].max())
    rows["elapsed_days"] = (rows["test_time"] - min_t).round(4)
    record_cols = [
        "test_time", "elapsed_days", "motor_UPDRS", "total_UPDRS",
        "HNR", "Jitter(%)", "Shimmer", "RPDE", "DFA", "PPE"
    ]
    available_cols = [c for c in record_cols if c in rows.columns]
    first_row = rows.iloc[0]
    last_row = rows.iloc[-1]
    pre_baseline_count = int((rows["test_time"] < 0).sum())
    return {
        "patient_id": patient_id,
        "summary": {
            "patient_id": patient_id,
            "age": int(first_row["age"]),
            "sex": int(first_row["sex"]),
            "sex_label": "Female" if int(first_row["sex"]) == 1 else "Male",
            "observations": int(len(rows)),
            "test_time_min": min_t,
            "test_time_max": max_t,
            "elapsed_span_days": round(max_t - min_t, 2),
            "pre_baseline_records": pre_baseline_count,
            "motor_mean": float(rows["motor_UPDRS"].mean()),
            "motor_std": float(rows["motor_UPDRS"].std()),
            "motor_min": float(rows["motor_UPDRS"].min()),
            "motor_max": float(rows["motor_UPDRS"].max()),
            "motor_first": float(first_row["motor_UPDRS"]),
            "motor_last": float(last_row["motor_UPDRS"]),
            "motor_delta": float(last_row["motor_UPDRS"] - first_row["motor_UPDRS"]),
            "total_mean": float(rows["total_UPDRS"].mean()),
            "total_std": float(rows["total_UPDRS"].std()),
            "total_min": float(rows["total_UPDRS"].min()),
            "total_max": float(rows["total_UPDRS"].max()),
            "total_first": float(first_row["total_UPDRS"]),
            "total_last": float(last_row["total_UPDRS"]),
            "total_delta": float(last_row["total_UPDRS"] - first_row["total_UPDRS"]),
        },
        "records": rows[available_cols].to_dict(orient="records"),
        "notice": "Descriptive repeated measurements; not a forecast."
    }

@router.get("/progression-summary")
def progression_summary():
    df = dataset("severity").copy()
    df["period_days"] = (np.floor(df.test_time / 30) * 30).astype(int)
    grouped = (
        df.groupby("period_days")
        .agg(
            motor_UPDRS=("motor_UPDRS", "mean"),
            total_UPDRS=("total_UPDRS", "mean"),
            records=("subject#", "size"),
            participants=("subject#", "nunique"),
            min_day=("test_time", "min"),
            max_day=("test_time", "max"),
        )
        .reset_index()
    )
    period_records = []
    for r in grouped.to_dict(orient="records"):
        p = int(r["period_days"])
        label = f"Pre-baseline ({r['min_day']:.1f} to 0d)" if p < 0 else f"Days {p}–{p + 29}"
        period_records.append({**r, "period_label": label})

    # Also compute elapsed-time bins aligned to each subject's first recording (0 to 210 days)
    df["subject_start"] = df.groupby("subject#")["test_time"].transform("min")
    df["elapsed_days"] = df["test_time"] - df["subject_start"]
    df["elapsed_bin"] = (np.floor(df["elapsed_days"] / 30) * 30).astype(int)
    elapsed_grouped = (
        df.groupby("elapsed_bin")
        .agg(
            motor_UPDRS=("motor_UPDRS", "mean"),
            total_UPDRS=("total_UPDRS", "mean"),
            records=("subject#", "size"),
            participants=("subject#", "nunique"),
        )
        .reset_index()
        .rename(columns={"elapsed_bin": "period_days"})
    )
    elapsed_records = [
        {**r, "period_label": f"Elapsed Days {int(r['period_days'])}–{int(r['period_days']) + 29}"}
        for r in elapsed_grouped.to_dict(orient="records")
    ]

    # Per-participant summary across all 42 subjects
    df_sorted = df.sort_values(["subject#", "test_time"])
    subj_list = []
    for sid, group in df_sorted.groupby("subject#"):
        first_r = group.iloc[0]
        last_r = group.iloc[-1]
        t_min, t_max = float(group["test_time"].min()), float(group["test_time"].max())
        subj_list.append({
            "patient_id": int(sid),
            "age": int(first_r["age"]),
            "sex": int(first_r["sex"]),
            "sex_label": "Female" if int(first_r["sex"]) == 1 else "Male",
            "observations": int(len(group)),
            "test_time_min": t_min,
            "test_time_max": t_max,
            "elapsed_span_days": round(t_max - t_min, 2),
            "pre_baseline_records": int((group["test_time"] < 0).sum()),
            "motor_mean": float(group["motor_UPDRS"].mean()),
            "motor_min": float(group["motor_UPDRS"].min()),
            "motor_max": float(group["motor_UPDRS"].max()),
            "motor_delta": float(last_r["motor_UPDRS"] - first_r["motor_UPDRS"]),
            "total_mean": float(group["total_UPDRS"].mean()),
            "total_min": float(group["total_UPDRS"].min()),
            "total_max": float(group["total_UPDRS"].max()),
            "total_delta": float(last_r["total_UPDRS"] - first_r["total_UPDRS"]),
        })

    return {
        "period_days": period_records,
        "elapsed_period_days": elapsed_records,
        "participants": subj_list,
        "time_alignment_note": (
            "test_time is measured in days relative to each participant's formal baseline recruitment date (Day 0). "
            "12 observations across Subjects 34 (-3.31d) and 42 (-4.26d) were recorded 3–4 days before formal baseline enrollment, "
            "which places them in the <0 day pre-baseline bin when grouped by recruitment date. "
            "Aligning by each participant's first observation (Elapsed Days ≥ 0) spans 0 to 204.0 days."
        ),
        "notice": "Descriptive means in approximate 30-day study-time bins. Sampling density varies over time; this is not a forecast."
    }

@router.post("/explain")
def explain(body: Features):
    art = artifact("classification")
    cleaned = _validate_feature_values(art["features"], body.values)
    try:
        import shap
    except ImportError:
        raise HTTPException(503, "SHAP dependency is not installed")
    row = pd.DataFrame([cleaned])
    background = pd.DataFrame(art["background"])
    baseline = art.get("baseline", {})
    explainer = shap.Explainer(art["model"].predict_proba, background)
    explanation = explainer(row, max_evals=2 * len(art["features"]) + 1)
    vals = np.asarray(explanation.values)[0]
    if vals.ndim == 2:
        vals = vals[:, 1]
    order_all = np.argsort(np.abs(vals))[::-1]
    all_contribs = [
        {
            "feature": art["features"][i],
            "value": float(cleaned[art["features"][i]]),
            "baseline": float(baseline.get(art["features"][i], 0.0)),
            "contribution": float(vals[i]),
            "direction": "increases estimate" if vals[i] > 0 else "decreases estimate"
        }
        for i in order_all
    ]
    return {
        "method": "SHAP permutation explainer; signed contributions to Parkinson’s probability",
        "base_value": float(np.asarray(explanation.base_values).reshape(-1)[-1]),
        "causal_claim": False,
        "contributions": all_contribs[:10],
        "all_contributions": all_contribs,
    }

@router.get("/models/importance")
def importance():
    try:
        import shap
    except ImportError:
        raise HTTPException(503, "SHAP dependency is not installed")
    art = artifact("classification")
    df = dataset("classification")
    X = df[art["features"]].head(12)
    bg = pd.DataFrame(art["background"])
    explanation = shap.Explainer(art["model"].predict_proba, bg)(X, max_evals=2 * len(art["features"]) + 1)
    vals = np.asarray(explanation.values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.plots.beeswarm(explanation[:, :, 1], max_display=12, show=False)
    plt.gcf().savefig(ARTIFACTS / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(plt.gcf())
    scores = np.abs(vals).mean(axis=0)
    order = np.argsort(scores)[::-1][:12]
    return {
        "method": "Mean absolute SHAP value on 12 supplied records",
        "causal_claim": False,
        "features": [{"feature": art["features"][i], "importance": float(scores[i])} for i in order]
    }

@router.get("/models/shap-summary")
def shap_summary():
    path = ARTIFACTS / "shap_summary.png"
    if not path.exists():
        raise HTTPException(503, "Load global model importance to generate the SHAP summary plot")
    return FileResponse(path, media_type="image/png", filename="voxsense-shap-summary.png")

