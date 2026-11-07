from fastapi.testclient import TestClient
from app.main import app
from app.services.data import dataset
from app.services.training import grouped_holdout, severity_features
from app.core.config import ARTIFACTS
import pytest

client=TestClient(app)
def test_health():
    r=client.get('/api/health'); assert r.status_code==200 and r.json()['status']=='ok'
def test_overview_uses_real_dataset():
    r=client.get('/api/overview'); assert r.status_code==200
    body=r.json()
    assert body['classification_records']==195 and body['severity_subjects']==42
    assert body['severity_pre_baseline_records']==12
    assert body['severity_pre_baseline_subjects']==[34,42]
def test_dataset_schema_and_no_missing():
    d=client.get('/api/datasets').json()['items']
    assert d[0]['missing_cells']==0 and 'status' in [c['name'] for c in d[0]['column_info']]
def test_unknown_progression_subject():
    assert client.get('/api/progression/99999').status_code==404
def test_aggregate_progression_has_observed_bins():
    response=client.get('/api/progression-summary'); assert response.status_code==200
    body=response.json(); assert body['period_days'] and body['period_days'][0]['participants']>0
    assert body['elapsed_period_days'] and body['elapsed_period_days'][0]['period_days']==0
    assert len(body['participants'])==42
def test_grouped_holdout_keeps_people_separate():
    frame=dataset('classification'); X=frame.drop(columns=['name','status']); groups=frame.name.str.rsplit('_',n=1).str[0]
    train,test=grouped_holdout(X,frame.status,groups)
    assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))
def test_severity_predictors_exclude_leakage_fields():
    fields=severity_features(dataset('severity').columns)
    assert len(fields)==16 and not {'subject#','test_time','age','sex','motor_UPDRS','total_UPDRS'} & set(fields)
def test_classification_inference_when_trained():
    if not (ARTIFACTS/'classification.joblib').exists(): pytest.skip('Train models before inference test')
    row=dataset('classification').drop(columns=['name','status']).iloc[0].to_dict()
    response=client.post('/api/predict/classification',json={'values':row})
    assert response.status_code==200 and 0<=response.json()['probability']<=1
    assert 'probabilities' in response.json()
    assert abs(response.json()['probabilities']['parkinsons']+response.json()['probabilities']['healthy']-1.0)<1e-6
    assert 'not a diagnosis' in response.json()['research_notice'].lower()
def test_manual_classification_rejects_missing_features():
    if not (ARTIFACTS/'classification.joblib').exists(): pytest.skip('Train models before inference test')
    response=client.post('/api/predict/classification',json={'values':{'MDVP:Fo(Hz)':150.0}})
    assert response.status_code==422
    assert 'Missing features' in response.json()['detail']
def test_feature_schema_has_22_documented_features():
    r=client.get('/api/features/schema'); assert r.status_code==200
    feats=r.json()['features']
    assert len(feats)==22
    assert all('label' in f and 'unit' in f and 'group' in f and 'min' in f and 'max' in f for f in feats)
def test_samples_distinguish_actual_labels_and_partitions():
    r=client.get('/api/samples'); assert r.status_code==200
    samples=r.json()['samples']
    assert len(samples)>=5
    for s in samples:
        assert s['partition'] in ('train','test')
        assert s['actual_status'] in (0,1)
        assert len(s['values'])==22
def test_batch_csv_prediction_and_invalid_row_handling():
    if not (ARTIFACTS/'classification.joblib').exists(): pytest.skip('Train models before batch test')
    df=dataset('classification').drop(columns=['status']).head(3).copy()
    csv_lines=df.to_csv(index=False).strip().splitlines()
    # Add an invalid 4th row with non-numeric text in first feature
    bad_row=csv_lines[1].split(',')
    bad_row[0]='corrupted_row'
    bad_row[1]='NOT_A_NUMBER'
    csv_text='\n'.join(csv_lines+[','.join(bad_row)])
    r=client.post('/api/predict/batch',json={'csv_text':csv_text})
    assert r.status_code==200
    body=r.json()
    assert body['total_rows']==4
    assert body['valid_rows']==3
    assert body['invalid_rows']==1
    assert body['results'][0]['valid'] is True
    assert 0.0<=body['results'][0]['probability_parkinsons']<=1.0
    assert body['results'][3]['valid'] is False
    assert 'MDVP:Fo(Hz)' in body['results'][3]['error']
    assert 'probability_parkinsons' in body['csv_download']
def test_batch_csv_missing_column_rejected():
    if not (ARTIFACTS/'classification.joblib').exists(): pytest.skip('Train models before batch test')
    r=client.post('/api/predict/batch',json={'csv_text':'MDVP:Fo(Hz),MDVP:Fhi(Hz)\n150.0,180.0'})
    assert r.status_code==422
    assert 'Missing required acoustic feature columns' in r.json()['detail']
def test_severity_inference_uses_voice_features_when_trained():
    if not (ARTIFACTS/'motor_UPDRS.joblib').exists(): pytest.skip('Train models before inference test')
    row=dataset('severity').iloc[0]; values={field:float(row[field]) for field in severity_features(dataset('severity').columns)}
    response=client.post('/api/predict/severity',json={'values':values})
    assert response.status_code==200
    assert set(response.json()['estimates'])=={'motor_UPDRS','total_UPDRS'}
def test_trained_metrics_include_roc_curve_when_available():
    if not (ARTIFACTS/'metrics.json').exists(): pytest.skip('Train models before metrics test')
    response=client.get('/api/models/metrics'); assert response.status_code==200
    body=response.json(); selected=body['classification_selected']
    assert body['classification'][selected]['roc_curve']
    assert len(body['classification'][selected]['confusion_matrix'])==2
def test_shap_summary_image_when_generated():
    if not (ARTIFACTS/'shap_summary.png').exists(): pytest.skip('Load model importance to generate plot')
    response=client.get('/api/models/shap-summary')
    assert response.status_code==200 and response.headers['content-type']=='image/png'

