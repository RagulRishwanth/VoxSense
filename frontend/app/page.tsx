'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertCircle,
  ArrowRight,
  AudioLines,
  BookOpen,
  CheckCircle2,
  Database,
  Download,
  FileSpreadsheet,
  HelpCircle,
  History,
  Info,
  LayoutDashboard,
  Mic2,
  Play,
  RotateCcw,
  ShieldAlert,
  Sliders,
  Sparkles,
  Target,
  Trash2,
  Trophy,
  UploadCloud,
  UserCheck,
  Zap,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const API = (process.env.NEXT_PUBLIC_API_URL || '/api').replace(/\/+$/, '');
const HISTORY_STORAGE_KEY = 'voxsense_recent_analyses_v1';

type NavSection = 'Overview' | 'Analyze Voice Biomarkers' | 'Patient Insights' | 'About VoxSense';
type AnalyzeTab = 'manual' | 'sample' | 'batch';

type FeatureMeta = {
  name: string;
  label: string;
  unit: string;
  group: string;
  description: string;
  min: number;
  max: number;
  median: number;
  mean: number;
  std: number;
  q1: number;
  q3: number;
  healthy_median: number;
  parkinsons_median: number;
  healthy_mean: number;
  parkinsons_mean: number;
  valid_min: number;
  valid_max: number;
};

type SampleRecord = {
  id: string;
  subject_group: string;
  actual_status: number;
  actual_label: string;
  partition: 'train' | 'test';
  partition_label: string;
  is_independent_test: boolean;
  partition_notice: string;
  description: string;
  values: Record<string, number>;
};

type OverviewData = {
  classification_records: number;
  healthy_records: number;
  parkinsons_records: number;
  classification_subjects: number;
  severity_records: number;
  severity_subjects: number;
  severity_days: [number, number];
  severity_post_recruitment_days: [number, number];
  severity_pre_baseline_records: number;
  severity_pre_baseline_subjects: number[];
  severity_pre_baseline_range: [number, number];
  severity_subject_span_days: [number, number];
  severity_time_explanation: string;
  motor_mean: number;
  motor_std: number;
  motor_min: number;
  motor_max: number;
  total_mean: number;
  total_std: number;
  total_min: number;
  total_max: number;
  classification_features: string[];
  severity_features: string[];
  key_biomarkers: {
    feature: string;
    label: string;
    unit: string;
    healthy_mean: number;
    healthy_median: number;
    parkinsons_mean: number;
    parkinsons_median: number;
  }[];
};

type PredictionResult = {
  class: number;
  label: string;
  probability: number;
  probabilities: {
    parkinsons: number;
    healthy: number;
  };
  model: string;
  research_notice: string;
};

type ShapContribution = {
  feature: string;
  value?: number;
  baseline?: number;
  contribution: number;
  direction: string;
};

type ExplanationResult = {
  method: string;
  base_value: number;
  causal_claim: boolean;
  contributions: ShapContribution[];
  all_contributions?: ShapContribution[];
};

type HistoryItem = {
  id: string;
  timestamp: string;
  source: string;
  label: string;
  probability: number;
  model: string;
  topFeatures: string[];
  values: Record<string, number>;
};

const NAV_ITEMS: { name: NavSection; icon: any }[] = [
  { name: 'Overview', icon: LayoutDashboard },
  { name: 'Analyze Voice Biomarkers', icon: Sliders },
  { name: 'Patient Insights', icon: Activity },
  { name: 'About VoxSense', icon: BookOpen },
];

function formatNumber(n: number | undefined | null, digits = 4): string {
  if (n === undefined || n === null || !Number.isFinite(n)) return '—';
  if (Math.abs(n) > 0 && Math.abs(n) < 0.0005) return n.toExponential(2);
  if (Math.abs(n) >= 100) return n.toFixed(2);
  if (Math.abs(n) >= 10) return n.toFixed(3);
  return Number(n.toFixed(digits)).toString();
}

export default function Home() {
  const [page, setPage] = useState<NavSection>('Overview');
  const [analyzeTab, setAnalyzeTab] = useState<AnalyzeTab>('manual');

  // Backend state
  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [featureSchema, setFeatureSchema] = useState<FeatureMeta[]>([]);
  const [samples, setSamples] = useState<SampleRecord[]>([]);
  const [metrics, setMetrics] = useState<Record<string, any> | null>(null);
  const [importance, setImportance] = useState<any>(null);
  const [rocModel, setRocModel] = useState<string>('');

  // Manual analysis state
  const [manualInputStrings, setManualInputStrings] = useState<Record<string, string>>({});
  const [showTooltips, setShowTooltips] = useState<boolean>(false);
  const [manualPrediction, setManualPrediction] = useState<PredictionResult | null>(null);
  const [manualExplanation, setManualExplanation] = useState<ExplanationResult | null>(null);
  const [manualBusy, setManualBusy] = useState<boolean>(false);
  const [manualError, setManualError] = useState<string>('');

  // Sample patient analysis state
  const [selectedSampleId, setSelectedSampleId] = useState<string>('');
  const [samplePrediction, setSamplePrediction] = useState<PredictionResult | null>(null);
  const [sampleExplanation, setSampleExplanation] = useState<ExplanationResult | null>(null);
  const [sampleBusy, setSampleBusy] = useState<boolean>(false);
  const [sampleError, setSampleError] = useState<string>('');

  // CSV batch analysis state
  const [csvFileName, setCsvFileName] = useState<string>('');
  const [csvRawText, setCsvRawText] = useState<string>('');
  const [csvPreviewHeaders, setCsvPreviewHeaders] = useState<string[]>([]);
  const [csvPreviewRows, setCsvPreviewRows] = useState<Record<string, string>[]>([]);
  const [csvMissingCols, setCsvMissingCols] = useState<string[]>([]);
  const [batchResult, setBatchResult] = useState<any>(null);
  const [batchBusy, setBatchBusy] = useState<boolean>(false);
  const [batchError, setBatchError] = useState<string>('');
  const [dragActive, setDragActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Longitudinal UPDRS state
  const [progressionSummary, setProgressionSummary] = useState<any>(null);
  const [primaryPatientId, setPrimaryPatientId] = useState<number>(1);
  const [comparePatientId, setComparePatientId] = useState<string>('none');
  const [primaryTrajectory, setPrimaryTrajectory] = useState<any>(null);
  const [compareTrajectory, setCompareTrajectory] = useState<any>(null);
  const [timeAxisMode, setTimeAxisMode] = useState<'test_time' | 'elapsed_days'>('test_time');
  const [selectedVoiceMetric, setSelectedVoiceMetric] = useState<string>('HNR');
  const [participantFilter, setParticipantFilter] = useState<string>('');

  // Local storage history
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [globalError, setGlobalError] = useState<string>('');

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(HISTORY_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) setHistory(parsed);
      }
    } catch {
      // ignore
    }
  }, []);

  function saveToHistory(entry: Omit<HistoryItem, 'id' | 'timestamp'>) {
    try {
      const newItem: HistoryItem = {
        ...entry,
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      const updated = [newItem, ...history].slice(0, 8);
      setHistory(updated);
      window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(updated));
    } catch {
      // ignore
    }
  }

  function clearHistory() {
    setHistory([]);
    try {
      window.localStorage.removeItem(HISTORY_STORAGE_KEY);
    } catch {
      // ignore
    }
  }

  // Load backend data
  useEffect(() => {
    async function initLoad() {
      try {
        const [ovRes, featRes, sampRes, metRes, progSumRes, p1Res] = await Promise.all([
          fetch(`${API}/overview`).then((r) => r.json()),
          fetch(`${API}/features/schema`).then((r) => r.json()),
          fetch(`${API}/samples`).then((r) => r.json()),
          fetch(`${API}/models/metrics`).then((r) => r.json()),
          fetch(`${API}/progression-summary`).then((r) => r.json()),
          fetch(`${API}/progression/1`).then((r) => r.json()),
        ]);

        setOverview(ovRes);
        const schemaList: FeatureMeta[] = featRes.features || [];
        setFeatureSchema(schemaList);

        const initialStrings: Record<string, string> = {};
        for (const f of schemaList) {
          initialStrings[f.name] = String(f.median);
        }
        setManualInputStrings(initialStrings);

        const sampList: SampleRecord[] = sampRes.samples || [];
        setSamples(sampList);
        if (sampList.length > 0) {
          setSelectedSampleId(sampList[0].id);
        }

        setMetrics(metRes);
        if (metRes?.classification_selected) {
          setRocModel(metRes.classification_selected);
        }

        setProgressionSummary(progSumRes);
        setPrimaryTrajectory(p1Res);
        setGlobalError('');
      } catch {
        setGlobalError('Backend is offline. Ensure the VoxSense FastAPI backend is running and reachable.');
      }
    }
    initLoad();
  }, []);

  // Fetch SHAP importance for About page
  useEffect(() => {
    if ((page === 'About VoxSense' || page === 'Overview') && metrics?.trained && !importance) {
      fetch(`${API}/models/importance`)
        .then((r) => r.json())
        .then(setImportance)
        .catch(() => {});
    }
  }, [page, metrics, importance]);

  // Load participant trajectory
  async function loadPrimaryParticipant(pid: number) {
    setPrimaryPatientId(pid);
    try {
      const r = await fetch(`${API}/progression/${pid}`);
      const j = await r.json();
      if (r.ok) setPrimaryTrajectory(j);
    } catch {
      // ignore
    }
  }

  async function handleCompareParticipantChange(val: string) {
    setComparePatientId(val);
    if (val === 'none') {
      setCompareTrajectory(null);
      return;
    }
    try {
      const r = await fetch(`${API}/progression/${Number(val)}`);
      const j = await r.json();
      if (r.ok) setCompareTrajectory(j);
    } catch {
      setCompareTrajectory(null);
    }
  }

  const groupedFeatures = useMemo(() => {
    const groups: Record<string, FeatureMeta[]> = {};
    for (const f of featureSchema) {
      if (!groups[f.group]) groups[f.group] = [];
      groups[f.group].push(f);
    }
    return Object.entries(groups);
  }, [featureSchema]);

  const manualValidation = useMemo(() => {
    const errors: Record<string, string> = {};
    const warnings: Record<string, string> = {};
    const numericValues: Record<string, number> = {};

    for (const f of featureSchema) {
      const raw = manualInputStrings[f.name];
      if (raw === undefined || raw.trim() === '') {
        errors[f.name] = 'Required';
        continue;
      }
      const num = Number(raw);
      if (!Number.isFinite(num)) {
        errors[f.name] = 'Invalid number';
        continue;
      }
      if (num < f.valid_min || num > f.valid_max) {
        errors[f.name] = `Out of bounds`;
        continue;
      }
      if (num < f.min || num > f.max) {
        warnings[f.name] = `Outside range (${formatNumber(f.min, 3)}–${formatNumber(f.max, 3)})`;
      }
      numericValues[f.name] = num;
    }
    return {
      errors,
      warnings,
      numericValues,
      isValid: Object.keys(errors).length === 0 && featureSchema.length === 22,
    };
  }, [featureSchema, manualInputStrings]);

  function applyManualPreset(mode: 'median' | 'healthy' | 'parkinsons') {
    const next: Record<string, string> = {};
    for (const f of featureSchema) {
      if (mode === 'median') next[f.name] = String(f.median);
      else if (mode === 'healthy') next[f.name] = String(f.healthy_median);
      else next[f.name] = String(f.parkinsons_median);
    }
    setManualInputStrings(next);
    setManualPrediction(null);
    setManualExplanation(null);
    setManualError('');
  }

  async function runManualAnalysis() {
    if (!manualValidation.isValid) {
      setManualError('Please fill all 22 required features with valid numeric values.');
      return;
    }
    setManualBusy(true);
    setManualError('');
    try {
      const [predRes, expRes] = await Promise.all([
        fetch(`${API}/predict/classification`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ values: manualValidation.numericValues }),
        }),
        fetch(`${API}/explain`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ values: manualValidation.numericValues }),
        }),
      ]);
      const predJson = await predRes.json();
      const expJson = await expRes.json();
      if (!predRes.ok) throw new Error(predJson.detail || 'Prediction failed');
      if (!expRes.ok) throw new Error(expJson.detail || 'Explanation failed');

      setManualPrediction(predJson);
      setManualExplanation(expJson);
      saveToHistory({
        source: 'Manual Analysis',
        label: predJson.label,
        probability: predJson.probability,
        model: predJson.model,
        topFeatures: (expJson.contributions || []).slice(0, 3).map((c: any) => c.feature),
        values: manualValidation.numericValues,
      });
    } catch (e) {
      setManualError(e instanceof Error ? e.message : 'Analysis failed');
    } finally {
      setManualBusy(false);
    }
  }

  const selectedSample = useMemo(
    () => samples.find((s) => s.id === selectedSampleId) || samples[0] || null,
    [samples, selectedSampleId]
  );

  async function runSampleAnalysis(sample: SampleRecord) {
    setSampleBusy(true);
    setSampleError('');
    setSamplePrediction(null);
    setSampleExplanation(null);
    try {
      const [predRes, expRes] = await Promise.all([
        fetch(`${API}/predict/classification`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ values: sample.values }),
        }),
        fetch(`${API}/explain`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ values: sample.values }),
        }),
      ]);
      const predJson = await predRes.json();
      const expJson = await expRes.json();
      if (!predRes.ok) throw new Error(predJson.detail || 'Prediction failed');
      if (!expRes.ok) throw new Error(expJson.detail || 'Explanation failed');

      setSamplePrediction(predJson);
      setSampleExplanation(expJson);
      saveToHistory({
        source: `Sample ${sample.id}`,
        label: predJson.label,
        probability: predJson.probability,
        model: predJson.model,
        topFeatures: (expJson.contributions || []).slice(0, 3).map((c: any) => c.feature),
        values: sample.values,
      });
    } catch (e) {
      setSampleError(e instanceof Error ? e.message : 'Sample analysis failed');
    } finally {
      setSampleBusy(false);
    }
  }

  function parseAndPreviewCsv(text: string, filename: string) {
    setCsvFileName(filename);
    setCsvRawText(text);
    setBatchResult(null);
    setBatchError('');

    const lines = text
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    if (lines.length === 0) {
      setCsvPreviewHeaders([]);
      setCsvPreviewRows([]);
      setCsvMissingCols([]);
      setBatchError('CSV file is empty.');
      return;
    }

    const headers = lines[0].split(',').map((h) => h.trim());
    setCsvPreviewHeaders(headers);

    const requiredNames = featureSchema.map((f) => f.name);
    const missing = requiredNames.filter((req) => !headers.includes(req));
    setCsvMissingCols(missing);

    const parsedRows: Record<string, string>[] = [];
    for (let i = 1; i < lines.length; i++) {
      const cells = lines[i].split(',');
      const rowObj: Record<string, string> = {};
      headers.forEach((h, idx) => {
        rowObj[h] = cells[idx] !== undefined ? cells[idx].trim() : '';
      });
      parsedRows.push(rowObj);
    }
    setCsvPreviewRows(parsedRows);

    if (missing.length > 0) {
      setBatchError(`Missing ${missing.length} required columns: ${missing.join(', ')}`);
    } else if (parsedRows.length === 0) {
      setBatchError('CSV has headers but no rows.');
    }
  }

  function handleFileSelect(file: File | undefined) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv') && file.type !== 'text/csv') {
      setBatchError('Please select a valid .csv file.');
      return;
    }
    const MAX_CLIENT_CSV_BYTES = 4 * 1024 * 1024; // 4MB
    if (file.size > MAX_CLIENT_CSV_BYTES) {
      setBatchError(`File size (${(file.size / (1024 * 1024)).toFixed(2)} MB) exceeds the maximum upload limit of 4 MB.`);
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      const content = String(ev.target?.result || '');
      parseAndPreviewCsv(content, file.name);
    };
    reader.readAsText(file);
  }

  function loadDemoBatchCsv() {
    if (!samples.length || !featureSchema.length) return;
    const cols = ['name', ...featureSchema.map((f) => f.name)];
    const rows = samples.slice(0, 6).map((s) => {
      return [s.id, ...featureSchema.map((f) => String(s.values[f.name]))].join(',');
    });
    const csvContent = [cols.join(','), ...rows].join('\n');
    parseAndPreviewCsv(csvContent, 'voxsense_demo_batch.csv');
  }

  function downloadTemplateCsv() {
    if (!featureSchema.length) return;
    const cols = ['record_id', ...featureSchema.map((f) => f.name)];
    const medianRow = ['patient_01', ...featureSchema.map((f) => String(f.median))];
    const healthyRow = ['patient_02', ...featureSchema.map((f) => String(f.healthy_median))];
    const csvContent = [cols.join(','), medianRow.join(','), healthyRow.join(',')].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'voxsense_template.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  async function runBatchPrediction() {
    if (!csvRawText.trim()) return;
    setBatchBusy(true);
    setBatchError('');
    try {
      const res = await fetch(`${API}/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ csv_text: csvRawText }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Batch analysis failed');
      setBatchResult(data);
    } catch (e) {
      setBatchError(e instanceof Error ? e.message : 'Batch prediction failed');
    } finally {
      setBatchBusy(false);
    }
  }

  function downloadBatchResultsCsv() {
    if (!batchResult?.csv_download) return;
    const blob = new Blob([batchResult.csv_download], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `voxsense_results_${csvFileName || 'batch.csv'}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const comparisonChartData = useMemo(() => {
    if (!primaryTrajectory?.records) return [];
    const primaryRecs = primaryTrajectory.records;
    if (!compareTrajectory?.records) {
      return primaryRecs.map((r: any) => ({
        time: Number(r[timeAxisMode]),
        primary_motor: Number(r.motor_UPDRS),
        primary_total: Number(r.total_UPDRS),
        voice_metric: Number(r[selectedVoiceMetric]),
      }));
    }
    const bins = new Map<number, any>();
    for (const r of primaryRecs) {
      const b = Math.round(Number(r[timeAxisMode]) / 5) * 5;
      const cur = bins.get(b) || { time: b };
      cur.primary_motor = Number(r.motor_UPDRS);
      cur.primary_total = Number(r.total_UPDRS);
      cur.voice_metric = Number(r[selectedVoiceMetric]);
      bins.set(b, cur);
    }
    for (const r of compareTrajectory.records) {
      const b = Math.round(Number(r[timeAxisMode]) / 5) * 5;
      const cur = bins.get(b) || { time: b };
      cur.compare_motor = Number(r.motor_UPDRS);
      cur.compare_total = Number(r.total_UPDRS);
      cur.compare_voice = Number(r[selectedVoiceMetric]);
      bins.set(b, cur);
    }
    return Array.from(bins.values()).sort((a, b) => a.time - b.time);
  }, [primaryTrajectory, compareTrajectory, timeAxisMode, selectedVoiceMetric]);

  const filteredParticipants = useMemo(() => {
    const list = progressionSummary?.participants || [];
    if (!participantFilter.trim()) return list;
    const q = participantFilter.trim().toLowerCase();
    return list.filter(
      (p: any) =>
        String(p.patient_id).includes(q) ||
        p.sex_label.toLowerCase().includes(q) ||
        String(p.age).includes(q)
    );
  }, [progressionSummary, participantFilter]);

  function renderPredictionAndShap(
    pred: PredictionResult | null,
    exp: ExplanationResult | null,
    contextSample?: SampleRecord | null
  ) {
    if (!pred) return null;
    const isPD = pred.class === 1;
    const probPD = (pred.probabilities?.parkinsons ?? pred.probability) * 100;
    const probHC = (pred.probabilities?.healthy ?? 1 - pred.probability) * 100;
    const topContribs = exp?.contributions || [];
    const maxAbsShap = Math.max(...topContribs.map((c) => Math.abs(c.contribution)), 0.01);

    return (
      <div style={{ marginTop: 20 }}>
        <div className={`result-banner ${isPD ? '' : 'healthy'}`}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
            <div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
                <span className={`badge ${isPD ? 'amber' : 'blue'}`}>
                  Prediction ({pred.model})
                </span>
                {contextSample && (
                  <>
                    <span className={`badge ${contextSample.actual_status === 1 ? 'rose' : 'teal'}`}>
                      Ground Truth: {contextSample.actual_label}
                    </span>
                    <span className={`badge ${contextSample.is_independent_test ? 'teal' : 'amber'}`}>
                      {contextSample.partition_label}
                    </span>
                  </>
                )}
              </div>
              <h3 style={{ fontSize: 22, fontWeight: 800, margin: '4px 0', color: isPD ? '#92400e' : '#1e40af' }}>
                {pred.label}
              </h3>
              <p style={{ color: '#475569', fontSize: 13.5 }}>
                Estimated likelihood: <b>{probPD.toFixed(1)}%</b> Parkinson&apos;s acoustic profile vs <b>{probHC.toFixed(1)}%</b> Healthy Control profile.
              </p>
            </div>
            <div style={{ minWidth: 260, background: '#ffffff', padding: '14px 18px', borderRadius: 12, border: '1px solid var(--line)', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
              <div className="bar-head">
                <span style={{ fontWeight: 600 }}>Parkinson&apos;s Probability</span>
                <b>{probPD.toFixed(1)}%</b>
              </div>
              <div className="track" style={{ marginBottom: 12 }}>
                <div className="fill" style={{ width: `${probPD}%` }} />
              </div>
              <div className="bar-head">
                <span style={{ fontWeight: 600 }}>Healthy Control</span>
                <b>{probHC.toFixed(1)}%</b>
              </div>
              <div className="track">
                <div className="fill blue" style={{ width: `${probHC}%` }} />
              </div>
            </div>
          </div>

          <div className="notice info" style={{ marginTop: 14, marginBottom: 0 }}>
            <ShieldAlert size={16} style={{ flexShrink: 0, marginTop: 2 }} />
            <div>
              <b>Research Disclaimer:</b> {pred.research_notice} This is a statistical estimate based on acoustic features and is not a clinical diagnosis.
            </div>
          </div>
        </div>

        {exp && topContribs.length > 0 && (
          <div className="card" style={{ marginTop: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
              <div>
                <h2>Why Did the Model Make This Decision? (SHAP Explainability)</h2>
                <p>
                  Shows which acoustic features pushed the probability up (towards Parkinson&apos;s) or down (towards Healthy).
                </p>
              </div>
              <span className="badge teal">Top 10 Influential Features</span>
            </div>

            <div style={{ marginTop: 14 }}>
              <div className="shap-row" style={{ fontWeight: 700, color: '#475569', borderBottom: '2px solid var(--line)' }}>
                <span>Acoustic Feature</span>
                <span>Value vs Median</span>
                <span>Impact on Probability</span>
                <span style={{ textAlign: 'right' }}>Contribution</span>
              </div>
              {topContribs.map((c) => {
                const pctWidth = Math.min(100, (Math.abs(c.contribution) / maxAbsShap) * 100);
                const isPos = c.contribution > 0;
                return (
                  <div className="shap-row" key={c.feature}>
                    <div>
                      <div style={{ fontWeight: 700, fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>{c.feature}</div>
                      <div className="small">{c.direction}</div>
                    </div>
                    <div className="small">
                      <b>{formatNumber(c.value, 4)}</b>
                      <br />
                      <span>Med: {formatNumber(c.baseline, 4)}</span>
                    </div>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div className="track" style={{ flex: 1, height: 8 }}>
                          <div
                            className={`fill ${isPos ? 'amber' : 'blue'}`}
                            style={{ width: `${pctWidth}%` }}
                          />
                        </div>
                        <span className="small" style={{ minWidth: 120 }}>
                          {isPos ? '↑ Increases PD' : '↓ Decreases PD'}
                        </span>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right', fontWeight: 750 }} className={isPos ? 'positive' : 'negative'}>
                      {isPos ? '+' : ''}
                      {(c.contribution * 100).toFixed(2)}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="layout">
      {/* 4-SECTION SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <AudioLines size={22} />
          </div>
          <div>
            <div className="brand-title">VoxSense</div>
            <div className="brand-sub">Parkinson&apos;s Voice AI</div>
          </div>
        </div>

        <div>
          <div className="nav-label">Navigation</div>
          <nav className="nav" aria-label="Sidebar Navigation">
            {NAV_ITEMS.map(({ name, icon: Icon }) => (
              <button
                key={name}
                className={page === name ? 'active' : ''}
                onClick={() => setPage(name)}
              >
                <Icon size={18} />
                <span>{name}</span>
              </button>
            ))}
          </nav>
        </div>

        {history.length > 0 && (
          <div className="side-history">
            <div className="side-history-head">
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                <History size={13} /> Recent Runs ({history.length})
              </span>
              <button
                onClick={clearHistory}
                title="Clear local history"
                style={{ border: 0, background: 'transparent', color: '#94a3b8', cursor: 'pointer', padding: 2 }}
              >
                <Trash2 size={13} />
              </button>
            </div>
            <div className="side-history-list">
              {history.map((h) => (
                <button
                  key={h.id}
                  className="side-history-item"
                  onClick={() => {
                    const nextStrings: Record<string, string> = {};
                    for (const [k, v] of Object.entries(h.values)) nextStrings[k] = String(v);
                    setManualInputStrings(nextStrings);
                    setPage('Analyze Voice Biomarkers');
                    setAnalyzeTab('manual');
                  }}
                >
                  <div style={{ fontWeight: 650, fontSize: 11.5, color: 'var(--ink)' }}>{h.source}</div>
                  <div className="small" style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
                    <span>{(h.probability * 100).toFixed(1)}% PD</span>
                    <span>{h.timestamp}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="side-note">
          <ShieldAlert size={15} style={{ marginBottom: 6, color: '#d97706' }} />
          <div>
            <b>Research &amp; Education Only</b>
            <br />
            Analyzes pre-extracted numerical measurements. Never use for clinical medical decisions.
          </div>
        </div>
      </aside>

      {/* MAIN VIEW */}
      <main className="main">
        <header className="top">
          <div>
            <div className="eyebrow">VOXSENSE AI RESEARCH SUITE</div>
            <h1 className="title">{page}</h1>
            <p className="subtitle">
              {page === 'Overview' && 'A simple guide to what Parkinson’s is, how vocal biomarkers were analyzed, what the AI models discovered, and how to use the interactive tools.'}
              {page === 'Analyze Voice Biomarkers' && 'Enter 22 vocal acoustic features, test real patient recordings, or upload a CSV to get instant predictions and SHAP explainability.'}
              {page === 'Patient Insights' && 'Explore descriptive symptom scores (UPDRS) and voice stability over time across 42 patients in a 6-month trial.'}
              {page === 'About VoxSense' && 'Full dataset citations, 22-feature dictionary, cross-validation metrics, ROC curves, and research limits.'}
            </p>
          </div>
          <div className="top-badges">
            <span className="pill">
              <span style={{ color: overview ? '#0d9488' : '#d97706' }}>●</span>
              {overview ? 'Models Ready' : 'Connecting...'}
            </span>
            {metrics?.classification_selected && (
              <span className="pill">Active Model: {metrics.classification_selected}</span>
            )}
          </div>
        </header>

        {globalError && (
          <div className="notice danger">
            <AlertCircle size={16} />
            <div>{globalError}</div>
          </div>
        )}

        {/* =========================================================
            SECTION 1: OVERVIEW (SUPER SIMPLE, VISUAL & ENGAGING)
           ========================================================= */}
        {page === 'Overview' && (
          <>
            {/* HERO BANNER */}
            <div className="hero-banner">
              <div>
                <h2>Can Subtle Voice Changes Help Detect Parkinson&apos;s?</h2>
                <p>
                  VoxSense is an interactive research tool that analyzes 22 numerical vocal sound measurements (like micro-tremors, pitch instability, and breathiness) using machine learning and explainable AI.
                </p>
              </div>
              <div className="hero-actions">
                <button
                  className="btn-cta"
                  onClick={() => {
                    setPage('Analyze Voice Biomarkers');
                    setAnalyzeTab('manual');
                  }}
                >
                  Try VoxSense Now <ArrowRight size={17} />
                </button>
                <button
                  className="btn-hero-secondary"
                  onClick={() => {
                    setPage('Analyze Voice Biomarkers');
                    setAnalyzeTab('sample');
                  }}
                >
                  <UserCheck size={16} /> Test Patient Samples
                </button>
              </div>
            </div>

            {/* 4 CORE QUESTIONS: SIMPLE STORY FLOW */}
            <div className="grid two-equal" style={{ gap: 20, marginBottom: 24 }}>
              {/* Card 1: What is Parkinson's? */}
              <div className="card">
                <div style={{ display: 'flex', gap: 14, alignItems: 'center', marginBottom: 12 }}>
                  <div style={{ width: 42, height: 42, borderRadius: 12, background: '#fef3c7', color: '#b45309', display: 'grid', placeItems: 'center', flexShrink: 0 }}>
                    <Activity size={22} />
                  </div>
                  <div>
                    <h2 style={{ fontSize: 17, margin: 0 }}>1. What is Parkinson&apos;s Disease?</h2>
                    <span className="small">Background &amp; Vocal Manifestation</span>
                  </div>
                </div>
                <p style={{ fontSize: 13.5, lineHeight: 1.6, color: '#334155' }}>
                  Parkinson&apos;s disease is a neurodegenerative disorder that primarily affects dopamine-producing neurons in the brain, leading to tremors, muscle stiffness, and slowed movement.
                </p>
                <div style={{ marginTop: 12, background: '#f8fafc', padding: '12px 14px', borderRadius: 10, border: '1px solid #e2e8f0', fontSize: 13, color: '#475569' }}>
                  💡 <b>Why Voice?</b> Up to <b>90%</b> of people with Parkinson&apos;s experience <i>dysphonia</i> (voice impairment) early in the condition. Vocal cords lose flexibility, producing quiet, breathy speech, frequency jitter, and micro-tremors that human ears often miss.
                </div>
              </div>

              {/* Card 2: What was the Objective? */}
              <div className="card">
                <div style={{ display: 'flex', gap: 14, alignItems: 'center', marginBottom: 12 }}>
                  <div style={{ width: 42, height: 42, borderRadius: 12, background: '#e0f2fe', color: '#0369a1', display: 'grid', placeItems: 'center', flexShrink: 0 }}>
                    <Target size={22} />
                  </div>
                  <div>
                    <h2 style={{ fontSize: 17, margin: 0 }}>2. What was the Research Objective?</h2>
                    <span className="small">Goals &amp; Feasibility</span>
                  </div>
                </div>
                <p style={{ fontSize: 13.5, lineHeight: 1.6, color: '#334155' }}>
                  The research conducted by Oxford University evaluated whether sustained vowel phonations (saying &quot;aaah&quot; into a microphone) could be converted into numerical acoustic features to:
                </p>
                <ul style={{ paddingLeft: 18, margin: '8px 0 0', fontSize: 13, color: '#475569', lineHeight: 1.6 }}>
                  <li><b>Distinguish Parkinson&apos;s from healthy controls</b> using non-invasive voice measurements.</li>
                  <li><b>Track symptom progression (UPDRS scores)</b> over 6 months from home telemonitoring devices.</li>
                  <li>Provide transparent <b>explainable AI (SHAP)</b> rather than a black-box model.</li>
                </ul>
              </div>

              {/* Card 3: How was it Done? */}
              <div className="card">
                <div style={{ display: 'flex', gap: 14, alignItems: 'center', marginBottom: 12 }}>
                  <div style={{ width: 42, height: 42, borderRadius: 12, background: '#ccfbf1', color: '#0f766e', display: 'grid', placeItems: 'center', flexShrink: 0 }}>
                    <Database size={22} />
                  </div>
                  <div>
                    <h2 style={{ fontSize: 17, margin: 0 }}>3. How Was It Done? (Data &amp; Models)</h2>
                    <span className="small">Datasets &amp; Leakage-Free ML</span>
                  </div>
                </div>
                <p style={{ fontSize: 13.5, lineHeight: 1.6, color: '#334155' }}>
                  Pre-extracted biomedical acoustic measurements from two University of Oxford studies were used (no raw audio files):
                </p>
                <div style={{ display: 'grid', gap: 8, marginTop: 10 }}>
                  <div style={{ background: '#f8fafc', padding: '10px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12.5 }}>
                    <b>Classification Dataset:</b> 195 records across 31 individuals (48 healthy controls, 147 with Parkinson&apos;s). Models: <i>SVM, Random Forest, Gradient Boosting, Logistic Regression</i>.
                  </div>
                  <div style={{ background: '#f8fafc', padding: '10px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 12.5 }}>
                    <b>Telemonitoring Dataset:</b> 5,875 records tracking 42 early-stage patients over ~6 months measuring motor and total UPDRS clinical scores.
                  </div>
                </div>
              </div>

              {/* Card 4: What Was Achieved? */}
              <div className="card">
                <div style={{ display: 'flex', gap: 14, alignItems: 'center', marginBottom: 12 }}>
                  <div style={{ width: 42, height: 42, borderRadius: 12, background: '#f3e8ff', color: '#7e22ce', display: 'grid', placeItems: 'center', flexShrink: 0 }}>
                    <Trophy size={22} />
                  </div>
                  <div>
                    <h2 style={{ fontSize: 17, margin: 0 }}>4. What Was Achieved &amp; Discovered?</h2>
                    <span className="small">Key Findings &amp; Honest Limitations</span>
                  </div>
                </div>
                <p style={{ fontSize: 13.5, lineHeight: 1.6, color: '#334155' }}>
                  The models revealed distinct acoustic signatures, but also critical real-world limitations when tested on new, unseen patients:
                </p>
                <ul style={{ paddingLeft: 18, margin: '8px 0 0', fontSize: 13, color: '#475569', lineHeight: 1.6 }}>
                  <li><b>Key Predictive Biomarkers:</b> Nonlinear pitch variation (<code>PPE</code> and <code>spread1</code>) and fundamental frequency jitter are the strongest indicators.</li>
                  <li><b>High Sensitivity, Low Specificity:</b> On held-out patients, the models catch 86%–100% of Parkinson&apos;s cases, but also flag many healthy controls as positive.</li>
                  <li><b>Honest Conclusion:</b> Voice tests show strong promise as supportive biomarkers, but cannot stand alone as a medical diagnosis.</li>
                </ul>
              </div>
            </div>

            {/* WHAT THE INTERACTIVE FEATURES DO (HOW TO USE) */}
            <div className="card" style={{ marginBottom: 24 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10, marginBottom: 14 }}>
                <div>
                  <h2 style={{ fontSize: 18 }}>What You Can Do in VoxSense (Interactive Features)</h2>
                  <p>Three ways to interact with the machine learning models right now:</p>
                </div>
                <button
                  className="btn"
                  onClick={() => {
                    setPage('Analyze Voice Biomarkers');
                    setAnalyzeTab('manual');
                  }}
                >
                  Launch Predictor <Play size={15} />
                </button>
              </div>

              <div className="grid three" style={{ gap: 16 }}>
                <div
                  style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 14, padding: 18, cursor: 'pointer' }}
                  onClick={() => {
                    setPage('Analyze Voice Biomarkers');
                    setAnalyzeTab('manual');
                  }}
                >
                  <div style={{ width: 36, height: 36, borderRadius: 10, background: '#ccfbf1', color: '#0f766e', display: 'grid', placeItems: 'center', marginBottom: 10 }}>
                    <Sliders size={20} />
                  </div>
                  <h3 style={{ fontSize: 15, margin: '0 0 6px' }}>A. Manual Analysis</h3>
                  <p style={{ fontSize: 12.5, lineHeight: 1.5 }}>
                    Adjust 22 vocal feature sliders (pitch, jitter, shimmer, harmonic noise) or load healthy/Parkinson&apos;s presets to see instant probability estimates and SHAP feature explanations.
                  </p>
                </div>

                <div
                  style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 14, padding: 18, cursor: 'pointer' }}
                  onClick={() => {
                    setPage('Analyze Voice Biomarkers');
                    setAnalyzeTab('sample');
                  }}
                >
                  <div style={{ width: 36, height: 36, borderRadius: 10, background: '#e0f2fe', color: '#0369a1', display: 'grid', placeItems: 'center', marginBottom: 10 }}>
                    <UserCheck size={20} />
                  </div>
                  <h3 style={{ fontSize: 15, margin: '0 0 6px' }}>B. Sample Patients</h3>
                  <p style={{ fontSize: 12.5, lineHeight: 1.5 }}>
                    Choose from 8 real dataset records (both unseen held-out test patients and training patients) to test how the model performs on real-world recordings.
                  </p>
                </div>

                <div
                  style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 14, padding: 18, cursor: 'pointer' }}
                  onClick={() => {
                    setPage('Analyze Voice Biomarkers');
                    setAnalyzeTab('batch');
                  }}
                >
                  <div style={{ width: 36, height: 36, borderRadius: 10, background: '#fef3c7', color: '#b45309', display: 'grid', placeItems: 'center', marginBottom: 10 }}>
                    <FileSpreadsheet size={20} />
                  </div>
                  <h3 style={{ fontSize: 15, margin: '0 0 6px' }}>C. CSV Batch Upload</h3>
                  <p style={{ fontSize: 12.5, lineHeight: 1.5 }}>
                    Upload your own CSV table of acoustic features. VoxSense validates every column, runs vectorized predictions, and gives you a downloadable results CSV.
                  </p>
                </div>
              </div>
            </div>

            {/* VISUAL ACOUSTIC DISTRIBUTION PLOTS */}
            <div className="grid two" style={{ gap: 20 }}>
              <div className="card">
                <h2>Vocal Acoustic Distributions in the Study Cohort</h2>
                <p>Histograms of the 16 core vocal measures across the Parkinson&apos;s telemonitoring study:</p>
                <div style={{ marginTop: 12, borderRadius: 12, overflow: 'hidden', border: '1px solid #e2e8f0', background: '#f8fafc' }}>
                  <img
                    src="/parkinsons_updrs_distributions.png"
                    alt="Acoustic Feature Distributions"
                    style={{ width: '100%', height: 'auto', display: 'block' }}
                  />
                </div>
                <span className="small" style={{ display: 'block', marginTop: 8 }}>
                  Features show characteristic log-normal and skewed distributions in jitter and amplitude shimmer.
                </span>
              </div>

              <div className="card">
                <h2>Global SHAP Feature Importance</h2>
                <p>Which voice measurements had the highest statistical influence across all tested samples?</p>
                <div style={{ marginTop: 12, borderRadius: 12, overflow: 'hidden', border: '1px solid #e2e8f0', background: '#f8fafc' }}>
                  <img
                    src="/shap_summary.png"
                    alt="SHAP Summary Beeswarm Plot"
                    style={{ width: '100%', height: 'auto', display: 'block' }}
                  />
                </div>
                <span className="small" style={{ display: 'block', marginTop: 8 }}>
                  Beeswarm plot: Red dots show high feature values; dots to the right indicate higher predicted likelihood of Parkinson&apos;s.
                </span>
              </div>
            </div>
          </>
        )}

        {/* =========================================================
            SECTION 2: ANALYZE VOICE BIOMARKERS (MAIN FEATURE)
           ========================================================= */}
        {page === 'Analyze Voice Biomarkers' && (
          <>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
              <div className="subtabs" role="tablist">
                <button
                  className={`subtab ${analyzeTab === 'manual' ? 'active' : ''}`}
                  onClick={() => setAnalyzeTab('manual')}
                >
                  <Sliders size={15} /> A. Manual Biomarker Analysis
                </button>
                <button
                  className={`subtab ${analyzeTab === 'sample' ? 'active' : ''}`}
                  onClick={() => setAnalyzeTab('sample')}
                >
                  <UserCheck size={15} /> B. Sample Patient Analysis ({samples.length})
                </button>
                <button
                  className={`subtab ${analyzeTab === 'batch' ? 'active' : ''}`}
                  onClick={() => setAnalyzeTab('batch')}
                >
                  <FileSpreadsheet size={15} /> C. CSV Batch Analysis
                </button>
              </div>
            </div>

            {/* A: MANUAL */}
            {analyzeTab === 'manual' && (
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 14 }}>
                  <div>
                    <h2>Manual Acoustic Biomarker Analysis (22 Model Features)</h2>
                    <p>
                      Enter pre-extracted numerical voice measurements across all 22 required acoustic features, or load a preset profile below.
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                    <button className="btn ghost" onClick={() => setShowTooltips(!showTooltips)}>
                      <HelpCircle size={14} /> {showTooltips ? 'Hide Descriptions' : 'Show Descriptions'}
                    </button>
                    <button className="btn ghost" onClick={() => applyManualPreset('healthy')}>
                      Healthy Median Profile
                    </button>
                    <button className="btn ghost" onClick={() => applyManualPreset('parkinsons')}>
                      Parkinson&apos;s Median Profile
                    </button>
                    <button className="btn ghost" onClick={() => applyManualPreset('median')}>
                      <RotateCcw size={14} /> Reset
                    </button>
                  </div>
                </div>

                {manualError && (
                  <div className="notice danger" style={{ marginTop: 14, marginBottom: 0 }}>
                    <AlertCircle size={16} />
                    <div>{manualError}</div>
                  </div>
                )}

                <div style={{ marginTop: 16 }}>
                  {groupedFeatures.map(([groupName, groupItems]) => (
                    <div className="feature-group" key={groupName}>
                      <div className="feature-group-title">
                        <span>{groupName}</span>
                        <span className="small">{groupItems.length} features</span>
                      </div>
                      <div className="feature-grid">
                        {groupItems.map((f) => {
                          const err = manualValidation.errors[f.name];
                          const warn = manualValidation.warnings[f.name];
                          return (
                            <div
                              key={f.name}
                              className={`field-card ${err ? 'has-error' : warn ? 'has-warning' : ''}`}
                            >
                              <div className="field-top">
                                <label htmlFor={`feat-${f.name}`} className="field-name">
                                  {f.name}
                                </label>
                                <span className="field-unit">{f.unit}</span>
                              </div>
                              <div className="field-label">{f.label}</div>
                              <input
                                id={`feat-${f.name}`}
                                className={`text ${err ? 'invalid' : ''}`}
                                style={{ width: '100%', padding: '7px 9px' }}
                                type="number"
                                step="any"
                                value={manualInputStrings[f.name] ?? ''}
                                onChange={(e) =>
                                  setManualInputStrings({
                                    ...manualInputStrings,
                                    [f.name]: e.target.value,
                                  })
                                }
                              />
                              <div className="field-meta">
                                <span>Range: {formatNumber(f.min, 3)} – {formatNumber(f.max, 3)}</span>
                                <span>Med: {formatNumber(f.median, 3)}</span>
                              </div>
                              {showTooltips && <div className="field-desc">{f.description}</div>}
                              {err && <div style={{ color: 'var(--danger)', fontSize: 10.5, marginTop: 4 }}>{err}</div>}
                              {!err && warn && <div style={{ color: '#d97706', fontSize: 10.5, marginTop: 4 }}>{warn}</div>}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="actions" style={{ justifyContent: 'space-between', borderTop: '1px solid var(--line)', paddingTop: 16, marginBottom: 0 }}>
                  <div className="small">
                    {Object.keys(manualValidation.errors).length === 0 ? (
                      <span style={{ color: '#0d9488', fontWeight: 650 }}>
                        ✓ All 22 numerical features validated
                      </span>
                    ) : (
                      <span style={{ color: 'var(--danger)', fontWeight: 650 }}>
                        {Object.keys(manualValidation.errors).length} feature(s) need numbers
                      </span>
                    )}
                  </div>
                  <div style={{ display: 'flex', gap: 10 }}>
                    <button className="btn ghost" onClick={() => applyManualPreset('median')}>
                      <RotateCcw size={15} /> Reset
                    </button>
                    <button
                      className="btn"
                      disabled={manualBusy || !manualValidation.isValid}
                      onClick={runManualAnalysis}
                    >
                      <Play size={15} /> {manualBusy ? 'Analyzing...' : 'Analyze Biomarkers'}
                    </button>
                  </div>
                </div>

                {renderPredictionAndShap(manualPrediction, manualExplanation, null)}
              </div>
            )}

            {/* B: SAMPLE */}
            {analyzeTab === 'sample' && (
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
                  <div>
                    <h2>Sample Patient Analysis (Real Dataset Records)</h2>
                    <p>
                      Select from {samples.length} real patient recordings. Ground-truth diagnosis and whether the patient was in the training or unseen test set are clearly displayed.
                    </p>
                  </div>
                </div>

                <div className="sample-grid">
                  {samples.map((s) => {
                    const isSelected = selectedSample?.id === s.id;
                    return (
                      <button
                        key={s.id}
                        type="button"
                        className={`sample-card ${isSelected ? 'selected' : ''}`}
                        onClick={() => {
                          setSelectedSampleId(s.id);
                          setSamplePrediction(null);
                          setSampleExplanation(null);
                          setSampleError('');
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                          <span style={{ fontFamily: 'ui-monospace, monospace', fontWeight: 750, fontSize: 13.5, color: 'var(--ink)' }}>
                            {s.id}
                          </span>
                          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                            <span className={`badge ${s.actual_status === 1 ? 'rose' : 'blue'}`}>
                              Actual: {s.actual_status === 1 ? "Parkinson's" : 'Healthy'}
                            </span>
                            <span className={`badge ${s.is_independent_test ? 'teal' : 'amber'}`}>
                              {s.is_independent_test ? 'Unseen Test Set' : 'Training Set'}
                            </span>
                          </div>
                        </div>
                        <p style={{ marginTop: 7, fontSize: 12, color: '#475569', lineHeight: 1.4 }}>{s.description}</p>
                        <div className="small" style={{ marginTop: 8, display: 'flex', gap: 14 }}>
                          <span>Fo: {s.values['MDVP:Fo(Hz)']?.toFixed(1)} Hz</span>
                          <span>Jitter: {(s.values['MDVP:Jitter(%)'] * 100)?.toFixed(2)}%</span>
                          <span>HNR: {s.values['HNR']?.toFixed(1)} dB</span>
                          <span>PPE: {s.values['PPE']?.toFixed(3)}</span>
                        </div>
                      </button>
                    );
                  })}
                </div>

                {selectedSample && (
                  <div style={{ marginTop: 20, borderTop: '1px solid var(--line)', paddingTop: 18 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
                      <div>
                        <h3>
                          Selected Recording: <code>{selectedSample.id}</code> (Subject Group <code>{selectedSample.subject_group}</code>)
                        </h3>
                        <p>{selectedSample.partition_notice}</p>
                      </div>
                      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                        <button
                          className="btn"
                          disabled={sampleBusy}
                          onClick={() => runSampleAnalysis(selectedSample)}
                        >
                          <Play size={15} /> {sampleBusy ? 'Running...' : `Run Analysis on ${selectedSample.id}`}
                        </button>
                      </div>
                    </div>

                    {sampleError && (
                      <div className="notice danger" style={{ marginTop: 12 }}>
                        <AlertCircle size={16} />
                        <div>{sampleError}</div>
                      </div>
                    )}

                    <div className="table-wrap" style={{ marginTop: 14, maxHeight: 220 }}>
                      <table>
                        <thead>
                          <tr>
                            <th>Feature</th>
                            <th>Description</th>
                            <th>Sample Value</th>
                            <th>Healthy Cohort Median</th>
                            <th>Parkinson&apos;s Cohort Median</th>
                          </tr>
                        </thead>
                        <tbody>
                          {featureSchema.map((f) => {
                            const val = selectedSample.values[f.name];
                            return (
                              <tr key={f.name}>
                                <td style={{ fontFamily: 'ui-monospace, monospace', fontWeight: 700 }}>{f.name}</td>
                                <td>{f.label}</td>
                                <td style={{ fontWeight: 700 }}>{formatNumber(val, 4)}</td>
                                <td>{formatNumber(f.healthy_median, 4)}</td>
                                <td>{formatNumber(f.parkinsons_median, 4)}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>

                    {renderPredictionAndShap(samplePrediction, sampleExplanation, selectedSample)}
                  </div>
                )}
              </div>
            )}

            {/* C: BATCH */}
            {analyzeTab === 'batch' && (
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
                  <div>
                    <h2>CSV Batch Biomarker Analysis</h2>
                    <p>
                      Upload a CSV file containing the 22 required acoustic features. Target labels are not needed. Everything is processed locally in memory.
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    <button className="btn ghost" onClick={downloadTemplateCsv}>
                      <Download size={14} /> Download Template
                    </button>
                    <button className="btn secondary" onClick={loadDemoBatchCsv}>
                      Load 6-Patient Demo CSV
                    </button>
                  </div>
                </div>

                <div
                  className={`dropzone ${dragActive ? 'active' : ''}`}
                  style={{ marginTop: 16 }}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragActive(true);
                  }}
                  onDragLeave={() => setDragActive(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragActive(false);
                    const file = e.dataTransfer.files?.[0];
                    handleFileSelect(file);
                  }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,text/csv"
                    style={{ display: 'none' }}
                    onChange={(e) => handleFileSelect(e.target.files?.[0])}
                  />
                  <UploadCloud size={34} style={{ margin: '0 auto 8px', color: 'var(--teal)' }} />
                  <div style={{ fontWeight: 700, fontSize: 15 }}>
                    {csvFileName ? `Loaded: ${csvFileName}` : 'Drag & drop a CSV here, or click to browse'}
                  </div>
                  <p style={{ marginTop: 4 }}>
                    Requires the 22 Oxford acoustic feature columns (<code>MDVP:Fo(Hz)</code> through <code>PPE</code>).
                  </p>
                </div>

                {batchError && (
                  <div className="notice danger" style={{ marginTop: 14 }}>
                    <AlertCircle size={16} />
                    <div>{batchError}</div>
                  </div>
                )}

                {csvPreviewHeaders.length > 0 && (
                  <div style={{ marginTop: 18 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
                      <div>
                        <h3>CSV Preview ({csvPreviewRows.length} rows)</h3>
                        <p>
                          {csvMissingCols.length === 0 ? (
                            <span style={{ color: '#0d9488', fontWeight: 650 }}>
                              ✓ All 22 required feature columns present and validated.
                            </span>
                          ) : (
                            <span style={{ color: 'var(--danger)', fontWeight: 650 }}>
                              ✗ Missing {csvMissingCols.length} columns: {csvMissingCols.join(', ')}
                            </span>
                          )}
                        </p>
                      </div>
                      <button
                        className="btn"
                        disabled={batchBusy || csvMissingCols.length > 0 || csvPreviewRows.length === 0}
                        onClick={runBatchPrediction}
                      >
                        <Play size={15} /> {batchBusy ? 'Running Batch...' : `Run Analysis (${csvPreviewRows.length} rows)`}
                      </button>
                    </div>

                    <div className="table-wrap" style={{ maxHeight: 200 }}>
                      <table>
                        <thead>
                          <tr>
                            <th>#</th>
                            {csvPreviewHeaders.slice(0, 8).map((h) => (
                              <th key={h}>{h}</th>
                            ))}
                            {csvPreviewHeaders.length > 8 && <th>+{csvPreviewHeaders.length - 8} more cols</th>}
                          </tr>
                        </thead>
                        <tbody>
                          {csvPreviewRows.slice(0, 6).map((r, idx) => (
                            <tr key={idx}>
                              <td>{idx + 1}</td>
                              {csvPreviewHeaders.slice(0, 8).map((h) => (
                                <td key={h}>{r[h]}</td>
                              ))}
                              {csvPreviewHeaders.length > 8 && <td className="small">...</td>}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {batchResult && (
                  <div style={{ marginTop: 22, borderTop: '1px solid var(--line)', paddingTop: 18 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
                      <div>
                        <h3>Batch Results</h3>
                        <p>{batchResult.research_notice}</p>
                      </div>
                      <button className="btn" onClick={downloadBatchResultsCsv}>
                        <Download size={15} /> Download Results CSV
                      </button>
                    </div>

                    <div className="grid metrics" style={{ marginTop: 14 }}>
                      <div className="card" style={{ padding: 14 }}>
                        <div className="metric-label">Total Rows</div>
                        <div className="metric-value" style={{ fontSize: 24 }}>{batchResult.total_rows}</div>
                      </div>
                      <div className="card" style={{ padding: 14 }}>
                        <div className="metric-label">Parkinson&apos;s Estimates</div>
                        <div className="metric-value" style={{ fontSize: 24, color: '#0d9488' }}>{batchResult.predicted_parkinsons}</div>
                      </div>
                      <div className="card" style={{ padding: 14 }}>
                        <div className="metric-label">Healthy Control Estimates</div>
                        <div className="metric-value" style={{ fontSize: 24, color: '#2563eb' }}>{batchResult.predicted_healthy}</div>
                      </div>
                      <div className="card" style={{ padding: 14 }}>
                        <div className="metric-label">Invalid Rows</div>
                        <div className="metric-value" style={{ fontSize: 24, color: batchResult.invalid_rows > 0 ? 'var(--danger)' : 'var(--ink)' }}>
                          {batchResult.invalid_rows}
                        </div>
                      </div>
                    </div>

                    <div className="table-wrap" style={{ maxHeight: 280 }}>
                      <table>
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>Record ID</th>
                            <th>Status</th>
                            <th>Prediction</th>
                            <th>Parkinson&apos;s Prob</th>
                            <th>Healthy Prob</th>
                          </tr>
                        </thead>
                        <tbody>
                          {batchResult.results.map((r: any) => (
                            <tr key={r.row_number}>
                              <td>{r.row_number}</td>
                              <td style={{ fontFamily: 'ui-monospace, monospace' }}>{r.record_id}</td>
                              <td>{r.valid ? <span className="badge teal">Valid</span> : <span className="badge rose">Invalid</span>}</td>
                              <td>
                                {r.valid ? (
                                  <span className={`badge ${r.predicted_class === 1 ? 'amber' : 'blue'}`}>
                                    {r.predicted_label}
                                  </span>
                                ) : (
                                  <span style={{ color: 'var(--danger)', fontSize: 11 }}>{r.error}</span>
                                )}
                              </td>
                              <td>{r.probability_parkinsons !== null ? `${(r.probability_parkinsons * 100).toFixed(1)}%` : '—'}</td>
                              <td>{r.probability_healthy !== null ? `${(r.probability_healthy * 100).toFixed(1)}%` : '—'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}

        {/* =========================================================
            SECTION 3: PATIENT INSIGHTS (LONGITUDINAL UPDRS)
           ========================================================= */}
        {page === 'Patient Insights' && (
          <>
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
                <div>
                  <h2>Longitudinal UPDRS &amp; Voice Trajectory Explorer</h2>
                  <p>
                    Follow repeated home telemonitoring observations across all 42 early-stage Parkinson&apos;s participants (5,875 total observations).
                  </p>
                </div>
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
                  <div>
                    <label className="small" style={{ display: 'block', marginBottom: 3 }}>Primary Participant</label>
                    <select
                      className="select"
                      value={primaryPatientId}
                      onChange={(e) => loadPrimaryParticipant(Number(e.target.value))}
                    >
                      {(progressionSummary?.participants || []).map((p: any) => (
                        <option key={p.patient_id} value={p.patient_id}>
                          Subject #{p.patient_id} ({p.sex_label}, age {p.age} · {p.observations} obs)
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="small" style={{ display: 'block', marginBottom: 3 }}>Compare With</label>
                    <select
                      className="select"
                      value={comparePatientId}
                      onChange={(e) => handleCompareParticipantChange(e.target.value)}
                    >
                      <option value="none">None (Single Participant)</option>
                      {(progressionSummary?.participants || [])
                        .filter((p: any) => p.patient_id !== primaryPatientId)
                        .map((p: any) => (
                          <option key={p.patient_id} value={String(p.patient_id)}>
                            Subject #{p.patient_id} ({p.sex_label}, age {p.age})
                          </option>
                        ))}
                    </select>
                  </div>

                  <div>
                    <label className="small" style={{ display: 'block', marginBottom: 3 }}>Time Axis</label>
                    <select
                      className="select"
                      value={timeAxisMode}
                      onChange={(e) => setTimeAxisMode(e.target.value as 'test_time' | 'elapsed_days')}
                    >
                      <option value="test_time">Recruitment Day (test_time)</option>
                      <option value="elapsed_days">Elapsed Days from 1st Obs</option>
                    </select>
                  </div>
                </div>
              </div>

              {primaryTrajectory?.summary && (
                <>
                  <div className="grid metrics" style={{ marginTop: 18 }}>
                    <div className="card" style={{ padding: 14 }}>
                      <div className="metric-label">Subject #{primaryTrajectory.summary.patient_id}</div>
                      <div className="metric-value" style={{ fontSize: 22 }}>
                        {primaryTrajectory.summary.observations} observations
                      </div>
                      <div className="metric-sub">{primaryTrajectory.summary.sex_label}, age {primaryTrajectory.summary.age}</div>
                    </div>
                    <div className="card" style={{ padding: 14 }}>
                      <div className="metric-label">Motor UPDRS Mean</div>
                      <div className="metric-value" style={{ fontSize: 22 }}>
                        {primaryTrajectory.summary.motor_mean.toFixed(1)}
                      </div>
                      <div className="metric-sub">Range: {primaryTrajectory.summary.motor_min.toFixed(1)} to {primaryTrajectory.summary.motor_max.toFixed(1)}</div>
                    </div>
                    <div className="card" style={{ padding: 14 }}>
                      <div className="metric-label">Total UPDRS Mean</div>
                      <div className="metric-value" style={{ fontSize: 22 }}>
                        {primaryTrajectory.summary.total_mean.toFixed(1)}
                      </div>
                      <div className="metric-sub">Range: {primaryTrajectory.summary.total_min.toFixed(1)} to {primaryTrajectory.summary.total_max.toFixed(1)}</div>
                    </div>
                    <div className="card" style={{ padding: 14 }}>
                      <div className="metric-label">Follow-up Span</div>
                      <div className="metric-value" style={{ fontSize: 22 }}>
                        {primaryTrajectory.summary.elapsed_span_days} days
                      </div>
                      <div className="metric-sub">
                        {primaryTrajectory.summary.pre_baseline_records > 0 ? `${primaryTrajectory.summary.pre_baseline_records} pre-baseline recordings` : 'All post-recruitment'}
                      </div>
                    </div>
                  </div>

                  <div className="grid two-equal" style={{ marginTop: 16 }}>
                    <div className="card">
                      <h3>Motor &amp; Total UPDRS Scores Over Time</h3>
                      <p>Clinician-interpolated assessment scores across follow-up days:</p>
                      <div className="chart">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={comparisonChartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                            <XAxis
                              dataKey="time"
                              type="number"
                              domain={['dataMin', 'dataMax']}
                              tick={{ fontSize: 10 }}
                              tickFormatter={(v) => `${Number(v).toFixed(0)}d`}
                            />
                            <YAxis tick={{ fontSize: 10 }} />
                            <Tooltip />
                            <Legend wrapperStyle={{ fontSize: 11 }} />
                            <Line
                              type="monotone"
                              dataKey="primary_total"
                              stroke="#0d9488"
                              strokeWidth={2.4}
                              dot={false}
                              name={`Subj #${primaryTrajectory.summary.patient_id} Total UPDRS`}
                            />
                            <Line
                              type="monotone"
                              dataKey="primary_motor"
                              stroke="#2563eb"
                              strokeWidth={2.4}
                              dot={false}
                              name={`Subj #${primaryTrajectory.summary.patient_id} Motor UPDRS`}
                            />
                            {compareTrajectory?.summary && (
                              <>
                                <Line
                                  type="monotone"
                                  dataKey="compare_total"
                                  stroke="#d97706"
                                  strokeWidth={2}
                                  strokeDasharray="4 4"
                                  dot={false}
                                  name={`Subj #${compareTrajectory.summary.patient_id} Total`}
                                />
                                <Line
                                  type="monotone"
                                  dataKey="compare_motor"
                                  stroke="#dc2626"
                                  strokeWidth={2}
                                  strokeDasharray="4 4"
                                  dot={false}
                                  name={`Subj #${compareTrajectory.summary.patient_id} Motor`}
                                />
                              </>
                            )}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="card">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <div>
                          <h3>Voice Biomarker Trajectory</h3>
                          <p>Concurrent vocal stability measure recorded at each session:</p>
                        </div>
                        <select
                          className="select"
                          style={{ minWidth: 110, padding: '4px 8px' }}
                          value={selectedVoiceMetric}
                          onChange={(e) => setSelectedVoiceMetric(e.target.value)}
                        >
                          {['HNR', 'PPE', 'RPDE', 'DFA', 'Jitter(%)', 'Shimmer'].map((m) => (
                            <option key={m} value={m}>{m}</option>
                          ))}
                        </select>
                      </div>
                      <div className="chart">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={comparisonChartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                            <XAxis
                              dataKey="time"
                              type="number"
                              domain={['dataMin', 'dataMax']}
                              tick={{ fontSize: 10 }}
                              tickFormatter={(v) => `${Number(v).toFixed(0)}d`}
                            />
                            <YAxis tick={{ fontSize: 10 }} />
                            <Tooltip />
                            <Legend wrapperStyle={{ fontSize: 11 }} />
                            <Area
                              type="monotone"
                              dataKey="voice_metric"
                              stroke="#0d9488"
                              fill="#ccfbf1"
                              name={`Subj #${primaryTrajectory.summary.patient_id} ${selectedVoiceMetric}`}
                            />
                            {compareTrajectory?.summary && (
                              <Area
                                type="monotone"
                                dataKey="compare_voice"
                                stroke="#d97706"
                                fill="#fef3c7"
                                name={`Subj #${compareTrajectory.summary.patient_id} ${selectedVoiceMetric}`}
                              />
                            )}
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>

            <div className="card" style={{ marginTop: 20 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
                <div>
                  <h2>All 42 Study Participants</h2>
                  <p>Click any row to load their repeated measurements above.</p>
                </div>
                <input
                  className="text"
                  placeholder="Filter by ID, sex, age..."
                  value={participantFilter}
                  onChange={(e) => setParticipantFilter(e.target.value)}
                />
              </div>
              <div className="table-wrap" style={{ maxHeight: 300 }}>
                <table>
                  <thead>
                    <tr>
                      <th>Subject</th>
                      <th>Demographics</th>
                      <th>Records</th>
                      <th>Time Range</th>
                      <th>Motor UPDRS Mean</th>
                      <th>Total UPDRS Mean</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredParticipants.map((p: any) => (
                      <tr
                        key={p.patient_id}
                        className={`clickable-row ${p.patient_id === primaryPatientId ? 'selected-row' : ''}`}
                        onClick={() => loadPrimaryParticipant(p.patient_id)}
                      >
                        <td style={{ fontWeight: 700 }}>Subject #{p.patient_id}</td>
                        <td>{p.sex_label}, {p.age} yrs</td>
                        <td>{p.observations}</td>
                        <td>{p.test_time_min.toFixed(1)}d to {p.test_time_max.toFixed(1)}d</td>
                        <td><b>{p.motor_mean.toFixed(1)}</b></td>
                        <td><b>{p.total_mean.toFixed(1)}</b></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {/* =========================================================
            SECTION 4: ABOUT VOXSENSE
           ========================================================= */}
        {page === 'About VoxSense' && (
          <>
            <div className="grid two">
              <div className="card">
                <h2>Project Scope &amp; Provenance</h2>
                <p style={{ marginTop: 8, lineHeight: 1.6 }}>
                  VoxSense is a local-only educational AI platform exploring whether vocal acoustic biomarkers can assist in identifying voice characteristics of Parkinson&apos;s disease.
                </p>
                <h3 style={{ marginTop: 14 }}>Dataset Sources</h3>
                <ul style={{ paddingLeft: 18, margin: '8px 0 0', lineHeight: 1.6, fontSize: 13, color: '#475569' }}>
                  <li>
                    <b>Oxford Parkinson&apos;s Disease Detection Dataset (Little et al.):</b> 195 records from 31 individuals (48 healthy controls, 147 with Parkinson&apos;s). Contains 22 acoustic features.
                  </li>
                  <li style={{ marginTop: 6 }}>
                    <b>Oxford Telemonitoring Dataset (Tsanas et al.):</b> 5,875 records from 42 early-stage patients monitoring motor &amp; total UPDRS scores over 6 months from home.
                  </li>
                </ul>
              </div>

              <div className="card">
                <h2>Model Validation &amp; Leakage Prevention</h2>
                <p style={{ marginTop: 8, lineHeight: 1.6 }}>
                  Because an individual&apos;s voice recordings share baseline anatomical traits, placing recordings of the same patient in both train and test sets leads to massive data leakage and fake high accuracy.
                </p>
                <div style={{ marginTop: 12, background: '#f8fafc', padding: 14, borderRadius: 10, border: '1px solid #e2e8f0', fontSize: 13, color: '#334155' }}>
                  🔒 <b>Patient-Disjoint Holdout:</b> 25% of patient groups were completely held out. Zero recordings from those people were ever seen during training or cross-validation.
                </div>
              </div>
            </div>

            {/* ROC Curve & Confusion Matrix */}
            <div className="grid two" style={{ marginTop: 20 }}>
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
                  <div>
                    <h2>Patient-Disjoint ROC Curve</h2>
                    <p>Performance on held-out unseen patients:</p>
                  </div>
                  {metrics?.classification && (
                    <select
                      className="select"
                      value={rocModel || metrics.classification_selected}
                      onChange={(e) => setRocModel(e.target.value)}
                    >
                      {Object.entries(metrics.classification)
                        .filter(([, v]) => typeof v === 'object' && v !== null && 'roc_curve' in (v as object))
                        .map(([k]) => (
                          <option key={k} value={k}>{k}</option>
                        ))}
                    </select>
                  )}
                </div>
                <div className="chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={metrics?.classification?.[rocModel || metrics?.classification_selected]?.roc_curve || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis type="number" dataKey="fpr" domain={[0, 1]} tickCount={5} tick={{ fontSize: 10 }} />
                      <YAxis type="number" dataKey="tpr" domain={[0, 1]} tickCount={5} tick={{ fontSize: 10 }} />
                      <Tooltip />
                      <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#94a3b8" strokeDasharray="4 4" />
                      <Line type="linear" dataKey="tpr" stroke="#0d9488" dot={false} strokeWidth={2.4} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="card">
                <h2>Honest Scientific Limits &amp; Medical Disclaimer</h2>
                <div className="notice danger" style={{ marginTop: 10, marginBottom: 14 }}>
                  <ShieldAlert size={16} style={{ flexShrink: 0, marginTop: 2 }} />
                  <div>
                    <b>Not A Medical Device:</b> VoxSense is for research exploration only. It cannot diagnose disease or substitute for a neurologist.
                  </div>
                </div>
                <p style={{ fontSize: 13, lineHeight: 1.6, color: '#475569' }}>
                  On the strict patient-disjoint holdout set, the classifier demonstrates high sensitivity/recall (identifying true positive Parkinson&apos;s cases) but low specificity on this small 31-person cohort (tending to over-predict positive). This honest reporting underscores why acoustic biomarkers require comprehensive clinical context.
                </p>
              </div>
            </div>

            {/* Feature Table */}
            <div className="card" style={{ marginTop: 20 }}>
              <h2>Complete 22 Acoustic Biomarker Feature Dictionary</h2>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Category</th>
                      <th>Full Name</th>
                      <th>Unit</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {featureSchema.map((f) => (
                      <tr key={f.name}>
                        <td style={{ fontFamily: 'ui-monospace, monospace', fontWeight: 700 }}>{f.name}</td>
                        <td><span className="badge teal">{f.group}</span></td>
                        <td>{f.label}</td>
                        <td>{f.unit}</td>
                        <td>{f.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
