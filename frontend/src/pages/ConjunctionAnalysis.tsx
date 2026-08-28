import React, { useEffect, useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  CheckCircle,
  Clock,
  Compass,
  FileCode,
  Layers,
  Play,
  RotateCw,
  Search,
  ShieldCheck,
  Zap,
} from 'lucide-react';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { MetricCard } from '../components/MetricCard';
import { QuantileDistributionChart } from '../components/QuantileDistributionChart';
import { listEvents, runInference } from '../services/api';
import { EventSummary, InferenceResponse } from '../types/api';

interface ConjunctionAnalysisProps {
  activeHorizon: string;
  onHorizonChange: (h: string) => void;
  selectedEventId: string;
  onSelectEvent: (eventId: string) => void;
  onNavigateToDetail: () => void;
}

export const ConjunctionAnalysis: React.FC<ConjunctionAnalysisProps> = ({
  activeHorizon,
  onHorizonChange,
  selectedEventId,
  onSelectEvent,
  onNavigateToDetail,
}) => {
  const [events, setEvents] = useState<EventSummary[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [riskFilter, setRiskFilter] = useState<'all' | 'high' | 'moderate' | 'low'>('all');
  const [loadingEvents, setLoadingEvents] = useState(false);

  const [inferenceResult, setInferenceResult] = useState<InferenceResponse | null>(null);
  const [loadingInference, setLoadingInference] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Load candidate demo events list for the dropdown
  useEffect(() => {
    setLoadingEvents(true);
    let minR: number | undefined;
    if (riskFilter === 'high') minR = -4.0;
    else if (riskFilter === 'moderate') minR = -6.0;

    listEvents({ search: searchTerm, min_risk: minR, page_size: 100 })
      .then((data) => {
        setEvents(data.events);
        if (!selectedEventId && data.events.length > 0) {
          onSelectEvent(data.events[0].event_id);
        }
      })
      .catch((err) => setErrorMsg(err.message))
      .finally(() => setLoadingEvents(false));
  }, [searchTerm, riskFilter]);

  const selectedEventSummary = events.find((e) => e.event_id === selectedEventId);

  // Explicit inference execution triggered SOLELY by the "Analyze Conjunction" button
  const handleRunInference = () => {
    if (!selectedEventId) return;
    setLoadingInference(true);
    setErrorMsg(null);

    runInference(selectedEventId, activeHorizon, 0.10)
      .then((res) => {
        setInferenceResult(res);
      })
      .catch((err) => {
        setErrorMsg(err.message);
      })
      .finally(() => setLoadingInference(false));
  };

  const isStale =
    inferenceResult &&
    (inferenceResult.event_id !== selectedEventId || inferenceResult.horizon !== activeHorizon);

  const horizons = [
    { key: 'H2', name: 'H2 (48h)', leadHours: 48, desc: 'TCA - 2 days' },
    { key: 'H3', name: 'H3 (72h)', leadHours: 72, desc: 'TCA - 3 days' },
    { key: 'H5', name: 'H5 (120h)', leadHours: 120, desc: 'TCA - 5 days' },
    { key: 'H6', name: 'H6 (144h)', leadHours: 144, desc: 'TCA - 6 days' },
  ];

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Compass className="w-5 h-5 text-cyan-400" />
            <span>Orbital Conjunction Risk Prioritization</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Interactive inference workbench backed by frozen Candidate C (Quantile M4 Causal TCN + Conformalized Quantile Regression).
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onNavigateToDetail}
            className="px-3 py-1.5 bg-space-850 hover:bg-space-800 border border-slate-700 text-slate-300 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all"
          >
            <span>Inspect Approach CDMs</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Configuration Bar: Event Picker & Warning Horizon */}
      <div className="panel-card p-5 space-y-4">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-end">
          {/* Event Selector (Updates ONLY local state) */}
          <div className="lg:col-span-6 space-y-2">
            <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center justify-between">
              <span>Select Conjunction Event</span>
              <span className="text-[10px] text-slate-500 font-mono">Phase 5 Non-Sealed Data</span>
            </label>

            <div className="flex flex-col sm:flex-row gap-2">
              <div className="relative flex-1">
                <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-3" />
                <input
                  type="text"
                  placeholder="Search Event ID..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-8 pr-3 py-2 bg-space-950 border border-slate-800 rounded-lg text-xs font-mono text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                />
              </div>

              {/* Quick Risk Category filter */}
              <div className="flex items-center bg-space-950 p-1 rounded-lg border border-slate-800">
                <button
                  onClick={() => setRiskFilter('all')}
                  className={`text-[11px] px-2 py-1 rounded font-medium ${riskFilter === 'all' ? 'bg-cyan-500/20 text-cyan-300 font-bold' : 'text-slate-400'}`}
                >
                  All
                </button>
                <button
                  onClick={() => setRiskFilter('high')}
                  className={`text-[11px] px-2 py-1 rounded font-medium ${riskFilter === 'high' ? 'bg-rose-500/20 text-rose-300 font-bold' : 'text-slate-400'}`}
                >
                  High
                </button>
                <button
                  onClick={() => setRiskFilter('moderate')}
                  className={`text-[11px] px-2 py-1 rounded font-medium ${riskFilter === 'moderate' ? 'bg-amber-500/20 text-amber-300 font-bold' : 'text-slate-400'}`}
                >
                  Moderate
                </button>
              </div>
            </div>

            {/* Event Dropdown */}
            <select
              value={selectedEventId}
              onChange={(e) => onSelectEvent(e.target.value)}
              className="w-full py-2 px-3 bg-space-950 border border-slate-800 rounded-lg text-xs font-mono text-cyan-300 focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              {events.map((ev) => (
                <option key={ev.event_id} value={ev.event_id}>
                  Event {ev.event_id} | Final Risk: {ev.target_final_risk.toFixed(2)} | {ev.primary_object_type} ({ev.total_cdms} CDMs)
                </option>
              ))}
            </select>
          </div>

          {/* Horizon Selector (Updates ONLY local state) */}
          <div className="lg:col-span-4 space-y-2">
            <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider block">
              Warning Horizon
            </label>
            <div className="grid grid-cols-4 gap-1.5">
              {horizons.map((h) => {
                const active = activeHorizon === h.key;
                return (
                  <button
                    key={h.key}
                    onClick={() => onHorizonChange(h.key)}
                    className={`py-2 px-1 rounded-lg border text-center transition-all ${
                      active
                        ? 'bg-cyan-500/15 border-cyan-500 text-cyan-300 shadow-md font-bold'
                        : 'bg-space-950 border-slate-800 text-slate-400 hover:border-slate-700'
                    }`}
                  >
                    <div className="text-xs font-mono font-bold">{h.key}</div>
                    <div className="text-[10px] text-slate-500 font-mono">{h.leadHours}h</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Action Button: SOLE trigger for live inference */}
          <div className="lg:col-span-2">
            <button
              onClick={handleRunInference}
              disabled={loadingInference || !selectedEventId}
              className="w-full py-2.5 px-4 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg text-xs font-bold uppercase tracking-wider shadow-lg shadow-cyan-500/20 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
            >
              {loadingInference ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Analyze Conjunction</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Selected Event Context Badges */}
        {selectedEventSummary && (
          <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-800/60 text-xs font-mono">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-1.5 text-slate-400">
                <span>Total CDMs:</span>
                <span className="text-white font-semibold">{selectedEventSummary.total_cdms}</span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-400">
                <span>Approach Window:</span>
                <span className="text-white">
                  {selectedEventSummary.earliest_time_to_tca.toFixed(1)}d → {selectedEventSummary.latest_time_to_tca.toFixed(1)}d to TCA
                </span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-400">
                <span>Object Type:</span>
                <span className="text-cyan-300 font-semibold">{selectedEventSummary.primary_object_type}</span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-400">
                <span>Final CDM True Risk:</span>
                <span className="text-emerald-400 font-semibold">{selectedEventSummary.target_final_risk.toFixed(4)}</span>
              </div>
            </div>

            {isStale && (
              <span className="text-[11px] text-amber-300 bg-amber-500/10 border border-amber-500/30 px-2 py-0.5 rounded font-sans font-medium flex items-center gap-1">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                Parameters changed. Click "Analyze Conjunction" to update.
              </span>
            )}
          </div>
        )}
      </div>

      {/* Error Alert */}
      {errorMsg && (
        <div className="bg-rose-950/50 border border-rose-500/40 p-4 rounded-xl text-xs text-rose-200 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
          <div>
            <div className="font-bold text-rose-300">Inference Error</div>
            <div className="mt-0.5">{errorMsg}</div>
          </div>
        </div>
      )}

      {/* Initial Ready Prompt when no analysis has been executed yet */}
      {!inferenceResult && !loadingInference && !errorMsg && (
        <div className="panel-card p-10 text-center space-y-3">
          <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mx-auto text-cyan-400">
            <Compass className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h3 className="text-sm font-bold text-white font-mono">Conjunction Workbench Ready</h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              Select an event and warning horizon above, then click <strong className="text-cyan-300">"Analyze Conjunction"</strong> to run real-time inference using the frozen Candidate C model.
            </p>
          </div>
        </div>
      )}

      {/* Main Results Section (Displayed after user clicks Analyze Conjunction) */}
      {inferenceResult && (
        <div className="space-y-6 animate-fadeIn">
          {/* Key Metric Highlights */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Median Predicted Risk (q50)"
              value={inferenceResult.quantiles.q50.toFixed(3)}
              subtitle={`Equivalent Pc: ~10^(${inferenceResult.quantiles.q50.toFixed(2)})`}
              badge={inferenceResult.risk_assessment.level}
              badgeType={
                inferenceResult.risk_assessment.level.includes('HIGH')
                  ? 'rose'
                  : inferenceResult.risk_assessment.level.includes('MODERATE')
                  ? 'amber'
                  : 'cyan'
              }
            />

            <MetricCard
              title="90% CQR Uncertainty Interval"
              value={`[${inferenceResult.cqr_interval.lower.toFixed(2)}, ${inferenceResult.cqr_interval.upper.toFixed(2)}]`}
              subtitle={`Finite-sample calibrated width: ${inferenceResult.cqr_interval.width.toFixed(2)} log-units`}
              badge={`q̂ = ${inferenceResult.cqr_interval.conformal_shift_qhat.toFixed(3)}`}
              badgeType="amber"
            />

            <MetricCard
              title="Qualifying CDMs Ingested"
              value={`${inferenceResult.qualifying_cdms_count} / ${inferenceResult.total_cdms_count}`}
              subtitle={`CDMs with time_to_tca >= ${inferenceResult.horizon_days.toFixed(1)} days`}
              badge={`${inferenceResult.horizon} (${inferenceResult.lead_time_hours.toFixed(0)}h Lead)`}
              badgeType="cyan"
              icon={Clock}
            />

            <MetricCard
              title="Model Architecture & Status"
              value="Candidate C"
              subtitle="Frozen Multi-Quantile TCN + CQR"
              badge="SHA-256 Verified"
              badgeType="emerald"
              icon={ShieldCheck}
              mono={false}
            />
          </div>

          {/* Interactive Chart */}
          <QuantileDistributionChart
            quantiles={inferenceResult.quantiles}
            cqrInterval={inferenceResult.cqr_interval}
            targetRisk={selectedEventSummary?.target_final_risk}
          />

          {/* Interpretation and Model Metadata Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {/* Risk Assessment Details */}
            <div className="panel-card p-5 space-y-3">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2 border-b border-slate-800 pb-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <span>Risk Categorization & Astrodynamics Interpretation</span>
              </h3>

              <div className="space-y-2 text-xs">
                <div className="flex items-center justify-between p-2.5 bg-space-850 rounded-lg border border-slate-800">
                  <span className="text-slate-400">Assessed Risk Tier:</span>
                  <span className="font-bold text-white font-mono">{inferenceResult.risk_assessment.level}</span>
                </div>

                <div className="p-3 bg-space-850/60 rounded-lg border border-slate-800 text-slate-300 leading-relaxed">
                  {inferenceResult.risk_assessment.description}
                </div>

                <div className="p-2.5 bg-amber-950/20 border border-amber-500/20 rounded-lg text-[11px] text-amber-300 flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                  <span>
                    <strong>Uncertainty Awareness:</strong> Even when median risk $q_{50}$ is low, the upper 90% CQR bound indicates the potential worst-case risk consistent with observation dispersion.
                  </span>
                </div>
              </div>
            </div>

            {/* Model Governance & Artifact Hashes */}
            <div className="panel-card p-5 space-y-3">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2 border-b border-slate-800 pb-2">
                <FileCode className="w-4 h-4 text-cyan-400" />
                <span>Model Governance & SHA-256 Audit Hashes</span>
              </h3>

              <div className="space-y-2 text-xs font-mono">
                <div className="p-2 bg-space-850 rounded border border-slate-800 space-y-1">
                  <div className="text-[10px] text-slate-400 uppercase">PyTorch Model Weights Hash:</div>
                  <div className="text-cyan-300 text-[11px] truncate">{inferenceResult.model_info.model_weights_sha256}</div>
                </div>

                <div className="p-2 bg-space-850 rounded border border-slate-800 space-y-1">
                  <div className="text-[10px] text-slate-400 uppercase">CQR Calibrator Hash:</div>
                  <div className="text-cyan-300 text-[11px] truncate">{inferenceResult.model_info.cqr_calibrator_sha256}</div>
                </div>

                <div className="p-2 bg-space-850 rounded border border-slate-800 space-y-1">
                  <div className="text-[10px] text-slate-400 uppercase">M4 Preprocessor Hash:</div>
                  <div className="text-cyan-300 text-[11px] truncate">{inferenceResult.model_info.preprocessor_sha256}</div>
                </div>

                <div className="text-[11px] text-emerald-400 flex items-center gap-1.5 pt-1">
                  <CheckCircle className="w-3.5 h-3.5" />
                  <span>All model artifacts verified read-only against Candidate Freeze Manifest.</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
