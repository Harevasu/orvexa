import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertCircle,
  ArrowLeft,
  Clock,
  Compass,
  FileSpreadsheet,
  Globe,
  Radio,
  Sun,
  TrendingDown,
} from 'lucide-react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { MetricCard } from '../components/MetricCard';
import { fetchEventDetail } from '../services/api';
import { CDMRecord, EventDetailResponse } from '../types/api';

interface EventDetailProps {
  eventId: string;
  onBackToAnalysis: () => void;
}

export const EventDetail: React.FC<EventDetailProps> = ({ eventId, onBackToAnalysis }) => {
  const [detail, setDetail] = useState<EventDetailResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!eventId) return;
    setLoading(true);
    setErrorMsg(null);

    fetchEventDetail(eventId)
      .then((data) => setDetail(data))
      .catch((err) => setErrorMsg(err.message))
      .finally(() => setLoading(false));
  }, [eventId]);

  if (loading) {
    return (
      <div className="panel-card p-12 text-center text-slate-400 font-mono text-xs space-y-2">
        <Activity className="w-6 h-6 text-cyan-400 animate-spin mx-auto" />
        <div>Loading conjunction approach history for Event {eventId}...</div>
      </div>
    );
  }

  if (errorMsg || !detail) {
    return (
      <div className="space-y-4">
        <DisclaimerBanner />
        <div className="bg-rose-950/50 border border-rose-500/40 p-4 rounded-xl text-xs text-rose-200 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
          <div>
            <div className="font-bold text-rose-300">Error Loading Event Detail</div>
            <div className="mt-0.5">{errorMsg || 'Event not found.'}</div>
            <button
              onClick={onBackToAnalysis}
              className="mt-3 px-3 py-1.5 bg-space-800 text-slate-300 rounded text-xs hover:bg-space-700"
            >
              Return to Analysis
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Prepare chart data (sorted oldest to newest, so time_to_tca decreases towards 0)
  const chartData = detail.cdms.map((c, i) => ({
    index: i + 1,
    time_to_tca: c.time_to_tca,
    time_label: `${c.time_to_tca.toFixed(2)}d`,
    miss_distance: c.miss_distance,
    relative_speed: c.relative_speed,
    mahalanobis_distance: c.mahalanobis_distance,
    risk: c.risk,
    t_sigma_r: c.t_sigma_r,
    c_sigma_r: c.c_sigma_r,
  }));

  const latestCDM = detail.cdms[detail.cdms.length - 1];
  const earliestCDM = detail.cdms[0];

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <button
            onClick={onBackToAnalysis}
            className="p-2 bg-space-900 border border-slate-800 hover:border-slate-700 text-slate-300 rounded-lg transition-all"
            title="Back to Workbench"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <h1 className="text-xl font-bold text-white flex items-center gap-2">
              <Compass className="w-5 h-5 text-cyan-400" />
              <span>Conjunction Event Detail: {detail.event_id}</span>
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Full chronological approach sequence ({detail.total_cdms} CDMs) and relative orbital geometry.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-mono px-3 py-1 bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 rounded-lg font-semibold">
            Split: {detail.split.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Overview Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Target Object Type"
          value={detail.primary_object_type}
          subtitle={`Chaser secondary object classification`}
          badge="Verified CDM"
          badgeType="cyan"
          mono={false}
        />

        <MetricCard
          title="Total CDMs Received"
          value={detail.total_cdms}
          subtitle={`Span: ${earliestCDM.time_to_tca.toFixed(2)}d → ${latestCDM.time_to_tca.toFixed(2)}d to TCA`}
          badge="Full History"
          badgeType="emerald"
        />

        <MetricCard
          title="Minimum Miss Distance"
          value={
            latestCDM.miss_distance !== null && latestCDM.miss_distance !== undefined
              ? `${latestCDM.miss_distance.toFixed(1)} m`
              : 'N/A'
          }
          subtitle="Estimated closest geometric separation"
          badge="Approach Geometry"
          badgeType="amber"
        />

        <MetricCard
          title="Final Target Risk"
          value={detail.target_final_risk.toFixed(4)}
          subtitle={`log10(Pc) at final observation`}
          badge={detail.target_final_risk >= -4 ? 'HIGH' : detail.target_final_risk >= -6 ? 'MODERATE' : 'LOW'}
          badgeType={detail.target_final_risk >= -4 ? 'rose' : detail.target_final_risk >= -6 ? 'amber' : 'cyan'}
        />
      </div>

      {/* Approach Geometry Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Miss Distance Progression Chart */}
        <div className="panel-card p-5 space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-cyan-400" />
              <span>Miss Distance vs Approach Time</span>
            </h3>
            <span className="text-[10px] font-mono text-slate-400">Oldest → Newest</span>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                <XAxis
                  dataKey="time_label"
                  stroke="#64748B"
                  tick={{ fill: '#94A3B8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                />
                <YAxis
                  stroke="#64748B"
                  tick={{ fill: '#94A3B8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  label={{ value: 'Miss Dist (m)', angle: -90, position: 'insideLeft', fill: '#94A3B8', fontSize: 10 }}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const p = payload[0].payload;
                      return (
                        <div className="bg-space-900 border border-slate-700 p-2.5 rounded shadow-xl text-xs font-mono">
                          <div className="text-cyan-400 font-bold">CDM #{p.index} ({p.time_label} to TCA)</div>
                          <div className="text-white">Miss Distance: {p.miss_distance?.toFixed(1)} m</div>
                          <div className="text-slate-400">Relative Speed: {p.relative_speed?.toFixed(2)} m/s</div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="miss_distance"
                  stroke="#06b6d4"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: '#06b6d4' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Mahalanobis Distance Progression Chart */}
        <div className="panel-card p-5 space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Radio className="w-4 h-4 text-amber-400" />
              <span>Mahalanobis Distance & Covariance Scaling</span>
            </h3>
            <span className="text-[10px] font-mono text-slate-400">Normalized Uncertainty</span>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                <XAxis
                  dataKey="time_label"
                  stroke="#64748B"
                  tick={{ fill: '#94A3B8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                />
                <YAxis
                  stroke="#64748B"
                  tick={{ fill: '#94A3B8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  label={{ value: 'Mahalanobis Dist', angle: -90, position: 'insideLeft', fill: '#94A3B8', fontSize: 10 }}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const p = payload[0].payload;
                      return (
                        <div className="bg-space-900 border border-slate-700 p-2.5 rounded shadow-xl text-xs font-mono">
                          <div className="text-amber-400 font-bold">CDM #{p.index} ({p.time_label} to TCA)</div>
                          <div className="text-white">Mahalanobis Distance: {p.mahalanobis_distance?.toFixed(2)}</div>
                          <div className="text-slate-400">Target Sigma R: {p.t_sigma_r?.toFixed(2)} m</div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="mahalanobis_distance"
                  stroke="#f59e0b"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: '#f59e0b' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Sequential CDM Data Table */}
      <div className="panel-card p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <FileSpreadsheet className="w-4 h-4 text-cyan-400" />
              <span>Chronological Conjunction Data Messages (CDMs)</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Exact recorded measurements from earliest approach to closest encounter.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400 bg-space-850 px-2.5 py-1 rounded border border-slate-800">
            {detail.cdms.length} Updates
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px] bg-space-850">
                <th className="py-2.5 px-3">#</th>
                <th className="py-2.5 px-3">Time to TCA (d)</th>
                <th className="py-2.5 px-3">Miss Dist (m)</th>
                <th className="py-2.5 px-3">Rel Speed (m/s)</th>
                <th className="py-2.5 px-3">Mahalanobis</th>
                <th className="py-2.5 px-3">Pos (R, T, N) m</th>
                <th className="py-2.5 px-3">T-Sigma R (m)</th>
                <th className="py-2.5 px-3">Obs Used</th>
                <th className="py-2.5 px-3">Risk log10(Pc)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {detail.cdms.map((c, i) => (
                <tr key={i} className="hover:bg-space-850/60 transition-colors">
                  <td className="py-2 px-3 text-slate-500">{i + 1}</td>
                  <td className="py-2 px-3 font-semibold text-cyan-300">{c.time_to_tca.toFixed(3)}</td>
                  <td className="py-2 px-3 text-white">{c.miss_distance !== null ? c.miss_distance.toFixed(1) : '-'}</td>
                  <td className="py-2 px-3 text-slate-300">{c.relative_speed !== null ? c.relative_speed.toFixed(2) : '-'}</td>
                  <td className="py-2 px-3 text-amber-300">{c.mahalanobis_distance !== null ? c.mahalanobis_distance.toFixed(2) : '-'}</td>
                  <td className="py-2 px-3 text-slate-400 text-[11px]">
                    {c.relative_position_r !== null
                      ? `(${c.relative_position_r.toFixed(0)}, ${c.relative_position_t?.toFixed(0)}, ${c.relative_position_n?.toFixed(0)})`
                      : '-'}
                  </td>
                  <td className="py-2 px-3 text-slate-300">{c.t_sigma_r !== null ? c.t_sigma_r.toFixed(1) : '-'}</td>
                  <td className="py-2 px-3 text-slate-400">{c.t_obs_used ?? '-'}</td>
                  <td className="py-2 px-3 font-bold text-emerald-400">
                    {c.risk !== null ? c.risk.toFixed(4) : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
