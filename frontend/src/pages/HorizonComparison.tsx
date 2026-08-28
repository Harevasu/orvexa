import React, { useEffect, useState } from 'react';
import {
  BarChart2,
  CheckCircle,
  Clock,
  Layers,
  Scale,
  ShieldAlert,
  Sparkles,
  TrendingDown,
  Zap,
} from 'lucide-react';
import {
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
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { MetricCard } from '../components/MetricCard';
import { fetchBenchmarks, runMultiHorizonInference } from '../services/api';
import { BenchmarksResponse, MultiHorizonInferenceResponse } from '../types/api';

interface HorizonComparisonProps {
  selectedEventId: string;
  onSelectHorizon: (h: string) => void;
}

export const HorizonComparison: React.FC<HorizonComparisonProps> = ({
  selectedEventId,
  onSelectHorizon,
}) => {
  const [benchmarks, setBenchmarks] = useState<BenchmarksResponse | null>(null);
  const [multiInference, setMultiInference] = useState<MultiHorizonInferenceResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetchBenchmarks(),
      selectedEventId ? runMultiHorizonInference(selectedEventId, 0.10) : Promise.resolve(null),
    ])
      .then(([bData, mData]) => {
        setBenchmarks(bData);
        setMultiInference(mData);
      })
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, [selectedEventId]);

  if (!benchmarks) {
    return (
      <div className="panel-card p-12 text-center text-slate-400 font-mono text-xs">
        Loading warning horizon benchmarks...
      </div>
    );
  }

  // Chart data for benchmark performance across horizons
  const benchmarkChartData = benchmarks.horizons.map((h) => ({
    horizon: h.horizon,
    lead_time_hours: h.lead_time_hours,
    r2: h.q50_r2,
    spearman: h.q50_spearman_rho,
    cqr_coverage: h.cqr_90pct_coverage,
    interval_width: h.cqr_mean_width,
  }));

  // Chart data for current event multi-horizon inference
  const currentEventData = ['H2', 'H3', 'H5', 'H6'].map((hKey) => {
    const res = multiInference?.horizons[hKey];
    return {
      horizon: hKey,
      lead_time_hours: hKey === 'H2' ? 48 : hKey === 'H3' ? 72 : hKey === 'H5' ? 120 : 144,
      q50: res ? res.quantiles.q50 : null,
      cqr_low: res ? res.cqr_interval.lower : null,
      cqr_high: res ? res.cqr_interval.upper : null,
      width: res ? res.cqr_interval.width : null,
      qualifying_cdms: res ? res.qualifying_cdms_count : 0,
    };
  });

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Layers className="w-5 h-5 text-cyan-400" />
            <span>Warning Horizon Comparison & Trade-off Analysis</span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Cross-horizon evaluation across H2 (48h), H3 (72h), H5 (120h), and H6 (144h).
          </p>
        </div>

        <div className="text-xs font-mono px-3 py-1 bg-space-850 rounded-lg border border-slate-800 text-slate-300">
          Evaluated Partition: <span className="text-cyan-400 font-bold">Phase 5 Internal Test (1,677 events)</span>
        </div>
      </div>

      {/* Distinction Alert */}
      <div className="bg-space-900 border border-cyan-500/30 p-4 rounded-xl text-xs space-y-1">
        <div className="font-bold text-cyan-300 flex items-center gap-1.5">
          <Sparkles className="w-4 h-4 text-cyan-400" />
          <span>Scientific Horizon Taxonomy:</span>
        </div>
        <p className="text-slate-300">
          As warning lead time extends from 48 hours to 144 hours (6 days), ephemeris uncertainty increases exponentially due to atmospheric drag fluctuations.
          While point accuracy ($R^2$) degrades gracefully from $+0.585$ (H2) to negative values at H6 ($-0.166$), Conformalized Quantile Regression (CQR) maintains guaranteed distribution-free coverage ($\ge 90\%$) across all horizons.
        </p>
      </div>

      {/* Benchmark Metric Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {benchmarks.horizons.map((h) => (
          <div key={h.horizon} className="panel-card p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <span className="text-sm font-bold text-white font-mono">{h.horizon} ({h.lead_time_hours}h Lead)</span>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${h.q50_r2 > 0 ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30' : 'bg-rose-500/10 text-rose-400 border-rose-500/30'}`}>
                R² = {h.q50_r2.toFixed(3)}
              </span>
            </div>

            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex justify-between text-slate-400">
                <span>Spearman ρ:</span>
                <span className="text-white font-semibold">{h.q50_spearman_rho.toFixed(3)}</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>CQR 90% Coverage:</span>
                <span className="text-emerald-400 font-semibold">{h.cqr_90pct_coverage.toFixed(2)}%</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Mean Interval Width:</span>
                <span className="text-amber-300 font-semibold">{h.cqr_mean_width.toFixed(2)} log-units</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Pinball Loss:</span>
                <span className="text-slate-300">{h.mean_pinball_loss.toFixed(4)}</span>
              </div>
            </div>

            <button
              onClick={() => onSelectHorizon(h.horizon)}
              className="w-full py-1.5 bg-space-850 hover:bg-space-800 border border-slate-700 text-cyan-300 rounded text-xs font-mono font-semibold transition-all"
            >
              Analyze in {h.horizon} Workbench
            </button>
          </div>
        ))}
      </div>

      {/* Comparative Charts: Benchmark Point R² vs CQR Coverage */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Point Accuracy Degradation vs Horizon */}
        <div className="panel-card p-5 space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-cyan-400" />
              <span>Benchmark Point Prediction Accuracy (R² & Spearman ρ)</span>
            </h3>
            <span className="text-[10px] font-mono text-slate-400">Phase 5 Internal Test</span>
          </div>

          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={benchmarkChartData} margin={{ top: 15, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                <XAxis dataKey="horizon" stroke="#64748B" tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                <YAxis domain={[-0.3, 0.8]} stroke="#64748B" tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const p = payload[0].payload;
                      return (
                        <div className="bg-space-900 border border-slate-700 p-2.5 rounded shadow-xl text-xs font-mono space-y-1">
                          <div className="text-cyan-400 font-bold">{p.horizon} ({p.lead_time_hours}h Lead Time)</div>
                          <div className="text-white">Point R²: <span className={p.r2 >= 0 ? 'text-cyan-300' : 'text-rose-400'}>{p.r2.toFixed(4)}</span></div>
                          <div className="text-slate-300">Spearman Rho: {p.spearman.toFixed(4)}</div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <ReferenceLine y={0} stroke="#EF4444" strokeDasharray="3 3" />
                <Bar dataKey="r2" name="R² Score" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="spearman" name="Spearman Rho" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'JetBrains Mono', paddingTop: '8px' }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* CQR Empirical Coverage vs Nominal Guarantee */}
        <div className="panel-card p-5 space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Scale className="w-4 h-4 text-emerald-400" />
              <span>CQR Empirical Coverage vs Nominal 90% Guarantee</span>
            </h3>
            <span className="text-[10px] font-mono text-emerald-400">Guaranteed Valid</span>
          </div>

          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={benchmarkChartData} margin={{ top: 15, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                <XAxis dataKey="horizon" stroke="#64748B" tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                <YAxis domain={[85, 96]} stroke="#64748B" tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'JetBrains Mono' }} unit="%" />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const p = payload[0].payload;
                      return (
                        <div className="bg-space-900 border border-slate-700 p-2.5 rounded shadow-xl text-xs font-mono space-y-1">
                          <div className="text-emerald-400 font-bold">{p.horizon} CQR Coverage</div>
                          <div className="text-white">Empirical: <span className="text-emerald-300 font-bold">{p.cqr_coverage.toFixed(2)}%</span></div>
                          <div className="text-amber-400">Mean Width: {p.interval_width.toFixed(2)} log-units</div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <ReferenceLine
                  y={90.0}
                  stroke="#F59E0B"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  label={{ value: '90.0% Nominal Target', fill: '#F59E0B', fontSize: 10, position: 'insideTopRight' }}
                />
                <Bar dataKey="cqr_coverage" name="CQR Empirical Coverage (%)" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'JetBrains Mono', paddingTop: '8px' }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Active Workbench Event Multi-Horizon Sweep */}
      {selectedEventId && (
        <div className="panel-card p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Clock className="w-4 h-4 text-cyan-400" />
                <span>Active Event ({selectedEventId}) Live Multi-Horizon Sweep</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Real-time predictions and uncertainty bounds across all warning horizons.
              </p>
            </div>
            <span className="text-xs font-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-2.5 py-1 rounded">
              Event #{selectedEventId}
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px] bg-space-850">
                  <th className="py-2.5 px-3">Horizon</th>
                  <th className="py-2.5 px-3">Lead Time</th>
                  <th className="py-2.5 px-3">Qualifying CDMs</th>
                  <th className="py-2.5 px-3">Predicted q50</th>
                  <th className="py-2.5 px-3">90% CQR Lower</th>
                  <th className="py-2.5 px-3">90% CQR Upper</th>
                  <th className="py-2.5 px-3">Interval Width</th>
                  <th className="py-2.5 px-3">Risk Assessment</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {currentEventData.map((d) => {
                  const res = multiInference?.horizons[d.horizon];
                  return (
                    <tr key={d.horizon} className="hover:bg-space-850/60 transition-colors">
                      <td className="py-2.5 px-3 font-bold text-cyan-300">{d.horizon}</td>
                      <td className="py-2.5 px-3 text-slate-400">{d.lead_time_hours}h</td>
                      <td className="py-2.5 px-3 text-slate-300">{d.qualifying_cdms} CDMs</td>
                      <td className="py-2.5 px-3 font-bold text-white">
                        {d.q50 !== null ? d.q50.toFixed(3) : <span className="text-slate-500">No CDMs</span>}
                      </td>
                      <td className="py-2.5 px-3 text-amber-300">
                        {d.cqr_low !== null ? d.cqr_low.toFixed(3) : '-'}
                      </td>
                      <td className="py-2.5 px-3 text-amber-300">
                        {d.cqr_high !== null ? d.cqr_high.toFixed(3) : '-'}
                      </td>
                      <td className="py-2.5 px-3 text-slate-300">
                        {d.width !== null ? `${d.width.toFixed(2)} log-units` : '-'}
                      </td>
                      <td className="py-2.5 px-3">
                        {res ? (
                          <span
                            className={`text-[10px] px-2 py-0.5 rounded border font-semibold ${
                              res.risk_assessment.level.includes('HIGH')
                                ? 'bg-rose-500/20 text-rose-400 border-rose-500/40'
                                : res.risk_assessment.level.includes('MODERATE')
                                ? 'bg-amber-500/20 text-amber-400 border-amber-500/40'
                                : 'bg-cyan-500/20 text-cyan-400 border-cyan-500/40'
                            }`}
                          >
                            {res.risk_assessment.level}
                          </span>
                        ) : (
                          <span className="text-slate-600">Unqualified</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
