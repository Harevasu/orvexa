import React, { useEffect, useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  BarChart3,
  HelpCircle,
  Info,
  ShieldCheck,
  TrendingDown,
  Zap,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
import { fetchBenchmarks } from '../services/api';
import { BenchmarksResponse } from '../types/api';

export const Robustness: React.FC = () => {
  const [benchmarks, setBenchmarks] = useState<BenchmarksResponse | null>(null);

  useEffect(() => {
    fetchBenchmarks().then((data) => setBenchmarks(data));
  }, []);

  const pointDegradationData = [
    { horizon: 'H2 (48h)', lead_time: 48, r2: 0.58497, mae: 3.342, rmse: 6.200, cqr_cov: 92.15 },
    { horizon: 'H3 (72h)', lead_time: 72, r2: 0.48416, mae: 3.729, rmse: 6.922, cqr_cov: 92.58 },
    { horizon: 'H5 (120h)', lead_time: 120, r2: 0.18967, mae: 5.340, rmse: 8.602, cqr_cov: 92.96 },
    { horizon: 'H6 (144h)', lead_time: 144, r2: -0.16651, mae: 5.973, rmse: 10.122, cqr_cov: 90.20 },
  ];

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-amber-400" />
            <span>Robustness & Scientific Record Audit</span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Transparent analysis of long-horizon point prediction degradation and uncertainty calibration limits.
          </p>
        </div>

        <div className="text-xs font-mono px-3 py-1 bg-amber-500/10 text-amber-300 border border-amber-500/30 rounded-lg font-semibold flex items-center gap-1.5">
          <AlertTriangle className="w-4 h-4 text-amber-400" />
          <span>Scientific Integrity First</span>
        </div>
      </div>

      {/* Primary Highlight: H6 Negative R2 Card */}
      <div className="bg-gradient-to-r from-rose-950/60 via-space-900 to-space-900 border border-rose-500/40 rounded-xl p-5 shadow-xl space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-lg bg-rose-500/20 border border-rose-500/40 flex items-center justify-center">
              <TrendingDown className="w-5 h-5 text-rose-400" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white font-mono">
                Long-Horizon Reality: H6 (144h / 6-Day) Point R² = -0.16651
              </h2>
              <p className="text-xs text-slate-400">
                Phase 5 Blind Internal Test result across 1,071 qualifying conjunction events.
              </p>
            </div>
          </div>
          <span className="text-xs font-mono font-bold px-3 py-1 bg-rose-500/20 text-rose-300 border border-rose-500/40 rounded-full">
            Negative R² Unmasked
          </span>
        </div>

        <p className="text-xs text-slate-300 leading-relaxed pt-1">
          <strong>Scientific Principle:</strong> Point predictions for conjunction risk become statistically unreliable at 6-day lead times ($R^2 &lt; 0$) because small initial orbital velocity covariance errors propagate non-linearly over millions of kilometers.
          <strong> ORVEXA does not hide this negative $R^2$.</strong> While point prediction fails, Conformalized Quantile Regression (CQR) maintains an empirical coverage of <strong>90.20%</strong> (meeting the 90.0% nominal target), providing operators with faithful decision envelopes rather than false scalar precision.
        </p>
      </div>

      {/* Point Error vs CQR Coverage Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* R2 Score Progression */}
        <div className="panel-card p-5 space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-rose-400" />
              <span>Point R² Degradation Curve</span>
            </h3>
            <span className="text-[10px] font-mono text-slate-400">Phase 5 Internal Test</span>
          </div>

          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={pointDegradationData} margin={{ top: 15, right: 30, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                <XAxis dataKey="horizon" stroke="#64748B" tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                <YAxis domain={[-0.3, 0.8]} stroke="#64748B" tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const p = payload[0].payload;
                      return (
                        <div className="bg-space-900 border border-slate-700 p-2.5 rounded shadow-xl text-xs font-mono">
                          <div className="text-cyan-400 font-bold">{p.horizon} Point Metrics</div>
                          <div className="text-white">R² Score: <span className={p.r2 >= 0 ? 'text-cyan-300' : 'text-rose-400 font-bold'}>{p.r2.toFixed(5)}</span></div>
                          <div className="text-slate-400">MAE: {p.mae.toFixed(3)} log-units</div>
                          <div className="text-slate-400">RMSE: {p.rmse.toFixed(3)} log-units</div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <ReferenceLine y={0} stroke="#EF4444" strokeDasharray="4 4" label={{ value: 'R² = 0 Baseline (Mean Predictor)', fill: '#EF4444', fontSize: 10 }} />
                <Line
                  type="monotone"
                  dataKey="r2"
                  name="R² Score"
                  stroke="#f43f5e"
                  strokeWidth={3}
                  dot={{ r: 4, fill: '#f43f5e' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* CQR Coverage Stability Across Horizons */}
        <div className="panel-card p-5 space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>CQR Coverage Resiliency (All Horizons ≥ 90%)</span>
            </h3>
            <span className="text-[10px] font-mono text-emerald-400">Stable Calibration</span>
          </div>

          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={pointDegradationData} margin={{ top: 15, right: 30, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                <XAxis dataKey="horizon" stroke="#64748B" tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                <YAxis domain={[85, 96]} stroke="#64748B" tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'JetBrains Mono' }} unit="%" />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const p = payload[0].payload;
                      return (
                        <div className="bg-space-900 border border-slate-700 p-2.5 rounded shadow-xl text-xs font-mono">
                          <div className="text-emerald-400 font-bold">{p.horizon} CQR Interval</div>
                          <div className="text-white">Empirical Coverage: <span className="text-emerald-300 font-bold">{p.cqr_cov}%</span></div>
                          <div className="text-slate-300">Target Level: 90.0%</div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <ReferenceLine y={90.0} stroke="#F59E0B" strokeWidth={2} strokeDasharray="4 4" label={{ value: '90.0% Nominal Target', fill: '#F59E0B', fontSize: 10 }} />
                <Bar dataKey="cqr_cov" name="Empirical Coverage (%)" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Scientific Insights & Nuance Table */}
      <div className="panel-card p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2 border-b border-slate-800 pb-2">
          <Info className="w-4 h-4 text-cyan-400" />
          <span>Scientific Summary & Nuanced Conclusions</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-300 leading-relaxed">
          <div className="p-3 bg-space-850 rounded-lg border border-slate-800 space-y-1.5">
            <div className="font-bold text-white flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span>Short Horizons (H2 = 48h, H3 = 72h):</span>
            </div>
            <p className="text-slate-400">
              Orbital geometry and tracking observations are dense. Both point prediction ($R^2 = +0.585$) and conformal bounds ($12.54$ log-units width) provide highly informative decision prioritization.
            </p>
          </div>

          <div className="p-3 bg-space-850 rounded-lg border border-slate-800 space-y-1.5">
            <div className="font-bold text-white flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-amber-400" />
              <span>Long Horizons (H5 = 120h, H6 = 144h):</span>
            </div>
            <p className="text-slate-400">
              Epistemic state dispersion dominates. Point predictions degrade to near zero or negative $R^2$. CQR intervals widen ($18.03$ to $20.83$ log-units) to transparently reflect genuine astrodynamic uncertainty without overconfidence.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
