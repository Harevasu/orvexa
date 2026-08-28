import React, { useEffect, useState } from 'react';
import {
  Award,
  CheckCircle2,
  HelpCircle,
  Layers,
  Scale,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
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

export const Reliability: React.FC = () => {
  const [benchmarks, setBenchmarks] = useState<BenchmarksResponse | null>(null);

  useEffect(() => {
    fetchBenchmarks().then((data) => setBenchmarks(data));
  }, []);

  const coverageData = [
    {
      horizon: 'H2 (48h)',
      coverage: 92.15,
      nominal: 90.0,
      mean_width: 12.54,
      median_width: 9.41,
      q_hat: 1.348,
      lead_time: '48h',
      status: 'Valid (Coverage >= 90%)',
    },
    {
      horizon: 'H3 (72h)',
      coverage: 92.58,
      nominal: 90.0,
      mean_width: 17.95,
      median_width: 14.94,
      q_hat: 4.283,
      lead_time: '72h',
      status: 'Valid (Coverage >= 90%)',
    },
    {
      horizon: 'H5 (120h)',
      coverage: 92.96,
      nominal: 90.0,
      mean_width: 20.83,
      median_width: 20.18,
      q_hat: 3.479,
      lead_time: '120h',
      status: 'Valid (Coverage >= 90%)',
    },
    {
      horizon: 'H6 (144h)',
      coverage: 90.20,
      nominal: 90.0,
      mean_width: 18.03,
      median_width: 19.53,
      q_hat: 0.100,
      lead_time: '144h',
      status: 'Valid (Coverage >= 90%)',
    },
  ];

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Scale className="w-5 h-5 text-cyan-400" />
            <span>Reliability & Conformal Calibration Audit</span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Empirical validation of distribution-free coverage guarantees on the Phase 5 Internal Test dataset.
          </p>
        </div>

        <div className="text-xs font-mono px-3 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 rounded-lg font-semibold flex items-center gap-1.5">
          <ShieldCheck className="w-4 h-4" />
          <span>Nominal Guarantee: 1 - α = 90.0%</span>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="H2 Empirical Coverage"
          value="92.15%"
          subtitle="Nominal Target: 90.00% (+2.15% margin)"
          badge="H2 (48h)"
          badgeType="emerald"
          icon={CheckCircle2}
        />
        <MetricCard
          title="H3 Empirical Coverage"
          value="92.58%"
          subtitle="Nominal Target: 90.00% (+2.58% margin)"
          badge="H3 (72h)"
          badgeType="emerald"
          icon={CheckCircle2}
        />
        <MetricCard
          title="H5 Empirical Coverage"
          value="92.96%"
          subtitle="Nominal Target: 90.00% (+2.96% margin)"
          badge="H5 (120h)"
          badgeType="emerald"
          icon={CheckCircle2}
        />
        <MetricCard
          title="H6 Empirical Coverage"
          value="90.20%"
          subtitle="Nominal Target: 90.00% (+0.20% margin)"
          badge="H6 (144h)"
          badgeType="emerald"
          icon={CheckCircle2}
        />
      </div>

      {/* Main Chart: Empirical Coverage vs Nominal 90% Target */}
      <div className="panel-card p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Award className="w-4 h-4 text-emerald-400" />
              <span>Phase 5 Blind Internal Test Conformal Coverage Audit</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Strict finite-sample non-parametric coverage evaluated on 1,677 unmanipulated test events.
            </p>
          </div>
          <span className="text-xs font-mono text-emerald-400 bg-space-850 px-2.5 py-1 rounded border border-slate-800">
            All 4 Horizons Met (≥ 90.0%)
          </span>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={coverageData} margin={{ top: 20, right: 30, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
              <XAxis
                dataKey="horizon"
                stroke="#64748B"
                tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'JetBrains Mono' }}
              />
              <YAxis
                domain={[85, 96]}
                stroke="#64748B"
                tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                unit="%"
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const p = payload[0].payload;
                    return (
                      <div className="bg-space-900 border border-slate-700 p-3 rounded-lg shadow-xl text-xs font-mono space-y-1.5">
                        <div className="text-emerald-400 font-bold">{p.horizon} Reliability</div>
                        <div className="text-white">Empirical Coverage: <span className="text-emerald-300 font-bold">{p.coverage}%</span></div>
                        <div className="text-slate-300">Nominal Guarantee: 90.0%</div>
                        <div className="text-amber-400">Mean Interval Width: {p.mean_width} log-units</div>
                        <div className="text-cyan-400">Conformal Shift q̂: {p.q_hat}</div>
                        <div className="text-emerald-400 font-semibold text-[10px]">{p.status}</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <ReferenceLine
                y={90.0}
                stroke="#F59E0B"
                strokeWidth={2.5}
                strokeDasharray="4 4"
                label={{
                  value: 'Nominal 90.0% Confidence Guarantee (1 - α)',
                  fill: '#F59E0B',
                  fontSize: 11,
                  fontFamily: 'JetBrains Mono',
                  position: 'insideTopRight',
                }}
              />
              <Bar dataKey="coverage" name="Empirical CQR Coverage (%)" radius={[4, 4, 0, 0]}>
                {coverageData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill="#10b981" />
                ))}
              </Bar>
              <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'JetBrains Mono', paddingTop: '8px' }} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Conformal Prediction Mathematical & Governance Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="panel-card p-5 space-y-3">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2 border-b border-slate-800 pb-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <span>Inductive Split Conformal Prediction (CQR) Guarantees</span>
          </h3>

          <div className="text-xs text-slate-300 space-y-2.5 leading-relaxed">
            <p>
              Under exchangeability of calibration and test conjunction samples, Inductive Split Conformal Prediction satisfies the finite-sample non-parametric coverage theorem:
            </p>
            <div className="p-3 bg-space-950 rounded-lg border border-slate-800 font-mono text-cyan-300 text-center text-[13px]">
              P( Y ∈ [ q̂₀₅(X) - q̂, q̂₉₅(X) + q̂ ] ) ≥ 1 - α
            </div>
            <p>
              where <code className="text-amber-300 font-mono">q̂</code> is the <code className="text-amber-300 font-mono">⌈(n_cal + 1)(1 - α)⌉ / n_cal</code> quantile of nonconformity scores computed on the strictly disjoint Phase 5 Calibration set.
            </p>
            <div className="p-2.5 bg-emerald-950/20 border border-emerald-500/20 rounded-lg text-[11px] text-emerald-300 flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
              <span>
                <strong>Zero Leakage Enforcement:</strong> The calibration set (1,008 events) was never seen during model training, and the internal test set (1,677 events) was evaluated blindly without refitting.
              </span>
            </div>
          </div>
        </div>

        <div className="panel-card p-5 space-y-3">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2 border-b border-slate-800 pb-2">
            <Layers className="w-4 h-4 text-amber-400" />
            <span>Calibration Artifact Audit Summary</span>
          </h3>

          <div className="overflow-x-auto text-xs font-mono">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 text-[10px] uppercase">
                  <th className="py-2 px-2">Horizon</th>
                  <th className="py-2 px-2">N_cal Samples</th>
                  <th className="py-2 px-2">Conformal q̂</th>
                  <th className="py-2 px-2">Mean Width</th>
                  <th className="py-2 px-2">Reduction vs Constant Residual</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                <tr>
                  <td className="py-2 px-2 font-bold text-cyan-400">H2 (48h)</td>
                  <td className="py-2 px-2">1,008</td>
                  <td className="py-2 px-2 text-amber-300">1.348</td>
                  <td className="py-2 px-2">12.54</td>
                  <td className="py-2 px-2 text-emerald-400 font-semibold">49.21% narrower</td>
                </tr>
                <tr>
                  <td className="py-2 px-2 font-bold text-cyan-400">H3 (72h)</td>
                  <td className="py-2 px-2">951</td>
                  <td className="py-2 px-2 text-amber-300">4.283</td>
                  <td className="py-2 px-2">17.95</td>
                  <td className="py-2 px-2 text-emerald-400 font-semibold">47.69% narrower</td>
                </tr>
                <tr>
                  <td className="py-2 px-2 font-bold text-cyan-400">H5 (120h)</td>
                  <td className="py-2 px-2">797</td>
                  <td className="py-2 px-2 text-amber-300">3.479</td>
                  <td className="py-2 px-2">20.83</td>
                  <td className="py-2 px-2 text-emerald-400 font-semibold">47.15% narrower</td>
                </tr>
                <tr>
                  <td className="py-2 px-2 font-bold text-cyan-400">H6 (144h)</td>
                  <td className="py-2 px-2">700</td>
                  <td className="py-2 px-2 text-amber-300">0.100</td>
                  <td className="py-2 px-2">18.03</td>
                  <td className="py-2 px-2 text-emerald-400 font-semibold">52.45% narrower</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
