import React from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { CQRIntervalOutput, QuantilesOutput } from '../types/api';

interface QuantileDistributionChartProps {
  quantiles: QuantilesOutput;
  cqrInterval: CQRIntervalOutput;
  targetRisk?: number | null;
}

export const QuantileDistributionChart: React.FC<QuantileDistributionChartProps> = ({
  quantiles,
  cqrInterval,
  targetRisk,
}) => {
  // Construct sequence data for the distribution fan curve
  const data = [
    { tau: 0.05, label: 'q05', risk: quantiles.q05, cqrLow: cqrInterval.lower },
    { tau: 0.10, label: 'q10', risk: quantiles.q10 },
    { tau: 0.25, label: 'q25', risk: quantiles.q25 },
    { tau: 0.50, label: 'q50 (Median)', risk: quantiles.q50 },
    { tau: 0.75, label: 'q75', risk: quantiles.q75 },
    { tau: 0.90, label: 'q90', risk: quantiles.q90 },
    { tau: 0.95, label: 'q95', risk: quantiles.q95, cqrHigh: cqrInterval.upper },
  ];

  const minVal = Math.floor(Math.min(cqrInterval.lower, quantiles.q05, targetRisk ?? 0) - 2);
  const maxVal = Math.ceil(Math.max(cqrInterval.upper, quantiles.q95, targetRisk ?? 0) + 2);

  return (
    <div className="panel-card p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <span>Predicted Quantile Spectrum & 90% CQR Uncertainty Bounds</span>
            <span className="text-[10px] font-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-2 py-0.5 rounded">
              Monotonic Step Head
            </span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Log10 collision risk across quantiles τ ∈ [0.05, 0.95] expanded by conformal shift q̂ = {cqrInterval.conformal_shift_qhat.toFixed(3)}
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-cyan-400 inline-block" />
            <span className="text-slate-300">q50 Median: {quantiles.q50.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-1 bg-amber-400 inline-block border-t border-dashed" />
            <span className="text-amber-300">90% CQR Bounds: [{cqrInterval.lower.toFixed(2)}, {cqrInterval.upper.toFixed(2)}]</span>
          </div>
        </div>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 15, right: 30, left: 10, bottom: 5 }}>
            <defs>
              <linearGradient id="quantileGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
            <XAxis
              dataKey="label"
              stroke="#64748B"
              tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'JetBrains Mono' }}
            />
            <YAxis
              domain={[minVal, maxVal]}
              stroke="#64748B"
              tick={{ fill: '#94A3B8', fontSize: 11, fontFamily: 'JetBrains Mono' }}
              label={{
                value: 'log10(Pc) Risk',
                angle: -90,
                position: 'insideLeft',
                fill: '#94A3B8',
                fontSize: 11,
              }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const p = payload[0].payload;
                  return (
                    <div className="bg-space-900 border border-slate-700 p-2.5 rounded-lg shadow-xl text-xs font-mono space-y-1">
                      <div className="text-cyan-400 font-bold">{p.label} (τ = {p.tau})</div>
                      <div className="text-white">Predicted log10 Risk: <span className="text-cyan-300">{p.risk.toFixed(4)}</span></div>
                      <div className="text-slate-400 text-[10px]">Equivalent Pc: ~10^({p.risk.toFixed(2)})</div>
                    </div>
                  );
                }
                return null;
              }}
            />
            {/* CQR Upper & Lower Bound Reference Lines */}
            <ReferenceLine
              y={cqrInterval.upper}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              label={{
                value: `CQR Upper (${cqrInterval.upper.toFixed(2)})`,
                fill: '#f59e0b',
                fontSize: 10,
                position: 'top',
                fontFamily: 'JetBrains Mono',
              }}
            />
            <ReferenceLine
              y={cqrInterval.lower}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              label={{
                value: `CQR Lower (${cqrInterval.lower.toFixed(2)})`,
                fill: '#f59e0b',
                fontSize: 10,
                position: 'bottom',
                fontFamily: 'JetBrains Mono',
              }}
            />
            {/* Final True Target Risk if available */}
            {targetRisk !== undefined && targetRisk !== null && (
              <ReferenceLine
                y={targetRisk}
                stroke="#10b981"
                strokeWidth={2}
                label={{
                  value: `Final CDM Target (${targetRisk.toFixed(2)})`,
                  fill: '#10b981',
                  fontSize: 10,
                  position: 'right',
                  fontFamily: 'JetBrains Mono',
                }}
              />
            )}
            <Area
              type="monotone"
              dataKey="risk"
              stroke="#06b6d4"
              strokeWidth={3}
              fillOpacity={1}
              fill="url(#quantileGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-slate-800/80 text-xs font-mono">
        <div className="bg-space-850 p-2 rounded border border-slate-800">
          <span className="text-slate-400 text-[10px] block">q05 (Lower Model Bound)</span>
          <span className="text-cyan-300 font-semibold">{quantiles.q05.toFixed(2)}</span>
        </div>
        <div className="bg-space-850 p-2 rounded border border-slate-800">
          <span className="text-slate-400 text-[10px] block">q50 (Median Point Risk)</span>
          <span className="text-white font-bold">{quantiles.q50.toFixed(2)}</span>
        </div>
        <div className="bg-space-850 p-2 rounded border border-slate-800">
          <span className="text-slate-400 text-[10px] block">q95 (Upper Model Bound)</span>
          <span className="text-cyan-300 font-semibold">{quantiles.q95.toFixed(2)}</span>
        </div>
        <div className="bg-space-850 p-2 rounded border border-slate-800">
          <span className="text-slate-400 text-[10px] block">90% CQR Interval Width</span>
          <span className="text-amber-300 font-semibold">{cqrInterval.width.toFixed(2)} log-units</span>
        </div>
      </div>
    </div>
  );
};
