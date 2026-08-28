import React, { useEffect, useState } from 'react';
import {
  AlertOctagon,
  ArrowUpRight,
  ChevronRight,
  Filter,
  Layers,
  ListOrdered,
  RotateCw,
  Search,
  ShieldAlert,
} from 'lucide-react';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { fetchRankedAlerts } from '../services/api';
import { RankedAlertItem, RankedAlertsResponse } from '../types/api';

interface RankedAlertsProps {
  activeHorizon: string;
  onHorizonChange: (h: string) => void;
  onSelectEventAndNavigate: (eventId: string) => void;
}

export const RankedAlerts: React.FC<RankedAlertsProps> = ({
  activeHorizon,
  onHorizonChange,
  onSelectEventAndNavigate,
}) => {
  const [alertsData, setAlertsData] = useState<RankedAlertsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [riskFilter, setRiskFilter] = useState<number | undefined>(undefined);
  const [searchTerm, setSearchTerm] = useState<string>('');

  useEffect(() => {
    setLoading(true);
    fetchRankedAlerts(activeHorizon, riskFilter, 100)
      .then((data) => setAlertsData(data))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, [activeHorizon, riskFilter]);

  const filteredAlerts = (alertsData?.alerts || []).filter((a) =>
    searchTerm ? a.event_id.toLowerCase().includes(searchTerm.toLowerCase()) : true
  );

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <AlertOctagon className="w-5 h-5 text-rose-400" />
            <span>Prioritized Conjunction Risk Alert Queue</span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Conjunction events ranked by predicted median log10 risk $q_{50}$ with conformalized uncertainty envelopes.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-mono px-3 py-1 bg-space-850 rounded-lg border border-slate-800 text-slate-300">
            Source: <span className="text-cyan-400 font-bold">Phase 5 Validation Partition</span>
          </span>
        </div>
      </div>

      {/* Filter and Horizon Toolbar */}
      <div className="panel-card p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Horizon Buttons */}
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <span className="text-xs text-slate-400 font-mono hidden sm:inline">Horizon:</span>
          <div className="flex items-center bg-space-950 p-1 rounded-lg border border-slate-800">
            {['H2', 'H3', 'H5', 'H6'].map((h) => (
              <button
                key={h}
                onClick={() => onHorizonChange(h)}
                className={`text-xs px-3 py-1 rounded font-mono font-medium transition-all ${
                  activeHorizon === h
                    ? 'bg-cyan-500 text-space-950 font-bold shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {h}
              </button>
            ))}
          </div>
        </div>

        {/* Search & Risk Filter */}
        <div className="flex items-center gap-3 w-full sm:w-auto">
          <div className="relative flex-1 sm:w-48">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2.5" />
            <input
              type="text"
              placeholder="Filter Event ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 bg-space-950 border border-slate-800 rounded-lg text-xs font-mono text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
            />
          </div>

          <div className="flex items-center bg-space-950 p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setRiskFilter(undefined)}
              className={`px-2.5 py-1 rounded font-medium ${
                riskFilter === undefined ? 'bg-cyan-500/20 text-cyan-300 font-bold' : 'text-slate-400'
              }`}
            >
              All Risks
            </button>
            <button
              onClick={() => setRiskFilter(-4.0)}
              className={`px-2.5 py-1 rounded font-medium ${
                riskFilter === -4.0 ? 'bg-rose-500/20 text-rose-300 font-bold' : 'text-slate-400'
              }`}
            >
              Critical (≥-4)
            </button>
            <button
              onClick={() => setRiskFilter(-6.0)}
              className={`px-2.5 py-1 rounded font-medium ${
                riskFilter === -6.0 ? 'bg-amber-500/20 text-amber-300 font-bold' : 'text-slate-400'
              }`}
            >
              Moderate (≥-6)
            </button>
          </div>
        </div>
      </div>

      {/* Ranked Alert Table */}
      <div className="panel-card p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <ListOrdered className="w-4 h-4 text-cyan-400" />
              <span>Prioritized Queue at Horizon {activeHorizon}</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Ranked from highest estimated risk to lowest. Click any event to load live in the analysis workbench.
            </p>
          </div>
          <span className="text-xs font-mono text-cyan-400 bg-space-850 px-2.5 py-1 rounded border border-slate-800">
            {filteredAlerts.length} Events Listed
          </span>
        </div>

        {loading ? (
          <div className="py-12 text-center text-slate-400 font-mono text-xs space-y-2">
            <RotateCw className="w-5 h-5 text-cyan-400 animate-spin mx-auto" />
            <div>Loading ranked alerts for horizon {activeHorizon}...</div>
          </div>
        ) : filteredAlerts.length === 0 ? (
          <div className="py-12 text-center text-slate-500 font-mono text-xs">
            No events match the selected search or risk filters.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px] bg-space-850">
                  <th className="py-2.5 px-3">Rank</th>
                  <th className="py-2.5 px-3">Event ID</th>
                  <th className="py-2.5 px-3">Horizon</th>
                  <th className="py-2.5 px-3">Median Risk (q50)</th>
                  <th className="py-2.5 px-3">90% CQR Lower</th>
                  <th className="py-2.5 px-3">90% CQR Upper</th>
                  <th className="py-2.5 px-3">Uncertainty Width</th>
                  <th className="py-2.5 px-3">Risk Tier</th>
                  <th className="py-2.5 px-3">Final Target</th>
                  <th className="py-2.5 px-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {filteredAlerts.map((item) => (
                  <tr key={item.event_id} className="hover:bg-space-850/60 transition-colors">
                    <td className="py-2.5 px-3 font-bold text-slate-400">#{item.rank}</td>
                    <td className="py-2.5 px-3 font-bold text-cyan-300">{item.event_id}</td>
                    <td className="py-2.5 px-3 text-slate-400">{item.horizon}</td>
                    <td className="py-2.5 px-3 font-bold text-white">{item.median_risk_q50.toFixed(3)}</td>
                    <td className="py-2.5 px-3 text-amber-300">{item.cqr_lower.toFixed(3)}</td>
                    <td className="py-2.5 px-3 text-amber-300">{item.cqr_upper.toFixed(3)}</td>
                    <td className="py-2.5 px-3 text-slate-300">{item.cqr_width.toFixed(2)} log-units</td>
                    <td className="py-2.5 px-3">
                      <span
                        className={`text-[10px] px-2 py-0.5 rounded border font-semibold ${
                          item.risk_level.includes('CRITICAL')
                            ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                            : item.risk_level.includes('MODERATE')
                            ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                            : 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                        }`}
                      >
                        {item.risk_level}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-emerald-400">
                      {item.target_final_risk.toFixed(3)}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <button
                        onClick={() => onSelectEventAndNavigate(item.event_id)}
                        className="px-2.5 py-1 bg-space-800 hover:bg-cyan-500 hover:text-space-950 text-cyan-300 border border-slate-700 hover:border-cyan-400 rounded text-[11px] font-semibold flex items-center gap-1 ml-auto transition-all"
                      >
                        <span>Analyze</span>
                        <ArrowUpRight className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
