import React, { useEffect, useState } from 'react';
import { Activity, CheckCircle2, Orbit, ShieldCheck, Sparkles } from 'lucide-react';
import { fetchHealth } from '../services/api';
import { HealthResponse } from '../types/api';

interface HeaderProps {
  activeHorizon: string;
  onHorizonChange: (h: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ activeHorizon, onHorizonChange }) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(false);

  useEffect(() => {
    fetchHealth()
      .then((data) => {
        setHealth(data);
        setIsOnline(true);
      })
      .catch(() => setIsOnline(false));

    const interval = setInterval(() => {
      fetchHealth()
        .then((data) => {
          setHealth(data);
          setIsOnline(true);
        })
        .catch(() => setIsOnline(false));
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  const horizons = [
    { key: 'H2', label: 'H2 (48h)', days: 2 },
    { key: 'H3', label: 'H3 (72h)', days: 3 },
    { key: 'H5', label: 'H5 (120h)', days: 5 },
    { key: 'H6', label: 'H6 (144h)', days: 6 },
  ];

  return (
    <header className="bg-space-900/95 border-b border-slate-800 sticky top-0 z-40 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Left: Branding & Model Identity */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Orbit className="w-6 h-6 text-white animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold tracking-wider text-white font-mono">ORVEXA</span>
                <span className="text-xs font-semibold px-2 py-0.5 bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 rounded">
                  v1.0 Demo
                </span>
              </div>
              <p className="text-xs text-slate-400 font-medium">Orbital Conjunction Risk Prioritization</p>
            </div>
          </div>

          <div className="hidden md:flex items-center gap-2 pl-4 border-l border-slate-800 text-xs">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span className="text-slate-300 font-mono">Candidate C: Quantile M4 + CQR (Frozen)</span>
          </div>
        </div>

        {/* Right: Horizon Switcher & Backend Health Status */}
        <div className="flex items-center gap-4">
          {/* Quick Horizon Selector */}
          <div className="flex items-center bg-space-850 p-1 rounded-lg border border-slate-800">
            <span className="text-xs text-slate-400 px-2 font-medium hidden sm:inline">Horizon:</span>
            {horizons.map((h) => {
              const active = activeHorizon === h.key;
              return (
                <button
                  key={h.key}
                  onClick={() => onHorizonChange(h.key)}
                  className={`text-xs px-2.5 py-1 rounded font-mono font-medium transition-all ${
                    active
                      ? 'bg-cyan-500 text-space-950 font-bold shadow-md shadow-cyan-500/20'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-space-800'
                  }`}
                >
                  {h.key}
                </button>
              );
            })}
          </div>

          {/* Online status indicator */}
          <div className="flex items-center gap-1.5 px-3 py-1 bg-space-850 rounded-lg border border-slate-800 text-xs font-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                isOnline ? 'bg-emerald-400 shadow-sm shadow-emerald-400 animate-pulse' : 'bg-rose-500'
              }`}
            />
            <span className={isOnline ? 'text-slate-300' : 'text-rose-400'}>
              {isOnline ? 'API Connected' : 'API Offline'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
