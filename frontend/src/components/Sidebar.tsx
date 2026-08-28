import React from 'react';
import {
  AlertOctagon,
  BarChart3,
  BookOpen,
  Compass,
  FileText,
  Layers,
  Orbit,
  Scale,
  ShieldAlert,
} from 'lucide-react';

export type NavTab =
  | 'analysis'
  | 'event_detail'
  | 'horizon_comparison'
  | 'ranked_alerts'
  | 'reliability'
  | 'robustness'
  | 'explanations'
  | 'orbital_demo';

interface SidebarProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  selectedEventId: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  selectedEventId,
}) => {
  const navItems: { id: NavTab; label: string; icon: React.FC<{ className?: string }>; badge?: string }[] = [
    { id: 'analysis', label: 'Conjunction Risk Analysis', icon: Compass, badge: 'Main' },
    { id: 'event_detail', label: 'Event Detail & CDMs', icon: FileText },
    { id: 'horizon_comparison', label: 'Horizon Comparison', icon: Layers },
    { id: 'ranked_alerts', label: 'Ranked Alerts Queue', icon: AlertOctagon },
    { id: 'reliability', label: 'Reliability & Calibration', icon: Scale },
    { id: 'robustness', label: 'Robustness & Degradation', icon: BarChart3 },
    { id: 'explanations', label: 'Scientific Explanations', icon: BookOpen },
    { id: 'orbital_demo', label: 'Auxiliary Orbit Ephemeris', icon: Orbit },
  ];

  return (
    <aside className="w-64 bg-space-900/90 border-r border-slate-800/80 flex flex-col justify-between p-3 min-h-[calc(100vh-4rem)]">
      <div className="space-y-1">
        <div className="px-3 py-2 text-[10px] font-mono tracking-widest text-slate-400 uppercase font-semibold">
          Navigation Modules
        </div>

        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                isActive
                  ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10 font-semibold'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-space-850 border border-transparent'
              }`}
            >
              <div className="flex items-center gap-2.5">
                <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                <span className="truncate">{item.label}</span>
              </div>
              {item.badge && (
                <span className="text-[10px] font-mono bg-cyan-500/20 text-cyan-300 px-1.5 py-0.2 rounded border border-cyan-500/30">
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Selected Event Quick Info */}
      <div className="bg-space-850/80 border border-slate-800 rounded-lg p-3 text-xs space-y-1.5">
        <div className="text-[10px] font-mono uppercase text-slate-400 font-semibold flex items-center justify-between">
          <span>Active Workbench Event</span>
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
        </div>
        <div className="font-mono text-sm text-cyan-300 font-bold">
          ID: {selectedEventId || 'None'}
        </div>
        <div className="text-[11px] text-slate-400">
          Selected for live inference & detailed inspection
        </div>
      </div>
    </aside>
  );
};
