import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  badge?: string;
  badgeType?: 'cyan' | 'rose' | 'amber' | 'emerald' | 'slate';
  icon?: React.FC<{ className?: string }>;
  mono?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  badge,
  badgeType = 'cyan',
  icon: Icon,
  mono = true,
}) => {
  const badgeClasses = {
    cyan: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30',
    rose: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
    amber: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    slate: 'bg-slate-800 text-slate-400 border-slate-700',
  };

  return (
    <div className="panel-card p-4 flex flex-col justify-between">
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className="text-xs font-medium text-slate-400 tracking-wide uppercase">{title}</span>
        {Icon && <Icon className="w-4 h-4 text-slate-400" />}
      </div>

      <div className="flex items-baseline justify-between mt-1">
        <div className={`text-2xl font-bold text-white ${mono ? 'font-mono' : ''}`}>
          {value}
        </div>
        {badge && (
          <span className={`text-[11px] font-mono px-2 py-0.5 rounded border font-semibold ${badgeClasses[badgeType]}`}>
            {badge}
          </span>
        )}
      </div>

      {subtitle && (
        <div className="text-[11px] text-slate-400 mt-2 border-t border-slate-800/60 pt-1.5 font-normal">
          {subtitle}
        </div>
      )}
    </div>
  );
};
