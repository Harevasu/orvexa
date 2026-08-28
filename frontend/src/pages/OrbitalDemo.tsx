import React, { useEffect, useState } from 'react';
import {
  Compass,
  Globe,
  Info,
  Orbit,
  Play,
  RotateCw,
  TrendingUp,
  Zap,
} from 'lucide-react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { MetricCard } from '../components/MetricCard';
import { fetchOrbitalSatellites, propagateOrbit } from '../services/api';
import { OrbitalPropagationResponse } from '../types/api';

export const OrbitalDemo: React.FC = () => {
  const [satellites, setSatellites] = useState<Record<string, { name: string; norad_id: string }>>({});
  const [selectedSat, setSelectedSat] = useState<string>('SENTINEL_1A');
  const [durationHours, setDurationHours] = useState<number>(3.0);
  const [propagation, setPropagation] = useState<OrbitalPropagationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchOrbitalSatellites().then((data) => {
      setSatellites(data);
      if (Object.keys(data).length > 0 && !data[selectedSat]) {
        setSelectedSat(Object.keys(data)[0]);
      }
    });
  }, []);

  const handlePropagate = () => {
    setLoading(true);
    propagateOrbit(selectedSat, durationHours, 120.0)
      .then((data) => setPropagation(data))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    handlePropagate();
  }, [selectedSat, durationHours]);

  const groundTrackData = (propagation?.trajectory || []).map((p) => ({
    lon: p.longitude_deg,
    lat: p.latitude_deg,
    alt: p.altitude_km,
    t_min: Math.round(p.timestamp_offset_sec / 60),
    v: Math.sqrt(p.vx_km_s ** 2 + p.vy_km_s ** 2 + p.vz_km_s ** 2).toFixed(2),
  }));

  const altitudeData = (propagation?.trajectory || []).map((p) => ({
    time_min: Math.round(p.timestamp_offset_sec / 60),
    altitude_km: p.altitude_km,
    velocity_km_s: Number(Math.sqrt(p.vx_km_s ** 2 + p.vy_km_s ** 2 + p.vz_km_s ** 2).toFixed(3)),
  }));

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Orbit className="w-5 h-5 text-cyan-400" />
            <span>Auxiliary Orbital Ephemeris & Propagation Demonstration</span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            SGP4 and J2-perturbed analytical orbit propagation engine from Two-Line Element sets (TLE).
          </p>
        </div>

        <div className="text-xs font-mono px-3 py-1 bg-space-850 rounded-lg border border-slate-800 text-amber-300 flex items-center gap-1.5">
          <Info className="w-4 h-4 text-amber-400" />
          <span>Decoupled from ESA ML Risk Scoring</span>
        </div>
      </div>

      {/* Control Panel */}
      <div className="panel-card p-5 space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-end">
          <div className="sm:col-span-6 space-y-1.5">
            <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider block">
              Select Resident Space Object (TLE Target)
            </label>
            <select
              value={selectedSat}
              onChange={(e) => setSelectedSat(e.target.value)}
              className="w-full py-2 px-3 bg-space-950 border border-slate-800 rounded-lg text-xs font-mono text-cyan-300 focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              {Object.entries(satellites).map(([key, s]) => (
                <option key={key} value={key}>
                  {s.name} (NORAD {s.norad_id})
                </option>
              ))}
            </select>
          </div>

          <div className="sm:col-span-3 space-y-1.5">
            <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider block">
              Propagation Horizon
            </label>
            <select
              value={durationHours}
              onChange={(e) => setDurationHours(Number(e.target.value))}
              className="w-full py-2 px-3 bg-space-950 border border-slate-800 rounded-lg text-xs font-mono text-white focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              <option value={1.5}>1.5 Hours (~1 Orbit)</option>
              <option value={3.0}>3.0 Hours (~2 Orbits)</option>
              <option value={6.0}>6.0 Hours (~4 Orbits)</option>
              <option value={12.0}>12.0 Hours (~8 Orbits)</option>
            </select>
          </div>

          <div className="sm:col-span-3">
            <button
              onClick={handlePropagate}
              disabled={loading}
              className="w-full py-2.5 px-4 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg text-xs font-bold uppercase tracking-wider shadow-lg shadow-cyan-500/20 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
            >
              {loading ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin" />
                  <span>Propagating...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Run Ephemeris</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Overview Metrics */}
      {propagation && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Target Satellite"
            value={propagation.satellite_name.split('(')[0].trim()}
            subtitle={`NORAD Catalog ID: ${propagation.norad_id || 'N/A'}`}
            badge="LEO Object"
            badgeType="cyan"
            mono={false}
          />

          <MetricCard
            title="Calculated Ephemeris Points"
            value={propagation.trajectory.length}
            subtitle={`Step: ${propagation.step_seconds}s interval`}
            badge={`${propagation.duration_hours}h Duration`}
            badgeType="emerald"
          />

          <MetricCard
            title="Current Altitude"
            value={`${propagation.trajectory[0]?.altitude_km.toFixed(1)} km`}
            subtitle="Geodetic WGS84 altitude"
            badge="Apogee / Perigee"
            badgeType="amber"
          />

          <MetricCard
            title="Orbital Speed"
            value={`${Math.sqrt(
              propagation.trajectory[0]?.vx_km_s ** 2 +
              propagation.trajectory[0]?.vy_km_s ** 2 +
              propagation.trajectory[0]?.vz_km_s ** 2
            ).toFixed(2)} km/s`}
            subtitle="ECI velocity magnitude"
            badge="Orbital Mechanics"
            badgeType="cyan"
          />
        </div>
      )}

      {/* Ground Track and Altitude Profiles */}
      {propagation && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* Ground Track Latitude vs Longitude */}
          <div className="panel-card p-5 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Globe className="w-4 h-4 text-cyan-400" />
                <span>Geodetic Ground Track (Latitude vs Longitude)</span>
              </h3>
              <span className="text-[10px] font-mono text-slate-400">Sub-satellite Point</span>
            </div>

            <div className="h-60 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 15, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis
                    type="number"
                    dataKey="lon"
                    name="Longitude"
                    domain={[-180, 180]}
                    stroke="#64748B"
                    tick={{ fill: '#94A3B8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                    unit="°"
                  />
                  <YAxis
                    type="number"
                    dataKey="lat"
                    name="Latitude"
                    domain={[-90, 90]}
                    stroke="#64748B"
                    tick={{ fill: '#94A3B8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                    unit="°"
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const p = payload[0].payload;
                        return (
                          <div className="bg-space-900 border border-slate-700 p-2.5 rounded shadow-xl text-xs font-mono">
                            <div className="text-cyan-400 font-bold">t = {p.t_min} mins</div>
                            <div className="text-white">Lat: {p.lat.toFixed(2)}°, Lon: {p.lon.toFixed(2)}°</div>
                            <div className="text-slate-400">Altitude: {p.alt.toFixed(1)} km | Speed: {p.v} km/s</div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Scatter data={groundTrackData} fill="#06b6d4" line={{ stroke: '#0891b2', strokeWidth: 1.5 }} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Altitude Profile vs Elapsed Time */}
          <div className="panel-card p-5 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-amber-400" />
                <span>Altitude & Orbital Velocity Profile</span>
              </h3>
              <span className="text-[10px] font-mono text-slate-400">J2 Analytical Orbit</span>
            </div>

            <div className="h-60 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={altitudeData} margin={{ top: 15, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                  <XAxis
                    dataKey="time_min"
                    stroke="#64748B"
                    tick={{ fill: '#94A3B8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                    unit="m"
                  />
                  <YAxis
                    stroke="#64748B"
                    tick={{ fill: '#94A3B8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                    label={{ value: 'Alt (km)', angle: -90, position: 'insideLeft', fill: '#94A3B8', fontSize: 10 }}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const p = payload[0].payload;
                        return (
                          <div className="bg-space-900 border border-slate-700 p-2.5 rounded shadow-xl text-xs font-mono">
                            <div className="text-amber-400 font-bold">Elapsed: {p.time_min} mins</div>
                            <div className="text-white">Altitude: {p.altitude_km.toFixed(2)} km</div>
                            <div className="text-slate-400">Velocity: {p.velocity_km_s.toFixed(3)} km/s</div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="altitude_km"
                    stroke="#f59e0b"
                    strokeWidth={2.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Trajectory Table */}
      {propagation && (
        <div className="panel-card p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Compass className="w-4 h-4 text-cyan-400" />
                <span>Ephemeris State Vectors (ECI & Geodetic)</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Exact propagated state vectors at 120-second cadence.
              </p>
            </div>
            <span className="text-xs font-mono text-slate-400 bg-space-850 px-2.5 py-1 rounded border border-slate-800">
              {propagation.trajectory.length} State Vectors
            </span>
          </div>

          <div className="overflow-x-auto max-h-80">
            <table className="w-full text-left text-xs font-mono border-collapse">
              <thead className="sticky top-0 bg-space-850">
                <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px]">
                  <th className="py-2 px-3">Time (s)</th>
                  <th className="py-2 px-3">ECI Pos X (km)</th>
                  <th className="py-2 px-3">ECI Pos Y (km)</th>
                  <th className="py-2 px-3">ECI Pos Z (km)</th>
                  <th className="py-2 px-3">Velocity (km/s)</th>
                  <th className="py-2 px-3">Latitude (°)</th>
                  <th className="py-2 px-3">Longitude (°)</th>
                  <th className="py-2 px-3">Altitude (km)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {propagation.trajectory.map((pt, i) => (
                  <tr key={i} className="hover:bg-space-850/60 transition-colors">
                    <td className="py-1.5 px-3 text-cyan-300">+{pt.timestamp_offset_sec}s</td>
                    <td className="py-1.5 px-3 text-slate-300">{pt.x_km.toFixed(1)}</td>
                    <td className="py-1.5 px-3 text-slate-300">{pt.y_km.toFixed(1)}</td>
                    <td className="py-1.5 px-3 text-slate-300">{pt.z_km.toFixed(1)}</td>
                    <td className="py-1.5 px-3 text-amber-300">
                      {Math.sqrt(pt.vx_km_s ** 2 + pt.vy_km_s ** 2 + pt.vz_km_s ** 2).toFixed(3)}
                    </td>
                    <td className="py-1.5 px-3 text-white">{pt.latitude_deg.toFixed(2)}°</td>
                    <td className="py-1.5 px-3 text-white">{pt.longitude_deg.toFixed(2)}°</td>
                    <td className="py-1.5 px-3 font-semibold text-emerald-400">{pt.altitude_km.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
