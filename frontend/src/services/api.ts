import {
  BenchmarksResponse,
  EventDetailResponse,
  EventListResponse,
  HealthResponse,
  InferenceResponse,
  MultiHorizonInferenceResponse,
  OrbitalPropagationResponse,
  RankedAlertsResponse,
} from '../types/api';

const API_BASE = '/api';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Failed to check health: ${res.statusText}`);
  return res.json();
}

export async function fetchManifest(): Promise<Record<string, any>> {
  const res = await fetch(`${API_BASE}/manifest`);
  if (!res.ok) throw new Error(`Failed to fetch manifest: ${res.statusText}`);
  return res.json();
}

export async function fetchBenchmarks(): Promise<BenchmarksResponse> {
  const res = await fetch(`${API_BASE}/benchmarks`);
  if (!res.ok) throw new Error(`Failed to fetch benchmarks: ${res.statusText}`);
  return res.json();
}

export async function listEvents(params: {
  search?: string;
  split?: string;
  min_risk?: number;
  horizon_qual?: string;
  page?: number;
  page_size?: number;
}): Promise<EventListResponse> {
  const query = new URLSearchParams();
  if (params.search) query.set('search', params.search);
  if (params.split) query.set('split', params.split);
  if (params.min_risk !== undefined) query.set('min_risk', params.min_risk.toString());
  if (params.horizon_qual) query.set('horizon_qual', params.horizon_qual);
  if (params.page) query.set('page', params.page.toString());
  if (params.page_size) query.set('page_size', params.page_size.toString());

  const res = await fetch(`${API_BASE}/events?${query.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch events: ${res.statusText}`);
  return res.json();
}

export async function fetchEventDetail(eventId: string): Promise<EventDetailResponse> {
  const res = await fetch(`${API_BASE}/events/${encodeURIComponent(eventId)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to fetch event detail for ${eventId}`);
  }
  return res.json();
}

export async function runInference(
  eventId: string,
  horizon: string = 'H2',
  alpha: number = 0.10
): Promise<InferenceResponse> {
  const res = await fetch(`${API_BASE}/inference`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event_id: eventId, horizon, alpha }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Inference error on event ${eventId}`);
  }
  return res.json();
}

export async function runMultiHorizonInference(
  eventId: string,
  alpha: number = 0.10
): Promise<MultiHorizonInferenceResponse> {
  const query = new URLSearchParams({ alpha: alpha.toString() });
  const res = await fetch(`${API_BASE}/inference/${encodeURIComponent(eventId)}/multi-horizon?${query.toString()}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Multi-horizon inference error on event ${eventId}`);
  }
  return res.json();
}

export async function fetchRankedAlerts(
  horizon: string = 'H2',
  minRisk?: number,
  limit: number = 100
): Promise<RankedAlertsResponse> {
  const query = new URLSearchParams({ horizon, limit: limit.toString() });
  if (minRisk !== undefined) query.set('min_risk', minRisk.toString());

  const res = await fetch(`${API_BASE}/ranked_alerts?${query.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch ranked alerts: ${res.statusText}`);
  return res.json();
}

export async function fetchOrbitalSatellites(): Promise<Record<string, { name: string; norad_id: string }>> {
  const res = await fetch(`${API_BASE}/orbital/satellites`);
  if (!res.ok) throw new Error(`Failed to list satellites: ${res.statusText}`);
  return res.json();
}

export async function propagateOrbit(
  satellite: string = 'SENTINEL_1A',
  durationHours: number = 3.0,
  stepSeconds: number = 120.0
): Promise<OrbitalPropagationResponse> {
  const query = new URLSearchParams({
    satellite,
    duration_hours: durationHours.toString(),
    step_seconds: stepSeconds.toString(),
  });
  const res = await fetch(`${API_BASE}/orbital/propagate?${query.toString()}`);
  if (!res.ok) throw new Error(`Failed to propagate orbit: ${res.statusText}`);
  return res.json();
}
