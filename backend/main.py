"""ORVEXA FastAPI Backend Application.

Main entry point for ORVEXA Conjunction Risk Prioritization inference and API services.
"""

from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root and src are discoverable in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.benchmarks_service import benchmarks_service
from backend.data_service import data_service
from backend.inference import inference_engine
from backend.orbital_service import orbital_service
from backend.schemas import (
    BenchmarksResponse,
    EventDetailResponse,
    EventListResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    MultiHorizonInferenceResponse,
    OrbitalPropagationResponse,
    RankedAlertsResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up data caches and verify frozen model artifacts on server startup."""
    print("[ORVEXA Backend] Initializing data services and loading non-sealed demo events...")
    data_service.load_data()
    print(f"[ORVEXA Backend] Loaded {len(data_service.allowed_event_ids)} allowed non-sealed demo events.")
    print("[ORVEXA Backend] Loading and verifying frozen Candidate C artifacts...")
    inference_engine.load_artifacts()
    print("[ORVEXA Backend] Candidate C artifacts successfully verified and ready.")
    yield
    print("[ORVEXA Backend] Shutting down.")


app = FastAPI(
    title="ORVEXA Conjunction Risk Prioritization API",
    description=(
        "Production demonstration API for ORVEXA: Deep Conformal Risk Prioritization for Orbital Conjunctions. "
        "Operates strictly with frozen Phase 5 Candidate C (Quantile M4 Causal TCN + CQR) artifacts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local Vite dev server and arbitrary frontend hosts
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "app": "ORVEXA Conjunction Risk Prioritization API",
        "status": "operational",
        "docs": "/docs",
        "disclaimer": "Research estimate only. ORVEXA is not an operational collision-avoidance authority.",
    }


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check and governance candidate verification."""
    return HealthResponse()


@app.get("/api/manifest", tags=["Governance"])
async def get_freeze_manifest():
    """Return the full frozen Candidate C freeze manifest and artifact hashes."""
    try:
        return benchmarks_service.get_candidate_manifest()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/benchmarks", response_model=BenchmarksResponse, tags=["Scientific Record"])
async def get_benchmarks():
    """Retrieve Phase 5 Internal Test benchmarks across all 4 operational warning horizons."""
    try:
        return benchmarks_service.get_benchmarks()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/events", response_model=EventListResponse, tags=["Data Catalog"])
async def list_events(
    search: Optional[str] = Query(None, description="Search by event ID or object type"),
    split: Optional[str] = Query(None, description="Filter by split: validation or calibration"),
    min_risk: Optional[float] = Query(None, description="Minimum final target risk filter"),
    horizon_qual: Optional[str] = Query(None, description="Filter events qualifying for horizon (H2, H3, H5, H6)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
):
    """List allowed non-sealed demo conjunction events with optional filtering and pagination."""
    try:
        events, total = data_service.list_events(
            search=search,
            split=split,
            min_risk=min_risk,
            horizon_qual=horizon_qual,
            page=page,
            page_size=page_size,
        )
        return EventListResponse(
            total_count=total,
            page=page,
            page_size=page_size,
            events=events,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/events/{event_id}", response_model=EventDetailResponse, tags=["Data Catalog"])
async def get_event_detail(event_id: str):
    """Retrieve full CDM approach history, relative geometry, and covariance dispersion for an event."""
    try:
        return data_service.get_event_detail(event_id)
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
    except KeyError as ke:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ke))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/inference", response_model=InferenceResponse, tags=["Model Inference"])
async def run_inference(request: InferenceRequest):
    """Execute live Candidate C inference on an allowed demo conjunction event.

    Outputs monotonic quantile predictions and calibrated 90% CQR uncertainty interval.
    """
    try:
        return inference_engine.predict_event(
            event_id=request.event_id,
            horizon=request.horizon,
            alpha=request.alpha,
        )
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
    except (KeyError, ValueError) as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/inference/{event_id}/multi-horizon", response_model=MultiHorizonInferenceResponse, tags=["Model Inference"])
async def run_multi_horizon_inference(
    event_id: str,
    alpha: float = Query(0.10, ge=0.01, le=0.50, description="Significance level"),
):
    """Execute Candidate C inference across all 4 operational horizons (H2, H3, H5, H6) for side-by-side evaluation."""
    try:
        return inference_engine.predict_multi_horizon(event_id=event_id, alpha=alpha)
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
    except KeyError as ke:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ke))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/ranked_alerts", response_model=RankedAlertsResponse, tags=["Prioritization"])
async def get_ranked_alerts(
    horizon: str = Query("H2", description="Horizon key: H2, H3, H5, or H6"),
    min_risk: Optional[float] = Query(None, description="Optional minimum risk filter"),
    limit: int = Query(100, ge=1, le=500, description="Maximum alerts to return"),
):
    """Retrieve operational risk prioritization alert queue for the selected horizon."""
    try:
        return benchmarks_service.get_ranked_alerts(
            horizon=horizon,
            min_risk=min_risk,
            limit=limit,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/orbital/satellites", tags=["Orbital Tools"])
async def list_orbital_satellites():
    """List available demo orbital propagation targets."""
    return orbital_service.get_demo_satellites()


@app.get("/api/orbital/propagate", response_model=OrbitalPropagationResponse, tags=["Orbital Tools"])
async def propagate_orbit_endpoint(
    satellite: str = Query("SENTINEL_1A", description="Satellite key (SENTINEL_1A, ENVISAT_DEBRIS, COSMOS_2251_DEBRIS)"),
    duration_hours: float = Query(3.0, ge=0.5, le=24.0, description="Propagation duration in hours"),
    step_seconds: float = Query(120.0, ge=30.0, le=600.0, description="Ephemeris time step in seconds"),
):
    """Compute SGP4 / analytical orbital trajectory points for demo visualization."""
    try:
        return orbital_service.propagate_demo_satellite(
            satellite_key=satellite,
            duration_hours=duration_hours,
            step_seconds=step_seconds,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
