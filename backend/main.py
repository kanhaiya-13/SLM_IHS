"""
main.py — FastAPI application for Vector-Borne Disease Outbreak Forecasting Dashboard.

Serves 4–6 week ahead outbreak probability forecasts, historical surveillance trends,
custom scenario simulations, and model transparency metrics for Maharashtra,
Karnataka, and Tamil Nadu.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
from pathlib import Path

# Ensure backend/ and src/ are on sys.path whether run from project root or backend directory
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
for p in [str(BACKEND_DIR), str(ROOT_DIR / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from schemas import (
    StatesListResponse, HistoryResponse, ForecastResponse,
    CustomForecastRequest, ModelInfoResponse
)
from inference import engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("outbreak-backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload trained PyTorch models and state data on server startup."""
    logger.info("Initializing Outbreak Forecasting Engine...")
    try:
        engine.initialize()
        logger.info("Engine successfully initialized all state models.")
    except Exception as e:
        logger.error(f"Failed to initialize forecasting engine: {e}", exc_info=True)
        raise e
    yield
    logger.info("Shutting down Outbreak Forecasting Engine...")


app = FastAPI(
    title="Outbreak Forecasting Dashboard API",
    description=(
        "Public health surveillance forecasting API. Serves 4–6 week ahead vector-borne "
        "(Dengue, Chikungunya, Malaria) outbreak risk predictions powered by PatchTST "
        "time-series transformers trained on IDSP epidemiological data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Service health and overview."""
    return {
        "status": "healthy",
        "service": "Outbreak Forecasting Dashboard Backend",
        "version": "1.0.0",
        "supported_states": ["Maharashtra", "Karnataka", "Tamil Nadu"],
        "docs_url": "/docs",
        "disclaimer": "Research prototype trained on historical IDSP reports (2009–2022). Not a live feed."
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": len(engine.models) == 3,
        "states": list(engine.models.keys())
    }


@app.get(
    "/states",
    response_model=StatesListResponse,
    tags=["Surveillance"],
    summary="List supported states and summary performance"
)
async def get_states():
    """Returns list of supported target states with evaluation metrics and coverage metadata."""
    try:
        states_info = engine.get_supported_states()
        return StatesListResponse(
            states=states_info,
            default_state="Maharashtra",
            total_states=len(states_info)
        )
    except Exception as e:
        logger.error(f"Error retrieving states: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error retrieving supported states: {str(e)}"
        )


@app.get(
    "/history/{state}",
    response_model=HistoryResponse,
    tags=["Surveillance"],
    summary="Retrieve historical case counts and weather trends"
)
async def get_history(
    state: str,
    weeks: int = Query(24, ge=4, le=104, description="Number of historical weeks to retrieve (default 24)")
):
    """
    Returns the most recent N weeks of actual surveillance data for the specified state.
    Includes case counts, weather covariates (preci, LAI, Temp), and confirmed outbreak flags.
    """
    try:
        return engine.get_history(state, weeks=weeks)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"State '{state}' not found. Supported states: Maharashtra, Karnataka, Tamil Nadu."
        )
    except Exception as e:
        logger.error(f"Error retrieving history for {state}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch history for state '{state}': {str(e)}"
        )


@app.get(
    "/forecast/{state}",
    response_model=ForecastResponse,
    tags=["Forecasting"],
    summary="Generate 4–6 week ahead forecast from latest historical snapshot"
)
async def get_forecast(state: str):
    """
    Runs the state's trained PatchTST model on the most recent 12-week historical window.
    Returns:
    - Window-level outbreak probability (t+4..t+6)
    - Per-lead-week breakdown (t+4, t+5, t+6)
    - Risk classification (Low / Moderate / High)
    - 12-week input window summary
    """
    try:
        return engine.forecast_latest(state)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"State '{state}' not supported. Choose from: Maharashtra, Karnataka, Tamil Nadu."
        )
    except Exception as e:
        logger.error(f"Error computing forecast for {state}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate forecast for '{state}': {str(e)}"
        )


@app.post(
    "/forecast/custom",
    response_model=ForecastResponse,
    tags=["Forecasting"],
    summary="Run custom scenario simulation on 12-week input"
)
async def post_custom_forecast(request: CustomForecastRequest):
    """
    Accepts a user-supplied 12-week sequence of case counts and weather parameters to evaluate
    what-if outbreak risk scenarios.
    """
    try:
        return engine.forecast_custom(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during custom simulation for {request.state}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation error for '{request.state}': {str(e)}"
        )


@app.get(
    "/model-info/{state}",
    response_model=ModelInfoResponse,
    tags=["Transparency"],
    summary="Get model architecture, test metrics, and baseline comparisons"
)
async def get_model_info(state: str):
    """
    Returns full ablation configuration, test set metrics (precision/recall/F1/AUCs),
    and comparison against LSTM and Persistence baselines for complete auditability.
    """
    try:
        return engine.get_model_info(state)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"State '{state}' not found. Supported states: Maharashtra, Karnataka, Tamil Nadu."
        )
    except Exception as e:
        logger.error(f"Error retrieving model info for {state}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model info for '{state}': {str(e)}"
        )
