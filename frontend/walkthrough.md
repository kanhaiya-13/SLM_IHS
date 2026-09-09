# Walkthrough: Outbreak Forecasting Dashboard (Frontend + Backend)

A complete, production-grade application for the **Vector-Borne Disease Outbreak Forecasting Early Warning System** (Maharashtra, Karnataka, Tamil Nadu).

---

## 1. What Was Built

### Backend (`backend/`)
- **[main.py](file:///c:/Users/23101A0034/SLM_IHS/backend/main.py)**: FastAPI application with CORS, lifespan startup preloading, and endpoints for `/states`, `/history/{state}`, `/forecast/{state}`, `/forecast/custom`, and `/model-info/{state}`.
- **[schemas.py](file:///c:/Users/23101A0034/SLM_IHS/backend/schemas.py)**: Pydantic v2 schemas for all request/response models with validation (e.g., ensuring exactly 12-week windows).
- **[inference.py](file:///c:/Users/23101A0034/SLM_IHS/backend/inference.py)**: Inference engine loading the best PatchTST checkpoints per state, applying fitted `StandardScaler` scalers, and generating window-level & per-lead probabilities ($t+4, t+5, t+6$).
- **[test_endpoints.py](file:///c:/Users/23101A0034/SLM_IHS/backend/test_endpoints.py)**: Automated test suite validating health, data retrieval, inference, custom scenario simulations, and error handling.
- **[requirements.txt](file:///c:/Users/23101A0034/SLM_IHS/backend/requirements.txt)** & **[README.md](file:///c:/Users/23101A0034/SLM_IHS/backend/README.md)**.

### Frontend (`frontend/`)
- **[App.tsx](file:///c:/Users/23101A0034/SLM_IHS/frontend/src/App.tsx)**: React + TypeScript single-page application orchestrating state switching, tab views, and live API communication.
- **[Header.tsx](file:///c:/Users/23101A0034/SLM_IHS/frontend/src/components/Header.tsx)**: Persistent disclaimer banner, state switcher tabs, and live backend connection badge.
- **[RiskDashboard.tsx](file:///c:/Users/23101A0034/SLM_IHS/frontend/src/components/RiskDashboard.tsx)**: Prominent risk meter (Low/Moderate/High), probability breakdown, per-lead-week ($t+4, t+5, t+6$) progression bars, historical snapshot tags, and 12-week input window inspector.
- **[HistoricalTrendChart.tsx](file:///c:/Users/23101A0034/SLM_IHS/frontend/src/components/HistoricalTrendChart.tsx)**: Recharts-powered multi-variate time-series tracking cases against rainfall, temperature, and vegetation (LAI) with confirmed IDSP outbreak markers.
- **[ScenarioSimulator.tsx](file:///c:/Users/23101A0034/SLM_IHS/frontend/src/components/ScenarioSimulator.tsx)**: Interactive "What-If" testing laboratory with scenario presets (*Monsoon Surge*, *Severe Drought*, *Case Cluster*, *Reset Baseline*) and week-by-week table inputs.
- **[ModelTransparencyPanel.tsx](file:///c:/Users/23101A0034/SLM_IHS/frontend/src/components/ModelTransparencyPanel.tsx)**: Plain-English officer interpretation, full test-set evaluation metrics table, benchmark comparisons against LSTM and Persistence baselines, and architectural design breakdown.
- **[index.css](file:///c:/Users/23101A0034/SLM_IHS/frontend/src/index.css)**: Modern dark-slate public health surveillance design system with high-contrast risk badges and glassmorphism styling.

---

## 2. Verification Results

### Backend Automated Test Suite
Ran `python backend/test_endpoints.py`:
- `GET /health` &rarr; **PASS** (Models loaded: Maharashtra, Karnataka, Tamil Nadu)
- `GET /states` &rarr; **PASS** (Returned 3 states with test-set PR-AUC and F1 metrics)
- `GET /history/Maharashtra?weeks=24` &rarr; **PASS** (Returned 24 continuous weekly records)
- `GET /forecast/Maharashtra` &rarr; **PASS** (Risk: Low 11.7%, Leads: `{'t+4': 0.2839, 't+5': 0.4428, 't+6': 0.3448}`)
- `GET /forecast/Karnataka` &rarr; **PASS** (Risk: Low 6.9%, Leads: `{'t+4': 0.1643, 't+5': 0.2694, 't+6': 0.0521}`)
- `GET /forecast/Tamil Nadu` &rarr; **PASS** (Risk: High 87.2%, Leads: `{'t+4': 0.3368, 't+5': 0.1882, 't+6': 0.3722}`)
- `POST /forecast/custom` (Synthetic scenario) &rarr; **PASS** (Risk: High 95.5%)
- `POST /forecast/custom` (Validation check on invalid window length) &rarr; **PASS** (Rejected with 422)
- `GET /model-info/Maharashtra` &rarr; **PASS** (Window PR-AUC 0.847, Recall 0.906, Precision 0.716)
- `GET /forecast/Atlantis` (404 Error handling) &rarr; **PASS** (Returned clean 404 with helpful error message)

### Frontend Build Verification
Ran `npm run build` in `frontend/`:
- **TypeScript Type Checking**: 0 errors
- **Vite Production Bundle**: `dist/index.html` (0.96 kB), `dist/assets/index-CQ4UrDgE.css` (9.76 kB), `dist/assets/index-DhUga7Mk.js` (633 kB). Build completed cleanly.

---

## 3. How to Run the System

### Start Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Swagger API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Start Frontend
```bash
cd frontend
npm run dev
```
Dashboard: [http://localhost:5173](http://localhost:5173)

---

## 4. Viva Presentation Guide

1. **Surveillance & Risk Dashboard**:
   - Open [http://localhost:5173](http://localhost:5173).
   - Point out the **Persistent IDSP Prototype Disclaimer** at the top.
   - Switch between **Maharashtra** (Low baseline risk 11.7%), **Karnataka** (6.9%), and **Tamil Nadu** (High risk 87.2% due to seasonal winter monsoon dynamics).
   - Explain the **Per-Lead-Week Curve** ($t+4, t+5, t+6$).
   - Expand the **12-Week Model Input Window** to demonstrate model transparency.
2. **Surveillance History Chart**:
   - Toggle rainfall, temperature, and LAI overlays on the 24-week multi-variate chart.
   - Hover on red outbreak dots to show how outbreaks follow weather triggers.
3. **Scenario Laboratory**:
   - Switch to the **Scenario Simulator** tab.
   - Click **"Monsoon Surge"** or **"Case Cluster"** and click **"Run Forecast Simulation"**.
   - Show the live probability comparison and $\Delta\%$ risk jump tag.
4. **Model Transparency & Evaluation**:
   - Switch to the **Model Transparency & Audit** tab.
   - Present the plain-English summary, empirical test-set metrics (Recall 90.6%, Precision 71.6%, PR-AUC 0.847 for Maharashtra), and benchmark comparisons showing PatchTST outperforming LSTM and Persistence baselines.
