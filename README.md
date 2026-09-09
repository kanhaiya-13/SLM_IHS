# Vector-Borne Disease Outbreak Forecasting (FYP)

An early-warning surveillance system that predicts vector-borne disease outbreaks (Dengue, Chikungunya, Malaria) **4–6 weeks in advance** for Indian states (**Maharashtra**, **Karnataka**, **Tamil Nadu**) using a **PatchTST-style Time-Series Transformer** with a dedicated **FastAPI Backend** and **React + TypeScript Frontend Dashboard**.

---

## 🚀 How to Run for the Viva / Demo

The application runs locally in two lightweight processes (Backend + Frontend):

### Step 1: Start the FastAPI Backend

```bash
# In project root:
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 2: Start the React Frontend

Open a second terminal:
```bash
cd frontend
npm install   # (only once)
npm run dev
```
- **Frontend Dashboard**: [http://localhost:5173](http://localhost:5173)

---

## 📋 Viva Presentation & Demonstration Walkthrough

When presenting to examiners during the viva, follow this 4-stage click-through flow:

### Stage 1: Public Health Surveillance & Risk Dashboard
1. Open [http://localhost:5173](http://localhost:5173).
2. Point out the **Persistent Disclaimer Banner** at the top: explain that the model operates on IDSP historical reports (2009–2022) with clear acknowledgment of reporting constraints and no live stream assumptions.
3. Review **Maharashtra**:
   - Show the **Risk Assessment Gauge** (e.g. Low/Moderate/High risk bucket with exact decimal probability and clear threshold guidance).
   - Point out the **Input Window Tag** (`2022-W41 to 2022-W52`) and **Forecast Horizon** (`Weeks t+4..t+6: 2023-W04 to 2023-W06`).
   - Examine the **Per-Lead-Week Confidence Breakdown** ($t+4, t+5, t+6$).
   - Click **"Inspect 12-Week Model Input Window"** to show what multi-channel features were passed to the Transformer.
4. Switch to **Tamil Nadu** or **Karnataka** using the state selector tabs to demonstrate state-specific models and varying risk levels.

### Stage 2: Meteorological & Epidemiological History (24 Weeks)
1. In the **Surveillance History Chart**, show the multi-variate trend over time.
2. Toggle the **Rainfall (Precipitation)**, **Temperature**, and **Vegetation (LAI)** overlays.
3. Hover over the red outbreak markers to explain how IDSP-confirmed outbreaks align with pre-monsoon precipitation spikes.
4. Toggle between 12W, 24W, and 52W time horizons.

### Stage 3: Scenario Laboratory ("What-If" Sensitivity Simulator)
1. Click the **"Scenario Simulator"** tab.
2. Demonstrate how surveillance officers can test hypothetical conditions when no live feed exists:
   - Click **"Monsoon Surge (+75mm rain)"**: show rainfall and vegetation rise in weeks 8–12.
   - Click **"Run Forecast Simulation"**: observe the live output and the $\Delta\%$ risk increase badge.
   - Click **"Case Cluster (+50 cases)"**: demonstrate sudden late surge impact on $t+4..t+6$ risk.
   - Click **"Reset Baseline"** to return to actual historical records.
   - Edit any specific cell directly in the table to test custom values.

### Stage 4: Model Transparency & Auditability Panel
1. Click the **"Model Transparency & Audit"** tab.
2. Highlight that the model is **NOT a black box**:
   - Plain-English officer verdict: *"On held-out test data (2021–2022), this model identified 90.6% of actual outbreak windows (Recall) with 71.6% Precision and 0.847 PR-AUC."*
   - Review the **Held-Out Test Set Metrics Table** across Window, $t+4, t+5, t+6$.
   - Review the **Benchmark Comparison** demonstrating that PatchTST outperforms LSTM and Persistence baselines.
   - Explain the architectural decision (Channel-Independent attention prevents noisy sensor channels from corrupting case autocorrelation).

---

## 🏗️ Architecture & Project Structure

```
SLM_IHS/
├── backend/
│   ├── main.py                  ← FastAPI server & CORS setup
│   ├── schemas.py               ← Pydantic v2 request/response schemas
│   ├── inference.py             ← Checkpoint loading, feature engineering, prediction
│   ├── test_endpoints.py        ← Automated endpoint test suite
│   ├── requirements.txt         ← Backend Python dependencies
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx                 ← Navigation, disclaimer banner, state switcher
│   │   │   ├── RiskDashboard.tsx          ← Risk meter, per-lead cards, input inspector
│   │   │   ├── HistoricalTrendChart.tsx   ← Interactive Recharts multi-variate trends
│   │   │   ├── ScenarioSimulator.tsx      ← "What-If" preset and custom simulator
│   │   │   ├── ModelTransparencyPanel.tsx ← Test metrics, baselines, architecture audit
│   │   │   └── Footer.tsx                 ← Prototype disclaimer
│   │   ├── services/api.ts                ← API client calling FastAPI endpoints
│   │   ├── types/index.ts                 ← TypeScript interfaces
│   │   ├── App.tsx                        ← Tab orchestration & state management
│   │   └── index.css                      ← Dark-mode surveillance design system
│   ├── package.json
│   ├── vite.config.ts
│   └── README.md
├── src/
│   ├── config.py                ← Central config (states, window size, hyperparams)
│   ├── data_pipeline.py         ← Grid construction, scalers, sliding windows
│   ├── model.py                 ← PatchTST Transformer & LSTM architectures
│   ├── train.py                 ← Training loop (AdamW, cosine LR, BCEWithLogits)
│   ├── evaluate.py              ← Test set evaluation & ablation matrix
│   └── generate_summary.py      ← Plain-language results generator
├── data/
│   ├── raw/EpiClim.csv          ← Raw IDSP data (2009–2022)
│   └── processed/               ← Serialized pickle grids and scalers per state
├── results/
│   ├── checkpoints/             ← Best trained model weights per state
│   ├── ablation_table.csv       ← Full ablation results across 15 configurations
│   ├── baseline_results.csv     ← LSTM & persistence baseline results
│   └── plots/                   ← Loss curves, lead-week degradation plots
├── RESULTS_SUMMARY.md           ← Complete summary of research findings
└── README.md                    ← Top-level guide (this file)
```

---

## 🧪 Automated Testing

To verify the backend and frontend:

```bash
# 1. Backend test suite
python backend/test_endpoints.py

# 2. Frontend build verification
cd frontend && npm run build
```

---

## 📊 Summary of Best Models Per State

| State | Best Config | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Maharashtra** | PatchTST (`CI_W_NC`) | **0.716** | **0.906** | **0.800** | 0.779 | **0.847** |
| **Karnataka** | PatchTST (`CI_W_C`) | 0.059 | 0.056 | 0.057 | 0.509 | **0.202** |
| **Tamil Nadu** | PatchTST (`CM_W_C`) | 0.295 | **0.947** | **0.450** | **0.734** | **0.475** |
