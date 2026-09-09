# Outbreak Forecasting Dashboard — React Frontend

A modern, responsive React + TypeScript surveillance dashboard built with Vite and Recharts, designed for public health surveillance officers to assess vector-borne disease outbreak risk 4–6 weeks in advance.

---

## 1. Quick Start

### Prerequisites
Make sure Node.js (v18+) is installed.

### Install Dependencies
```bash
cd frontend
npm install
```

### Configure Backend API URL
Create or edit `.env` in the `frontend/` directory (defaults to `http://localhost:8000`):
```env
VITE_API_BASE_URL=http://localhost:8000
```

### Start Development Server
```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your web browser.

---

## 2. Features

- **Executive Outbreak Risk Assessment**:
  - Live 4–6 week ahead window-level probability meter with color-coded risk buckets (Low / Moderate / High).
  - Per-lead-week progression ($t+4, t+5, t+6$) confidence curve.
  - Transparent 12-week model input inspector.
- **Multi-Variate Historical Surveillance (12W, 24W, 52W)**:
  - Synchronized chart tracking case counts against monsoon precipitation, temperature, and vegetation (LAI).
  - Confirmed IDSP outbreak week indicator markers.
- **Scenario Laboratory (&ldquo;What-If&rdquo; Simulator)**:
  - Interactive table allowing manual adjustments of weekly cases, rainfall, and temperatures across the 12-week input window.
  - One-click presets: *Monsoon Surge*, *Severe Drought*, *Late Case Cluster*, and *Baseline Reset*.
  - Live side-by-side risk difference badge ($\Delta\%$ risk jump).
- **Model Transparency & Auditability Panel**:
  - Plain-English officer interpretation summary.
  - Empirical test set metrics (Precision, Recall, F1, ROC-AUC, PR-AUC).
  - Benchmark comparison against LSTM and Persistence baselines.
- **Persistent Disclaimer**:
  - Unobtrusive, persistent banner clarifying the research prototype nature and historical IDSP dataset context.

---

## 3. Production Build
```bash
npm run build
npm run preview
```
