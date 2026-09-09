# Project Prompt: Outbreak Forecasting Dashboard — Frontend + Backend

## Context

The forecasting core (data pipeline + PatchTST-style model, trained per state for Maharashtra,
Karnataka, Tamil Nadu, predicting combined vector-borne — Dengue/Chikungunya/Malaria — outbreak
probability 4–6 weeks ahead) is already built and trained. This prompt covers building a
**usable application around that trained model**: a FastAPI backend serving predictions, and a
React frontend presenting them to the intended end user — a **public health / district
surveillance officer** — for **local demo/viva use**, not production deployment.

Design for the actual user: someone assessing outbreak risk to plan a response, not a data
scientist. Prioritize clarity and honesty about uncertainty over visual complexity.

---

## 1. Backend (FastAPI)

### 1.1 Core responsibilities
- Load the trained model checkpoint(s) per state on startup (one model per state, per the
  training setup — Maharashtra, Karnataka, Tamil Nadu).
- Load the processed historical data (the same state-week grid used at training/eval time) so
  the app can show recent case-count/weather history without needing a live feed.
- Serve inference endpoints that take a 12-week input window and return the forecast.

### 1.2 Required endpoints

1. `GET /states` — returns the list of supported states.
2. `GET /history/{state}?weeks=N` — returns the last N weeks (default 24) of actual data for
   that state: case counts, weather (`preci`, `LAI`, `Temp`), and whether each week was a
   confirmed outbreak week (from the ground-truth label) — this powers the historical trend
   chart on the frontend.
3. `GET /forecast/{state}` — runs the trained model on the **most recent available 12-week
   window** for that state (from the processed historical data) and returns:
   - `window_level_probability`: probability of ≥1 outbreak week in t+4..t+6
   - `per_lead_week`: `{t+4: prob, t+5: prob, t+6: prob}` breakdown
   - `input_window_summary`: the 12 weeks of data the prediction was based on (for
     transparency — the officer should be able to see what the model looked at)
   - `risk_label`: a human-readable bucket (e.g. Low / Moderate / High) derived from the
     probability using thresholds you choose and document — do not invent false precision by
     showing only a raw decimal with no interpretation aid.
4. `POST /forecast/custom` — accepts a manually supplied 12-week input window (case counts +
   weather values) in the request body and returns the same forecast shape as above. This lets
   the officer test hypothetical scenarios during a demo/viva, since there's no live data feed
   to pull a genuinely "current" window from beyond what's in the historical file.
5. `GET /model-info/{state}` — returns which ablation configuration / metrics this model
   achieved on the test set (precision/recall/F1/AUC), so the frontend can show "this
   forecast is produced by a model with X% recall on held-out data" rather than presenting the
   number as ground truth.

### 1.3 Backend requirements
- Use Pydantic models for all request/response schemas — typed, validated, self-documenting
  (FastAPI's auto-generated `/docs` should be usable as-is for the viva if needed).
- No authentication required for local demo use, but structure the code so adding auth later
  wouldn't require a rewrite (e.g., don't hardcode "no user" assumptions into business logic).
- CORS enabled for local frontend dev origin (e.g. `http://localhost:5173`).
- Clear error handling: if a state isn't supported, if a custom input window is malformed
  (wrong length, non-numeric), return a proper 4xx with a message the frontend can display
  directly to the officer — not a stack trace.

---

## 2. Frontend (React)

### 2.1 Primary screen: State risk dashboard
- State selector (Maharashtra / Karnataka / Tamil Nadu).
- Prominent risk display: the `risk_label` (Low/Moderate/High) with the underlying probability
  shown alongside it, not hidden — officers should see both the bucket and the number.
- Per-lead-week breakdown (t+4, t+5, t+6) as a small bar or line — makes clear this is a
  window, not a single-day prediction, and that confidence naturally varies by how far out
  you're looking.
- Historical trend chart (last ~24 weeks): case counts over time with outbreak weeks
  highlighted, and weather (temp/precipitation) as a secondary overlay or small multiple —
  this gives the officer context for *why* the model might be predicting elevated risk
  (e.g., rising precipitation before monsoon).

### 2.2 Secondary screen/panel: "Test a scenario"
- A form letting the officer manually adjust the 12-week input (or start from the real recent
  window and tweak a few weeks) and re-run the forecast via `/forecast/custom` — useful for
  demonstrating model behavior live during a viva ("what if cases had been higher two weeks
  ago").

### 2.3 Model transparency panel
- Show the `/model-info` metrics for the selected state's model plainly: "On historical
  test data, this model correctly identified X% of actual outbreak windows (recall), with Y%
  precision." Don't bury this — it's what makes the tool honest rather than a black box.

### 2.4 Mandatory disclaimer (do not omit)
Display, persistently visible (e.g. a banner or footer, not a one-time modal that can be
dismissed and forgotten):
> This is a research prototype trained on historical IDSP outbreak reports (2009–2022) for
> three states. It is not connected to a live surveillance feed, has not been clinically or
> operationally validated, and should not be used as the sole basis for public health
> decisions. Predictions reflect patterns in historical *reported* outbreaks, which are
> subject to known underreporting.

This isn't optional boilerplate — the intended user is a real-world decision-maker persona,
and overclaiming certainty to that audience is the single worst failure mode for this kind of
tool, demo or not.

### 2.5 Visual/UX notes
- Clean, dashboard-style layout — this is being evaluated in a viva, so clarity and
  professionalism matter more than novelty. A sidebar or top-nav for state selection, a main
  panel for the risk display and charts, is a safe, defensible layout.
- Use a charting library (Recharts or Chart.js) for the trend charts — don't hand-roll SVG
  charts for this.
- Color-code risk levels consistently (e.g., green/amber/red) but always pair color with text
  labels — never color alone to convey risk level.

---

## 3. Data flow (important — no live feed exists)

Be explicit in the implementation and in any on-screen copy that this system runs on
**historical data snapshots**, not real-time ingestion — this was established earlier in the
project as a known constraint (IDSP has no accessible live/streaming feed). `/forecast/{state}`
uses the most recent window available in the processed historical dataset, which will be a
historical date, not "today." Label the returned window's date range clearly in the UI (e.g.,
"Forecast based on data through week of [date]") so this isn't misread as a live prediction.

---

## 4. Deliverables

1. `backend/` — FastAPI app per §1, with `requirements.txt`, model-loading code pointing at
   the checkpoints produced by the training pipeline, and a `README` with exact run
   instructions (`uvicorn ...`).
2. `frontend/` — React app (Vite-based) per §2, with a `README` covering `npm install` /
   `npm run dev`, and the backend base URL configurable via `.env`.
3. A top-level `README.md` with a single combined "how to run this for the viva" walkthrough:
   start backend, start frontend, which URL to open, what to click through to demonstrate each
   feature (risk dashboard → scenario testing → model transparency panel).
4. No Docker/cloud deployment config required given local-only demo use — keep the setup to
   "clone, install, run two processes" simplicity.

---

## 5. Constraints and working notes

- This is a demo/viva tool, not a production system — don't over-invest in auth, scaling,
  or deployment infra that won't be evaluated or used.
- Do not silently invent additional data sources or "live" behavior to make the demo look more
  impressive than the underlying model — the honesty of the disclaimer and the transparency
  panel is a feature of this project, not a hedge to minimize.
- If the trained model checkpoints, the exact input feature order, or the processed dataset
  format from the training phase don't match an assumption made here, stop and ask rather than
  guessing a mismatched interface — a silent shape mismatch between training-time preprocessing
  and inference-time preprocessing is the most common way these systems break.
