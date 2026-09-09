# Outbreak Forecasting Dashboard — FastAPI Backend

FastAPI backend serving vector-borne disease outbreak forecasts (4–6 weeks ahead), historical surveillance trends, scenario simulations, and model evaluation metrics for **Maharashtra**, **Karnataka**, and **Tamil Nadu**.

---

## 1. Quick Start

### Prerequisites
Make sure your Python environment has dependencies installed:
```bash
# In the project root with the virtual environment activated:
pip install -r backend/requirements.txt
```

### Run the Server
From the project root:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 2. Endpoints Overview

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check & loaded model status |
| `GET` | `/states` | List supported states with test-set performance metrics |
| `GET` | `/history/{state}?weeks=24` | Historical case counts and weather trends |
| `GET` | `/forecast/{state}` | 4–6 week ahead forecast based on latest historical snapshot |
| `POST` | `/forecast/custom` | What-if scenario forecast on custom 12-week input |
| `GET` | `/model-info/{state}` | Full model transparency metrics, ablation info, and baselines |

---

## 3. Running Automated Tests
```bash
python backend/test_endpoints.py
```
