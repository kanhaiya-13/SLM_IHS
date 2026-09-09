"""
test_endpoints.py — Comprehensive automated test suite for FastAPI backend endpoints.

Tests:
1. GET /health
2. GET /states
3. GET /history/{state}?weeks=24
4. GET /forecast/{state}
5. POST /forecast/custom (valid 12 weeks)
6. POST /forecast/custom (error case: <12 weeks)
7. GET /model-info/{state}
8. 404 Error handling for invalid state
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from starlette.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    print("Testing GET /health...")
    with TestClient(app) as c:
        resp = c.get("/health")
        assert resp.status_code == 200, f"Health check failed: {resp.text}"
        data = resp.json()
        assert data["models_loaded"] is True
        assert len(data["states"]) == 3
        print(f"  [PASS] Health check OK. Models loaded: {data['states']}")


def test_states():
    print("Testing GET /states...")
    with TestClient(app) as c:
        resp = c.get("/states")
        assert resp.status_code == 200, f"GET /states failed: {resp.text}"
        data = resp.json()
        assert data["total_states"] == 3
        states = [s["name"] for s in data["states"]]
        assert "Maharashtra" in states
        assert "Karnataka" in states
        assert "Tamil Nadu" in states
        print(f"  [PASS] States endpoint returned {states}")


def test_history():
    print("Testing GET /history/Maharashtra...")
    with TestClient(app) as c:
        resp = c.get("/history/Maharashtra?weeks=24")
        assert resp.status_code == 200, f"GET /history failed: {resp.text}"
        data = resp.json()
        assert data["state"] == "Maharashtra"
        assert len(data["data"]) == 24
        sample = data["data"][-1]
        assert "Cases" in sample
        assert "preci" in sample
        assert "LAI" in sample
        assert "Temp" in sample
        assert "outbreak" in sample
        print(f"  [PASS] History returned 24 records. Last week: {sample['date_label']}, Cases: {sample['Cases']}")


def test_forecast():
    print("Testing GET /forecast/Maharashtra, Karnataka, Tamil Nadu...")
    with TestClient(app) as c:
        for state in ["Maharashtra", "Karnataka", "Tamil Nadu"]:
            resp = c.get(f"/forecast/{state}")
            assert resp.status_code == 200, f"GET /forecast/{state} failed: {resp.text}"
            data = resp.json()
            assert 0.0 <= data["window_level_probability"] <= 1.0
            assert "t+4" in data["per_lead_week"]
            assert "t+5" in data["per_lead_week"]
            assert "t+6" in data["per_lead_week"]
            assert data["risk_label"] in ["Low", "Moderate", "High"]
            assert len(data["input_window_summary"]) == 12
            print(
                f"  [PASS] Forecast for {state}: Risk={data['risk_label']} ({data['window_level_probability']*100:.1f}%), "
                f"Leads={data['per_lead_week']}, Model={data['model_used']}"
            )


def test_custom_forecast():
    print("Testing POST /forecast/custom with synthetic 12-week scenario...")
    with TestClient(app) as c:
        # Create a 12-week scenario with elevated cases in weeks 10-12
        weeks_payload = []
        for i in range(1, 13):
            weeks_payload.append({
                "week_index": i,
                "Cases": 5.0 if i < 10 else 45.0,
                "preci": 80.0 if i >= 8 else 10.0,
                "LAI": 2.5,
                "Temp": 28.5,
                "week_num": i,
                "year": 2023,
            })

        req_body = {
            "state": "Maharashtra",
            "weeks": weeks_payload
        }
        resp = c.post("/forecast/custom", json=req_body)
        assert resp.status_code == 200, f"Custom forecast failed: {resp.text}"
        data = resp.json()
        assert data["is_simulated"] is True
        assert 0.0 <= data["window_level_probability"] <= 1.0
        print(
            f"  [PASS] Custom scenario for Maharashtra: Risk={data['risk_label']} "
            f"({data['window_level_probability']*100:.1f}%), Leads={data['per_lead_week']}"
        )


def test_custom_forecast_validation_error():
    print("Testing POST /forecast/custom validation error on invalid length...")
    with TestClient(app) as c:
        req_body = {
            "state": "Maharashtra",
            "weeks": [{"week_index": 1, "Cases": 0, "preci": 0, "LAI": 0, "Temp": 25}]
        }
        resp = c.post("/forecast/custom", json=req_body)
        assert resp.status_code == 422, f"Expected 422 validation error, got {resp.status_code}"
        print(f"  [PASS] Custom forecast correctly rejected malformed length with status 422.")


def test_model_info():
    print("Testing GET /model-info/Maharashtra...")
    with TestClient(app) as c:
        resp = c.get("/model-info/Maharashtra")
        assert resp.status_code == 200, f"GET /model-info failed: {resp.text}"
        data = resp.json()
        assert data["state"] == "Maharashtra"
        assert data["test_metrics_window"]["pr_auc"] > 0
        assert len(data["baseline_comparison"]) > 0
        print(
            f"  [PASS] Model info: {data['best_config_name']} | Window PR-AUC={data['test_metrics_window']['pr_auc']}, "
            f"Recall={data['test_metrics_window']['recall']}, Precision={data['test_metrics_window']['precision']}"
        )


def test_invalid_state_404():
    print("Testing GET /forecast/NonExistentState for 404 handling...")
    with TestClient(app) as c:
        resp = c.get("/forecast/Atlantis")
        assert resp.status_code == 404
        assert "Atlantis" in resp.json()["detail"]
        print("  [PASS] Invalid state returned clean 404 with helpful message.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING BACKEND TEST SUITE")
    print("=" * 60)
    test_health()
    test_states()
    test_history()
    test_forecast()
    test_custom_forecast()
    test_custom_forecast_validation_error()
    test_model_info()
    test_invalid_state_404()
    print("=" * 60)
    print("ALL BACKEND TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
