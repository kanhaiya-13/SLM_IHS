import React, { useState, useEffect } from "react";
import { AlertTriangle, Activity, Sliders, ShieldCheck } from "lucide-react";
import type { StateInfo, ForecastResponse, ModelInfoResponse, WeekRecord } from "./types";
import { api } from "./services/api";
import { Header } from "./components/Header";
import { RiskDashboard } from "./components/RiskDashboard";
import { HistoricalTrendChart } from "./components/HistoricalTrendChart";
import { ScenarioSimulator } from "./components/ScenarioSimulator";
import { ModelTransparencyPanel } from "./components/ModelTransparencyPanel";
import { Footer } from "./components/Footer";

export const App: React.FC = () => {
  const [states, setStates] = useState<StateInfo[]>([]);
  const [selectedState, setSelectedState] = useState<string>("Maharashtra");
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [history, setHistory] = useState<WeekRecord[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [historyWeeks, setHistoryWeeks] = useState<number>(24);

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isOnline, setIsOnline] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"dashboard" | "simulator" | "transparency">("dashboard");

  // Load states on initial mount
  useEffect(() => {
    async function init() {
      try {
        const health = await api.checkHealth();
        setIsOnline(health.status === "ok");

        const statesData = await api.getStates();
        setStates(statesData.states);
        if (statesData.states.length > 0) {
          setSelectedState(statesData.states[0].id);
        }
      } catch (err: any) {
        setIsOnline(false);
        setError("Could not connect to FastAPI backend at http://localhost:8000. Ensure uvicorn is running.");
        // Set fallback state options
        setStates([
          {
            id: "Maharashtra",
            name: "Maharashtra",
            best_config: "CI_W_NC",
            window_pr_auc: 0.847,
            window_f1: 0.8,
            window_recall: 0.906,
            window_precision: 0.716,
            historical_year_range: "2009–2022",
            total_recorded_weeks: 728,
          },
          {
            id: "Karnataka",
            name: "Karnataka",
            best_config: "CI_W_C",
            window_pr_auc: 0.202,
            window_f1: 0.057,
            window_recall: 0.056,
            window_precision: 0.059,
            historical_year_range: "2009–2022",
            total_recorded_weeks: 728,
          },
          {
            id: "Tamil Nadu",
            name: "Tamil Nadu",
            best_config: "CM_W_C",
            window_pr_auc: 0.475,
            window_f1: 0.45,
            window_recall: 0.947,
            window_precision: 0.295,
            historical_year_range: "2009–2022",
            total_recorded_weeks: 728,
          },
        ]);
      }
    }
    init();
  }, []);

  // Fetch state data whenever selectedState or historyWeeks changes
  useEffect(() => {
    if (!selectedState) return;

    let isSubscribed = true;
    async function loadStateData() {
      setIsLoading(true);
      setError(null);
      try {
        const [forecastRes, historyRes, modelInfoRes] = await Promise.all([
          api.getForecast(selectedState),
          api.getHistory(selectedState, historyWeeks),
          api.getModelInfo(selectedState),
        ]);

        if (isSubscribed) {
          setForecast(forecastRes);
          setHistory(historyRes.data);
          setModelInfo(modelInfoRes);
          setIsOnline(true);
        }
      } catch (err: any) {
        if (isSubscribed) {
          setIsOnline(false);
          setError(`Failed to fetch live data for ${selectedState}: ${err.message}`);
        }
      } finally {
        if (isSubscribed) {
          setIsLoading(false);
        }
      }
    }

    loadStateData();
    return () => {
      isSubscribed = false;
    };
  }, [selectedState, historyWeeks]);

  return (
    <div>
      <Header
        states={states}
        selectedState={selectedState}
        onSelectState={setSelectedState}
        isOnline={isOnline}
        isLoading={isLoading}
      />

      <main className="app-container">
        {/* Navigation Tabs */}
        <div className="view-nav">
          <button
            className={`view-nav-btn ${activeTab === "dashboard" ? "active" : ""}`}
            onClick={() => setActiveTab("dashboard")}
          >
            <Activity size={17} />
            Surveillance &amp; Risk Dashboard
          </button>
          <button
            className={`view-nav-btn ${activeTab === "simulator" ? "active" : ""}`}
            onClick={() => setActiveTab("simulator")}
          >
            <Sliders size={17} />
            Scenario Simulator (&ldquo;What-If&rdquo;)
          </button>
          <button
            className={`view-nav-btn ${activeTab === "transparency" ? "active" : ""}`}
            onClick={() => setActiveTab("transparency")}
          >
            <ShieldCheck size={17} />
            Model Transparency &amp; Audit
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div
            style={{
              background: "rgba(244, 63, 94, 0.12)",
              border: "1px solid rgba(244, 63, 94, 0.35)",
              borderRadius: "var(--radius-md)",
              padding: "14px 20px",
              marginBottom: "20px",
              display: "flex",
              alignItems: "center",
              gap: "10px",
              color: "#fb7185",
              fontSize: "0.86rem",
            }}
          >
            <AlertTriangle size={18} />
            <div>{error}</div>
          </div>
        )}

        {/* View 1: Surveillance Dashboard */}
        {activeTab === "dashboard" && (
          <div className="dashboard-grid">
            <RiskDashboard forecast={forecast} isLoading={isLoading} />
            <HistoricalTrendChart
              data={history}
              stateName={selectedState}
              onWeeksChange={setHistoryWeeks}
              currentWeeks={historyWeeks}
            />
          </div>
        )}

        {/* View 2: Scenario Simulator */}
        {activeTab === "simulator" && (
          <div className="dashboard-grid">
            <ScenarioSimulator
              stateName={selectedState}
              baselineForecast={forecast}
              historicalRecent12={history.slice(-12)}
            />
          </div>
        )}

        {/* View 3: Model Transparency Panel */}
        {activeTab === "transparency" && (
          <div className="dashboard-grid">
            <ModelTransparencyPanel modelInfo={modelInfo} isLoading={isLoading} />
          </div>
        )}

        <Footer />
      </main>
    </div>
  );
};

export default App;
