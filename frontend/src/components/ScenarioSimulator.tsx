import React, { useState, useEffect } from "react";
import { Sliders, Play, RotateCcw, CloudRain, Sun, AlertTriangle, ArrowUpRight, ArrowDownRight } from "lucide-react";
import type { CustomWeekInput, ForecastResponse, WeekRecord } from "../types";
import { api } from "../services/api";

interface ScenarioSimulatorProps {
  stateName: string;
  baselineForecast: ForecastResponse | null;
  historicalRecent12: WeekRecord[];
}

export const ScenarioSimulator: React.FC<ScenarioSimulatorProps> = ({
  stateName,
  baselineForecast,
  historicalRecent12,
}) => {
  const [simWeeks, setSimWeeks] = useState<CustomWeekInput[]>([]);
  const [simResult, setSimResult] = useState<ForecastResponse | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [activePreset, setActivePreset] = useState<string>("baseline");

  // Initialize 12 weeks from actual historical data
  useEffect(() => {
    if (historicalRecent12 && historicalRecent12.length >= 12) {
      const init: CustomWeekInput[] = historicalRecent12.slice(-12).map((w, idx) => ({
        week_index: idx + 1,
        Cases: Number(w.Cases.toFixed(1)),
        preci: Number(w.preci.toFixed(1)),
        LAI: Number(w.LAI.toFixed(2)),
        Temp: Number(w.Temp.toFixed(1)),
        week_num: w.week,
        year: w.year,
      }));
      setSimWeeks(init);
      setSimResult(null);
      setActivePreset("baseline");
    }
  }, [stateName, historicalRecent12]);

  const handleCellChange = (index: number, field: keyof CustomWeekInput, value: number) => {
    const updated = [...simWeeks];
    updated[index] = {
      ...updated[index],
      [field]: isNaN(value) ? 0 : Math.max(0, value),
    };
    setSimWeeks(updated);
    setActivePreset("custom");
  };

  // Preset Handlers
  const applyPreset = (presetKey: string) => {
    if (!historicalRecent12 || historicalRecent12.length < 12) return;

    const base: CustomWeekInput[] = historicalRecent12.slice(-12).map((w, idx) => ({
      week_index: idx + 1,
      Cases: Number(w.Cases.toFixed(1)),
      preci: Number(w.preci.toFixed(1)),
      LAI: Number(w.LAI.toFixed(2)),
      Temp: Number(w.Temp.toFixed(1)),
      week_num: w.week,
      year: w.year,
    }));

    if (presetKey === "baseline") {
      setSimWeeks(base);
    } else if (presetKey === "monsoon_surge") {
      // Elevate rainfall and LAI in weeks 8-12
      const modified = base.map((w, i) => {
        if (i >= 7) {
          return {
            ...w,
            preci: Number((w.preci + 75.0).toFixed(1)),
            LAI: Number((w.LAI + 1.2).toFixed(2)),
            Cases: Number((w.Cases + 15.0).toFixed(1)),
          };
        }
        return w;
      });
      setSimWeeks(modified);
    } else if (presetKey === "drought") {
      // Suppress rainfall to near zero
      const modified = base.map((w) => ({
        ...w,
        preci: 0.0,
        LAI: Math.max(0.2, Number((w.LAI * 0.4).toFixed(2))),
        Cases: 0.0,
      }));
      setSimWeeks(modified);
    } else if (presetKey === "outbreak_spike") {
      // Sharp jump in cases in weeks 10, 11, 12
      const modified = base.map((w, i) => {
        if (i >= 9) {
          return {
            ...w,
            Cases: Number((w.Cases + 50.0).toFixed(1)),
            preci: Number((w.preci + 30.0).toFixed(1)),
          };
        }
        return w;
      });
      setSimWeeks(modified);
    }

    setActivePreset(presetKey);
    setSimResult(null);
  };

  const runSimulation = async () => {
    setIsSimulating(true);
    setSimError(null);
    try {
      const res = await api.postCustomForecast({
        state: stateName,
        weeks: simWeeks,
      });
      setSimResult(res);
    } catch (err: any) {
      setSimError(err.message || "Simulation failed");
    } finally {
      setIsSimulating(false);
    }
  };

  // Difference calculation
  const baselineProb = baselineForecast ? baselineForecast.window_level_probability : 0;
  const simProb = simResult ? simResult.window_level_probability : null;
  const diffPct = simProb !== null ? (simProb - baselineProb) * 100 : null;

  return (
    <div className="card">
      <div className="card-title">
        <Sliders size={20} color="#06b6d4" />
        Scenario Laboratory: &ldquo;What-If&rdquo; Outbreak Simulation
      </div>
      <p className="card-subtitle">
        Adjust weekly case counts or meteorological signals across the 12-week input window to test model sensitivity live during a viva.
      </p>

      {/* Preset Scenario Buttons */}
      <div style={{ marginBottom: "14px" }}>
        <div style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "8px" }}>
          Quick Scenario Presets:
        </div>
        <div className="preset-grid">
          <button
            className={`preset-btn ${activePreset === "baseline" ? "active" : ""}`}
            onClick={() => applyPreset("baseline")}
          >
            <RotateCcw size={16} color="#94a3b8" />
            <div>
              <div style={{ fontWeight: 600 }}>Reset Baseline</div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Actual IDSP records</div>
            </div>
          </button>

          <button
            className={`preset-btn ${activePreset === "monsoon_surge" ? "active" : ""}`}
            onClick={() => applyPreset("monsoon_surge")}
          >
            <CloudRain size={16} color="#38bdf8" />
            <div>
              <div style={{ fontWeight: 600 }}>Monsoon Surge</div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>+75mm rain &amp; vegetation</div>
            </div>
          </button>

          <button
            className={`preset-btn ${activePreset === "drought" ? "active" : ""}`}
            onClick={() => applyPreset("drought")}
          >
            <Sun size={16} color="#f59e0b" />
            <div>
              <div style={{ fontWeight: 600 }}>Severe Drought</div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>0mm rain (suppression)</div>
            </div>
          </button>

          <button
            className={`preset-btn ${activePreset === "outbreak_spike" ? "active" : ""}`}
            onClick={() => applyPreset("outbreak_spike")}
          >
            <AlertTriangle size={16} color="#f43f5e" />
            <div>
              <div style={{ fontWeight: 600 }}>Case Cluster (+50)</div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Sudden late surge</div>
            </div>
          </button>
        </div>
      </div>

      {/* 12-Week Interactive Input Table */}
      <div className="sim-table-container">
        <table className="sim-table">
          <thead>
            <tr>
              <th style={{ width: "80px" }}>Week #</th>
              <th>Reported Cases</th>
              <th>Precipitation (mm)</th>
              <th>LAI (Vegetation)</th>
              <th>Temperature (°C)</th>
            </tr>
          </thead>
          <tbody>
            {simWeeks.map((w, idx) => (
              <tr key={idx}>
                <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                  W{w.week_index} {w.week_num ? `(W${String(w.week_num).padStart(2, "0")})` : ""}
                </td>
                <td>
                  <input
                    type="number"
                    min="0"
                    step="1"
                    className="sim-input"
                    value={w.Cases}
                    onChange={(e) => handleCellChange(idx, "Cases", parseFloat(e.target.value))}
                  />
                </td>
                <td>
                  <input
                    type="number"
                    min="0"
                    step="5"
                    className="sim-input"
                    value={w.preci}
                    onChange={(e) => handleCellChange(idx, "preci", parseFloat(e.target.value))}
                  />
                </td>
                <td>
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    className="sim-input"
                    value={w.LAI}
                    onChange={(e) => handleCellChange(idx, "LAI", parseFloat(e.target.value))}
                  />
                </td>
                <td>
                  <input
                    type="number"
                    step="0.5"
                    className="sim-input"
                    value={w.Temp}
                    onChange={(e) => handleCellChange(idx, "Temp", parseFloat(e.target.value))}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Action Row & Simulation Output */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "18px", flexWrap: "wrap", gap: "16px" }}>
        <button
          className="sim-run-btn"
          onClick={runSimulation}
          disabled={isSimulating || simWeeks.length !== 12}
        >
          <Play size={16} />
          {isSimulating ? "Running Inference..." : "Run Forecast Simulation"}
        </button>

        {/* Live Diff Summary */}
        {simResult && diffPct !== null && (
          <div style={{ display: "flex", alignItems: "center", gap: "14px", flexWrap: "wrap" }}>
            <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Baseline Risk: <strong style={{ color: "var(--text-primary)" }}>{(baselineProb * 100).toFixed(1)}%</strong>
            </div>

            <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Simulated Risk:{" "}
              <strong style={{ color: simResult.risk_color === "emerald" ? "#34d399" : simResult.risk_color === "amber" ? "#fbbf24" : "#fb7185", fontSize: "1.1rem" }}>
                {(simResult.window_level_probability * 100).toFixed(1)}%
              </strong>
            </div>

            <div
              className="sim-diff-badge"
              style={{
                background: diffPct > 0 ? "rgba(244, 63, 94, 0.15)" : "rgba(16, 185, 129, 0.15)",
                color: diffPct > 0 ? "#fb7185" : "#34d399",
                border: `1px solid ${diffPct > 0 ? "rgba(244, 63, 94, 0.3)" : "rgba(16, 185, 129, 0.3)"}`,
              }}
            >
              {diffPct > 0 ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
              {diffPct > 0 ? `+${diffPct.toFixed(1)}% Risk` : `${diffPct.toFixed(1)}% Risk`}
            </div>
          </div>
        )}
      </div>

      {simError && (
        <div style={{ color: "#f43f5e", fontSize: "0.84rem", marginTop: "12px" }}>
          &bull; Error running simulation: {simError}
        </div>
      )}
    </div>
  );
};
