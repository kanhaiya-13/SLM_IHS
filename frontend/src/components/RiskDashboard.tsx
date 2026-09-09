import React, { useState } from "react";
import { AlertCircle, Calendar, ChevronDown, ChevronUp, Eye, Info, ShieldCheck } from "lucide-react";
import type { ForecastResponse } from "../types";

interface RiskDashboardProps {
  forecast: ForecastResponse | null;
  isLoading: boolean;
}

export const RiskDashboard: React.FC<RiskDashboardProps> = ({ forecast, isLoading }) => {
  const [showInputSummary, setShowInputSummary] = useState(false);

  if (isLoading || !forecast) {
    return (
      <div className="card" style={{ textAlign: "center", padding: "60px 20px" }}>
        <div style={{ color: "var(--text-secondary)", fontSize: "1rem" }}>
          Generating 4–6 week ahead forecast from historical snapshot...
        </div>
      </div>
    );
  }

  const probPercent = (forecast.window_level_probability * 100).toFixed(1);
  const colorKey = forecast.risk_color; // 'emerald', 'amber', or 'rose'

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <div className="card-title">
            <AlertCircle size={20} color={colorKey === "emerald" ? "#10b981" : colorKey === "amber" ? "#f59e0b" : "#f43f5e"} />
            Outbreak Risk Assessment (4–6 Weeks Ahead)
          </div>
          <div className="card-subtitle">
            Target State: <strong style={{ color: "var(--text-primary)" }}>{forecast.state}</strong> &bull; Model: {forecast.model_used}
          </div>
        </div>

        {/* Snapshot & Horizon Badges */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "4px" }}>
          <div className="snapshot-tag">
            <Calendar size={13} />
            <span>{forecast.input_date_range} (Input Window)</span>
          </div>
          <div className="snapshot-tag" style={{ color: "var(--accent-cyan)", borderColor: "rgba(6, 182, 212, 0.3)" }}>
            <span>&rarr; {forecast.forecast_lead_range}</span>
          </div>
        </div>
      </div>

      <div className="dashboard-grid-2col" style={{ alignItems: "center", marginTop: "12px" }}>
        {/* Left Column: Big Risk Display */}
        <div className="risk-meter-container">
          <div className={`risk-badge-large ${colorKey}`}>
            {colorKey === "emerald" && <ShieldCheck size={28} />}
            {colorKey === "amber" && <Info size={28} />}
            {colorKey === "rose" && <AlertCircle size={28} />}
            {forecast.risk_label} RISK
          </div>

          <div className="risk-prob-text">
            {probPercent}<span>%</span>
          </div>
          <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", maxWidth: "340px" }}>
            Estimated probability of &ge;1 confirmed outbreak in weeks t+4 through t+6
          </div>

          {/* Actionable Health Officer Guidance */}
          <div
            style={{
              marginTop: "20px",
              padding: "14px 18px",
              background: "var(--bg-surface-raised)",
              borderRadius: "var(--radius-md)",
              border: `1px solid var(--color-${colorKey}-border)`,
              fontSize: "0.85rem",
              lineHeight: "1.5",
              textAlign: "left",
              color: "var(--text-primary)",
            }}
          >
            <strong style={{ color: colorKey === "emerald" ? "#34d399" : colorKey === "amber" ? "#fbbf24" : "#fb7185" }}>
              Surveillance Directive:
            </strong>{" "}
            {forecast.risk_description}
          </div>
        </div>

        {/* Right Column: Per-Lead-Week Confidence Curve */}
        <div>
          <div style={{ fontSize: "0.88rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "4px" }}>
            Forecast Horizon Progression (Confidence by Lead Week)
          </div>
          <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "16px" }}>
            Confidence varies across time horizons as meteorological covariates propagate into vector breeding cycles.
          </p>

          <div className="leads-grid">
            {(["t+4", "t+5", "t+6"] as const).map((lead) => {
              const p = forecast.per_lead_week[lead] ?? 0;
              const pPct = (p * 100).toFixed(1);
              const barColor = p < 0.35 ? "#10b981" : p < 0.65 ? "#f59e0b" : "#f43f5e";

              return (
                <div key={lead} className="lead-card">
                  <div className="lead-label">Lead {lead}</div>
                  <div className="lead-prob" style={{ color: barColor }}>
                    {pPct}%
                  </div>
                  <div className="lead-bar-bg">
                    <div
                      className="lead-bar-fill"
                      style={{
                        width: `${Math.min(Math.max(p * 100, 4), 100)}%`,
                        backgroundColor: barColor,
                      }}
                    />
                  </div>
                  <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "6px" }}>
                    {lead === "t+4" ? "4 Weeks Out" : lead === "t+5" ? "5 Weeks Out" : "6 Weeks Out"}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Threshold Legend */}
          <div style={{ marginTop: "20px", fontSize: "0.76rem", color: "var(--text-muted)", lineHeight: "1.6" }}>
            <div>&bull; <strong style={{ color: "#34d399" }}>Low Risk (P &lt; 35%):</strong> Baseline sentinel surveillance</div>
            <div>&bull; <strong style={{ color: "#fbbf24" }}>Moderate Risk (35% &le; P &lt; 65%):</strong> Pre-position larvicide inventory</div>
            <div>&bull; <strong style={{ color: "#fb7185" }}>High Risk (P &ge; 65%):</strong> Preemptive district mobilization</div>
          </div>
        </div>
      </div>

      {/* Accordion: 12-Week Input Window Inspector */}
      <div style={{ marginTop: "24px", borderTop: "1px solid var(--border-subtle)", paddingTop: "16px" }}>
        <button
          onClick={() => setShowInputSummary(!showInputSummary)}
          style={{
            background: "transparent",
            border: "none",
            color: "var(--accent-cyan)",
            fontSize: "0.85rem",
            fontWeight: 500,
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "6px",
          }}
        >
          <Eye size={15} />
          {showInputSummary ? "Hide 12-Week Input Feature Breakdown" : "Inspect 12-Week Model Input Window (Transparency)"}
          {showInputSummary ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </button>

        {showInputSummary && (
          <div style={{ marginTop: "14px" }}>
            <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: "10px" }}>
              Exact multi-channel sequence ingested by the PatchTST model to compute this forecast:
            </p>
            <div className="sim-table-container">
              <table className="sim-table">
                <thead>
                  <tr>
                    <th>Week Label</th>
                    <th>Reported Cases</th>
                    <th>Precipitation (mm)</th>
                    <th>LAI (Vegetation)</th>
                    <th>Temperature (°C)</th>
                    <th>IDSP Outbreak</th>
                  </tr>
                </thead>
                <tbody>
                  {forecast.input_window_summary.map((w, idx) => (
                    <tr key={idx}>
                      <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)", fontWeight: 500 }}>
                        {w.date_label}
                      </td>
                      <td style={{ fontFamily: "var(--font-mono)", color: w.Cases > 0 ? "#f43f5e" : "inherit" }}>
                        {w.Cases.toFixed(1)}
                      </td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{w.preci.toFixed(1)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{w.LAI.toFixed(2)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{w.Temp.toFixed(1)}</td>
                      <td>
                        {w.outbreak ? (
                          <span style={{ color: "#f43f5e", fontWeight: 600, fontSize: "0.75rem" }}>CONFIRMED</span>
                        ) : (
                          <span style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>None</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
