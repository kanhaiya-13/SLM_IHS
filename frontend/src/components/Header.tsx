import React from "react";
import { ShieldAlert, Activity, Layers } from "lucide-react";
import type { StateInfo } from "../types";

interface HeaderProps {
  states: StateInfo[];
  selectedState: string;
  onSelectState: (state: string) => void;
  isOnline: boolean;
  isLoading: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  states,
  selectedState,
  onSelectState,
  isOnline,
  isLoading,
}) => {
  return (
    <>
      {/* Persistent Disclaimer Banner */}
      <div className="disclaimer-banner">
        <ShieldAlert size={18} className="disclaimer-icon" />
        <div>
          <strong>IDSP Surveillance Research Prototype:</strong> Trained on historical IDSP outbreak reports
          (2009–2022) for three states. Not connected to a live surveillance feed, not clinically validated, and
          not intended as the sole basis for operational public health decisions. Output reflects historical reported
          outbreak patterns subject to known reporting biases.
        </div>
      </div>

      {/* Main App Navigation Bar */}
      <header className="header">
        <div className="logo-section">
          <div className="logo-badge">
            <Activity size={24} color="#ffffff" />
          </div>
          <div className="logo-text">
            <h1>Outbreak Forecast Surveillance</h1>
            <p>PatchTST Early Warning System (4–6 Weeks Horizon)</p>
          </div>
        </div>

        {/* State Selector Buttons */}
        <div style={{ display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap" }}>
          <div className="state-tabs">
            {states.map((s) => (
              <button
                key={s.id}
                className={`state-tab-btn ${selectedState === s.id ? "active" : ""}`}
                onClick={() => onSelectState(s.id)}
                disabled={isLoading}
              >
                <Layers size={14} />
                {s.name}
              </button>
            ))}
          </div>

          {/* Backend Connection Indicator */}
          <div className="connection-status">
            <div className={`status-dot ${!isOnline ? "offline" : ""}`} />
            <span>{isOnline ? "FastAPI Backend Connected" : "Backend Offline"}</span>
          </div>
        </div>
      </header>
    </>
  );
};
