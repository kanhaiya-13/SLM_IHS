import React, { useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceDot,
} from "recharts";
import { TrendingUp, CloudRain, Thermometer, Leaf } from "lucide-react";
import type { WeekRecord } from "../types";

interface HistoricalTrendChartProps {
  data: WeekRecord[];
  stateName: string;
  onWeeksChange: (weeks: number) => void;
  currentWeeks: number;
}

export const HistoricalTrendChart: React.FC<HistoricalTrendChartProps> = ({
  data,
  stateName,
  onWeeksChange,
  currentWeeks,
}) => {
  const [showPreci, setShowPreci] = useState(true);
  const [showTemp, setShowTemp] = useState(true);
  const [showLai, setShowLai] = useState(false);

  // Custom rich tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const item: WeekRecord = payload[0].payload;
      return (
        <div
          style={{
            background: "rgba(15, 22, 36, 0.95)",
            border: "1px solid rgba(255, 255, 255, 0.15)",
            borderRadius: "8px",
            padding: "12px 16px",
            backdropFilter: "blur(8px)",
            boxShadow: "0 10px 25px rgba(0,0,0,0.5)",
            fontSize: "0.82rem",
            color: "#f1f5f9",
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: "8px", color: "var(--accent-cyan)", borderBottom: "1px solid rgba(255,255,255,0.1)", paddingBottom: "4px" }}>
            {item.date_label} (Year {item.year}, Week {item.week})
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "16px", margin: "3px 0" }}>
            <span style={{ color: "#fb7185" }}>Reported Cases:</span>
            <strong style={{ fontFamily: "var(--font-mono)" }}>{item.Cases.toFixed(1)}</strong>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "16px", margin: "3px 0" }}>
            <span style={{ color: "#38bdf8" }}>Precipitation:</span>
            <strong style={{ fontFamily: "var(--font-mono)" }}>{item.preci.toFixed(1)} mm</strong>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "16px", margin: "3px 0" }}>
            <span style={{ color: "#fbbf24" }}>Temperature:</span>
            <strong style={{ fontFamily: "var(--font-mono)" }}>{item.Temp.toFixed(1)} °C</strong>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "16px", margin: "3px 0" }}>
            <span style={{ color: "#4ade80" }}>Leaf Area Index (LAI):</span>
            <strong style={{ fontFamily: "var(--font-mono)" }}>{item.LAI.toFixed(2)}</strong>
          </div>
          <div style={{ marginTop: "6px", paddingTop: "4px", borderTop: "1px solid rgba(255,255,255,0.08)" }}>
            {item.outbreak ? (
              <span style={{ color: "#f43f5e", fontWeight: 700, fontSize: "0.76rem" }}>
                &bull; IDSP Confirmed Outbreak Week
              </span>
            ) : (
              <span style={{ color: "var(--text-muted)", fontSize: "0.76rem" }}>
                &bull; No confirmed outbreak reported
              </span>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="card">
      <div className="chart-controls">
        <div>
          <div className="card-title" style={{ marginBottom: "2px" }}>
            <TrendingUp size={20} color="#38bdf8" />
            Surveillance &amp; Meteorological History: {stateName}
          </div>
          <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            Vector-borne case counts overlaid with monsoon precipitation and temperature drivers
          </p>
        </div>

        {/* Range & Variable Filter Controls */}
        <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
          {/* Channel Toggles */}
          <div className="pill-group">
            <button
              className={`pill-btn ${showPreci ? "active" : ""}`}
              onClick={() => setShowPreci(!showPreci)}
              title="Toggle Rainfall"
            >
              <CloudRain size={12} style={{ display: "inline", marginRight: "4px" }} />
              Rainfall
            </button>
            <button
              className={`pill-btn ${showTemp ? "active" : ""}`}
              onClick={() => setShowTemp(!showTemp)}
              title="Toggle Temperature"
            >
              <Thermometer size={12} style={{ display: "inline", marginRight: "4px" }} />
              Temp
            </button>
            <button
              className={`pill-btn ${showLai ? "active" : ""}`}
              onClick={() => setShowLai(!showLai)}
              title="Toggle LAI"
            >
              <Leaf size={12} style={{ display: "inline", marginRight: "4px" }} />
              LAI
            </button>
          </div>

          {/* Time Window Selector */}
          <div className="pill-group">
            {[12, 24, 52].map((w) => (
              <button
                key={w}
                className={`pill-btn ${currentWeeks === w ? "active" : ""}`}
                onClick={() => onWeeksChange(w)}
              >
                {w}W
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Chart Canvas */}
      <div style={{ width: "100%", height: "360px", marginTop: "16px" }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.06)" />
            
            <XAxis
              dataKey="date_label"
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              interval="preserveStartEnd"
            />
            
            {/* Left Y-Axis: Cases */}
            <YAxis
              yAxisId="left"
              stroke="#f43f5e"
              fontSize={11}
              tickLine={false}
              domain={[0, "auto"]}
              label={{ value: "Reported Cases", angle: -90, position: "insideLeft", fill: "#f43f5e", fontSize: 10, offset: 15 }}
            />

            {/* Right Y-Axis: Weather */}
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#38bdf8"
              fontSize={11}
              tickLine={false}
              domain={[0, "auto"]}
              label={{ value: "Precipitation (mm) / Temp (°C)", angle: 90, position: "insideRight", fill: "#38bdf8", fontSize: 10, offset: 15 }}
            />

            <Tooltip content={<CustomTooltip />} />
            
            <Legend
              wrapperStyle={{ paddingTop: "12px", fontSize: "0.78rem" }}
              iconType="circle"
            />

            {/* Precipitation Bar */}
            {showPreci && (
              <Bar
                yAxisId="right"
                dataKey="preci"
                name="Precipitation (mm)"
                fill="#0284c7"
                opacity={0.35}
                radius={[2, 2, 0, 0]}
              />
            )}

            {/* Temperature Line */}
            {showTemp && (
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="Temp"
                name="Temperature (°C)"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
              />
            )}

            {/* LAI Line */}
            {showLai && (
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="LAI"
                name="LAI (Vegetation)"
                stroke="#10b981"
                strokeWidth={1.5}
                strokeDasharray="4 4"
                dot={false}
              />
            )}

            {/* Outbreak Cases Area */}
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="Cases"
              name="Vector-Borne Cases"
              stroke="#f43f5e"
              strokeWidth={2.5}
              fill="url(#caseGradient)"
            />

            <defs>
              <linearGradient id="caseGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.0} />
              </linearGradient>
            </defs>

            {/* Outbreak Markers */}
            {data.map((entry, index) =>
              entry.outbreak ? (
                <ReferenceDot
                  key={`dot-${index}`}
                  yAxisId="left"
                  x={entry.date_label}
                  y={entry.Cases}
                  r={5}
                  fill="#f43f5e"
                  stroke="#ffffff"
                  strokeWidth={1.5}
                />
              ) : null
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "8px", fontSize: "0.75rem", color: "var(--text-muted)" }}>
        <span>&bull; Red dots indicate IDSP confirmed outbreak weeks (Cases &gt; 0)</span>
        <span>Weather covariates averaged across reporting districts</span>
      </div>
    </div>
  );
};
