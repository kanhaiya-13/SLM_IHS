import React from "react";
import { Award, ShieldCheck, Cpu } from "lucide-react";
import type { ModelInfoResponse } from "../types";

interface ModelTransparencyPanelProps {
  modelInfo: ModelInfoResponse | null;
  isLoading: boolean;
}

export const ModelTransparencyPanel: React.FC<ModelTransparencyPanelProps> = ({
  modelInfo,
  isLoading,
}) => {
  if (isLoading || !modelInfo) {
    return (
      <div className="card" style={{ textAlign: "center", padding: "40px" }}>
        <div style={{ color: "var(--text-secondary)" }}>Loading model auditability metrics...</div>
      </div>
    );
  }

  const w = modelInfo.test_metrics_window;

  return (
    <div className="card">
      <div className="card-title">
        <ShieldCheck size={20} color="#10b981" />
        Model Transparency &amp; Rigorous Evaluation
      </div>
      <p className="card-subtitle">
        Empirical performance metrics on the chronologically held-out test split (2021–2022) with zero data leakage.
      </p>

      {/* Plain English Summary Callout */}
      <div
        style={{
          background: "linear-gradient(90deg, rgba(6, 182, 212, 0.12) 0%, rgba(59, 130, 246, 0.08) 100%)",
          border: "1px solid rgba(6, 182, 212, 0.3)",
          borderRadius: "var(--radius-md)",
          padding: "16px 20px",
          marginBottom: "24px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: 700, color: "var(--accent-cyan)", marginBottom: "4px" }}>
          <Award size={18} />
          Surveillance Performance Verdict ({modelInfo.state}):
        </div>
        <p style={{ fontSize: "0.9rem", lineHeight: "1.6", color: "#f8fafc" }}>
          {modelInfo.plain_english_summary}
        </p>
      </div>

      <div className="dashboard-grid-2col">
        {/* Left Column: Test Set Metrics Breakdown */}
        <div>
          <div style={{ fontSize: "0.88rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "4px" }}>
            Held-Out Test Set Performance (2021–2022)
          </div>
          <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: "12px" }}>
            Evaluated on sliding windows across 104 held-out weeks.
          </p>

          <table className="metrics-table">
            <thead>
              <tr>
                <th>Horizon</th>
                <th>Recall</th>
                <th>Precision</th>
                <th>F1 Score</th>
                <th>PR-AUC</th>
                <th>ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              <tr className="best-row">
                <td style={{ color: "var(--accent-cyan)" }}>Window (t+4..t+6)</td>
                <td>{(w.recall * 100).toFixed(1)}%</td>
                <td>{(w.precision * 100).toFixed(1)}%</td>
                <td>{w.f1.toFixed(3)}</td>
                <td style={{ color: "var(--accent-cyan)", fontWeight: 700 }}>{w.pr_auc.toFixed(3)}</td>
                <td>{w.roc_auc.toFixed(3)}</td>
              </tr>
              {(["t+4", "t+5", "t+6"] as const).map((lead) => {
                const lm = modelInfo.test_metrics_leads[lead];
                return (
                  <tr key={lead}>
                    <td>Lead {lead}</td>
                    <td>{(lm.recall * 100).toFixed(1)}%</td>
                    <td>{(lm.precision * 100).toFixed(1)}%</td>
                    <td>{lm.f1.toFixed(3)}</td>
                    <td>{lm.pr_auc.toFixed(3)}</td>
                    <td>{lm.roc_auc.toFixed(3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Right Column: Benchmark Comparison against Baselines */}
        <div>
          <div style={{ fontSize: "0.88rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "4px" }}>
            Benchmark Comparison (vs Baselines)
          </div>
          <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: "12px" }}>
            Comparison against LSTM and naive persistence on the same test split.
          </p>

          <table className="metrics-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>PR-AUC</th>
                <th>F1 Score</th>
                <th>Recall</th>
                <th>Precision</th>
              </tr>
            </thead>
            <tbody>
              {modelInfo.baseline_comparison.map((b, idx) => (
                <tr key={idx} className={b.is_selected_best ? "best-row" : ""}>
                  <td>
                    {b.model} {b.is_selected_best ? <span style={{ color: "var(--accent-cyan)", fontSize: "0.75rem" }}>(Best)</span> : ""}
                  </td>
                  <td style={{ fontWeight: b.is_selected_best ? 700 : 400 }}>{b.pr_auc.toFixed(3)}</td>
                  <td>{b.f1.toFixed(3)}</td>
                  <td>{(b.recall * 100).toFixed(1)}%</td>
                  <td>{(b.precision * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Architectural Configuration Box */}
      <div style={{ marginTop: "24px", padding: "16px 20px", background: "var(--bg-surface-raised)", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: 600, color: "var(--text-primary)", marginBottom: "6px", fontSize: "0.86rem" }}>
          <Cpu size={16} color="#38bdf8" />
          Trained Architecture &amp; Feature Strategy:
        </div>
        <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: "1.6" }}>
          <div>&bull; <strong>Configuration:</strong> {modelInfo.best_config_name}</div>
          <div>&bull; <strong>Attention Design:</strong> {modelInfo.channel_strategy} (Each channel processed via isolated multi-head attention blocks)</div>
          <div>&bull; <strong>Input Channels ({modelInfo.active_channels.length}):</strong> {modelInfo.active_channels.join(", ")}</div>
          <div>&bull; <strong>Domain Insight:</strong> {modelInfo.interpretation}</div>
        </div>
      </div>
    </div>
  );
};
