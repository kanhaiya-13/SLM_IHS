import React from "react";

export const Footer: React.FC = () => {
  return (
    <footer className="footer">
      <div style={{ maxWidth: "900px", margin: "0 auto" }}>
        <p style={{ fontWeight: 600, color: "var(--text-secondary)", marginBottom: "6px" }}>
          Vector-Borne Disease Outbreak Forecasting Surveillance Prototype
        </p>
        <p style={{ fontSize: "0.76rem", lineHeight: "1.6" }}>
          Trained on historical IDSP outbreak records (2009–2022) across Maharashtra, Karnataka, and Tamil Nadu.
          This tool is designed for academic demonstration and decision-support exploration during local evaluation.
          It does not replace clinical laboratory diagnosis or formal epidemiological field investigations.
        </p>
      </div>
    </footer>
  );
};
