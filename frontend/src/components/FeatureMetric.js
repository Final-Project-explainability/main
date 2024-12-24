import React, { useState } from "react";
import "../styles/Global.css";

const FeatureMetric = () => {
  const [isAgreementOn, setIsAgreementOn] = useState(false);

  const toggleAgreement = () => {
    setIsAgreementOn(!isAgreementOn);
  };

  return (
    <div className="feature-metric">
      <div className="toggle-container">
        <span>Agreement</span>
        <label className="switch">
          <input type="checkbox" checked={isAgreementOn} onChange={toggleAgreement} />
          <span className="slider round"></span>
        </label>
      </div>
      <h2>Feature Metric:</h2>
      <p>Placeholder for metrics or list data</p>
    </div>
  );
};

export default FeatureMetric;