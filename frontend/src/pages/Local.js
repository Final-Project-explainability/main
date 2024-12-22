import React from "react";

const Local = () => {
  return (
    <div className="local-page">
      <h1>Personal Patient Data</h1>
      <p>
        Welcome to the Personal Patient Data page. Here you can view and analyze
        patient-specific data and insights.
      </p>
      <div className="patient-data-container">
        <h2>Example Patient Details</h2>
        <ul>
          <li><strong>Name:</strong> John Doe</li>
          <li><strong>Age:</strong> 45</li>
          <li><strong>Condition:</strong> Stable</li>
          <li><strong>Last Visit:</strong> Dec 20, 2024</li>
        </ul>
      </div>
    </div>
  );
};

export default Local;
