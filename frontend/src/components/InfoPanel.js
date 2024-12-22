import React from "react";
import Card from "./Card";

const InfoPanel = ({ isLoggedIn }) => {
  return (
    <div className="info-panel">
      <h2>Medical AI Mortality Prediction Model</h2>
      <p>
        ICU MediClear specializes in mortality prediction for ICU patients,
        combining artificial intelligence models with explainability to help
        clinicians and enhance decision-making in critical care.
      </p>
      <div className="card-container">
        {/* Global Data Card */}
        <Card
          title="Global Data"
          buttonText="GO Global"
          onClick={() => console.log("Global button clicked")}
          imageClass="global-image"
          disabled={false}
        />

        {/* Personal Patient Data Card */}
        <Card
          title="Personal Patient Data"
          buttonText="GO Personal"
          onClick={() => console.log("Personal button clicked")}
          imageClass="local-image"
          disabled={!isLoggedIn}
          overlayText="You need to login to get access"
        />
      </div>
    </div>
  );
};

export default InfoPanel;
