import React from "react";
import { useNavigate } from "react-router-dom"; // Import useNavigate for navigation
import Card from "./Card";

const InfoPanel = ({ isLoggedIn }) => {
  const navigate = useNavigate(); // Initialize navigation

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
          onClick={() => navigate("/global")} // Navigate to the Global page
          imageClass="global-image"
          disabled={false} // Card is always enabled
        />

        {/* Personal Patient Data Card */}
        <Card
          title="Personal Patient Data"
          buttonText="GO Personal"
          onClick={() => navigate("/local")} // Navigate to the Local page
          imageClass="local-image"
          disabled={!isLoggedIn} // Disable card if the user is not logged in
          overlayText="You need to login to get access" // Overlay text shown when disabled
        />
      </div>
    </div>
  );
};

export default InfoPanel;
