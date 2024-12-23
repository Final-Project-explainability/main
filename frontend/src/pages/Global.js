import React from "react";
import MenuBar from "../components/MenuBar"; // נייבא את ה-MenuBar
import FeatureMetric from "../components/FeatureMetric"; // קומפוננטה לתיבת המידע השמאלית
import GraphContainer from "../components/GraphContainer"; // קומפוננטה לגרפים
import "../styles/Global.css";

const GlobalPage = () => {
  return (
    <div className="global-page">
      <MenuBar />
      <div className="main-content">
        <FeatureMetric />
        <GraphContainer />
      </div>
    </div>
  );
};

export default GlobalPage;