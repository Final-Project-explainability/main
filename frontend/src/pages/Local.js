import React from "react";
import MenuBar from "../components/MenuBar"; // ייבוא התפריט העליון
import "../styles/Local.css"; // ייבוא עיצוב מותאם לדף Local

const Local = () => {
  return (
    <div className="local-page">
      {/* תפריט עליון */}
      <MenuBar />

      {/* תוכן הדף */}
      <main className="local-content">
        <h1>Personal Patient Data</h1>
        <p>
          Welcome to the Personal Patient Data page. Here you can view and
          analyze patient-specific data and insights.
        </p>
        <div className="patient-data-container">
          <h2>Example Patient Details</h2>
          <ul>
            <li>
              <strong>Name:</strong> John Doe
            </li>
            <li>
              <strong>Age:</strong> 45
            </li>
            <li>
              <strong>Condition:</strong> Stable
            </li>
            <li>
              <strong>Last Visit:</strong> Dec 20, 2024
            </li>
          </ul>
        </div>
      </main>
    </div>
  );
};

export default Local;