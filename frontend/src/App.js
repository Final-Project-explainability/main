import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./components/Dashboard";
import Login from "./components/Login";
import InfoPanel from "./components/InfoPanel";
import Global from "./pages/Global";
import Local from "./pages/Local"; // Import Local page
import "./styles.css";

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false); // State to track login status
  const [userDetails, setUserDetails] = useState(null); // State to store user details

  // Handle user login
  const handleLogin = (username) => {
    setUserDetails({
      name: username,
      licenseId: "123465",
      specialty: "ER Physician",
    });
    setIsLoggedIn(true);
  };

  // Handle user logout
  const handleLogout = () => {
    setUserDetails(null);
    setIsLoggedIn(false);
  };

  return (
    <Router>
      <Routes>
        {/* Main page */}
        <Route
          path="/"
          element={
            <div className="main-container">
              <div className="left-panel">
                {/* Show Dashboard if logged in, otherwise show Login */}
                {isLoggedIn ? (
                  <Dashboard user={userDetails} onLogout={handleLogout} />
                ) : (
                  <Login onLogin={handleLogin} />
                )}
              </div>
              <div className="right-panel">
                {/* InfoPanel for global and local navigation */}
                <InfoPanel isLoggedIn={isLoggedIn} />
              </div>
            </div>
          }
        />

        {/* Global page route */}
        <Route path="/global" element={<Global />} />

        {/* Local page route */}
        <Route path="/local" element={<Local />} />
      </Routes>
    </Router>
  );
}

export default App;
