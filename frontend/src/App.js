import React, { useState } from "react";
import Login from "./components/Login";
import Dashboard from "./components/Dashboard"; // Import Dashboard
import InfoPanel from "./components/InfoPanel"; // InfoPanel remains in App
import "./styles.css";

function App() {
    // State to manage login status and user details
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [userDetails, setUserDetails] = useState(null);

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
        <div className="main-container">
            <div className="left-panel">
                {isLoggedIn ? (
                    <Dashboard user={userDetails} onLogout={handleLogout} />
                ) : (
                    <Login onLogin={handleLogin} />
                )}
            </div>
            <div className="right-panel">
                <InfoPanel isLoggedIn={isLoggedIn} />
            </div>
        </div>
    );
}

export default App;
