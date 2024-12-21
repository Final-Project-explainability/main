import React, { useState } from 'react';
import Login from './components/Login';
import Dashboard from './components/Dashboard'; // Import the Dashboard component

function App() {
    // State to manage the logged-in user
    const [user, setUser] = useState(null);

    // Handle user login
    const handleLogin = (username) => {
        setUser(username); // Save the username after login
    };

    // Handle user logout
    const handleLogout = () => {
        setUser(null); // Clear the user on logout
    };

    return (
        <div>
            {/* Render Dashboard if user is logged in, otherwise show Login page */}
            {user ? (
                <Dashboard user={user} onLogout={handleLogout} />
            ) : (
                <Login onLogin={handleLogin} />
            )}
        </div>
    );
}

export default App; // Export the App component
