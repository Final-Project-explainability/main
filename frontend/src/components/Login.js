import React, { useState } from 'react';
import './styles.css'; // Import the CSS file

const Login = ({ onLogin }) => {
    // State for managing username and password inputs
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');

    // Handle form submission
    const handleSubmit = (e) => {
        e.preventDefault(); // Prevent default form behavior
        if (username && password) {
            onLogin(username); // Trigger login callback
        } else {
            alert('Please fill in both fields'); // Alert if fields are empty
        }
    };

    return (
        <div className="login-container">
            <div className="login-box">
                <h1 className="login-title">ICU MediClear</h1>
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label>Username:</label>
                        <input
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            placeholder="Enter your username"
                        />
                    </div>
                    <div className="form-group">
                        <label>Password:</label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="Enter your password"
                        />
                    </div>
                    <button type="submit" className="login-button">
                        Login
                    </button>
                </form>
                <div className="extra-links">
                    <a href="#" className="forgot-password">Forgot password?</a>
                    <a href="#" className="register-link">Register</a>
                </div>
            </div>
        </div>
    );
};

export default Login;
