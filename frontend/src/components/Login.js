
import React, { useState } from 'react';

const Login = ({ onLogin }) => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (username && password) {
            onLogin(username);
        } else {
            alert('Please fill in both fields');
        }
    };

    return (
        <div className="login-section">
            <h1 className="login-title">ICU MediClear</h1>
             <div class="logo-image-container"></div>
            <div className="login-background">
                <form onSubmit={handleSubmit} className="login-form">
                    <div className="form-group">
                        <label>Username</label>
                        <input
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            placeholder="Enter your username"
                        />
                    </div>
                    <div className="form-group">
                        <label>Password</label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="Enter your password"
                        />
                    </div>
                    <div className="forgot-password">
                        <a href="#">Forgot password? Click here</a>
                    </div>
                    <button type="submit" className="login-button">Login</button>
                </form>
            </div>
            <div className="register-section">
                <span>Don't have an account yet?</span>{' '}
                <a href="#" className="register-link">Register</a>
            </div>
        </div>
    );
};

export default Login;
