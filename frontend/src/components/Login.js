import React, { useState } from 'react';

// Login component to handle user login
const Login = ({ onLogin }) => {
    // State to manage username and password inputs
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');

    // Handle form submission
    const handleSubmit = (e) => {
        e.preventDefault(); // Prevent default form submission behavior
        if (username && password) {
            onLogin(username); // Trigger the login function with the username
        } else {
            alert('Please fill in both fields'); // Alert user if fields are empty
        }
    };

    return (
        <div>
            <h1>ICU MediClear</h1>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>Username:</label>
                    <input
                        type="text"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)} // Update username state
                    />
                </div>
                <div>
                    <label>Password:</label>
                    <input
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)} // Update password state
                    />
                </div>
                <button type="submit">Login</button>
            </form>
        </div>
    );
};

export default Login; // Export the Login component
