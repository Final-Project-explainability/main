import React from 'react';

// Dashboard component to display user information and options
const Dashboard = ({ user, onLogout }) => {
    return (
        <div>
            <h1>Welcome, {user}!</h1>
            <p>Here is your information:</p>
            <ul>
                <li><strong>ID:</strong> 123456</li>
                <li><strong>Specialty:</strong> Emergency Physician</li>
            </ul>
            <div>
                <button onClick={() => alert('Global Data Accessed!')}>Global Data</button>
                <button onClick={() => alert('Personal Patient Data Accessed!')}>Personal Patient Data</button>
            </div>
            <button onClick={onLogout}>Logout</button>
        </div>
    );
};

export default Dashboard;
