import React from "react";

const Dashboard = ({ user, onLogout }) => {
    return (
        <div className="user-profile">
            {/* User Avatar Section */}
            <div className="user-avatar"></div>

            {/* Separator */}
            <div className="separator"></div>

            {/* User Details */}
            <div className="user-details">
                <h2>Hello, Dr. {user.name}!</h2>
                <p>
                    <strong>Medical License ID:</strong> {user.licenseId}
                </p>
                <p>
                    <strong>Medical Specialties:</strong> {user.specialty}
                </p>
            </div>

            {/* Separator */}
            <div className="separator"></div>

            {/* Action Buttons */}
            <div className="action-buttons">
                <button className="action-button">Patient List</button>
                <button className="action-button">Personal Area</button>
                <button className="logout-button" onClick={onLogout}>
                    Logout
                </button>
            </div>
        </div>
    );
};

export default Dashboard;
