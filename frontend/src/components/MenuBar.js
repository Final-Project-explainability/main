import React from "react";
import { FaBars, FaHome, FaSearch } from "react-icons/fa"; // React icons

const MenuBar = () => {
    return (
        <div className="menu-bar">
            {/* Left section - Hamburger menu */}
            <button className="menu-icon">
                <FaBars />
            </button>

            {/* Center section - Logo/Home */}
            <div className="menu-logo">
                <FaHome className="home-icon" />
                ICU MediClear
            </div>

            {/* Right section - Search */}
            <div className="search-container">
                <input
                    type="text"
                    placeholder="Search"
                    className="search-input"
                />
                <button className="search-icon">
                    <FaSearch />
                </button>
            </div>
        </div>
    );
};

export default MenuBar;
