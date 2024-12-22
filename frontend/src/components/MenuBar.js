
import React from "react";
import { FaBars, FaHome, FaSearch } from "react-icons/fa";

const MenuBar = () => {
    return (
        <div className="menu-bar">
            <button className="menu-icon">
                <FaBars />
            </button>
            <div className="menu-logo">
                <FaHome className="home-icon" />
                ICU MediClear
            </div>
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
