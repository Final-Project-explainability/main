import React, { useState } from "react";
import "../styles/MenuBar.css"; // נייבא את ה-CSS של התפריט

const MenuBar = ({ isLoggedIn }) => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const toggleDropdown = () => {
    setIsDropdownOpen(!isDropdownOpen);
  };

  return (
    <div className="menu-bar">
      {/* צד שמאל */}
      <div className="menu-left">
        <button className="menu-icon">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="icon-svg"
          >
            <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z" />
          </svg>
        </button>
      </div>

      {/* מרכז */}
      {isLoggedIn && (
        <div className="menu-center">
          <input
            type="text"
            placeholder="Search..."
            className="search-input"
          />
        </div>
      )}

      {/* לוגו חצי דהוי */}
      <div className="menu-center logo-center">
        <span className="logo-text">ICU MediClear</span>
      </div>

      {/* אייקון מימין ללוגו */}
      <div className="logo-icon"></div>

      {/* צד ימין */}
      <div className="menu-right">
        <button className="user-menu" onClick={toggleDropdown}>
          <span>Hello, Dr. Israel</span>
          <div className="user-avatar-menu"></div>
        </button>
        {isDropdownOpen && (
          <div className="user-dropdown">
            <ul>
              <li>Profile</li>
              <li>Settings</li>
              <li>Logout</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default MenuBar;