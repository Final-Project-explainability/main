import React, { useRef, useState } from "react";
import "../styles/Global.css";

const GraphContainer = () => {
  const graphRef = useRef(); // הפניה לתיבת הגרפים
  const [isFullscreen, setIsFullscreen] = useState(false); // מעקב אחרי מצב המסך
  const [activeGraph, setActiveGraph] = useState("Row"); // מצב הכפתור הפעיל

  // פונקציה להפעלת מסך מלא
  const handleFullscreen = () => {
    if (!isFullscreen) {
      if (graphRef.current.requestFullscreen) {
        graphRef.current.requestFullscreen();
      } else if (graphRef.current.webkitRequestFullscreen) {
        // Safari
        graphRef.current.webkitRequestFullscreen();
      } else if (graphRef.current.msRequestFullscreen) {
        // IE/Edge
        graphRef.current.msRequestFullscreen();
      }
      setIsFullscreen(true); // עדכון מצב למסך מלא
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      } else if (document.webkitExitFullscreen) {
        // Safari
        document.webkitExitFullscreen();
      } else if (document.msExitFullscreen) {
        // IE/Edge
        document.msExitFullscreen();
      }
      setIsFullscreen(false); // חזרה ממצב מסך מלא
    }
  };

  // פונקציה לטיפול בלחיצה על כפתור גרף
  const handleGraphChange = (graphType) => {
    setActiveGraph(graphType);
  };

  return (
    <div className="graph-container" ref={graphRef}>
      {/* שורת כפתורים */}
      <div className="graph-buttons-container">
        {/* כפתורי גרפים בצד שמאל */}
        <div className="graph-buttons">
          <button
            className={`graph-button ${activeGraph === "Row" ? "active" : ""}`}
            onClick={() => handleGraphChange("Row")}
          >
            Row
          </button>
          <button
            className={`graph-button ${activeGraph === "Radar" ? "active" : ""}`}
            onClick={() => handleGraphChange("Radar")}
          >
            Radar
          </button>
          <button
            className={`graph-button ${activeGraph === "Scatterplot" ? "active" : ""}`}
            onClick={() => handleGraphChange("Scatterplot")}
          >
            Scatterplot
          </button>
        </div>

        {/* כפתור מסך מלא בצד ימין */}
        <button className="fullscreen-button" onClick={handleFullscreen}>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth="2"
            stroke="currentColor"
            className="fullscreen-icon"
          >
            {isFullscreen ? (
              // אייקון יציאה ממסך מלא
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M4 10v-3a1 1 0 011-1h3m7 0h3a1 1 0 011 1v3m0 7v3a1 1 0 01-1 1h-3m-7 0H5a1 1 0 01-1-1v-3"
              />
            ) : (
              // אייקון כניסה למסך מלא
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M8 3H5a2 2 0 00-2 2v3m0 8v3a2 2 0 002 2h3m8 0h3a2 2 0 002-2v-3m0-8V5a2 2 0 00-2-2h-3"
              />
            )}
          </svg>
          {isFullscreen ? "Exit Full Screen" : "Full Screen"}
        </button>
      </div>

      {/* אזור הגרף */}
      <h2>Graph Placeholder</h2>
      <p>This area will display the {activeGraph} graph.</p>
    </div>
  );
};

export default GraphContainer;
