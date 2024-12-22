import React from "react";

const Card = ({ title, buttonText, onClick, disabled, overlayText, imageClass }) => {
  return (
      <div className={`card ${disabled ? "disabled" : ""}`}>
          <h3>{title}</h3>
          <div className={`card-image ${imageClass}`}></div>
          <div className="card-content">

              <button onClick={onClick} disabled={disabled}>
                  {buttonText}
              </button>
          </div>
          {disabled && overlayText && <div className="overlay">{overlayText}</div>}
      </div>
  );
};

export default Card;
