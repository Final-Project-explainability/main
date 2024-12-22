import React from "react";

const Card = ({ title, buttonText, onClick, imageClass, disabled, overlayText }) => {
  return (
    <div className={`card ${disabled ? "disabled" : ""}`}>
      <div className={`card-image ${imageClass}`}></div>
      <h3>{title}</h3>
      <button onClick={onClick} disabled={disabled}>
        {buttonText}
      </button>
      {disabled && <div className="overlay">{overlayText}</div>}
    </div>
  );
};

export default Card;
