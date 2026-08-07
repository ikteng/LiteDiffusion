import React from "react";
import ReactDOM from "react-dom/client";
import { MotionConfig } from "framer-motion";
import "./styles.css";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {/* One place decides that motion is optional. `reducedMotion="user"` drops every transform and layout animation
        for anyone who has asked their system for less movement, while keeping opacity changes so nothing appears or
        disappears without warning. */}
    <MotionConfig reducedMotion="user">
      <App />
    </MotionConfig>
  </React.StrictMode>,
);
