import React from "react";
import ReactDOM from "react-dom/client";
import { Tooltip } from "@base-ui/react/tooltip";
import { MotionConfig } from "framer-motion";
import "./styles.css";
import App from "./App";
import { warmGeneratorConnection } from "./api";

// Start the Hugging Face parent-frame quota handshake before the first render. Waiting until Generate is clicked can
// race the queue submission and silently charge the anonymous allowance even while the user is signed in.
warmGeneratorConnection();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {/* One place decides that motion is optional. `reducedMotion="user"` drops every transform and layout animation
        for anyone who has asked their system for less movement, while keeping opacity changes so nothing appears or
        disappears without warning. */}
    <MotionConfig reducedMotion="user">
      {/* The shared provider is what makes the second tooltip in a toolbar open instantly instead of waiting out the
          delay again — without it, every icon button feels sticky. */}
      <Tooltip.Provider delay={400} closeDelay={80}>
        <App />
      </Tooltip.Provider>
    </MotionConfig>
  </React.StrictMode>,
);
