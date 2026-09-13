import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { SiteCopilot } from "./copilot";
import "./style.css";
import "./planner.css";
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <SiteCopilot><App /></SiteCopilot>
  </React.StrictMode>,
);
if ("serviceWorker" in navigator)
  navigator.serviceWorker.register("/sw.js").catch(() => {});
