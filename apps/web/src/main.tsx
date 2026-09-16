import React, { lazy, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { PublicPages } from "./PublicPages";
import "./public.css";
import "./style.css";
import "./planner.css";
const PlannerApp = lazy(() => import("./PlannerApp"));
const publicPage = ["/welcome", "/privacy", "/terms", "/support"].includes(location.pathname) || ["eridani.app", "www.eridani.app"].includes(location.hostname);
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {publicPage ? <PublicPages/> : <Suspense fallback={<div className="splash">Opening Eridani…</div>}><PlannerApp/></Suspense>}
  </React.StrictMode>,
);
if (!publicPage && "serviceWorker" in navigator)
  navigator.serviceWorker.register("/sw.js").catch(() => {});
