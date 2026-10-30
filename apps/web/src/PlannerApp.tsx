import "./ux.css";
import App from "./App";
import { SiteCopilot } from "./copilot";
import { RemoteaProvider } from "./remotea-context";
export default function PlannerApp() { return <SiteCopilot><RemoteaProvider><App/></RemoteaProvider></SiteCopilot>; }
