import { createRoot } from "react-dom/client";
import { RemoteaProvider, useRemoteaActions } from "../../src/remotea-context";
import { setDurableScope } from "../../src/api";
import { RunPanel } from "../../src/RunPanel";
import { useMaxWidth } from "../../src/ux";
import "../../src/theme.css";
import "../../src/shell.css";
setDurableScope("panel-fixture:personal");
function Fixture() {
  const actions = useRemoteaActions();
  const phone = useMaxWidth(700);
  return <div className="app-shell"><div className="main-shell">
    <div style={{ height: "var(--topbar-height)", flex: "none", padding: 16 }}>Run panel browser fixture</div>
    <div className="workspace">
      <main className="content"><button className="btn" onClick={() => actions.openRun("11111111-1111-4111-8111-111111111111")}>Open run</button><p>Page content remains usable when the run is hidden.</p></main>
      <RunPanel phone={phone}/>
    </div>
  </div></div>;
}
createRoot(document.getElementById("root")!).render(<RemoteaProvider><Fixture/></RemoteaProvider>);
