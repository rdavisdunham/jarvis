import { useState } from "react";
import { Check, Plus, X } from "lucide-react";
import { Dialog } from "./ux";
import { useStructureActions } from "./structure-actions";
import { addNode, childKinds, editNode, removeNode, treeImpact, TREE_KIND_LABEL, walkTree,
  type OrganizationProposal, type TreeKind, type TreeNode } from "./setup-model";
import "./details.css";

/** Set up with Dani: review the proposed spaces, areas, goals and projects, then create them in one step.
 * Edits re-preview before applying, so the backend only ever applies a tree it validated. */
export function SetupTreeReview({ proposal: initial, onClose, onKeepTalking, onByHand, onApplied }: {
  proposal: OrganizationProposal; onClose: () => void; onKeepTalking: () => void; onByHand: () => void;
  onApplied: (result: { created_count: number; changed_count: number }) => void;
}) {
  const [proposal, setProposal] = useState(initial);
  const [tree, setTree] = useState<TreeNode[]>(initial.tree);
  const [edited, setEdited] = useState(false);
  const { run, busy, error } = useStructureActions();
  const issues = edited ? [] : proposal.impact.issues;
  const flagged = new Set(issues.map(issue => issue.node_id));
  const mixed = walkTree(tree).some(row => row.node.existing_id);
  const change = (next: TreeNode[]) => { setTree(next); setEdited(true); };
  const create = async () => {
    let current = proposal;
    if (edited) {
      current = await run<OrganizationProposal>("setup.tree_preview", { tree });
      setProposal(current); setTree(current.tree); setEdited(false);
      if (current.impact.blocking_count) return;
    }
    const result = await run<{ created_count: number; changed_count: number }>("setup.tree_apply",
      { proposal_id: current.id, preview_hash: current.preview_hash });
    onApplied(result);
  };
  const status = (node: TreeNode) => node.action === "create" ? (mixed ? "New" : null) :
    node.action === "rename" ? "Rename" : node.action === "archive" ? null : "Already exists";
  return <Dialog className="schema-dialog setup-tree-dialog" aria-label="Proposed organization" onEscape={busy ? undefined : onClose}>
    <header className="schema-dialog-head"><div><h2>Proposed organization</h2>
      <p>Here’s a first draft of your organization. Change anything, or tell Dani what’s off.</p></div>
      <button className="btn-icon" aria-label="Close proposed organization" disabled={busy} onClick={onClose}><X size={20}/></button></header>
    {error && <p role="alert" className="detail-alert schema-alert">{error}</p>}
    <div className="schema-dialog-body single"><section className="setup-tree">
      <ul className="setup-tree-rows" aria-label="Spaces, areas, goals and projects">
        {walkTree(tree).map(({ node, depth, parent }) => <li key={node.id} className={"setup-tree-row" + (flagged.has(node.id) ? " flagged" : "")}
          style={{ marginInlineStart: Math.min(depth, 3) * 20 }}>
          <div className="setup-tree-main">
            {node.action === "create" && childKinds(parent?.kind ?? null).length > 1
              ? <select className="setup-tree-kind" aria-label={"Kind of " + node.title} value={node.kind} disabled={busy}
                  onChange={e => change(editNode(tree, node.id, { kind: e.target.value as TreeKind }))}>
                  {childKinds(parent?.kind ?? null).map(kind => <option key={kind} value={kind}>{TREE_KIND_LABEL[kind]}</option>)}
                </select>
              : <span className="chip">{TREE_KIND_LABEL[node.kind]}</span>}
            <input className="setup-tree-title" aria-label={TREE_KIND_LABEL[node.kind] + " title"} value={node.title} maxLength={200} disabled={busy}
              onChange={e => change(editNode(tree, node.id, { title: e.target.value }))}/>
            {status(node) && <span className="chip">{status(node)}</span>}
            {node.action === "archive" && <label className="setup-tree-archive"><input type="checkbox" checked={!!node.archive_confirmed} disabled={busy}
              onChange={e => change(editNode(tree, node.id, { archive_confirmed: e.target.checked }))}/>Archive</label>}
            <button type="button" className="btn-icon" aria-label={"Remove " + node.title} disabled={busy}
              onClick={() => change(removeNode(tree, node.id))}><X size={16}/></button>
          </div>
          <input className="setup-tree-description" aria-label={"Description of " + node.title} placeholder="One line about it"
            value={node.description} maxLength={2000} disabled={busy || node.action !== "create"}
            onChange={e => change(editNode(tree, node.id, { description: e.target.value }))}/>
          {flagged.has(node.id) && issues.filter(issue => issue.node_id === node.id).map((issue, n) =>
            <p role="alert" className="schema-issue" key={n}>{issue.message}</p>)}
          {!!childKinds(node.kind).length && node.action !== "archive" && <button type="button" className="btn btn-ghost btn-sm" disabled={busy}
            onClick={() => change(addNode(tree, node.id, childKinds(node.kind).at(-1)!))}><Plus size={15}/>Add under this</button>}
        </li>)}
      </ul>
      <button type="button" className="btn btn-soft btn-sm" disabled={busy} onClick={() => change(addNode(tree, null, "space"))}><Plus size={15}/>Add a space</button>
      {issues.filter(issue => !issue.node_id).map((issue, n) => <p role="alert" className="schema-issue" key={n}>{issue.message}</p>)}
      <p className="setup-tree-impact" role="status">{treeImpact(tree)}</p>
    </section></div>
    <footer className="schema-dialog-foot">
      <button className="btn" disabled={busy} onClick={onKeepTalking}>Keep talking</button>
      <button className="btn" disabled={busy} onClick={onByHand}>Fill in by hand</button>
      <button className="btn btn-primary" disabled={busy || !tree.length || (!edited && !!proposal.impact.blocking_count)}
        onClick={() => void create().catch(() => {})}><Check size={17}/>Create this organization</button>
    </footer>
  </Dialog>;
}
