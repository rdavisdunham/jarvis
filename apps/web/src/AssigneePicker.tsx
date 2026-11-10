import { useId, useState, type KeyboardEvent } from "react";
import { Check, UserRound } from "lucide-react";
import { Popover } from "./ux";
import type { Actor } from "./productivity";
import "./assignee-picker.css";
import { currentAssignee, filterOptions, useAssigneeOptions, type AssigneeOption } from "./assignees";

/** Searchable "Assigned to" dropdown over accounts, the assistant and Unassigned. A current assignee who isn't
 * offered (a Linear person on a synced task) is shown on the button but not listed. */
export function AssigneePicker({value, actors, disabled = false, buttonClassName = "prop-text", onChange}: {
  value: {id?: string | null; name?: string | null}; actors?: Actor[]; disabled?: boolean; buttonClassName?: string;
  onChange: (option: AssigneeOption) => void;
}) {
  const { options, actors: all } = useAssigneeOptions(actors);
  const current = currentAssignee(options, value, all);
  return <Popover floating label="Change assignee" title="Assigned to" disabled={disabled} buttonClassName={buttonClassName + " assignee-trigger"}
    panelClassName="assignee-panel" button={<><UserRound size={14} aria-hidden="true"/><span>{current.label}</span></>}>
    {close => <AssigneeList options={options} selected={current.option?.key ?? null} onPick={option => { close(); if (option.key !== current.option?.key) onChange(option); }}/>}
  </Popover>;
}

function AssigneeList({options, selected, onPick}: {options: AssigneeOption[]; selected: string | null; onPick: (option: AssigneeOption) => void}) {
  const [query, setQuery] = useState("");
  const shown = filterOptions(options, query);
  const [active, setActive] = useState(() => Math.max(0, shown.findIndex(o => o.key === selected)));
  const id = useId();
  const at = Math.min(active, shown.length - 1);
  const keys = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); setActive(Math.max(0, Math.min(shown.length - 1, at + (e.key === "ArrowDown" ? 1 : -1)))); }
    else if (e.key === "Enter" && shown[at]) { e.preventDefault(); onPick(shown[at]); }
  };
  let group = "";
  return <div className="assignee-list">
    <input type="search" aria-label="Find an assignee" placeholder="Search people…" value={query} autoComplete="off"
      role="combobox" aria-expanded="true" aria-controls={id} aria-activedescendant={shown[at] ? id + "-" + at : undefined}
      onChange={e => { setQuery(e.target.value); setActive(0); }} onKeyDown={keys}/>
    <div id={id} role="listbox" aria-label="Assignees">
      {shown.map((o, i) => {
        const heading = o.group && o.group !== group ? o.group : "";
        group = o.group;
        return <div key={o.key}>
          {heading && <div className="assignee-group" role="presentation">{heading}</div>}
          <div id={id + "-" + i} role="option" aria-selected={o.key === selected} className={"assignee-option" + (i === at ? " is-active" : "")}
            onMouseEnter={() => setActive(i)} onMouseDown={e => e.preventDefault()} onClick={() => onPick(o)}>
            <span>{o.label}</span>{o.key === selected && <Check size={14} aria-hidden="true"/>}
          </div>
        </div>;
      })}
      {!shown.length && <p className="footnote">No one matches “{query.trim()}”.</p>}
    </div>
  </div>;
}
