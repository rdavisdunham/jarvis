import { useEffect, useState } from "react";
import { Plus, X } from "lucide-react";

export const ALIAS_LIMIT = 10;
const tidy = (value: string) => value.trim().replace(/\s+/g, " ");
const key = (value: string) => tidy(value).toLocaleLowerCase();

/** Why `value` cannot be added to `names`, or "" when it can (the server checks names other people have). */
export function aliasProblem(names: string[], value: string) {
  const name = tidy(value);
  if (!name) return "Type a name first.";
  if (name.length > 80) return "Keep each name to 80 characters.";
  if (name.includes("@")) return "Use a name, not an email address.";
  if (["me", "myself", "owner", "i"].includes(key(name))) return `“${name}” can't be used as a name.`;
  if (names.some(n => key(n) === key(name))) return `“${name}” is already in the list.`;
  if (names.length >= ALIAS_LIMIT) return `Add at most ${ALIAS_LIMIT} names.`;
  return "";
}

/** "Also known as": the other names the user goes by, so work assigned to any of them (by Dani, in meetings,
 * by name) lands on them. Saved with the profile as `name_aliases` (replacing the list). */
export function NameAliases({ aliases, busy, onSave }: { aliases: string[]; busy: boolean; onSave: (names: string[]) => Promise<unknown> }) {
  const saved = aliases.join("\n");
  const [names, setNames] = useState(aliases), [draft, setDraft] = useState(""), [problem, setProblem] = useState("");
  useEffect(() => { setNames(saved ? saved.split("\n") : []); }, [saved]);
  const dirty = names.join("\n") !== saved;
  function add() {
    const why = aliasProblem(names, draft);
    setProblem(why);
    if (why) return;
    setNames([...names, tidy(draft)]); setDraft("");
  }
  return <div className="name-aliases">
    {names.length ? <ul className="chip-row" aria-label="Other names">
      {names.map(name => <li key={key(name)} className="chip">{name}
        <button type="button" className="text-button" aria-label={`Remove ${name}`} disabled={busy}
          onClick={() => { setNames(names.filter(n => n !== name)); setProblem(""); }}><X size={12}/></button></li>)}
    </ul> : <p className="settings-empty">No other names yet.</p>}
    <form className="setting-inline-form" onSubmit={e => { e.preventDefault(); add(); }}>
      <input aria-label="Add another name" value={draft} maxLength={80} placeholder="For example, your full name"
        onChange={e => { setDraft(e.target.value); setProblem(""); }}/>
      <button className="btn btn-ghost" disabled={busy || !draft.trim()}><Plus size={15}/>Add</button>
      <button type="button" className="btn btn-soft" disabled={busy || !dirty} onClick={() => void onSave(names)}>Save names</button>
    </form>
    {problem && <p className="error-banner" role="alert">{problem}</p>}
  </div>;
}
