import { useEffect, useRef, useState } from "react";
import { Bookmark, Link, Trash2, Plus } from "lucide-react";
import { api, post } from "./api";
import { type SavedView, type ViewState, viewLink } from "./saved-views";
export function SavedViews({
  state,
  onApply,
}: {
  state: ViewState;
  onApply: (value: ViewState) => void;
}) {
  const [items, setItems] = useState<SavedView[]>([]),
    [selected, setSelected] = useState(""),
    [naming, setNaming] = useState(false),
    [name, setName] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const retry = useRef<{ key: string; body: unknown } | null>(null);
  const load = () =>
    api<{ items: SavedView[] }>("/task-views").then((r) => setItems(r.items));
  useEffect(() => {
    void load().catch((e) => setError(e.message));
  }, []);
  async function save() {
    if (!name.trim()) return;
    setBusy(true);
    setError("");
    try {
      const key = JSON.stringify({ name, state });
      if (retry.current?.key !== key)
        retry.current = {
          key,
          body: { id: crypto.randomUUID(), name, expected_revision: 0, state },
        };
      const result = await post<SavedView>("/task-views", retry.current.body);
      retry.current = null;
      await load();
      setSelected(result.id);
      setNaming(false);
      setName("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function remove() {
    const view = items.find((v) => v.id === selected);
    if (!view) return;
    setBusy(true);
    try {
      await post("/task-views/remove", {
        id: view.id,
        expected_revision: view.revision,
      });
      await load();
      setSelected("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="saved-views">
      <Bookmark size={14} />
      <select
        aria-label="Saved task views"
        value={selected}
        disabled={busy}
        onChange={(e) => {
          setSelected(e.target.value);
          const v = items.find((x) => x.id === e.target.value);
          if (v) onApply(v.state);
        }}
      >
        <option value="">Saved views</option>
        {items.map((v) => (
          <option key={v.id} value={v.id}>
            {v.name}
          </option>
        ))}
      </select>
      <button className="text-button" onClick={() => setNaming(!naming)}>
        <Plus size={14} />
        Save current view
      </button>
      <button
        className="text-button"
        onClick={() =>
          void navigator.clipboard
            .writeText(viewLink(state).href)
            .catch(() =>
              setError(
                "Could not copy. The address bar contains this view's link.",
              ),
            )
        }
      >
        <Link size={14} />
        Copy view link
      </button>
      {selected && (
        <button
          className="icon-button"
          aria-label="Delete saved view"
          disabled={busy}
          onClick={() => void remove()}
        >
          <Trash2 size={14} />
        </button>
      )}
      {naming && (
        <form
          className="saved-view-name"
          onSubmit={(e) => {
            e.preventDefault();
            void save();
          }}
        >
          <input
            autoFocus
            aria-label="View name"
            value={name}
            maxLength={80}
            onChange={(e) => setName(e.target.value)}
          />
          <button className="secondary compact" disabled={busy || !name.trim()}>
            Save view
          </button>
        </form>
      )}
      {error && (
        <span className="error-banner" role="alert">
          {error}
        </span>
      )}
    </div>
  );
}
