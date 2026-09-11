import { useEffect, useRef, useState } from "react";
import { Archive, ArrowRight, FileText, Plus, Sparkles, X } from "lucide-react";
import { api, post } from "./api";
import { useDialogFocus } from "./components";
import type { Project, Task } from "./types";

export type NoteRecord = {
  id: string;
  title: string;
  content?: string;
  excerpt: string;
  tags: string[];
  project_id: string | null;
  conversation_id: string | null;
  archived: boolean;
  revision: number;
  index_state: string;
  updated_at: string;
  tasks: {
    id: string;
    title: string;
    status: string;
    evidence: string;
    note_revision: number | null;
  }[];
};
type Mutate = <T>(
  tool: string,
  args: unknown,
  message: string,
) => Promise<T | undefined>;

export function blankNote(
  task?: Task,
  conversationId?: string | null,
): NoteRecord {
  return {
    id: "new",
    title: "",
    content: "",
    excerpt: "",
    tags: [],
    project_id: task?.project_id ?? null,
    conversation_id: conversationId ?? null,
    archived: false,
    revision: 0,
    index_state: "queued",
    updated_at: new Date().toISOString(),
    tasks: task
      ? [
          {
            id: task.id,
            title: task.title,
            status: task.status,
            evidence: "",
            note_revision: null,
          },
        ]
      : [],
  };
}

export function NotesPage({
  query,
  projects,
  project,
  onProject,
  revision,
  onVisible,
  onOpen,
  onNew,
}: {
  query: string;
  projects: Project[];
  project: string;
  onProject: (name: string) => void;
  revision: number;
  onVisible: (ids: string[]) => void;
  onOpen: (id: string) => void;
  onNew: () => void;
}) {
  const [items, setItems] = useState<NoteRecord[]>([]);
  const [archived, setArchived] = useState(false);
  const [meaningKey, setMeaningKey] = useState("");
  const generation = useRef(0);
  const [offset, setOffset] = useState<number | null>(null);
  const [loading, setLoading] = useState(false),
    [error, setError] = useState("");
  const [hint, setHint] = useState("");
  const [refresh, setRefresh] = useState(0);
  const projectId = projects.find((p) => p.name === project)?.id;
  const searchKey = JSON.stringify([query, projectId, archived]);
  const meaning = meaningKey === searchKey;
  useEffect(() => {
    let active = true;
    generation.current++;
    setLoading(true);
    setError("");
    setHint("");
    const params = new URLSearchParams({
      q: query,
      archived: String(archived),
    });
    if (projectId) params.set("project_id", projectId);
    api<{
      items: NoteRecord[];
      next_offset?: number | null;
      mode?: string;
      truncated?: boolean;
    }>(
      (meaning && query.trim() && !archived ? "/notes/search?" : "/notes?") +
        params,
    )
      .then((data) => {
        if (!active) return;
        setItems(data.items);
        setOffset(data.next_offset ?? null);
        setHint(
          data.mode === "keyword_fallback"
            ? "Meaning search is unavailable. Showing keyword matches."
            : data.truncated
              ? "Showing the closest matches. Narrow your search for more."
              : "",
        );
      })
      .catch((e) => {
        if (active) setError(e.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
      generation.current++;
    };
  }, [query, projectId, archived, meaning, revision, refresh]);
  const visibleKey = items
    .slice(0, 60)
    .map((n) => n.id)
    .join(",");
  useEffect(
    () => onVisible(visibleKey ? visibleKey.split(",") : []),
    [visibleKey, onVisible],
  );
  async function more() {
    if (offset === null || loading) return;
    const requestGeneration = generation.current;
    setLoading(true);
    try {
      const params = new URLSearchParams({
        q: query,
        archived: String(archived),
        offset: String(offset),
      });
      if (projectId) params.set("project_id", projectId);
      const data = await api<{
        items: NoteRecord[];
        next_offset: number | null;
      }>("/notes?" + params);
      if (requestGeneration !== generation.current) return;
      setItems((old) => [...old, ...data.items]);
      setOffset(data.next_offset);
    } catch (e) {
      if (requestGeneration === generation.current)
        setError((e as Error).message);
    } finally {
      if (requestGeneration === generation.current) setLoading(false);
    }
  }
  return (
    <section className="notes-workspace" aria-label="Notes workspace">
      <div className="workspace-actions">
        <div className="note-filters">
          <label>
            Project
            <select
              aria-label="Notes project"
              value={project}
              onChange={(e) => onProject(e.target.value)}
            >
              <option value="">All projects</option>
              {projects.map((p) => (
                <option key={p.id} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Collection
            <select
              aria-label="Notes collection"
              value={archived ? "archived" : "active"}
              onChange={(e) => setArchived(e.target.value === "archived")}
            >
              <option value="active">Active notes</option>
              <option value="archived">Archived notes</option>
            </select>
          </label>
        </div>
        <button className="primary compact" onClick={onNew}>
          <Plus size={16} />
          New note
        </button>
      </div>
      {query.trim() && !archived && (
        <button
          className="secondary compact"
          disabled={loading}
          onClick={() => {
            setMeaningKey(searchKey);
            setRefresh((n) => n + 1);
          }}
        >
          <Sparkles size={15} />
          Search by meaning
        </button>
      )}
      {loading && (
        <p role="status" className="footnote">
          Loading notes…
        </p>
      )}
      {error && (
        <p role="alert" className="error-banner">
          {error}
          <button onClick={() => setRefresh((n) => n + 1)}>Retry</button>
        </p>
      )}
      {hint && (
        <p role="status" className="footnote">
          {hint}
        </p>
      )}
      <div className="note-grid">
        {items.map((n) => (
          <button key={n.id} className="note-card" onClick={() => onOpen(n.id)}>
            <span className="note-card-heading">
              <FileText size={18} />
              <strong>{n.title}</strong>
            </span>
            <span className="note-excerpt">
              {n.excerpt || "A little room for your thoughts."}
            </span>
            <span className="task-meta">
              {projects.find((p) => p.id === n.project_id)?.name}
              {n.tags.length
                ? " · " + n.tags.map((t) => "#" + t).join(" ")
                : ""}
            </span>
            <span className="note-card-footer">
              {n.tasks.length ? n.tasks.length + " linked tasks" : "Note"}
              <span>{new Date(n.updated_at).toLocaleDateString()}</span>
            </span>
          </button>
        ))}
      </div>
      {!loading && !error && !items.length && (
        <div className="empty-state">
          <FileText size={28} />
          <h3>
            {query
              ? "No matching notes."
              : archived
                ? "No archived notes."
                : "A place to put your thoughts."}
          </h3>
          <p>
            {query
              ? "Try another phrase or search by meaning."
              : "Keep notes, connect them to your work, and ask Eri to find the next steps."}
          </p>
        </div>
      )}
      {offset !== null && (
        <button
          className="secondary"
          disabled={loading}
          onClick={() => void more()}
        >
          Load more notes
        </button>
      )}
    </section>
  );
}

export function NoteEditor({
  note,
  projects,
  tasks,
  busy,
  error,
  mutate,
  onSaved,
  onClose,
  onTask,
  onConversation,
}: {
  note: NoteRecord;
  projects: Project[];
  tasks: Task[];
  busy: boolean;
  error: string;
  mutate: Mutate;
  onSaved: (note: NoteRecord) => void;
  onClose: () => void;
  onTask: (id: string) => void;
  onConversation: (id: string) => void;
}) {
  useDialogFocus();
  const [title, setTitle] = useState(note.title),
    [content, setContent] = useState(note.content ?? "");
  const [tags, setTags] = useState(note.tags.join(", ")),
    [project, setProject] = useState(note.project_id ?? "");
  const [linked, setLinked] = useState(note.tasks.map((t) => t.id)),
    [taskQuery, setTaskQuery] = useState("");
  const [extracting, setExtracting] = useState(false),
    [localError, setLocalError] = useState("");
  const [proposals, setProposals] = useState<
    {
      title: string;
      evidence: string;
      existing_task_id: string | null;
      checked: boolean;
    }[]
  >([]);
  const [proposalRevision, setProposalRevision] = useState(0);
  const dirty =
    title !== note.title ||
    content !== (note.content ?? "") ||
    tags !== note.tags.join(", ") ||
    project !== (note.project_id ?? "") ||
    linked.join(",") !== note.tasks.map((t) => t.id).join(",");
  async function save() {
    const args = {
      title,
      content,
      tags: tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean),
      project_id: project || null,
      task_ids: linked,
      conversation_id: note.conversation_id,
    };
    const result = await mutate<NoteRecord>(
      note.id === "new" ? "note.create" : "note.update",
      note.id === "new"
        ? args
        : { note_id: note.id, expected_revision: note.revision, ...args },
      "Note saved",
    );
    if (result) onSaved(result);
  }
  async function extract() {
    setExtracting(true);
    setLocalError("");
    try {
      const data = await post<{ revision: number; items: typeof proposals }>(
        "/notes/" + note.id + "/suggest-tasks",
      );
      setProposals(
        data.items.map((p) => ({ ...p, checked: !p.existing_task_id })),
      );
      setProposalRevision(data.revision);
      if (!data.items.length)
        setLocalError("No unfinished to-dos found in this note.");
    } catch (e) {
      setLocalError((e as Error).message);
    } finally {
      setExtracting(false);
    }
  }
  async function createTasks() {
    const items = proposals
      .filter((p) => p.checked && !p.existing_task_id)
      .map(({ title, evidence }) => ({ title, evidence }));
    const result = await mutate(
      "note.tasks",
      { note_id: note.id, expected_revision: proposalRevision, items },
      "Tasks created from note",
    );
    if (result) {
      setProposals([]);
      try {
        onSaved(await api<NoteRecord>("/notes/" + note.id));
      } catch (e) {
        setLocalError((e as Error).message);
      }
    }
  }
  return (
    <div className="modal-backdrop">
      <form
        className="dialog note-editor"
        role="dialog"
        aria-modal="true"
        aria-labelledby="note-title"
        onSubmit={(e) => {
          e.preventDefault();
          void save();
        }}
      >
        <div className="dialog-heading">
          <h2 id="note-title">{note.id === "new" ? "New note" : "Note"}</h2>
          <button
            type="button"
            className="icon-button"
            aria-label="Close note"
            disabled={busy}
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
        {(error || localError) && (
          <p className="error-banner" role="alert">
            {error || localError}
          </p>
        )}
        <label>
          Title
          <input
            autoFocus
            required
            maxLength={200}
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
        </label>
        <label>
          Note
          <textarea
            className="note-body"
            maxLength={30000}
            rows={11}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Start writing…"
          />
        </label>
        <div className="form-grid">
          <label>
            Project
            <select
              aria-label="Note project"
              value={project}
              onChange={(e) => setProject(e.target.value)}
            >
              <option value="">No project</option>
              {projects
                .filter((p) => !p.archived || p.id === project)
                .map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
            </select>
          </label>
          <label>
            Tags
            <input
              value={tags}
              maxLength={800}
              placeholder="Separate with commas"
              onChange={(e) => setTags(e.target.value)}
            />
          </label>
        </div>
        <details className="note-links">
          <summary>Linked tasks · {linked.length}</summary>
          <input
            aria-label="Find tasks to link"
            placeholder="Find a task…"
            value={taskQuery}
            onChange={(e) => setTaskQuery(e.target.value)}
          />
          <div className="link-task-list">
            {tasks
              .filter(
                (t) =>
                  linked.includes(t.id) ||
                  t.title.toLowerCase().includes(taskQuery.toLowerCase()),
              )
              .map((t) => (
                <label key={t.id}>
                  <input
                    type="checkbox"
                    checked={linked.includes(t.id)}
                    onChange={(e) =>
                      setLinked((ids) =>
                        e.target.checked
                          ? [...ids, t.id]
                          : ids.filter((id) => id !== t.id),
                      )
                    }
                  />
                  {t.title}
                </label>
              ))}
          </div>
        </details>
        {!!note.tasks.length && (
          <div className="note-linked-records">
            {note.tasks.map((t) => (
              <div key={t.id}>
                <button
                  className="text-button"
                  type="button"
                  disabled={dirty || busy}
                  onClick={() => onTask(t.id)}
                >
                  {t.title}
                  <ArrowRight size={14} />
                </button>
                {t.evidence && (
                  <small>
                    From note version {t.note_revision}: {t.evidence}
                  </small>
                )}
              </div>
            ))}
          </div>
        )}
        {note.conversation_id && (
          <button
            type="button"
            className="text-button"
            disabled={dirty || busy}
            onClick={() => onConversation(note.conversation_id!)}
          >
            Open linked conversation
            <ArrowRight size={14} />
          </button>
        )}
        {note.id !== "new" && !note.archived && (
          <section className="note-extraction">
            <button
              type="button"
              className="secondary compact"
              disabled={busy || extracting || dirty}
              onClick={() => void extract()}
            >
              <Sparkles size={15} />
              {extracting ? "Finding to-dos…" : "Find to-dos"}
            </button>
            <p className="footnote">
              {dirty
                ? "Save your edits before extracting to-dos or opening linked records."
                : "Review suggestions before creating tasks. Existing extractions won't create duplicates."}
            </p>
            {note.index_state !== "ready" && (
              <p className="footnote">
                {note.index_state === "failed"
                  ? "Meaning search could not be prepared. Save again to retry; keyword search works."
                  : "Preparing this note for meaning search."}
              </p>
            )}
            {proposals.map((p, i) => (
              <div className="note-proposal" key={i}>
                <label>
                  <input
                    type="checkbox"
                    checked={p.checked}
                    disabled={!!p.existing_task_id}
                    onChange={(e) =>
                      setProposals((rows) =>
                        rows.map((r, index) =>
                          index === i ? { ...r, checked: e.target.checked } : r,
                        ),
                      )
                    }
                  />
                  {p.existing_task_id ? "Already extracted" : "Create task"}
                </label>
                <input
                  aria-label={"Suggested task " + (i + 1)}
                  value={p.title}
                  maxLength={500}
                  disabled={!!p.existing_task_id}
                  onChange={(e) =>
                    setProposals((rows) =>
                      rows.map((r, index) =>
                        index === i ? { ...r, title: e.target.value } : r,
                      ),
                    )
                  }
                />
                <blockquote>{p.evidence}</blockquote>
              </div>
            ))}
            {!!proposals.length && (
              <button
                type="button"
                className="primary"
                disabled={
                  busy ||
                  dirty ||
                  !proposals.some((p) => p.checked && !p.existing_task_id) ||
                  proposals.some((p) => p.checked && !p.title.trim())
                }
                onClick={() => void createTasks()}
              >
                Create selected tasks
              </button>
            )}
          </section>
        )}
        <div className="dialog-actions">
          {note.id !== "new" && (
            <button
              type="button"
              className="text-button"
              disabled={busy || dirty}
              onClick={async () => {
                const result = await mutate(
                  "note.update",
                  {
                    note_id: note.id,
                    expected_revision: note.revision,
                    archived: !note.archived,
                  },
                  note.archived ? "Note restored" : "Note archived",
                );
                if (result) onClose();
              }}
            >
              <Archive size={15} />
              {note.archived ? "Restore note" : "Archive note"}
            </button>
          )}
          <button
            className="primary"
            disabled={busy || extracting || !title.trim()}
          >
            Save note
          </button>
        </div>
      </form>
    </div>
  );
}

export function TaskNotes({
  taskId,
  revision,
  onOpen,
  onNew,
}: {
  taskId: string;
  revision: number;
  onOpen: (id: string) => void;
  onNew: () => void;
}) {
  const [items, setItems] = useState<NoteRecord[]>([]),
    [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    api<{ items: NoteRecord[] }>("/notes?task_id=" + encodeURIComponent(taskId))
      .then((data) => {
        if (active) setItems(data.items);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [taskId, revision]);
  return (
    <section className="task-reminders">
      <h3>Linked notes</h3>
      {error && <p role="alert">{error}</p>}
      {items.map((n) => (
        <button
          type="button"
          className="text-button"
          key={n.id}
          onClick={() => onOpen(n.id)}
        >
          <FileText size={14} />
          {n.title}
        </button>
      ))}
      <button type="button" className="text-button" onClick={onNew}>
        <Plus size={14} />
        New linked note
      </button>
    </section>
  );
}
