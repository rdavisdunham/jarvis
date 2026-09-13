import { z } from "zod";
import { useEditor, nullableId, choice, tagsField } from "./editor-control";
import { HomeFields, LinkPicker, OrganizationFilters } from "./Productivity";
import {
  emptyOrganization,
  type Organization,
  type Home,
  type OrganizationFilter,
} from "./productivity";
import { useEffect, useRef, useState } from "react";
import { Archive, ArrowRight, FileText, Plus, Sparkles, X } from "lucide-react";
import { api, post } from "./api";
import { useDialogFocus } from "./components";
import type { Project, Task } from "./types";

export type NoteRecord = {
  space_id?: string | null;
  area_id?: string | null;
  goals?: { id: string; name: string; archived?: boolean }[];
  projects?: { id: string; name: string; archived?: boolean }[];
  related_notes?: { id: string; title: string; archived?: boolean }[];
  backlinks?: { id: string; title: string; archived?: boolean }[];
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
  organization,
  organizationFilter,
  onOrganizationFilter,
  query,
  projects,
  project,
  onProject,
  revision,
  onVisible,
  onOpen,
  onNew,
  archived,
  onArchived,
  mode,
  onMode,
}: {
  archived: boolean;
  onArchived: (v: boolean) => void;
  mode: "keyword" | "semantic";
  onMode: (v: "keyword" | "semantic") => void;
  organization: Organization;
  organizationFilter: OrganizationFilter;
  onOrganizationFilter: (value: OrganizationFilter) => void;
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

  const generation = useRef(0);
  const [offset, setOffset] = useState<number | null>(null);
  const [loading, setLoading] = useState(false),
    [error, setError] = useState("");
  const [hint, setHint] = useState("");
  const [refresh, setRefresh] = useState(0);
  const projectId = projects.find((p) => p.name === project)?.id;
  const meaning = mode === "semantic";
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
    if (organizationFilter.space)
      params.set("space_id", organizationFilter.space);
    if (organizationFilter.area) params.set("area_id", organizationFilter.area);
    if (organizationFilter.goal) params.set("goal_id", organizationFilter.goal);
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
  }, [
    query,
    projectId,
    archived,
    meaning,
    revision,
    refresh,
    organizationFilter,
  ]);
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
      if (organizationFilter.space)
        params.set("space_id", organizationFilter.space);
      if (organizationFilter.area)
        params.set("area_id", organizationFilter.area);
      if (organizationFilter.goal)
        params.set("goal_id", organizationFilter.goal);
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
  const filterLabels = [
    organization.spaces.find((v) => v.id === organizationFilter.space)?.name,
    organization.areas.find((v) => v.id === organizationFilter.area)?.name,
    organization.goals.find((v) => v.id === organizationFilter.goal)?.name,
    project,
    archived ? "Archived" : "",
  ].filter(Boolean);
  return (
    <section className="notes-workspace" aria-label="Notes workspace">
      <div className="workspace-actions">
        <details className="filter-panel note-filter-panel">
          <summary>
            Filters{" "}
            {filterLabels.length > 0 && (
              <span>{filterLabels.length} active</span>
            )}
          </summary>
          <div className="note-filters">
            <OrganizationFilters
              organization={organization}
              value={organizationFilter}
              onChange={onOrganizationFilter}
            />
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
                onChange={(e) => onArchived(e.target.value === "archived")}
              >
                <option value="active">Active notes</option>
                <option value="archived">Archived notes</option>
              </select>
            </label>
          </div>
        </details>
        <button className="primary compact" onClick={onNew}>
          <Plus size={16} />
          New note
        </button>
      </div>
      {filterLabels.length > 0 && (
        <div className="active-filters" aria-label="Active note filters">
          {filterLabels.map((label, i) => (
            <span key={i}>{label}</span>
          ))}
        </div>
      )}
      {query.trim() && !archived && (
        <button
          className="secondary compact"
          disabled={loading}
          onClick={() => {
            onMode(meaning ? "keyword" : "semantic");
            setRefresh((n) => n + 1);
          }}
        >
          <Sparkles size={15} />
          {meaning ? "Use keyword search" : "Search by meaning"}
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
            <span className="note-excerpt">{n.excerpt || "Empty note"}</span>
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
  organization = emptyOrganization,
  onOpenNote,
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
  organization?: Organization;
  onOpenNote?: (id: string) => void;
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
  const [home, setHome] = useState<Home>({
    space_id: note.space_id,
    area_id: note.area_id,
  });
  const [goalIds, setGoalIds] = useState((note.goals ?? []).map((g) => g.id));
  const [projectIds, setProjectIds] = useState(
    (note.projects ?? []).map((p) => p.id),
  );
  const [noteIds, setNoteIds] = useState(
    (note.related_notes ?? []).map((n) => n.id),
  );
  const [noteChoices, setNoteChoices] = useState<
    { id: string; name: string; archived?: boolean }[]
  >((note.related_notes ?? []).map((n) => ({ ...n, name: n.title })));
  useEffect(() => {
    let current = true;
    async function load() {
      let offset: number | null = 0;
      const choices = new Map<
        string,
        { id: string; name: string; archived?: boolean }
      >(
        (note.related_notes ?? []).map((n) => [
          n.id,
          { id: n.id, name: n.title, archived: n.archived },
        ]),
      );
      while (offset !== null && current) {
        const result: { items: NoteRecord[]; next_offset: number | null } =
          await api("/notes?limit=100&offset=" + offset);
        for (const n of result.items)
          if (n.id !== note.id)
            choices.set(n.id, {
              id: n.id,
              name: n.title,
              archived: n.archived,
            });
        offset = result.next_offset;
      }
      if (current) setNoteChoices([...choices.values()]);
    }
    void load().catch(() => {
      if (current)
        setLocalError(
          "Could not load notes to link. Existing links are preserved.",
        );
    });
    return () => {
      current = false;
    };
  }, [note.id]);
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
    (home.space_id ?? null) !== (note.space_id ?? null) ||
    (home.area_id ?? null) !== (note.area_id ?? null) ||
    goalIds.join(",") !== (note.goals ?? []).map((g) => g.id).join(",") ||
    projectIds.join(",") !== (note.projects ?? []).map((p) => p.id).join(",") ||
    noteIds.join(",") !==
      (note.related_notes ?? []).map((n) => n.id).join(",") ||
    title !== note.title ||
    content !== (note.content ?? "") ||
    tags !== note.tags.join(", ") ||
    project !== (note.project_id ?? "") ||
    linked.join(",") !== note.tasks.map((t) => t.id).join(",");
  const values = {
    title,
    content,
    tags: tags
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean),
    space_id: home.space_id || null,
    area_id: home.area_id || null,
    goal_ids: goalIds,
    project_ids: projectIds,
    related_note_ids: noteIds,
    project_id: project || null,
    task_ids: linked,
  };
  const baseline = {
    title: note.title,
    content: note.content ?? "",
    tags: note.tags,
    space_id: note.space_id || null,
    area_id: note.area_id || null,
    goal_ids: (note.goals ?? []).map((g) => g.id),
    project_ids: (note.projects ?? []).map((p) => p.id),
    related_note_ids: (note.related_notes ?? []).map((n) => n.id),
    project_id: note.project_id || null,
    task_ids: note.tasks.map((t) => t.id),
  };
  const editorSchema = z.object({
    title: z.string().min(1).max(200),
    content: z.string().max(30000),
    tags: tagsField,
    space_id: nullableId(organization.spaces.map((s) => s.id)),
    area_id: nullableId(organization.areas.map((a) => a.id)),
    project_id: nullableId(projects.map((p) => p.id)),
    task_ids: z.array(choice([...tasks.map((t) => t.id), ...linked])).max(200),
    goal_ids: z
      .array(choice([...organization.goals.map((g) => g.id), ...goalIds]))
      .max(100),
    project_ids: z
      .array(choice([...projects.map((p) => p.id), ...projectIds]))
      .max(100),
    related_note_ids: z
      .array(choice([...noteChoices.map((n) => n.id), ...noteIds]))
      .max(100),
  });
  async function save() {
    if (!title.trim()) {
      setLocalError("Give the note a title.");
      return;
    }
    const args: Record<string, unknown> =
      note.id === "new"
        ? { ...values, conversation_id: note.conversation_id }
        : Object.fromEntries(
            Object.entries(values).filter(
              ([key, value]) =>
                JSON.stringify(value) !==
                JSON.stringify(baseline[key as keyof typeof baseline]),
            ),
          );
    if (project) {
      delete args.space_id;
      delete args.area_id;
    }
    if (note.id !== "new" && !Object.keys(args).length) return note;
    const result = await mutate<NoteRecord>(
      note.id === "new" ? "note.create" : "note.update",
      note.id === "new"
        ? args
        : { note_id: note.id, expected_revision: note.revision, ...args },
      "Note saved",
    );
    if (result) onSaved(result);
    return result;
  }
  useEditor({
    kind: "note",
    record_id: note.id === "new" ? null : note.id,
    dirty,
    busy: busy || extracting,
    schema: editorSchema,
    values,
    save,
    close: onClose,
    patch: (v) => {
      if ("title" in v) setTitle(v.title as string);
      if ("content" in v) setContent(v.content as string);
      if ("tags" in v) setTags((v.tags as string[]).join(", "));
      if ("project_id" in v) setProject((v.project_id as string) || "");
      if ("space_id" in v || "area_id" in v)
        setHome((h) => ({
          ...h,
          ...Object.fromEntries(
            Object.entries(v).filter(
              ([k]) => k === "space_id" || k === "area_id",
            ),
          ),
        }));
      if ("task_ids" in v) setLinked(v.task_ids as string[]);
      if ("goal_ids" in v) setGoalIds(v.goal_ids as string[]);
      if ("project_ids" in v) setProjectIds(v.project_ids as string[]);
      if ("related_note_ids" in v) setNoteIds(v.related_note_ids as string[]);
    },
  });
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
            Home project
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
        <HomeFields
          organization={organization}
          value={projects.find((p) => p.id === project) ?? home}
          onChange={setHome}
          disabled={!!project}
        />
        <details className="note-links">
          <summary>Connected goals, projects and notes</summary>
          <LinkPicker
            label="Goals"
            items={organization.goals}
            selected={goalIds}
            onChange={setGoalIds}
          />
          <LinkPicker
            label="Related projects"
            items={projects}
            selected={projectIds}
            onChange={setProjectIds}
          />
          <LinkPicker
            label="Notes"
            items={noteChoices}
            selected={noteIds}
            onChange={setNoteIds}
          />
        </details>
        {!!(
          (note.related_notes?.length ?? 0) + (note.backlinks?.length ?? 0)
        ) && (
          <div className="note-backlinks">
            <strong>Connected notes & backlinks</strong>
            {[
              ...new Map(
                [...(note.related_notes ?? []), ...(note.backlinks ?? [])].map(
                  (n) => [n.id, n],
                ),
              ).values(),
            ].map((n) => (
              <button
                key={n.id}
                type="button"
                className="text-button"
                disabled={dirty || busy}
                onClick={() => onOpenNote?.(n.id)}
              >
                {n.title}
                <ArrowRight size={14} />
              </button>
            ))}
          </div>
        )}
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
