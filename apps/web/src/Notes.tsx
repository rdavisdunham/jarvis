import { NoteLists, NoteFiling } from "./NoteLists";
import { RecordTools } from "./record-links";
import { humanLabel, Dialog } from "./ux";
import { z } from "zod";
import { useEditor, nullableId, choice, tagsField } from "./editor-control";
import { HomeFields, LinkPicker, OrganizationFilters } from "./Productivity";
import {
  emptyOrganization,
  type Organization,
  type Home,
  type OrganizationFilter,
} from "./productivity";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Archive, ChevronRight, FileText, FolderOpen, ListChecks, MessageSquare, Plus, Search, ListFilter, Sparkles, X } from "lucide-react";
import "./notes.css";
import { api, post } from "./api";
import { notePreview } from "./note-preview";
import type { Project, Task } from "./types";

type NoteSource = {id:string; title:string; evidence:string; source_revision:number; source_changed:boolean};
export type NoteRecord = {
  organization?: {status:string; tags_locked:boolean; generated:boolean; saved_count?:number; uncertain_count?:number} | null;
  sources?: NoteSource[];
  saved_entries?: NoteSource[];
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
  listId = "", onList, canEdit = true,
  onQuery,
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
  listId?: string;
  onList?: (id: string) => void;
  onQuery?: (value: string) => void;
  canEdit?: boolean;
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
    if (listId === "uncategorized") params.set("uncategorized", "true");
    else if (listId) params.set("list_id", listId);
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
    listId,
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
      if (listId === "uncategorized") params.set("uncategorized", "true");
      else if (listId) params.set("list_id", listId);
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
  const toolbar = (
    <div className="notes-toolbar">
      {onQuery && (
        <label className="page-search notes-search">
          <Search size={16} aria-hidden />
          <input aria-label="Search notes" placeholder="Search notes…" value={query} onChange={(e) => onQuery(e.target.value)} />
        </label>
      )}
      <FilterMenu count={filterLabels.length}>
        <OrganizationFilters organization={organization} value={organizationFilter} onChange={onOrganizationFilter} />
        <label>
          Project
          <select aria-label="Notes project" value={project} onChange={(e) => onProject(e.target.value)}>
            <option value="">All projects</option>
            {projects.map((p) => (
              <option key={p.id} value={p.name}>{p.name}</option>
            ))}
          </select>
        </label>
        <label>
          Collection
          <select aria-label="Notes collection" value={archived ? "archived" : "active"} onChange={(e) => onArchived(e.target.value === "archived")}>
            <option value="active">Active notes</option>
            <option value="archived">Archived notes</option>
          </select>
        </label>
      </FilterMenu>
      <button className="btn btn-primary" disabled={!canEdit} onClick={onNew}>
        <Plus size={16} />
        New note
      </button>
    </div>
  );
  return (
    <section className="notes-workspace notes-page" aria-label="Notes workspace">
      <NoteLists selected={listId} onSelect={id => onList?.(id)} onOpen={onOpen} revision={revision} canEdit={canEdit} toolbar={toolbar}>
      {(filterLabels.length > 0 || (query.trim() && !archived)) && (
        <div className="notes-context">
          {filterLabels.length > 0 && (
            <div className="chip-row" aria-label="Active note filters">
              {filterLabels.map((label, i) => (
                <span className="chip" key={i}>{label}</span>
              ))}
            </div>
          )}
          {query.trim() && !archived && (
            <button
              className="btn btn-ghost btn-sm notes-mode"
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
        </div>
      )}
      {loading && (
        <p role="status" className="notes-status">
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
        <p role="status" className="notes-status">
          {hint}
        </p>
      )}
      <div className="note-grid">
        {items.map((n) => {
          const home = projects.find((p) => p.id === n.project_id)?.name;
          const preview = notePreview(n);
          return (
            <button key={n.id} className="note-card" onClick={() => onOpen(n.id)}>
              <strong className="note-card-title">{n.title}</strong>
              {preview && <span className="note-card-preview">{preview}</span>}
              <span className="note-card-foot">
                <span className="note-card-tags">
                  {home && <span className="chip chip-home"><FolderOpen size={13} aria-hidden />{home}</span>}
                  {n.tags.map((t) => <span className="chip" key={t}>{t}</span>)}
                  {n.tasks.length > 0 && <span className="chip"><ListChecks size={13} aria-hidden />{n.tasks.length === 1 ? "1 task" : n.tasks.length + " tasks"}</span>}
                </span>
                <time className="note-card-date" dateTime={n.updated_at}>{shortDate(n.updated_at)}</time>
              </span>
            </button>
          );
        })}
      </div>
      {!loading && !error && !items.length && (
        <div className="empty notes-empty">
          <p>
            {query
              ? "No notes match this search. Try another phrase or search by meaning."
              : archived
                ? "Archived notes will appear here."
                : listId === "uncategorized"
                  ? "Every note is filed in a list."
                  : listId
                    ? "Nothing is filed in this list yet."
                    : "Keep notes here, connect them to your work, and ask Eri to find the next steps."}
          </p>
          {!query && !archived && canEdit && !listId && (
            <button className="btn btn-soft" onClick={onNew}>
              <Plus size={16} />
              Write a note
            </button>
          )}
          {query && onQuery && (
            <button className="btn btn-soft" onClick={() => onQuery("")}>
              Clear search
            </button>
          )}
        </div>
      )}
      {offset !== null && (
        <button
          className="btn notes-more"
          disabled={loading}
          onClick={() => void more()}
        >
          Load more notes
        </button>
      )}
      </NoteLists>
    </section>
  );
}

function shortDate(iso: string) {
  const d = new Date(iso), now = new Date();
  return d.toLocaleDateString([], d.getFullYear() === now.getFullYear()
    ? { month: "short", day: "numeric" }
    : { month: "short", day: "numeric", year: "numeric" });
}

/** A "Filters" button that opens a small popover; closes on outside click or Escape. */
function FilterMenu({ count, children }: { count: number; children: ReactNode }) {
  const ref = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const outside = (event: Event) => {
      const el = ref.current;
      if (el?.open && !el.contains(event.target as Node)) el.open = false;
    };
    const escape = (event: KeyboardEvent) => {
      const el = ref.current;
      if (event.key === "Escape" && el?.open && el.contains(document.activeElement)) {
        event.stopPropagation();
        el.open = false;
        el.querySelector("summary")?.focus();
      }
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape, true);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape, true);
    };
  }, []);
  return (
    <details className="notes-filter" ref={ref}>
      <summary className="btn">
        <ListFilter size={16} aria-hidden />
        Filters
        {count > 0 && <span className="notes-filter-count">{count}</span>}
      </summary>
      <div className="popover notes-filter-popover">{children}</div>
    </details>
  );
}

export function NoteEditor({
  readOnly = false,
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
  readOnly?: boolean;
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
  const [writing, setWriting] = useState(note.id === "new");
  const [discarding, setDiscarding] = useState(false);
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
    mode: writing ? "edit" : "detail",
    record_id: note.id === "new" ? null : note.id,
    dirty,
    busy: busy || extracting,
    schema: editorSchema,
    values,
    save,
    close: onClose,
    patch: (v) => {
      setWriting(true);
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
  const homeProject = projects.find((p) => p.id === project)?.name;
  const tagList = tags.split(",").map((t) => t.trim()).filter(Boolean);
  const connected = [
    ...new Map(
      [...(note.related_notes ?? []), ...(note.backlinks ?? [])].map((n) => [n.id, n]),
    ).values(),
  ];
  return (
    <Dialog as="form"
        className={"dialog note-editor" + (!writing ? " note-detail" : "")}
        onChange={() => setWriting(true)}
        aria-labelledby="note-title"
        onSubmit={(e) => {
          e.preventDefault();
          void save();
        }}
      >
        <div className="dialog-heading note-editor-head">
          <div className="note-editor-crumb">
            <h2 id="note-title">{note.id === "new" ? "New note" : "Note"}</h2>
            <span className="note-editor-saved" role="status">
              {dirty ? "Unsaved changes" : note.id === "new" ? "Start with a title" : "Saved " + new Date(note.updated_at).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}
            </span>
          </div>
          {note.id !== "new" && <RecordTools kind="note" id={note.id}/>}
          <button
            type="button"
            className="btn-icon"
            aria-label="Close note"
            disabled={busy}
            onClick={() => dirty ? setDiscarding(true) : onClose()}
          >
            <X size={20} />
          </button>
        </div>
        {(error || localError) && (
          <p className="error-banner" role="alert">
            {error || localError}
          </p>
        )}
        {discarding && <div role="alert" className="draft-warning"><p>Your note has unsaved changes.</p><button type="button" className="btn btn-sm" onClick={() => setDiscarding(false)}>Keep writing</button><button type="button" className="btn btn-danger btn-sm" onClick={onClose}>Discard draft</button></div>}
        <div className="note-doc">
        {writing ? <div className="note-writing">
          <label className="note-title-field">
            <span className="sr-only">Title</span>
            <input
              autoFocus
              required
              maxLength={200}
              value={title}
              placeholder="Untitled note"
              onChange={(e) => setTitle(e.target.value)}
            />
          </label>
          <label className="note-body-field">
            <span className="sr-only">Note</span>
            <textarea
              className="note-body"
              maxLength={30000}
              rows={11}
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Start writing…"
            />
          </label>
        </div> : <div className="note-reading">
          <button type="button" className="note-reading-title" aria-label="Change note title" onClick={() => setWriting(true)}>{title}</button>
          <button type="button" className={"note-reading-body" + (content ? "" : " is-empty")} aria-label="Edit note content" onClick={() => setWriting(true)}>{content || "Add note content"}</button>
        </div>}
        {(homeProject || tagList.length > 0) && (
          <div className="chip-row note-meta" aria-label="Project and tags">
            {homeProject && <span className="chip chip-home"><FolderOpen size={13} aria-hidden />{homeProject}</span>}
            {tagList.map((t) => <span className="chip" key={t}>{t}</span>)}
          </div>
        )}
        {note.id !== "new" && <NoteFiling note={note} readOnly={readOnly} disabled={busy || dirty || note.archived} onSaved={onSaved} onOpen={onOpenNote}/>}
        {connected.length > 0 && (
          <div className="note-backlinks">
            <h3 className="note-section-title">Connected notes and backlinks</h3>
            <div className="chip-row">
              {connected.map((n) => (
                <button
                  key={n.id}
                  type="button"
                  className="chip note-chip-link"
                  disabled={dirty || busy}
                  onClick={() => onOpenNote?.(n.id)}
                >
                  <FileText size={13} aria-hidden />
                  {n.title}
                </button>
              ))}
            </div>
          </div>
        )}
        {(!!note.tasks.length || note.conversation_id) && (
          <div className="note-linked-records">
            <h3 className="note-section-title">Linked work</h3>
            {note.tasks.map((t) => (
              <div className="note-linked-task" key={t.id}>
                <button
                  className="note-linked-open"
                  type="button"
                  disabled={dirty || busy}
                  onClick={() => onTask(t.id)}
                >
                  <span className="note-linked-name">{t.title}</span>
                  <span className={"chip" + (t.status === "done" ? " chip-done" : "")}>{humanLabel(t.status)}</span>
                  <ChevronRight size={16} aria-hidden className="note-linked-chevron" />
                </button>
                {t.evidence && (
                  <small>
                    From note version {t.note_revision}: {t.evidence}
                  </small>
                )}
              </div>
            ))}
            {note.conversation_id && (
              <div className="note-linked-task">
                <button
                  type="button"
                  className="note-linked-open"
                  disabled={dirty || busy}
                  onClick={() => onConversation(note.conversation_id!)}
                >
                  <MessageSquare size={15} aria-hidden />
                  <span className="note-linked-name">Open linked conversation</span>
                  <ChevronRight size={16} aria-hidden className="note-linked-chevron" />
                </button>
              </div>
            )}
          </div>
        )}
        {note.id !== "new" && !note.archived && (
          <section className="note-extraction" aria-label="To-dos in this note">
            <div className="note-extraction-head">
              <button
                type="button"
                className="btn btn-soft btn-sm"
                disabled={busy || extracting || dirty}
                onClick={() => void extract()}
              >
                <Sparkles size={15} />
                {extracting ? "Finding to-dos…" : "Find to-dos"}
              </button>
              <p className="note-hint">
                {dirty
                  ? "Save your edits before extracting to-dos or opening linked records."
                  : "Review suggestions before creating tasks. Existing extractions won't create duplicates."}
              </p>
            </div>
            {note.index_state !== "ready" && (
              <p className="note-hint">
                {note.index_state === "failed"
                  ? "Meaning search could not be prepared. Save again to retry; keyword search works."
                  : "Preparing this note for meaning search."}
              </p>
            )}
            {proposals.map((p, i) => (
              <div className="note-proposal" key={i}>
                <label className="check-label">
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
                className="btn btn-primary"
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
        <details className="note-attribution"><summary>Organization and links</summary>
        <div className="note-attribution-grid">
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
        <details className="note-links">
          <summary>Linked tasks <span className="note-links-count">{linked.length}</span></summary>
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
        </details>
        </div>
        <div className="dialog-actions">
          {note.id !== "new" && (
            <button
              type="button"
              className="btn btn-ghost dialog-actions-start"
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
          {!writing && !dirty && <span className="note-hint">Select the title or text to edit</span>}
          {(writing || dirty) && <button className="btn btn-primary" disabled={busy || extracting || !title.trim()}>Save note</button>}
        </div>
      </Dialog>
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
