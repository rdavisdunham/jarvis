import { ProductivityPage, OrganizationFilters } from "./Productivity";
import { emptyOrganization, emptyFilter, matchesOrganization, scheduledBy, type Organization } from "./productivity";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Bell,
  Brain,
  CalendarDays,
  Check,
  ChevronRight,
  Clock3,
  Inbox,
  ListTodo,
  FileText,
  LogOut,
  Menu,
  MessageCircle,
  Mic,
  Plus,
  Repeat2,
  Search,
  Settings2,
  Shield,
  Sparkles,
  Square,
  Sun,
  Trash2,
  VolumeX,
  X,
} from "lucide-react";
import { api, ApiError, command, post, setCsrf } from "./api";
import { Voice, type VoiceState } from "./voice";
import { subscribeEvents } from "./events";
import {
  NotesPage,
  NoteEditor,
  TaskNotes,
  blankNote,
  type NoteRecord,
} from "./Notes";
import { GoogleSettings, startGoogle } from "./GoogleSettings";
import { GoogleEventDialog } from "./GoogleEventDialog";
import { PlanningDialog } from "./PlanningDialog";
import { LinearSettings, LinearTask } from "./Linear";
import type { CalendarEntry } from "./types";
import { BulkTaskDialog } from "./BulkTaskDialog";
import { Workspace } from "./Workspace";
import { ScheduleDialog } from "./ScheduleDialog";
import type { Project } from "./types";
import { MemoryReviewCard } from "./MemoryReviewCard";
import { WakeWord, recognitionType } from "./wake-word";
import { useSiteControl } from "./copilot";
import type {
  Bootstrap,
  UIAction,
  UIContext,
  VoiceProvider,
  ChatMessage,
  Memory,
  MemoryReview,
  MemoryMaintenance,
  Notice,
  Schedule,
  Task,
  View,
} from "./types";
import {
  dayInZone,
  timeLabel,
  recurrenceLabel,
  TaskRow,
  TaskDialog,
  ReminderDialog,
  MemoryCapture,
  SettingsPanel,
} from "./components";
const nav: { id: View; label: string; icon: typeof Sun }[] = [
  { id: "today", label: "Today", icon: Sun },
  { id: "inbox", label: "Inbox", icon: Inbox },
  { id: "week", label: "This week", icon: CalendarDays },
  { id: "all", label: "Work", icon: ListTodo },
  { id: "organize", label: "Goals & projects", icon: ListTodo },
  { id: "calendar", label: "Calendar", icon: CalendarDays },
  { id: "notes", label: "Notes", icon: FileText },
  { id: "memory", label: "Memory", icon: Brain },
];
export default function App() {
  const [boot, setBoot] = useState<Bootstrap | null>(null),
    [loading, setLoading] = useState(true);
  const [view, setView] = useState<View>(
    new URLSearchParams(location.search).get("view") === "notifications"
      ? "notifications"
      : new URLSearchParams(location.search).get("view") === "settings"
        ? "settings"
        : "today",
  );
  const [tasks, setTasks] = useState<Task[]>([]),
    [schedules, setSchedules] = useState<Schedule[]>([]),
    [notices, setNotices] = useState<Notice[]>([]);
  const [googleLogin, setGoogleLogin] = useState(false);
  const [googleEvent, setGoogleEvent] = useState<CalendarEntry | null>(null);
  const [noteEditor, setNoteEditor] = useState<NoteRecord | null>(null);
  const [noteRevision, setNoteRevision] = useState(0);
  const [noteVisible, setNoteVisible] = useState<string[]>([]);
  const [selectedTaskIds, setSelectedTaskIds] = useState<string[]>([]);
  const [selectingTasks, setSelectingTasks] = useState(false);
  const pendingSelection = useRef<string[] | null>(null);
  const [selectionRequest, setSelectionRequest] = useState(0);
  const [bulkEditor, setBulkEditor] = useState<Task[] | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [organization, setOrganization] = useState<Organization>(emptyOrganization);
  const [organizationFilter, setOrganizationFilter] = useState(emptyFilter);
  const [organizationEditing, setOrganizationEditing] = useState(false);
  const [calendarDay, setCalendarDay] = useState("");
  const [calendarMode, setCalendarMode] = useState<"month" | "week" | "day">(
    () => {
      const stored = localStorage.getItem("eri-calendar-view");
      return stored === "week" || stored === "day" ? stored : "month";
    },
  );
  useEffect(
    () => localStorage.setItem("eri-calendar-view", calendarMode),
    [calendarMode],
  );
  const [workKind, setWorkKind] = useState<"all" | "task" | "reminder">("all");
  const [workVisible, setWorkVisible] = useState<string[]>([]);
  const [scheduleEditor, setScheduleEditor] = useState<{
    schedule: Schedule | null;
    date?: string;
    task?: Task | null;
  } | null>(null);
  const pendingNotices = notices.filter((notice) => !notice.completed_at);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [memoryReviews, setMemoryReviews] = useState<MemoryReview[]>([]);
  const [maintenance, setMaintenance] = useState<MemoryMaintenance | null>(
    null,
  );
  const [query, setQuery] = useState(""),
    [quick, setQuick] = useState(""),
    [pair, setPair] = useState("");
  const [error, setError] = useState(""),
    [toast, setToast] = useState(""),
    [busy, setBusy] = useState(false);
  const [sidebar, setSidebar] = useState(false),
    [companion, setCompanion] = useState(false);
  const [selected, setSelected] = useState<Task | null>(null),
    [reminder, setReminder] = useState(false);
  const [privateMode, setPrivate] = useState(false),
    [messages, setMessages] = useState<ChatMessage[]>([]),
    [chatText, setChatText] = useState(""),
    [thinking, setThinking] = useState(false);
  const [syncWarning, setSyncWarning] = useState("");
  const [voiceState, setVoiceState] = useState<VoiceState | null>(null);
  // Realtime is paused. Old device preferences must not start a disabled session.
  const [voiceProvider, setVoiceProvider] = useState<VoiceProvider>("live");
  const [voiceName, setVoiceName] = useState(
    () => localStorage.getItem("eri-voice-live") || "marin",
  );
  useEffect(() => {
    localStorage.setItem("eri-voice-provider", "live");
  }, []);
  const [taskStatus, setTaskStatus] = useState<
    | "all"
    | "open"
    | "in_progress"
    | "waiting"
    | "deferred"
    | "completed"
    | "cancelled"
  >("all");
  const [projectFilter, setProjectFilter] = useState("");
  const [memoryStatus, setMemoryStatus] = useState({
    enabled: true,
    pending: 0,
    retrying: 0,
    deferred: 0,
  });
  const [memoryRevision, setMemoryRevision] = useState(0);
  const [editingMemory, setEditingMemory] = useState<Memory | null>(null);
  const [mobile, setMobile] = useState(() => window.innerWidth <= 1000);
  const uiResults = useRef<{ id: string; status: string; message?: string }[]>(
    [],
  );
  const syncUIRef = useRef<() => Promise<void>>(async () => {});
  const [highlight, setHighlight] = useState<string | null>(null);
  const [wakeEnabled, setWakeEnabled] = useState(false);
  const [wakeStatus, setWakeStatus] = useState("");
  const wake = useRef<WakeWord | null>(null);
  const wakeStart = useRef<() => void>(() => {});
  const glowRef = useRef<HTMLDivElement>(null);
  const displayedActions = useRef(new Set<string>());
  const voiceGeneration = useRef(0);
  const voice = useRef<Voice | null>(null),
    conversationRef = useRef<string | null>(null),
    retryRef = useRef<null | (() => Promise<void>)>(null),
    lastVoiceReceipt = useRef("");
  const pendingCommands = useRef(
    new Map<string, ReturnType<typeof command<unknown>>>(),
  );
  useEffect(() => {
    if (!boot)
      api<{ google: boolean }>("/auth/options")
        .then((d) => setGoogleLogin(d.google))
        .catch(() => {});
  }, [!!boot]);
  useEffect(() => {
    const result = new URLSearchParams(location.search).get("google");
    if (!result) return;
    history.replaceState({}, "", location.pathname + "?view=settings");
    if (result === "connected") setToast("Google account connected");
    else
      setError(
        (
          {
            cancelled: "Google connection cancelled.",
            permission: "Calendar permission was not granted.",
            account: "Use the Google account already linked to Eri.",
          } as Record<string, string>
        )[result] ??
          "Google sign-in could not finish. Start again from Settings.",
      );
  }, []);
  const searchRef = useRef<HTMLInputElement>(null),
    messageEnd = useRef<HTMLDivElement>(null);
  const load = useCallback(async () => {
    const [taskData, scheduleData, noticeData, projectData] = await Promise.all(
      [
        api<{ items: Task[]; next_cursor: string | null }>("/tasks?limit=200"),
        api<{ items: Schedule[]; next_cursor?: string | null }>("/schedules"),
        api<{ items: Notice[] }>("/notifications"),
        api<Organization>("/organization"),
      ],
    );
    let items = taskData.items,
      cursor = taskData.next_cursor;
    while (cursor) {
      const page = await api<{ items: Task[]; next_cursor: string | null }>(
        "/tasks?limit=200&before=" + cursor,
      );
      items = [...items, ...page.items];
      cursor = page.next_cursor;
    }
    setTasks(items);
    let scheduleItems = scheduleData.items,
      scheduleCursor = scheduleData.next_cursor;
    while (scheduleCursor) {
      const page = await api<{
        items: Schedule[];
        next_cursor?: string | null;
      }>("/schedules?before=" + scheduleCursor);
      scheduleItems = [...scheduleItems, ...page.items];
      scheduleCursor = page.next_cursor;
    }
    setSchedules(scheduleItems);
    setProjects(projectData.projects);
    setOrganization(projectData);
    setNotices(noticeData.items);
    setNoteRevision((n) => n + 1);
  }, []);
  const initialize = useCallback(async () => {
    try {
      const info = await api<Bootstrap>("/bootstrap");
      setCsrf(info.csrf);
      setBoot(info);
      setCalendarDay((day) => day || dayInZone(info.preferences.timezone));
      const saved = sessionStorage.getItem("jarvis-conversation");
      if (saved && !conversationRef.current) {
        try {
          const conversation = await api<{
            id: string;
            private: boolean;
            messages: ChatMessage[];
          }>("/conversations/" + saved);
          conversationRef.current = conversation.id;
          setPrivate(conversation.private);
          setMessages(conversation.messages);
        } catch {
          sessionStorage.removeItem("jarvis-conversation");
        }
      }
      await load();
      setError("");
    } catch (e) {
      if (!(e instanceof ApiError && e.code === "NOT_AUTHORIZED"))
        setError((e as Error).message);
      setBoot(null);
    } finally {
      setLoading(false);
    }
  }, [load]);
  useEffect(() => {
    void initialize();
    return () => {
      void voice.current?.stop();
    };
  }, [initialize]);
  useEffect(() => {
    if (!boot) return;
    const stopEvents = subscribeEvents(
      boot.event_cursor,
      () => {
        setMemoryRevision((v) => v + 1);
        void load().catch(() =>
          setSyncWarning("Task updates are reconnecting…"),
        );
      },
      (online) =>
        setSyncWarning(
          online
            ? ""
            : "Live task updates are reconnecting. Voice can continue.",
        ),
    );
    const timer = setInterval(() => {
      api<Bootstrap>("/bootstrap")
        .then(async (info) => {
          setBoot(info);
          await load();
          setSyncWarning("");
        })
        .catch(() => {});
    }, 30000);
    return () => {
      stopEvents();
      clearInterval(timer);
    };
  }, [!!boot, load]);
  useEffect(() => {
    if (view !== "memory" || !boot) return;
    let current = true;
    const timer = setTimeout(() => {
      api<{
        items: Memory[];
        reviews: MemoryReview[];
        maintenance: MemoryMaintenance;
        learning: typeof memoryStatus;
      }>("/memory?q=" + encodeURIComponent(query))
        .then((data) => {
          if (current) {
            setMemories(data.items);
            setMemoryReviews(data.reviews ?? []);
            setMaintenance(data.maintenance);
            if (data.learning) setMemoryStatus(data.learning);
          }
        })
        .catch((e) => setError(e.message));
    }, 200);
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [view, query, !!boot, toast, memoryRevision]);
  useEffect(() => {
    const listen = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        searchRef.current?.focus();
      }
      if (e.key === "Escape") {
        setSelected(null);
        setReminder(false);
        setScheduleEditor(null);
        setNoteEditor(null);
        setGoogleEvent(null);
        setBulkEditor(null);
        setSidebar(false);
        setCompanion(false);
      }
    };
    window.addEventListener("keydown", listen);
    return () => window.removeEventListener("keydown", listen);
  }, []);
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(""), 4500);
      return () => clearTimeout(timer);
    }
  }, [toast]);
  useEffect(() => {
    messageEnd.current?.scrollIntoView({
      block: "nearest",
      behavior: "smooth",
    });
  }, [messages]);
  useEffect(() => {
    const handler = (event: MessageEvent) => {
      if (event.data?.type === "open-inbox") setView("notifications");
    };
    navigator.serviceWorker?.addEventListener("message", handler);
    return () =>
      navigator.serviceWorker?.removeEventListener("message", handler);
  }, []);
  async function mutate<T>(
    tool: string,
    args: unknown,
    success: string,
  ): Promise<T | undefined> {
    // A lost response may hide a committed write. Repeated Save must reuse its receipt.
    const key = JSON.stringify([tool, args]);
    const request =
      pendingCommands.current.get(key) ?? command<unknown>(tool, args);
    pendingCommands.current.set(key, request);
    async function send() {
      setBusy(true);
      setError("");
      try {
        const result = await request.send();
        pendingCommands.current.delete(key);
        await load().catch(() =>
          setSyncWarning(
            "Saved. The workspace will refresh when the connection returns.",
          ),
        );
        setToast(success);
        retryRef.current = null;
        return result.data as T;
      } catch (e) {
        const err = e as ApiError;
        if (err.code !== "NETWORK") pendingCommands.current.delete(key);
        setError(
          tool === "task.batch" && err.code === "REVISION_CONFLICT"
            ? "One of these tasks changed. Close this editor and reopen the selection to review its latest values."
            : err.message,
        );
        if (
          err.code === "REVISION_CONFLICT" &&
          err.data &&
          typeof err.data === "object" &&
          "id" in err.data
        ) {
          if (tool.startsWith("task.") && tool !== "task.batch")
            setSelected(err.data as Task);
          else if (tool.startsWith("schedule."))
            setScheduleEditor({ schedule: err.data as Schedule });
        }
        if (err.code === "NETWORK")
          retryRef.current = async () => {
            await send();
          };
        return undefined;
      } finally {
        setBusy(false);
      }
    }
    return send();
  }
  async function signIn(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      const result = await post<{ csrf: string }>("/auth/login", {
        token: pair.trim(),
      });
      setCsrf(result.csrf);
      setPair("");
      await initialize();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function add(e: React.FormEvent) {
    e.preventDefault();
    if (!quick.trim() || busy) return;
    const result = await mutate<Task>(
      "task.create",
      { title: quick.trim() },
      "Added to your Inbox",
    );
    if (result) setQuick("");
  }
  async function toggle(task: Task) {
    const completed = task.status === "completed";
    await mutate(
      completed ? "task.reopen" : "task.complete",
      { task_id: task.id, expected_revision: task.revision },
      completed ? "Task reopened" : "Task completed",
    );
  }
  async function ensureConversation() {
    if (conversationRef.current) return conversationRef.current;
    const data = await post<{ id: string }>("/conversations", {
      private: privateMode,
    });
    conversationRef.current = data.id;
    sessionStorage.setItem("jarvis-conversation", data.id);
    return data.id;
  }
  useEffect(() => {
    if (!boot?.voice_options) return;
    const options = boot.voice_options[voiceProvider];
    if (!options) return;
    if (!options.voices.includes(voiceName)) {
      setVoiceName(options.default_voice);
      localStorage.setItem("eri-voice-" + voiceProvider, options.default_voice);
    }
  }, [boot?.voice_options, voiceProvider, voiceName]);

  function createTask(date?: string) {
    setSelected({
      id: "new",
      title: "",
      notes: "",
      status: "open",
      priority: 0,
      project: null,
      project_id: projects.find((p) => p.name === projectFilter)?.id ?? null,
      parent_task_id: null,
      assignee: "owner",
      work_type: "",
      tags: [],
      space_id: organizationFilter.space || null,
      area_id: organizationFilter.area || null,
      planned_date: date ?? null,
      due_date: null,
      due_time: null,
      due_timezone: null,
      revision: 1,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      completed_at: null,
      archived: false,
      occurrence_id: null,
    });
  }
  async function openNote(id: string) {
    try {
      const note = await api<NoteRecord>("/notes/" + encodeURIComponent(id));
      setError("");
      setSelected(null);
      setNoteEditor(note);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function openNoteTask(id: string) {
    try {
      const task = await api<Task>("/tasks/" + encodeURIComponent(id));
      setError("");
      setNoteEditor(null);
      setSelected(task);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function openNoteConversation(id: string) {
    if (voice.current) {
      setError("End voice before switching conversations.");
      return;
    }
    try {
      const data = await api<{
        id: string;
        private: boolean;
        messages: ChatMessage[];
      }>("/conversations/" + id);
      conversationRef.current = data.id;
      sessionStorage.setItem("jarvis-conversation", data.id);
      setMessages(data.messages);
      setPrivate(data.private);
      setNoteEditor(null);
      setCompanion(true);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  useEffect(() => {
    const requested = pendingSelection.current;
    pendingSelection.current = null;
    setSelectedTaskIds(requested ?? []);
    setSelectingTasks(requested !== null);
  }, [
    view,
    query,
    taskStatus,
    projectFilter,
    workKind,
    calendarDay,
    selectionRequest,
  ]);

  async function applyAction(action: UIAction) {
    const kind = action.kind ?? "show";
    if (kind === "chat") {
      setCompanion(action.mode === "auto" ? !mobile : action.mode === "open");
      return;
    }
    if (
      (selected && selected.id !== action.entity_id) ||
      reminder ||
      scheduleEditor ||
      organizationEditing ||
      editingMemory ||
      noteEditor ||
      googleEvent ||
      bulkEditor
    )
      throw new Error(
        "An editor is open. Save or close it before changing pages.",
      );
    setSidebar(false);
    if (mobile) setCompanion(false);
    if (kind === "select") {
      const ids = action.task_ids ?? [];
      if (ids.some((id) => !tasks.some((t) => t.id === id)))
        throw new Error(
          "Some tasks are no longer available. Refresh the workspace.",
        );
      setView("all");
      setQuery("");
      setTaskStatus("all");
      setProjectFilter("");
      setOrganizationFilter(emptyFilter);
      setWorkKind("task");
      pendingSelection.current = ids;
      setSelectionRequest((n) => n + 1);
    } else if (kind === "calendar") {
      const day = action.date ?? "";
      if (
        !/^\d{4}-\d{2}-\d{2}$/.test(day) ||
        isNaN(Date.parse(day + "T12:00:00Z")) ||
        new Date(day + "T12:00:00Z").toISOString().slice(0, 10) !== day
      )
        throw new Error("Choose a valid calendar date.");
      if (action.entity_id)
        await api("/calendar/events/" + encodeURIComponent(action.entity_id));
      setHighlight(action.entity_id ?? null);
      setCalendarDay(day);
      if (action.calendar_view || action.entity_id)
        setCalendarMode(action.calendar_view ?? "day");
      setView("calendar");
      setQuery("");
      setTaskStatus("all");
      setProjectFilter("");
      setOrganizationFilter(emptyFilter);
      setWorkKind("all");
    } else if (kind === "search") {
      setQuery(action.query ?? "");
      setTaskStatus("all");
      setProjectFilter("");
      setOrganizationFilter(emptyFilter);
      setWorkKind("all");
      setView(action.view ?? "all");
    } else if (kind === "filter") {
      setView(
        action.view ??
          (["all", "calendar", "reminders"].includes(view) ? view : "all"),
      );
      if (action.status !== undefined) setTaskStatus(action.status);
      if (action.space_id !== undefined || action.area_id !== undefined || action.goal_id !== undefined) setOrganizationFilter(current => ({ space: action.space_id ?? current.space, area: action.area_id ?? current.area, goal: action.goal_id ?? current.goal }));
      if (action.project !== undefined) setProjectFilter(action.project);
      if (action.work_kind !== undefined) setWorkKind(action.work_kind);
    } else if (kind === "form") {
      if (action.form === "reminder") setScheduleEditor({ schedule: null });
      else if (action.form === "note") {
        setView("notes");
        setNoteEditor(blankNote());
      } else {
        setView("all");
        createTask();
      }
    } else {
      setQuery("");
      setTaskStatus("all");
      setProjectFilter("");
      setOrganizationFilter(emptyFilter);
      setWorkKind("all");
      setView(action.view ?? "today");
      if (action.entity_id && action.view === "all") {
        setSelected(
          await api<Task>("/tasks/" + encodeURIComponent(action.entity_id)),
        );
      } else if (action.entity_id && action.view === "organize") {
        const current = await api<Organization>("/organization");
        if (![...current.spaces, ...current.areas, ...current.goals, ...current.projects, ...current.actors].some(row => row.id === action.entity_id))
          throw new Error("That organization record is no longer available.");
        setOrganization(current);
        setProjects(current.projects);
        setHighlight(action.entity_id);
      } else if (action.entity_id && action.view === "notes") {
        // Fetch directly so failures are acknowledged as failed site actions.
        setNoteEditor(
          await api<NoteRecord>(
            "/notes/" + encodeURIComponent(action.entity_id),
          ),
        );
      } else if (action.entity_id && action.view === "reminders") {
        const record = await api<Schedule>(
          "/schedules/" + encodeURIComponent(action.entity_id),
        );
        setSchedules((items) => [
          ...items.filter((item) => item.id !== record.id),
          record,
        ]);
        setHighlight(record.id);
      }
    }
  }
  async function showActions(actions: UIAction[] = []) {
    for (const action of actions) {
      if (displayedActions.current.has(action.id)) continue;
      displayedActions.current.add(action.id);
      try {
        await runSiteControl(action);
        uiResults.current.push({ id: action.id, status: "displayed" });
      } catch (e) {
        const message = (e as Error).message;
        uiResults.current.push({ id: action.id, status: "failed", message });
        setError(message);
      }
    }
  }
  useEffect(() => {
    const resize = () => setMobile(window.innerWidth <= 1000);
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);
  useEffect(() => {
    if (!boot) return;
    let inFlight = false;
    const sync = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        await syncUIRef.current();
      } catch {
        /* retry with the same acknowledgements */
      } finally {
        inFlight = false;
      }
    };
    void sync();
    const timer = setInterval(sync, 700);
    return () => clearInterval(timer);
  }, [!!boot]);
  useEffect(() => {
    if (!highlight || !["reminders", "calendar"].includes(view)) return;
    let attempts = 0;
    const timer = setInterval(() => {
      const target = document.getElementById("record-" + highlight);
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        target.focus({ preventScroll: true });
        clearInterval(timer);
      } else if (++attempts >= 30) clearInterval(timer);
    }, 100);
    const clear = setTimeout(() => setHighlight(null), 12000);
    return () => {
      clearInterval(timer);
      clearTimeout(clear);
    };
  }, [highlight, view]);
  async function newChat() {
    voiceGeneration.current += 1;
    const previous = voice.current;
    voice.current = null;
    setVoiceState(
      previous
        ? {
            state: "closing",
            error: null,
            text: "",
            closed: false,
            receipts: [],
          }
        : null,
    );
    if (previous) await previous.stop();
    conversationRef.current = null;
    sessionStorage.removeItem("jarvis-conversation");
    retryRef.current = null;
    setMessages([]);
    setChatText("");
    setVoiceState(null);
    setError("");
  }
  function chooseProvider(provider: VoiceProvider) {
    if (!boot?.voice_options[provider]) return;
    setVoiceProvider(provider);
    localStorage.setItem("eri-voice-provider", provider);
    const saved = localStorage.getItem("eri-voice-" + provider);
    setVoiceName(
      saved && boot?.voice_options[provider]?.voices.includes(saved)
        ? saved
        : "marin",
    );
  }
  async function sendChat(e?: React.FormEvent) {
    e?.preventDefault();
    if (!chatText.trim() || thinking || voice.current) return;
    const content = chatText.trim(),
      turn_id = crypto.randomUUID();
    setChatText("");
    setThinking(true);
    setMessages((m) => [...m, { id: turn_id, role: "user", content }]);
    let conversation_id: string;
    try {
      conversation_id = await ensureConversation();
    } catch (e) {
      setThinking(false);
      setError((e as Error).message);
      return;
    }
    const body = {
      turn_id,
      conversation_id,
      message: content,
      focus: selected?.id ?? null,
    };
    async function submit() {
      setThinking(true);
      setError("");
      try {
        await syncUIRef.current();
        const result = await post<{
          message: string;
          status: string;
          ui_actions?: UIAction[];
        }>("/chat", body);
        setMessages((m) => [
          ...m.filter((item) => item.id !== turn_id + "-reply"),
          {
            id: turn_id + "-reply",
            role: "assistant",
            content: result.message,
          },
        ]);
        retryRef.current = null;
        await load();
        await showActions(result.ui_actions);
      } catch (e) {
        const err = e as ApiError;
        setError(err.message);
        if (err.code === "NETWORK" || err.code === "IN_PROGRESS")
          retryRef.current = submit;
      } finally {
        setThinking(false);
      }
    }
    await submit();
  }
  async function startVoice() {
    if (voice.current) {
      voiceGeneration.current += 1;
      const previous = voice.current;
      voice.current = null;
      setVoiceState({
        state: "closing",
        error: null,
        text: "",
        closed: false,
        receipts: [],
      });
      await previous.stop();
      setVoiceState(null);
      setMessages((items) =>
        items.map((item) => ({ ...item, pending: false })),
      );
      return;
    }
    if (voiceState && ["connecting", "closing"].includes(voiceState.state))
      return;
    const generation = ++voiceGeneration.current;
    wake.current?.stop();
    setError("");
    setCompanion(true);
    setVoiceState({
      state: "connecting",
      error: null,
      text: "",
      closed: false,
      receipts: [],
    });
    try {
      const id = await ensureConversation();
      if (generation !== voiceGeneration.current) return;
      const controller = new Voice(
        (state) => {
          if (voice.current !== controller) return;
          if (state.closed) voice.current = null;
          setVoiceState(state);
          void showActions(state.ui_actions);
          if (state.text && state.text_id) {
            setMessages((m) => {
              const existing = m.find((item) => item.id === state.text_id);
              if (existing?.content === state.text) return m;
              const message: ChatMessage = {
                id: state.text_id!,
                role: "assistant",
                content: state.text,
              };
              return existing
                ? m.map((item) => (item.id === message.id ? message : item))
                : [...m, message];
            });
          }
          const receipt = state.receipts.at(-1);
          if (receipt && receipt !== lastVoiceReceipt.current) {
            lastVoiceReceipt.current = receipt;
            void load().catch(() => {});
          }
        },
        (message) => {
          if (voice.current !== controller) return;
          setMessages((m) =>
            m.some((item) => item.id === message.id)
              ? m.map((item) => (item.id === message.id ? message : item))
              : [...m, message],
          );
        },
        (level) =>
          glowRef.current?.style.setProperty("--voice-level", String(level)),
      );
      voice.current = controller;
      await syncUIRef.current();
      await controller.start(id, selected?.id, {
        provider: voiceProvider,
        voice: voiceName,
      });
    } catch (e) {
      if (generation !== voiceGeneration.current) return;
      voice.current = null;
      setVoiceState(null);
      setError((e as Error).message);
    }
  }
  wakeStart.current = () => {
    if (!voice.current && !thinking) void startVoice();
  };
  useEffect(() => {
    if (
      !wakeEnabled ||
      !boot ||
      (voiceState && !voiceState.closed) ||
      thinking
    ) {
      wake.current?.stop();
      return;
    }
    let listener: WakeWord | null = null;
    const begin = () => {
      listener?.stop();
      if (document.visibilityState !== "visible") {
        setWakeStatus("Wake word paused while this page is hidden.");
        return;
      }
      listener = new WakeWord(
        () => wakeStart.current(),
        (message) => {
          setWakeStatus(message);
          if (!message.startsWith("Listening")) setWakeEnabled(false);
        },
      );
      wake.current = listener;
      listener.start();
    };
    const timer = setTimeout(begin, 300);
    document.addEventListener("visibilitychange", begin);
    return () => {
      clearTimeout(timer);
      listener?.stop();
      document.removeEventListener("visibilitychange", begin);
    };
  }, [wakeEnabled, !!boot, !!voiceState && !voiceState.closed, thinking]);
  function changePrivacy() {
    if (thinking || voice.current) return;
    setPrivate(!privateMode);
    conversationRef.current = null;
    sessionStorage.removeItem("jarvis-conversation");
    setMessages([]);
  }
  async function enablePush() {
    if (!("PushManager" in window)) {
      setError(
        "This browser does not support notifications. Your Inbox still works.",
      );
      return;
    }
    try {
      const permission = await Notification.requestPermission();
      if (permission !== "granted") {
        setToast("Notifications are off. Reminders stay in your Inbox.");
        return;
      }
      const registration = await navigator.serviceWorker.ready;
      const raw = atob(
        boot!.vapid_public_key.replace(/-/g, "+").replace(/_/g, "/"),
      );
      const key = Uint8Array.from(raw, (c) => c.charCodeAt(0));
      const subscription =
        (await registration.pushManager.getSubscription()) ??
        (await registration.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: key,
        }));
      await post("/push", subscription.toJSON());
      setToast("Notifications enabled on this device");
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const effectivePrivate = privateMode || !boot?.preferences.history_enabled;
  const today = dayInZone(boot?.preferences.timezone ?? "America/Chicago");
  const endOfWeek = new Date(today + "T12:00:00");
  endOfWeek.setDate(endOfWeek.getDate() + 6);
  const weekEnd = endOfWeek.toISOString().slice(0, 10);
  const open = tasks.filter(
    (t) => !["completed", "cancelled"].includes(t.status),
  );
  const filtered = tasks
    .filter((t) => {
      if (
        query &&
        !t.title.toLowerCase().includes(query.toLowerCase()) &&
        !t.project?.toLowerCase().includes(query.toLowerCase())
      )
        return false;
      if (taskStatus !== "all" && t.status !== taskStatus) return false;
      if (projectFilter && t.project !== projectFilter) return false;
      if (!matchesOrganization(t, organizationFilter, organization)) return false;
      if (view === "today") return scheduledBy(t, today);
      if (view === "week") return scheduledBy(t, weekEnd);
      if (view === "inbox") return !t.project_id && !t.area_id && !t.space_id;
      return true;
    })
    .sort(
      (a, b) =>
        b.priority - a.priority ||
        (a.due_date ?? "9999").localeCompare(b.due_date ?? "9999") ||
        (a.due_time ?? "99:99").localeCompare(b.due_time ?? "99:99") ||
        b.created_at.localeCompare(a.created_at),
    );
  const active = filtered.filter(
      (t) => !["completed", "cancelled"].includes(t.status),
    ),
    done = filtered.filter((t) => t.status === "completed"),
    unread = notices.filter((n) => !n.read_at).length;
  const uiContext: UIContext = {
    view,
    chat_open: companion,
    mobile,
    voice_active: !!voiceState && !voiceState.closed,
    query: query.slice(0, 300),
    selected_task_id: selected?.id === "new" ? null : (selected?.id ?? null),
    selected_schedule_id: scheduleEditor?.schedule?.id ?? null,
    selected_task_ids: selectedTaskIds
      .filter((id) => tasks.some((t) => t.id === id))
      .slice(0, 100),
    selected_calendar_event_id:
      googleEvent?.kind === "google" ? googleEvent.entity_id : null,
    selected_note_id:
      noteEditor?.id === "new" ? null : (noteEditor?.id ?? null),
    calendar_date: calendarDay || undefined,
    calendar_view: calendarMode,
    space_id: organizationFilter.space,
    area_id: organizationFilter.area,
    goal_id: organizationFilter.goal,
    work_kind: view === "reminders" ? "reminder" : workKind,
    visible_ids:
      view === "notes"
        ? noteVisible
        : ["all", "calendar", "reminders"].includes(view)
          ? workVisible
          : (view === "reminders"
              ? schedules
              : view === "memory"
                ? memories
                : filtered
            )
              .slice(0, 60)
              .map((item) => item.id),
    task_status: taskStatus,
    project: projectFilter,
  };
  const runSiteControl = useSiteControl(uiContext, applyAction, !!boot);
  syncUIRef.current = async () => {
    const acknowledged = [...uiResults.current];
    const response = await post<{ actions: UIAction[] }>("/ui/sync", {
      context: uiContext,
      results: acknowledged,
    });
    uiResults.current = uiResults.current.filter(
      (r) => !acknowledged.includes(r),
    );
    await showActions(response.actions);
  };
  const titles: Record<View, string> = {
    organize: "Outcomes, connected to your work.",
    today: "A little space for your day.",
    inbox: "Catch it. Clear your head.",
    week: "A look at the week ahead.",
    all: "Everything, in one place.",
    reminders: "The things worth a nudge.",
    calendar: "Make room for what matters.",
    notes: "Thoughts, connected to your work.",
    memory: "A companion that remembers.",
    notifications: "Your reminders are here.",
    settings: "Make yourself at home.",
  };
  if (loading)
    return (
      <div className="splash">
        <div className="brand-mark">
          E<span>·</span>
        </div>
        <p>Opening Eridani…</p>
      </div>
    );
  if (!boot)
    return (
      <div className="login-page">
        <form className="login-card" onSubmit={signIn}>
          <div className="brand-mark">
            E<span>·</span>
          </div>
          <p className="eyebrow">PRIVATELY YOURS</p>
          <h1>Welcome home.</h1>
          <p>
            Your day, with Eri.
            <br />
            Pair this device to get started.
          </p>
          {googleLogin && (
            <button
              type="button"
              className="google-signin"
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                setError("");
                try {
                  await startGoogle("login");
                } catch (e) {
                  setError((e as Error).message);
                  setBusy(false);
                }
              }}
            >
              <img src="/google-sign-in.png" alt="Sign in with Google" />
            </button>
          )}
          <label>
            Pairing code
            <input
              autoComplete="current-password"
              type="password"
              value={pair}
              onChange={(e) => setPair(e.target.value)}
              autoFocus
              placeholder="Enter your pairing code"
            />
          </label>
          <button className="primary" disabled={busy || !pair}>
            {busy ? "Connecting…" : "Connect to Eridani"}
            <ChevronRight size={17} />
          </button>
          {error && (
            <p role="alert" className="error-text">
              {error}
            </p>
          )}
          <small>The pairing code is stored on your home server.</small>
        </form>
      </div>
    );
  return (
    <div
      className={
        "app-shell " + (voiceState && !voiceState.closed ? "voice-active" : "")
      }
    >
      {sidebar && (
        <button
          className="nav-scrim"
          aria-label="Close navigation"
          onClick={() => setSidebar(false)}
        />
      )}
      <aside className={"sidebar " + (sidebar ? "open" : "")}>
        <a
          className="brand"
          href="/"
          onClick={(e) => {
            e.preventDefault();
            setView("today");
          }}
        >
          <span className="brand-mark small">
            E<span>·</span>
          </span>
          <span>eridani</span>
          <span className="brand-caption">PERSONAL</span>
        </a>
        <div className="nav-label">YOUR SPACE</div>
        <nav>
          {nav.map((item) => (
            <button
              key={item.id}
              aria-label={item.label}
              className={view === item.id ? "nav-item active" : "nav-item"}
              onClick={() => {
                setView(item.id);
                setQuery("");
                setSidebar(false);
              }}
            >
              <item.icon size={18} />
              <span>{item.label}</span>
              {item.id === "inbox" &&
                open.filter((t) => !t.project).length > 0 && (
                  <span className="count">
                    {open.filter((t) => !t.project).length}
                  </span>
                )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <button
            className={"nav-item " + (view === "settings" ? "active" : "")}
            onClick={() => {
              setView("settings");
              setSidebar(false);
            }}
          >
            <Settings2 size={18} />
            <span>Settings</span>
          </button>
          <div className="owner">
            <span className="avatar">{boot.name[0]}</span>
            <div>
              <strong>{boot.name}</strong>
              <span>
                <i className="status-dot" />
                Connected
              </span>
            </div>
            <button
              className="icon-button"
              aria-label="Sign out"
              onClick={async () => {
                await voice.current?.stop();
                await post("/auth/logout");
                setBoot(null);
                setTasks([]);
                setMessages([]);
                conversationRef.current = null;
                sessionStorage.removeItem("jarvis-conversation");
              }}
            >
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            aria-label="Open navigation"
            onClick={() => setSidebar(true)}
          >
            <Menu size={21} />
          </button>
          <div className="breadcrumb">
            <span>Your space</span>
            <ChevronRight size={14} />
            <strong>
              {nav.find((n) => n.id === view)?.label ??
                (view === "settings"
                  ? "Settings"
                  : view === "reminders"
                    ? "Reminders"
                    : "Notifications")}
            </strong>
          </div>
          <div className="top-actions">
            <div className="search">
              <Search size={16} />
              <input
                ref={searchRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={
                  view === "notes"
                    ? "Search notes…"
                    : view === "memory"
                      ? "Search memories"
                      : "Search work…"
                }
                aria-label="Search"
              />
              <kbd>⌘ K</kbd>
            </div>
            <button
              className={"icon-button " + (unread ? "has-notice" : "")}
              aria-label={
                "Notifications" + (unread ? ", " + unread + " unread" : "")
              }
              onClick={() => setView("notifications")}
            >
              <Bell size={19} />
              {unread > 0 && <i />}
            </button>
            <button
              className="icon-button companion-toggle"
              aria-label="Open Eridani"
              onClick={() => setCompanion(!companion)}
            >
              <MessageCircle size={20} />
            </button>
          </div>
        </header>
        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            {retryRef.current && !noteEditor && !bulkEditor && (
              <button onClick={() => void retryRef.current?.()}>
                Retry same request
              </button>
            )}
            <button
              className="icon-button"
              aria-label="Dismiss error"
              onClick={() => setError("")}
            >
              <X size={16} />
            </button>
          </div>
        )}
        <div className="workspace">
          <main className="content">
            <div className="page-heading">
              <p className="eyebrow">
                {view === "today"
                  ? new Intl.DateTimeFormat(undefined, {
                      weekday: "long",
                      month: "long",
                      day: "numeric",
                      timeZone: boot.preferences.timezone,
                    }).format(new Date())
                  : "YOUR PERSONAL SPACE"}
              </p>
              <h1>
                {view === "today"
                  ? "Good " +
                    (new Date().getHours() < 12
                      ? "morning"
                      : new Date().getHours() < 18
                        ? "afternoon"
                        : "evening") +
                    ", " +
                    boot.name +
                    "."
                  : (nav.find((n) => n.id === view)?.label ??
                    (view === "settings"
                      ? "Settings"
                      : view === "reminders"
                        ? "Reminders"
                        : "Notifications"))}
              </h1>
              <p>{titles[view]}</p>
            </div>
            {[
              "today",
              "inbox",
              "week",
              "all",
              "calendar",
              "reminders",
            ].includes(view) && (
              <div className="task-filters">
                <OrganizationFilters organization={organization} value={organizationFilter} onChange={setOrganizationFilter} />
                <label>
                  Status
                  <select
                    aria-label="Task status filter"
                    value={taskStatus}
                    onChange={(e) =>
                      setTaskStatus(e.target.value as typeof taskStatus)
                    }
                  >
                    <option value="all">All</option>
                    <option value="open">Open</option>
                    <option value="in_progress">In progress</option>
                    <option value="waiting">Waiting</option>
                    <option value="deferred">Deferred</option>
                    <option value="cancelled">Cancelled</option>
                    <option value="completed">Completed</option>
                  </select>
                </label>
                <label>
                  Project
                  <select
                    aria-label="Project filter"
                    value={projectFilter}
                    onChange={(e) => setProjectFilter(e.target.value)}
                  >
                    <option value="">All projects</option>
                    {projects.map((p) => (
                      <option key={p.id} value={p.name}>
                        {p.name}
                        {p.archived ? " (archived)" : ""}
                      </option>
                    ))}
                  </select>
                </label>
                {["all", "calendar", "reminders"].includes(view) && (
                  <label>
                    Kind
                    <select
                      aria-label="Work kind filter"
                      value={workKind}
                      onChange={(e) =>
                        setWorkKind(e.target.value as typeof workKind)
                      }
                    >
                      <option value="all">
                        {view === "calendar" ? "All items" : "All tasks"}
                      </option>
                      <option value="task">Tasks</option>
                      <option value="reminder">Reminders</option>
                    </select>
                  </label>
                )}
              </div>
            )}
            {["all", "calendar", "reminders"].includes(view) && (
              <>
                <Workspace
                  onGoogleEvent={setGoogleEvent}
                  organization={organization}
                  organizationFilter={organizationFilter}
                  calendar={view === "calendar"}
                  calendarMode={calendarMode}
                  onCalendarMode={setCalendarMode}
                  onCreateEvent={(day) =>
                    setGoogleEvent({
                      id: "new",
                      entity_id: "new",
                      kind: "event",
                      title: "New event",
                      date: day,
                      at: null,
                      status: "active",
                      project_id: null,
                      task_id: null,
                      revision: 1,
                      projected: false,
                      notification_id: null,
                    })
                  }
                  day={calendarDay || today}
                  onDay={setCalendarDay}
                  today={today}
                  tasks={tasks}
                  schedules={schedules}
                  notices={notices}
                  projects={projects}
                  zone={boot.preferences.timezone}
                  query={query}
                  status={taskStatus}
                  project={projectFilter}
                  kind={view === "reminders" ? "reminder" : workKind}
                  highlight={highlight}
                  busy={busy}
                  onTask={setSelected}
                  onSchedule={(schedule) => setScheduleEditor({ schedule })}
                  toggle={(task) => void toggle(task)}
                  createTask={createTask}
                  createReminder={(date) =>
                    setScheduleEditor({ schedule: null, date })
                  }
                  mutate={mutate}
                  onVisible={setWorkVisible}
                  selecting={selectingTasks}
                  selectedIds={selectedTaskIds}
                  onSelecting={(value) => {
                    setSelectingTasks(value);
                    if (!value) setSelectedTaskIds([]);
                  }}
                  onSelection={setSelectedTaskIds}
                  onBulk={() => {
                    setError("");
                    setBulkEditor(
                      tasks.filter((t) => selectedTaskIds.includes(t.id)),
                    );
                  }}
                />

              </>
            )}
            {["today", "inbox", "week"].includes(view) && (
              <>
                <form className="quick-add" onSubmit={add}>
                  <Plus size={21} />
                  <input
                    value={quick}
                    onChange={(e) => setQuick(e.target.value)}
                    aria-label="New task"
                    placeholder="What's on your mind? Add a task…"
                    maxLength={500}
                  />
                  <button disabled={!quick.trim() || busy} type="submit">
                    {busy ? "Saving…" : "Add task"}
                    <span>↵</span>
                  </button>
                </form>
                <div className="section-head">
                  <h2>
                    {view === "today"
                      ? "Your focus"
                      : view === "week"
                        ? "Coming up"
                        : "Tasks"}
                    <span>{active.length}</span>
                  </h2>
                  <button
                    className="text-button"
                    onClick={() => setReminder(true)}
                  >
                    <Clock3 size={15} />
                    Set a reminder
                  </button>
                </div>
                {active.length ? (
                  <div className="task-list">
                    {active.map((task) => (
                      <TaskRow
                        key={task.id}
                        task={task}
                        today={today}
                        busy={busy}
                        onToggle={() => void toggle(task)}
                        onOpen={() => setSelected(task)}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="empty-state">
                    <div className="empty-symbol">
                      <Check size={27} />
                    </div>
                    <h3>
                      {query
                        ? "Nothing matches that search."
                        : view === "today"
                          ? "A clear day ahead."
                          : "Room for what matters."}
                    </h3>
                    <p>
                      {query
                        ? "Try another word or check All tasks."
                        : view === "today"
                          ? "Add a task, give it a due date, or ask Eridani to help plan your day."
                          : "Capture a thought above, or tell Eridani what you need to do."}
                    </p>
                    {view === "today" && open.length > 0 && (
                      <button
                        className="text-button"
                        onClick={() => setView("inbox")}
                      >
                        See your Inbox
                        <ChevronRight size={15} />
                      </button>
                    )}
                  </div>
                )}
                {done.length > 0 && (
                  <details className="completed">
                    <summary>
                      <ChevronRight size={14} />
                      Completed<span>{done.length}</span>
                    </summary>
                    {done.map((task) => (
                      <TaskRow
                        key={task.id}
                        task={task}
                        today={today}
                        busy={busy}
                        onToggle={() => void toggle(task)}
                        onOpen={() => setSelected(task)}
                      />
                    ))}
                  </details>
                )}
                {view === "today" &&
                  schedules.some((s) => s.status === "active") && (
                    <section className="up-next">
                      <div className="section-head">
                        <h2>Next reminder</h2>
                        <button
                          className="text-button"
                          onClick={() => setView("reminders")}
                        >
                          View all
                          <ChevronRight size={14} />
                        </button>
                      </div>
                      {schedules
                        .filter((s) => s.status === "active" && s.next_run_at)
                        .sort((a, b) =>
                          a.next_run_at!.localeCompare(b.next_run_at!),
                        )
                        .slice(0, 1)
                        .map((s) => (
                          <div key={s.id} className="next-reminder">
                            <Clock3 size={19} />
                            <div>
                              <strong>{s.title}</strong>
                              <span>
                                {timeLabel(
                                  s.next_run_at!,
                                  boot.preferences.timezone,
                                )}
                              </span>
                            </div>
                            {s.recurrence && <Repeat2 size={16} />}
                          </div>
                        ))}
                    </section>
                  )}
              </>
            )}
            {view === "notifications" && (
              <>
                <div className="section-head">
                  <h2>
                    Your Inbox<span>{pendingNotices.length}</span>
                  </h2>
                  <button
                    className="text-button"
                    onClick={() => void enablePush()}
                  >
                    <Bell size={15} />
                    Enable notifications
                  </button>
                </div>
                {pendingNotices.map((n) => (
                  <article
                    className={"notice " + (!n.read_at ? "unread" : "")}
                    key={n.id}
                  >
                    <div className="notice-heading">
                      <Bell size={17} />
                      <strong>{n.title}</strong>
                      <span>
                        {timeLabel(n.scheduled_at, boot.preferences.timezone)}
                      </span>
                    </div>
                    {n.body && <p>{n.body}</p>}
                    <div className="notice-actions">
                      <button
                        onClick={() =>
                          void mutate(
                            "notification.complete",
                            { notification_id: n.id },
                            "Reminder completed",
                          )
                        }
                      >
                        <Check size={14} />
                        Complete
                      </button>
                      <button
                        onClick={() =>
                          void mutate(
                            "notification.snooze",
                            { notification_id: n.id, minutes: 10 },
                            "Snoozed for 10 minutes",
                          )
                        }
                      >
                        <Clock3 size={14} />
                        10 min
                      </button>
                      <button
                        onClick={() =>
                          void mutate(
                            "notification.dismiss",
                            { notification_id: n.id },
                            "Dismissed",
                          )
                        }
                      >
                        Dismiss
                      </button>
                    </div>
                  </article>
                ))}
                {!pendingNotices.length && (
                  <div className="empty-state">
                    <Bell size={30} />
                    <h3>You're all caught up.</h3>
                    <p>Reminders appear here when they are due.</p>
                  </div>
                )}
              </>
            )}
            {view === "organize" && (
              <ProductivityPage organization={organization} busy={busy} mutate={mutate} query={query} highlight={highlight}
                onEditing={setOrganizationEditing}
                onProject={(p) => { setProjectFilter(p.name); setOrganizationFilter(emptyFilter); setQuery(""); setView("all"); }}
                onNote={(id) => { void openNote(id); }}
              />
            )}
            {view === "notes" && (
              <NotesPage
                organization={organization}
                organizationFilter={organizationFilter}
                onOrganizationFilter={setOrganizationFilter}
                query={query}
                projects={projects}
                project={projectFilter}
                onProject={setProjectFilter}
                revision={noteRevision}
                onVisible={setNoteVisible}
                onOpen={(id) => void openNote(id)}
                onNew={() => {
                  setError("");
                  setNoteEditor(blankNote());
                }}
              />
            )}
            {view === "memory" && (
              <>
                <p className="learning-status" role="status">
                  {memoryStatus.enabled
                    ? "Automatic learning is on"
                    : "Automatic learning is off"}
                  {memoryStatus.pending > 0
                    ? " · Learning from " +
                      memoryStatus.pending +
                      " saved messages…"
                    : memoryStatus.deferred > 0
                      ? " · Paused near the model budget; resumes when room is available"
                      : memoryStatus.retrying > 0
                        ? ""
                        : " · Up to date"}
                  {memoryStatus.retrying > 0
                    ? " · Some memories need a retry"
                    : ""}
                </p>

                {memoryStatus.retrying > 0 && (
                  <button
                    className="text-button"
                    onClick={async () => {
                      const result = await post<{ queued: number }>(
                        "/memory/retry",
                      );
                      setToast(
                        result.queued
                          ? "Memory learning queued again"
                          : "Learning is already retrying",
                      );
                      setMemoryRevision((v) => v + 1);
                    }}
                  >
                    Retry memory learning
                  </button>
                )}
                <div className="memory-maintenance">
                  <p className="subtle">
                    {maintenance?.enabled
                      ? maintenance.status === "failed"
                        ? "Deep sleep could not finish. Your memories are unchanged; retry the review."
                        : maintenance.status === "retrying"
                          ? "Deep sleep hit a problem and is waiting to retry."
                          : maintenance.running
                            ? "Deep sleep is reviewing memories…"
                            : "Weekly deep sleep · Next review " +
                              new Date(maintenance.next_run_at).toLocaleString(
                                [],
                                {
                                  month: "short",
                                  day: "numeric",
                                  hour: "numeric",
                                  minute: "2-digit",
                                  timeZone: boot.preferences.timezone,
                                },
                              )
                      : "Weekly deep sleep is off"}
                  </p>
                  <button
                    className="text-button"
                    disabled={
                      busy || !maintenance?.enabled || maintenance.running
                    }
                    onClick={async () => {
                      try {
                        await post("/memory/review");
                        setToast("Memory review queued");
                        setMemoryRevision((v) => v + 1);
                      } catch (e) {
                        setError((e as Error).message);
                      }
                    }}
                  >
                    {maintenance?.status === "failed"
                      ? "Retry review"
                      : "Review now"}
                  </button>
                  {maintenance?.last_run_at && (
                    <p className="footnote">
                      Last successful review:{" "}
                      {new Date(maintenance.last_run_at).toLocaleString([], {
                        timeZone: boot.preferences.timezone,
                      })}
                    </p>
                  )}
                </div>
                {memoryReviews.map((review) => (
                  <MemoryReviewCard
                    key={review.id}
                    review={review}
                    busy={busy}
                    onResolve={(action, content) =>
                      mutate(
                        "memory.resolve",
                        {
                          review_id: review.id,
                          expected_revision: review.revision,
                          action,
                          ...(content ? { content } : {}),
                        },
                        action === "merge"
                          ? "Memory corrected"
                          : action === "distinct"
                            ? "Kept as separate facts"
                            : "Eri can ask next week",
                      )
                    }
                  />
                ))}
                <MemoryCapture
                  busy={busy}
                  onCapture={(content) =>
                    mutate(
                      "memory.capture",
                      { content },
                      "Memory saved with its source",
                    )
                  }
                />
                <div className="section-head">
                  <h2>
                    Remembered<span>{memories.length}</span>
                  </h2>
                  <span className="subtle">Source-backed</span>
                </div>
                {memories.map((m) => (
                  <article className="memory" key={m.id}>
                    <p>{m.content}</p>
                    {!!m.tags?.length && (
                      <div className="memory-tags">
                        {m.tags.map((tag) => (
                          <span key={tag}>{tag}</span>
                        ))}
                      </div>
                    )}
                    <footer>
                      <span>
                        {m.attribution === "owner_statement"
                          ? "You shared this"
                          : "Derived from a source"}
                      </span>
                      <button
                        className="text-button"
                        onClick={async () => {
                          const source = await api<{ content: string }>(
                            "/sources/" + m.source_id,
                          );
                          setMessages((ms) => [
                            ...ms,
                            {
                              id: crypto.randomUUID(),
                              role: "assistant",
                              content: "Original source:\n" + source.content,
                            },
                          ]);
                          setCompanion(true);
                        }}
                      >
                        View source
                        <ChevronRight size={13} />
                      </button>
                      <button
                        className="text-button"
                        onClick={() => setEditingMemory(m)}
                      >
                        Correct
                      </button>
                      <button
                        className="icon-button"
                        aria-label="Forget this memory"
                        onClick={() =>
                          void mutate(
                            "memory.forget",
                            { memory_id: m.id, delete_source: true },
                            "Memory and its source deleted",
                          )
                        }
                      >
                        <Trash2 size={14} />
                      </button>
                    </footer>
                  </article>
                ))}
                {!memories.length && (
                  <div className="empty-state">
                    <Brain size={30} />
                    <h3>
                      {query
                        ? "No matching memories yet."
                        : "Keep the useful little things."}
                    </h3>
                    <p>
                      Eri learns useful facts and preferences from saved
                      conversations. You can also save something above.
                    </p>
                  </div>
                )}
              </>
            )}
            {view === "settings" && (
              <>
                <section className="voice-settings settings-sections">
                  <h2>Voice & conversation</h2>
                  <p>
                    Choose how Eri sounds on this device. Changes apply to the
                    next voice session. Voice ends after 15 quiet seconds
                    following the last response.
                  </p>
                  <div className="voice-options">
                    {Object.keys(boot.voice_options).length > 1 && (
                      <label>
                        Voice mode
                        <select
                          aria-label="Voice provider"
                          value={voiceProvider}
                          disabled={!!voiceState && !voiceState.closed}
                          onChange={(e) =>
                            chooseProvider(e.target.value as VoiceProvider)
                          }
                        >
                          {Object.entries(boot.voice_options).map(([provider, option]) => (
                            <option key={provider} value={provider}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                    <label>
                      Voice
                      <select
                        aria-label="Voice"
                        value={voiceName}
                        disabled={!!voiceState && !voiceState.closed}
                        onChange={(e) => {
                          setVoiceName(e.target.value);
                          localStorage.setItem(
                            "eri-voice-" + voiceProvider,
                            e.target.value,
                          );
                        }}
                      >
                        {(
                          boot.voice_options?.[voiceProvider]?.voices ?? [
                            "marin",
                          ]
                        ).map((name) => (
                          <option key={name} value={name}>
                            {name.charAt(0).toUpperCase() + name.slice(1)}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  <p className="voice-model-note">
                    {voiceProvider === "live"
                      ? "GPT-Live · natural, simultaneous listening and speaking · $0.05 per connected minute, plus task work."
                      : "Realtime · the existing turn-based voice experience."}
                  </p>
                  <p className="voice-model-note">
                    Task agent: {boot.agent_model || "gpt-5.4-mini"}. GPT-Live
                    delegates task work to this agent using your saved records
                    and tools.
                  </p>
                  {voiceState && !voiceState.closed && (
                    <p className="voice-model-note">
                      End the active voice session before changing its voice.
                    </p>
                  )}
                  <div className="wake-controls">
                    <label>
                      <input
                        type="checkbox"
                        checked={wakeEnabled}
                        disabled={!recognitionType()}
                        onChange={(e) => {
                          setWakeEnabled(e.target.checked);
                          if (!e.target.checked) setWakeStatus("");
                        }}
                      />
                      “Eri” or “Hey, Eri”
                    </label>
                    <span>
                      {wakeEnabled
                        ? voiceState && !voiceState.closed
                          ? "Paused during voice"
                          : wakeStatus
                        : wakeStatus ||
                          (recognitionType()
                            ? "Opt in · browser speech service · page open"
                            : "Unavailable in this browser")}
                    </span>
                  </div>
                </section>
                <LinearSettings revision={noteRevision} mutate={mutate} />
                <GoogleSettings
                  revision={noteRevision}
                  voiceActive={!!voiceState && !voiceState.closed}
                  mutate={mutate}
                />
                <SettingsPanel
                  boot={boot}
                  busy={busy}
                  onSave={async (args) => {
                    await mutate("settings.update", args, "Preferences saved");
                    const data = await api<Bootstrap>("/bootstrap");
                    setBoot(data);
                  }}
                  onPush={enablePush}
                />
              </>
            )}
            <footer className="page-footer">
              <Shield size={13} />
              <span>{syncWarning || "Up to date"}</span>
            </footer>
          </main>
          <aside
            className={
              "companion " +
              (companion ? "visible " : "") +
              (voiceState && !voiceState.closed ? "voice-mode" : "")
            }
            aria-label="Eridani conversation"
          >
            <div className="companion-header">
              <button
                className="text-button new-chat"
                disabled={
                  thinking ||
                  (!!voiceState &&
                    ["connecting", "closing"].includes(voiceState.state))
                }
                onClick={() => void newChat()}
              >
                <Plus size={16} />
                New chat
              </button>
              <div className="grow" />
              <button
                className="icon-button"
                aria-label="Voice settings"
                title="Voice settings"
                onClick={() => {
                  setView("settings");
                  setCompanion(false);
                }}
              >
                <Settings2 size={16} />
              </button>
              <button
                className={
                  "icon-button privacy " + (effectivePrivate ? "on" : "")
                }
                aria-label={
                  effectivePrivate
                    ? "Private session on"
                    : "Start a private session"
                }
                title={
                  effectivePrivate
                    ? "Private: no transcript history"
                    : "Start a private conversation"
                }
                disabled={
                  thinking ||
                  !!voice.current ||
                  !boot.preferences.history_enabled
                }
                onClick={changePrivacy}
              >
                <Shield size={17} />
              </button>
              <button
                className="icon-button close-companion"
                aria-label="Close conversation"
                onClick={() => setCompanion(false)}
              >
                <X size={18} />
              </button>
            </div>
            {effectivePrivate && (
              <div className="private-note">
                <Shield size={13} />
                Private conversation · Tasks still save
              </div>
            )}
            <div className="messages">
              {!messages.length && (
                <div className="conversation-empty">
                  <Sparkles size={23} />
                  <p>What can I help with?</p>
                </div>
              )}
              {messages.map((m) => (
                <div className={"message " + m.role} key={m.id}>
                  <span className="message-label">
                    {m.role === "assistant" ? "ERIDANI" : "YOU"}
                  </span>
                  <p>{m.content || (m.pending ? "Listening…" : "")}</p>
                  {m.pending && (
                    <span className="transcript-status">
                      {m.role === "user" ? "Transcribing…" : "Speaking…"}
                    </span>
                  )}
                </div>
              ))}
              {thinking && (
                <div className="thinking">
                  <span />
                  Thinking and checking your saved data…
                </div>
              )}
              <div ref={messageEnd} />
            </div>
            {voiceState && !voiceState.closed && (
              <div className="voice-panel">
                <span className="status-dot" />
                <span>
                  {voiceState.idle_seconds !== null &&
                  voiceState.idle_seconds !== undefined &&
                  voiceState.idle_seconds <= 5
                    ? "Voice ends in " + voiceState.idle_seconds + "s"
                    : voiceLabel(voiceState.state)}
                </span>
                <button
                  className="icon-button"
                  aria-label="Stop speaking"
                  onClick={() => {
                    void voice.current
                      ?.interrupt()
                      .catch((e) => setError(e.message));
                  }}
                >
                  <VolumeX size={17} />
                </button>
                {voiceProvider === "realtime" && (
                  <button
                    className="text-button"
                    title="Ask Eri to answer your latest speech after you finish talking."
                    disabled={!voiceState.can_submit}
                    onClick={() => {
                      void voice.current
                        ?.submit()
                        .catch((e) => setError(e.message));
                    }}
                  >
                    Respond now
                  </button>
                )}
              </div>
            )}
            {voiceState?.error && (
              <p className="voice-error">{voiceState.error}</p>
            )}
            <form className="chat-compose" onSubmit={sendChat}>
              <textarea
                value={chatText}
                onChange={(e) => setChatText(e.target.value)}
                placeholder={
                  boot.capabilities.chat
                    ? "Ask Eridani anything…"
                    : "Chat needs a configured API key"
                }
                aria-label="Message Eridani"
                rows={2}
                disabled={!!voiceState && !voiceState.closed}
                maxLength={12000}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void sendChat();
                  }
                }}
              />
              <div>
                <button
                  type="button"
                  className={
                    "voice-button " + (voice.current ? "recording" : "")
                  }
                  disabled={
                    !boot.capabilities.voice ||
                    thinking ||
                    (!!voiceState &&
                      ["connecting", "closing"].includes(voiceState.state))
                  }
                  onClick={() => void startVoice()}
                  title={
                    boot.capabilities.voice
                      ? "Start voice"
                      : "Realtime needs an OpenAI API key"
                  }
                >
                  {voice.current ? <Square size={15} /> : <Mic size={17} />}
                  <span>{voice.current ? "End voice" : "Talk to Eridani"}</span>
                </button>
                <button
                  className="send-button"
                  type="submit"
                  aria-label="Send message"
                  disabled={
                    !chatText.trim() ||
                    thinking ||
                    (!!voiceState && !voiceState.closed) ||
                    !boot.capabilities.chat
                  }
                >
                  <ArrowUp size={18} />
                </button>
              </div>
            </form>

            {!boot.capabilities.voice && (
              <p className="chat-footnote">
                Voice needs an OpenAI API key. Text is ready.
              </p>
            )}
            <p className="chat-footnote">
              {effectivePrivate
                ? "This conversation won’t be retained."
                : "Conversation history is on."}{" "}
              Cloud models process what you send.
            </p>
          </aside>
        </div>
      </div>
      <div className="voice-glow" ref={glowRef} aria-hidden="true">
        <i />
        <i />
        <i />
      </div>
      {voiceState && !voiceState.closed && !companion && (
        <div className="voice-dock" aria-label="Active voice controls">
          <button className="text-button" onClick={() => setCompanion(true)}>
            <MessageCircle size={17} />
            {voiceState.idle_seconds !== null &&
            voiceState.idle_seconds !== undefined &&
            voiceState.idle_seconds <= 5
              ? "Listening · " + voiceState.idle_seconds + "s"
              : voiceLabel(voiceState.state)}
          </button>
          <button
            className="icon-button"
            aria-label="Stop speaking"
            onClick={() =>
              void voice.current?.interrupt().catch((e) => setError(e.message))
            }
          >
            <VolumeX size={17} />
          </button>
          <button
            className="icon-button"
            aria-label="End voice"
            disabled={voiceState.state === "closing"}
            onClick={() => void startVoice()}
          >
            <Square size={16} />
          </button>
        </div>
      )}
      {editingMemory && (
        <div className="modal-backdrop">
          <form
            className="dialog"
            onSubmit={async (e) => {
              e.preventDefault();
              const content = String(
                new FormData(e.currentTarget).get("content"),
              ).trim();
              if (
                content &&
                (await mutate(
                  "memory.correct",
                  { memory_id: editingMemory.id, content },
                  "Memory corrected",
                ))
              )
                setEditingMemory(null);
            }}
          >
            <div className="dialog-heading">
              <h2>Correct memory</h2>
              <button
                type="button"
                className="icon-button"
                aria-label="Close memory editor"
                onClick={() => setEditingMemory(null)}
              >
                <X size={18} />
              </button>
            </div>
            <textarea
              name="content"
              aria-label="Memory correction"
              defaultValue={editingMemory.content}
              required
              maxLength={20000}
              rows={5}
            />
            <button className="primary" disabled={busy}>
              Save correction
            </button>
          </form>
        </div>
      )}
      {toast && (
        <div className="toast" role="status">
          <Check size={16} />
          {toast}
        </div>
      )}
      {selected && (
        <TaskDialog
          organization={organization}
          timezone={boot.preferences.timezone}
          error={error}
          projects={projects}
          tasks={tasks}
          reminders={schedules.filter((s) => s.task_id === selected.id)}
          onReminder={(schedule) => {
            setSelected(null);
            setScheduleEditor({ schedule });
          }}
          onAddReminder={() => {
            setScheduleEditor({ schedule: null, task: selected });
            setSelected(null);
          }}
          linkedNotes={
            selected.id !== "new" ? (
              <>
                <LinearTask
                  task={selected}
                  mutate={mutate}
                  onChanged={() => {
                    setSelected(null);
                    void load();
                  }}
                />
                {!selected.is_template && (
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => {
                      setGoogleEvent({
                        id: "new",
                        entity_id: "new",
                        kind: "block",
                        title: "Work block",
                        date: selected.due_date ?? today,
                        at: null,
                        status: "active",
                        project_id: selected.project_id,
                        task_id: selected.id,
                        revision: 1,
                        projected: false,
                        notification_id: null,
                      });
                      setSelected(null);
                    }}
                  >
                    Reserve time for this task
                  </button>
                )}
                <TaskNotes
                  taskId={selected.id}
                  revision={noteRevision}
                  onOpen={(id) => void openNote(id)}
                  onNew={() => {
                    setNoteEditor(blankNote(selected));
                    setSelected(null);
                    setError("");
                  }}
                />
              </>
            ) : null
          }
          task={selected}
          busy={busy}
          onClose={() => setSelected(null)}
          onSave={async (args) => {
            const values = { ...(args as Record<string, unknown>) };
            if (selected.id === "new") {
              delete values.task_id;
              delete values.expected_revision;
              delete values.status;
            }
            const result = await mutate<Task>(
              selected.id === "new" ? "task.create" : "task.update",
              values,
              "Task saved",
            );
            if (result) setSelected(null);
          }}
          onArchive={async () => {
            const result = await mutate(
              "task.update",
              {
                task_id: selected.id,
                expected_revision: selected.revision,
                archived: true,
              },
              "Task archived",
            );
            if (result) setSelected(null);
          }}
        />
      )}
      {googleEvent && googleEvent.kind !== "google" && (
        <PlanningDialog
          key={googleEvent.entity_id}
          event={googleEvent}
          tasks={tasks}
          timezone={boot.preferences.timezone}
          mutate={mutate}
          onClose={() => setGoogleEvent(null)}
          onSaved={() => {
            setGoogleEvent(null);
            void load();
          }}
        />
      )}
      {googleEvent && googleEvent.kind === "google" && (
        <GoogleEventDialog
          key={
            googleEvent.entity_id +
            ":" +
            (googleEvent.occurrence_start ?? googleEvent.date)
          }
          event={googleEvent}
          timezone={boot.preferences.timezone}
          mutate={mutate}
          onSettings={() => {
            setGoogleEvent(null);
            setView("settings");
          }}
          onSaved={() => {
            setGoogleEvent(null);
            setToast("Google Calendar updated");
            void load();
          }}
          onClose={() => {
            setGoogleEvent(null);
            setError("");
          }}
        />
      )}
      {noteEditor && (
        <NoteEditor
          organization={organization}
          onOpenNote={(id) => { void openNote(id); }}
          key={
            noteEditor.id +
            ":" +
            noteEditor.revision +
            ":" +
            noteEditor.tasks.map((t) => t.id).join(",")
          }
          note={noteEditor}
          projects={projects}
          tasks={tasks}
          busy={busy}
          error={error}
          mutate={mutate}
          onSaved={setNoteEditor}
          onClose={() => {
            setNoteEditor(null);
            setError("");
          }}
          onTask={(id) => void openNoteTask(id)}
          onConversation={(id) => void openNoteConversation(id)}
        />
      )}
      {bulkEditor && (
        <BulkTaskDialog
          tasks={bulkEditor}
          projects={projects}
          busy={busy}
          error={error}
          onClose={() => {
            setBulkEditor(null);
            setError("");
          }}
          onSave={async (items) => {
            const result = await mutate(
              "task.batch",
              { items },
              "Tasks updated",
            );
            if (result) {
              setBulkEditor(null);
              setSelectedTaskIds([]);
              setSelectingTasks(false);
            }
          }}
        />
      )}
      {scheduleEditor && (
        <ScheduleDialog
          key={
            scheduleEditor.schedule
              ? scheduleEditor.schedule.id +
                ":" +
                scheduleEditor.schedule.revision
              : "new-reminder"
          }
          error={error}
          schedule={scheduleEditor.schedule}
          initialDate={scheduleEditor.date}
          linkedTask={scheduleEditor.task}
          zone={boot.preferences.timezone}
          tasks={tasks}
          projects={projects}
          notices={notices}
          busy={busy}
          onClose={() => setScheduleEditor(null)}
          mutate={mutate}
        />
      )}
      {reminder && (
        <ReminderDialog
          zone={boot.preferences.timezone}
          busy={busy}
          onClose={() => setReminder(false)}
          onSave={async (args) => {
            const result = await mutate(
              "schedule.create",
              args,
              "Reminder saved",
            );
            if (result) setReminder(false);
          }}
        />
      )}
    </div>
  );
}

function voiceLabel(state: string) {
  return (
    (
      {
        connecting: "Connecting…",
        closing: "Ending voice…",
        listening: "Listening",
        waiting: "Take your time. I’m listening.",
        evaluating: "Listening…",
        thinking: "Thinking…",
        acting: "Saving that…",
        working: "Taking care of that…",
        speaking: "Eri is speaking",
        unresolved: "Ready for your next words",
        disconnected: "Voice disconnected",
        closed: "Voice ended",
        ended: "Voice ended · Say Hey Eri when you need me",
        idle_timeout: "Voice ended after 15 quiet seconds",
      } as Record<string, string>
    )[state] ?? "Voice is on"
  );
}
