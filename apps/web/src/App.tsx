import { NoticeSnooze } from "./NoticeSnooze";
import { RoutingReviewPanel, NotificationPreferences } from "./PlannerPreferences";
import { StructureWorkspace } from "./Structure";
import { ProfileMenu } from "./ProfileMenu";
import { RecordNavigator, readRecordLink, type LinkedRecord } from "./record-links";
import { MemoryActions } from "./MemoryActions";
import { Tabs, humanLabel, PlannerGuide, useBodyLock } from "./ux";
import { BotSettings } from "./BotSettings";
import { PublicFooter } from "./PublicPages";
import { ActivityPanel, WorkCard, useWork, workActive, workAttention, type ActionChange, type WorkItem } from "./Activity";
import { VOICE_IDLE_SECONDS } from "./voice-idle";
import { SavedViews } from "./SavedViews";
import {
  readView,
  viewLink,
  type ViewState as SavedViewState,
  type SavedView,
} from "./saved-views";
import { AccountSwitcher, SharingSettings } from "./Accounts";
import { TaskDetails } from "./TaskDetails";
import { CalendarDetails } from "./CalendarDetails";
import { TaskTabs } from "./TaskTabs";
import { initialView, isTaskTab, taskTabs, type TaskTab } from "./task-presets";
import { validateSiteAction } from "./site-validation";
import { MemoryEditor } from "./MemoryEditor";
import { useEditorBridge } from "./editor-control";
import {
  emptyTaskFilters,
  matchesTaskFilters,
  sortTasks,
  type WorkLayout,
  type WorkSort,
  type WorkGroup,
  type TimelineSpan,
} from "./work-views";
import { LayoutSwitch } from "./WorkViews";
import { ProductivityPage, OrganizationFilters } from "./Productivity";
import {
  emptyOrganization,
  emptyFilter,
  matchesOrganization,
  scheduledBy,
  type Organization,
} from "./productivity";
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
import { api, ApiError, command, post, setCsrf, setDevice } from "./api";
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
  useDialogFocus,
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
  { id: "all", label: "Tasks", icon: ListTodo },
  { id: "organize", label: "Organization", icon: ListTodo },
  { id: "calendar", label: "Calendar", icon: CalendarDays },
  { id: "notes", label: "Notes", icon: FileText },
];
export default function App() {
  const editors = useEditorBridge();
  const initialRecord = useRef(readRecordLink(location.search));
  const recordOpened = useRef(false);
  const [linkWorkspace, setLinkWorkspace] = useState<string | null>(null);
  const initialSavedView = useRef(readView(location.search)).current;
  const [workLayout, setWorkLayout] = useState<WorkLayout>(
    initialSavedView?.layout ?? "list",
  );
  const [workSort, setWorkSort] = useState<WorkSort>(
    initialSavedView?.sort ?? "priority",
  );
  const [workGroup, setWorkGroup] = useState<WorkGroup>(
    initialSavedView?.group ?? "status",
  );
  const [taskFilters, setTaskFilters] = useState(
    initialSavedView
      ? {
          assignee: initialSavedView.assignee,
          work_type: initialSavedView.work_type,
          tag: initialSavedView.tag,
          due_from: initialSavedView.due_from,
          due_through: initialSavedView.due_through,
        }
      : emptyTaskFilters,
  );
  const [timelineDate, setTimelineDate] = useState(
    initialSavedView?.timeline_date ?? "",
  );
  const [timelineSpan, setTimelineSpan] = useState<TimelineSpan>(
    initialSavedView?.timeline_span ?? (window.innerWidth <= 600 ? 14 : 30),
  );
  const [organizationTab, setOrganizationTab] = useState<
    "goal" | "project" | "area" | "space" | "actor"
  >("project");
  const [organizationLayout, setOrganizationLayout] =
    useState<WorkLayout>("list");
  const [organizationVisible, setOrganizationVisible] = useState<string[]>([]);
  const [organizationEditor, setOrganizationEditor] = useState<{
    kind: "goal" | "project" | "area" | "space" | "actor";
    id?: string;
    sequence: number;
  } | null>(null);
  const [showArchived, setShowArchived] = useState(false);
  const [notesMode, setNotesMode] = useState<"keyword" | "semantic">("keyword");
  const [collectionContext,setCollectionContext]=useState<Record<string,string|number|null>>({});
  const [recordControl,setRecordControl]=useState<{nonce:string;type_id?:string;parent_id?:string;layout?:string;group?:string;record_id?:string;proposal_id?:string;field?:string;value?:string}>();
  useEffect(()=>{const open=(e:Event)=>{setView("organize");setOrganizationEditor(null);setRecordControl({nonce:crypto.randomUUID(),record_id:(e as CustomEvent).detail.id});};window.addEventListener("eri-open-custom-record",open);return()=>window.removeEventListener("eri-open-custom-record",open);},[]);
  const [settingsSection, setSettingsSection] = useState<
    "profile" | "voice" | "integrations" | "privacy" | "system" | "sharing"
  >(() => {
    const params = new URLSearchParams(location.search), section = params.get("section");
    if (section === "profile" || section === "voice" || section === "integrations" || section === "privacy" || section === "system" || section === "sharing") return section;
    return params.has("sharing") ? "sharing" : params.has("google") ? "integrations" : "profile";
  });
  const [density, setDensity] = useState<"compact" | "comfortable">(() =>
    localStorage.getItem("eri-density") === "comfortable"
      ? "comfortable"
      : "compact",
  );
  useEffect(() => {
    localStorage.setItem("eri-density", density);
  }, [density]);
  const [boot, setBoot] = useState<Bootstrap | null>(null),
    [loading, setLoading] = useState(true);
  const [view, setView] = useState<View>(
    () => initialSavedView?.tab ?? initialView(location.search),
  );
  const [planToday, setPlanToday] = useState(true);
  const [searchOpen, setSearchOpen] = useState(false);
  const searchAvailable = view !== "settings";
  const searchLabel = view === "notes" ? "Search notes" : view === "memory" ? "Search memories" : view === "organize" ? "Search organization" : view === "calendar" ? "Search calendar" : view === "notifications" ? "Search notifications" : "Search tasks";
  useEffect(() => { setPlanToday(true); setSearchOpen(false); }, [view]);
  const lastTaskTab = useRef<TaskTab>(isTaskTab(view) ? view : "today");
  useEffect(() => {
    if (isTaskTab(view)) lastTaskTab.current = view;
  }, [view]);
  const [calendarDetail, setCalendarDetail] = useState<CalendarEntry | null>(
    null,
  );
  function openTaskCard(task: Task) {
    setSelected(null);
    setCalendarDetail({
      id: "task-detail:" + task.id,
      entity_id: task.id,
      kind: "task",
      title: task.title,
      date: task.due_date ?? task.planned_date ?? "",
      at: null,
      status: task.status,
      project_id: task.project_id,
      task_id: task.id,
      revision: task.revision,
      projected: false,
      notification_id: null,
    });
  }
  function openCalendarEntry(entry: CalendarEntry) {
    if (entry.kind === "google") setGoogleEvent(entry);
    else setCalendarDetail(entry);
  }
  const [tasks, setTasks] = useState<Task[]>([]),
    [schedules, setSchedules] = useState<Schedule[]>([]),
    [notices, setNotices] = useState<Notice[]>([]);
  const [googleLogin, setGoogleLogin] = useState(false);
  const [pairingLogin, setPairingLogin] = useState(false);
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
  const [organization, setOrganization] =
    useState<Organization>(emptyOrganization);
  const [organizationFilter, setOrganizationFilter] = useState(
    initialSavedView
      ? {
          space: initialSavedView.space,
          area: initialSavedView.area,
          goal: initialSavedView.goal,
        }
      : emptyFilter,
  );
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
  const [workKind, setWorkKind] = useState<"all" | "task" | "reminder">(
    initialSavedView?.kind ?? "all",
  );
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
  const [query, setQuery] = useState(initialSavedView?.query ?? ""),
    [quick, setQuick] = useState(""),
    [pair, setPair] = useState("");
  const [error, setError] = useState(""),
    [toast, setToast] = useState(""),
    [busy, setBusy] = useState(false);
  const [sidebar, setSidebar] = useState(false),
    [companion, setCompanion] = useState(false);
  const [selected, setSelected] = useState<Task | null>(null),
    [reminder, setReminder] = useState(false);
  const [conversationHistoryOff, setConversationHistoryOff] = useState(false),
    [messages, setMessages] = useState<ChatMessage[]>([]),
    [chatText, setChatText] = useState(""),
    [thinking, setThinking] = useState(false);
  const [syncWarning, setSyncWarning] = useState("");
  const [activityOpen, setActivityOpen] = useState(() => new URLSearchParams(location.search).get("activity") === "1");
  const work = useWork(!!boot, (boot?.account_id ?? "") + ":" + (boot?.workspace?.id ?? "personal"));
  const activeWork = work.items.filter(workActive).length;
  const attentionWork = work.items.filter(item => workAttention(item) && !item.seen).length;
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
    | "active"
    | "backlog"
    | "open"
    | "in_progress"
    | "waiting"
    | "deferred"
    | "completed"
    | "cancelled"
  >(initialSavedView?.status ?? "active");
  const [projectFilter, setProjectFilter] = useState(
    initialSavedView?.project ?? "",
  );
  const [savedViewRevision, setSavedViewRevision] = useState(0);
  const savedViewState: SavedViewState = {
    tab: isTaskTab(view) ? view : "all",
    query,
    status: taskStatus,
    project: projectFilter,
    ...organizationFilter,
    ...taskFilters,
    kind: workKind,
    layout: workLayout,
    sort: workSort,
    group: workGroup,
    timeline_date: timelineDate,
    timeline_span: timelineSpan,
  };
  function applySavedView(value: SavedViewState) {
    setView(value.tab);
    setQuery(value.query);
    setTaskStatus(value.status);
    setProjectFilter(value.project);
    setOrganizationFilter({
      space: value.space,
      area: value.area,
      goal: value.goal,
    });
    setTaskFilters({
      assignee: value.assignee,
      work_type: value.work_type,
      tag: value.tag,
      due_from: value.due_from,
      due_through: value.due_through,
    });
    setWorkKind(value.kind);
    setWorkLayout(value.layout);
    setWorkSort(value.sort);
    setWorkGroup(value.group);
    setTimelineDate(value.timeline_date);
    setTimelineSpan(value.timeline_span);
  }
  useEffect(() => {
    const url = isTaskTab(view)
      ? viewLink(savedViewState)
      : new URL(location.href);
    if (!isTaskTab(view)) {
      url.searchParams.set("view", view);
      url.searchParams.delete("tab");
      url.searchParams.delete("state");
    }
    history.replaceState(null, "", url);
  }, [view, JSON.stringify(savedViewState)]);
  const [memoryStatus, setMemoryStatus] = useState<{enabled:boolean;pending:number;retrying:number;deferred:number;queued?:number;active?:number;failed?:number;retry_waiting?:number}>({
    enabled: true,
    pending: 0,
    retrying: 0,
    deferred: 0,
  });
  const [memoryRevision, setMemoryRevision] = useState(0);
  const [editingMemory, setEditingMemory] = useState<Memory | null>(null);
  const [mobile, setMobile] = useState(() => window.innerWidth <= 1000);
  useBodyLock(mobile && (companion || sidebar || searchOpen));
  const uiResults = useRef<
    { id: string; status: string; message?: string; data?: unknown }[]
  >([]);
  const syncUIRef = useRef<() => Promise<void>>(async () => {});
  const [highlight, setHighlight] = useState<string | null>(null);
  const [wakeEnabled, setWakeEnabled] = useState(false);
  const [wakeStatus, setWakeStatus] = useState("");
  const wake = useRef<WakeWord | null>(null);
  const wakeStart = useRef<(request: string) => void>(() => {});
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
      api<{ google: boolean; pairing: boolean }>("/auth/options")
        .then((d) => { setGoogleLogin(d.google); setPairingLogin(d.pairing); })
        .catch(() => {});
  }, [!!boot]);
  useEffect(() => {
    const result = new URLSearchParams(location.search).get("google");
    if (!result) return;
    const cleaned = new URL(location.href);
    cleaned.searchParams.delete("google");
    history.replaceState({}, "", cleaned);
    if (result === "connected") setToast("Google account connected");
    else
      setError(
        (
          {
            cancelled: "Google connection cancelled.",
            permission: "Calendar permission was not granted.",
            account: "Use your linked or invited Google account. If the invitation expired, ask its sender for a new one.",
          } as Record<string, string>
        )[result] ??
          "Google sign-in could not finish. Try again using your invited account.",
      );
  }, []);
  const workspaceSwitching = useRef(false);
  useEffect(() => {
    let recovering = false;
    const recover = (event: Event) => {
      if (recovering || workspaceSwitching.current) return;
      recovering = true;
      setTasks([]);
      setProjects([]);
      setMemories([]);
      setMessages([]);
      setCalendarDetail(null);
      setBoot(null);
      void (async () => {
        try {
          await voice.current?.stop();
          if ((event as CustomEvent).detail !== "WORKSPACE_CHANGED")
            await post("/accounts/switch", { workspace_id: null });
        } finally {
          sessionStorage.removeItem("jarvis-conversation");
          location.assign("/?view=tasks");
        }
      })();
    };
    window.addEventListener("eri-access-ended", recover);
    return () => window.removeEventListener("eri-access-ended", recover);
  }, []);
  const searchRef = useRef<HTMLInputElement>(null),
    messageEnd = useRef<HTMLDivElement>(null);
  const load = useCallback(async () => {
    const [taskData, scheduleData, noticeData, projectData, preferences] =
      await Promise.all([
        api<{ items: Task[]; next_cursor: string | null }>("/tasks?limit=200"),
        api<{ items: Schedule[]; next_cursor?: string | null }>("/schedules"),
        api<{ items: Notice[] }>("/notifications"),
        api<Organization>("/organization"),
        api<Bootstrap>("/bootstrap"),
      ]);
    let items = taskData.items,
      cursor = taskData.next_cursor;
    while (cursor) {
      const page = await api<{ items: Task[]; next_cursor: string | null }>(
        "/tasks?limit=200&before=" + cursor,
      );
      items = [...items, ...page.items];
      cursor = page.next_cursor;
    }
    setBoot(preferences);
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
      setDevice(info.device_id);
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
          setConversationHistoryOff(conversation.private);
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
        if (view === "settings") return;
        if (mobile) { setSearchOpen(true); requestAnimationFrame(() => searchRef.current?.focus()); }
        else searchRef.current?.focus();
      }
      if (e.key === "Escape") {
        const activeEditor = editors.current();
        if (activeEditor?.busy || activeEditor?.dirty) return;
        if (activeEditor) {void editors.act({operation:"close"}).catch(e => setError(e.message));return;}
        setSearchOpen(false);
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
  }, [view, mobile]);
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
      if(event.data?.type==="open-notification"&&typeof event.data.url==="string"&&event.data.url.startsWith("/?"))location.assign(event.data.url);
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
        if (result.data && typeof result.data === "object")
          Object.defineProperty(result.data, "__command_id", {value: result.command_id, enumerable: false});
        return result.data as T;
      } catch (e) {
        const err = e as ApiError;
        if (err.code !== "NETWORK") pendingCommands.current.delete(key);
        setError(
          tool === "task.batch" && err.code === "REVISION_CONFLICT"
            ? "One of these tasks changed. Close this editor and reopen the selection to review its latest values."
            : err.message,
        );
        // Keep the authored draft intact on revision conflict. Replacing the record
        // prop here used to silently reset task/reminder fields to the remote copy.
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
      {
        title: quick.trim(),
        project_id:
          view === "inbox"
            ? null
            : (projects.find((p) => p.name === projectFilter)?.id ?? null),
        space_id: view === "inbox" ? null : organizationFilter.space || null,
        area_id: view === "inbox" ? null : organizationFilter.area || null,
        planned_date: planToday && ["today", "week"].includes(view) ? today : null,
      },
      "Task added",
    );
    if (result) {
      setQuick("");
      // Keep a newly captured record discoverable even when the current filters exclude it.
      const matches = (!query || result.title.toLowerCase().includes(query.toLowerCase())) && matchesTaskFilters(result, taskFilters) && (taskStatus === "active" || taskStatus === "all" || taskStatus === result.status);
      if (!matches || (!result.planned_date && ["today", "week"].includes(view))) openTaskCard(result);
    }
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
    const data = await post<{ id: string; private: boolean }>("/conversations", {});
    setConversationHistoryOff(data.private);
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
      planned_date: date ?? (planToday && ["today", "week"].includes(view) ? dayInZone(boot?.preferences.timezone ?? "UTC") : null),
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
      openTaskCard(task);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function openLinkedRecord(record: LinkedRecord) {
    if(record.kind === "record") {await api("/structure/records/"+record.id);setView("organize");setOrganizationEditor(null);setNoteEditor(null);setCalendarDetail(null);setRecordControl({nonce:crypto.randomUUID(),record_id:record.id});return;}
    if (record.kind === "task") { const task = await api<Task>("/tasks/" + record.id); setNoteEditor(null); setSelected(null); openTaskCard(task); }
    else if (record.kind === "note") { const note = await api<NoteRecord>("/notes/" + record.id); setCalendarDetail(null); setNoteEditor(note); }
    else {
      const current = await api<Organization>("/organization");
      const rows = current[record.kind === "project" ? "projects" : record.kind === "goal" ? "goals" : record.kind === "area" ? "areas" : record.kind === "space" ? "spaces" : "actors"];
      if (!rows.some(row => row.id === record.id)) throw new Error("This record is missing or you no longer have access.");
      setOrganization(current); setView("organize"); setOrganizationTab(record.kind); setNoteEditor(null);setCalendarDetail(null);
      setOrganizationEditor({kind:record.kind,id:record.id,sequence:Date.now()});
    }
  }
  useEffect(() => {
    const record = initialRecord.current;
    if (!boot || loading || !record || recordOpened.current) return;
    recordOpened.current = true;
    if (record.workspace !== (boot.workspace?.id ?? "personal")) {setLinkWorkspace(record.workspace);return;}
    void openLinkedRecord(record).catch(() => setError("This record is missing or you no longer have access. Check the selected workspace or ask its owner."));
  }, [!!boot, loading]);
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
      setConversationHistoryOff(data.private);
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

  async function applyAction(
    action: UIAction,
  ): Promise<Record<string, unknown> | void> {
    validateSiteAction(action, view, organization, organizationTab);
    if (action.kind === "editor") return editors.act(action);
    if (action.kind === "device") {
      if (
        action.voice !== undefined &&
        (!boot?.voice_options.live?.voices.includes(action.voice) ||
          (!!voiceState && !voiceState.closed))
      )
        throw new Error(
          "Choose an available Live voice after the current voice session ends.",
        );
      if (action.wake_enabled && !recognitionType())
        throw new Error("Wake word is unavailable in this browser.");
      if (action.voice !== undefined) {
        setVoiceName(action.voice);
        localStorage.setItem("eri-voice-live", action.voice);
      }
      if (action.wake_enabled !== undefined) {
        setWakeEnabled(action.wake_enabled);
        setWakeStatus("");
      }
      if (action.density !== undefined) setDensity(action.density);
      return { outcome: "device_preferences_updated", saved: true };
    }
    const kind = action.kind ?? "show";
    if (kind === "activity") {
      if (action.mode === "open" && editors.current()?.mode === "edit")
        throw new Error("Finish or close the current draft before opening Activity.");
      if (editors.current()?.auto_save) await editors.current()?.beforeLeave?.();
      if (action.mode === "open") setCalendarDetail(null);
      setActivityOpen(action.mode === "open");
      return;
    }
    if (kind === "chat") {
      setCompanion(action.mode === "auto" ? !mobile : action.mode === "open");
      return;
    }
    const currentEditor = editors.current();
    if (currentEditor?.auto_save) await currentEditor.beforeLeave?.();
    if ((currentEditor && currentEditor.mode !== "detail") || (!currentEditor && (selected || reminder || scheduleEditor || organizationEditing || editingMemory || noteEditor || googleEvent || bulkEditor)))
      throw new Error("An editor is open. Save or close it before changing pages.");
    if (currentEditor?.mode === "detail") await editors.act({operation:"close"});
    setCalendarDetail(null);
    if(kind==="records"){
      setView("organize");setOrganizationEditor(null);
      setRecordControl({nonce:action.id,type_id:action.type_id,parent_id:action.parent_id,layout:action.layout,group:action.record_group,record_id:action.record_id,proposal_id:action.proposal_id,field:action.field,value:action.value});
      return {outcome:"collection_opened",record_id:action.record_id??null};
    }
    if (kind === "saved_view") {
      const collection = await api<{ items: SavedView[] }>("/task-views");
      if (action.view_operation === "list") return { items: collection.items };
      const saved = collection.items.find((v) => v.id === action.saved_view_id);
      if (action.view_operation === "load") {
        if (!saved) throw new Error("That saved view is unavailable.");
        applySavedView(saved.state);
        return { outcome: "view_loaded", name: saved.name };
      }
      if (action.view_operation === "delete") {
        if (!saved) throw new Error("That saved view is unavailable.");
        const result = await post("/task-views/remove", {
          id: saved.id,
          expected_revision: saved.revision,
        });
        setSavedViewRevision((n) => n + 1);
        return { outcome: "view_deleted", result };
      }
      if (!isTaskTab(view))
        throw new Error("Open Tasks before saving a task view.");
      if (!action.view_name?.trim())
        throw new Error("Give the saved view a name.");
      const result = await post("/task-views", {
        id: action.id.slice(-36),
        name: action.view_name,
        expected_revision: 0,
        state: savedViewState,
      });
      setSavedViewRevision((n) => n + 1);
      return { outcome: "view_saved", result };
    } else if (kind === "workspace") {
      const target = action.view ?? view;
      if (
        action.layout &&
        !["all", "inbox", "today", "week", "organize"].includes(target)
      )
        throw new Error(
          "List, board and timeline layouts are available in Tasks and Projects.",
        );
      if (action.organization_tab && target !== "organize")
        throw new Error("Organization tabs belong to Projects.");
      if (action.notes_mode && target !== "notes")
        throw new Error("Choose Notes for note search mode.");
      if (action.settings_section && target !== "settings")
        throw new Error("Choose Settings for that section.");
      if (action.layout)
        (target === "organize" ? setOrganizationLayout : setWorkLayout)(
          action.layout,
        );
      if (action.sort) setWorkSort(action.sort);
      if (action.group_by) setWorkGroup(action.group_by);
      if (action.timeline_date) setTimelineDate(action.timeline_date);
      if (action.timeline_span) setTimelineSpan(action.timeline_span);
      if (action.organization_tab) setOrganizationTab(action.organization_tab);
      if (action.show_archived !== undefined)
        setShowArchived(action.show_archived);
      if (action.notes_mode) setNotesMode(action.notes_mode);
      if (action.settings_section) setSettingsSection(action.settings_section);
      setView(target);
    } else if (kind === "select") {
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
      setTaskFilters(emptyTaskFilters);
      setWorkKind("task");
      setWorkLayout("list");
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
      if (
        action.entity_id &&
        !tasks.some((t) => t.id === action.entity_id) &&
        !schedules.some((s) => s.id === action.entity_id)
      ) {
        try {
          await api("/planning/" + encodeURIComponent(action.entity_id));
        } catch {
          await api("/calendar/events/" + encodeURIComponent(action.entity_id));
        }
      }
      setHighlight(action.entity_id ?? null);
      setCalendarDay(day);
      if (action.calendar_view || action.entity_id)
        setCalendarMode(action.calendar_view ?? "day");
      setView("calendar");
      setQuery("");
      setTaskStatus("all");
      setProjectFilter("");
      setOrganizationFilter(emptyFilter);
      setTaskFilters(emptyTaskFilters);
      setWorkKind("all");
      if (action.open_details && action.entity_id) {
        const data = await api<{ items: CalendarEntry[] }>(
          "/calendar?start=" +
            day +
            "&end=" +
            new Date(Date.parse(day + "T12:00:00Z") + 86400000)
              .toISOString()
              .slice(0, 10) +
            "&timezone=" +
            encodeURIComponent(boot?.preferences.timezone ?? "America/Chicago"),
        );
        const entry = data.items.find((e) => e.entity_id === action.entity_id);
        if (!entry)
          throw new Error(
            "That item is not in this calendar day. Check its date before opening details.",
          );
        openCalendarEntry(entry);
      }
    } else if (kind === "search") {
      setQuery(action.query ?? "");
      setTaskStatus("all");
      setProjectFilter("");
      setOrganizationFilter(emptyFilter);
      setTaskFilters(emptyTaskFilters);
      setWorkKind("all");
      const target = action.view ?? "all";
      if (target === "settings") {
        const phrase = (action.query ?? "").toLowerCase();
        setSettingsSection(
          /shar|invit|member|account|permission/.test(phrase)
            ? "sharing"
            : /voice|wake|sound/.test(phrase)
              ? "voice"
              : /google|calendar|linear|integration/.test(phrase)
                ? "integrations"
                : /memory|history|privacy|learning/.test(phrase)
                  ? "privacy"
                  : /model|agent|backup|export|budget|cost|system/.test(phrase)
                    ? "system"
                    : "profile",
        );
      }
      setView(target);
    } else if (kind === "filter") {
      const from = action.due_from ?? taskFilters.due_from,
        through = action.due_through ?? taskFilters.due_through;
      if (from && through && from > through)
        throw new Error("Due-from must be on or before due-through.");
      setView(
        action.view ??
          ([
            "all",
            "calendar",
            "reminders",
            "today",
            "inbox",
            "week",
            "notes",
            "organize",
          ].includes(view)
            ? view
            : "all"),
      );
      if (action.status !== undefined) setTaskStatus(action.status);
      if (
        action.space_id !== undefined ||
        action.area_id !== undefined ||
        action.goal_id !== undefined
      )
        setOrganizationFilter((current) => ({
          space: action.space_id ?? current.space,
          area: action.area_id ?? current.area,
          goal: action.goal_id ?? current.goal,
        }));
      if (action.project_id !== undefined) {
        const project = projects.find((p) => p.id === action.project_id);
        if (action.project_id && !project)
          throw new Error(
            "That project is unavailable. Read current organization records.",
          );
        setProjectFilter(project?.name ?? "");
      } else if (action.project !== undefined) {
        if (action.project && !projects.some((p) => p.name === action.project))
          throw new Error("That project is unavailable.");
        setProjectFilter(action.project);
      }
      setTaskFilters((current) => ({
        assignee: action.assignee ?? current.assignee,
        work_type: action.work_type ?? current.work_type,
        tag: action.tag ?? current.tag,
        due_from: action.due_from ?? current.due_from,
        due_through: action.due_through ?? current.due_through,
      }));
      if (action.work_kind !== undefined) setWorkKind(action.work_kind);
    } else if (kind === "form") {
      if (
        ["goal", "project", "space", "area", "actor"].includes(
          action.form ?? "",
        )
      ) {
        setView("organize");
        setOrganizationEditor({
          kind: action.form as "goal" | "project" | "space" | "area" | "actor",
          id: action.entity_id ?? undefined,
          sequence: Date.now(),
        });
      } else if (action.form === "reminder") {
        setScheduleEditor({
          schedule: action.entity_id
            ? await api<Schedule>(
                "/schedules/" + encodeURIComponent(action.entity_id),
              )
            : null,
        });
      } else if (action.form === "note") {
        setView("notes");
        setNoteEditor(
          action.entity_id
            ? await api<NoteRecord>(
                "/notes/" + encodeURIComponent(action.entity_id),
              )
            : blankNote(),
        );
      } else if (action.form === "event" || action.form === "google_event") {
        if (action.entity_id) {
          const detail = await api<Record<string, unknown>>(
            (action.form === "event" ? "/planning/" : "/calendar/events/") +
              encodeURIComponent(action.entity_id),
          );
          const fields = (detail.fields ?? detail) as Record<string, unknown>;
          setGoogleEvent({
            id: action.entity_id,
            entity_id: action.entity_id,
            kind:
              action.form === "google_event"
                ? "google"
                : detail.kind === "block"
                  ? "block"
                  : "event",
            title: String(fields.title ?? fields.summary ?? ""),
            date: calendarDay || today,
            at: null,
            status: "active",
            project_id: null,
            task_id: typeof detail.task_id === "string" ? detail.task_id : null,
            revision: Number(detail.revision ?? 1),
            projected: false,
            notification_id: null,
            recurring: !!detail.recurring,
            occurrence_start:
              typeof detail.occurrence_start === "string"
                ? detail.occurrence_start
                : undefined,
          });
        } else
          setGoogleEvent({
            id: "new",
            entity_id: "new",
            kind: action.form === "google_event" ? "google" : "event",
            title: "",
            date: calendarDay || today,
            at: null,
            status: "active",
            project_id: null,
            task_id: null,
            revision: 1,
            projected: false,
            notification_id: null,
          });
      } else if (action.form === "bulk") {
        const chosen = tasks.filter((t) => selectedTaskIds.includes(t.id));
        if (!chosen.length)
          throw new Error("Select tasks before opening their bulk editor.");
        setBulkEditor(chosen);
      } else if (action.form === "memory") {
        const memory = memories.find((m) => m.id === action.entity_id);
        if (!memory)
          throw new Error(
            "Open Memory and choose a visible memory before correcting it.",
          );
        setEditingMemory(memory);
      } else {
        setView("all");
        if (action.entity_id)
          openTaskCard(
            await api<Task>("/tasks/" + encodeURIComponent(action.entity_id)),
          );
        else createTask();
      }
    } else {
      if (
        action.entity_id &&
        !["all", "organize", "notes", "reminders", "memory"].includes(
          action.view ?? "",
        )
      )
        throw new Error(
          "Choose Tasks, Projects & goals, Notes, Task alerts or Memory to open a record.",
        );
      setQuery("");
      setTaskStatus("all");
      setProjectFilter("");
      setOrganizationFilter(emptyFilter);
      setTaskFilters(emptyTaskFilters);
      setWorkKind("all");
      setView(action.view ?? "today");
      if (action.entity_id && action.view === "all") {
        openTaskCard(
          await api<Task>("/tasks/" + encodeURIComponent(action.entity_id)),
        );
      } else if (action.entity_id && action.view === "organize") {
        const current = await api<Organization>("/organization");
        if (
          ![
            ...current.spaces,
            ...current.areas,
            ...current.goals,
            ...current.projects,
            ...current.actors,
          ].some((row) => row.id === action.entity_id)
        )
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
      } else if (action.entity_id && action.view === "memory") {
        const data = await api<{ items: Memory[] }>("/memory?q=");
        const found = data.items.find((m) => m.id === action.entity_id);
        if (!found)
          throw new Error("That memory is unavailable. Search Memory first.");
        setMemories(data.items);
        setEditingMemory(found);
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
    setError("");
    setSidebar(false);
    if (mobile) setCompanion(false);
  }
  async function showActions(actions: UIAction[] = []) {
    for (const action of actions) {
      if (displayedActions.current.has(action.id)) continue;
      displayedActions.current.add(action.id);
      try {
        const result = (await runSiteControl(action)) as
          | { data?: unknown }
          | undefined;
        uiResults.current.push({
          id: action.id,
          status: "displayed",
          data: result?.data,
        });
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
    setConversationHistoryOff(false);
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
        await post<WorkItem>("/work", body);
        retryRef.current = null;
        await work.refresh();
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
  async function openWorkRecord(action: ActionChange) {
    if (!action.entity_id) return;
    if(action.kind==="record"){setView("organize");setOrganizationEditor(null);setRecordControl({nonce:crypto.randomUUID(),record_id:action.entity_id});return;}
    const form = action.kind === "schedule" ? "reminder" : action.kind === "planning" ? "event" : action.kind;
    await applyAction({ id: crypto.randomUUID(), kind: "form", form, entity_id: action.entity_id } as UIAction);
    setActivityOpen(false);
    if (mobile) setCompanion(false);
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
  wakeStart.current = (request) => {
    if (voice.current || thinking) return;
    void (async () => {
      if (request) {
        const id = crypto.randomUUID();
        try {
          const conversation_id = await ensureConversation();
          await post("/work", {turn_id:id, conversation_id, message:request});
          setMessages(items => [...items, {id, role:"user", content:request}]);
          await work.refresh();
        } catch (e) { setError((e as Error).message); return; }
      }
      await startVoice();
    })();
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
        (request) => wakeStart.current(request),
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
  async function enablePush() {
    if (!("PushManager" in window)) {
      setError(
        "This browser does not support notifications. Notifications still work.",
      );
      return;
    }
    try {
      const permission = await Notification.requestPermission();
      if (permission !== "granted") {
        setToast("Notifications are off. Reminders stay in Notifications.");
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
  const historyOff = conversationHistoryOff || !boot?.preferences.history_enabled;
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
      if (
        taskStatus === "active"
          ? ["completed", "cancelled"].includes(t.status)
          : taskStatus !== "all" && t.status !== taskStatus
      )
        return false;
      if (!matchesTaskFilters(t, taskFilters)) return false;
      if (projectFilter && t.project !== projectFilter) return false;
      if (!matchesOrganization(t, organizationFilter, organization))
        return false;
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
    layout: view === "organize" ? organizationLayout : workLayout,
    sort: workSort,
    group_by: workGroup,
    timeline_date: timelineDate || today,
    timeline_span: timelineSpan,
    organization_tab: organizationTab,
    settings_section: settingsSection,
    notes_mode: notesMode,
    show_archived: showArchived,
    ...taskFilters,
    editor: editors.summary,
    device_preferences: {
      voice: voiceName,
      voices: boot?.voice_options.live?.voices ?? [],
      wake_enabled: wakeEnabled,
      wake_supported: !!recognitionType(),
      density,
    },

    view,
    activity_open: activityOpen,
    collection: collectionContext,
    chat_open: companion,
    mobile,
    voice_active: !!voiceState && !voiceState.closed,
    query: query.slice(0, 300),
    selected_task_id:
      calendarDetail?.kind === "task"
        ? calendarDetail.entity_id
        : selected?.id === "new"
          ? null
          : (selected?.id ?? null),
    selected_schedule_id: scheduleEditor?.schedule?.id ?? null,
    selected_task_ids: selectedTaskIds
      .filter((id) => tasks.some((t) => t.id === id))
      .slice(0, 100),
    selected_calendar_event_id:
      calendarDetail?.entity_id ??
      (googleEvent?.kind === "google" ? googleEvent.entity_id : null),
    selected_note_id:
      noteEditor?.id === "new" ? null : (noteEditor?.id ?? null),
    calendar_date: calendarDay || undefined,
    calendar_view: calendarMode,
    space_id: organizationFilter.space,
    area_id: organizationFilter.area,
    goal_id: organizationFilter.goal,
    work_kind: view === "reminders" ? "reminder" : workKind,
    visible_ids:
      view === "organize"
        ? organizationVisible
        : view === "notes"
          ? noteVisible
          : ["all", "calendar", "reminders", "today", "inbox", "week"].includes(
                view,
              )
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
    organize: "Organization",
    today: "Tasks",
    inbox: "Tasks",
    week: "Tasks",
    all: "Tasks",
    reminders: "Task alerts",
    calendar: "Calendar",
    notes: "Notes",
    memory: "Memory",
    notifications: "Notifications",
    settings: "Settings",
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
          <p className="eyebrow">WELCOME TO ERIDANI</p>
          <h1>A little more organized.</h1>
          <p>
            Your day, with Eri.
            <br />
            {pairingLogin ? "Sign in with Google, or pair an owner device." : "Sign in with your linked or invited Google account."}
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
          {pairingLogin && <>
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
          </>}
          {error && (
            <p role="alert" className="error-text">
              {error}
            </p>
          )}
          {pairingLogin && <small>
            The pairing code opens the owner account. Invited people use Google sign-in.
          </small>}
          {!pairingLogin && <p className="footnote">Invitation-only access. <a href="/support#access">Need an invitation?</a></p>}
          {busy && <p role="status">Opening Google sign-in…</p>}
        </form>
        <PublicFooter/>
      </div>
    );
  return (
    <RecordNavigator workspace={boot.workspace?.id ?? "personal"} view={view} onOpen={openLinkedRecord}><div
      className={
        "app-shell density-" +
        density +
        " " +
        (voiceState && !voiceState.closed ? "voice-active" : "")
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
          <span className="brand-caption">
            {boot.workspace?.id ? "SHARED" : "PERSONAL"}
          </span>
        </a>
        <AccountSwitcher
          onSharing={() => {
            setView("settings");
            setSettingsSection("sharing");
            setSidebar(false);
          }}
          onSwitch={async (id) => {
            const active = editors.current();
            if (active?.auto_save) await active.beforeLeave?.();
            else if (active?.dirty)
              throw new Error(
                "Finish or discard the current form before switching workspaces.",
              );
            await voice.current?.stop();
            workspaceSwitching.current = true;
            try {
              await post("/accounts/switch", { workspace_id: id });
            } catch (e) {
              workspaceSwitching.current = false;
              throw e;
            }
            sessionStorage.removeItem("jarvis-conversation");
            location.assign("/?view=tasks");
          }}
        />
        <div className="nav-label">
          {boot.workspace?.id
            ? boot.workspace.name + " · " + boot.workspace.role
            : "YOUR SPACE"}
        </div>
        <nav>
          {nav.map((item) => (
              <button
                key={item.id}
                aria-label={item.label}
                className={
                  (item.id === "all" ? isTaskTab(view) : view === item.id)
                    ? "nav-item active"
                    : "nav-item"
                }
                onClick={() => {
                  setView(item.id === "all" ? lastTaskTab.current : item.id);
                  setQuery("");
                  setSidebar(false);
                }}
              >
                <item.icon size={18} />
                <span>{item.label}</span>
              </button>
            ))}
        </nav>
        <div className="sidebar-bottom">
          <ProfileMenu name={boot.name} view={view} personal={!boot.workspace?.id} navigationOpen={sidebar}
            onNavigate={target => {
              setView(target);
              setQuery("");
              setSidebar(false);
              if (matchMedia("(max-width: 700px)").matches)
                document.querySelector<HTMLButtonElement>(".mobile-menu")?.focus({preventScroll: true});
            }}
            onLogout={async () => {
              await voice.current?.stop();
              await post("/auth/logout");
              setBoot(null);
              setTasks([]);
              setMessages([]);
              conversationRef.current = null;
              sessionStorage.removeItem("jarvis-conversation");
            }}/>
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
            <span>{boot.workspace?.name ?? "Personal workspace"}</span>
            <ChevronRight size={14} />
            <strong>
              {titles[view]}
            </strong>
          </div>
          <div className="top-actions">
            <button className={"activity-toggle " + (attentionWork ? "needs-attention" : "")}
              aria-label={"Eri activity, " + activeWork + " pending, " + attentionWork + " need attention"}
              title="Eri activity" onClick={() => setActivityOpen(true)}>
              <Clock3 size={18}/><span>Activity</span>{(activeWork + attentionWork > 0) && <b>{activeWork + attentionWork}</b>}
            </button>
            {searchAvailable && <>
              {mobile && <button className="icon-button mobile-search" aria-label={searchLabel} onClick={() => {setSearchOpen(true); requestAnimationFrame(() => searchRef.current?.focus());}}><Search size={18}/></button>}
            <div className={"search " + (searchOpen ? "search-expanded" : "")}>
              <Search size={16} />
              <input
                ref={searchRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={searchLabel + "…"}
                aria-label={searchLabel}
              />
              <kbd>{/Mac|iPhone|iPad/.test(navigator.platform) ? "⌘ K" : "Ctrl K"}</kbd>
              {mobile && <button className="icon-button" aria-label="Close search" onClick={() => setSearchOpen(false)}><X size={18}/></button>}
            </div>
            </>}
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
        {linkWorkspace && <div className="error-banner" role="status"><span>This record link belongs to another workspace. Switch to open it; the link does not grant access.</span><button onClick={async () => {try {await post("/accounts/switch", {workspace_id:linkWorkspace === "personal" ? null : linkWorkspace});location.reload();} catch {setError("That workspace is unavailable to your account. Ask its owner for access.");}}}>Open linked workspace</button><button onClick={() => setLinkWorkspace(null)}>Dismiss</button></div>}
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
        <div className={"workspace " + (mobile && companion ? "chat-sheet-open" : "")}>
          <main className="content" inert={mobile && companion}>
            <div className="page-heading">
              <h1>{titles[view]}</h1>
            </div>
            {isTaskTab(view) && (
              <>
                <TaskTabs value={view} onChange={setView} />

              </>
            )}
            {[
              "today",
              "inbox",
              "week",
              "all",
              "calendar",
              "reminders",
            ].includes(view) && (
              <div className="view-control-row" hidden={isTaskTab(view)}>
              {isTaskTab(view) && <SavedViews key={savedViewRevision} state={savedViewState} onApply={applySavedView}/>}
              <details className="filter-panel">
                <summary>
                  Filters & sort{" "}
                  <span>
                    {
                      [
                        taskStatus !== "active",
                        !!projectFilter,
                        ...Object.values(organizationFilter).map(Boolean),
                        ...Object.values(taskFilters).map(Boolean),
                        workKind !== "all",
                      ].filter(Boolean).length
                    }
                  </span>
                </summary>
                <div className="task-filters">
                  <OrganizationFilters
                    organization={organization}
                    value={organizationFilter}
                    onChange={setOrganizationFilter}
                  />
                  <label>
                    Status
                    <select
                      aria-label="Task status filter"
                      value={taskStatus}
                      onChange={(e) =>
                        setTaskStatus(e.target.value as typeof taskStatus)
                      }
                    >
                      <option value="active">Active</option>
                      <option value="all">All</option>
                      <option value="backlog">Backlog</option>
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
                  <label>
                    Assignee
                    <select
                      aria-label="Assignee filter"
                      value={taskFilters.assignee}
                      onChange={(e) =>
                        setTaskFilters({
                          ...taskFilters,
                          assignee: e.target.value,
                        })
                      }
                    >
                      <option value="">Everyone</option>
                      {organization.actors.map((a) => (
                        <option key={a.id} value={a.id}>
                          {a.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Work type
                    <input
                      aria-label="Work type filter"
                      value={taskFilters.work_type}
                      onChange={(e) =>
                        setTaskFilters({
                          ...taskFilters,
                          work_type: e.target.value,
                        })
                      }
                    />
                  </label>
                  <label>
                    Tag
                    <input
                      aria-label="Tag filter"
                      value={taskFilters.tag}
                      onChange={(e) =>
                        setTaskFilters({ ...taskFilters, tag: e.target.value })
                      }
                    />
                  </label>
                  <label>
                    Due from
                    <input
                      aria-label="Due from filter"
                      type="date"
                      value={taskFilters.due_from}
                      onChange={(e) =>
                        setTaskFilters({
                          ...taskFilters,
                          due_from: e.target.value,
                        })
                      }
                    />
                  </label>
                  <label>
                    Due through
                    <input
                      aria-label="Due through filter"
                      type="date"
                      value={taskFilters.due_through}
                      onChange={(e) =>
                        setTaskFilters({
                          ...taskFilters,
                          due_through: e.target.value,
                        })
                      }
                    />
                  </label>
                  <label>
                    Sort
                    <select
                      aria-label="Sort work"
                      value={workSort}
                      onChange={(e) => setWorkSort(e.target.value as WorkSort)}
                    >
                      {["priority", "due", "planned", "title", "updated"].map(
                        (v) => (
                          <option key={v} value={v}>
                            {v}
                          </option>
                        ),
                      )}
                    </select>
                  </label>
                  <label>
                    Group board
                    <select
                      aria-label="Group work"
                      value={workGroup}
                      onChange={(e) =>
                        setWorkGroup(e.target.value as WorkGroup)
                      }
                    >
                      {["status", "project", "assignee"].map((v) => (
                        <option key={v} value={v}>
                          {v}
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
                  <button
                    className="text-button"
                    onClick={() => {
                      setTaskStatus("active");
                      setProjectFilter("");
                      setOrganizationFilter(emptyFilter);
                      setTaskFilters(emptyTaskFilters);
                      setWorkKind("all");
                      setQuery("");
                    }}
                  >
                    Reset to this tab
                  </button>
                </div>
              </details>
              </div>
            )}
            {[
              "today",
              "inbox",
              "week",
              "all",
              "calendar",
              "reminders",
            ].includes(view) && (
              <div className="active-filters" aria-label="Active filters" hidden={isTaskTab(view)}>
                {[
                  {label: query ? 'Search: ' + query : '', clear: () => setQuery('')},
                  {label: projectFilter, clear: () => setProjectFilter('')},
                  ...(['space', 'area', 'goal'] as const).map(key => ({label: organization[key === 'space' ? 'spaces' : key === 'area' ? 'areas' : 'goals'].find(r => r.id === organizationFilter[key])?.name ?? '', clear: () => setOrganizationFilter({...organizationFilter, [key]: '', ...(key === 'space' ? {area: ''} : {})})})),
                  {label: taskStatus !== 'active' ? 'Status: ' + humanLabel(taskStatus) : '', clear: () => setTaskStatus('active')},
                  ...Object.entries(taskFilters).map(([key, value]) => ({label: value ? humanLabel(key) + ': ' + (key === 'assignee' ? organization.actors.find(a => a.id === value)?.name ?? value : value) : '', clear: () => setTaskFilters({...taskFilters, [key]: ''})})),
                  {label: workKind !== 'all' ? humanLabel(workKind) : '', clear: () => setWorkKind('all')},
                ].filter(chip => chip.label).map((chip, index) => <button key={index} className="filter-chip" aria-label={'Remove filter: ' + chip.label} onClick={chip.clear}>{chip.label}<X size={13}/></button>)}
              </div>
            )}
            {[
              "calendar",
              "reminders",
            ].includes(view) && (
              <>
                {["today", "inbox", "week", "all"].includes(view) && (
                  <div className="capture-bar"><form className="quick-add compact-capture" onSubmit={add}>
                    <Plus size={17} />
                    <input
                      value={quick}
                      onChange={(e) => setQuick(e.target.value)}
                      aria-label="New task"
                      placeholder="Add a task…"
                      maxLength={500}
                    />
                    <button disabled={!quick.trim() || busy} type="submit">
                      Add<span>↵</span>
                    </button>
                  </form>{["today", "week"].includes(view) && <button className="plan-chip" aria-pressed={planToday} onClick={() => setPlanToday(!planToday)} title="Planned day is when you intend to work on this task">{planToday ? <>Planned today <X size={12}/></> : "Plan today"}</button>}</div>
                )}
                <Workspace
                  layout={workLayout}
                  onLayout={setWorkLayout}
                  sort={workSort}
                  group={workGroup}
                  taskFilters={taskFilters}
                  timelineDate={timelineDate || today}
                  timelineSpan={timelineSpan}
                  onTimelineDate={setTimelineDate}
                  onTimelineSpan={setTimelineSpan}
                  onGoogleEvent={openCalendarEntry}
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
                  preset={
                    ["today", "inbox", "week"].includes(view)
                      ? (view as "today" | "inbox" | "week")
                      : "all"
                  }
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
                  onTask={openTaskCard}
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
            {isTaskTab(view)&&<StructureWorkspace selecting={selectingTasks} selectedIds={selectedTaskIds} onSelecting={value=>{setSelectingTasks(value);if(!value)setSelectedTaskIds([]);}} onSelection={setSelectedTaskIds} onBulk={()=>{setError("");setBulkEditor(tasks.filter(t=>selectedTaskIds.includes(t.id)));}} key={boot.workspace?.id??"personal"} onContext={setCollectionContext} statusFilter={taskStatus} homeFilter={organizationFilter.area||organizationFilter.space||projects.find(p=>p.name===projectFilter)?.id||""} layoutFilter={workLayout} groupFilter={workGroup} onQuery={setQuery} onTab={setView} capability="work" tab={view} today={today} query={query} canEdit={boot.workspace?.role!=="viewer"} canDesign={!boot.workspace?.id||boot.workspace?.role==="owner"} refresh={noteRevision} onChanged={()=>void load()} onVisible={setWorkVisible} onTask={id=>{const task=tasks.find(t=>t.id===id);if(task)openTaskCard(task);}}/>}
            {view === "notifications" && (
              <>
                <div className="section-head">
                  <h2>
                    Notifications<span>{pendingNotices.length}</span>
                  </h2>
                  <button
                    className="text-button"
                    onClick={() => void enablePush()}
                  >
                    <Bell size={15} />
                    Enable notifications
                  </button>
                </div>
                {pendingNotices
                  .filter((n) =>
                    JSON.stringify(n)
                      .toLowerCase()
                      .includes(query.toLowerCase()),
                  )
                  .map((n) => (
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
                        <button onClick={()=>{if(n.task_id){const task=tasks.find(t=>t.id===n.task_id);if(task)openTaskCard(task);}else if(n.target?.view==="activity")setActivityOpen(true);else setView("today");void mutate("notification.read",{notification_id:n.id},"");}}>Open</button>
                        {n.task_id&&["reminder","deadline"].includes(n.category??"reminder")&&<button
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
                        </button>}
                        <NoticeSnooze onSnooze={args=>mutate("notification.snooze",{notification_id:n.id,...args},"Notification snoozed")}/>
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
            {view === "organize" && organizationEditor && (
              <ProductivityPage
                tasks={tasks}
                organization={organization}
                busy={busy}
                mutate={mutate}
                query={query}
                highlight={highlight}
                onEditing={setOrganizationEditing}
                tab={organizationTab}
                onTab={setOrganizationTab}
                layout={organizationLayout}
                onLayout={setOrganizationLayout}
                archived={showArchived}
                onArchived={setShowArchived}
                space={organizationFilter.space}
                onSpace={(space) =>
                  setOrganizationFilter({ ...emptyFilter, space })
                }
                editorRequest={organizationEditor}
                onEditorRequestHandled={() => setOrganizationEditor(null)}
                onVisible={setOrganizationVisible}
                today={today}
                timelineDate={timelineDate || today}
                timelineSpan={timelineSpan}
                onTimelineDate={setTimelineDate}
                onTimelineSpan={setTimelineSpan}
                onProject={(p) => {
                  setProjectFilter(p.name);
                  setOrganizationFilter(emptyFilter);
                  setQuery("");
                  setView("all");
                }}
                onNote={(id) => {
                  void openNote(id);
                }}
              />
            )}
            {view === "organize" && !organizationEditor && <StructureWorkspace onVisible={setOrganizationVisible} onContext={setCollectionContext} onQuery={setQuery} control={recordControl} query={query} canEdit={boot.workspace?.role!=="viewer"} canDesign={!boot.workspace?.id||boot.workspace?.role==="owner"} refresh={noteRevision} onChanged={()=>void load()}/>}

            {view === "notes" && (
              <NotesPage
                archived={showArchived}
                onArchived={setShowArchived}
                mode={notesMode}
                onMode={setNotesMode}
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
                <div className="learning-status" role="status"><strong>{memoryStatus.enabled ? "Automatic learning is on" : "Automatic learning is off"}</strong><p>
                  {!boot.capabilities.worker ? "Background processing is paused. " : ""}
                  {[memoryStatus.queued ? `${memoryStatus.queued} waiting` : "", memoryStatus.active ? `${memoryStatus.active} processing` : "", memoryStatus.retry_waiting ? `${memoryStatus.retry_waiting} waiting to retry` : "", memoryStatus.failed ? `${memoryStatus.failed} failed` : "", memoryStatus.deferred ? `${memoryStatus.deferred} paused for budget` : "", memoryReviews.length ? `${memoryReviews.length} questions to review` : ""].filter(Boolean).join(" · ") || (memoryStatus.pending ? "Learning is queued" : "No learning waiting")}
                </p><small>These are learning steps, not a count of messages. Correct a fact to change what Eri remembers; View source shows where it came from.</small></div>

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
                  <span className="subtle">Personal facts & preferences</span>
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
                          ? "You told Eri"
                          : "Learned from a saved source"} · {new Date(m.created_at).toLocaleDateString()}
                      </span>
                      <MemoryActions memory={m} onCorrect={() => setEditingMemory(m)} onForget={delete_source => mutate("memory.forget", {memory_id:m.id, delete_source}, delete_source ? "Memory and its stored source deleted" : "Memory forgotten")}/>
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
                <Tabs id="settings-tab" label="Settings sections" className="settings-tabs" panel="settings-panel"
                  value={settingsSection} onChange={setSettingsSection} items={(["profile", "voice", "integrations", "privacy", "system", "sharing"] as const).map(id => ({id, label: humanLabel(id)}))}/>
                <div id="settings-panel" role="tabpanel" aria-labelledby={"settings-tab-" + settingsSection}>
                {settingsSection === "sharing" && <SharingSettings />}
                {settingsSection === "profile"&&!boot.workspace?.id&&<><RoutingReviewPanel/><NotificationPreferences/></>}
                {boot.workspace?.id &&
                  ["profile", "privacy", "system", "integrations"].includes(
                    settingsSection,
                  ) && (
                    <p className="footnote">
                      Switch to Personal to manage your profile, memory,
                      notifications and connected accounts.
                    </p>
                  )}
                {settingsSection === "profile" && (
                  <section className="density-setting"><PlannerGuide />
                    <label>
                      Display density
                      <select
                        aria-label="Display density"
                        value={density}
                        onChange={(e) =>
                          setDensity(e.target.value as typeof density)
                        }
                      >
                        <option value="compact">Compact</option>
                        <option value="comfortable">Comfortable</option>
                      </select>
                    </label>
                  </section>
                )}
                {settingsSection === "voice" && (
                  <section className="voice-settings settings-sections">
                    <h2>Voice & conversation</h2>
                    <p>
                      Choose how Eri sounds on this device. Changes apply to the
                      next voice session. Voice ends after {VOICE_IDLE_SECONDS} quiet seconds
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
                            {Object.entries(boot.voice_options).map(
                              ([provider, option]) => (
                                <option key={provider} value={provider}>
                                  {option.label}
                                </option>
                              ),
                            )}
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
                      Task agent: {boot.agent_model || "gpt-5.6-luna"}. GPT-Live
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
                )}
                {settingsSection === "integrations" && <BotSettings canDesign={!boot.workspace?.id||boot.workspace?.role==="owner"} key={boot.workspace?.id ?? boot.account_id} workspace={boot.workspace?.name ?? "Personal"} readOnly={boot.workspace?.role === "viewer"}/>}
                {settingsSection === "integrations" && !boot.workspace?.id && (
                  <>
                    <LinearSettings revision={noteRevision} mutate={mutate} />
                    <GoogleSettings
                      revision={noteRevision}
                      voiceActive={!!voiceState && !voiceState.closed}
                      mutate={mutate}
                    />
                  </>
                )}
                {settingsSection !== "sharing" && !boot.workspace?.id && (
                  <SettingsPanel
                    section={settingsSection}
                    boot={boot}
                    busy={busy}
                    onSave={async (args) => {
                      await mutate(
                        "settings.update",
                        args,
                        "Preferences saved",
                      );
                      const data = await api<Bootstrap>("/bootstrap");
                      setBoot(data);
                    }}
                    onPush={enablePush}
                  />
                )}
                </div>
              </>
            )}
            <footer className="page-footer">
              <Shield size={13} />
              <span>{syncWarning || (!boot.capabilities.worker ? "Background work is paused" : activeWork ? `${activeWork} request${activeWork === 1 ? "" : "s"} in progress` : attentionWork ? "Eri has work that needs attention" : "Connected · Changes save automatically")}</span>
            </footer>
          </main>
          <aside
            className={
              "companion " +
              (companion ? "visible " : "") +
              (voiceState && !voiceState.closed ? "voice-mode" : "")
            }
            aria-label="Eridani conversation"
            role={mobile && companion ? "dialog" : undefined}
            aria-modal={mobile && companion ? true : undefined}
          >
            {mobile && companion && <MobileChatFocus />}
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
                  setSettingsSection("voice");
                  setCompanion(false);
                }}
              >
                <Settings2 size={16} />
              </button>
              {mobile && <button className="text-button return-to-work" onClick={() => setCompanion(false)}>Back to work</button>}
              <button
                className="icon-button close-companion"
                aria-label="Close conversation"
                onClick={() => setCompanion(false)}
              >
                <X size={18} />
              </button>
            </div>
            {historyOff && (
              <div className="history-note">
                <Shield size={13} />
                Conversation history is off · Tasks still save
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
                  Sending your request…
                </div>
              )}
              {work.items.filter(item => item.conversation_id === conversationRef.current).slice().reverse().map(item =>
                <WorkCard key={item.id} item={item} onRefresh={work.refresh} onOpen={openWorkRecord}/>)}
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
                      : "Voice is unavailable. Check Settings."
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
                Voice is unavailable. {boot.capabilities.chat ? "You can send a text request." : "The task model also needs to be configured."}
              </p>
            )}
            <p className="chat-footnote">
              {historyOff
                ? "This conversation won’t be retained."
                : "Conversation history is on."}{" "}
              Cloud models process what you send.
            </p>
          </aside>
        </div>
      </div>
      {activityOpen && <ActivityPanel items={work.items} error={work.error} onClose={() => setActivityOpen(false)}
        onRefresh={work.refresh} onOpen={openWorkRecord}/>}
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
        <MemoryEditor
          key={editingMemory.id}
          memory={editingMemory}
          busy={busy}
          mutate={mutate}
          onClose={() => setEditingMemory(null)}
        />
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
            }
            const result = await mutate<Task>(
              selected.id === "new" ? "task.create" : "task.update",
              values,
              "Task saved",
            );
            if (result) { setSelected(null); if (selected.id === "new") openTaskCard(result); }
            return result;
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
      {calendarDetail?.kind === "task" && (
        <TaskDetails
          key={calendarDetail.entity_id}
          id={calendarDetail.entity_id}
          canEdit={boot.workspace?.role !== "viewer"}
          onNewNote={(task) => {
            setNoteEditor(blankNote(task));
            setCalendarDetail(null);
            setError("");
          }}
          onReminder={(schedule, task) => {
            setScheduleEditor({ schedule, task });
            setCalendarDetail(null);
          }}
          onBlock={(task) => {
            setGoogleEvent({
              id: "new",
              entity_id: "new",
              kind: "block",
              title: "Work block",
              date: task.due_date ?? today,
              at: null,
              status: "active",
              project_id: task.project_id,
              task_id: task.id,
              revision: 1,
              projected: false,
              notification_id: null,
            });
            setCalendarDetail(null);
          }}
          organization={organization}
          tasks={tasks}
          schedules={schedules}
          zone={boot.preferences.timezone}
          mutate={mutate}
          onClose={() => setCalendarDetail(null)}
          onTask={openTaskCard}
          onNote={(id) => {
            setCalendarDetail(null);
            void openNote(id);
          }}
        />
      )}
      {calendarDetail && calendarDetail.kind !== "task" && (
        <CalendarDetails
          key={calendarDetail.id}
          event={calendarDetail}
          allTasks={tasks}
          allSchedules={schedules}
          organization={organization}
          zone={boot.preferences.timezone}
          onClose={() => setCalendarDetail(null)}
          onTask={(task) => {
            setCalendarDetail(null);
            openTaskCard(task);
          }}
          onNote={(id) => {
            setCalendarDetail(null);
            void openNote(id);
          }}
          onComplete={async (task) =>
            mutate(
              "task.update",
              {
                task_id: task.id,
                expected_revision: task.revision,
                status: task.status === "completed" ? "open" : "completed",
              },
              task.status === "completed" ? "Task reopened" : "Task completed",
            )
          }
          onEdit={(entry, task, schedule) => {
            setCalendarDetail(null);
            if (entry.kind === "event" || entry.kind === "block")
              setGoogleEvent(entry);
            else if (entry.kind === "task" && task) openTaskCard(task);
            else if (schedule) setScheduleEditor({ schedule });
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
            setSettingsSection("integrations");
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
          onOpenNote={(id) => {
            void openNote(id);
          }}
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
          hideProject
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
            return result;
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
    </div></RecordNavigator>
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
        idle_timeout: `Voice ended after ${VOICE_IDLE_SECONDS} quiet seconds`,
      } as Record<string, string>
    )[state] ?? "Voice is on"
  );
}

function MobileChatFocus() {
  useDialogFocus();
  useEffect(() => { document.querySelector<HTMLButtonElement>(".companion .return-to-work")?.focus({preventScroll:true}); }, []);
  return null;
}
