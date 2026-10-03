import { SourceColorSettings } from "./SourceDetails";
import { VoiceDrafts } from "./VoiceDrafts";
import { useAppHistory } from "./app-history";
import { SettingsLayout, SettingsGroup, SettingRow } from "./SettingsLayout";
import { readSettingsSection, type SettingsSection } from "./settings-sections";
import "./shell.css";
import { NoticeSnooze } from "./NoticeSnooze";
import { StructureWorkspace } from "./Structure";
import { ProfileMenu } from "./ProfileMenu";
import { RecordNavigator, readRecordLink, type LinkedRecord } from "./record-links";
import { MemoryRow, MemoryStatus } from "./MemoryActions";
import { BrandMark, humanLabel, PlannerGuide, Popover, useBodyLock, useMaxWidth, matchesMaxWidth, COMPACT_MAX_WIDTH, PHONE_MAX_WIDTH, NARROW_MAX_WIDTH } from "./ux";
import { PublicFooter } from "./PublicPages";
import { chatTimeline, mergeWorkReplies } from "./chat-timeline";
import { ActivityPanel, WorkCard, useWork, workActive, workAttention, type ActionChange, type WorkItem } from "./Activity";
import { VOICE_IDLE_SECONDS } from "./voice-idle";
import {
  readView,
  viewLink,
  viewStateSchema,
  type ViewState as SavedViewState,
  type SavedView,
} from "./saved-views";
import { AccountSwitcher, SharingSettings } from "./Accounts";
import { TaskDetails } from "./TaskDetails";
import { TaskTabs } from "./TaskTabs";
import { Today } from "./Today";
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
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Bell,
  Brain,
  CalendarDays,
  CalendarPlus,
  ListFilter,
  Check,
  Ellipsis,
  FolderTree,
  NotebookPen,
  ListChecks,
  Clock3,
  Inbox,
  MessageCircle,
  Mic,
  Plus,
  Repeat2,
  Search,
  Settings2,
  Shield,
  Square,
  Sun,
  Trash2,
  VolumeX,
  X,
} from "lucide-react";
import { api, ApiError, command, post, setCsrf, setDevice } from "./api";
import { Voice, type VoiceState } from "./voice";
import { bridgeInterval, createLoader, signalScope, subscribeEvents } from "./events";
import {
  NotesPage,
  NoteEditor,
  TaskNotes,
  blankNote,
  type NoteRecord,
} from "./Notes";
import { GoogleSettings, startGoogle } from "./GoogleSettings";
import type { CalendarEntry } from "./types";
import { BulkTaskDialog } from "./BulkTaskDialog";
import { Workspace } from "./Workspace";
import { ScheduleDialog } from "./ScheduleDialog";
import type { Project } from "./types";
import { MemoryReviewCard } from "./MemoryReviewCard";
import { WakeWord, recognitionType } from "./wake-word";
import { useSiteControl } from "./copilot";
import { ensurePushSubscription, watchPushSubscription } from "./push-subscription";
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
// Heavy views and dialogs that most sessions never open load on demand (and are
// prefetched once the app is idle so opening them stays instant).
const viewChunks = {
  calendarDetails: () => import("./CalendarDetails"),
  googleEvent: () => import("./GoogleEventDialog"),
  planning: () => import("./PlanningDialog"),
  linear: () => import("./Linear"),
  bots: () => import("./BotSettings"),
  preferences: () => import("./PlannerPreferences"),
};
const CalendarDetails = lazy(() => viewChunks.calendarDetails().then((m) => ({ default: m.CalendarDetails })));
const GoogleEventDialog = lazy(() => viewChunks.googleEvent().then((m) => ({ default: m.GoogleEventDialog })));
const PlanningDialog = lazy(() => viewChunks.planning().then((m) => ({ default: m.PlanningDialog })));
const LinearSettings = lazy(() => viewChunks.linear().then((m) => ({ default: m.LinearSettings })));
const LinearTask = lazy(() => viewChunks.linear().then((m) => ({ default: m.LinearTask })));
const BotSettings = lazy(() => viewChunks.bots().then((m) => ({ default: m.BotSettings })));
const RoutingReviewPanel = lazy(() => viewChunks.preferences().then((m) => ({ default: m.RoutingReviewPanel })));
const SchedulingPreferences = lazy(() => viewChunks.preferences().then((m) => ({ default: m.SchedulingPreferences })));
const NotificationPreferences = lazy(() => viewChunks.preferences().then((m) => ({ default: m.NotificationPreferences })));
const ThemeSetting = lazy(() => viewChunks.preferences().then((m) => ({ default: m.ThemeSetting })));
const nav: { id: View; label: string; icon: typeof Sun; tab?: boolean }[] = [
  { id: "today", label: "Today", icon: Sun, tab: true },
  { id: "all", label: "Tasks", icon: ListChecks, tab: true },
  { id: "organize", label: "Organization", icon: FolderTree },
  { id: "calendar", label: "Calendar", icon: CalendarDays, tab: true },
  { id: "notes", label: "Notes", icon: NotebookPen, tab: true },
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
    initialSavedView?.timeline_span ?? (matchesMaxWidth(NARROW_MAX_WIDTH) ? 14 : 30),
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
  const [noteListId, setNoteListId] = useState(() => new URLSearchParams(location.search).get("note_list") ?? "");
  const [notesMode, setNotesMode] = useState<"keyword" | "semantic">("keyword");
  const [collectionContext,setCollectionContext]=useState<Record<string,string|number|null>>({});
  const [recordControl,setRecordControl]=useState<{nonce:string;type_id?:string;parent_id?:string;layout?:string;group?:string;record_id?:string;record_ids?:string[];search_id?:string;proposal_id?:string;field?:string;value?:string;status?:string;archived?:boolean;design?:boolean}>();
  const [taskRecordControl,setTaskRecordControl]=useState<typeof recordControl>();
  useEffect(()=>{const open=(e:Event)=>{setView("organize");setOrganizationEditor(null);setRecordControl({nonce:crypto.randomUUID(),record_id:(e as CustomEvent).detail.id});};window.addEventListener("eri-open-custom-record",open);return()=>window.removeEventListener("eri-open-custom-record",open);},[]);
  const [settingsSection, setSettingsSection] = useState<SettingsSection>(() => {
    const params = new URLSearchParams(location.search), section = params.get("section");
    if (section) return readSettingsSection(section);
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
  const planToday = true; // New tasks on Today and Next 7 days are planned for today.
  const [searchOpen, setSearchOpen] = useState(false);
  const searchLabel = view === "notes" ? "Search notes" : view === "memory" ? "Search memories" : view === "organize" ? "Search organization" : view === "calendar" ? "Search calendar" : view === "notifications" ? "Search notifications" : "Search tasks";
  // View "today" is the Today page; the Tasks page's Today tab shows the same view as a list.
  const [todayList, setTodayList] = useState(() => view === "today" && new URLSearchParams(location.search).get("view") === "tasks");
  useEffect(() => { if (view !== "today") setTodayList(false); }, [view]);
  const dashboard = view === "today" && !todayList;
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
  const pendingNotices = notices.filter((notice) => !notice.completed_at && notice.category !== "work_result");
  const [memories, setMemories] = useState<Memory[]>([]);
  const [memoryReviews, setMemoryReviews] = useState<MemoryReview[]>([]);
  const [maintenance, setMaintenance] = useState<MemoryMaintenance | null>(
    null,
  );
  const [query, setQuery] = useState(initialSavedView?.query ?? ""),
    [quick, setQuick] = useState(""),
    [pair, setPair] = useState("");
  const [taskSearchText, setTaskSearchText] = useState(initialSavedView?.query ?? "");
  useEffect(() => { if (isTaskTab(view)) setTaskSearchText(query); }, [view, query]);
  const [error, setError] = useState(""),
    [toast, setToast] = useState("");
  // Count in-flight foreground operations instead of one shared flag, so overlapping
  // saves cannot clear each other's busy state. Background work (quick captures,
  // completion toggles) never blocks dialogs.
  const [pendingOps, setPendingOps] = useState(0);
  const busy = pendingOps > 0;
  const setBusy = useCallback(
    (on: boolean) => setPendingOps((n) => Math.max(0, n + (on ? 1 : -1))),
    [],
  );
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
  const conversationRef = useRef<string | null>(null);
  const [activeConversation, setActiveConversation] = useState<string | null>(null);
  const work = useWork(!!boot, (boot?.account_id ?? "") + ":" + (boot?.workspace?.id ?? "personal"), activeConversation);
  const chatAnchors = useRef({conversation: conversationRef.current, items: new Map<string, string | null>()});
  if (chatAnchors.current.conversation !== conversationRef.current) chatAnchors.current = {conversation: conversationRef.current, items: new Map()};
  const conversationWork = work.chatItems.filter(item => item.conversation_id === conversationRef.current);
  const chatEntries = chatTimeline(messages, conversationWork, chatAnchors.current.items);
  useEffect(() => {
    setMessages(current => mergeWorkReplies(current, work.chatItems.filter(item => item.conversation_id === conversationRef.current)));
  }, [work.chatItems]);
  const presentedSearchWork = useRef(new Set<string>());
  useEffect(()=>{
    if(!companion)return;
    const observer=new IntersectionObserver(entries=>{
      if(document.visibilityState!=="visible")return;
      for(const entry of entries){
        if(!entry.isIntersecting)continue;
        const native=entry.target.getAttribute("data-native-id");
        const item=work.chatItems.find(w=>w.response_native_id===native);
        if(!item||item.voice_session_id||!item.finished_at||!native||presentedSearchWork.current.has(native))continue;
        presentedSearchWork.current.add(native);
        void post("/search/events",{work_id:item.id,kind:"presented"}).catch(()=>presentedSearchWork.current.delete(native));
      }
    },{threshold:.25});
    document.querySelectorAll(".messages .message.assistant").forEach(element=>observer.observe(element));
    return()=>observer.disconnect();
  },[companion,messages,work.chatItems]);
  const activeWork = work.items.filter(workActive).length;
  const attentionWork = work.items.filter(item => workAttention(item) && !item.seen).length;
  const [voiceState, setVoiceState] = useState<VoiceState | null>(null);
  useEffect(()=>{const colors=boot?.preferences.source_colors??{};for(const [key,value] of Object.entries({eridani:"#9edac8",linear:"#e4a261",google:"#e995bd",...colors}))document.documentElement.style.setProperty("--source-"+key,value);},[boot?.preferences.source_colors]);
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
  const [memoryStatus, setMemoryStatus] = useState<{enabled:boolean;pending:number;retrying:number;deferred:number;queued?:number;active?:number;failed?:number;retry_waiting?:number}>({
    enabled: true,
    pending: 0,
    retrying: 0,
    deferred: 0,
  });
  const [memoryRevision, setMemoryRevision] = useState(0);
  const [editingMemory, setEditingMemory] = useState<Memory | null>(null);
  const mobile = useMaxWidth(COMPACT_MAX_WIDTH);
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
  // One coalesced, single-flight loader for the workspace snapshot (see events.ts).
  // /bootstrap is read first so its event_cursor is a lower bound for the rest of the
  // snapshot; change-feed signals at or below it (including our own write echoes) are dropped.
  const loader = useRef<ReturnType<typeof createLoader> | null>(null);
  if (!loader.current)
    loader.current = createLoader(async ({ revision, current }) => {
      const preferences = await api<Bootstrap>("/bootstrap");
      const [taskData, scheduleData, noticeData, projectData] =
        await Promise.all([
          api<{ items: Task[]; next_cursor: string | null }>("/tasks?limit=200"),
          api<{ items: Schedule[]; next_cursor?: string | null }>("/schedules"),
          api<{ items: Notice[] }>("/notifications"),
          api<Organization>("/organization"),
        ]);
      let items = taskData.items,
        cursor = taskData.next_cursor;
      while (cursor && current()) {
        const page = await api<{ items: Task[]; next_cursor: string | null }>(
          "/tasks?limit=200&before=" + cursor,
        );
        items = [...items, ...page.items];
        cursor = page.next_cursor;
      }
      let scheduleItems = scheduleData.items,
        scheduleCursor = scheduleData.next_cursor;
      while (scheduleCursor && current()) {
        const page = await api<{
          items: Schedule[];
          next_cursor?: string | null;
        }>("/schedules?before=" + scheduleCursor);
        scheduleItems = [...scheduleItems, ...page.items];
        scheduleCursor = page.next_cursor;
      }
      // A superseded (cancelled) load must not overwrite newer state.
      if (!current()) return;
      setBoot(preferences);
      setTasks(items);
      setSchedules(scheduleItems);
      setProjects(projectData.projects);
      setOrganization(projectData);
      setNotices(noticeData.items);
      if (revision) setNoteRevision((n) => n + 1);
      return preferences.event_cursor;
    });
  /** Explicit reload (user action or local change): immediate, refreshes dependent views. */
  const load = useCallback(
    () => loader.current!.request({ immediate: true, revision: true }),
    [],
  );
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
          conversationRef.current = conversation.id; setActiveConversation(conversation.id);
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
    // Warm the on-demand chunks once the first screen has settled.
    const warm = () => Object.values(viewChunks).forEach((chunk) => void chunk().catch(() => {}));
    const idle = window.requestIdleCallback?.(warm, { timeout: 5000 });
    const timer = idle === undefined ? setTimeout(warm, 3000) : undefined;
    return () => {
      if (idle !== undefined) window.cancelIdleCallback?.(idle);
      if (timer) clearTimeout(timer);
    };
  }, [!!boot]);
  const eventScope = boot
    ? boot.account_id + ":" + (boot.workspace?.id ?? "personal")
    : "";
  const eventCursor = useRef(0);
  eventCursor.current = boot?.event_cursor ?? 0;
  useEffect(() => {
    if (!eventScope) return;
    const loads = loader.current!;
    // A new scope starts from that scope's cursor; earlier snapshots say nothing about it.
    loads.reset();
    let online = false;
    const stopEvents = subscribeEvents(
      eventCursor.current,
      (signal) => {
        const scope = signalScope(signal);
        if (scope.memory) setMemoryRevision((v) => v + 1);
        if (!scope.load && !scope.revision) return;
        void loads
          .request({
            revision: scope.revision,
            cursor: signal.reason === "event" ? signal.id : undefined,
          })
          .catch(() => setSyncWarning("Task updates are reconnecting…"));
      },
      (connected) => {
        online = connected;
        setSyncWarning(
          connected
            ? ""
            : "Live task updates are reconnecting. Voice can continue.",
        );
      },
    );
    // Fallback poll only while the change feed is down (or the tab was hidden).
    const timer = setInterval(() => {
      if (online || document.hidden) return;
      loads
        .request({ revision: true })
        .then(() => setSyncWarning(""))
        .catch(() => {});
    }, 30000);
    return () => {
      stopEvents();
      clearInterval(timer);
    };
  }, [eventScope]);
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
  }, [view, query, !!boot, memoryRevision]);
  useEffect(() => {
    const listen = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
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
  // Follow new chat output only while the reader is at (or near) the bottom; never
  // yank someone who scrolled up to reread. Their own new message always scrolls.
  const chatPinned = useRef(true);
  const lastMessage = messages.at(-1);
  useEffect(() => {
    if (!chatPinned.current && lastMessage?.role !== "user") return;
    chatPinned.current = true;
    messageEnd.current?.scrollIntoView({
      block: "nearest",
      behavior: "smooth",
    });
  }, [messages.length, lastMessage?.id, lastMessage?.content, thinking, chatEntries.length]);
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
    options: { background?: boolean } = {},
  ): Promise<T | undefined> {
    // A lost response may hide a committed write. Repeated Save must reuse its receipt.
    const key = JSON.stringify([tool, args]);
    const request =
      pendingCommands.current.get(key) ?? command<unknown>(tool, args);
    pendingCommands.current.set(key, request);
    async function send() {
      if (!options.background) {
        setBusy(true);
        setError("");
      }
      try {
        const result = await request.send();
        pendingCommands.current.delete(key);
        // The memory list is fetched separately; refresh it after memory writes.
        if (tool.startsWith("memory.")) setMemoryRevision((v) => v + 1);
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
        if (!options.background) setBusy(false);
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
  // Quick captures are queued and sent in order; nothing typed is ever dropped while
  // another save is in flight. A failed capture is put back into the input.
  const captureQueue = useRef<{ title: string; args: Record<string, unknown> }[]>([]);
  const capturing = useRef(false);
  const captureContext = useRef({ query, taskFilters, taskStatus, view });
  captureContext.current = { query, taskFilters, taskStatus, view };
  async function drainCaptures() {
    if (capturing.current) return;
    capturing.current = true;
    try {
      while (captureQueue.current.length) {
        const item = captureQueue.current[0];
        const result = await mutate<Task>("task.create", item.args, "Task added", { background: true });
        captureQueue.current.shift();
        if (!result) {
          setQuick((current) => current.trim() ? current : item.title);
          continue;
        }
        // Keep a newly captured record discoverable even when the current filters exclude it.
        const { query, taskFilters, taskStatus, view } = captureContext.current;
        const matches = (!query || result.title.toLowerCase().includes(query.toLowerCase())) && matchesTaskFilters(result, taskFilters) && (taskStatus === "active" || taskStatus === "all" || taskStatus === result.status);
        if (!matches || (!result.planned_date && ["today", "week"].includes(view))) openTaskCard(result);
      }
    } finally {
      capturing.current = false;
    }
  }
  function add(e: React.FormEvent) {
    e.preventDefault();
    const title = quick.trim();
    if (!title) return;
    setError("");
    captureQueue.current.push({
      title,
      args: {
        title,
        project_id:
          view === "inbox"
            ? null
            : (projects.find((p) => p.name === projectFilter)?.id ?? null),
        space_id: view === "inbox" ? null : organizationFilter.space || null,
        area_id: view === "inbox" ? null : organizationFilter.area || null,
        planned_date: planToday && ["today", "week"].includes(view) ? today : null,
      },
    });
    setQuick("");
    void drainCaptures();
  }
  // Completion toggles apply immediately and roll back if the server rejects them.
  const toggling = useRef(new Set<string>());
  async function toggle(task: Task) {
    if (toggling.current.has(task.id)) return;
    toggling.current.add(task.id);
    const completed = task.status === "completed";
    const optimistic: Task = {
      ...task,
      status: completed ? "open" : "completed",
      completed_at: completed ? null : new Date().toISOString(),
    };
    setTasks((items) => items.map((t) => (t.id === task.id ? optimistic : t)));
    try {
      const result = await mutate(
        completed ? "task.reopen" : "task.complete",
        { task_id: task.id, expected_revision: task.revision },
        completed ? "Task reopened" : "Task completed",
        { background: true },
      );
      if (result === undefined)
        setTasks((items) =>
          items.map((t) => (t.id === task.id && t.revision === task.revision ? task : t)),
        );
    } finally {
      toggling.current.delete(task.id);
    }
  }
  async function ensureConversation() {
    if (conversationRef.current) return conversationRef.current;
    const data = await post<{ id: string; private: boolean }>("/conversations", {});
    setConversationHistoryOff(data.private);
    conversationRef.current = data.id; setActiveConversation(data.id);
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
  async function searchAllTasks() {
    try {
      if (editors.current()) await editors.act({operation: "close"});
      setView("all"); setQuery(taskSearchText.trim()); setTaskStatus("all");
      setProjectFilter(""); setOrganizationFilter(emptyFilter); setTaskFilters(emptyTaskFilters);
      setWorkKind("all"); setWorkLayout("list"); setSearchOpen(false); setSidebar(false);
      setTaskRecordControl({nonce: crypto.randomUUID(), type_id: "", parent_id: "", field: "", value: "", status: "all", archived: false, layout: "list"});
    } catch (error) {setError((error as Error).message);}
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
      conversationRef.current = data.id; setActiveConversation(data.id);
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
      setRecordControl({nonce:action.id,type_id:action.type_id,parent_id:action.parent_id,layout:action.layout,group:action.record_group,record_id:action.record_id,record_ids:action.record_ids,search_id:action.search_id,proposal_id:action.proposal_id,field:action.field,value:action.value});
      if(action.record_ids!==undefined){setQuery("");setRecordControl(c=>c?{...c,status:"all",parent_id:"",field:"",value:"",type_id:""}:c);}
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
      // Today opens the Today page unless the request is about the task list (layout, sort or grouping).
      if (action.view === "today") setTodayList(!!(action.layout || action.sort || action.group_by));
      if (
        action.layout &&
        !["all", "inbox", "today", "week", "organize"].includes(target)
      )
        throw new Error(
          "List, board and timeline layouts are available in Tasks and Projects.",
        );
      if (action.organization_tab && target !== "organize")
        throw new Error("Organization tabs belong to Projects.");
      if (action.note_list_id !== undefined && target !== "notes") throw new Error("Choose Notes for a list.");
      if (action.note_list_id && action.note_list_id !== "uncategorized") {
        const lists = await api<{items:{id:string}[]}>("/note-lists");
        if (!lists.items.some(l => l.id === action.note_list_id)) throw new Error("That list is unavailable.");
      }
      if (action.notes_mode && target !== "notes")
        throw new Error("Choose Notes for note search mode.");
      if (action.settings_section && target !== "settings")
        throw new Error("Choose Settings for that section.");
      if (action.layout) {
        if(target === "organize") {
          setRecordControl({nonce:crypto.randomUUID(),layout:action.layout});
          if(action.layout!=="tree")setOrganizationLayout(action.layout);
        } else {
          if(action.layout==="tree")throw new Error("The tree layout belongs to Organization.");
          setWorkLayout(action.layout);
        }
      }
      if (action.sort) setWorkSort(action.sort);
      if (action.group_by) setWorkGroup(action.group_by);
      if (action.timeline_date) setTimelineDate(action.timeline_date);
      if (action.timeline_span) setTimelineSpan(action.timeline_span);
      if (action.organization_tab) setOrganizationTab(action.organization_tab);
      if (action.show_archived !== undefined)
        setShowArchived(action.show_archived);
      if (action.notes_mode) setNotesMode(action.notes_mode);
      if (action.note_list_id !== undefined) setNoteListId(action.note_list_id);
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
              : /notification|quiet|reminder|summary/.test(phrase) ? "notifications"
              : /routing|organization|work hours|filing|rule/.test(phrase) ? "organization"
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
  // Device bridge cadence. The server keeps the screen context for 60 s and a dispatched
  // screen action for 12 s (it waits ~10 s for our acknowledgement), so: poll fast while an
  // agent, chat or voice turn is active; slower when idle; slowest (but well inside the
  // 60 s context window) when hidden. Results are acknowledged immediately.
  const bridgeActive = thinking || activeWork > 0 || (!!voiceState && !voiceState.closed);
  const bridgeInFlight = useRef(false);
  useEffect(() => {
    if (!boot) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const interval = () => bridgeInterval(bridgeActive, document.hidden);
    const schedule = (wait: number) => {
      if (timer) clearTimeout(timer);
      if (!stopped) timer = setTimeout(sync, wait);
    };
    const sync = async () => {
      // Shared across cadence changes so a restart never overlaps an in-flight sync.
      if (bridgeInFlight.current) return schedule(interval());
      bridgeInFlight.current = true;
      let failed = false;
      try {
        await syncUIRef.current();
      } catch {
        failed = true; /* retry with the same acknowledgements */
      } finally {
        bridgeInFlight.current = false;
        // Pending results (from an action just shown) are acknowledged right away.
        schedule(!failed && uiResults.current.length ? 0 : interval());
      }
    };
    const visibility = () => { if (!document.hidden) void sync(); else schedule(interval()); };
    document.addEventListener("visibilitychange", visibility);
    void sync();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [!!boot, bridgeActive]);
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
    conversationRef.current = null; setActiveConversation(null);
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
    setMessages((m) => [...m, { id: turn_id, role: "user", content, created_at: new Date().toISOString() }]);
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
                created_at: existing?.created_at ?? new Date().toISOString(),
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
              ? m.map((item) => (item.id === message.id ? {...message, created_at: item.created_at ?? message.created_at} : item))
              : [...m, {...message, created_at: message.created_at ?? new Date().toISOString()}],
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
          setMessages(items => [...items, {id, role:"user", content:request, created_at:new Date().toISOString()}]);
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
  const pushKey = boot?.capabilities.push ? boot.vapid_public_key : null;
  useEffect(() => {
    if (!pushKey) return;
    void ensurePushSubscription(pushKey);
    return watchPushSubscription(pushKey);
  }, [pushKey, boot?.account_id, boot?.workspace?.id]);
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
  const openCount = open.filter((t) => !t.archived).length;
  const todayCount = open.filter((t) => !t.archived && scheduledBy(t, today)).length;
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
    layout: view === "organize" ? (organizationEditor ? organizationLayout : (collectionContext.layout??"tree") as "tree"|WorkLayout) : workLayout,
    sort: workSort,
    group_by: workGroup,
    timeline_date: timelineDate || today,
    timeline_span: timelineSpan,
    organization_tab: organizationTab,
    settings_section: settingsSection,
    note_list_id: noteListId,
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
  const previousLayer = useRef({record: null as string | null, structure: false});
  useEffect(() => {
    const previous = previousLayer.current;
    const record = editors.summary?.kind === "record" ? editors.summary.record_id : null;
    const structure = !!collectionContext.design;
    if ((previous.record && !record) || (previous.structure && !structure)) {
      // Opening a card is a one-time command, not a default for the next page visit.
      const consume = (control: typeof recordControl) => control ? {...control,
        ...(control.record_id === previous.record && !record ? {record_id: undefined} : {}),
        ...(previous.structure && !structure ? {design: false, proposal_id: undefined} : {})} : control;
      setRecordControl(consume); setTaskRecordControl(consume);
    }
    previousLayer.current = {record, structure};
  }, [editors.summary?.kind, editors.summary?.record_id, collectionContext.design]);
  const detailKey = (detail: typeof editors.summary) => detail ? `${detail.kind}:${detail.record_id ?? "new"}:${detail.mode}` : "";
  const navigation = {view, settingsSection, query, savedViewState, collectionContext, calendarMode, calendarDay,
    noteListId, notesMode, showArchived, todayList, companion, activityOpen, sidebar, searchOpen, detail: editors.summary,
    selected, reminder, scheduleEditor, noteEditor, googleEvent, calendarDetail, bulkEditor, editingMemory, organizationEditor};
  const navigationUrl = isTaskTab(view) && !dashboard ? viewLink(savedViewState) : new URL(location.href);
  if (!isTaskTab(view) || dashboard) {navigationUrl.searchParams.set("view", view); navigationUrl.searchParams.delete("tab"); navigationUrl.searchParams.delete("state");}
  if (view === "settings") navigationUrl.searchParams.set("section", settingsSection); else navigationUrl.searchParams.delete("section");
  if (editors.summary?.record_id && ["record", "task", "note", "project", "goal", "area", "space", "actor"].includes(editors.summary.kind)) {
    navigationUrl.searchParams.set("record", editors.summary.kind + ":" + editors.summary.record_id);
    navigationUrl.searchParams.set("workspace", boot?.workspace?.id ?? "personal");
  } else if (!linkWorkspace) {navigationUrl.searchParams.delete("record"); navigationUrl.searchParams.delete("workspace");}
  if (view === "notes" && noteListId) navigationUrl.searchParams.set("note_list", noteListId); else navigationUrl.searchParams.delete("note_list");
  if (activityOpen) navigationUrl.searchParams.set("activity", "1"); else navigationUrl.searchParams.delete("activity");
  useAppHistory({enabled: !!boot && !loading, snapshot: navigation,
    page: view + (view === "settings" ? ":" + settingsSection : view === "notes" ? ":" + noteListId : ""),
    layers: [sidebar ? "navigation" : "", searchOpen ? "search" : "", companion ? "chat" : "", activityOpen ? "activity" : "", collectionContext.design ? "structure" : "", detailKey(editors.summary)].filter(Boolean),
    url: navigationUrl.href, onError: setError,
    restore: async target => {
      if (collectionContext.design && collectionContext.design_dirty && (!target.collectionContext.design || view !== target.view)) {
        throw new Error("Apply your structure changes or close the structure editor before leaving.");
      }
      const detailChanged = detailKey(editors.summary) !== detailKey(target.detail);
      if (detailChanged && editors.current()) await editors.act({operation: "close"});
      if (isTaskTab(target.view)) applySavedView(target.savedViewState); else setView(target.view);
      setTodayList(target.todayList);
      setQuery(target.query); setSettingsSection(target.settingsSection); setCalendarMode(target.calendarMode);
      setCalendarDay(target.calendarDay); setNotesMode(target.notesMode); setNoteListId(target.noteListId); setShowArchived(target.showArchived);
      setCompanion(target.companion); setActivityOpen(target.activityOpen); setSidebar(target.sidebar); setSearchOpen(target.searchOpen);
      const collection = target.collectionContext;
      (isTaskTab(target.view) ? setTaskRecordControl : setRecordControl)({nonce: crypto.randomUUID(), type_id: String(collection.type_id ?? ""), parent_id: String(collection.parent_id ?? ""),
        layout: String(collection.layout ?? "list"), group: String(collection.group ?? "status"), field: String(collection.field ?? ""),
        value: String(collection.value ?? ""), status: String(collection.status ?? "active"), archived: !!collection.archived, design: !!collection.design,
        ...(detailChanged && target.detail?.kind === "record" && target.detail.record_id ? {record_id: target.detail.record_id} : {})});
      if (detailChanged) {
        setSelected(target.selected); setReminder(target.reminder); setScheduleEditor(target.scheduleEditor);
        setNoteEditor(target.noteEditor); setGoogleEvent(target.googleEvent); setCalendarDetail(target.calendarDetail);
        setBulkEditor(target.bulkEditor); setEditingMemory(target.editingMemory); setOrganizationEditor(target.organizationEditor);
        const record = target.detail;
        if (record?.record_id && ["task", "note", "project", "goal", "area", "space", "actor"].includes(record.kind)) {
          await openLinkedRecord({kind: record.kind as LinkedRecord["kind"], id: record.record_id});
          setView(target.view);
        }
        if (record) {
          const deadline = performance.now() + 5000;
          while ((editors.current()?.kind !== record.kind || (editors.current()?.record_id ?? null) !== record.record_id) && performance.now() < deadline) {
            await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
          }
        }
      }
    }});
  useEffect(() => {
    if (companion) requestAnimationFrame(() => document.getElementById("eri-conversation")?.focus({preventScroll: true}));
  }, [companion]);
  if (loading)
    return (
      <div className="splash">
        <BrandMark size={44} />
        <p>Opening Eridani…</p>
      </div>
    );
  if (!boot)
    return (
      <div className="login-page">
        <form className="login-card" onSubmit={signIn}>
          <div className="login-brand">
            <BrandMark size={36} />
            <span>eridani</span>
          </div>
          <h1>A little more organized.</h1>
          <p>
            Your day, with Eri.{" "}
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
  const workFilterCount = [
    taskStatus !== "active",
    !!projectFilter,
    ...Object.values(organizationFilter).map(Boolean),
    ...Object.values(taskFilters).map(Boolean),
    workKind !== "all",
  ].filter(Boolean).length;
  // Calendar and reminder filters, shared by the calendar toolbar popover and the reminders panel.
  const workFilterFields = (
    <>
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
                            {humanLabel(v)}
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
                          {humanLabel(v)}
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
                    className="btn btn-ghost btn-sm filter-reset"
                    onClick={() => {
                      setTaskStatus("active");
                      setProjectFilter("");
                      setOrganizationFilter(emptyFilter);
                      setTaskFilters(emptyTaskFilters);
                      setWorkKind("all");
                      setQuery("");
                    }}
                  >
                    Reset filters
                  </button>
    </>
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
      <aside className={"sidebar " + (sidebar ? "open" : "")} aria-label="Sidebar">
        <a
          className="brand"
          href="/"
          onClick={(e) => {
            e.preventDefault();
            setView("today");
            setSidebar(false);
          }}
        >
          <BrandMark size={28} />
          <span className="brand-word">eridani</span>
          <span className="sr-only">{boot.workspace?.id ? "Shared workspace" : "Personal workspace"}</span>
        </a>
        <nav aria-label="Main">
          {nav.map((item) => {
            const current = item.id === "today" ? dashboard : item.id === "all" ? isTaskTab(view) && !dashboard : view === item.id;
            const count = item.id === "today" ? todayCount : item.id === "all" ? openCount : 0;
            return (
              <button
                key={item.id}
                aria-label={item.label}
                aria-current={current ? "page" : undefined}
                className={current ? "nav-item active" : "nav-item"}
                onClick={() => {
                  setView(item.id === "all" ? (lastTaskTab.current === "today" ? "inbox" : lastTaskTab.current) : item.id);
                  if (item.id === "today") setTodayList(false);
                  setQuery("");
                  setSidebar(false);
                }}
              >
                <item.icon size={18} aria-hidden="true" />
                <span>{item.label}</span>
                {count > 0 && <span className="nav-count" aria-hidden="true">{count}</span>}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
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
          <ProfileMenu name={boot.name} view={view} personal={!boot.workspace?.id} navigationOpen={sidebar}
            onNavigate={target => {
              setView(target);
              setQuery("");
              setSidebar(false);
              if (matchesMaxWidth(PHONE_MAX_WIDTH))
                document.querySelector<HTMLButtonElement>(".mobile-menu")?.focus({preventScroll: true});
            }}
            onLogout={async () => {
              await voice.current?.stop();
              await post("/auth/logout");
              setBoot(null);
              setTasks([]);
              setMessages([]);
              conversationRef.current = null; setActiveConversation(null);
              sessionStorage.removeItem("jarvis-conversation");
            }}/>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <h1 className="topbar-title">{dashboard ? "Today" : titles[view]}</h1>
          {mobile && <button className="icon-button mobile-search" aria-label="Search tasks" onClick={() => {setSearchOpen(true); requestAnimationFrame(() => searchRef.current?.focus());}}><Search size={20}/></button>}
          <form role="search" aria-label="Task search" className={"search " + (searchOpen ? "search-expanded" : "")} onSubmit={e => {e.preventDefault();void searchAllTasks();}}>
            <Search size={17} aria-hidden="true" />
            <input
              ref={searchRef}
              value={taskSearchText}
              onChange={(e) => setTaskSearchText(e.target.value)}
              placeholder="Search tasks, notes, people…"
              aria-label="Search tasks"
            />
            <kbd aria-hidden="true">{/Mac|iPhone|iPad/.test(navigator.platform) ? "⌘K" : "Ctrl K"}</kbd>
            <button type="submit" className="icon-button search-submit" aria-label="Find tasks"><ArrowUp size={16}/></button>
            {mobile && <button type="button" className="icon-button" aria-label="Close search" onClick={() => setSearchOpen(false)}><X size={18}/></button>}
          </form>
          <div className="top-actions">
            <button className={"btn btn-ghost activity-toggle " + (attentionWork ? "needs-attention" : "")}
              aria-label={"Eri activity, " + activeWork + " pending, " + attentionWork + " need attention"}
              title="Eri activity" onClick={() => setActivityOpen(true)}>
              <Clock3 size={18}/><span>Activity</span>{(activeWork + attentionWork > 0) && <b>{activeWork + attentionWork}</b>}
            </button>
            <button
              className={"icon-button " + (unread ? "has-notice" : "")}
              aria-label={
                "Notifications" + (unread ? ", " + unread + " unread" : "")
              }
              aria-current={view === "notifications" ? "page" : undefined}
              onClick={() => setView("notifications")}
            >
              <Bell size={19} />
              {unread > 0 && <i />}
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
          <main className={"content" + (view === "settings" ? " settings-page" : "")} inert={mobile && companion}>
            {!dashboard && view !== "calendar" && <div className="page-heading">
              <h1>{titles[view]}</h1>
            </div>}
            {view === "calendar" && <div className="page-heading cal-page-head">
              <h1>{titles[view]}</h1>
              <div className="cal-toolbar">
                <label className="toolbar-search"><Search size={16} aria-hidden="true"/><input type="search" aria-label={searchLabel} placeholder={searchLabel + "…"} value={query} onChange={e => setQuery(e.target.value)}/></label>
                <Popover label={"Filters" + (workFilterCount ? ", " + workFilterCount + " active" : "")} panelClassName="filter-popover cal-filter-popover"
                  button={<><ListFilter size={16} aria-hidden="true"/><span className="toolbar-label">Filters</span>{workFilterCount > 0 && <span className="toolbar-count">{workFilterCount}</span>}</>}>
                  {() => <div className="filter-fields task-filters">{workFilterFields}</div>}
                </Popover>
                <Popover label="Actions" className="cal-actions" panelClassName="menu-popover"
                  button={<><Plus size={16} aria-hidden="true"/><span className="toolbar-label">Actions</span></>}>
                  {close => <div className="menu-list">
                    <button type="button" className="menu-item" onClick={() => { close(); setGoogleEvent({ id: "new", entity_id: "new", kind: "event", title: "New event", date: calendarDay || today, at: null, status: "active", project_id: null, task_id: null, revision: 1, projected: false, notification_id: null }); }}><CalendarPlus size={16} aria-hidden="true"/>New event</button>
                    <button type="button" className="menu-item" onClick={() => { close(); createTask(calendarDay || today); }}><Plus size={16} aria-hidden="true"/>New task</button>
                    <button type="button" className="menu-item" onClick={() => { close(); setScheduleEditor({ schedule: null, date: calendarDay || today }); }}><Clock3 size={16} aria-hidden="true"/>Task reminder</button>
                  </div>}
                </Popover>
              </div>
            </div>}
            {!isTaskTab(view) && view !== "settings" && view !== "notes" && view !== "organize" && view !== "calendar" && <label className="page-search"><Search size={16}/><input aria-label={searchLabel} placeholder={searchLabel + "…"} value={query} onChange={e => setQuery(e.target.value)}/></label>}
            {dashboard && boot && <Today tasks={tasks} schedules={schedules} today={today} zone={boot.preferences.timezone} busy={busy}
              noteRevision={noteRevision} quick={quick} onQuick={setQuick} onAdd={add} onTask={openTaskCard} onToggle={task => void toggle(task)}
              onEntry={openCalendarEntry} onNote={id => void openNote(id)}
              onNavigate={target => { setQuery(""); if (target === "tasks-today") { setTodayList(true); } else setView(target); }}/>}
            {view === "reminders" && (
              <div className="view-control-row">
              <details className="filter-panel">
                <summary>
                  Filters{workFilterCount > 0 && <span>{workFilterCount}</span>}
                </summary>
                <div className="task-filters">{workFilterFields}</div>
              </details>
              </div>
            )}
            {["calendar", "reminders"].includes(view) && (
              <div className="active-filters" aria-label="Active filters">
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
            {isTaskTab(view)&&!dashboard&&<StructureWorkspace tabs={<TaskTabs value={view} onChange={tab=>{setTodayList(tab==="today");setView(tab);}}/>} viewLink={()=>viewLink(savedViewState).href} onApplyTaskView={state=>{const parsed=viewStateSchema.safeParse(state);if(parsed.success){applySavedView(parsed.data);setTodayList(parsed.data.tab==="today");}}} sortFilter={workSort} control={taskRecordControl} selecting={selectingTasks} selectedIds={selectedTaskIds} onSelecting={value=>{setSelectingTasks(value);if(!value)setSelectedTaskIds([]);}} onSelection={setSelectedTaskIds} onBulk={()=>{setError("");setBulkEditor(tasks.filter(t=>selectedTaskIds.includes(t.id)));}} key={boot.workspace?.id??"personal"} onContext={setCollectionContext} statusFilter={taskStatus} homeFilter={organizationFilter.area||organizationFilter.space||projects.find(p=>p.name===projectFilter)?.id||""} layoutFilter={workLayout} groupFilter={workGroup} onQuery={setQuery} onTab={tab=>{setTodayList(tab==="today");setView(tab);}} capability="work" tab={view} today={today} query={query} canEdit={boot.workspace?.role!=="viewer"} canDesign={!boot.workspace?.id||boot.workspace?.role==="owner"} refresh={noteRevision} onChanged={()=>void load()} onVisible={setWorkVisible} onTask={id=>{const task=tasks.find(t=>t.id===id);if(task)openTaskCard(task);}}/>}
            {view === "notifications" && (
              <div className="notice-page">
                <div className="notice-page-head">
                  <h2>
                    Notifications<span>{pendingNotices.length}</span>
                  </h2>
                  <button
                    className="btn btn-ghost btn-sm"
                    onClick={() => void enablePush()}
                  >
                    <Bell size={15} />
                    Enable notifications
                  </button>
                </div>
                {!!pendingNotices.length && <div className="notice-list">
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
                      <span className="notice-icon" aria-hidden="true"><Bell size={17} /></span>
                      <div className="notice-text">
                        <strong className="notice-title">{n.title}</strong>
                        {n.body && <p className="notice-body">{n.body}</p>}
                      </div>
                      <time className="notice-time" dateTime={n.scheduled_at}>
                        {timeLabel(n.scheduled_at, boot.preferences.timezone)}
                      </time>
                      <div className="notice-actions">
                        <button className="notice-primary" onClick={()=>{if(n.task_id){const task=tasks.find(t=>t.id===n.task_id);if(task)openTaskCard(task);}else if(n.target?.view==="activity")setActivityOpen(true);else setView("today");void mutate("notification.read",{notification_id:n.id},"");}}>Open</button>
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
                </div>}
                {!pendingNotices.length && (
                  <div className="notice-empty">
                    <strong>You're all caught up.</strong>
                    <p>Reminders appear here when they are due.</p>
                  </div>
                )}
              </div>
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
            {view === "organize" && !organizationEditor && <StructureWorkspace today={today} searchLabel="Search organization" onVisible={setOrganizationVisible} onContext={setCollectionContext} onQuery={setQuery} control={recordControl} query={query} canEdit={boot.workspace?.role!=="viewer"} canDesign={!boot.workspace?.id||boot.workspace?.role==="owner"} refresh={noteRevision} onChanged={()=>void load()}/>}

            {view === "notes" && (
              <NotesPage
                listId={noteListId} onList={setNoteListId} canEdit={boot.workspace?.role !== "viewer"} onQuery={setQuery}
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
              <div className="memory-page">
                <MemoryStatus status={memoryStatus} worker={!!boot.capabilities.worker} reviews={memoryReviews.length} maintenance={maintenance}
                  timezone={boot.preferences.timezone} busy={busy}
                  onRetry={async () => {
                    const result = await post<{ queued: number }>("/memory/retry");
                    setToast(result.queued ? "Memory learning queued again" : "Learning is already retrying");
                    setMemoryRevision((v) => v + 1);
                  }}
                  onReview={async () => {
                    try {
                      await post("/memory/review");
                      setToast("Memory review queued");
                      setMemoryRevision((v) => v + 1);
                    } catch (e) {
                      setError((e as Error).message);
                    }
                  }}
                />
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
                <section className="memory-section" aria-labelledby="memory-remembered">
                  <div className="memory-section-head">
                    <h2 id="memory-remembered">Remembered</h2>
                    <span className="panel-count">{memories.length}</span>
                    <span className="memory-section-sub">Personal facts and preferences</span>
                  </div>
                  {memories.length > 0 && (
                    <div className="row-list memory-list">
                      {memories.map((m) => (
                        <MemoryRow key={m.id} memory={m} onCorrect={() => setEditingMemory(m)} onForget={delete_source => mutate("memory.forget", {memory_id:m.id, delete_source}, delete_source ? "Memory and its stored source deleted" : "Memory forgotten")}/>
                      ))}
                    </div>
                  )}
                  {!memories.length && (
                    <div className="empty">
                      <p>
                        {query
                          ? "No memories match this search."
                          : "Nothing remembered yet. Tell Eri something above, or let it learn from saved conversations."}
                      </p>
                    </div>
                  )}
                </section>
              </div>
            )}
            {view === "settings" && (
              <>
                <SettingsLayout section={settingsSection} onChange={setSettingsSection}>
                <Suspense fallback={<p className="subtle" role="status">Loading…</p>}>
                {settingsSection === "sharing" && <SharingSettings />}
                {settingsSection === "organization"&&!boot.workspace?.id&&<><SchedulingPreferences/><RoutingReviewPanel/></>}
                {settingsSection === "notifications"&&!boot.workspace?.id&&<NotificationPreferences/>}
                {boot.workspace?.id &&
                  ["profile", "organization", "notifications", "privacy", "system", "integrations"].includes(
                    settingsSection,
                  ) && (
                    <p className="footnote">
                      Switch to Personal to manage your profile, memory,
                      notifications and connected accounts.
                    </p>
                  )}
                {settingsSection === "profile" && (
                  <SettingsGroup className="density-setting" title="Display">
                    <ThemeSetting />
                    {!boot.workspace?.id&&<SourceColorSettings/>}
                    <SettingRow as="label" label="Display density" hint="Compact fits more rows on screen.">
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
                    </SettingRow>
                    <PlannerGuide />
                  </SettingsGroup>
                )}
                {settingsSection === "voice" && (
                  <SettingsGroup className="voice-settings" title="Voice & conversation"
                    description={"Choose how Eri sounds on this device. Changes apply to the next voice session. Voice ends after " + VOICE_IDLE_SECONDS + " quiet seconds following the last response."}>
                    {voiceState && !voiceState.closed && (
                      <p className="settings-callout">
                        End the active voice session before changing its voice.
                      </p>
                    )}
                    {Object.keys(boot.voice_options).length > 1 && (
                      <SettingRow label="Voice mode" hint={voiceProvider === "live"
                        ? "GPT-Live: natural, simultaneous listening and speaking. $0.05 per connected minute, plus task work."
                        : "Realtime: the existing turn-based voice experience."}>
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
                      </SettingRow>
                    )}
                    <SettingRow label="Voice" hint={"Task agent: " + (boot.agent_model || "gpt-5.6-luna") + ". GPT-Live delegates task work to this agent using your saved records and tools."}>
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
                    </SettingRow>
                    <SettingRow as="label" className="switch-row" label="Wake word"
                      hint={<>Say “Eri” or “Hey, Eri”. {wakeEnabled
                          ? voiceState && !voiceState.closed
                            ? "Paused during voice."
                            : wakeStatus
                          : wakeStatus ||
                            (recognitionType()
                              ? "Opt in. Uses the browser speech service while the page is open."
                              : "Unavailable in this browser.")}</>}>
                      <input
                        type="checkbox"
                        className="switch"
                        role="switch"
                        aria-label="“Eri” or “Hey, Eri”"
                        checked={wakeEnabled}
                        disabled={!recognitionType()}
                        onChange={(e) => {
                          setWakeEnabled(e.target.checked);
                          if (!e.target.checked) setWakeStatus("");
                        }}
                      />
                    </SettingRow>
                  </SettingsGroup>
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
                </Suspense>
                </SettingsLayout>
              </>
            )}
            <footer className="page-footer">
              <Shield size={13} />
              <span>{syncWarning || (!boot.capabilities.worker ? "Background work is paused" : activeWork ? `${activeWork} request${activeWork === 1 ? "" : "s"} in progress` : attentionWork ? "Eri has work that needs attention" : "Connected. Changes save automatically.")}</span>
            </footer>
          </main>
          <aside
            id="eri-conversation"
            tabIndex={-1}
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
                Conversation history is off. Tasks still save.
              </div>
            )}
            <div
              className="messages"
              onScroll={(event) => {
                const el = event.currentTarget;
                chatPinned.current =
                  el.scrollHeight - el.scrollTop - el.clientHeight < 80;
              }}
            >
              {!messages.length && (
                <div className="conversation-empty">
                  <span className="eri-orb conversation-orb" aria-hidden="true" />
                  <p>What can I help with?</p>
                </div>
              )}
              {chatEntries.map(entry => {
                if (entry.kind === "work") return <WorkCard key={"work:" + entry.work.id} item={entry.work} compact onRefresh={work.refresh} onOpen={openWorkRecord}/>;
                const m = entry.message;
                return (
                <div className={"message " + m.role} key={m.id} data-message-id={m.id} data-native-id={m.native_id??m.id}>
                  <span className="message-label">
                    {m.role === "assistant" ? <><span className="eri-orb" aria-hidden="true"/>Eri</> : "You"}
                  </span>
                  <p>{m.content || (m.pending ? "Listening…" : "")}</p>
                  {m.pending && (
                    <span className="transcript-status">
                      {m.role === "user" ? "Transcribing…" : "Speaking…"}
                    </span>
                  )}
                </div>
              );})}
              {thinking && (
                <div className="thinking">
                  <span />
                  Sending your request…
                </div>
              )}
              <VoiceDrafts key={(boot.account_id ?? "") + ":" + (boot.workspace?.id ?? "personal")} />
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
      {mobile && companion && <button className="chat-scrim" aria-label="Close conversation backdrop" onClick={() => setCompanion(false)}/>}
      <div className="tabbar" role="navigation" aria-label="Phone navigation">
        {nav.filter(item => item.tab).map(item => {
          const current = item.id === "today" ? dashboard : item.id === "all" ? isTaskTab(view) && !dashboard : view === item.id;
          return <button key={item.id} type="button" className={"tabbar-item" + (current ? " active" : "")} aria-current={current ? "page" : undefined}
            onClick={() => {
              setView(item.id === "all" ? (lastTaskTab.current === "today" ? "inbox" : lastTaskTab.current) : item.id);
              if (item.id === "today") setTodayList(false);
              setQuery("");
              setSidebar(false);
            }}>
            <item.icon size={22} aria-hidden="true" /><span>{item.label}</span>
          </button>;
        })}
        <button type="button" className={"tabbar-item mobile-menu" + (sidebar ? " active" : "")} aria-label="Open navigation" aria-expanded={sidebar}
          onClick={() => setSidebar(true)}>
          <Ellipsis size={22} aria-hidden="true" /><span aria-hidden="true">More</span>
        </button>
      </div>
      <button type="button" className={"chat-launcher" + (companion ? " is-open" : "") + (voiceState && !voiceState.closed ? " is-listening" : "")}
        aria-label={companion ? "Close Eridani" : "Open Eridani"} aria-controls="eri-conversation" aria-expanded={companion}
        onClick={() => setCompanion(!companion)}>
        <span className="chat-launcher-icon" aria-hidden="true">{companion ? <X size={23}/> : <MessageCircle size={24}/>}</span>
        {!companion && <span className="chat-launcher-label">Eri</span>}
        {voiceState && !voiceState.closed && <span className="chat-live-dot" aria-label="Voice active"/>}
      </button>
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
              ? "Listening, " + voiceState.idle_seconds + "s left"
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
                <Suspense fallback={null}><LinearTask
                  task={selected}
                  mutate={mutate}
                  onChanged={() => {
                    setSelected(null);
                    void load();
                  }}
                /></Suspense>
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
        <Suspense fallback={null}><CalendarDetails
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
        /></Suspense>
      )}
      {googleEvent && googleEvent.kind !== "google" && (
        <Suspense fallback={null}><PlanningDialog
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
        /></Suspense>
      )}
      {googleEvent && googleEvent.kind === "google" && (
        <Suspense fallback={null}><GoogleEventDialog
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
        /></Suspense>
      )}
      {noteEditor && (
        <NoteEditor
          readOnly={boot.workspace?.role === "viewer"}
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
        ended: "Voice ended. Say Hey Eri when you need me.",
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
