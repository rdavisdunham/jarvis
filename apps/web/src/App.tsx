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
import type {
  Bootstrap,
  ChatMessage,
  Memory,
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
  { id: "all", label: "All tasks", icon: ListTodo },
  { id: "reminders", label: "Reminders", icon: Clock3 },
  { id: "memory", label: "Memory", icon: Brain },
];
export default function App() {
  const [boot, setBoot] = useState<Bootstrap | null>(null),
    [loading, setLoading] = useState(true);
  const [view, setView] = useState<View>(
    new URLSearchParams(location.search).get("view") === "notifications"
      ? "notifications"
      : "today",
  );
  const [tasks, setTasks] = useState<Task[]>([]),
    [schedules, setSchedules] = useState<Schedule[]>([]),
    [notices, setNotices] = useState<Notice[]>([]);
  const [memories, setMemories] = useState<Memory[]>([]),
    [legacy, setLegacy] = useState<
      { id: string; content: string; attribution: string }[]
    >([]);
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
  const [voiceState, setVoiceState] = useState<VoiceState | null>(null);
  const voice = useRef<Voice | null>(null),
    conversationRef = useRef<string | null>(null),
    retryRef = useRef<null | (() => Promise<void>)>(null),
    lastVoiceText = useRef("");
  const searchRef = useRef<HTMLInputElement>(null),
    messageEnd = useRef<HTMLDivElement>(null);
  const load = useCallback(async () => {
    const [taskData, scheduleData, noticeData] = await Promise.all([
      api<{ items: Task[]; next_cursor: string | null }>("/tasks?limit=200"),
      api<{ items: Schedule[] }>("/schedules"),
      api<{ items: Notice[] }>("/notifications"),
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
    setTasks(items);
    setSchedules(scheduleData.items);
    setNotices(noticeData.items);
  }, []);
  const initialize = useCallback(async () => {
    try {
      const info = await api<Bootstrap>("/bootstrap");
      setCsrf(info.csrf);
      setBoot(info);
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
    const events = new EventSource("/api/v1/events?after=" + boot.event_cursor);
    events.onmessage = () => {
      void load().catch(() =>
        setError("Connection lost. Reconnecting to your saved data…"),
      );
    };
    events.addEventListener("refresh", () => {
      void load().catch(() => {});
    });
    events.onopen = () => setError("");
    events.onerror = () => setError("Connection interrupted. Reconnecting…");
    const timer = setInterval(() => {
      api<Bootstrap>("/bootstrap")
        .then((info) => setBoot(info))
        .catch(() => {});
    }, 30000);
    return () => {
      events.close();
      clearInterval(timer);
    };
  }, [!!boot, load]);
  useEffect(() => {
    if (view !== "memory" || !boot) return;
    let current = true;
    const timer = setTimeout(() => {
      api<{ items: Memory[]; legacy: typeof legacy }>(
        "/memory?q=" + encodeURIComponent(query),
      )
        .then((data) => {
          if (current) {
            setMemories(data.items);
            setLegacy(data.legacy);
          }
        })
        .catch((e) => setError(e.message));
    }, 200);
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [view, query, !!boot, toast]);
  useEffect(() => {
    const listen = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        searchRef.current?.focus();
      }
      if (e.key === "Escape") {
        setSelected(null);
        setReminder(false);
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
    const request = command<T>(tool, args);
    async function send() {
      setBusy(true);
      setError("");
      try {
        const result = await request.send();
        await load();
        setToast(success);
        retryRef.current = null;
        return result.data;
      } catch (e) {
        const err = e as ApiError;
        setError(err.message);
        if (
          err.code === "REVISION_CONFLICT" &&
          tool.startsWith("task.") &&
          err.data
        )
          setSelected(err.data as Task);
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
  async function sendChat(e?: React.FormEvent) {
    e?.preventDefault();
    if (!chatText.trim() || thinking) return;
    const content = chatText.trim(),
      turn_id = crypto.randomUUID();
    setChatText("");
    setMessages((m) => [...m, { id: turn_id, role: "user", content }]);
    let conversation_id: string;
    try {
      conversation_id = await ensureConversation();
    } catch (e) {
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
        const result = await post<{ message: string; status: string }>(
          "/chat",
          body,
        );
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
      await voice.current.stop();
      voice.current = null;
      setVoiceState(null);
      return;
    }
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
      const controller = new Voice((state) => {
        setVoiceState(state);
        if (state.text && state.text !== lastVoiceText.current) {
          lastVoiceText.current = state.text;
          setMessages((m) => [
            ...m,
            { id: crypto.randomUUID(), role: "assistant", content: state.text },
          ]);
        }
        if (state.closed) voice.current = null;
        if (state.receipts.length) void load();
      });
      voice.current = controller;
      await controller.start(id, selected?.id);
    } catch (e) {
      voice.current = null;
      setVoiceState(null);
      setError((e as Error).message);
    }
  }
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
      if (view === "today") return !!t.due_date && t.due_date <= today;
      if (view === "week") return !!t.due_date && t.due_date <= weekEnd;
      if (view === "inbox") return !t.project;
      return true;
    })
    .sort(
      (a, b) =>
        b.priority - a.priority ||
        (a.due_date ?? "9999").localeCompare(b.due_date ?? "9999") ||
        b.created_at.localeCompare(a.created_at),
    );
  const active = filtered.filter(
      (t) => !["completed", "cancelled"].includes(t.status),
    ),
    done = filtered.filter((t) => t.status === "completed"),
    unread = notices.filter((n) => !n.read_at).length;
  const titles: Record<View, string> = {
    today: "A little space for your day.",
    inbox: "Catch it. Clear your head.",
    week: "A look at the week ahead.",
    all: "Everything, in one place.",
    reminders: "The things worth a nudge.",
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
    <div className="app-shell">
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
        <div className="sidebar-note">
          <Shield size={17} />
          <div>
            <strong>Your records live at home</strong>
            <span>Saved on your personal server</span>
          </div>
        </div>
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
                Connected to home
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
                (view === "settings" ? "Settings" : "Notifications")}
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
                  view === "memory" ? "Search memories" : "Find a task…"
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
            {retryRef.current && (
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
                    (view === "settings" ? "Settings" : "Notifications"))}
              </h1>
              <p>{titles[view]}</p>
            </div>
            {["today", "inbox", "week", "all"].includes(view) && (
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
            {view === "reminders" && (
              <>
                <div className="section-head">
                  <h2>
                    Scheduled
                    <span>
                      {schedules.filter((s) => s.status === "active").length}
                    </span>
                  </h2>
                  <button
                    className="primary compact"
                    onClick={() => setReminder(true)}
                  >
                    <Plus size={16} />
                    New reminder
                  </button>
                </div>
                {schedules
                  .filter((s) => s.status === "active")
                  .map((s) => (
                    <div className="schedule-row" key={s.id}>
                      <span className="item-icon">
                        {s.recurrence ? (
                          <Repeat2 size={19} />
                        ) : (
                          <Clock3 size={19} />
                        )}
                      </span>
                      <div className="grow">
                        <strong>{s.title}</strong>
                        <span>
                          {s.next_run_at &&
                            timeLabel(
                              s.next_run_at,
                              boot.preferences.timezone,
                            )}{" "}
                          · {recurrenceLabel(s.recurrence)}
                          {s.kind === "recurring_task"
                            ? " · New task each time"
                            : ""}
                        </span>
                      </div>
                      <button
                        className="icon-button"
                        aria-label={"Cancel " + s.title}
                        disabled={busy}
                        onClick={() =>
                          void mutate(
                            "schedule.cancel",
                            {
                              schedule_id: s.id,
                              expected_revision: s.revision,
                            },
                            "Reminder cancelled",
                          )
                        }
                      >
                        <X size={17} />
                      </button>
                    </div>
                  ))}
                {!schedules.some((s) => s.status === "active") && (
                  <div className="empty-state">
                    <Clock3 size={30} />
                    <h3>Let Eridani keep the time.</h3>
                    <p>
                      Set a one-time reminder or a repeating routine. It stays
                      saved when you close the app.
                    </p>
                  </div>
                )}
                <p className="footnote">
                  {boot.capabilities.worker
                    ? "Scheduler is running."
                    : "Scheduler is reconnecting. Saved reminders will catch up when it returns."}{" "}
                  All times shown in {boot.preferences.timezone}.
                </p>
              </>
            )}
            {view === "notifications" && (
              <>
                <div className="section-head">
                  <h2>
                    Your Inbox<span>{notices.length}</span>
                  </h2>
                  <button
                    className="text-button"
                    onClick={() => void enablePush()}
                  >
                    <Bell size={15} />
                    Enable notifications
                  </button>
                </div>
                {notices.map((n) => (
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
                            "notification.read",
                            { notification_id: n.id },
                            "Marked as read",
                          )
                        }
                      >
                        <Check size={14} />
                        Read
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
                {!notices.length && (
                  <div className="empty-state">
                    <Bell size={30} />
                    <h3>You're all caught up.</h3>
                    <p>Reminders appear here when they are due.</p>
                  </div>
                )}
              </>
            )}
            {view === "memory" && (
              <>
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
                {legacy.length > 0 && (
                  <>
                    <div className="section-head">
                      <h2>From earlier conversations</h2>
                    </div>
                    {legacy.map((m) => (
                      <article className="memory legacy" key={m.id}>
                        <p>{m.content}</p>
                        <footer>
                          <span>
                            Legacy memory · Original transcript unavailable
                          </span>
                        </footer>
                      </article>
                    ))}
                  </>
                )}
                {!memories.length && !legacy.length && (
                  <div className="empty-state">
                    <Brain size={30} />
                    <h3>
                      {query
                        ? "No matching memories yet."
                        : "Keep the useful little things."}
                    </h3>
                    <p>
                      Tell Eridani “remember this,” or save something above.
                      Search also checks your available legacy memories.
                    </p>
                  </div>
                )}
              </>
            )}
            {view === "settings" && (
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
            )}
            <footer className="page-footer">
              <Shield size={13} />
              <span>Saved at home. Available across your devices.</span>
            </footer>
          </main>
          <aside
            className={"companion " + (companion ? "visible" : "")}
            aria-label="Eridani conversation"
          >
            <div className="companion-header">
              <span className="companion-logo">
                <Sparkles size={18} />
              </span>
              <div>
                <strong>Eridani</strong>
                <span>
                  {thinking
                    ? "Working on it…"
                    : voiceState && !voiceState.closed
                      ? voiceState.state
                      : "Here when you need me"}
                </span>
              </div>
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
                <div className="companion-welcome">
                  <div className="welcome-symbol">
                    <Sparkles size={25} />
                  </div>
                  <h3>Hey, {boot.name}.</h3>
                  <p>What are we getting out of your head today?</p>
                  <div className="suggestions">
                    {[
                      "What should I focus on today?",
                      "Remind me tomorrow at 10 AM to plan my day",
                      "Remember that I prefer short, useful answers",
                    ].map((s) => (
                      <button key={s} onClick={() => setChatText(s)}>
                        {s}
                        <ArrowUp size={13} />
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {messages.map((m) => (
                <div className={"message " + m.role} key={m.id}>
                  {m.role === "assistant" && (
                    <span className="message-label">ERIDANI</span>
                  )}
                  <p>{m.content}</p>
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
                  {voiceState.state === "waiting"
                    ? "Take your time. I’m listening."
                    : voiceState.state}
                </span>
                <button
                  className="icon-button"
                  aria-label="Stop speaking"
                  onClick={() => void voice.current?.interrupt()}
                >
                  <VolumeX size={17} />
                </button>
                <button
                  className="text-button"
                  onClick={() => void voice.current?.submit()}
                >
                  Submit
                </button>
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
                  disabled={!boot.capabilities.voice || thinking}
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
                    !chatText.trim() || thinking || !boot.capabilities.chat
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
      {toast && (
        <div className="toast" role="status">
          <Check size={16} />
          {toast}
        </div>
      )}
      {selected && (
        <TaskDialog
          task={selected}
          busy={busy}
          onClose={() => setSelected(null)}
          onSave={async (args) => {
            const result = await mutate<Task>(
              "task.update",
              args,
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
