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
import { subscribeEvents } from "./events";
import { ReminderList } from "./ReminderList";
import { WakeWord, recognitionType } from "./wake-word";
import type {
  Bootstrap,
  UIAction,
  VoiceProvider,
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
  const pendingNotices = notices.filter((notice) => !notice.completed_at);
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
  const [syncWarning, setSyncWarning] = useState("");
  const [voiceState, setVoiceState] = useState<VoiceState | null>(null);
  const [voiceProvider, setVoiceProvider] = useState<VoiceProvider>(() =>
    localStorage.getItem("eri-voice-provider") === "live" ? "live" : "realtime",
  );
  const [voiceName, setVoiceName] = useState(
    () =>
      localStorage.getItem(
        "eri-voice-" +
          (localStorage.getItem("eri-voice-provider") === "live"
            ? "live"
            : "realtime"),
      ) || "marin",
  );
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
    const stopEvents = subscribeEvents(
      boot.event_cursor,
      () => {
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
  useEffect(() => {
    if (!boot?.voice_options) return;
    const options = boot.voice_options[voiceProvider];
    if (!options.voices.includes(voiceName)) {
      setVoiceName(options.default_voice);
      localStorage.setItem("eri-voice-" + voiceProvider, options.default_voice);
    }
  }, [boot?.voice_options, voiceProvider, voiceName]);
  async function showActions(actions: UIAction[] = []) {
    for (const action of actions) {
      if (displayedActions.current.has(action.id)) continue;
      displayedActions.current.add(action.id);
      setQuery("");
      setView(action.view);
      setSidebar(false);
      setCompanion(false);
      try {
        if (action.entity_id && action.view === "all") {
          setSelected(
            await api<Task>("/tasks/" + encodeURIComponent(action.entity_id)),
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
      } catch (e) {
        setError((e as Error).message);
      }
    }
  }
  useEffect(() => {
    if (!highlight || view !== "reminders") return;
    const timer = setTimeout(() => {
      const target = document.getElementById("record-" + highlight);
      target?.scrollIntoView({ behavior: "smooth", block: "center" });
      target?.focus({ preventScroll: true });
    }, 100);
    const clear = setTimeout(() => setHighlight(null), 12000);
    return () => {
      clearTimeout(timer);
      clearTimeout(clear);
    };
  }, [highlight, view, schedules]);
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
    setVoiceProvider(provider);
    localStorage.setItem("eri-voice-provider", provider);
    const saved = localStorage.getItem("eri-voice-" + provider);
    setVoiceName(
      saved && boot?.voice_options[provider].voices.includes(saved)
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
                <ReminderList
                  schedules={schedules}
                  notices={notices}
                  zone={boot.preferences.timezone}
                  busy={busy}
                  highlight={highlight}
                  create={() => setReminder(true)}
                  mutate={mutate}
                />
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
              <>
                <section className="voice-settings settings-sections">
                  <h2>Voice & conversation</h2>
                  <p>
                    Choose how Eri sounds on this device. Changes apply to the
                    next voice session.
                  </p>
                  <div className="voice-options">
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
                        <option value="realtime">Realtime</option>
                        <option value="live">GPT-Live</option>
                      </select>
                    </label>
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
                    Task agent: {boot.agent_model || "gpt-5.4-mini"}. Both voice
                    modes use the same saved tasks, reminders, and tools.
                  </p>
                  {voiceState && !voiceState.closed && (
                    <p className="voice-model-note">
                      End the active voice session before changing its model or
                      voice.
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
                      “Hey, Eri”
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
                    <span className="transcript-status">Transcribing…</span>
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
                <span>{voiceLabel(voiceState.state)}</span>
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
            <div className="voice-glow" ref={glowRef} aria-hidden="true">
              <i />
              <i />
              <i />
            </div>
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
      } as Record<string, string>
    )[state] ?? "Voice is on"
  );
}
