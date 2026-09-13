import {
  useEffect,
  useRef,
  useState,
  type PointerEvent,
  type KeyboardEvent,
} from "react";
type Item = { id: string; title: string };
type Drag = {
  id: string;
  title: string;
  x: number;
  y: number;
  over: string | null;
  keyboard: boolean;
};
export function useBoardDrag<T extends Item>({
  items,
  keys,
  disabled,
  move,
}: {
  items: T[];
  keys: string[];
  disabled: boolean;
  move: (item: T, key: string) => Promise<unknown>;
}) {
  const root = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const current = useRef<Drag | null>(null);
  const pending = useRef<{ id: string; x: number; y: number } | null>(null);
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);
  const latest = useRef({ items, keys, disabled, move });
  latest.current = { items, keys, disabled, move };
  const set = (value: Drag | null) => {
    current.current = value;
    setDrag(value);
  };
  const hit = (x: number, y: number) => {
    const element = document
      .elementFromPoint(x, y)
      ?.closest<HTMLElement>("[data-board-key]");
    return element && root.current?.contains(element)
      ? (element.dataset.boardKey ?? null)
      : null;
  };
  const cancel = () => {
    pending.current = null;
    set(null);
  };
  const finish = async () => {
    const value = current.current;
    cancel();
    const item = latest.current.items.find((t) => t.id === value?.id);
    if (!value || !item || value.over === null || latest.current.disabled)
      return;
    setSaving(true);
    setMessage("Saving move for " + item.title + ".");
    try {
      const saved = await latest.current.move(item, value.over);
      setMessage(
        saved
          ? "Moved " + item.title + "."
          : "Move was not saved. Check the error and try again.",
      );
    } catch {
      setMessage("Move was not saved. Check the error and try again.");
    } finally {
      setSaving(false);
    }
  };
  useEffect(() => {
    if (!drag || drag.keyboard) return;
    let frame = 0;
    const tick = () => {
      const value = current.current,
        el = root.current;
      if (!value || !el) return;
      const rect = el.getBoundingClientRect(),
        edge = 42;
      const amount = (point: number, min: number, max: number) =>
        point < min + edge
          ? -Math.ceil((min + edge - point) / 4)
          : point > max - edge
            ? Math.ceil((point - max + edge) / 4)
            : 0;
      const dx = amount(value.x, rect.left, Math.min(rect.right, innerWidth));
      if (dx) el.scrollLeft += Math.max(-16, Math.min(16, dx));
      if (value.y > innerHeight - 45) window.scrollBy(0, 10);
      else if (value.y < 65) window.scrollBy(0, -10);
      const over = hit(value.x, value.y);
      if (over !== value.over) set({ ...value, over });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [!!drag, drag?.keyboard]);
  useEffect(() => {
    if (!drag) return;
    const escape = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        cancel();
        setMessage("Move cancelled.");
      }
    };
    const blur = () => cancel();
    window.addEventListener("keydown", escape);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", escape);
      window.removeEventListener("blur", blur);
    };
  }, [!!drag]);
  const handle = (item: T, column: string) => ({
    disabled: disabled || saving,
    "aria-label": "Move task " + item.title,
    "aria-describedby": "board-drag-help",
    "aria-pressed": drag?.id === item.id,
    onPointerDown: (e: PointerEvent<HTMLButtonElement>) => {
      if (e.button !== 0 || disabled || saving) return;
      e.preventDefault();
      e.stopPropagation();
      e.currentTarget.focus();
      pending.current = { id: item.id, x: e.clientX, y: e.clientY };
      e.currentTarget.setPointerCapture(e.pointerId);
    },
    onPointerMove: (e: PointerEvent<HTMLButtonElement>) => {
      const start = pending.current;
      if (!start || start.id !== item.id) return;
      if (
        !current.current &&
        Math.hypot(e.clientX - start.x, e.clientY - start.y) < 6
      )
        return;
      e.preventDefault();
      set({
        id: item.id,
        title: item.title,
        x: e.clientX,
        y: e.clientY,
        over: hit(e.clientX, e.clientY),
        keyboard: false,
      });
    },
    onPointerUp: (e: PointerEvent<HTMLButtonElement>) => {
      if (pending.current?.id !== item.id) return;
      if (e.currentTarget.hasPointerCapture(e.pointerId))
        e.currentTarget.releasePointerCapture(e.pointerId);
      void finish();
    },
    onPointerCancel: () => {
      cancel();
      setMessage("Move cancelled.");
    },
    onKeyDown: (e: KeyboardEvent<HTMLButtonElement>) => {
      if (disabled || saving) return;
      if (e.key === " " || e.key === "Enter") {
        e.preventDefault();
        if (current.current) void finish();
        else {
          set({
            id: item.id,
            title: item.title,
            x: 0,
            y: 0,
            over: column,
            keyboard: true,
          });
          setMessage(
            "Moving " +
              item.title +
              ". Use left or right, then Space to drop. Escape cancels.",
          );
        }
      } else if (
        current.current?.keyboard &&
        ["ArrowLeft", "ArrowRight"].includes(e.key)
      ) {
        e.preventDefault();
        const index = keys.indexOf(current.current.over ?? column);
        const next =
          keys[
            Math.max(
              0,
              Math.min(
                keys.length - 1,
                index + (e.key === "ArrowLeft" ? -1 : 1),
              ),
            )
          ];
        set({ ...current.current, over: next });
        root.current
          ?.querySelectorAll<HTMLElement>("[data-board-key]")
          .forEach((el) => {
            if (el.dataset.boardKey === next) {
              el.scrollIntoView({ block: "nearest", inline: "nearest" });
              setMessage("Drop in " + el.dataset.boardLabel + ".");
            }
          });
      }
    },
  });
  return { root, drag, handle, message, saving };
}
