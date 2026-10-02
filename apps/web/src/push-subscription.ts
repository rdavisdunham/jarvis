import { post } from "./api";

// Push services rotate or expire subscriptions (410). Re-register the current one on every boot,
// and whenever the service worker reports a change, so a device never silently goes dark.
export function vapidKey(base64url: string) {
  const raw = atob(base64url.replace(/-/g, "+").replace(/_/g, "/"));
  return Uint8Array.from(raw, (c) => c.charCodeAt(0));
}

export async function ensurePushSubscription(vapidPublicKey?: string | null) {
  if (
    typeof Notification === "undefined" ||
    typeof navigator === "undefined" ||
    !navigator.serviceWorker ||
    Notification.permission !== "granted"
  )
    return false;
  try {
    const registration = await navigator.serviceWorker.ready;
    if (!registration.pushManager) return false;
    const subscription =
      (await registration.pushManager.getSubscription()) ??
      (vapidPublicKey
        ? await registration.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey: vapidKey(vapidPublicKey),
          })
        : null);
    if (!subscription) return false;
    await post("/push", subscription.toJSON());
    return true;
  } catch {
    return false;
  }
}

// The worker cannot hold the session's CSRF token, so it asks open pages to register for it.
export function watchPushSubscription(vapidPublicKey?: string | null) {
  if (typeof navigator === "undefined" || !navigator.serviceWorker) return () => {};
  const handler = (event: MessageEvent) => {
    if (event.data?.type === "push-subscription-changed")
      void ensurePushSubscription(vapidPublicKey);
  };
  navigator.serviceWorker.addEventListener("message", handler);
  return () => navigator.serviceWorker.removeEventListener("message", handler);
}
