self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
// No API, transcript or task caching: private data never becomes a second browser archive.
self.addEventListener('push', event => {
  let data = {title: 'Eridani reminder', body: 'A reminder is waiting in your Inbox.', tag: 'jarvis'};
  try { data = {...data, ...event.data.json()}; } catch (_) {}
  event.waitUntil(self.registration.showNotification(data.title, {
    body: data.body, tag: data.tag, icon: '/icon.svg', badge: '/icon.svg', data: {id: data.id, url: typeof data.url === "string" && data.url.startsWith("/?") ? data.url : "/?view=notifications"},
  }));
});
// The browser rotated or expired the subscription. Resubscribe with the same server key, then have
// open pages register it: the API needs the session's CSRF token, which this worker never holds.
// Closed devices re-register on their next boot (ensurePushSubscription).
self.addEventListener('pushsubscriptionchange', event => {
  event.waitUntil((async () => {
    const key = event.oldSubscription?.options?.applicationServerKey;
    if (!event.newSubscription && key) {
      try { await self.registration.pushManager.subscribe({userVisibleOnly: true, applicationServerKey: key}); } catch (_) {}
    }
    const clients = await self.clients.matchAll({type:'window', includeUncontrolled:true});
    for (const client of clients) client.postMessage({type:'push-subscription-changed'});
  })());
});
self.addEventListener('notificationclick', event => {
  event.notification.close();
  event.waitUntil(self.clients.matchAll({type:'window', includeUncontrolled:true}).then(async clients => {
    for (const client of clients) { if ('focus' in client) { await client.focus(); client.postMessage({type:'open-notification', url:event.notification.data?.url || '/?view=notifications'}); return; } }
    await self.clients.openWindow(event.notification.data?.url || '/?view=notifications');
  }));
});