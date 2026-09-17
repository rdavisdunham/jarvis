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
self.addEventListener('notificationclick', event => {
  event.notification.close();
  event.waitUntil(self.clients.matchAll({type:'window', includeUncontrolled:true}).then(async clients => {
    for (const client of clients) { if ('focus' in client) { await client.focus(); client.postMessage({type:'open-notification', url:event.notification.data?.url || '/?view=notifications'}); return; } }
    await self.clients.openWindow(event.notification.data?.url || '/?view=notifications');
  }));
});