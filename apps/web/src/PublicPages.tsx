import { ArrowUpRight, CalendarDays, Check, FileText, ListTodo, Mic, Sparkles } from "lucide-react";

const apex = ["eridani.app", "www.eridani.app"].includes(location.hostname);
export const appLink = apex ? "https://app.eridani.app/" : "/";
export function PublicFooter() {
  return <footer className="public-footer"><a href="/welcome">Eridani</a><nav aria-label="Information">
    <a href="/privacy">Privacy</a><a href="/terms">Terms</a><a href="/support">Help & access</a></nav></footer>;
}
export function PublicPages() {
  const path = location.pathname;
  const policy = path === "/privacy" || path === "/terms" || path === "/support";
  return <div className="public-site">
    <header className="public-nav"><a className="public-brand" href="/welcome"><img src="/icon.svg" alt=""/>eridani</a>
      <a className="public-signin" href={appLink}>Sign in <ArrowUpRight size={15}/></a></header>
    {policy ? <article className="public-document">
      <a className="public-back" href="/welcome">← About Eridani</a>
      {path === "/privacy" ? <>
        <p className="eyebrow">YOUR INFORMATION</p><h1>Privacy</h1><p className="public-lead">What Eridani stores, what the assistant uses, and what you control.</p>
        <p>Updated September 16, 2026. Eridani is an invitation-only planner operated by its app owner.</p>
        <h2>Your account and records</h2><p>Google sign-in provides your verified email and account identifier. Eridani stores the tasks, projects, goals, notes, reminders, settings, and workspace memberships you create. Personal records are separate from shared workspace records. Members of a shared workspace can see records in that workspace according to their role; assignment alone does not grant access.</p>
        <h2>Conversations and AI</h2><p>When you use Eri, your request and relevant records, screen context, or saved memories may be sent to the configured AI provider. Live voice is processed by OpenAI. The task agent may use OpenAI, Google, or another configured provider shown in Settings. Provider terms and retention rules also apply. Voice transcription is displayed in the app; Eridani does not keep an audio-recording archive.</p>
        <p>Conversation history and automatic memory learning are separate controls in Settings → Privacy. History follows your configured retention period; zero means no automatic age-based deletion. Shared-workspace conversations do not become personal memories. A background request can temporarily retain encrypted input and recovery context for up to 24 hours, even with history off, so accepted work can finish. History-off input is removed when the request finishes; durable saved actions and their change records remain.</p>
        <h2>Connected accounts</h2><p>Google Calendar and Linear are optional. Eridani imports selected calendar or issue data to display it and perform actions you request. Integration credentials are stored encrypted. Disconnect an integration in Settings, and revoke its access with the provider if desired. Disconnecting Calendar is separate from removing your Google sign-in.</p>
        <h2>Hosting and backups</h2><p>Eridani uses Railway for application and database hosting and Cloudflare for domain services and, when configured, independent backups. Authorized service operators can administer stored data. Backups may retain earlier copies until their retention period ends; deleting a current record does not immediately erase historical backups.</p>
        <h2>Browser storage and microphone</h2><p>A secure session cookie keeps you signed in. Browser storage remembers interface preferences and the current conversation reference. Voice requires microphone permission. Optional wake recognition runs while the page is visible and may use the browser vendor’s recognition service. Ending voice releases the microphone capture used by that session; enabled wake listening can then resume.</p>
        <h2>Your controls and requests</h2><p>You can edit or remove records, review learned memories, change history and learning settings, export your data, and disconnect integrations. Contact the person who invited you or the app owner to request account deletion, access help, or clarification about this policy. <a href="/support">See help and access.</a></p>
      </> : path === "/terms" ? <>
        <p className="eyebrow">USING ERIDANI</p><h1>Terms of use</h1><p className="public-lead">A shared understanding for this early-access planner.</p>
        <p>Updated September 16, 2026. Use Eridani only with an account or invitation issued to you. Keep your sign-in secure and do not try to access another person’s private records or bypass workspace permissions.</p>
        <h2>Your content and connections</h2><p>You keep ownership of your content. By using the service, you allow its operator and configured providers to process the information needed to store, synchronize, and act on your requests. Connect only accounts and calendars you are authorized to use. Shared-workspace members receive access to the records placed there.</p>
        <h2>Assistant actions</h2><p>Eri can create and change records, but its interpretation may be mistaken. Review action cards and important dates. Cancel stops unfinished work; it does not automatically reverse saved changes. Revert is available only where Eridani can safely perform the stated reversal. Changes in Google or Linear may need to be corrected through those services.</p>
        <h2>Early access</h2><p>Features may change, and availability or successful notification delivery is not guaranteed. Keep a separate copy of information you cannot afford to lose. The operator may suspend access that misuses the service or threatens other users’ data.</p>
        <h2>Questions or leaving</h2><p>Use Settings to export your data and disconnect integrations. Contact the app owner through the person who invited you for account closure or questions. The <a href="/privacy">privacy page</a> describes information handling and retention.</p>
      </> : <>
        <p className="eyebrow">A LITTLE DIRECTION</p><h1>Help & access</h1><p className="public-lead">Get in, get oriented, or pick up where you left off.</p>
        <h2 id="access">Getting an invitation</h2><p>Eridani is currently invitation-only. Ask the person who shared it with you to invite your Google email. Invitations are copied and shared manually; Eridani does not send an invitation email. Sign in with the exact invited account, then review the invitation in Settings → Sharing.</p>
        <h2>Wrong account or expired invitation</h2><p>Choose Sign in with Google again and select the invited address. If the invitation expired or was revoked, ask its sender for a new invitation. A shared-workspace invitation does not expose the sender’s personal tasks, memories, or connected accounts.</p>
        <h2>Where did Eri’s work go?</h2><p>Open Activity in the top bar. Queued and working requests continue after voice ends. Waiting for you means Eri needs a clarification; a failed or partly completed request keeps its saved changes visible. Use Revise or Continue to move it forward.</p>
        <h2>Voice and wake words</h2><p>Allow microphone access, then tap Talk to Eridani. Voice ends after 30 quiet seconds or a clear goodbye. Enable wake listening in Settings → Voice to use “Eri” or “Hey Eri” while the page is visible. Locked-screen and background phone wake are not supported by the web app.</p>
        <h2>Calendar and synchronization</h2><p>Check Settings → Integrations for the connected account, selected calendars, permissions, and sync status. Local tasks stay in Eridani unless you explicitly publish or synchronize them. An external action is not complete until its provider confirms it.</p>
        <h2>Still need help?</h2><p>Contact the person who invited you or the app owner. Include the page, what you expected, and the visible error or Activity status. Do not send API keys, passwords, or private invitation details in a public post.</p>
      </>}
    </article> : <main>
      <section className="public-hero">
        <div className="public-hero-copy"><p className="eyebrow"><span/> A LITTLE MORE SPACE TO THINK</p>
          <h1>Your plans.<br/>In good company.</h1>
          <p className="public-lead">Tasks, time, and notes in one place. An assistant who helps you turn a passing thought into something you can follow through on.</p>
          <div className="public-cta"><a className="public-primary" href={appLink}>Meet your day <ArrowUpRight size={18}/></a><span>Early access · By invitation</span></div>
          <a className="public-access" href="/support#access">Have an invitation? Sign in with your Google account.</a>
        </div>
        <div className="public-orbit" aria-hidden="true"><div className="orbit-line"/><div className="orbit-line second"/><div className="orbit-star"/>
          <div className="public-preview"><div className="preview-heading"><span>AN EXAMPLE DAY</span><span>Today</span></div>
            <h2>A little less to keep in your head.</h2>
            <div className="preview-task"><span className="preview-check"><Check size={13}/></span><div>Finish the project brief<small>Work · Planned today</small></div><span>10:00</span></div>
            <div className="preview-task"><span className="preview-check empty"/><div>Leave room for a walk<small>Personal · 30 minutes</small></div><span>16:30</span></div>
            <div className="preview-note"><FileText size={15}/><span>The idea that started it all<small>Linked to the project</small></span></div>
            <div className="preview-eri"><Sparkles size={17}/><span>“Move the brief to tomorrow.”<small>A request. A saved change. Room to think.</small></span><Mic size={17}/></div>
          </div>
        </div>
      </section>
      <section className="public-features" aria-label="What you can do">
        <article><ListTodo size={22}/><h2>See the work clearly.</h2><p>A quick list for today, a board for the project, a timeline for what’s ahead. One set of tasks, viewed your way.</p></article>
        <article><CalendarDays size={22}/><h2>Give your plans time.</h2><p>Bring selected Google calendars alongside tasks, reminders, and work blocks. Keep appointments and intentions connected.</p></article>
        <article><FileText size={22}/><h2>Keep the thread.</h2><p>Connect notes to projects and tasks. Let Eri help find the detail, make the change, and show what happened.</p></article>
      </section>
      <section className="public-closing"><p className="eyebrow">MEET ERI</p><h2>Say it while it’s on your mind.</h2><p>Talk naturally or type a request. Keep going while Eri works.<br/>Every accepted request has a place in Activity.</p><a href={appLink}>Open Eridani <ArrowUpRight size={16}/></a></section>
    </main>}
    <PublicFooter/>
  </div>;
}
