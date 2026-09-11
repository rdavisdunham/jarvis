// Browser recognition is opt-in and foreground-only. No local model is downloaded.
export const isWakePhrase = (text: string) =>
  /^(hey|hi|okay|ok)[,\s]+(eri|erie|airy|eridani)[.!?,\s]*$/i.test(text.trim());

type Recognition = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  processLocally?: boolean;
  onresult: ((event: any) => void) | null;
  onerror: ((event: any) => void) | null;
  onend: (() => void) | null;
  start(): void;
  abort(): void;
};
export function recognitionType(): (new () => Recognition) | undefined {
  const browser = window as any;
  return browser.SpeechRecognition ?? browser.webkitSpeechRecognition;
}
export class WakeWord {
  private recognition: Recognition | null = null;
  private enabled = false;
  private timer: ReturnType<typeof setTimeout> | null = null;
  constructor(
    private wake: () => void,
    private status: (message: string) => void,
  ) {}
  start() {
    const Type = recognitionType();
    if (!Type) {
      this.status("Wake word is unavailable in this browser.");
      return;
    }
    this.enabled = true;
    const recognition = new Type();
    this.recognition = recognition;
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = "en-US";
    if ("processLocally" in recognition) recognition.processLocally = false;
    recognition.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (
          event.results[i].isFinal &&
          isWakePhrase(event.results[i][0].transcript)
        ) {
          this.stop();
          this.wake();
          break;
        }
      }
    };
    recognition.onerror = (event) => {
      if (event.error === "no-speech" || event.error === "aborted") return;
      this.stop();
      this.status(
        "Wake word paused: " +
          (event.error === "not-allowed"
            ? "allow microphone access, then enable it again."
            : "browser recognition is unavailable. Tap voice to talk."),
      );
    };
    recognition.onend = () => {
      if (this.enabled && document.visibilityState === "visible") {
        this.timer = setTimeout(() => {
          if (!this.enabled) return;
          try {
            recognition.start();
          } catch {
            this.stop();
            this.status("Enable wake word again to resume listening.");
          }
        }, 500);
      }
    };
    try {
      recognition.start();
      this.status("Listening for “Hey, Eri” while this page is visible.");
    } catch {
      this.stop();
      this.status("Wake word could not start. Tap voice to talk.");
    }
  }
  stop() {
    this.enabled = false;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    if (this.recognition) {
      this.recognition.onend = null;
      this.recognition.onresult = null;
      this.recognition.onerror = null;
      this.recognition.abort();
      this.recognition = null;
    }
  }
}
