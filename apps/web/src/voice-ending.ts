import type { ChatMessage } from "./types";

type ClosingQuestion = "more" | "done";

function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[’']/g, "")
    .replace(/[^a-z\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^(?:eri|eridani)\s+/, "")
    .replace(/\s+(?:eri|eridani)$/, "");
}

// Complete standalone endings only; never close on quoted words or an added request.
export function isVoiceEnding(text: string): boolean {
  const value = normalize(text)
    .replace(/^(?:(?:okay|ok|alright|yes|yeah|yep|no|nope|please)\s+)+/, "")
    .replace(/\s+(?:please|thanks|thank you)(?:\s+(?:very much|so much))?$/, "");
  return /^(?:goodbye|good bye|bye|bye bye|goodnight|good night|see you later|talk to you later|thank you|thanks|thank you very much|thanks so much|thanks for (?:your|the) help|(?:i think )?(?:thats|that is) (?:all|everything|it)(?: for (?:now|today|me))?|that(?:ll| will| would) (?:be all|do)(?: for (?:now|today))?|(?:im|i am|were|we are) (?:done|finished)(?: (?:talking|chatting|with (?:this|the|our) (?:conversation|chat)))?(?: for (?:now|today))?|(?:im|i am|were|we are) (?:all set|good)(?: for (?:now|today))|i dont need anything else|nothing else(?: for (?:now|today))?|(?:you can |lets )?(?:end|stop|close)(?: (?:the|this|our))? (?:conversation|voice chat|voice session|voice mode)|(?:you can )?stop listening(?: now)?)$/.test(value);
}

export function closingQuestion(text: string): ClosingQuestion | null {
  // Only a single, final, direct question. Quoted examples and compound questions
  // must not make an ordinary "yes" permission to hang up.
  if ((text.match(/\?/g) ?? []).length !== 1) return null;
  const tail = text.match(/(?:^|[.!]\s+)([^.!?]*\?)\s*$/)?.[1];
  if (!tail) return null;
  const value = normalize(tail);
  if (
    /^(?:anything else(?: (?:i can|can i) help(?: you)? with)?|is there anything else(?: (?:i can|you need|i can help(?: you)? with))?|do you need anything else|would you like anything else|can i help(?: you)? with anything else)$/.test(
      value,
    )
  )
    return "more";
  if (
    /^(?:will that be all|is that (?:all|everything)|are (?:we|you) (?:all )?(?:done|finished)|shall we wrap up)(?: for (?:now|today))?$/.test(
      value,
    )
  )
    return "done";
  return null;
}

function confirmsEnding(
  text: string,
  question: ClosingQuestion | null,
): boolean {
  const value = normalize(text);
  // Politeness does not reverse a yes/no answer: "yes, thank you" to an
  // offer of more help keeps listening rather than treating thanks as goodbye.
  if (
    question &&
    /^(?:yes|yeah|yep|yup)(?: please)?(?: (?:thanks|thank you))?$/.test(value)
  )
    return question === "done";
  if (question && /^(?:no|nope|nah)(?: (?:thanks|thank you))?$/.test(value))
    return question === "more";
  if (isVoiceEnding(text)) return true;
  if (!question) return false;
  if (
    /^(?:(?:yes|yeah|yep|no|nope|okay|ok) )?(?:nothing else|not (?:right )?now|(?:im|i am|were|we are) (?:all set|good)(?: for (?:now|today))?)(?: (?:thanks|thank you))?$/.test(
      value,
    )
  )
    return true;
  if (question === "more")
    return /^(?:no|nope|nah|no thank you|no thanks)(?: (?:im|i am) good)?$/.test(
      value,
    );
  return /^(?:yes|yeah|yep|yup|yes please|absolutely|correct|thats right|that will be all|that would be all)(?: (?:thanks|thank you))?$/.test(
    value,
  );
}

type Offer = {
  kind: ClosingQuestion;
  at: number;
  userId: string;
  userText: string;
};

// Session-local context, never restored from old chat history. Capture the question
// when the next user utterance begins so early assistant replies cannot change it.
export class VoiceEnding {
  private offer: Offer | null = null;
  private answer: {
    id: string;
    prefix: string;
    kind: ClosingQuestion | null;
  } | null = null;
  private lastUser = { id: "", content: "" };
  private lastUserFinal = false;
  private lastAssistant = { id: "", content: "" };
  private assistantBoundary = { id: "", content: "" };
  private assistantSinceUser = false;

  assistant(id: string, content: string, now = Date.now()) {
    if (id === this.lastAssistant.id && content === this.lastAssistant.content)
      return;
    this.lastAssistant = { id, content };
    this.assistantSinceUser = true;
    // Live can group several exchanges in one caption bubble. An earlier
    // question in that bubble must not hide the current closing question.
    const spoken =
      id === this.assistantBoundary.id &&
      content.startsWith(this.assistantBoundary.content)
        ? content.slice(this.assistantBoundary.content.length).trim()
        : content;
    const kind = closingQuestion(spoken);
    this.offer = kind
      ? {
          kind,
          at: now,
          userId: this.lastUser.id,
          userText: this.lastUser.content,
        }
      : null;
  }

  user(message: ChatMessage, now = Date.now()): boolean {
    if (!message.content.trim()) return false;
    const { id, content } = message;
    if (
      this.assistantSinceUser &&
      (id !== this.lastUser.id || content !== this.lastUser.content)
    ) {
      this.assistantBoundary = { ...this.lastAssistant };
      // A new spoken reply can extend the same display bubble. Strip the completed
      // prior utterance even when the assistant did not ask a closing question.
      if (this.lastUserFinal && id === this.lastUser.id && content.startsWith(this.lastUser.content)) {
        this.answer = { id, prefix: this.lastUser.content, kind: null };
      }
      this.assistantSinceUser = false;
    }
    if (
      this.offer &&
      (id !== this.offer.userId || content !== this.offer.userText)
    ) {
      const prefix =
        id === this.offer.userId && content.startsWith(this.offer.userText)
          ? this.offer.userText
          : "";
      this.answer = {
        id,
        prefix,
        kind: now - this.offer.at <= 60_000 ? this.offer.kind : null,
      };
      this.offer = null;
    } else if (id !== this.answer?.id) {
      this.answer = { id, prefix: "", kind: null };
    }
    this.lastUser = { id, content };
    this.lastUserFinal = !message.pending;
    if (message.pending) return false;
    const reply = content.startsWith(this.answer?.prefix ?? "")
      ? content.slice(this.answer?.prefix.length ?? 0)
      : content;
    return confirmsEnding(reply, this.answer?.kind ?? null);
  }
}
