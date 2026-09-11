// Only complete, standalone closing phrases can end the microphone session.
export function isVoiceEnding(text: string): boolean {
  let value = text.toLowerCase().replace(/[’']/g, "").replace(/[^a-z\s]/g, " ").replace(/\s+/g, " ").trim();
  value = value.replace(/^(?:eri|eridani)\s+/, "").replace(/\s+(?:eri|eridani)$/, "");
  return /^(?:(?:okay|ok|alright) )?(?:goodbye|good bye|bye|bye bye|thank you|thanks|thank you very much|thanks so much|thats all|that is all|thats all thank you|thank you thats all|end (?:the )?(?:conversation|voice chat|voice session)|stop listening)$/.test(value);
}
