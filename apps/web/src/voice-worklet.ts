import { CaptureProcessor } from "./voice-audio";
declare const sampleRate: number;
declare class AudioWorkletProcessor { port: MessagePort; }
declare function registerProcessor(name: string, processor: typeof AudioWorkletProcessor): void;
class VoiceCaptureWorklet extends AudioWorkletProcessor {
  private core = new CaptureProcessor(sampleRate, message => this.port.postMessage(message));
  constructor() {
    super();
    this.port.onmessage = event => this.core.command(event.data);
  }
  process(inputs: Float32Array[][], outputs: Float32Array[][]) {
    const output = outputs[0]?.[0];
    if (output) this.core.process(inputs[0]?.[0] ?? new Float32Array(output.length), output);
    return true;
  }
}
registerProcessor("eri-voice-capture", VoiceCaptureWorklet);
