import { expect, test } from "vitest";
import { wakeRequest } from "./wake-word";
test.each(["Eri", "Hey, Eri!", "Eridani"])("standalone wake %s", text => expect(wakeRequest(text)).toBe(""));
test.each(["Eri, add milk to my tasks", "hey Eri add milk to my tasks"])("preserves immediate request %s", text => expect(wakeRequest(text)).toBe("add milk to my tasks"));
test.each(["I told Eri yesterday", "Erin called me", "ordinary speech"])("does not wake on %s", text => expect(wakeRequest(text)).toBeNull());
import { afterEach, beforeEach, vi } from "vitest";
import { WakeWord } from "./wake-word";
class RecognitionFake {
  static current:RecognitionFake;
  continuous=false;interimResults=false;maxAlternatives=1;lang="";
  onstart:(()=>void)|null=null;onresult:((e:any)=>void)|null=null;onerror:((e:any)=>void)|null=null;onend:(()=>void)|null=null;
  constructor(){RecognitionFake.current=this;}
  start=vi.fn();abort=vi.fn();
  result(text:string, final=false, alternatives:string[]=[]){this.onresult?.({resultIndex:0,results:[Object.assign([text,...alternatives].map(transcript=>({transcript})),{isFinal:final})]});}
}
beforeEach(()=>{vi.useFakeTimers();vi.stubGlobal("window",{SpeechRecognition:RecognitionFake});});
afterEach(()=>{vi.useRealTimers();vi.unstubAllGlobals();});
test("stable interim prefix connects before request finalization; suffix continues growing",async()=>{
 const wake=vi.fn(),listener=new WakeWord(wake,vi.fn());listener.start();
 const r=RecognitionFake.current;
 expect(r.interimResults).toBe(true);expect(r.maxAlternatives).toBe(3);
 r.result("Hey Eri"); await vi.advanceTimersByTimeAsync(100);
 r.result("Hey Eri add milk");await vi.advanceTimersByTimeAsync(100);
 expect(wake).toHaveBeenCalledExactlyOnceWith("add milk");
 r.result("Hey Eri add milk",true);expect(wake).toHaveBeenCalledOnce();
});
test("interim revision invalidates candidate and alternatives can confirm a final wake",async()=>{
 const wake=vi.fn(),listener=new WakeWord(wake,vi.fn());listener.start();
 const r=RecognitionFake.current;r.result("Eri");await vi.advanceTimersByTimeAsync(100);r.result("Erin");
 await vi.advanceTimersByTimeAsync(300);expect(wake).not.toHaveBeenCalled();
 r.result("Erin add milk",true,["Eri add milk"]);expect(wake).toHaveBeenCalledExactlyOnceWith("add milk");
});
test("recognition readiness requires onstart and stale callbacks cannot reactivate a stopped listener",()=>{
 const wake=vi.fn(),started=vi.fn(),listener=new WakeWord(wake,vi.fn(),{started});listener.start();
 expect(started).not.toHaveBeenCalled();const r=RecognitionFake.current,late=r.onresult!;r.onstart?.();expect(started).toHaveBeenCalledOnce();
 listener.stop();late({resultIndex:0,results:[Object.assign([{transcript:"Eri"}],{isFinal:true})]});expect(wake).not.toHaveBeenCalled();
});
