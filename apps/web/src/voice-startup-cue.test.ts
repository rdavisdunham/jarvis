import { expect, test, vi } from "vitest";
import { StartupCue } from "./voice-startup-cue";
function setup(){const c={send:vi.fn(),chime:vi.fn(),interrupt:vi.fn(),gate:vi.fn()};return{controls:c,cue:new StartupCue(true,1,c)};}
test.each(["Dani", "Danny", "Danni", "Donny", "Donnie"])("split %s wake requests one startup cue", name => {
 const {controls,cue}=setup();cue.sessionReady();cue.input("Hey "+name.slice(0,2));
 cue.activity(false,1,false,true,0);cue.tick(1000);expect(controls.send).not.toHaveBeenCalled();
 cue.input(name.slice(2));cue.tick(1200);expect(controls.send).toHaveBeenCalledExactlyOnceWith("ready",1);
});
test.each(["Eri", "Erie", "Airy", "Eridani", "Hey Eri"])("retired wake %s cannot request a greeting", text => {
 const {controls,cue}=setup();cue.sessionReady();cue.input(text);cue.activity(false,1,false,true,0);cue.tick(1200);
 expect(controls.send).not.toHaveBeenCalled();expect(cue.eligible).toBe(false);
});
test("only delivered wake plus readiness and 750ms quiet requests one cue; audible output prevents fallback",()=>{
 const {controls,cue}=setup();cue.input("Hey Dani");cue.activity(false,1,false,true,0);cue.tick(1000);expect(controls.send).not.toHaveBeenCalled();
 cue.sessionReady();cue.tick(749);expect(controls.send).not.toHaveBeenCalled();cue.tick(750);cue.tick(900);
 expect(controls.send).toHaveBeenCalledExactlyOnceWith("ready",1);cue.output(true,1000);cue.tick(4000);expect(controls.chime).not.toHaveBeenCalled();
});
test("continuous or buffered request, resumed speech and uncertainty suppress cues",()=>{
 for(const text of ["Hey Dani add milk","Hey","Dani, I want to"]){const {controls,cue}=setup();cue.sessionReady();cue.input(text);cue.activity(false,1,false,true,0);cue.tick(5000);expect(controls.send).not.toHaveBeenCalled();}
 const {controls,cue}=setup();cue.sessionReady();cue.input("Dani");cue.activity(true,2,false,true,0);cue.activity(false,2,false,true,200);cue.tick(5000);expect(controls.send).not.toHaveBeenCalled();
});
test("silent Live fallback chimes once, cancels intent and gates delayed output until quiet boundary",()=>{
 const {controls,cue}=setup();cue.sessionReady();cue.input("Dani");cue.activity(false,1,false,true,0);cue.tick(750);cue.tick(2749);expect(controls.chime).not.toHaveBeenCalled();
 cue.tick(2750);cue.tick(3000);expect(controls.chime).toHaveBeenCalledOnce();expect(controls.send).toHaveBeenLastCalledWith("cancel",1);
 expect(controls.gate).toHaveBeenCalledWith(true);cue.output(true,3000);cue.tick(3200);expect(controls.gate).toHaveBeenLastCalledWith(true);cue.tick(3800);expect(controls.gate).toHaveBeenLastCalledWith(false);
});
test("speech during pending greeting interrupts it without leaving a normal playout gate",()=>{
 const {controls,cue}=setup();cue.sessionReady();cue.input("Dani");cue.activity(false,1,false,true,0);cue.tick(750);
 cue.activity(true,2,false,true,900);expect(controls.interrupt).toHaveBeenCalledOnce();cue.activity(false,2,false,true,1200);cue.tick(2000);expect(controls.gate).toHaveBeenLastCalledWith(false);expect(controls.chime).not.toHaveBeenCalled();
});

test("split raw wake fragments wait for positive recognition rather than cancelling the cue",()=>{
 const {controls,cue}=setup();cue.sessionReady();cue.input("Hey");cue.activity(false,1,false,true,0);cue.tick(1000);expect(controls.send).not.toHaveBeenCalled();
 cue.input(" Dani");cue.tick(1200);expect(controls.send).toHaveBeenCalledWith("ready",1);
});
test("a completed audible greeting disarms startup control before normal full-duplex speech",()=>{
 const {controls,cue}=setup();cue.sessionReady();cue.input("Dani");cue.activity(false,1,false,true,0);cue.tick(750);cue.output(true,1000);cue.tick(1800);
 expect(controls.gate).toHaveBeenLastCalledWith(false);cue.activity(true,2,false,true,1900);expect(controls.gate).not.toHaveBeenCalledWith(true);
});
