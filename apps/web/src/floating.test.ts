import { describe, expect, it } from "vitest";
import { GAP, GUTTER, place } from "./floating";

const anchor = (top: number, left: number, height = 30, width = 120) => ({ top, left, bottom: top + height, right: left + width });
const desk = { width: 1440, height: 900 };

describe("floating panel placement", () => {
  it("hangs below its anchor when there is room", () => {
    expect(place(anchor(100, 200), { width: 300, height: 400 }, desk)).toEqual({ mode: "float", top: 130 + GAP, left: 200, maxHeight: 900 - 130 - GAP - GUTTER, above: false });
  });
  it("flips above when the room below is short and above is larger", () => {
    const p = place(anchor(800, 200), { width: 300, height: 400 }, desk);
    expect(p).toMatchObject({ mode: "float", above: true, top: 800 - GAP - 400 });
  });
  it("stays below, scrolling, when neither side fits but below is larger", () => {
    const p = place(anchor(300, 200), { width: 300, height: 900 }, desk);
    expect(p).toMatchObject({ above: false, top: 330 + GAP, maxHeight: 900 - 330 - GAP - GUTTER });
  });
  it("is clamped to the viewport's right and left gutters", () => {
    expect(place(anchor(100, 1300), { width: 380, height: 200 }, desk)).toMatchObject({ left: 1440 - GUTTER - 380 });
    expect(place(anchor(100, 2), { width: 380, height: 200 }, desk)).toMatchObject({ left: GUTTER });
    expect(place(anchor(100, 500), { width: 2000, height: 200 }, desk)).toMatchObject({ left: GUTTER });
  });
  it("becomes a bottom sheet on phone widths", () => {
    expect(place(anchor(100, 20), { width: 300, height: 200 }, { width: 760, height: 800 })).toEqual({ mode: "sheet" });
    expect(place(anchor(100, 20), { width: 300, height: 200 }, { width: 761, height: 800 }).mode).toBe("float");
  });
});
