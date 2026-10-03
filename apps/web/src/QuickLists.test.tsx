import {describe,it,expect} from "vitest";
import {parseChecklist} from "./QuickLists";
import {renderToStaticMarkup} from "react-dom/server";
import {BrandMark} from "./ux";
describe("Quick capture",()=>{
 it("keeps sections as display labels without inventing items",()=>{expect(parseChecklist("# Hayes\nFood\n\n# Work\nLaptop")).toEqual([{title:"Food",section:"Hayes"},{title:"Laptop",section:"Work"}]);});
 it("accepts plain lines and pasted empty checkboxes",()=>{expect(parseChecklist("- [ ] Keys\n* Wallet\nPassport").map(i=>i.title)).toEqual(["Keys","Wallet","Passport"]);});
 it("does not infer completion from user text",()=>{expect(parseChecklist("- [x] Check with Alex")[0].title).toBe("[x] Check with Alex");});
 it("renders an accessible vector mark",()=>{const html=renderToStaticMarkup(<BrandMark title="Eridani"/>);expect(html).toContain('aria-label="Eridani"');expect(html).toContain('viewBox="0 0 48 48"');});
});
