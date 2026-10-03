import {describe,it,expect} from "vitest";
import {openTarget,type Schema} from "./structure-types";
const schema={types:[{id:"client",opens_as:"container"},{id:"task",opens_as:"item"},{id:"goal",opens_as:"auto"}]} as unknown as Pick<Schema,"types">;
describe("open target",()=>{
 it("follows the server-resolved setting",()=>{
  expect(openTarget({type_id:"client",opens_as:"container"},schema)).toBe("contents");
  expect(openTarget({type_id:"task",opens_as:"item"},schema)).toBe("details");
  expect(openTarget({type_id:"goal",opens_as:"container"},schema)).toBe("contents");
 });
 it("prefers the record value over its type setting",()=>{
  expect(openTarget({type_id:"client",opens_as:"item"},schema)).toBe("details");
 });
 it("falls back to an explicit type setting and otherwise opens details",()=>{
  expect(openTarget({type_id:"client"},schema)).toBe("contents");
  expect(openTarget({type_id:"task"},schema)).toBe("details");
  expect(openTarget({type_id:"goal"},schema)).toBe("details");
  expect(openTarget({type_id:"unknown"})).toBe("details");
 });
});
