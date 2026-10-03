import {chromium} from "@playwright/test";
import {readFileSync,writeFileSync} from "node:fs";
const svg=readFileSync("public/icon.svg","utf8");
const browser=await chromium.launch();
try {for(const size of [192,512]){const page=await browser.newPage({viewport:{width:size,height:size},deviceScaleFactor:1});await page.setContent(`<body style="margin:0;background:#faf9f6;display:grid;place-items:center;width:100vw;height:100vh"><div style="width:75%;height:75%">${svg}</div></body>`);await page.screenshot({path:`public/icon-${size}.png`});await page.close();}}finally{await browser.close();}
const manifest=JSON.parse(readFileSync("public/manifest.webmanifest","utf8"));manifest.icons=[{src:"/icon.svg",sizes:"any",type:"image/svg+xml",purpose:"any"},...[192,512].map(size=>({src:`/icon-${size}.png`,sizes:`${size}x${size}`,type:"image/png",purpose:"any maskable"}))];writeFileSync("public/manifest.webmanifest",JSON.stringify(manifest)+"\n");
