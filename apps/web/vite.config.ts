import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import pkg from "./package.json";

// Build identity for Report a bug (bug-report.ts `buildIdentity`). Railway passes RAILWAY_GIT_COMMIT_SHA to the
// Docker build (Dockerfile.upgrade declares the ARG); CI has GITHUB_SHA; a local build says "dev".
const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {};
const build = {
  version: pkg.version,
  commit: (env.RAILWAY_GIT_COMMIT_SHA || env.GITHUB_SHA || "dev").slice(0, 12),
  built_at: new Date().toISOString(),
  mode: env.NODE_ENV ?? "development",
};
export default defineConfig({
  plugins: [react()],
  define: { __ERIDANI_BUILD__: JSON.stringify(build) },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/health": "http://127.0.0.1:8765",
    },
  },
});
