import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Talk to the FastAPI backend on :8000 during development (see backend/app/main.py).
// In production (Phase 8, Docker) the built files are served BY that same backend, so this
// proxy is a dev-only convenience and CORS on the backend covers the rest.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
