import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Development: `npm run dev` serves the UI on http://localhost:5173 and forwards /api to the Python server
// (`uvicorn app:app --reload` from the repo root, port 8000). Production: `npm run build` -> dist/, served by app.py.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
});
