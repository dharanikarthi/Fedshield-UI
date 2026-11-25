import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `--mode demo` (npm run build:demo) bundles the recorded fixtures from demo-public/ and
// answers every API call from them, so the UI runs with no backend (e.g. on Vercel).
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  publicDir: mode === "demo" ? "demo-public" : "public",
  build: {
    rollupOptions: {
      output: {
        manualChunks: { react: ["react", "react-dom", "react-router-dom"], charts: ["recharts"], graph: ["cytoscape"] },
      },
    },
  },
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://localhost:8000", ws: true } },
  },
}));
