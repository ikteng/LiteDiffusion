import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  base: "/studio-assets/",
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/status": "http://127.0.0.1:7860",
      "/studio-config": "http://127.0.0.1:7860",
      "/gradio_api": {
        target: "http://127.0.0.1:7860",
        ws: true,
      },
    },
  },
});
