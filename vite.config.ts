import { defineConfig } from "vite";
import glsl from "vite-plugin-glsl";

export default defineConfig({
  plugins: [glsl()],
  // Relative base path works for both local dev and GitHub Pages
  base: "/",
});
