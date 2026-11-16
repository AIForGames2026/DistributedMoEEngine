import { defineConfig } from 'vite';

// Relative base so the build works under the GitHub Pages project path. The main page,
// index.html, is the engine's own live.html, built after this by scripts/showcase.mjs.
export default defineConfig({
  base: './',
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 800,
    rollupOptions: { input: { inspector: 'inspector.html' } },
  },
});
