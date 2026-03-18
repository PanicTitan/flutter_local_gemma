import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  build: {
    outDir: path.resolve(__dirname, '..'), // emit into the plugin web/ folder
    emptyOutDir: false,
    sourcemap: false,
    rollupOptions: {
      // bundle everything into a single ES module file named gemma_web.js
      input: path.resolve(__dirname, 'src', 'main.ts'),
      output: {
        entryFileNames: 'dist/gemma_web.js',
        chunkFileNames: 'dist/gemma_web.js',
        assetFileNames: 'dist/[name].[ext]',
        format: 'es',
        inlineDynamicImports: true,
      },
    },
  },
});
