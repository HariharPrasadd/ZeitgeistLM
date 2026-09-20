import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // Keep generated shadcn imports stable across local and production builds.
    alias: { '@': path.resolve(import.meta.dirname, './src') },
  },
})
