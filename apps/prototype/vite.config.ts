import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Standalone prototype dev server. Port 5174 to avoid clashing with the
// real app (apps/web runs on 5173).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
  },
});
