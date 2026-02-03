/// <reference types="vitest" />
import { defineConfig } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: '/static/dashboard/',
  plugins: [
    // The React and Tailwind plugins are both required for Make, even if
    // Tailwind is not being actively used – do not remove them
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      // Alias @ to the src directory
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return;
          }

          const packagePath = id.split('node_modules/')[1];
          if (!packagePath) {
            return;
          }

          const matchesPackage = (pkg: string, name: string) =>
            pkg === name || pkg.startsWith(`${name}/`);

          if (
            ['react', 'react-dom', 'react-router', 'react-router-dom'].some((pkg) =>
              matchesPackage(packagePath, pkg),
            )
          ) {
            return 'vendor-react';
          }

          if (
            [
              '@radix-ui',
              '@mui',
              '@emotion',
              'lucide-react',
              'react-resizable-panels',
            ].some((pkg) => matchesPackage(packagePath, pkg))
          ) {
            return 'vendor-ui';
          }

          if (
            ['recharts', 'd3', 'd3-array', 'd3-color', 'd3-format', 'd3-scale', 'd3-shape'].some(
              (pkg) => matchesPackage(packagePath, pkg),
            )
          ) {
            return 'vendor-charts';
          }

          if (
            ['date-fns', 'class-variance-authority', 'clsx', 'tailwind-merge'].some((pkg) =>
              matchesPackage(packagePath, pkg),
            )
          ) {
            return 'vendor-utils';
          }
        },
      },
    },
  },
})
