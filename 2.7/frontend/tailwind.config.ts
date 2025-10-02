import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // ZION 2.7 Color Palette
        zion: {
          primary: '#00ff00',
          secondary: '#00ffff', 
          accent: '#ff00ff',
          dark: '#0a0a0a',
          light: '#ffffff',
          success: '#22c55e',
          warning: '#f59e0b',
          error: '#ef4444',
        },
        cosmic: {
          blue: '#1e3a8a',
          purple: '#7c3aed',
          pink: '#ec4899',
          gold: '#f59e0b',
        }
      },
      backgroundImage: {
        'zion-gradient': 'linear-gradient(135deg, #0a0a0a, #1a1a2e, #16213e, #0f3460)',
        'cosmic-gradient': 'linear-gradient(45deg, #1e3a8a, #7c3aed, #ec4899)',
        'mining-gradient': 'linear-gradient(135deg, #f59e0b, #eab308)',
      },
      animation: {
        'zion-pulse': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'zion-spin': 'spin 3s linear infinite',
        'zion-bounce': 'bounce 1s infinite',
        'data-flow': 'dataFlow 2s ease-in-out infinite',
      },
      keyframes: {
        dataFlow: {
          '0%': { transform: 'translateX(-100%)', opacity: '0' },
          '50%': { opacity: '1' },
          '100%': { transform: 'translateX(100%)', opacity: '0' },
        }
      },
      fontFamily: {
        'mono': ['Monaco', 'Consolas', 'Courier New', 'monospace'],
        'cosmic': ['Orbitron', 'sans-serif'],
      }
    },
  },
  plugins: [
    // Add glow effects for ZION theme
    function({ addUtilities }) {
      const newUtilities = {
        '.glow-green': {
          'box-shadow': '0 0 20px rgba(0, 255, 0, 0.5)',
        },
        '.glow-cyan': {
          'box-shadow': '0 0 20px rgba(0, 255, 255, 0.5)',
        },
        '.glow-purple': {
          'box-shadow': '0 0 20px rgba(124, 58, 237, 0.5)',
        },
        '.text-glow-green': {
          'text-shadow': '0 0 10px rgba(0, 255, 0, 0.8)',
        },
        '.text-glow-cyan': {
          'text-shadow': '0 0 10px rgba(0, 255, 255, 0.8)',
        }
      }
      addUtilities(newUtilities)
    }
  ],
  darkMode: 'class',
}
export default config