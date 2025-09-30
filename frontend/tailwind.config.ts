import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: 'class',
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#effef6',
          100: '#d9fde9',
          200: '#b4fbd3',
          300: '#7df6b3',
          400: '#3aeb89',
          500: '#0fd96b',
          600: '#05b659',
          700: '#0a8d49',
          800: '#0f6f3d',
          900: '#0f5b34',
        },
      },
      boxShadow: {
        glow: '0 0 0 2px rgba(15, 217, 107, 0.2)',
      },
    },
  },
  plugins: [],
}

export default config
