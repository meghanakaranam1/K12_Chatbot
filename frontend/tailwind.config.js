/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#3E94A5',
        text: '#2A3B7C',
        accent: '#EFB340',
      },
    },
  },
  plugins: [],
}


