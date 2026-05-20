/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        sky: {
          lightest: "#F0F9FF",
          surface: "#c7e8fd",
          border: "#BAE6FD",
          light: "#7DD3FC",
          mid: "#38BDF8",
          DEFAULT: "#0284C7",
          dark: "#0C4A6E",
        },
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        heading: ["Space Grotesk", "sans-serif"],
      },
      borderRadius: {
        card: "8px",
        panel: "12px",
      },
    },
  },
  plugins: [],
};
