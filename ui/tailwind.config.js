/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        page: "#0d0d0d", surface: "#1a1a19", raised: "#232321", line: "#2c2c2a", axis: "#383835",
        ink: { DEFAULT: "#f0efec", 2: "#c3c2b7", muted: "#898781" },
        s1: "#3987e5", s2: "#d95926", s3: "#199e70",
        good: "#0ca30c", warn: "#fab219", serious: "#ec835a", critical: "#d03b3b",
      },
      fontFamily: { sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"], mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"] },
    },
  },
  plugins: [],
};
