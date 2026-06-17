import type { Config } from "tailwindcss";

/**
 * 기존 ui/styles.py 의 COLORS 토큰(Apple Finance 디자인 시스템)을 Tailwind 테마로 이식.
 * 인라인 style="color:#1d1d1f" → className="text-ink" 처럼 의미 토큰으로 변환한다.
 */
const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // 배경 레이어
        canvas: "#f5f5f7",
        surface: "#ffffff",
        elevated: "#fafafc",
        // 경계선
        hairline: "#e0e0e0",
        "hairline-md": "#d2d2d7",
        // 텍스트
        ink: "#1d1d1f",
        "ink-2": "#7a7a7a",
        "ink-3": "#b0b0b0",
        // 매매 신호
        gain: "#34c759",
        loss: "#ff3b30",
        accent: "#0066cc",
        sky: "#2997ff",
        // 다크 터미널 타일
        "term-1": "#161b22",
        "term-2": "#272729",
        "term-border": "#30363d",
        "term-ink": "#e6edf3",
        "term-muted": "#8b949e",
      },
      borderRadius: { card: "12px", lg: "16px" },
      boxShadow: {
        card: "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06)",
        elevated: "rgba(0,0,0,0.22) 3px 5px 30px 0",
      },
      fontFamily: {
        sans: ["-apple-system", "BlinkMacSystemFont", "Pretendard", "Segoe UI", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
