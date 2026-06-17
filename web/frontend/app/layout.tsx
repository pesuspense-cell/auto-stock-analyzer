import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ASA Service — AI 주식 분석 터미널",
  description: "FastAPI + Next.js 기반 AI 주식 분석 대시보드",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
