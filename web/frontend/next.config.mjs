/** @type {import('next').NextConfig} */

// ⚠️ 하이브리드 라우팅 주의:
//   배열 형태 rewrites 는 `afterFiles` 단계 → "정적 라우트 다음, 동적 라우트 이전"에 평가된다.
//   따라서 catch-all `/api/:path*` 를 쓰면 동적 로컬 라우트(/api/v1/quote/[ticker],
//   /api/v1/portfolio/[id] 등)가 프록시에 가려진다(shadowed).
//   → 해결: FastAPI(경로 A)로 폴백할 "미구현 경로"만 명시적으로 나열한다.
//      로컬(경로 B)로 이전 완료된 경로는 이 목록에서 제거하면 자동으로 로컬이 처리한다.
// ⚠️ trailing-slash 주의:
//   `:path*` 는 베이스 경로 매칭 시 끝에 슬래시를 붙여(/analysis → /analysis/) 보내,
//   슬래시 없는 FastAPI 라우트가 307 리다이렉트를 내고 그게 교차출처로 새어 CORS 가
//   터진다. → POST 베이스 엔드포인트는 "정확 경로"를 :path* 보다 **먼저** 나열해
//   슬래시 없이 프록시(same-origin 유지)한다.
const FASTAPI_FALLBACK_PATHS = [
  // 경량·즉답 엔드포인트만 FastAPI(경로 A)로 폴백한다.
  "/api/v1/news/summarize",   // 단일 기사 AI 요약(빠른 1회 호출) — 큐 미적용
  "/api/v1/market/overview",
  "/api/v1/stocks/list",
  // 주의: /analysis, /news(메인), /fundamental, /fundamental/ai-report 는 jobs 큐(워커
  //       interactive 레인)로 이전 완료 → 폴백 제외. 로컬 /…/run 라우트가 큐에 적재하고
  //       /jobs/[id] 폴링으로 결과를 받는다.
  //       /asa, /backtest 도 동일하게 jobs 큐 기반 로컬 라우트로 처리한다.
];

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // 기본값 = 프로덕션 asa-api. Vercel 의 BACKEND_URL 값이 비어도(빈 문자열=falsy)
    // 이 기본값으로 폴백해 rewrites 가 동작한다. 로컬 개발은 .env.local 의
    // BACKEND_URL=http://localhost:8000 이 우선(override)한다.
    // 스킴(http/https) 없는 값이면 rewrites 가 무효라 빌드 실패하므로 유효 URL 만 켠다.
    const raw = (process.env.BACKEND_URL || "https://asa-api-2wh0.onrender.com").trim();
    const backend = /^https?:\/\//i.test(raw) ? raw.replace(/\/+$/, "") : "";
    if (!backend) {
      console.warn(
        "[next.config] BACKEND_URL 미설정/형식오류 → FastAPI 폴백 비활성. " +
          "Render 배포 후 https://<asa-api> 형태로 지정하세요.",
      );
      return [];
    }
    return FASTAPI_FALLBACK_PATHS.map((source) => ({
      source,
      destination: `${backend}${source}`,
    }));
  },
};

export default nextConfig;
