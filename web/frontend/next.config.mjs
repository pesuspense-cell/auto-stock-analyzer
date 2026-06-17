/** @type {import('next').NextConfig} */

// ⚠️ 하이브리드 라우팅 주의:
//   배열 형태 rewrites 는 `afterFiles` 단계 → "정적 라우트 다음, 동적 라우트 이전"에 평가된다.
//   따라서 catch-all `/api/:path*` 를 쓰면 동적 로컬 라우트(/api/v1/quote/[ticker],
//   /api/v1/portfolio/[id] 등)가 프록시에 가려진다(shadowed).
//   → 해결: FastAPI(경로 A)로 폴백할 "미구현 경로"만 명시적으로 나열한다.
//      로컬(경로 B)로 이전 완료된 경로는 이 목록에서 제거하면 자동으로 로컬이 처리한다.
const FASTAPI_FALLBACK_PATHS = [
  "/api/v1/analysis/:path*",
  "/api/v1/news/:path*",
  "/api/v1/fundamental/:path*",
  "/api/v1/market/overview",
  "/api/v1/stocks/list",
  // 주의: /asa, /backtest, /jobs 는 jobs 큐 기반 로컬 라우트(경로 B)로 이전 완료 → 폴백 제외.
  //       Python 워커가 jobs 테이블을 처리한다(워커는 FastAPI HTTP 가 아님).
];

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // BACKEND_URL 이 비었거나 스킴(http/https) 없는 값이면 rewrites 가 무효가 되어
    // 빌드가 "Invalid rewrites found" 로 실패한다. → 유효한 절대 URL 일 때만 폴백을 켠다.
    const raw = (process.env.BACKEND_URL || "http://localhost:8000").trim();
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
