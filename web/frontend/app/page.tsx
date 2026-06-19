"use client";

import { useState } from "react";
import type { ReactNode } from "react";
import { StockSearch } from "@/components/StockSearch";
import { MarketTab } from "@/components/MarketTab";
import { ChartAnalysisTab } from "@/components/ChartAnalysisTab";
import { NewsTab } from "@/components/NewsTab";
import { FundamentalTab } from "@/components/FundamentalTab";
import { AsaTab } from "@/components/AsaTab";
import { BacktestTab } from "@/components/BacktestTab";
import { PortfolioTab } from "@/components/PortfolioTab";
import type { StockHit } from "@/lib/api-types";
import clsx from "clsx";

const TABS = [
  { id: "market", label: "🌐 시장 현황" },
  { id: "chart", label: "📊 차트 분석" },
  { id: "news", label: "📰 뉴스" },
  { id: "fund", label: "🏛️ 펀더멘털" },
  { id: "portfolio", label: "💼 포트폴리오" },
  { id: "asa", label: "🤖 ASA 추천" },
  { id: "backtest", label: "🔬 백테스트" },
] as const;
type TabId = (typeof TABS)[number]["id"];

export default function Dashboard() {
  const [tab, setTab] = useState<TabId>("market");
  const [picked, setPicked] = useState<StockHit | null>(null);
  // 한 번이라도 연 탭만 마운트(불필요한 초기 fetch 방지)하고, 이후엔 언마운트하지 않고
  // CSS(hidden)로만 숨겨 각 탭의 입력·결과 상태를 유지(keep-alive)한다.
  const [opened, setOpened] = useState<Set<TabId>>(() => new Set<TabId>(["market"]));

  function go(id: TabId) {
    setTab(id);
    setOpened((prev) => (prev.has(id) ? prev : new Set(prev).add(id)));
  }

  const panels: Record<TabId, ReactNode> = {
    market: <MarketTab />,
    chart: <ChartAnalysisTab picked={picked} />,
    news: <NewsTab picked={picked} />,
    fund: <FundamentalTab picked={picked} />,
    portfolio: <PortfolioTab />,
    asa: <AsaTab />,
    backtest: <BacktestTab />,
  };

  return (
    <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-4 p-4 md:flex-row">
      {/* 사이드바 — 데스크톱 고정, 모바일 상단 */}
      <aside className="md:w-72 md:shrink-0">
        <div className="rounded-card border border-hairline bg-surface p-4 shadow-card md:sticky md:top-4">
          <h1 className="mb-3 text-base font-bold text-ink">📈 AI 주식 분석 터미널</h1>
          <StockSearch
            onPick={(h) => {
              setPicked(h);
              go("chart");
            }}
          />
        </div>
      </aside>

      {/* 본문 */}
      <main className="flex-1">
        {/* 탭 헤더 — 클라이언트 전환, 전체 리프레시 없음 */}
        <nav className="mb-4 flex flex-wrap gap-1 border-b border-hairline">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => go(t.id)}
              className={clsx(
                "px-3 py-2 text-sm font-medium transition",
                tab === t.id
                  ? "border-b-2 border-accent text-accent"
                  : "text-ink-2 hover:text-ink"
              )}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {/* keep-alive: 연 탭은 유지, 비활성 탭은 hidden 으로 숨김(상태 보존) */}
        {TABS.map((t) =>
          opened.has(t.id) ? (
            <div key={t.id} hidden={tab !== t.id}>
              {panels[t.id]}
            </div>
          ) : null
        )}
      </main>
    </div>
  );
}
