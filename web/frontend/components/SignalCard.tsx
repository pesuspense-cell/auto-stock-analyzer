import { fmtPct, signClass, fmtNum } from "@/lib/format";
import type { AnalysisResponse } from "@/lib/types";

/**
 * 차트 분석 탭의 종합 매매신호 카드.
 * 기존 ui/layouts.py 의 다크 터미널 타일(#161B22/#30363D) 인라인 HTML을 Tailwind 토큰으로 변환.
 */
export function SignalCard({ a }: { a: AnalysisResponse }) {
  const rt = a.realtime;
  return (
    <section className="rounded-card border border-term-border bg-term-1 p-6 shadow-elevated">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm text-term-muted">{a.ticker}</div>
          <h2 className="text-xl font-bold text-term-ink">{a.sname || a.ticker}</h2>
        </div>
        <div className="text-right">
          <div className="tnum text-2xl font-bold text-term-ink">
            {fmtNum(rt.price, 2)}
          </div>
          <div className="text-xs text-term-muted">
            {rt.is_realtime ? `🟢 실시간 ${rt.ts}` : `⏸ ${rt.stale_msg || "지연 시세"}`}
          </div>
        </div>
      </div>

      {/* 하이브리드 신호 배지 */}
      <div className="mt-5 flex items-center gap-3 rounded-lg border border-term-border bg-black/20 px-4 py-3">
        <span className="text-3xl">{a.hybrid.badge}</span>
        <div>
          <div className="text-lg font-bold text-term-ink">{a.hybrid.label}</div>
          <div className="text-xs text-term-muted">
            종합 점수 <span className="tnum font-semibold text-term-ink">{a.hybrid.combined_score.toFixed(1)}</span> / 100
          </div>
        </div>
      </div>

      {/* 점수 그리드 */}
      <div className="mt-4 grid grid-cols-3 gap-2">
        <ScoreChip label="기술적" value={a.tech_score} />
        <ScoreChip label="뉴스 감성" value={a.news_score} />
        <ScoreChip
          label="펀더멘털"
          value={(a.fund_score_data?.fund_score as number) ?? 0}
        />
      </div>

      {/* 사유 */}
      {a.hybrid.reasons.length > 0 && (
        <ul className="mt-4 space-y-1 text-[0.82rem] text-term-muted">
          {a.hybrid.reasons.map((r, i) => (
            <li key={i}>• {r}</li>
          ))}
        </ul>
      )}
      {a.hybrid.warnings.map((w, i) => (
        <p key={i} className="mt-2 text-[0.82rem] text-amber-400">
          {w}
        </p>
      ))}
    </section>
  );
}

function ScoreChip({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border border-term-border bg-black/20 px-3 py-2 text-center">
      <div className="text-[0.62rem] text-term-muted">{label}</div>
      <div className={`tnum text-base font-bold ${signClass(value)}`}>
        {value >= 0 ? "+" : ""}
        {value.toFixed(1)}
      </div>
    </div>
  );
}
