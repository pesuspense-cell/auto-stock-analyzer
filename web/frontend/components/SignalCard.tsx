import { signClass, fmtNum } from "@/lib/format";
import type { AnalysisResponse } from "@/lib/types";
import { InfoTip } from "./InfoTip";

// 각 지표 설명 — "?" 툴팁으로 노출.
const HELP = {
  combined: "기술적·뉴스·펀더멘털 신호를 가중 결합한 0~100 종합 점수입니다. 높을수록 매수 우위, 낮을수록 매도 우위로 해석합니다.",
  tech: "이동평균·RSI·MACD 등 차트 기술지표를 종합한 점수입니다. 양수면 상승 신호 우위, 음수면 하락 신호 우위입니다.",
  news: "최근 뉴스의 긍정/부정 톤을 분석한 점수입니다. 양수면 호재 우위, 음수면 악재 우위로 봅니다.",
  fund: "PER·PBR·ROE 등 재무지표를 종합한 점수입니다. 기업의 실적·밸류에이션 건전성을 나타냅니다.",
  expReturn: "과거 변동성·추세를 바탕으로 향후 약 20거래일(1개월) 기대 수익률을 추정한 값입니다. 아래 구간은 통계적 변동 범위입니다.",
  winProb: "추정 기간 내 수익으로 끝날 확률입니다. 55% 이상이면 우위, 45% 미만이면 열위로 봅니다.",
  kelly: "켈리 공식 기반 권장 투자 비중입니다. 변동성을 줄이기 위해 그 절반(하프-켈리)을 적용합니다.",
  sharpe: "위험(변동성) 대비 초과수익입니다. 1 이상이면 위험 대비 수익이 양호, 0 미만이면 부진합니다.",
  beta: "시장(지수) 대비 민감도입니다. 1보다 크면 시장보다 크게 출렁이고, 작으면 더 안정적입니다.",
  vol: "연율화한 가격 변동 폭입니다. 클수록 등락이 심해 리스크가 큽니다.",
  mdd: "과거 고점 대비 최대 하락 폭(MDD)입니다. 해당 기간 가장 크게 손실 났던 정도를 나타냅니다.",
} as const;

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
          <div className="flex items-center gap-1 text-xs text-term-muted">
            <span>종합 점수 <span className="tnum font-semibold text-term-ink">{a.hybrid.combined_score.toFixed(1)}</span> / 100</span>
            <InfoTip text={HELP.combined} label="종합 점수" />
          </div>
        </div>
      </div>

      {/* 점수 그리드 */}
      <div className="mt-4 grid grid-cols-3 gap-2">
        <ScoreChip label="기술적" value={a.tech_score} help={HELP.tech} />
        <ScoreChip label="뉴스 감성" value={a.news_score} help={HELP.news} />
        <ScoreChip
          label="펀더멘털"
          value={(a.fund_score_data?.fund_score as number) ?? 0}
          help={HELP.fund}
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

/** 예상 수익률·승률·켈리 등 사용자 관심 지표 (analysis 의 expected/risk_adj). */
export function ExpectedReturnCard({ a }: { a: AnalysisResponse }) {
  const e = (a.expected ?? {}) as Record<string, unknown>;
  const risk = (a.risk_adj ?? {}) as Record<string, unknown>;
  const n = (k: string): number | null => (typeof e[k] === "number" ? (e[k] as number) : null);

  const exp = n("expected_return_pct");
  if (exp == null) return null; // expected 미산출(데이터 부족)
  const low = n("return_low");
  const high = n("return_high");
  const win = n("win_prob");
  const kelly = n("kelly_pct");
  const sharpe = n("sharpe");
  const beta = n("beta");
  const vol = n("hist_volatility");
  const mdd = n("max_drawdown");

  const consApplied = risk["conservative_applied"] === true;
  const consNote = typeof risk["conservative_note"] === "string" ? (risk["conservative_note"] as string) : null;

  return (
    <section className="rounded-card border border-term-border bg-term-1 p-5 shadow-elevated">
      <div className="mb-3 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-term-ink">📈 예상 수익률 · 리스크</h3>
        <span className="text-[0.66rem] text-term-muted">향후 20거래일 추정</span>
      </div>

      {/* 핵심 3종: 예상수익률(구간) / 승률 / 켈리 추천비중 */}
      <div className="grid grid-cols-3 gap-2">
        <BigStat
          label="예상 수익률"
          value={`${exp >= 0 ? "+" : ""}${exp.toFixed(1)}%`}
          tone={exp >= 0 ? "text-gain" : "text-loss"}
          sub={low != null && high != null ? `${low >= 0 ? "+" : ""}${low.toFixed(1)} ~ ${high >= 0 ? "+" : ""}${high.toFixed(1)}%` : undefined}
          help={HELP.expReturn}
        />
        <BigStat
          label="승률"
          value={win != null ? `${win.toFixed(0)}%` : "—"}
          tone={win != null ? (win >= 55 ? "text-gain" : win < 45 ? "text-loss" : "text-term-ink") : "text-term-ink"}
          sub={win != null ? (win >= 55 ? "우위" : win < 45 ? "열위" : "중립") : undefined}
          help={HELP.winProb}
        />
        <BigStat
          label="추천 비중(켈리)"
          value={kelly != null ? `${kelly.toFixed(0)}%` : "—"}
          tone="text-sky"
          sub="하프-켈리"
          help={HELP.kelly}
        />
      </div>

      {/* 보조: 샤프 / 베타 / 변동성 / 최대낙폭 */}
      <div className="mt-2 grid grid-cols-4 gap-2">
        <SmallStat label="샤프지수" value={sharpe != null ? sharpe.toFixed(2) : "—"} tone={sharpe != null && sharpe >= 1 ? "text-gain" : sharpe != null && sharpe < 0 ? "text-loss" : "text-term-ink"} help={HELP.sharpe} />
        <SmallStat label="베타(β)" value={beta != null ? beta.toFixed(2) : "—"} help={HELP.beta} />
        <SmallStat label="변동성(연)" value={vol != null ? `${vol.toFixed(0)}%` : "—"} help={HELP.vol} />
        <SmallStat label="최대낙폭" value={mdd != null ? `${mdd.toFixed(0)}%` : "—"} tone="text-loss" help={HELP.mdd} />
      </div>

      {consApplied && consNote && (
        <p className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[0.76rem] text-amber-300">
          ⚠️ {consNote}
        </p>
      )}
    </section>
  );
}

function StatLabel({ label, help }: { label: string; help?: string }) {
  return (
    <div className="flex items-center justify-center gap-1">
      <span className="text-term-muted">{label}</span>
      {help && <InfoTip text={help} label={label} />}
    </div>
  );
}

function BigStat({ label, value, tone, sub, help }: { label: string; value: string; tone: string; sub?: string; help?: string }) {
  return (
    <div className="rounded-lg border border-term-border bg-black/20 px-3 py-2.5 text-center">
      <div className="text-[0.62rem]"><StatLabel label={label} help={help} /></div>
      <div className={`tnum text-lg font-bold ${tone}`}>{value}</div>
      {sub && <div className="tnum text-[0.62rem] text-term-muted">{sub}</div>}
    </div>
  );
}

function SmallStat({ label, value, tone = "text-term-ink", help }: { label: string; value: string; tone?: string; help?: string }) {
  return (
    <div className="rounded-lg border border-term-border bg-black/20 px-2 py-1.5 text-center">
      <div className="text-[0.58rem]"><StatLabel label={label} help={help} /></div>
      <div className={`tnum text-[0.85rem] font-semibold ${tone}`}>{value}</div>
    </div>
  );
}

function ScoreChip({ label, value, help }: { label: string; value: number; help?: string }) {
  return (
    <div className="rounded-lg border border-term-border bg-black/20 px-3 py-2 text-center">
      <div className="text-[0.62rem]"><StatLabel label={label} help={help} /></div>
      <div className={`tnum text-base font-bold ${signClass(value)}`}>
        {value >= 0 ? "+" : ""}
        {value.toFixed(1)}
      </div>
    </div>
  );
}
