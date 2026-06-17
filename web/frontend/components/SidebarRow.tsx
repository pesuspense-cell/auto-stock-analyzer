import { signClass } from "@/lib/format";

/**
 * 기존 ui/layouts.py 의 `_sb_row` (f-string 인라인 HTML)를 Tailwind 컴포넌트로 변환.
 *
 * 원본:
 *   <div style="display:flex;justify-content:space-between;align-items:center;
 *               padding:5px 0;border-bottom:1px solid #e0e0e0;">
 *     <span style="font-size:.78rem;color:#7a7a7a;">{label}</span>
 *     <div style="text-align:right">
 *       <div style="font-size:.88rem;font-weight:600;color:#1d1d1f;
 *                   font-variant-numeric:tabular-nums;">{value}</div>
 *       <div style="font-size:.72rem;color:{chg_color};
 *                   font-variant-numeric:tabular-nums;">{change}</div>
 *     </div>
 *   </div>
 */
export function SidebarRow({
  label,
  value,
  change,
  changeValue,
}: {
  label: string;
  value: string;
  change?: string;
  changeValue?: number; // 색상 결정용 (양/음)
}) {
  return (
    <div className="flex items-center justify-between border-b border-hairline py-[5px]">
      <span className="text-[0.78rem] text-ink-2">{label}</span>
      <div className="text-right">
        <div className="tnum text-[0.88rem] font-semibold text-ink">{value}</div>
        {change != null && (
          <div
            className={`tnum text-[0.72rem] ${
              changeValue != null ? signClass(changeValue) : "text-ink-2"
            }`}
          >
            {change}
          </div>
        )}
      </div>
    </div>
  );
}
