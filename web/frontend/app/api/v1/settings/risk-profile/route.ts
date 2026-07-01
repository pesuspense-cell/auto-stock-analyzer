// GET /api/v1/settings/risk-profile   매매 위험성향 조회
// PUT /api/v1/settings/risk-profile   위험성향 저장 (user_settings.risk_profile)
//   "safe"       = 안전투자형 v4.6 (강한 방어, 기본값)
//   "aggressive" = 위험감수형 v5.5 (야수커트 70pt·종목캡 33%·당일손절 2종목·디펜스 눌림목 허용)
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import type { ApiError } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RiskProfile = "safe" | "aggressive";
interface RiskProfileResponse {
  riskProfile: RiskProfile;
}

function sanitize(raw: unknown): RiskProfile {
  const v = raw && typeof raw === "object" ? (raw as Record<string, unknown>).riskProfile : raw;
  return v === "aggressive" ? "aggressive" : "safe";
}

export async function GET(): Promise<NextResponse<RiskProfileResponse | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const { data, error } = await supabase
      .from("user_settings")
      .select("risk_profile")
      .eq("user_id", user.id)
      .maybeSingle();
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    return NextResponse.json({
      riskProfile: data?.risk_profile === "aggressive" ? "aggressive" : "safe",
    });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}

export async function PUT(req: Request): Promise<NextResponse<RiskProfileResponse | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const riskProfile = sanitize(await req.json());
    const { error } = await supabase
      .from("user_settings")
      .upsert(
        { user_id: user.id, risk_profile: riskProfile, updated_at: new Date().toISOString() },
        { onConflict: "user_id" },
      );
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    return NextResponse.json({ riskProfile });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}
