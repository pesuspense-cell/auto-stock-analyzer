// GET /api/v1/settings/alerts   시그널 봇 알림 on/off 조회
// PUT /api/v1/settings/alerts   알림 on/off 저장 (user_settings.alert_prefs)
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import { ALERT_TYPES } from "@/lib/api-types";
import type { ApiError, AlertPrefs, AlertPrefsResponse } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function sanitize(raw: unknown): AlertPrefs {
  const out: AlertPrefs = {};
  if (raw && typeof raw === "object") {
    for (const t of ALERT_TYPES) {
      const v = (raw as Record<string, unknown>)[t];
      if (typeof v === "boolean") out[t] = v;
    }
  }
  return out;
}

export async function GET(): Promise<NextResponse<AlertPrefsResponse | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const { data, error } = await supabase
      .from("user_settings")
      .select("alert_prefs")
      .eq("user_id", user.id)
      .maybeSingle();
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    return NextResponse.json({ alertPrefs: sanitize(data?.alert_prefs) });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}

export async function PUT(req: Request): Promise<NextResponse<AlertPrefsResponse | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const prefs = sanitize(await req.json());
    const { error } = await supabase
      .from("user_settings")
      .upsert(
        { user_id: user.id, alert_prefs: prefs, updated_at: new Date().toISOString() },
        { onConflict: "user_id" },
      );
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    return NextResponse.json({ alertPrefs: prefs });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}
