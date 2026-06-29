// GET /api/v1/telegram/status   텔레그램 연동/수신 상태 조회
// PUT /api/v1/telegram/status   알림 수신 on/off (연동된 경우만)
import { NextResponse } from "next/server";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import type { ApiError, TelegramStatus } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const mask = (id: string | null): string | null =>
  id ? id.slice(0, 3) + "***" + id.slice(-2) : null;

export async function GET(): Promise<NextResponse<TelegramStatus | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const { data, error } = await supabase
      .from("user_settings")
      .select("telegram_chat_id,telegram_enabled")
      .eq("user_id", user.id)
      .maybeSingle();
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    return NextResponse.json({
      linked: !!data?.telegram_chat_id,
      enabled: !!data?.telegram_enabled,
      chatId: mask(data?.telegram_chat_id ?? null),
    });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}

export async function PUT(req: Request): Promise<NextResponse<TelegramStatus | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const body = (await req.json()) as { enabled?: boolean };
    const enabled = !!body.enabled;

    const { data: cur } = await supabase
      .from("user_settings")
      .select("telegram_chat_id")
      .eq("user_id", user.id)
      .maybeSingle();
    if (enabled && !cur?.telegram_chat_id) {
      return NextResponse.json({ error: "먼저 텔레그램 연동을 완료해 주세요." }, { status: 400 });
    }

    const { error } = await supabase
      .from("user_settings")
      .upsert(
        { user_id: user.id, telegram_enabled: enabled, updated_at: new Date().toISOString() },
        { onConflict: "user_id" },
      );
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    return NextResponse.json({
      linked: !!cur?.telegram_chat_id,
      enabled,
      chatId: mask(cur?.telegram_chat_id ?? null),
    });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}
