// POST   /api/v1/telegram/link   딥링크 연동 토큰 발급 → t.me/<bot>?start=<token>
// DELETE /api/v1/telegram/link   연동 해제(chat_id 제거 + 알림 off)
import { NextResponse } from "next/server";
import { randomUUID } from "crypto";

import { createServerSupabase } from "@/lib/supabase/server";
import { requireUser, UnauthorizedError, unauthorized } from "@/lib/auth-route";
import type { ApiError, OkResponse, TelegramLinkResponse } from "@/lib/api-types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BOT_USERNAME =
  process.env.TELEGRAM_BOT_USERNAME ||
  process.env.NEXT_PUBLIC_TELEGRAM_BOT_USERNAME ||
  "asa_jhs_bot";

export async function POST(): Promise<NextResponse<TelegramLinkResponse | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const token = randomUUID().replace(/-/g, ""); // 32 hex, 딥링크 start 파라미터로 안전

    const { error } = await supabase
      .from("user_settings")
      .upsert(
        { user_id: user.id, telegram_link_token: token, updated_at: new Date().toISOString() },
        { onConflict: "user_id" },
      );
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });

    return NextResponse.json({
      token,
      botUsername: BOT_USERNAME,
      deepLink: `https://t.me/${BOT_USERNAME}?start=${token}`,
    });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}

export async function DELETE(): Promise<NextResponse<OkResponse | ApiError>> {
  const supabase = await createServerSupabase();
  try {
    const user = await requireUser(supabase);
    const { error } = await supabase
      .from("user_settings")
      .update({
        telegram_chat_id: null,
        telegram_enabled: false,
        telegram_link_token: null,
        updated_at: new Date().toISOString(),
      })
      .eq("user_id", user.id);
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    return NextResponse.json({ ok: true });
  } catch (e) {
    if (e instanceof UnauthorizedError) return unauthorized();
    throw e;
  }
}
