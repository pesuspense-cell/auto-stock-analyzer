// Route Handler 인증 헬퍼 — 세션 사용자(uuid)를 반환하거나 401 을 던진다.
import { NextResponse } from "next/server";
import type { SupabaseClient, User } from "@supabase/supabase-js";

import type { Database } from "@/lib/supabase/types";

export class UnauthorizedError extends Error {}

/** 세션 사용자 반환. 없으면 UnauthorizedError. */
export async function requireUser(supabase: SupabaseClient<Database>): Promise<User> {
  const { data, error } = await supabase.auth.getUser();
  if (error || !data.user) throw new UnauthorizedError("로그인이 필요합니다.");
  return data.user;
}

/** UnauthorizedError → 401 JSON 응답으로 변환하는 래퍼. */
export function unauthorized() {
  return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
}
