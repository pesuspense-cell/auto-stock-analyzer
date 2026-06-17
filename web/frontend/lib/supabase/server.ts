// 서버(Route Handler / 서버 컴포넌트)용 Supabase 클라이언트.
//
// - createServerSupabase(): 쿠키 기반 사용자 세션 클라이언트 (RLS 적용, auth.uid() 동작)
// - createServiceClient(): service_role 키 클라이언트 (RLS 우회 — 캐시 테이블 쓰기 전용)
//
// service_role 키는 절대 클라이언트로 노출되지 않으며, 서버리스 함수 내부에서만 쓰인다.
import { createServerClient } from "@supabase/ssr";
import { createClient } from "@supabase/supabase-js";
import { cookies } from "next/headers";

import type { Database } from "@/lib/supabase/types";

/** 로그인 사용자 컨텍스트 (RLS 정책 + auth.uid() 적용). */
export async function createServerSupabase() {
  const cookieStore = await cookies();
  return createServerClient<Database>(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll: () => cookieStore.getAll(),
        setAll: (toSet) => {
          try {
            toSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options)
            );
          } catch {
            // 서버 컴포넌트에서 set 호출 시 무시 (미들웨어가 세션 갱신 담당)
          }
        },
      },
    }
  );
}

/** RLS를 우회하는 서비스 롤 클라이언트 — 캐시 upsert 등 서버 전용 쓰기에만 사용. */
export function createServiceClient() {
  return createClient<Database>(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!,
    { auth: { persistSession: false, autoRefreshToken: false } }
  );
}
