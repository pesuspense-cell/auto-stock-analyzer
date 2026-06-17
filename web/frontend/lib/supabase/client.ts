// 브라우저(클라이언트 컴포넌트)용 Supabase 클라이언트.
// 공개 anon 키 사용 — 모든 접근은 RLS 정책으로 보호된다.
import { createBrowserClient } from "@supabase/ssr";

import type { Database } from "@/lib/supabase/types";

export function createClient() {
  return createBrowserClient<Database>(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );
}
