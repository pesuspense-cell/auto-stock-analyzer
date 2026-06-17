import { type NextRequest } from "next/server";

import { updateSession } from "@/lib/supabase/middleware";

// 정적 자산을 제외한 모든 요청에서 Supabase 세션 쿠키를 갱신한다.
export async function middleware(request: NextRequest) {
  return updateSession(request);
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
