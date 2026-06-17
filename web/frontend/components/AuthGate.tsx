"use client";

import { useState } from "react";
import type { User } from "@supabase/supabase-js";

import { createClient } from "@/lib/supabase/client";
import { useUser } from "@/hooks/useUser";

/**
 * Supabase Auth 로그인/회원가입 게이트.
 * 기존 database.py(werkzeug + session_token) 인증을 Supabase Auth 로 대체.
 * 로그인 시 children 렌더, 비로그인 시 폼 표시.
 */
export function AuthGate({ children }: { children: (user: User) => React.ReactNode }) {
  const { user, loading } = useUser();

  if (loading) {
    return <div className="p-12 text-center text-ink-2">세션 확인 중…</div>;
  }
  if (!user) return <AuthForm />;
  return <>{children(user)}</>;
}

function AuthForm() {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    setMsg(null);
    const supabase = createClient();
    try {
      if (mode === "login") {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        // onAuthStateChange 가 자동 반영
      } else {
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        setMsg("가입 완료! 이메일 인증 후 로그인하세요. (기존 사용자는 데이터가 자동 이관됩니다)");
        setMode("login");
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : "인증 실패");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-sm rounded-card border border-hairline bg-surface p-6 shadow-card">
      <h2 className="mb-1 text-lg font-bold text-ink">
        {mode === "login" ? "로그인" : "회원가입"}
      </h2>
      <p className="mb-4 text-sm text-ink-2">포트폴리오는 로그인 후 이용할 수 있습니다.</p>
      <form onSubmit={submit} className="space-y-3">
        <input
          type="email" required value={email} onChange={(e) => setEmail(e.target.value)}
          placeholder="이메일"
          className="w-full rounded-lg border border-hairline-md bg-surface px-3 py-2 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
        />
        <input
          type="password" required minLength={6} value={password} onChange={(e) => setPassword(e.target.value)}
          placeholder="비밀번호 (6자 이상)"
          className="w-full rounded-lg border border-hairline-md bg-surface px-3 py-2 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
        />
        <button
          disabled={busy}
          className="w-full rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white hover:brightness-110 disabled:opacity-40"
        >
          {busy ? "처리 중…" : mode === "login" ? "로그인" : "회원가입"}
        </button>
      </form>
      {err && <p className="mt-3 text-sm text-loss">{err}</p>}
      {msg && <p className="mt-3 text-sm text-gain">{msg}</p>}
      <button
        onClick={() => { setMode(mode === "login" ? "signup" : "login"); setErr(null); setMsg(null); }}
        className="mt-4 text-sm text-accent hover:underline"
      >
        {mode === "login" ? "계정이 없으신가요? 회원가입" : "이미 계정이 있으신가요? 로그인"}
      </button>
    </div>
  );
}

/** 헤더용 로그아웃 버튼 + 이메일 표시. */
export function UserMenu() {
  const { user } = useUser();
  if (!user) return null;
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="text-ink-2">{user.email}</span>
      <button
        onClick={async () => { await createClient().auth.signOut(); }}
        className="rounded-md border border-hairline-md px-2 py-1 text-xs text-ink-2 hover:bg-elevated"
      >
        로그아웃
      </button>
    </div>
  );
}
