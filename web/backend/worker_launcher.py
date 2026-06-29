"""
worker_launcher.py — asa-worker 통합 런처 (Render, 단일 유료 워커)

한 워커 프로세스에서 셋을 각각 자식 프로세스로 띄우고, 죽으면 자동 재시작한다.
PC·웹과 무관하게 24/7 구동되며, 시그널 봇을 별도 워커로 분리하지 않아 비용을 아낀다.

  · worker.jobs_worker     — ASA/백테스트 잡 큐 처리            (cwd = web/backend)
  · signal_bot.py          — 텔레그램 1:N 맞춤 시그널 발송       (cwd = 레포 루트)
  · telegram_link_bot.py   — 딥링크 계정 연동(/start <token>)    (cwd = 레포 루트)

⚠️ 텔레그램 getUpdates 폴러는 전 시스템에서 1개만 가능. 이 워커 가동 시 로컬
   bot.py(start_all.bat)·시작프로그램은 끄세요(동시 폴링 시 409 Conflict).

start command (Render asa-worker):  python worker_launcher.py
필요 env:  (jobs) SUPABASE_DB_URL  +  (signal) TELEGRAM_BOT_TOKEN, SUPABASE_URL,
           SUPABASE_SERVICE_ROLE_KEY
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))            # web/backend
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))       # 레포 루트

CHILDREN: dict[str, dict] = {
    "jobs_worker":       {"cmd": [sys.executable, "-m", "worker.jobs_worker"],    "cwd": HERE},
    "signal_bot":        {"cmd": [sys.executable, "-u", "signal_bot.py"],         "cwd": ROOT},
    "telegram_link_bot": {"cmd": [sys.executable, "-u", "telegram_link_bot.py"],  "cwd": ROOT},
}
RESTART_BACKOFF_SEC = 10   # 즉시 재크래시 루프 방지용 최소 재시작 간격


def _check_env() -> None:
    """시작 시 필수 env 존재 여부를 (값 노출 없이) 로그로 알려 진단을 돕는다."""
    checks = {
        "TELEGRAM_BOT_TOKEN":        bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "SUPABASE_URL":              bool(os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")),
        "SUPABASE_SERVICE_ROLE_KEY": bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SECRET_KEY")),
        "SUPABASE_DB_URL(jobs)":     bool(os.getenv("SUPABASE_DB_URL")),
    }
    print("[worker_launcher] env 점검: "
          + ", ".join(f"{k}={'SET' if v else 'MISSING'}" for k, v in checks.items()), flush=True)
    missing = [k for k, v in checks.items() if not v and not k.endswith("(jobs)")]
    if missing:
        print(f"[worker_launcher] ⚠️ 누락 env: {', '.join(missing)} — '이 서비스(asa-worker)'의 "
              "Environment 에 값을 넣고 재배포하세요. 그때까지 signal_bot/telegram_link_bot 은 "
              "재시작 루프를 돕니다(jobs_worker 는 SUPABASE_DB_URL 로 정상 동작).", flush=True)


def main() -> None:
    print("[worker_launcher] 시작 — " + " + ".join(CHILDREN), flush=True)
    _check_env()
    procs: dict[str, subprocess.Popen] = {}
    last_start: dict[str, float] = {}
    while True:
        for name, spec in CHILDREN.items():
            p = procs.get(name)
            if p is not None and p.poll() is None:
                continue  # 정상 구동 중
            now = time.time()
            if p is not None:
                print(f"[worker_launcher] {name} 종료(code {p.returncode}) → 재시작 예정", flush=True)
            if now - last_start.get(name, 0) < RESTART_BACKOFF_SEC:
                continue
            procs[name] = subprocess.Popen(spec["cmd"], cwd=spec["cwd"])
            last_start[name] = now
            print(f"[worker_launcher] {name} 시작 pid={procs[name].pid}", flush=True)
        time.sleep(15)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[worker_launcher] 종료", flush=True)
