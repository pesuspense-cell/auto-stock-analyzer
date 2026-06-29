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


def main() -> None:
    print("[worker_launcher] 시작 — " + " + ".join(CHILDREN), flush=True)
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
