"""
worker_signal.py — Render Background Worker 런처 (PC 없이 24/7 구동)

한 워커 프로세스에서 두 가지를 자식 프로세스로 띄우고, 죽으면 자동 재시작한다.
  · signal_bot.py        — 장중 시그널 1:N 맞춤 발송(연동·수신 ON 사용자 전체)
  · telegram_link_bot.py — 딥링크 계정 연동 처리(/start <token>)

⚠️ 텔레그램 getUpdates 폴러는 전 시스템에서 '단 하나'만 가능하다. 이 워커를 띄우면
   로컬 bot.py(start_all.bat)와 시작프로그램은 반드시 끄세요(동시 폴링 시 409 Conflict).
   /분석 등 대화형 비서까지 클라우드에서 쓰려면 telegram_link_bot 대신 bot.py 를 넣으면
   된다(bot.py 가 연동 /start 도 처리). 단, 그 경우에도 로컬 폴러는 끄세요.

start command (Render):  python worker_signal.py
필요 env:  TELEGRAM_BOT_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# telegram_link_bot 대신 bot.py 로 바꾸면 대화형 비서(/분석)까지 함께 구동된다(폴러 1개 유지).
CHILDREN: dict[str, list[str]] = {
    "signal_bot":        [sys.executable, "-u", "signal_bot.py"],
    "telegram_link_bot": [sys.executable, "-u", "telegram_link_bot.py"],
}

RESTART_BACKOFF_SEC = 10   # 즉시 재크래시 루프 방지용 최소 간격


def main() -> None:
    print("[worker_signal] 시작 — " + " + ".join(CHILDREN), flush=True)
    procs: dict[str, subprocess.Popen] = {}
    last_start: dict[str, float] = {}
    while True:
        for name, cmd in CHILDREN.items():
            p = procs.get(name)
            if p is not None and p.poll() is None:
                continue  # 정상 구동 중
            now = time.time()
            if p is not None:
                print(f"[worker_signal] {name} 종료(code {p.returncode}) → 재시작 예정", flush=True)
            # 너무 빠른 재시작은 백오프
            if now - last_start.get(name, 0) < RESTART_BACKOFF_SEC:
                continue
            procs[name] = subprocess.Popen(cmd, cwd=HERE)
            last_start[name] = now
            print(f"[worker_signal] {name} 시작 pid={procs[name].pid}", flush=True)
        time.sleep(15)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[worker_signal] 종료", flush=True)
