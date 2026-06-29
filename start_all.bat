@echo off
cd /d "%~dp0"

:: logs 폴더 생성
if not exist logs mkdir logs

:: ⚠️ 이 파일은 한 번만 실행하세요. 두 번 실행하면 bot.py 가 중복 폴링되어
::    텔레그램 409 Conflict 가 납니다. (시작프로그램 등록 시 로그인당 1회 자동 실행)

:: Redis 시작 (백그라운드)
echo [1/3] Redis 시작...
start /min "StockRedis" redis\redis-server.exe redis\redis.windows.conf --port 6379 --loglevel warning
timeout /t 2 /nobreak > nul

:: Telegram 비서 봇 + 딥링크 계정 연동 처리 (/start ^<token^>) — 상시 폴링
echo [2/3] Telegram Bot(연동 처리 포함) 시작...
start /min "StockBot" cmd /c ".venv\Scripts\python.exe -u bot.py >> logs\bot.log 2>&1"

:: 실시간 매매 시그널 봇 (장중 09:00~15:30 자동 감시, 장외엔 대기) — 상시
echo [3/3] Signal Bot 시작...
start /min "StockSignalBot" cmd /c ".venv\Scripts\python.exe -u signal_bot.py >> logs\signal_bot.log 2>&1"

echo 완료. 로그: logs\bot.log , logs\signal_bot.log
