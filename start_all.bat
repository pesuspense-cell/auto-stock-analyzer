@echo off
cd /d "%~dp0"

:: logs 폴더 생성
if not exist logs mkdir logs

:: Redis 시작 (백그라운드)
echo [1/2] Redis 시작...
start /min "StockRedis" redis\redis-server.exe redis\redis.windows.conf --port 6379 --loglevel warning
timeout /t 2 /nobreak > nul

:: Bot 시작 (백그라운드, 로그 파일 기록)
echo [2/2] Telegram Bot 시작...
start /min "StockBot" cmd /c ".venv\Scripts\python.exe -u bot.py >> logs\bot.log 2>&1"

echo 완료. 로그: logs\bot.log
