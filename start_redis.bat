@echo off
echo Starting Redis server on port 6379...
cd /d "%~dp0"
redis\redis-server.exe redis\redis.windows.conf --port 6379 --loglevel notice
