"""
analysis_worker.py - 실시간 분석 워커 (Consumer)

역할:
  - Redis Pub/Sub 채널 구독
  - 수신 메시지 처리 및 분석
  - stock_ai.py 핵심 함수 호출
  - 분석 결과에 따라 Telegram 알림 발송

아키텍처:
  Redis Pub/Sub (market_updates)
    ↓
  analysis_worker (Subscriber)
    ↓
  stock_ai.py (분석 로직)
    ↓
  Telegram Bot API (알림)
"""
import asyncio
import json
import logging
import signal
from datetime import datetime
from typing import Optional, Dict, Any

try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# 프로젝트 모듈
try:
    from stock_ai import (
        get_stock_data,
        analyze_technicals,
        get_advanced_sentiment,
    )
    HAS_STOCK_AI = True
except ImportError:
    HAS_STOCK_AI = False
    logger_temp = logging.getLogger("analysis_worker")
    logger_temp.warning("stock_ai.py 임포트 실패 — 분석 기능 제한됨")

# ─── 로깅 설정 ─────────────────────────────────────────────────────────────
logger = logging.getLogger("analysis_worker")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] analysis_worker: %(message)s",
            datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ─── 상수 ──────────────────────────────────────────────────────────────────
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
MARKET_UPDATES_CHANNEL = "market_updates"

# Telegram 설정
TELEGRAM_BOT_TOKEN = ""  # 환경 변수에서 주입
TELEGRAM_CHAT_ID = ""    # 환경 변수에서 주입

# 분석 임계값
PRICE_CHANGE_THRESHOLD = 2.0  # % (변동률 임계값)
VOLUME_SPIKE_THRESHOLD = 1.5  # 배수 (평균 대비)


class TelegramNotifier:
    """Telegram 봇 알림 발송기 (HTML 포맷, retry, 세션 재사용)"""

    _BASE = "https://api.telegram.org"

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._session: Optional[Any] = None

    def _url(self, method: str) -> str:
        return f"{self._BASE}/bot{self.bot_token}/{method}"

    async def _session_get(self):
        if not HAS_AIOHTTP:
            return None
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    async def close(self):
        """세션 정리 (워커 종료 시 호출)"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_me(self) -> Optional[Dict[str, Any]]:
        """봇 토큰 검증 — 봇 정보 반환"""
        session = await self._session_get()
        if not session:
            return None
        try:
            async with session.get(self._url("getMe")) as resp:
                data = await resp.json()
                return data.get("result") if data.get("ok") else None
        except Exception as e:
            logger.error(f"getMe 오류: {e}")
            return None

    async def get_updates(self, offset: int = 0) -> list:
        """최근 수신 메시지 조회 (Chat ID 확인용)"""
        session = await self._session_get()
        if not session:
            return []
        try:
            params = {"offset": offset, "limit": 10, "timeout": 0}
            async with session.get(self._url("getUpdates"), params=params) as resp:
                data = await resp.json()
                return data.get("result", []) if data.get("ok") else []
        except Exception as e:
            logger.error(f"getUpdates 오류: {e}")
            return []

    async def send_message(self, html: str, retries: int = 2) -> bool:
        """
        HTML 형식 메시지 발송 (retry 포함).

        Telegram HTML 태그: <b>, <i>, <code>, <pre>, <a href="...">
        """
        if not HAS_AIOHTTP:
            logger.warning("aiohttp 미설치 — Telegram 알림 불가")
            return False
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram 설정 미완 — 알림 스킵")
            return False

        session = await self._session_get()
        payload = {
            "chat_id": self.chat_id,
            "text": html,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        for attempt in range(retries + 1):
            try:
                async with session.post(self._url("sendMessage"), json=payload) as resp:
                    if resp.status == 200:
                        logger.debug("✓ Telegram 메시지 발송 성공")
                        return True
                    body = await resp.json()
                    logger.warning(f"Telegram API {resp.status}: {body.get('description', '')}")
                    # 429 Too Many Requests — retry_after 준수
                    if resp.status == 429:
                        retry_after = body.get("parameters", {}).get("retry_after", 5)
                        await asyncio.sleep(retry_after)
                    elif attempt < retries:
                        await asyncio.sleep(2 ** attempt)
            except asyncio.TimeoutError:
                logger.warning(f"Telegram 타임아웃 (시도 {attempt + 1})")
                if attempt < retries:
                    await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Telegram 발송 오류: {e}")
                break

        return False

    async def send_buy_signal(
        self, ticker: str, price: float, change: float, reason: str
    ) -> bool:
        html = (
            f"🟢 <b>BUY 신호</b>\n"
            f"종목: <code>{ticker}</code>\n"
            f"현재가: <b>{price:,.2f}</b>\n"
            f"변동: {change:+.2f}%\n"
            f"이유: {reason}\n"
            f"시간: {datetime.now().strftime('%H:%M:%S')}"
        )
        return await self.send_message(html)

    async def send_sell_signal(
        self, ticker: str, price: float, change: float, reason: str
    ) -> bool:
        html = (
            f"🔴 <b>SELL 신호</b>\n"
            f"종목: <code>{ticker}</code>\n"
            f"현재가: <b>{price:,.2f}</b>\n"
            f"변동: {change:+.2f}%\n"
            f"이유: {reason}\n"
            f"시간: {datetime.now().strftime('%H:%M:%S')}"
        )
        return await self.send_message(html)

    async def send_wait_signal(
        self, ticker: str, price: float, change: float, reason: str
    ) -> bool:
        html = (
            f"🟡 <b>WAIT 신호</b>\n"
            f"종목: <code>{ticker}</code>\n"
            f"현재가: <b>{price:,.2f}</b>\n"
            f"변동: {change:+.2f}%\n"
            f"이유: {reason}\n"
            f"시간: {datetime.now().strftime('%H:%M:%S')}"
        )
        return await self.send_message(html)

    async def send_alert(self, ticker: str, title: str, message: str) -> bool:
        html = f"⚠️ <b>{title}</b>\n<code>{ticker}</code>\n{message}"
        return await self.send_message(html)


class AnalysisWorker:
    """Redis 구독 및 실시간 분석"""

    def __init__(
        self,
        redis_host: str = REDIS_HOST,
        redis_port: int = REDIS_PORT,
        telegram_bot_token: str = "",
        telegram_chat_id: str = "",
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis = None
        self.pubsub = None
        self.running = False

        # Telegram 알림 설정
        self.notifier = TelegramNotifier(telegram_bot_token, telegram_chat_id)

        # 분석 상태 캐시
        self.last_analysis = {}  # {ticker: {last_price, last_signal, timestamp}}

    async def connect(self) -> bool:
        """Redis 연결"""
        try:
            if not HAS_REDIS:
                logger.error("aioredis 미설치 — Redis 서버와 통신 불가")
                return False

            self.redis = await aioredis.from_url(
                f"redis://{self.redis_host}:{self.redis_port}/{REDIS_DB}",
                decode_responses=True,
            )
            await self.redis.ping()
            logger.info(f"✓ Redis 연결 성공 ({self.redis_host}:{self.redis_port})")
            return True
        except Exception as e:
            logger.error(f"Redis 연결 실패: {e}")
            return False

    async def disconnect(self):
        """Redis 및 HTTP 세션 해제"""
        if self.pubsub:
            try:
                await self.pubsub.unsubscribe()
                await self.pubsub.reset()
            except Exception:
                pass
        if self.redis:
            await self.redis.aclose()
            logger.info("Redis 연결 해제")
        await self.notifier.close()

    async def subscribe(self) -> bool:
        """채널 구독"""
        try:
            self.pubsub = self.redis.pubsub()
            await self.pubsub.subscribe(MARKET_UPDATES_CHANNEL)
            logger.info(f"✓ 채널 구독 성공: {MARKET_UPDATES_CHANNEL}")
            return True
        except Exception as e:
            logger.error(f"채널 구독 실패: {e}")
            return False

    async def analyze_update(self, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        시장 데이터 분석.
        
        Args:
            update_data: 시장 데이터
            {
                "ticker": "005930.KS",
                "price": 70000,
                "change": 2.5,
                "volume": 1000000,
                ...
            }
        
        Returns:
            분석 결과
            {
                "signal": "BUY" | "SELL" | "HOLD",
                "confidence": 0.0~1.0,
                "reason": "...",
                "technicals": {...}
            }
        """
        try:
            ticker = update_data.get("ticker", "")
            price = update_data.get("price", 0)
            change = update_data.get("change", 0)
            volume = update_data.get("volume", 0)

            if not HAS_STOCK_AI:
                logger.warning("stock_ai.py 분석 불가 — 기본 로직 사용")
                # 간단한 기본 분석 (변동률 기반)
                if change > PRICE_CHANGE_THRESHOLD:
                    return {
                        "signal": "BUY",
                        "confidence": 0.5,
                        "reason": f"급상승 ({change:+.2f}%)",
                    }
                elif change < -PRICE_CHANGE_THRESHOLD:
                    return {
                        "signal": "SELL",
                        "confidence": 0.5,
                        "reason": f"급하락 ({change:+.2f}%)",
                    }
                else:
                    return {
                        "signal": "HOLD",
                        "confidence": 0.0,
                        "reason": "변동 없음",
                    }

            # stock_ai.py의 분석 함수 호출
            logger.info(f"📊 분석 시작: {ticker}")

            try:
                # 기술적 지표 분석
                stock_data = get_stock_data(ticker, period="1mo")
                if stock_data is not None and not stock_data.empty:
                    technicals = analyze_technicals(ticker)
                else:
                    technicals = {}

                # 감정 분석
                sentiment = get_advanced_sentiment(ticker)

                # 신호 결정 (예시 로직)
                signal = "HOLD"
                confidence = 0.0
                reason = "분석 중"

                # RSI 기반 신호
                if technicals.get("rsi", 50) > 70:
                    signal = "SELL"
                    confidence = 0.7
                    reason = "RSI 과매수"
                elif technicals.get("rsi", 50) < 30:
                    signal = "BUY"
                    confidence = 0.7
                    reason = "RSI 과매도"

                # 감정 분석 반영
                if sentiment and sentiment.get("score", 0) > 2:
                    signal = "BUY"
                    confidence = min(confidence + 0.2, 1.0)
                    reason += " + 긍정 뉴스"

                result = {
                    "signal": signal,
                    "confidence": confidence,
                    "reason": reason,
                    "technicals": technicals,
                    "sentiment": sentiment,
                }

                logger.info(f"✓ 분석 완료: {ticker} → {signal}")
                return result

            except Exception as e:
                logger.warning(f"분석 오류 [{ticker}]: {e}")
                return {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "reason": f"분석 오류: {e}",
                }

        except Exception as e:
            logger.error(f"분석 프로세싱 오류: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "오류 발생",
            }

    async def process_message(self, message: str):
        """
        수신 메시지 처리 및 분석.
        
        Args:
            message: JSON 형식의 시장 데이터
        """
        try:
            # JSON 파싱
            update_data = json.loads(message)
            ticker = update_data.get("ticker", "?")
            price = update_data.get("price", 0)
            change = update_data.get("change", 0)

            logger.debug(f"📥 메시지 수신: {ticker} {price} ({change:+.2f}%)")

            # 분석 실행
            analysis = await self.analyze_update(update_data)
            signal = analysis.get("signal", "HOLD")
            confidence = analysis.get("confidence", 0)
            reason = analysis.get("reason", "")

            logger.info(
                f"🔍 분석 결과: {ticker} → {signal} "
                f"(신뢰도: {confidence:.0%}) | {reason}"
            )

            # 신호에 따른 알림 발송
            if signal == "BUY" and confidence >= 0.5:
                await self.notifier.send_buy_signal(ticker, price, change, reason)
            elif signal == "SELL" and confidence >= 0.5:
                await self.notifier.send_sell_signal(ticker, price, change, reason)
            else:
                # 큰 변동은 경고로 발송
                if abs(change) > PRICE_CHANGE_THRESHOLD:
                    await self.notifier.send_alert(
                        ticker,
                        "큰 변동 감지",
                        f"변동률: {change:+.2f}% | 현재가: {price:,.2f}"
                    )

            # 상태 캐시 업데이트
            self.last_analysis[ticker] = {
                "last_price": price,
                "last_signal": signal,
                "timestamp": datetime.now().isoformat(),
            }

        except json.JSONDecodeError:
            logger.warning(f"JSON 파싱 실패: {message[:100]}")
        except Exception as e:
            logger.error(f"메시지 처리 오류: {e}")

    async def listen(self):
        """메시지 수신 루프"""
        logger.info("👂 메시지 수신 대기 중...")
        self.running = True

        try:
            async for message in self.pubsub.listen():
                if not self.running:
                    break
                if message["type"] == "message":
                    await self.process_message(message["data"])
        except asyncio.CancelledError:
            logger.info("수신 작업 취소됨")
        except Exception as e:
            logger.error(f"수신 루프 오류: {e}")
        finally:
            self.running = False

    async def run(self):
        """
        분석 워커 메인 루프.
        
        1. Redis 연결
        2. 채널 구독
        3. 메시지 수신 및 분석
        """
        # Redis 연결
        if not await self.connect():
            logger.error("Redis 연결 실패 — 종료")
            return

        # 채널 구독
        if not await self.subscribe():
            logger.error("채널 구독 실패 — 종료")
            await self.disconnect()
            return

        try:
            # 메시지 수신 루프
            await self.listen()
        except KeyboardInterrupt:
            logger.info("사용자 인터럽트 감지")
        except Exception as e:
            logger.error(f"실행 오류: {e}")
        finally:
            await self.disconnect()
            logger.info("✓ 분석 워커 종료")


async def main():
    """엔트리 포인트"""
    import os

    logger.info("=" * 80)
    logger.info("🚀 Analysis Worker (Consumer) 시작")
    logger.info("=" * 80)

    redis_host = os.getenv("REDIS_HOST", REDIS_HOST)
    redis_port = int(os.getenv("REDIS_PORT", REDIS_PORT))
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not telegram_token or not telegram_chat_id:
        logger.warning("⚠️ Telegram 설정이 없음 — 알림 비활성화")
        logger.warning("  설정: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")

    worker = AnalysisWorker(
        redis_host=redis_host,
        redis_port=redis_port,
        telegram_bot_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
    )

    # Graceful shutdown 처리
    def signal_handler(sig, frame):
        logger.info("\n🛑 Shutdown 신호 수신")
        worker.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 실행
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
