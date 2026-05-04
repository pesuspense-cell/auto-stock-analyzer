"""
data_ingestor.py - 실시간 데이터 수집기 (Producer)

역할:
  - 거래소 WebSocket 또는 실시간 API에서 시세 데이터 수집
  - 수집된 데이터를 파싱 및 정규화
  - Redis Pub/Sub으로 즉시 발행
  - 연결 끊김 시 자동 재접속

아키텍처:
  WebSocket (거래소) → data_ingestor (Parser/Normalizer)
                     → Redis Pub/Sub (market_updates)
                     → analysis_worker (Consumer)
"""
import asyncio
import json
import logging
import os
import time
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# ─── 로깅 설정 ─────────────────────────────────────────────────────────────
logger = logging.getLogger("data_ingestor")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] data_ingestor: %(message)s",
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

# app.py와 동일한 경로의 관심종목 파일
WATCHLIST_FILE = Path(__file__).parent / "watchlist.json"
WATCHLIST_RELOAD_SEC = 60  # 1분마다 재로드

# 관심종목 없을 때 기본 종목
WATCH_TICKERS_DEFAULT = [
    "005930.KS",  # 삼성전자
    "000660.KS",  # SK하이닉스
    "005380.KS",  # 현대차
    "035720.KS",  # 카카오
]

# WebSocket 재접속 설정
MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds
HEARTBEAT_INTERVAL = 30  # seconds


def load_watchlist_tickers(path: Path = WATCHLIST_FILE) -> List[str]:
    """watchlist.json에서 ticker 목록 로드. 없으면 기본 종목 반환."""
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                wl = json.load(f)
            tickers = [item["ticker"] for item in wl if item.get("ticker")]
            if tickers:
                return tickers
    except Exception as e:
        logger.warning(f"watchlist.json 로드 실패: {e}")
    return list(WATCH_TICKERS_DEFAULT)


@dataclass
class MarketUpdate:
    """시장 데이터 표준 포맷"""
    ticker: str
    timestamp: str  # ISO 8601
    price: float
    volume: int
    change: float  # 변동률 (%)
    high: float
    low: float
    open_: float
    market: str  # 'KRX' | 'NASDAQ' | 'NYSE' 등


class DataIngestor:
    """실시간 데이터 수집 및 Redis 발행"""

    def __init__(
        self,
        redis_host: str = REDIS_HOST,
        redis_port: int = REDIS_PORT,
        watch_tickers: list = None,
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        # watch_tickers가 명시적으로 전달되면 사용, 없으면 watchlist.json 우선
        self._override_tickers = watch_tickers
        self.watch_tickers = watch_tickers or load_watchlist_tickers()
        self.redis = None
        self.running = False
        self._retry_count = 0
        self._last_reload = time.monotonic()

    def _maybe_reload_watchlist(self):
        """WATCHLIST_RELOAD_SEC마다 watchlist.json을 다시 읽어 종목 목록 갱신."""
        if self._override_tickers:
            return  # 명시적으로 지정된 경우 갱신 안 함
        now = time.monotonic()
        if now - self._last_reload < WATCHLIST_RELOAD_SEC:
            return
        self._last_reload = now
        new_tickers = load_watchlist_tickers()
        if new_tickers != self.watch_tickers:
            added = set(new_tickers) - set(self.watch_tickers)
            removed = set(self.watch_tickers) - set(new_tickers)
            self.watch_tickers = new_tickers
            if added:
                logger.info(f"관심종목 추가: {sorted(added)}")
            if removed:
                logger.info(f"관심종목 제거: {sorted(removed)}")
            logger.info(f"현재 감시 종목 ({len(self.watch_tickers)}개): {self.watch_tickers}")

    async def connect(self):
        """Redis 연결"""
        try:
            if not HAS_REDIS:
                logger.error("aioredis 미설치 — redis 서버와 통신 불가")
                return False

            self.redis = await aioredis.from_url(
                f"redis://{self.redis_host}:{self.redis_port}/{REDIS_DB}",
                decode_responses=True,
            )
            await self.redis.ping()
            logger.info(f"✓ Redis 연결 성공 ({self.redis_host}:{self.redis_port})")
            self._retry_count = 0
            return True
        except Exception as e:
            logger.error(f"Redis 연결 실패: {e}")
            return False

    async def disconnect(self):
        """Redis 연결 해제"""
        if self.redis:
            await self.redis.aclose()
            logger.info("Redis 연결 해제")

    async def publish_update(self, update: MarketUpdate):
        """Redis Pub/Sub으로 시장 데이터 발행"""
        try:
            if not self.redis:
                logger.warning("Redis 연결 없음 — 데이터 발행 실패")
                return False

            # 메시지 포맷
            message = json.dumps(asdict(update), ensure_ascii=False)

            # Pub/Sub 발행
            await self.redis.publish(MARKET_UPDATES_CHANNEL, message)

            logger.debug(
                f"📤 발행 [{update.ticker}] {update.price:,.2f} "
                f"({update.change:+.2f}%) @ {update.timestamp}"
            )
            return True
        except Exception as e:
            logger.error(f"발행 오류 [{update.ticker}]: {e}")
            return False

    async def fetch_yfinance_update(self, ticker: str) -> Optional[MarketUpdate]:
        """yfinance로부터 단일 종목 데이터 조회 (시뮬레이션용)"""
        try:
            if not HAS_YFINANCE:
                logger.warning("yfinance 미설치")
                return None

            # yfinance에서 실시간 정보 조회 (지연이 있을 수 있음)
            data = yf.Ticker(ticker)
            info = data.info

            # 기본 정보 추출
            current = info.get("currentPrice", 0)
            high = info.get("fiftyTwoWeekHigh", 0)
            low = info.get("fiftyTwoWeekLow", 0)
            prev_close = info.get("previousClose", 0)
            volume = info.get("volume", 0)

            # 변동률 계산
            change = ((current - prev_close) / prev_close * 100) if prev_close else 0

            # 시장 결정
            market = self._determine_market(ticker)

            update = MarketUpdate(
                ticker=ticker,
                timestamp=datetime.utcnow().isoformat(),
                price=float(current),
                volume=int(volume),
                change=round(change, 2),
                high=float(high),
                low=float(low),
                open_=float(prev_close),
                market=market,
            )

            return update
        except Exception as e:
            logger.warning(f"yfinance 데이터 조회 실패 [{ticker}]: {e}")
            return None

    def _determine_market(self, ticker: str) -> str:
        """종목 코드로부터 시장 결정"""
        if ticker.endswith(".KS"):
            return "KRX"  # 한국거래소 유가증권시장
        elif ticker.endswith(".KQ"):
            return "KOSDAQ"  # 코스닥
        elif ticker in ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]:
            return "NASDAQ"
        else:
            return "NYSE"

    async def poll_data(self, interval: float = 5.0):
        """주기적으로 데이터 수집 및 발행 (시뮬레이션)"""
        logger.info(f"📊 데이터 폴링 시작 (간격: {interval}초)")
        logger.info(f"감시 종목 ({len(self.watch_tickers)}개): {self.watch_tickers}")
        self.running = True

        try:
            while self.running:
                self._maybe_reload_watchlist()
                for ticker in self.watch_tickers:
                    update = await self.fetch_yfinance_update(ticker)
                    if update:
                        await self.publish_update(update)
                    await asyncio.sleep(0.1)  # API 레이트 제한 회피

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("폴링 작업 취소됨")
        except Exception as e:
            logger.error(f"폴링 오류: {e}")
        finally:
            self.running = False

    async def simulate_websocket(self, interval: float = 5.0):
        """
        WebSocket 시뮬레이션 (거래소 실시간 API가 없을 때 테스트용)
        
        실제 환경에서는 다음과 같이 구현:
        - 한국거래소(KRX): KRX WebSocket API
        - NASDAQ: IEX Cloud, Finnhub 등 실시간 API
        """
        import random

        logger.info("🔄 WebSocket 시뮬레이션 시작")
        logger.info(f"감시 종목 ({len(self.watch_tickers)}개): {self.watch_tickers}")
        self.running = True

        try:
            base_prices = {ticker: random.uniform(10, 500) for ticker in self.watch_tickers}

            while self.running:
                self._maybe_reload_watchlist()
                # 새로 추가된 종목에 초기 가격 부여
                for ticker in self.watch_tickers:
                    if ticker not in base_prices:
                        base_prices[ticker] = random.uniform(10, 500)
                for ticker in self.watch_tickers:
                    # 가격 변동 시뮬레이션
                    base_prices[ticker] += random.uniform(-2, 2)
                    current_price = base_prices[ticker]
                    prev_price = base_prices[ticker] - random.uniform(-1, 1)

                    update = MarketUpdate(
                        ticker=ticker,
                        timestamp=datetime.utcnow().isoformat(),
                        price=round(current_price, 2),
                        volume=random.randint(100000, 10000000),
                        change=round((current_price - prev_price) / prev_price * 100, 2),
                        high=round(current_price + random.uniform(0, 5), 2),
                        low=round(current_price - random.uniform(0, 5), 2),
                        open_=round(prev_price, 2),
                        market=self._determine_market(ticker),
                    )

                    await self.publish_update(update)
                    await asyncio.sleep(0.1)

                logger.info(f"✓ 라운드 완료 ({len(self.watch_tickers)}종목)")
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("WebSocket 시뮬레이션 취소됨")
        except Exception as e:
            logger.error(f"WebSocket 시뮬레이션 오류: {e}")
        finally:
            self.running = False

    async def run(self, use_simulation: bool = True, interval: float = 5.0):
        """
        데이터 수집기 메인 루프.

        Args:
            use_simulation: True면 WebSocket 시뮬레이션, False면 yfinance 폴링
            interval: 데이터 수집 간격 (초)
        """
        # Redis 연결
        if not await self.connect():
            logger.error("Redis 연결 실패 — 종료")
            return

        try:
            if use_simulation:
                await self.simulate_websocket(interval)
            else:
                await self.poll_data(interval)
        except KeyboardInterrupt:
            logger.info("사용자 인터럽트 감지")
        except Exception as e:
            logger.error(f"실행 오류: {e}")
        finally:
            await self.disconnect()
            logger.info("✓ 데이터 수집기 종료")


async def main():
    """엔트리 포인트"""
    import os

    logger.info("=" * 80)
    logger.info("🚀 Data Ingestor (Producer) 시작")
    logger.info("=" * 80)

    redis_host = os.getenv("REDIS_HOST", REDIS_HOST)
    redis_port = int(os.getenv("REDIS_PORT", REDIS_PORT))
    interval = float(os.getenv("POLLING_INTERVAL", "60"))

    # 우선순위: watchlist.json > WATCH_TICKERS 환경변수 > 기본값
    tickers_from_wl = load_watchlist_tickers()
    watch_tickers_env = os.getenv("WATCH_TICKERS", "")
    env_tickers = [t.strip() for t in watch_tickers_env.split(",") if t.strip()]

    if tickers_from_wl != list(WATCH_TICKERS_DEFAULT):
        # watchlist.json에 실제 관심종목이 있는 경우
        watch_tickers = None  # DataIngestor가 직접 로드하게 놔둠
        logger.info(f"watchlist.json 사용 ({len(tickers_from_wl)}개)")
    elif env_tickers:
        watch_tickers = env_tickers
        logger.info(f"WATCH_TICKERS 환경변수 사용 ({len(env_tickers)}개)")
    else:
        watch_tickers = None
        logger.info(f"기본 종목 사용 ({len(WATCH_TICKERS_DEFAULT)}개)")

    ingestor = DataIngestor(
        redis_host=redis_host,
        redis_port=redis_port,
        watch_tickers=watch_tickers,
    )

    # Graceful shutdown 처리
    def signal_handler(sig, frame):
        logger.info("\n🛑 Shutdown 신호 수신")
        ingestor.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 실행
    await ingestor.run(use_simulation=True, interval=interval)


if __name__ == "__main__":
    asyncio.run(main())
