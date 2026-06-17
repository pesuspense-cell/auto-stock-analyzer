"""bootstrap.py — 기존 비즈니스 로직(stock_ai.py, src/*) 재사용을 위한 경로 설정.

FastAPI 백엔드는 web/backend/ 하위에 격리돼 있으나, 핵심 로직은 레포 루트의
`stock_ai.py` 와 `src/` 패키지에 그대로 존재한다. 이 모듈을 가장 먼저 import 하여
레포 루트를 sys.path 에 등록하면, 코드 복제 없이 기존 순수 함수를 직접 호출할 수 있다.

stock_ai.py 는 streamlit 의존이 전혀 없으므로 서버 환경에서 안전하게 import 된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

# web/backend/app/bootstrap.py → parents[3] == 레포 루트
_REPO_ROOT = Path(__file__).resolve().parents[3]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

REPO_ROOT = _REPO_ROOT
