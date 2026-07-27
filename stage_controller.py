"""
stage_controller.py
리니어 스테이지 컨트롤러 (Arduino Uno + MSD-224N + LSM1-NK235630-1610)
======================================================================

LinearStage_Commanded.ino 와 시리얼로 통신.

프로토콜
--------
  PC -> Uno:  "MOVE\\n" / "BACK\\n" / "HOME\\n" / "STOP\\n"
  Uno -> PC:  "OK:MOVE" / "OK:BACK" / ...   (동작 완료 시)

설계
----
  · hand_controller.AmazingHandController 와 동일한 패턴
    (connect/disconnect, event_callback 으로 Teensy 마커 동기화)
  · 모든 동작은 Uno 의 OK 응답을 기다리는 blocking 방식
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

try:
    import serial
except ImportError as exc:
    raise ImportError("pyserial 가 필요합니다: pip install pyserial") from exc

import config

LOG = logging.getLogger(__name__)


class StageController:
    """리니어 스테이지 제어 (Arduino Uno 시리얼)."""

    def __init__(
        self,
        port: str = config.STAGE_PORT,
        baudrate: int = config.STAGE_BAUDRATE,
        event_callback: Optional[Callable[[str], None]] = None,
        timeout: float = 0.5,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.event_callback = event_callback
        self.timeout = timeout
        self._serial: Optional[serial.Serial] = None

    # ----- 연결 -----
    def connect(self) -> None:
        LOG.info("리니어 스테이지 연결: %s @ %d bps", self.port, self.baudrate)
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(2.0)            # Uno 부팅(자동 리셋) 대기
        self._serial.reset_input_buffer()
        LOG.info("리니어 스테이지 연결 완료")

    def disconnect(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        LOG.info("리니어 스테이지 연결 해제")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    # ----- 마커 -----
    def _mark(self, name: str) -> None:
        if self.event_callback is not None:
            try:
                self.event_callback(name)
            except Exception as exc:
                LOG.warning("event_callback(%s) 실패: %s", name, exc)

    # ----- 저수준: 명령 송신 + OK 대기 -----
    def _send_wait(self, cmd: str, expect: str, max_wait: float = 20.0) -> bool:
        if self._serial is None:
            raise RuntimeError("connect() 먼저 호출하세요.")
        self._serial.reset_input_buffer()
        self._serial.write(f"{cmd}\n".encode("ascii"))
        self._serial.flush()
        t0 = time.time()
        while time.time() - t0 < max_wait:
            line = self._serial.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("#"):
                LOG.debug("Stage: %s", line)
                continue
            if line == expect:
                return True
            if line.startswith("ERR"):
                LOG.error("Stage 오류: %s", line)
                return False
        LOG.error("Stage 응답 timeout: %s (기대 %s)", cmd, expect)
        return False

    # ----- 고수준: phase 동작 -----
    def move(self) -> bool:
        """물체 방향으로 전진 (action 거리까지)."""
        self._mark("stage_move_start")
        ok = self._send_wait("MOVE", "OK:MOVE")
        self._mark("stage_move_done")
        return ok

    def back(self) -> bool:
        """출발점으로 복귀."""
        self._mark("stage_back_start")
        ok = self._send_wait("BACK", "OK:BACK")
        self._mark("stage_back_done")
        return ok

    def home(self) -> bool:
        """원점 확인(소프트)."""
        self._mark("stage_home")
        return self._send_wait("HOME", "OK:HOME")

    def stop(self) -> bool:
        return self._send_wait("STOP", "OK:STOP", max_wait=3.0)
