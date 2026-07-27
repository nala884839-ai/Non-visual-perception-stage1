"""
sensor_reader.py
Teensy 4.1 센서 스트림 수신 + CSV 저장 + 이벤트 마커 송신
===========================================================

teensy_sensor_logger.ino 의 출력 포맷을 그대로 파싱합니다.

수신 라인 포맷
--------------
  TAC,<t_us>,<s1>,<s2>,<s3>,<anyTouch>,<Cx>,<Cy>      → tactile.csv  (SingleTact 압력)
  BU,<t_us>,<N>,<v0>..<v(N-1)>                        → acoustic.csv (BU 진동 raw 버스트)
  ENS,<t_us>,<tvoc>,<eco2>,<aqi>                     → gas_ens160.csv
  BME688,<t_us>,<step>,<set_temp>,<gas_R>,<temp>,<humid>,<press>,<gas_valid> → gas_bme688.csv
  BME280,<t_us>,<temp>,<humid>,<press>               → environment.csv
  EVT,<t_us>,<phase_name>                            → events.csv
  # ...                                               → 주석 (READY / OK / ERROR 등)

송신 명령
---------
  send_marker("<phase_name>") → Teensy 에 "M:<phase_name>\n" 전송
  → Teensy 가 EVT 라인 에코 → PC 가 events.csv 에 기록

사용 예
-------
  reader = TeensySensorReader(output_dir=Path("data/raw/.../trial_001"))
  reader.start()
  try:
      reader.send_marker("baseline_start")
      time.sleep(5)
      reader.send_marker("hover_start")
      ...
  finally:
      reader.stop()
"""

from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, IO, Optional

try:
    import serial
except ImportError as exc:
    raise ImportError(
        "pyserial 가 설치되어 있지 않습니다.\n"
        "    pip install pyserial"
    ) from exc

import config

LOG = logging.getLogger(__name__)


# =============================================================================
# CSV 헤더 정의
# =============================================================================

_CSV_HEADERS: Dict[str, list] = {
    "tactile":     ["t_us", "s1", "s2", "s3", "any_touch", "cx", "cy"],
    "acoustic":    ["t_us", "n_samples", "values"],
    "gas_ens160":  ["t_us", "tvoc_ppb", "eco2_ppm", "aqi"],
    "gas_bme688":  ["t_us", "heater_step", "set_temp_c", "gas_resistance_ohm",
                    "temp_c", "humid_pct", "press_hpa", "gas_valid"],
    "environment": ["t_us", "temp_c", "humid_pct", "press_hpa"],
    "events":      ["t_us", "phase_name"],
}


# =============================================================================
# Reader
# =============================================================================

class TeensySensorReader:
    """Teensy USB Serial 스트림을 별도 스레드로 수신 → 센서별 CSV 로 분배."""

    def __init__(
        self,
        output_dir: Path,
        port: str = config.TEENSY_PORT,
        baudrate: int = config.TEENSY_BAUDRATE,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.port = port
        self.baudrate = baudrate

        self._serial: Optional[serial.Serial] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._tx_queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()

        self._files: Dict[str, IO] = {}
        self._writers: Dict[str, csv.writer] = {}
        self._lock = threading.Lock()
        self._error_count = 0

        # 최신 TAC(촉각) 값 캐시 (피드백 용도)
        # value = (t_us, s1, s2, s3, any_touch, cx, cy)
        self._latest_tac: Optional[tuple] = None
        self._tac_lock = threading.Lock()

    # -------------------------------------------------------------------------
    # 라이프사이클
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """시리얼 연결 + CSV 파일 열기 + 수신 스레드 시작."""
        LOG.info("Teensy 연결: %s @ %d bps", self.port, self.baudrate)
        self._serial = serial.Serial(self.port, self.baudrate, timeout=0.05)
        time.sleep(0.5)  # Teensy USB bringup
        self._serial.reset_input_buffer()

        for name, header in _CSV_HEADERS.items():
            path = self.output_dir / f"{name}.csv"
            fp = open(path, "w", newline="", encoding="utf-8")
            w = csv.writer(fp)
            w.writerow(header)
            self._files[name] = fp
            self._writers[name] = w

        self._stop.clear()
        self._rx_thread = threading.Thread(target=self._run_loop, name="TeensyRx", daemon=True)
        self._rx_thread.start()
        LOG.info("센서 수신 스레드 시작 (output=%s)", self.output_dir)

    def stop(self) -> None:
        """수신 스레드 종료 + 파일 flush/close + 시리얼 close."""
        self._stop.set()
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=2.0)

        with self._lock:
            for fp in self._files.values():
                try:
                    fp.flush()
                    fp.close()
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("CSV close 실패: %s", exc)
            self._files.clear()
            self._writers.clear()

        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:  # noqa: BLE001
                pass
            self._serial = None

        LOG.info("센서 수신 종료 (파싱 오류 %d 건)", self._error_count)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # -------------------------------------------------------------------------
    # 이벤트 마커 송신 (메인 스레드에서 호출)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # 최신 TAC 값 조회 (closed-loop 피드백 용도)
    # -------------------------------------------------------------------------

    def get_latest_tac(self) -> Optional[tuple]:
        """가장 최근 수신된 촉각 값 사본을 반환.

        Returns
        -------
        Optional[(t_us, s1, s2, s3, any_touch, cx, cy)]
            아직 수신 전이면 None.
        """
        with self._tac_lock:
            return self._latest_tac

    def send_marker(self, phase_name: str) -> None:
        """Teensy 에 'M:<phase>\\n' 송신 → Teensy 가 EVT 라인으로 에코."""
        if self._serial is None:
            LOG.warning("Teensy 미연결 — 마커 무시: %s", phase_name)
            return
        line = f"M:{phase_name}\n".encode("ascii", errors="ignore")
        try:
            self._serial.write(line)
            self._serial.flush()
        except Exception as exc:  # noqa: BLE001
            LOG.error("마커 송신 실패(%s): %s", phase_name, exc)

    # -------------------------------------------------------------------------
    # 수신 루프
    # -------------------------------------------------------------------------

    def _run_loop(self) -> None:
        assert self._serial is not None
        buf = bytearray()

        while not self._stop.is_set():
            try:
                chunk = self._serial.read(4096)
            except Exception as exc:  # noqa: BLE001
                LOG.error("시리얼 read 실패: %s", exc)
                break

            if not chunk:
                continue

            buf.extend(chunk)
            # 라인 단위 파싱
            while b"\n" in buf:
                line_bytes, _, rest = buf.partition(b"\n")
                buf = bytearray(rest)
                line = line_bytes.decode("ascii", errors="ignore").strip()
                if not line:
                    continue
                self._dispatch_line(line)

    def _dispatch_line(self, line: str) -> None:
        # 주석 / 상태 메시지
        if line.startswith("#"):
            LOG.debug("Teensy: %s", line)
            return

        # 타입 판별 후 해당 CSV 에 기록
        try:
            tag, rest = line.split(",", 1)
        except ValueError:
            self._error_count += 1
            return

        fields = rest.split(",")
        try:
            if tag == "TAC" and len(fields) == 7:
                self._write("tactile", fields)
                # 최신 촉각 캐시 갱신: t_us,s1,s2,s3,any,cx,cy
                try:
                    t_us = int(fields[0])
                    s1 = int(fields[1]); s2 = int(fields[2]); s3 = int(fields[3])
                    any_touch = int(fields[4])
                    cx = float(fields[5]); cy = float(fields[6])
                    with self._tac_lock:
                        self._latest_tac = (t_us, s1, s2, s3, any_touch, cx, cy)
                except (ValueError, IndexError):
                    pass
            elif tag == "BU":
                # t_us,N,v0..v(N-1)  (단일 채널 진동 raw 버스트)
                if len(fields) >= 2:
                    t_us, n_str = fields[0], fields[1]
                    n = int(n_str)
                    expected = 2 + n
                    if len(fields) == expected:
                        vals = " ".join(fields[2 : 2 + n])
                        self._write("acoustic", [t_us, n_str, vals])
                    else:
                        self._error_count += 1
            elif tag == "ENS" and len(fields) == 4:
                self._write("gas_ens160", fields)
            elif tag == "BME688" and len(fields) == 8:
                self._write("gas_bme688", fields)
            elif tag == "BME280" and len(fields) == 4:
                self._write("environment", fields)
            elif tag == "EVT" and len(fields) == 2:
                self._write("events", fields)
                LOG.info("EVT: %s @ %s us", fields[1], fields[0])
            else:
                self._error_count += 1
        except Exception as exc:  # noqa: BLE001
            LOG.warning("라인 파싱 실패 (%s): %s", line[:60], exc)
            self._error_count += 1

    def _write(self, name: str, row: list) -> None:
        with self._lock:
            writer = self._writers.get(name)
            if writer is None:
                return
            writer.writerow(row)
