"""
trial_runner.py
⑤ 1-Trial 프로토콜 3종 (Hand + Sensor 통합)
==============================================

Trial 구조
----------
  variant="tap"           : baseline → move → hover → sniff → tap                 → hover → back → recovery
  variant="tap_press"     : baseline → move → hover → sniff → tap → press        → hover → back → recovery
  variant="tap_press_rub" : baseline → move → hover → sniff → tap → press → rub  → hover → back → recovery

  move/back = 리니어 스테이지(Uno) 전진/복귀

각 phase 시작/끝마다 Teensy 에 이벤트 마커를 송신하여 센서 신호와
손 동작을 시간 동기화합니다.

사용법
------
  runner = TrialRunner(
      session="session_001",
      obj_id=1, obj_name="aluminum",
      trial_id=1, variant="tap_press_rub",
  )
  runner.run()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import config
from hand_controller import AmazingHandController
from sensor_reader import TeensySensorReader
from stage_controller import StageController

LOG = logging.getLogger(__name__)


# =============================================================================
# Metadata
# =============================================================================

@dataclass
class TrialMetadata:
    session: str
    obj_id: int
    obj_name: str
    trial_id: int
    variant: str
    start_iso: str = ""
    end_iso: str = ""
    phases: List[str] = field(default_factory=list)
    hand_side: int = config.HAND_SIDE
    hand_port: str = config.HAND_PORT
    teensy_port: str = config.TEENSY_PORT
    notes: str = ""


# =============================================================================
# Trial Runner
# =============================================================================

class TrialRunner:
    """1 trial 단위 실행기 (Hand + Teensy 통합)."""

    def __init__(
        self,
        session: str,
        obj_id: int,
        obj_name: str,
        trial_id: int,
        variant: str,
        data_root: str = config.DATA_ROOT,
        timing: config.PhaseTiming = config.PHASE_TIMING,
        dry_run: bool = False,
    ) -> None:
        if variant not in config.TRIAL_VARIANTS:
            raise ValueError(
                f"variant 는 {list(config.TRIAL_VARIANTS)} 중 하나여야 합니다: {variant}"
            )

        self.meta = TrialMetadata(
            session=session,
            obj_id=obj_id,
            obj_name=obj_name,
            trial_id=trial_id,
            variant=variant,
            phases=config.TRIAL_VARIANTS[variant],
        )
        self.timing = timing
        self.dry_run = dry_run

        # 저장 경로 생성
        rel_path = config.TRIAL_DIR_TEMPLATE.format(
            session=session,
            obj_id=obj_id,
            obj_name=obj_name,
            trial_id=trial_id,
            variant=variant,
        )
        self.trial_dir = Path(data_root) / rel_path
        self.trial_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 실행
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """1 trial 전체 실행."""
        LOG.info("=" * 70)
        LOG.info("Trial 시작: obj%02d_%s trial_%03d variant=%s",
                 self.meta.obj_id, self.meta.obj_name,
                 self.meta.trial_id, self.meta.variant)
        LOG.info("phases: %s", " → ".join(self.meta.phases))
        LOG.info("저장 경로: %s", self.trial_dir)
        LOG.info("=" * 70)

        self.meta.start_iso = datetime.now().isoformat()

        if self.dry_run:
            LOG.warning("[dry-run] 센서/손 연결 없이 시뮬레이션")
            self._simulate()
        else:
            self._run_real()

        self.meta.end_iso = datetime.now().isoformat()
        self._save_meta()
        LOG.info("Trial 완료: %s", self.trial_dir)

    # -------------------------------------------------------------------------
    # 실제 실행 (Hand + Teensy)
    # -------------------------------------------------------------------------

    def _run_real(self) -> None:
        with TeensySensorReader(output_dir=self.trial_dir) as sensors:
            # 센서 스트림 안정화 시간
            LOG.info("센서 버퍼 안정화 (1s)…")
            time.sleep(1.0)

            # 이벤트 콜백으로 Teensy 마커 송신 연결
            hand = AmazingHandController(
                port=config.HAND_PORT,
                side=config.HAND_SIDE,
                event_callback=sensors.send_marker,
            )
            stage = StageController(
                port=config.STAGE_PORT,
                event_callback=sensors.send_marker,
            )
            hand.connect()
            stage.connect()
            try:
                self._execute_phases(hand, stage, sensors)
            finally:
                # 종료: 손은 편 자세 유지(토크 유지), 스테이지 연결 해제
                try:
                    hand.disconnect()
                finally:
                    stage.disconnect()

    def _execute_phases(
        self,
        hand: AmazingHandController,
        stage: StageController,
        sensors: TeensySensorReader,
    ) -> None:
        t_trial_start = time.time()

        for phase in self.meta.phases:
            if time.time() - t_trial_start > config.SAFETY_LIMITS.max_trial_seconds:
                LOG.error("max_trial_seconds 초과 — 중단")
                break

            sensors.send_marker(f"{phase}_start")
            LOG.info(">> phase=%s", phase)

            if phase == "baseline":
                time.sleep(self.timing.baseline)

            elif phase == "move":
                # 스테이지가 물체 방향으로 전진 (Uno blocking)
                stage.move()
                time.sleep(self.timing.move)

            elif phase == "back":
                # 스테이지가 출발점으로 복귀
                stage.back()
                time.sleep(self.timing.back)

            elif phase == "hover":
                hand.hover()
                # 포즈 이동 후 남은 시간 대기
                time.sleep(max(0.0, self.timing.hover - config.SAFETY_LIMITS.settle_ms / 1000.0))

            elif phase == "sniff":
                hand.sniff()
                time.sleep(max(0.0, self.timing.sniff - config.SAFETY_LIMITS.settle_ms / 1000.0))

            elif phase == "tap":
                t_phase_start = time.time()
                # tap() 내부에서 pre_action_quiet_ms 만큼 무음 대기 후 3회 tap
                hand.tap()
                elapsed = time.time() - t_phase_start
                remaining = self.timing.tap - elapsed
                if remaining > 0:
                    LOG.info("   tap 완료 후 hover 유지 %.2fs (후처리 감쇠)", remaining)
                    time.sleep(remaining)

            elif phase == "press":
                hand.press()

            elif phase == "rub":
                t_phase_start = time.time()
                # rub() 내부에서 pre_action_quiet_ms 만큼 무음 대기 후 5 사이클
                hand.rub()
                elapsed = time.time() - t_phase_start
                remaining = self.timing.rub - elapsed
                if remaining > 0:
                    LOG.info("   rub 완료 후 hover 유지 %.2fs (후처리 감쇠)", remaining)
                    time.sleep(remaining)

            elif phase == "recovery":
                # 손가락 모두 편 상태(hover=네 손가락 폄)로 종료. safe(접기) 안 함.
                hand.hover()
                time.sleep(max(0.0, self.timing.recovery - config.SAFETY_LIMITS.settle_ms / 1000.0))

            else:
                LOG.warning("알 수 없는 phase: %s (skip)", phase)

            sensors.send_marker(f"{phase}_end")

    # -------------------------------------------------------------------------
    # Dry-run 시뮬레이션
    # -------------------------------------------------------------------------

    def _simulate(self) -> None:
        """하드웨어 없이 phase 순서/타이밍만 검증."""
        for phase in self.meta.phases:
            LOG.info("[sim] phase=%s duration=%.1fs",
                     phase, getattr(self.timing, phase, 0.0))
            time.sleep(getattr(self.timing, phase, 0.0))

    # -------------------------------------------------------------------------
    # 메타 저장
    # -------------------------------------------------------------------------

    def _save_meta(self) -> None:
        path = self.trial_dir / "meta.json"
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(asdict(self.meta), fp, indent=2, ensure_ascii=False)
        LOG.info("meta.json 저장: %s", path)
