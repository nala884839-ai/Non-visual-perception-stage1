"""
closedloop_hand_controller.py
WowSkin 힘 피드백 기반 Closed-loop Press 컨트롤러
====================================================

목적
----
  Open-loop press (시간 기반, 고정 각도) 의 한계 극복:
    - 부드러운 물체(스펀지) vs 딱딱한 물체(알루미늄) → 같은 각도여도 실제 힘 다름
    - press 피처가 물체 강성(stiffness)을 섞어서 반영

  Closed-loop press 는 WowSkin 으로 측정한 force proxy 가 임계값에
  도달하는 순간 즉시 검지를 정지 → **모든 물체에 동일 힘** 인가.

설계 방침
---------
  - AmazingHandController 를 상속하여 home/hover/sniff/safe/tap/rub 동작은 그대로 사용.
  - 새 메서드 press_closedloop() 만 추가 — 기존 open-loop press() 와 분리.
  - WowSkinForceProvider 를 외부에서 주입 (의존성 역전).

알고리즘
--------
  1. pre_action_quiet_ms 만큼 무음 대기 → baseline 측정
  2. step-wise descent: 일정 간격으로 검지 각도를 점진적 증가
     매 step 후 force_provider.get_force() 폴링
  3. force ≥ target_threshold → 즉시 정지 (현재 각도에서 hold)
     descend_max_a1/a2 도달 → 강제 정지 (안전)
     max_descend_ms timeout → 강제 정지 (force 미도달도 hold 진행)
  4. hold_ms 동안 그 자리에 유지
  5. release: hover 각도로 복귀
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import config
from hand_controller import AmazingHandController
from wowskin_force_provider import WowSkinForceProvider

LOG = logging.getLogger(__name__)


class ClosedLoopHandController(AmazingHandController):
    """AmazingHandController + WowSkin force feedback closed-loop press."""

    def __init__(
        self,
        force_provider: WowSkinForceProvider,
        port: str = config.HAND_PORT,
        baudrate: int = config.HAND_BAUDRATE,
        side: int = config.HAND_SIDE,
        event_callback=None,
    ) -> None:
        super().__init__(
            port=port,
            baudrate=baudrate,
            side=side,
            event_callback=event_callback,
        )
        self.force = force_provider

    # -------------------------------------------------------------------------
    # Closed-loop press
    # -------------------------------------------------------------------------

    def press_closedloop(
        self,
        params: "config.ClosedLoopPressParams" = config.CLOSEDLOOP_PRESS,
        target_force: Optional[float] = None,
    ) -> dict:
        """WowSkin 피드백 기반 press.

        Parameters
        ----------
        params : ClosedLoopPressParams
            descend / hold / release 타이밍 + 각도 한계.
        target_force : Optional[float]
            덮어쓸 목표 힘. None 이면 params.target_force_threshold 사용.

        Returns
        -------
        dict : 결과 메트릭
          {
            "target_force":    float,
            "final_force":     float,
            "final_a1":        float,
            "final_a2":        float,
            "contact_reached": bool,
            "descend_ms":      float,
            "n_steps":         int,
          }
        """
        threshold = target_force if target_force is not None else params.target_force_threshold

        self._mark("press_phase_start")

        # ── 1) 무음 대기 (서보 진동 감쇠) ───────────────────────────────────────
        self._quiet_wait("press")

        # ── 2) Baseline 자기장 측정 (검지가 hover 정지 상태) ─────────────────
        self._mark("baseline_measure_start")
        self.force.calibrate_baseline(duration_s=params.baseline_ms / 1000.0)
        self._mark("baseline_measure_end")

        # ── 3) Step-wise descent + force monitoring ─────────────────────────
        self._mark(f"descent_start_target={threshold:.1f}")
        a1, a2 = config.POSE_HOVER["index"]   # 시작: hover 각도
        speed = config.SERVO_SPEEDS["press"]
        step_s = params.descend_step_interval_ms / 1000.0
        timeout_s = params.max_descend_ms / 1000.0

        n_steps = 0
        contact_reached = False
        t_descend_start = time.time()
        final_force = 0.0

        while True:
            elapsed = time.time() - t_descend_start
            if elapsed >= timeout_s:
                self._mark("descent_timeout")
                LOG.warning("press_closedloop: max_descend_ms 초과 — force 미도달, hold 강행")
                break

            # 한 step 만큼 더 굽히기 (검지가 더 안쪽으로)
            a1 += params.descend_step_deg
            a2 -= params.descend_step_deg

            # 각도 안전 한계 도달 → 정지
            if a1 >= params.descend_max_a1 or a2 <= params.descend_max_a2:
                a1 = min(a1, params.descend_max_a1)
                a2 = max(a2, params.descend_max_a2)
                self._move_finger("index", a1, a2, speed)
                time.sleep(step_s)
                self._mark(f"descent_max_angle_a1={a1:.1f}_a2={a2:.1f}")
                LOG.warning("press_closedloop: 안전 한계 각도 도달 — 강제 정지")
                n_steps += 1
                final_force = self.force.get_force()
                break

            self._move_finger("index", a1, a2, speed)
            time.sleep(step_s)
            n_steps += 1

            # Force 측정
            f = self.force.get_force()
            final_force = f

            if f >= threshold:
                self._mark(f"contact_detected_force={f:.1f}_steps={n_steps}")
                LOG.info("press_closedloop: 목표 힘 도달  force=%.1f / %.1f, steps=%d, a1=%.1f",
                         f, threshold, n_steps, a1)
                contact_reached = True
                break

        # ── 4) Hold (목표 힘 도달 각도에서 유지) ─────────────────────────────
        descend_ms = (time.time() - t_descend_start) * 1000.0
        self._mark("press_hold_start")
        time.sleep(params.hold_ms / 1000.0)
        self._mark("press_hold_end")

        # ── 5) Release (hover 로 복귀) ───────────────────────────────────────
        self._mark("press_release")
        hover_a1, hover_a2 = config.POSE_HOVER["index"]
        self._move_finger("index", hover_a1, hover_a2, speed)
        time.sleep(params.release_ms / 1000.0)

        self._mark("press_phase_end")

        return {
            "target_force":    threshold,
            "final_force":     final_force,
            "final_a1":        a1,
            "final_a2":        a2,
            "contact_reached": contact_reached,
            "descend_ms":      descend_ms,
            "n_steps":         n_steps,
        }
