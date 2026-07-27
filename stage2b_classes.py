"""
stage2b_classes.py
Stage 2B 3-class 정의 (공용 모듈)
==================================

bio-surface 3 클래스: 사람 피부(skin) / Dragon Skin 10 실리콘(dragonskin) /
생닭(chicken). 촉각·음향으로는 구분이 어렵고, 후각(가스) 신호가 판별에 기여하는지
검증하는 것이 목표.

stage1_classes.py 와 동일 인터페이스(BY_OBJ_ID / BY_OBJ_NAME / resolve /
resolve_fuzzy)를 제공 → extract_features_v3.py `--stage 2b`, run_ml.py 무수정 동작.

> obj_id/obj_name 은 stage2b 수집 폴더명(obj01_skin/obj02_dragonskin/obj03_chicken)과 일치.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


_ORDER = ["skin", "dragonskin", "chicken"]


@dataclass(frozen=True)
class Stage2BClass:
    obj_id: int          # 1..3
    obj_name: str        # skin / dragonskin / chicken
    label: int           # 0..2 (ML 타깃)

    @property
    def group(self) -> str:
        return self.obj_name

    @property
    def material(self) -> str:
        return self.obj_name

    @property
    def content(self) -> Optional[str]:
        return None


def build_classes() -> List[Stage2BClass]:
    return [Stage2BClass(obj_id=i + 1, obj_name=name, label=i)
            for i, name in enumerate(_ORDER)]


CLASSES: List[Stage2BClass] = build_classes()

BY_OBJ_ID: Dict[int, Stage2BClass] = {c.obj_id: c for c in CLASSES}
BY_OBJ_NAME: Dict[str, Stage2BClass] = {c.obj_name: c for c in CLASSES}


def resolve(obj_id: int | None = None, obj_name: str | None = None) -> Optional[Stage2BClass]:
    if obj_id is not None and obj_id in BY_OBJ_ID:
        return BY_OBJ_ID[obj_id]
    if obj_name is not None and obj_name in BY_OBJ_NAME:
        return BY_OBJ_NAME[obj_name]
    return None


_ALIASES = {
    "skin": "skin", "human": "skin", "피부": "skin", "손": "skin",
    "dragonskin": "dragonskin", "dragon": "dragonskin", "silicone": "dragonskin",
    "실리콘": "dragonskin", "드래곤": "dragonskin",
    "chicken": "chicken", "raw": "chicken", "닭": "chicken", "생닭": "chicken",
}


def resolve_fuzzy(obj_id=None, obj_name=None, folder_name=None) -> Optional[Stage2BClass]:
    exact = resolve(obj_id, obj_name)
    if exact:
        return exact
    for raw in (s for s in (obj_name, folder_name) if s):
        s = str(raw).lower()
        # dragonskin 은 'skin' 을 포함하므로 먼저 검사(순서 중요)
        if "dragon" in s or "silicone" in s or "실리콘" in s or "드래곤" in s:
            return BY_OBJ_NAME.get("dragonskin")
        hit = next((v for k, v in _ALIASES.items() if k in s), None)
        if hit:
            return BY_OBJ_NAME.get(hit)
    return None


if __name__ == "__main__":
    print(f"{'obj_id':>6} {'label':>5}  {'obj_name'}")
    for c in CLASSES:
        print(f"{c.obj_id:>6} {c.label:>5}  {c.obj_name}")
