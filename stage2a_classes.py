"""
stage2a_classes.py
Stage 2A 8-class 정의 (공용 모듈)
==================================

과일 4 (orange/apple/pear/grapefruit) + 공 4 (pingpong/golf/tennis/baseball)
= 8 클래스. 촉각·음향으로 재질(과일 vs 공)을, 그리고 클래스 세부를 판별.

stage1_classes.py 와 동일한 인터페이스(BY_OBJ_ID / BY_OBJ_NAME / resolve /
resolve_fuzzy)를 제공하므로, extract_features_v3.py 의 `--stage 2a` 및
run_ml.py 가 무수정으로 label 을 얻는다.

> obj_id/obj_name 은 stage2a 수집 폴더명(obj01_orange … obj08_baseball)과 일치.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# 수집 폴더 순서 그대로 (obj01..obj08). group 은 상위 범주(과일/공).
_ORDER = [
    ("orange",     "fruit"),
    ("apple",      "fruit"),
    ("pear",       "fruit"),
    ("grapefruit", "fruit"),
    ("pingpong",   "ball"),
    ("golf",       "ball"),
    ("tennis",     "ball"),
    ("baseball",   "ball"),
]


@dataclass(frozen=True)
class Stage2AClass:
    obj_id: int          # 1..8
    obj_name: str        # 예: orange
    group: str           # fruit / ball  (상위 범주 = material 성격)
    label: int           # 0..7 (ML 타깃)

    # stage1 과 컬럼명 호환: material 속성으로도 group 을 노출
    @property
    def material(self) -> str:
        return self.group

    @property
    def content(self) -> Optional[str]:
        return None


def build_classes() -> List[Stage2AClass]:
    return [
        Stage2AClass(obj_id=i + 1, obj_name=name, group=grp, label=i)
        for i, (name, grp) in enumerate(_ORDER)
    ]


CLASSES: List[Stage2AClass] = build_classes()

BY_OBJ_ID: Dict[int, Stage2AClass] = {c.obj_id: c for c in CLASSES}
BY_OBJ_NAME: Dict[str, Stage2AClass] = {c.obj_name: c for c in CLASSES}


def resolve(obj_id: int | None = None, obj_name: str | None = None) -> Optional[Stage2AClass]:
    if obj_id is not None and obj_id in BY_OBJ_ID:
        return BY_OBJ_ID[obj_id]
    if obj_name is not None and obj_name in BY_OBJ_NAME:
        return BY_OBJ_NAME[obj_name]
    return None


# 이름 표기 변형 흡수용 별칭(한글/영문 혼용, 폴더명 접두 등)
_ALIASES = {
    "orange": "orange", "오렌지": "orange",
    "apple": "apple", "사과": "apple",
    "pear": "pear", "배": "pear",
    "grapefruit": "grapefruit", "자몽": "grapefruit",
    "pingpong": "pingpong", "ping": "pingpong", "탁구": "pingpong",
    "golf": "golf", "골프": "golf",
    "tennis": "tennis", "테니스": "tennis",
    "baseball": "baseball", "야구": "baseball",
}


def resolve_fuzzy(obj_id=None, obj_name=None, folder_name=None) -> Optional[Stage2AClass]:
    """정확 매칭 실패 시 obj_name/폴더명에서 키워드로 매칭.
    예) 'obj03_pear', 'PEAR', '배' 등 흡수."""
    exact = resolve(obj_id, obj_name)
    if exact:
        return exact
    for raw in (s for s in (obj_name, folder_name) if s):
        s = str(raw).lower()
        hit = next((v for k, v in _ALIASES.items() if k in s), None)
        if hit:
            return BY_OBJ_NAME.get(hit)
    return None


if __name__ == "__main__":
    print(f"{'obj_id':>6} {'label':>5}  {'group':<6} {'obj_name'}")
    for c in CLASSES:
        print(f"{c.obj_id:>6} {c.label:>5}  {c.group:<6} {c.obj_name}")
