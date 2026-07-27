"""
stage1_classes.py
Stage 1 9-class factorial 정의 (공용 모듈)
==========================================

컵재질 3 (glass/ceramic/plastic) × 내용물 3 (empty/ethanol/acetone) = 9 클래스.
obj_id, obj_name, material, content, label 을 한 곳에서 정의해
수집(collect)·피처추출(extract)·ML(run_ml) 이 동일한 매핑을 공유한다.

> 근거: ML_설계_정리.md §5 (9클래스 factorial), 메모리(내용물 IPA→acetone 변경).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


MATERIALS = ["glass", "ceramic", "plastic"]
CONTENTS = ["empty", "ethanol", "acetone"]

# VOC(휘발성) 내용물 — 이 클래스들은 trial 간격을 길게(60s+) 둔다.
VOC_CONTENTS = {"ethanol", "acetone"}


@dataclass(frozen=True)
class Stage1Class:
    obj_id: int          # 1..9
    obj_name: str        # 예: glass_ethanol  (디렉토리/파일명에 사용)
    material: str        # glass / ceramic / plastic
    content: str         # empty / ethanol / acetone
    label: int           # 0..8 (ML 타깃)

    @property
    def is_voc(self) -> bool:
        return self.content in VOC_CONTENTS


def build_classes() -> List[Stage1Class]:
    """9개 클래스를 (재질 바깥, 내용물 안쪽) 순서로 생성. obj_id/label = 1..9 / 0..8."""
    classes: List[Stage1Class] = []
    idx = 0
    for material in MATERIALS:
        for content in CONTENTS:
            classes.append(
                Stage1Class(
                    obj_id=idx + 1,
                    obj_name=f"{material}_{content}",
                    material=material,
                    content=content,
                    label=idx,
                )
            )
            idx += 1
    return classes


CLASSES: List[Stage1Class] = build_classes()

# 빠른 조회용 인덱스
BY_OBJ_ID: Dict[int, Stage1Class] = {c.obj_id: c for c in CLASSES}
BY_OBJ_NAME: Dict[str, Stage1Class] = {c.obj_name: c for c in CLASSES}


def resolve(obj_id: int | None = None, obj_name: str | None = None) -> Stage1Class | None:
    """obj_id 또는 obj_name 으로 클래스 메타를 찾는다 (둘 다 없으면 None)."""
    if obj_id is not None and obj_id in BY_OBJ_ID:
        return BY_OBJ_ID[obj_id]
    if obj_name is not None and obj_name in BY_OBJ_NAME:
        return BY_OBJ_NAME[obj_name]
    return None


# 이름 표기 변형까지 흡수하는 키워드 사전 (수집 방식이 달라도 매칭되도록)
_MATERIAL_ALIASES = {
    "glass": "glass", "유리": "glass",
    "ceramic": "ceramic", "porcelain": "ceramic", "도자기": "ceramic", "세라믹": "ceramic",
    "plastic": "plastic", "플라스틱": "plastic",
}
_CONTENT_ALIASES = {
    "empty": "empty", "none": "empty", "dry": "empty", "빈": "empty", "공": "empty",
    "ethanol": "ethanol", "etoh": "ethanol", "에탄올": "ethanol",
    "acetone": "acetone", "아세톤": "acetone",
    # 혹시 IPA 로 수집했다면 (Stage1 표엔 없지만) 알려주기 위해 감지만
    "ipa": "_ipa", "isopropyl": "_ipa", "이소프로": "_ipa",
}


def resolve_fuzzy(obj_id=None, obj_name=None, folder_name=None) -> Stage1Class | None:
    """정확 매칭 실패 시, obj_name/폴더명에서 재질·내용물 키워드를 찾아 매칭.

    예) 'obj02_유리_에탄올', 'glass-ethanol', 'GlassEtOH' 등도 흡수.
    IPA 등 9클래스에 없는 내용물이면 None 을 반환(경고는 호출부에서).
    """
    exact = resolve(obj_id, obj_name)
    if exact:
        return exact
    # 후보 문자열들(소문자)에서 키워드 탐색
    candidates = [s for s in (obj_name, folder_name) if s]
    for raw in candidates:
        s = str(raw).lower()
        mat = next((v for k, v in _MATERIAL_ALIASES.items() if k in s), None)
        con = next((v for k, v in _CONTENT_ALIASES.items() if k in s), None)
        if mat and con and not con.startswith("_"):
            return BY_OBJ_NAME.get(f"{mat}_{con}")
    return None


if __name__ == "__main__":
    print(f"{'obj_id':>6} {'label':>5}  {'material':<8} {'content':<8} {'is_voc':>6}")
    for c in CLASSES:
        print(f"{c.obj_id:>6} {c.label:>5}  {c.material:<8} {c.content:<8} {str(c.is_voc):>6}")
