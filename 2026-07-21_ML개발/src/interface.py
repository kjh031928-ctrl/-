"""interface.py — 입출력 명세 (두 사람 공용, 수정 시 협의 — CLAUDE.md §3).

역할: 학습·검증·변환 코드가 공유하는 상수와 결과 JSON 스키마를 한 곳에 정의한다.
다른 파일(data_loader, metrics, train_*, export_onnx)은 여기서 import 한다.
config/base.yaml 과 반드시 일치해야 하며, validate_against_config 로 표류를 검사한다.

하드코딩 금지 원칙(CLAUDE.md §4.2): 아래 상수는 config 값을 그대로 복제한 것이며
근거 주석을 달았다. validate_against_config 로 config 와의 불일치를 런타임에 잡는다.
"""

from __future__ import annotations

from typing import Any

# 입력세트 이름 — config/base.yaml features 의 키와 일치 (지시서 §3.1 / CLAUDE.md §2.1).
FEATURE_SETS: tuple[str, ...] = ("I1", "I2", "I3")

# 라벨 이름 → 데이터 컬럼명. config/base.yaml labels 블록과 일치해야 함.
#   r0 = X_propene_conversion  : 재순환 큐멘 0 mol% (원본 엑셀 컬럼)
#   r1 = X_propene_recycle1    : 재순환 큐멘 1 mol% (Phase 3 생성, labels_recycle1.csv)
LABELS: dict[str, str] = {"r0": "X_propene_conversion", "r1": "X_propene_recycle1"}

# ONNX 입력 순서 고정 (지시서 §4.11, config I1 주석). 이 3개가 항상 앞 순서.
#   압력세트(I2/I3) 선택 시 4번째로 아래 PRESSURE_TAG 를 덧붙인다.
ONNX_INPUT_ORDER: list[str] = ["TT-1006.PV", "FT-1004.PV", "FRC-1004.PV"]

# 입력세트별 추가 압력 태그 (config features 정의와 일치).
#   I2 = I1 + PT-1004.PV(프로펜 공급압력), I3 = I1 + PT-1100.PV(반응기 출구압력).
PRESSURE_TAG: dict[str, str | None] = {
    "I1": None,
    "I2": "PT-1004.PV",
    "I3": "PT-1100.PV",
}


def feature_columns(cfg: dict, feature_set: str) -> list[str]:
    """입력세트 이름 → 입력 컬럼 리스트(고정 순서). config features 를 그대로 사용.

    앞 3개가 ONNX_INPUT_ORDER 와 일치하는지, 압력 태그 위치가 맞는지 검증한다.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"알 수 없는 입력세트: {feature_set!r} (허용: {FEATURE_SETS})")
    cols = list(cfg["features"][feature_set])
    if cols[:3] != ONNX_INPUT_ORDER:
        raise ValueError(
            f"{feature_set}: 앞 3개 컬럼이 ONNX 고정 순서와 불일치. "
            f"config={cols[:3]} vs interface={ONNX_INPUT_ORDER}"
        )
    tag = PRESSURE_TAG[feature_set]
    if tag is None:
        if len(cols) != 3:
            raise ValueError(f"{feature_set}: 압력 태그가 없어야 하는데 {cols} 발견")
    else:
        if cols[3:] != [tag]:
            raise ValueError(f"{feature_set}: 4번째 컬럼이 {tag} 여야 하나 {cols[3:]} 발견")
    return cols


def label_column(label: str) -> str:
    """라벨 이름("r0"/"r1") → 데이터 컬럼명."""
    if label not in LABELS:
        raise ValueError(f"알 수 없는 라벨: {label!r} (허용: {tuple(LABELS)})")
    return LABELS[label]


def validate_against_config(cfg: dict) -> None:
    """interface 의 상수가 config/base.yaml 과 일치하는지 검사. 불일치 시 에러."""
    # 라벨 매핑 일치
    for name, col in LABELS.items():
        if cfg["labels"].get(name) != col:
            raise ValueError(
                f"labels[{name!r}] 불일치: config={cfg['labels'].get(name)!r} "
                f"vs interface={col!r}"
            )
    # 입력세트 존재 및 컬럼 순서 (feature_columns 내부 검증 재사용)
    for fs in FEATURE_SETS:
        if fs not in cfg["features"]:
            raise ValueError(f"config features 에 {fs} 없음")
        feature_columns(cfg, fs)


# --- 결과 JSON 스키마 ---------------------------------------------------------
# 24개 조합(모델2 x 입력세트3 x 재순환2 x 가중치2) 각각이 남기는 결과 dict.
# 지표는 두 스케일: conversion(전환율)과 cumene_kgh(큐멘 생산량). (지시서 §4.6)
RESULT_KEYS: tuple[str, ...] = (
    "model",
    "feature_set",
    "label",
    "weighted",
    "hyperparams",
    "seed",
    "timestamp",
    "n_train",
    "n_valid",
    "metrics",
)


def make_result(
    *,
    model: str,
    feature_set: str,
    label: str,
    weighted: bool,
    hyperparams: dict[str, Any],
    seed: int,
    timestamp: str,
    n_train: int,
    n_valid: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """표준 결과 dict 를 만든다. metrics 는 아래 두 블록을 담아야 한다.

    metrics = {
        "conversion": {"rmse":.., "mae":.., "mape":.., "r2":..},   # 전환율 스케일
        "cumene_kgh": {"rmse":.., "pct_of_mean":..},               # 큐멘 생산량 스케일
    }
    """
    result = {
        "model": model,
        "feature_set": feature_set,
        "label": label,
        "weighted": bool(weighted),
        "hyperparams": hyperparams,
        "seed": seed,
        "timestamp": timestamp,
        "n_train": int(n_train),
        "n_valid": int(n_valid),
        "metrics": metrics,
    }
    return result
