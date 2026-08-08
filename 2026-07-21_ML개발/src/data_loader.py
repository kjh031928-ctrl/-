"""data_loader.py — 로딩·분할·가중치 (두 사람 공용 — CLAUDE.md §3).

역할: 정본 엑셀(Data 시트)을 읽고 Phase 3 의 재순환 1 mol% 라벨(r1)을 병합한 뒤,
Split 컬럼(학습/검증/정상운전검사)으로 나눈다. 무작위 분할 금지(CLAUDE.md §1-3).

원칙:
  - 컬럼명·경로·라벨명 등 상수는 config/base.yaml 과 interface.py 에서 읽는다(하드코딩 금지).
  - 학습 sample_weight 로 ML_weight 제공. 검증 지표에는 가중치 미적용(지시서 §4.5).
  - float64 유지. float32 로 내리지 않는다(CLAUDE.md §2.4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import interface

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "config" / "base.yaml"

# n_propene 컬럼 — 정본 엑셀 파생 컬럼(kmol/h). 큐멘 생산량 환산에 사용(지시서 §4.6).
_N_PROPENE_COL = "n_propene_kmol_h"

_cfg_cache: dict[str, Any] | None = None


def load_config(config_path: Path | str = _CONFIG_PATH) -> dict:
    """config/base.yaml 을 읽어 dict 로 반환(모듈 내 캐시)."""
    global _cfg_cache
    if _cfg_cache is None:
        with open(config_path, "r", encoding="utf-8") as f:
            _cfg_cache = yaml.safe_load(f)
    return _cfg_cache


def load_data(config: dict) -> pd.DataFrame:
    """정본 엑셀 Data 시트를 읽고 r1 라벨을 Timestamp 로 병합해 반환.

    r1 라벨(interface.LABELS['r1'])이 최종 DataFrame 에 없으면 명확한 에러로 멈춘다.
    """
    interface.validate_against_config(config)  # 상수 표류 검사

    xlsx = _ROOT / config["paths"]["data_xlsx"]
    sheet = config["paths"]["data_sheet"]
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    r1_col = interface.LABELS["r1"]  # X_propene_recycle1

    if r1_col not in df.columns:
        # 정본 엑셀에 없으면 별도 파일(labels_recycle1.csv)에서 병합
        r1_path_key = config["paths"].get("labels_recycle1")
        if not r1_path_key:
            raise RuntimeError(
                f"r1 라벨 컬럼 {r1_col!r} 이 엑셀에 없고 config.paths.labels_recycle1 도 없음. "
                "Phase 3 라벨 생성(labels.py) 결과가 필요합니다."
            )
        r1_path = _ROOT / r1_path_key
        if not r1_path.exists():
            raise FileNotFoundError(
                f"r1 라벨 파일이 없습니다: {r1_path}. Phase 3(labels.py main)을 먼저 실행하세요."
            )
        r1 = pd.read_csv(r1_path)
        if r1_col not in r1.columns:
            raise RuntimeError(f"{r1_path.name} 에 {r1_col!r} 컬럼이 없습니다.")
        r1["Timestamp"] = pd.to_datetime(r1["Timestamp"])
        merged = df.merge(r1[["Timestamp", r1_col]], on="Timestamp", how="left")
        if len(merged) != len(df):
            raise RuntimeError(
                f"r1 병합 후 행수 변화({len(df)}→{len(merged)}). Timestamp 중복 의심."
            )
        df = merged

    if r1_col not in df.columns:
        raise RuntimeError(f"병합 후에도 r1 라벨 {r1_col!r} 이 없습니다.")
    # Timestamp 불일치로 인한 결측(브래킷 실패 NaN 과 구분): 병합 미스매치면 에러
    if df[r1_col].isna().all():
        raise RuntimeError(f"r1 라벨 {r1_col!r} 이 전부 결측입니다. 병합 키(Timestamp) 확인.")
    return df


def get_split(df: pd.DataFrame, which: str) -> pd.DataFrame:
    """Split 컬럼으로 필터. which ∈ {"학습","검증","정상운전검사"}. 무작위 분할 금지.

    config split.{train,valid,normalcheck}_value 로 유효값을 검사한다.
    """
    cfg = load_config()
    split_col = cfg["split"]["column"]
    allowed = {
        cfg["split"]["train_value"],
        cfg["split"]["valid_value"],
        cfg["split"]["normalcheck_value"],
    }
    if which not in allowed:
        raise ValueError(f"알 수 없는 split 값: {which!r} (허용: {allowed})")
    return df[df[split_col] == which].copy()


def get_Xy(
    df: pd.DataFrame, feature_set: str, label: str
) -> tuple[np.ndarray, np.ndarray]:
    """입력세트("I1/I2/I3")·라벨("r0/r1")로 X, y(둘 다 float64)를 반환.

    입력 컬럼 순서는 interface.feature_columns 를 따른다(ONNX 순서 고정).
    """
    cfg = load_config()
    cols = interface.feature_columns(cfg, feature_set)
    label_col = interface.label_column(label)

    missing = [c for c in cols + [label_col] if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame 에 필요한 컬럼 없음: {missing}")

    X = df[cols].to_numpy(dtype=np.float64)
    y = df[label_col].to_numpy(dtype=np.float64)
    return X, y


def get_weights(df: pd.DataFrame) -> np.ndarray:
    """ML_weight 컬럼을 float64 ndarray 로 반환. 학습에만 사용(검증 지표엔 미적용)."""
    cfg = load_config()
    wcol = cfg["weight_column"]  # "ML_weight"
    if wcol not in df.columns:
        raise KeyError(f"가중치 컬럼 {wcol!r} 이 DataFrame 에 없습니다.")
    return df[wcol].to_numpy(dtype=np.float64)


def get_n_propene(df: pd.DataFrame) -> np.ndarray:
    """큐멘 생산량 환산용 n_propene_kmol_h 를 float64 ndarray 로 반환."""
    if _N_PROPENE_COL not in df.columns:
        raise KeyError(f"{_N_PROPENE_COL!r} 컬럼이 DataFrame 에 없습니다.")
    return df[_N_PROPENE_COL].to_numpy(dtype=np.float64)
