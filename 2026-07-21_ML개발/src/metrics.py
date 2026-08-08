"""metrics.py — 지표 계산 및 결과 저장 (두 사람 공용 — CLAUDE.md §3).

역할:
  - conversion_metrics : 전환율 스케일 RMSE/MAE/MAPE/R2 (Attachment 2 예제와 동일 구성).
  - cumene_metrics     : 큐멘 생산량 스케일 RMSE[kg/h] 및 평균 생산량 대비 %.
  - save_result        : 결과 dict 를 results/ 에 파일명 충돌 없이 저장.

원칙:
  - 상수(큐멘 몰질량)는 config/base.yaml 에서 읽는다(하드코딩 금지 — CLAUDE.md §4.2).
  - 검증 지표에는 ML_weight 를 적용하지 않는다(가중치는 학습에만 — 지시서 §4.5).
  - 계산은 float64 (CLAUDE.md §2.4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import interface

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "config" / "base.yaml"

_mw_cumene_cache: float | None = None


def _cumene_mw(config_path: Path | str = _CONFIG_PATH) -> float:
    """config properties.MW_kg_per_kmol.cumene (=120.196 kg/kmol) 를 읽어 반환.

    큐멘 몰질량. AVEVA 제공값, 대체 금지(지시서 §3.4 / CLAUDE.md §2.3).
    """
    global _mw_cumene_cache
    if _mw_cumene_cache is None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        _mw_cumene_cache = float(cfg["properties"]["MW_kg_per_kmol"]["cumene"])
    return _mw_cumene_cache


def conversion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """전환율 스케일 지표. MAPE 는 % 단위. 반환 {rmse, mae, mape, r2}."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    # MAPE(%) — y_true=0 은 정의불가라 제외. 전환율은 0~1 사이 양수라 실제로는 전부 유효.
    nonzero = y_true != 0.0
    mape = float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100.0)

    # R2 = 1 - SS_res/SS_tot
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def cumene_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_propene: np.ndarray
) -> dict[str, float]:
    """전환율 → 큐멘 생산량[kg/h] 환산 후 RMSE 와 평균 생산량 대비 % 반환.

    환산(지시서 §4.6): cumene_kg_h = X × n_propene(kmol/h) × MW_cumene(120.196 kg/kmol).
    n_propene 은 y 와 정렬된 배열이어야 한다.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n_propene = np.asarray(n_propene, dtype=np.float64)

    mw = _cumene_mw()
    cum_true = y_true * n_propene * mw
    cum_pred = y_pred * n_propene * mw

    err = cum_pred - cum_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mean_true = float(np.mean(cum_true))
    pct_of_mean = float(rmse / mean_true * 100.0) if mean_true != 0 else float("nan")

    return {"rmse": rmse, "pct_of_mean": pct_of_mean}


def save_result(result_dict: dict[str, Any], results_dir: Path | str) -> Path:
    """결과 dict 를 results/ 에 저장. 파일명: {model}_{feature_set}_{label}_{weighted}.json.

    파일명에 model 이 들어가므로 두 사람(선형/부스팅) 결과가 서로 덮어쓰지 않는다.
    """
    for key in ("model", "feature_set", "label", "weighted"):
        if key not in result_dict:
            raise KeyError(f"result_dict 에 {key!r} 필요 (interface.make_result 사용 권장)")

    weighted_tag = "weighted" if result_dict["weighted"] else "unweighted"
    fname = (
        f"{result_dict['model']}_{result_dict['feature_set']}_"
        f"{result_dict['label']}_{weighted_tag}.json"
    )

    out_dir = Path(results_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    return out_path
