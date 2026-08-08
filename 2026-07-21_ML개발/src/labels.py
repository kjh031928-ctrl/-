"""labels.py — 재순환 큐멘 1 mol% 가정 라벨(X_propene_recycle1) 생성.

담당: 공용 (수정 시 상대에게 통보 — CLAUDE.md §3).

가정(지시서 §4.3): 재순환 스트림(FT-1300.PV)이 벤젠 99 mol% + 큐멘 1 mol%.
  MW_recycle   = 0.99·MW_benzene + 0.01·MW_cumene
  n_recycle    = FT-1300.PV / MW_recycle
  n_cumene_in  = n_recycle · 0.01
  n_benzene_in = FT-1001.PV / MW_benzene + n_recycle · 0.99   (신선 벤젠 + 재순환 벤젠)
  프로펜·프로판 입구 몰수는 0 mol% 경우와 동일(n_propene_kmol_h, n_propane_kmol_h).

이 입구 조성으로 srk.solve_conversion 을 다시 수행해 X_propene_recycle1 을 만든다.
brentq bracket 실패 행은 NaN 으로 두고 개수를 보고한다.

원본 정본 엑셀(Data 시트)은 절대 수정하지 않고(CLAUDE.md §1-1),
결과는 data/processed/labels_recycle1.csv 로 저장한다.

물성상수(MW 등)는 config/base.yaml 에서 읽는다 — 하드코딩 금지.
재순환 조성비 0.99 / 0.01 은 '재순환 큐멘 1 mol%' 가정 그 자체(지시서 §4.3)라 명시 상수.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from srk import COMPONENTS, load_properties, solve_conversion

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "config" / "base.yaml"

# 재순환 스트림 가정 조성 (지시서 §4.3): 벤젠 99 mol% + 큐멘 1 mol%.
RECYCLE_BENZENE_MOLFRAC = 0.99
RECYCLE_CUMENE_MOLFRAC = 0.01


def _mw_g_per_mol(cfg: dict, comp: str) -> float:
    """config 의 MW(kg/kmol == g/mol)를 그대로 반환. FT/유량 계산은 g/mol 단위 사용."""
    return float(cfg["properties"]["MW_kg_per_kmol"][comp])


def generate_recycle1_labels(
    xlsx_path: Path | str,
    sheet: str = "Data",
    config_path: Path | str = _CONFIG_PATH,
) -> tuple[pd.DataFrame, int]:
    """재순환 1 mol% 라벨을 생성해 DataFrame 과 bracket 실패 건수를 반환."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    props = load_properties(config_path)

    mw_benzene = _mw_g_per_mol(cfg, "benzene")  # g/mol
    mw_cumene = _mw_g_per_mol(cfg, "cumene")
    mw_recycle = (
        RECYCLE_BENZENE_MOLFRAC * mw_benzene + RECYCLE_CUMENE_MOLFRAC * mw_cumene
    )  # 재순환 혼합물 평균 몰질량 (g/mol)

    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    x_recycle1 = np.full(len(df), np.nan, dtype=np.float64)
    fails = 0
    for i, r in df.iterrows():
        # 재순환/신선 벤젠으로 입구 조성 재구성 (프로펜·프로판은 r0 와 동일)
        n_recycle = r["FT-1300.PV"] / mw_recycle  # kmol/h
        n_cumene_in = n_recycle * RECYCLE_CUMENE_MOLFRAC
        n_benzene_in = (
            r["FT-1001.PV"] / mw_benzene + n_recycle * RECYCLE_BENZENE_MOLFRAC
        )
        try:
            x_recycle1[i] = solve_conversion(
                target_rho=r["AT-1100.PV"],
                n_propene=r["n_propene_kmol_h"],
                n_propane=r["n_propane_kmol_h"],
                n_benzene=n_benzene_in,
                n_cumene_in=n_cumene_in,
                T_K=r["TT-1100.PV"] + 273.15,
                P_Pa=r["PT-1100.PV"] * 1000.0,
                props=props,
            )
        except (ValueError, RuntimeError):
            fails += 1  # bracket 실패 → NaN 유지

    out = pd.DataFrame(
        {
            "Timestamp": df["Timestamp"],
            "Campaign": df["Campaign"],
            "Split": df["Split"],
            "X_propene_conversion": df["X_propene_conversion"],  # r0 (원본, 대조용)
            "X_propene_recycle1": x_recycle1,  # r1 (신규)
        }
    )
    out["r0_minus_r1"] = out["X_propene_conversion"] - out["X_propene_recycle1"]
    return out, fails


def main() -> None:
    xlsx = _ROOT / "data" / "processed" / "cumene_ml_training_data.xlsx"
    out_csv = _ROOT / "data" / "processed" / "labels_recycle1.csv"

    out, fails = generate_recycle1_labels(xlsx)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    diff = out["r0_minus_r1"]
    print(f"저장: {out_csv}")
    print(f"bracket 실패 건수: {fails}")
    print(f"r1 NaN 건수: {int(out['X_propene_recycle1'].isna().sum())}")
    print(f"r0 - r1 평균 차이: {diff.mean():.6f}")
    print(f"r0 - r1 평균 |차이|: {diff.abs().mean():.6f}")
    print(f"상대차(평균차/평균 r0): {diff.mean() / out['X_propene_conversion'].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
