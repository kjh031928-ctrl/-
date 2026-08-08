"""recycle_sensitivity.py — [추가 3] 재순환 큐멘 몰분율 가정 민감도 (신설 2026-08-01).

rev2 §2: "benzene (containing <=1 mol% cumene) is recycled" — 1 mol% 는 상한이지 정확값이 아님.
재순환 큐멘 몰분율 f ∈ {0, 0.25, 0.5, 0.75, 1.0}% 각각으로 SRK 역산 라벨을 재산출하고,
로짓 1차·I1 모델의 계수·예측·큐멘 kg/h 를 비교한다. 원본/공용 labels.py 는 수정하지 않는다.

⚠ 가정별 검증 RMSE 로 "가장 좋은 가정"을 고르지 않는다(라벨이 달라 채점지가 다름). 표기만 함.
채택안: 1 mol% 유지(문제 명시 유일값, 가장 낮은 X → 보수적).
"""
from __future__ import annotations
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError): pass

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

from srk import load_properties, solve_conversion
import evaluation as ev

_ROOT = Path(__file__).resolve().parents[1]
_CFG = _ROOT / "config" / "base.yaml"
XLSX = _ROOT / "data" / "processed" / "cumene_ml_training_data.xlsx"
FRACS = (0.0, 0.0025, 0.005, 0.0075, 0.01)   # 재순환 큐멘 몰분율 (0~1 mol%)
MW_CUMENE = 120.196
CASE = {"T": 350.0, "FT": 3000.0, "FRC": 5.0}  # 케이스스터디 대표점


def gen_labels(df, props, mw_b, mw_c, f):
    """재순환 큐멘 몰분율 f 로 X 라벨 재산출. (배열, 실패건수)."""
    mw_rec = (1.0 - f) * mw_b + f * mw_c
    X = np.full(len(df), np.nan)
    fails = 0
    for i, r in df.iterrows():
        n_rec = r["FT-1300.PV"] / mw_rec
        n_cum_in = n_rec * f
        n_benz = r["FT-1001.PV"] / mw_b + n_rec * (1.0 - f)
        try:
            X[i] = solve_conversion(
                target_rho=r["AT-1100.PV"], n_propene=r["n_propene_kmol_h"],
                n_propane=r["n_propane_kmol_h"], n_benzene=n_benz, n_cumene_in=n_cum_in,
                T_K=r["TT-1100.PV"] + 273.15, P_Pa=r["PT-1100.PV"] * 1000.0, props=props)
        except (ValueError, RuntimeError):
            fails += 1
    return X, fails


def logit_fit(X_in, y):
    yc = np.clip(y, ev.LOGIT_EPS, 1 - ev.LOGIT_EPS)
    return LinearRegression().fit(X_in, np.log(yc / (1 - yc)))


def main():
    cfg = yaml.safe_load(open(_CFG, encoding="utf-8"))
    props = load_properties(_CFG)
    mw_b = cfg["properties"]["MW_kg_per_kmol"]["benzene"]
    mw_c = cfg["properties"]["MW_kg_per_kmol"]["cumene"]
    df = pd.read_excel(XLSX, sheet_name="Data")
    feats = ["TT-1006.PV", "FT-1004.PV", "FRC-1004.PV"]

    train = df[df["Split"] == "학습"]
    valid = df[df["Split"] == "검증"]
    tr_mask = df["Split"].to_numpy() == "학습"
    va_mask = df["Split"].to_numpy() == "검증"
    npr_all = df["n_propene_kmol_h"].to_numpy()
    ft1201 = df["FT-1201.PV"].to_numpy()
    n_prop_case = CASE["FT"] * 0.94773 / 42.081
    x_case = np.array([[CASE["T"], CASE["FT"], CASE["FRC"]]])

    labels = {}
    print(f"{'frac%':>6}{'fails':>6}{'X평균':>9}{'1%대비':>10}{'b_T':>9}{'b_R':>9}"
          f"{'큐멘@350/3000/5':>15}{'검증RMSE(비교불가)':>18}")
    x1 = None
    rows = []
    for f in FRACS:
        X, fails = gen_labels(df, props, mw_b, mw_c, f)
        labels[f] = X
        if f == 0.01:
            x1 = X
    xmean1 = np.nanmean(labels[0.01])
    for f in FRACS:
        X = labels[f]
        m = logit_fit(df.loc[tr_mask, feats].to_numpy(), X[tr_mask])
        bT, bR = m.coef_[0], m.coef_[2]
        cum_case = ev.from_logit(m.predict(x_case))[0] * n_prop_case * MW_CUMENE
        # 검증 RMSE (비교불가 — 표기만)
        pv = ev.from_logit(m.predict(df.loc[va_mask, feats].to_numpy()))
        vrmse = float(np.sqrt(np.nanmean((pv - X[va_mask]) ** 2)))
        xmean = np.nanmean(X)
        rows.append((f, fails, xmean, xmean - xmean1, bT, bR, cum_case, vrmse))
        print(f"{f*100:>6.2f}{fails:>6}{xmean:>9.5f}{xmean-xmean1:>+10.6f}{bT:>9.5f}"
              f"{bR:>9.4f}{cum_case:>15.1f}{vrmse:>18.5f}")

    # 큐멘 물질수지 (반응기 생성 큐멘 vs FT-1201, 전체 3066행)
    print(f"\n{'frac%':>6}{'평균차kg/h':>12}{'평균차%':>9}{'차이RMSE':>10}")
    for f in (0.0, 0.005, 0.01):
        X = labels[f]
        reactor = X * npr_all * MW_CUMENE
        d = reactor - ft1201
        print(f"{f*100:>6.2f}{np.nanmean(d):>12.1f}{np.nanmean(d)/np.nanmean(ft1201)*100:>+9.2f}"
              f"{np.sqrt(np.nanmean(d**2)):>10.1f}")

    # 재순환 조성 전 구간 일정 전제의 근사성 — 컬럼온도·유량 변화 (원본 CSV)
    raw = pd.read_csv(_ROOT.parent / "2026 Chemical Engineering Process design competition_rev0_Attachment1.csv")
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    raw["date"] = raw["Timestamp"].dt.strftime("%m-%d")
    raw = raw[raw["date"] != "10-12"]
    tc1215 = raw["TC-1215.PV Value °C"]
    ratio = raw["FT-1300.PV Value kg/h"] / raw["FT-1200.PV Value kg/h"]
    print(f"\nTC-1215(컬럼상부온도) 범위: {tc1215.min():.2f} ~ {tc1215.max():.2f} °C")
    print(f"FT-1300/FT-1200 비 범위: {ratio.min():.4f} ~ {ratio.max():.4f}")

    # 저장(CSV) — 재현용
    out = pd.DataFrame(rows, columns=["frac", "fails", "X_mean", "diff_vs_1pct",
                                      "b_T", "b_R", "cumene_case_kgh", "valid_rmse_비교불가"])
    out.to_csv(_ROOT / "results" / "recycle_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\n저장: results/recycle_sensitivity.csv")


if __name__ == "__main__":
    main()
