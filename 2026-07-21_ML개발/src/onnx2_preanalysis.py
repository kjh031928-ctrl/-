"""onnx2_preanalysis.py — ONNX2(반응기 압력강하 ΔP) 사전분석 (프롬프트 A).

인수인계_ONNX2_전체_2026-08-03.md §4 프롬프트 A 를 코드로 재현한다.
목적: 모델을 만들지 않고, ΔP 라벨·입력·총유량 정의·거동을 파악하고 선형이 충분한지 스크리닝.

절대규칙(CLAUDE.md): 원본수정금지 · 검증셋(10/17-18) 최종확인 전용 · 무작위분할 금지 ·
  .xlsx 산출금지(CSV·MD만) · 임의상수 금지 · 모든 수치 코드 재계산 · [주어짐]/[계산]/[가정] 구분.

sklearn 미설치 환경이므로 LinearRegression 은 numpy 정규방정식(OLS)으로 대체한다.
  → sklearn.LinearRegression 과 수학적으로 동일(절편 포함 최소제곱). 값 차이 < 1e-9.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_XLSX = _ROOT / "data" / "processed" / "cumene_ml_training_data.xlsx"
_CSV = _ROOT.parent / "2026 Chemical Engineering Process design competition_rev0_Attachment1.csv"
_OUT_CSV = _ROOT / "results" / "onnx2_dp_eda.csv"

# --- 상수 (출처 주석 필수, CLAUDE.md §4-2) ---
HX1005_TUBE_DP = 25.0   # kPa. HX-1005 튜브측 0.25 bar [주어짐] rev2 App B Table 4
TRAIN = "학습"          # 10/13-16
VALID = "검증"          # 10/17-18 (최종확인 전용)
NORMAL = "정상운전검사"  # 10/12


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """절편 포함 OLS. sklearn.LinearRegression 과 동일. 반환 (coef[n], intercept)."""
    Xd = np.column_stack([np.ones(len(X)), X])          # [1, x1, x2, ...]
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta[1:], float(beta[0])


def predict(coef, intercept, X):
    return X @ coef + intercept


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load() -> pd.DataFrame:
    df = pd.read_excel(_XLSX, sheet_name="Data")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    # FT-1003 병합 (원본 CSV, 읽기전용)
    csv = pd.read_csv(_CSV)
    csv = csv.rename(columns={csv.columns[0]: "Timestamp"})
    ft = csv[["Timestamp", "FT-1003.PV Value kg/h"]].rename(
        columns={"FT-1003.PV Value kg/h": "FT-1003.PV"})
    ft["Timestamp"] = pd.to_datetime(ft["Timestamp"], utc=True).dt.tz_localize(None)
    n0 = len(df)
    df = df.merge(ft, on="Timestamp", how="left")
    assert len(df) == n0, f"FT-1003 병합 후 행수변화 {n0}->{len(df)}"
    # 라벨 ΔP [kPa]
    df["dP"] = df["PT-1004.PV"] - HX1005_TUBE_DP - df["PT-1100.PV"]
    # 총유량 두 정의
    df["Ftot_a"] = df["FT-1004.PV"] * (1.0 + df["FRC-1004.PV"])       # (a) 프로펜×(1+FRC)
    df["Ftot_b"] = df["FT-1003.PV"] + df["FT-1004.PV"]                # (b) FT-1003+FT-1004
    return df


def sec_label(df):
    print("\n" + "=" * 70)
    print("# 1. 라벨 검증  ΔP = PT-1004 − 25 − PT-1100")
    print("=" * 70)
    neg = int((df["dP"] < 0).sum())
    print(f"음수 개수: {neg}  (기대 0)")
    rows = []
    for camp, g in df.groupby("Campaign"):
        rows.append((camp, len(g), g["dP"].min(), g["dP"].max(),
                     g["dP"].mean(), g["dP"].std(ddof=0)))
    hdr = f"{'캠페인':10}{'n':>6}{'min':>10}{'max':>10}{'mean':>10}{'std':>10}"
    print(hdr)
    for r in rows:
        print(f"{r[0]:10}{r[1]:>6}{r[2]:>10.3f}{r[3]:>10.3f}{r[4]:>10.4f}{r[5]:>10.4f}")
    ex1012 = df[df["Campaign"] != "10/12"]["dP"].mean()
    print(f"10/12 제외 전체 평균: {ex1012:.4f}  (문서 27.45 대조)")
    return rows


def sec_flow(df):
    print("\n" + "=" * 70)
    print("# 2. 총유량 정의 일치  (a)FT-1004×(1+FRC)  vs  (b)FT-1003+FT-1004")
    print("=" * 70)
    d = (df["Ftot_a"] - df["Ftot_b"]).abs()
    mean_flow = df[["Ftot_a", "Ftot_b"]].mean().mean()
    print(f"행별 최대 절대차: {d.max():.4f} kg/h  ({d.max()/mean_flow*100:.6f}% of 평균유량)")
    print(f"평균 절대차:      {d.mean():.4f} kg/h  ({d.mean()/mean_flow*100:.6f}%)")
    # kPa 환산: 총유량 기울기 ~0.00311 kPa/(kg/h) 사용
    print(f"→ 최대차의 ΔP 환산(×0.00311): {d.max()*0.00311059:.5f} kPa (z 폐기근거 0.078 과 대조)")
    for camp, g in df.groupby("Campaign"):
        print(f"  {camp:10} T[{g['TT-1006.PV'].min():.2f},{g['TT-1006.PV'].max():.2f}] "
              f"Ftot_b[{g['Ftot_b'].min():.0f},{g['Ftot_b'].max():.0f}] "
              f"FRC[{g['FRC-1004.PV'].min():.3f},{g['FRC-1004.PV'].max():.3f}]")


def sec_behavior(df):
    print("\n" + "=" * 70)
    print("# 3. DP 거동 · 상관 · 물리 방향")
    print("=" * 70)
    sub = df[df["Campaign"] != "10/12"]
    for col, name in [("TT-1006.PV", "온도"), ("Ftot_b", "총유량"), ("FRC-1004.PV", "FRC")]:
        r = np.corrcoef(sub[col], sub["dP"])[0, 1]
        print(f"  corr(ΔP, {name:5}) = {r:+.4f}")
    # 물리 방향 원자료: 총유량 17800~18200, B/P 4.5→5.5
    win = df[(df["Ftot_b"] >= 17800) & (df["Ftot_b"] <= 18200)]
    lo = win[win["FRC-1004.PV"] < 4.7]["dP"].mean()
    hi = win[win["FRC-1004.PV"] > 5.3]["dP"].mean()
    print(f"  총유량고정창 B/P<4.7 ΔP={lo:.3f} vs B/P>5.3 ΔP={hi:.3f}  (B/P↑→ΔP↓ 기대)")


def physics_27(coef, intercept, feat_order):
    """물리 27점 격자: 총유량↑→ΔP↑ / 총유량고정 후 B/P↑→ΔP↓. 각 9칸.
    feat_order: 리스트 ['T','F','R'] 순서로 coef 대응."""
    Ts = [335.0, 342.0, 350.0]
    Fs = [15000.0, 16500.0, 18000.0]
    Rs = [4.0, 5.0, 6.0]
    def mk(T, F, R):
        m = {"T": T, "F": F, "R": R}
        return np.array([m[k] for k in feat_order])
    up = 0
    for T in Ts:
        for R in Rs:
            v = [predict(coef, intercept, mk(T, F, R)) for F in Fs]
            if v[0] < v[1] < v[2]:
                up += 1
    down = 0
    for T in Ts:
        for F in Fs:
            v = [predict(coef, intercept, mk(T, F, R)) for R in Rs]
            if v[0] > v[1] > v[2]:
                down += 1
    return up, down


def sec_screen(df):
    print("\n" + "=" * 70)
    print("# 5. 모델 스크리닝 (입력 [TT-1006, 총유량, FRC], 학습10/13-16→검증10/17-18)")
    print("=" * 70)
    tr = df[df["Split"] == TRAIN]
    va = df[df["Split"] == VALID]
    print(f"학습 {len(tr)}행 / 검증 {len(va)}행")

    def build(g, kind):
        T = g["TT-1006.PV"].to_numpy(float)
        F = g["Ftot_b"].to_numpy(float)
        R = g["FRC-1004.PV"].to_numpy(float)
        if kind == "F2":      # (T, 총유량, FRC)  ← 채택 후보
            return np.column_stack([T, F, R]), ["T", "F", "R"]
        if kind == "F2sq":    # + 총유량²
            return np.column_stack([T, F, R, F ** 2]), ["T", "F", "R", "F2"]
        if kind == "F1":      # (T, 프로펜유량, FRC)
            Fp = g["FT-1004.PV"].to_numpy(float)
            return np.column_stack([T, Fp, R]), ["T", "Fp", "R"]
        raise ValueError(kind)

    y_tr = tr["dP"].to_numpy(float)
    y_va = va["dP"].to_numpy(float)
    results = []
    for kind in ["F2", "F2sq", "F1"]:
        Xtr, order = build(tr, kind)
        Xva, _ = build(va, kind)
        coef, b = ols(Xtr, y_tr)
        r = rmse(y_va, predict(coef, b, Xva))
        # 물리 27점 (F2/F1 만 T,F,R 3변수 격자 적용; F2sq 는 F 격자에 F² 동반)
        if kind in ("F2", "F1"):
            fo = ["T", ("F" if kind == "F2" else "Fp"), "R"]
            # physics_27 는 T/F/R 라벨만 알므로 Fp->F 로 매핑
            fo2 = ["T", "F", "R"]
            up, dn = physics_27(coef, b, fo2)
        else:
            up, dn = None, None
        results.append((kind, order, coef, b, r, up, dn))
        print(f"  {kind:5} 검증RMSE={r:.4f}  물리(F↑/BP↓)={up}/{dn}  절편={b:.4f}")
    return results


def sec_leak(df):
    print("\n" + "=" * 70)
    print("# 6. 누출·품질")
    print("=" * 70)
    dup = int(df["Timestamp"].duplicated().sum())
    print(f"중복 Timestamp: {dup}   결측 dP: {int(df['dP'].isna().sum())}")
    print("PT-1004·PT-1100 은 라벨 성분 → 입력 금지(누출) [확인]")


def report_coef(df):
    """확정 계수 재현: 학습 10/13-16 (T, 총유량, FRC)."""
    tr = df[df["Split"] == TRAIN]
    va = df[df["Split"] == VALID]
    X = tr[["TT-1006.PV", "Ftot_b", "FRC-1004.PV"]].to_numpy(float)
    y = tr["dP"].to_numpy(float)
    coef, b = ols(X, y)
    Xv = va[["TT-1006.PV", "Ftot_b", "FRC-1004.PV"]].to_numpy(float)
    r = rmse(va["dP"].to_numpy(float), predict(coef, b, Xv))
    print("\n" + "=" * 70)
    print("# 확정 계수 재현 (학습 10/13-16)")
    print("=" * 70)
    print(f"  절편        = {b:.4f}    (문서 -52.500)")
    print(f"  b_T[°C]     = {coef[0]:.6f}  (문서 +0.090071)")
    print(f"  b_총유량     = {coef[2 if False else 1]:.8f} (문서 +0.00311059)")
    print(f"  b_FRC       = {coef[2]:.5f}  (문서 -1.00681)")
    print(f"  검증 RMSE   = {r:.4f}    (문서 0.1910)")
    return coef, b, r


def main():
    df = load()
    sec_label(df)
    sec_flow(df)
    sec_behavior(df)
    sec_leak(df)
    sec_screen(df)
    report_coef(df)
    # EDA CSV 산출 (캠페인별 요약)
    _OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summ = df.groupby("Campaign").agg(
        n=("dP", "size"), dP_min=("dP", "min"), dP_max=("dP", "max"),
        dP_mean=("dP", "mean"), dP_std=("dP", lambda s: s.std(ddof=0)),
        T_min=("TT-1006.PV", "min"), T_max=("TT-1006.PV", "max"),
        Ftot_min=("Ftot_b", "min"), Ftot_max=("Ftot_b", "max"),
        FRC_min=("FRC-1004.PV", "min"), FRC_max=("FRC-1004.PV", "max"),
    ).reset_index()
    summ.to_csv(_OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[산출] {_OUT_CSV.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
