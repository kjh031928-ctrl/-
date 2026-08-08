# -*- coding: utf-8 -*-
"""보고서 본문 수치 일괄 재현 검증 (§3.2.2 · §3.5.2 · §3.5.3 · §3.5.4 · §3.6.4)

다른 재현 스크립트가 담당하지 않는 나머지 수치를 한 번에 대조한다.
  · §3.5.1 후보 25종 RMSE·clamp      → `스크리닝_재현_전종.py`
  · 부록 C 표 C4 순열 중요도            → `permutation_importance_검증.py`
  · 부록 C 표 C1–C3 L1 변수선택        → `26-08-06/L1_변수선택_교차검증_재현스크립트.py`
  · 그림 F1–F3                        → `figures/make_figures.py` (자체 assert 내장)
  · **그 외 본문 수치**                 → 이 스크립트

모든 값은 마스터 데이터에서 재계산하며, 기재값과 어긋나면 마지막에 목록으로 보고하고
종료코드 1로 멈춘다. 재현 조건(운전점·하이퍼파라미터)이 결과를 가르는 항목은
아래 상수에 그 조건을 명시해 두었다 — 조건 없이는 재현되지 않기 때문이다.

재현
  python 보고서수치_재현_검증.py    (repo 루트에서 실행)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent
SEED = 42

# 재현 조건 (보고서에 함께 적어야 하는 값 — 없으면 수치가 달라진다)
EXTRAP_TOTAL, EXTRAP_BP = 18000.0, 5.0      # §3.5.2 외삽 대표 운전점
GBR_DP = dict(random_state=SEED)                                # sklearn 기본값
RF_DP = dict(random_state=SEED, n_jobs=-1)                     # sklearn 기본값

df = pd.read_csv(ROOT / "ML_마스터데이터_PV.csv", encoding="utf-8-sig")
tr, va = df[df.Split == "학습"], df[df.Split == "검증"]
IC = ["TT-1006.PV", "FT-1004.PV", "FRC-1004.PV"]      # 전환율 입력 I1
ID = ["TT-1006.PV", "total_flow_kgh", "FRC-1004.PV"]  # ΔP 입력 F2

rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
sig = lambda z: 1.0 / (1.0 + np.exp(-z))
FAIL: list[str] = []


def chk(label, got, ref, tol, unit=""):
    ok = abs(got - ref) <= tol
    print(f"  [{'OK ' if ok else '불일치'}] {label}: 재현 {got:.5g}{unit}  기재 {ref:.5g}{unit}")
    if not ok:
        FAIL.append(f"{label}: 재현 {got:.5g}{unit} vs 기재 {ref:.5g}{unit}")


def acf_within_campaign(resid, camp, lag=1):
    """캠페인 경계를 넘지 않는 lag-k 자기상관.

    경계를 넘겨 이으면 서로 다른 날의 잔차가 짝지어져 값이 왜곡된다.
    """
    a, b = [], []
    for c in np.unique(camp):
        r = resid[camp == c]
        if len(r) > lag:
            a.extend(r[:-lag]); b.extend(r[lag:])
    return float(np.corrcoef(np.array(a), np.array(b))[0, 1])


# --------------------------------------------------------------- §3.2.2
print("=" * 88)
print("§3.2.2 잔차 자기상관 (학습 10/13–16, 1분 지연, 캠페인 내부)")
print("=" * 88)
y = tr["X_conv_r1_deployed"].values
m_c = LinearRegression().fit(tr[IC].values, np.log(y / (1 - y)))
res_c = y - sig(m_c.predict(tr[IC].values))
m_d = LinearRegression().fit(tr[ID].values, tr["DP_reactor_kPa"].values)
res_d = tr["DP_reactor_kPa"].values - m_d.predict(tr[ID].values)
camp = tr["Campaign"].values
chk("ΔP 1분 자기상관", acf_within_campaign(res_d, camp), 0.9997, 5e-4)
chk("전환율 1분 자기상관", acf_within_campaign(res_c, camp), 0.993, 1e-3)
print(f"        (60분 지연: 전환율 {acf_within_campaign(res_c, camp, 60):+.3f} · "
      f"ΔP {acf_within_campaign(res_d, camp, 60):+.3f} — 독립표본이 아님)")

# --------------------------------------------------------------- §3.5.2
print("\n" + "=" * 88)
print(f"§3.5.2 외삽 거동 (학습 재적합 · 총유량 {EXTRAP_TOTAL:.0f} kg/h · B/P {EXTRAP_BP:.0f})")
print("=" * 88)
ft = EXTRAP_TOTAL / (1.0 + EXTRAP_BP)
g_c = GradientBoostingRegressor(random_state=SEED).fit(tr[IC].values, np.log(y / (1 - y)))
lin_v, gbr_v = [], []
for T in (355.0, 360.0, 365.0):
    X = np.array([[T, ft, EXTRAP_BP]])
    lin_v.append(float(sig(m_c.predict(X))[0])); gbr_v.append(float(sig(g_c.predict(X))[0]))
print(f"  로짓-선형 355→360→365 °C: {lin_v[0]:.3f} → {lin_v[1]:.3f} → {lin_v[2]:.3f}  (기재 0.89→0.94→0.97)")
print(f"  GBR                     : {gbr_v[0]:.3f} → {gbr_v[1]:.3f} → {gbr_v[2]:.3f}  (기재 0.84 고정)")
for v, r in zip(lin_v, (0.89, 0.94, 0.97)):
    chk(f"  로짓-선형 {r}", v, r, 5e-3)
chk("  GBR 고정값", gbr_v[0], 0.84, 5e-3)
if not (gbr_v[0] == gbr_v[1] == gbr_v[2]):
    FAIL.append("GBR 이 355/360/365 에서 동일하지 않음(clamp 아님)")
else:
    print("  [OK ] GBR 세 온도 예측이 완전히 동일 — clamp 확인")

# 전환율 27점 물리방향 (노트 S2, 9/27)
T_G, TOT_G, BP_G = [335., 350., 365.], [15000., 16500., 18000.], [4., 5., 6.]
f = lambda t, tot, bp: float(sig(m_c.predict(np.array([[t, tot / (1 + bp), bp]])))[0])
up = lambda v: v[0] < v[1] < v[2]
t_ok = sum(up([f(t, tot, bp) for t in T_G]) for tot in TOT_G for bp in BP_G)
q_ok = sum(up([f(t, tot, bp) for tot in TOT_G][::-1]) for t in T_G for bp in BP_G)  # 유량↑→X↓
b_ok = sum(up([f(t, tot, bp) for bp in BP_G]) for t in T_G for tot in TOT_G)        # B/P↑→X↑
chk("전환율 27점 물리방향 합계(온도↑X↑ + 유량↑X↓ + B/P↑X↑)", t_ok + q_ok + b_ok, 9, 0)
print(f"        내역: 온도 {t_ok}/9 · 유량 {q_ok}/9 · B/P {b_ok}/9")

# --------------------------------------------------------------- §3.5.3
print("\n" + "=" * 88)
print("§3.5.3 압력강하 대표 후보 (학습 13–16 → held-out 10/17–18)")
print("=" * 88)
Xtr, ytr = tr[ID].values, tr["DP_reactor_kPa"].values
Xva, yva = va[ID].values, va["DP_reactor_kPa"].values


def mono(est):
    g = lambda t, fl, bp: float(est.predict(np.array([[t, fl, bp]]))[0])
    return (sum(g(t, TOT_G[0], bp) < g(t, TOT_G[1], bp) < g(t, TOT_G[2], bp)
                for t in T_G for bp in BP_G),
            sum(g(t, fl, BP_G[0]) > g(t, fl, BP_G[1]) > g(t, fl, BP_G[2])
                for t in T_G for fl in TOT_G))


chk("선형 F2 held-out RMSE", rmse(m_d.predict(Xva), yva), 0.1910, 5e-4, " kPa")
m_f1 = LinearRegression().fit(tr[IC].values, ytr)
chk("선형 F1 held-out RMSE", rmse(m_f1.predict(va[IC].values), yva), 0.1479, 5e-4, " kPa")
chk("F1 의 B/P 계수", m_f1.coef_[2], 7.98, 0.02, " kPa")
chk("F2 의 B/P 계수", m_d.coef_[2], -1.01, 0.02, " kPa")
fl2, bp2 = mono(m_d)
chk("F2 27점 유량 방향", fl2, 9, 0); chk("F2 27점 B/P 방향", bp2, 9, 0)
for nm, est, ref_r, ref_bp in (("GBR", GradientBoostingRegressor(**GBR_DP), 0.4413, 9),
                               ("RF", RandomForestRegressor(**RF_DP), 0.6680, 2)):
    est.fit(Xtr, ytr); fl, bp = mono(est)
    chk(f"{nm} held-out RMSE", rmse(est.predict(Xva), yva), ref_r, 5e-3, " kPa")
    chk(f"{nm} 27점 B/P 방향", bp, ref_bp, 0)

r2 = LinearRegression().fit(tr[["FT-1004.PV", "FRC-1004.PV"]].values,
                            tr["total_flow_kgh"].values).score(
     tr[["FT-1004.PV", "FRC-1004.PV"]].values, tr["total_flow_kgh"].values)
chk("[프로펜+B/P] → 총유량 복원 R²", r2, 0.95, 0.01)

c15 = df[df.Campaign == "10/15"]
ctrl = ["FRC-1004.PV", "total_flow_kgh", "TT-1006.PV"]
chk("10/15 B/P 스윕 밀도 기울기", LinearRegression().fit(c15[ctrl].values,
    c15["AT-1100.PV"].values).coef_[0], 0.108, 0.01, " kg/m³")
chk("10/15 B/P 스윕 ΔP 기울기", LinearRegression().fit(c15[ctrl].values,
    c15["DP_reactor_kPa"].values).coef_[0], -1.26, 0.02, " kPa")

# --------------------------------------------------------------- §3.5.4 · §3.6.4
print("\n" + "=" * 88)
print("§3.5.4 온도 범위 · §3.6.4 성능 종합표 · 10/12 앵커")
print("=" * 88)
chk("학습(13–16) TT-1006 최대", tr["TT-1006.PV"].max(), 349.98, 0.01, " °C")
chk("전체 관측 TT-1006 최대", df["TT-1006.PV"].max(), 359.99, 0.01, " °C")

p_c = sig(m_c.predict(va[IC].values)); p_d = m_d.predict(Xva)
yv = va["X_conv_r1_deployed"].values
r2f = lambda a, b: float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))
chk("ONNX1 held-out RMSE", rmse(p_c, yv), 0.021, 5e-4)
chk("ONNX1 held-out R²", r2f(yv, p_c), 0.983, 5e-4)
chk("ONNX1 held-out MAE", float(np.mean(np.abs(p_c - yv))), 0.015, 5e-4)
chk("ONNX2 held-out RMSE", rmse(p_d, yva), 0.191, 5e-4, " kPa")
chk("ONNX2 held-out R²", r2f(yva, p_d), 0.995, 5e-4)
chk("ONNX2 held-out MAE", float(np.mean(np.abs(p_d - yva))), 0.146, 5e-4, " kPa")

onnx_dir = next(p for p in (ROOT / "2026-07-21_ML개발" / "onnx",
                            ROOT / "대단원3_보고서작성" / "요청 자료") if p.is_dir())
s1 = ort.InferenceSession(str(onnx_dir / "reactor_conversion_r1.onnx"),
                          providers=["CPUExecutionProvider"])
s2 = ort.InferenceSession(str(onnx_dir / "reactor_dp.onnx"), providers=["CPUExecutionProvider"])
run = lambda s, X: np.asarray(s.run(None, {"X": np.asarray(X, np.float32)})[0]).ravel()
chk("배포 ONNX1 in-sample RMSE", rmse(run(s1, va[IC].values), yv), 0.018, 5e-4)
chk("배포 ONNX2 in-sample RMSE", rmse(run(s2, Xva), yva), 0.114, 5e-4, " kPa")
anc = df[df.Campaign == "10/12"]
chk("10/12 앵커 배포 ONNX1 예측", float(run(s1, anc[IC].values).mean()), 0.6093, 5e-4)

# --------------------------------------------------------------- 판정
print("\n" + "=" * 88)
if FAIL:
    print(f"불일치 {len(FAIL)}건 — 원인을 규명할 것:")
    for x in FAIL:
        print("  -", x)
    sys.exit(1)
print("본문 수치 전부 재현됨.")
print("=" * 88)
