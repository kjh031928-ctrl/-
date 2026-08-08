# -*- coding: utf-8 -*-
"""step1 — 시뮬레이션에 쓰인 두 회귀식 (1),(2) 를 실측 CSV 로 재현한다.

(1) R-1100_out.P = -226.32 - 0.5904*T[C] + 8.7337*F[kmol/h] + 886.994*z_BENZENE
(2) R-1100.DP    = -60.3808 + 0.08516*T[C] + 0.21436*F[kmol/h] + 5.60222*z_BENZENE
    (T, F, z 는 HX-1005 Tube Out = 반응기 입구 스트림)
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/sessions/relaxed-focused-carson/mnt/공정설계대회")
ML = ROOT / "2026-07-21_ML개발"

MW = {"benzene": 78.115, "propene": 42.081, "propane": 44.097, "cumene": 120.196}

df = pd.read_excel(ML / "data/processed/cumene_ml_training_data.xlsx", sheet_name="Data")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# --- 반응기 입구 스트림(HX-1005 Tube Out) 몰유량·조성 재구성 -------------------
# 재순환 = 벤젠 99 mol% + 큐멘 1 mol% (r1 가정). r0 = 재순환 전량 벤젠.
mw_rec = 0.99 * MW["benzene"] + 0.01 * MW["cumene"]
n_rec = df["FT-1300.PV"] / mw_rec                       # kmol/h
n_benz_r1 = df["FT-1001.PV"] / MW["benzene"] + n_rec * 0.99
n_cum_r1 = n_rec * 0.01
n_benz_r0 = df["n_benzene_kmol_h"]
n_prop = df["n_propene_kmol_h"]
n_pane = df["n_propane_kmol_h"]

F_r1 = n_benz_r1 + n_cum_r1 + n_prop + n_pane
z_r1 = n_benz_r1 / F_r1
F_r0 = n_benz_r0 + n_prop + n_pane
z_r0 = n_benz_r0 / F_r0

T = df["TT-1006.PV"].to_numpy()
Pout = df["PT-1100.PV"].to_numpy()
P1004 = df["PT-1004.PV"].to_numpy()


def fit(y, T, F, z, name):
    A = np.column_stack([np.ones_like(T), T, F, z])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ c
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rmse = np.sqrt(((y - pred) ** 2).mean())
    print(f"  {name:26s} b0={c[0]:11.4f} bT={c[1]:9.5f} bF={c[2]:9.5f} bz={c[3]:11.4f}"
          f"  R2={r2:.5f} RMSE={rmse:.4f}")
    return c


print("=" * 96)
print("[A] 식 (1) 재현 : y = PT-1100 (반응기 출구압, kPa)")
print("    문헌(팀 시뮬)     b0= -226.3200 bT= -0.59040 bF=  8.73370 bz=   886.9940")
for tag, F, z in [("r1(재순환 큐멘1%)", F_r1.to_numpy(), z_r1.to_numpy()),
                  ("r0(재순환 전량벤젠)", F_r0.to_numpy(), z_r0.to_numpy())]:
    for sub, m in [("전체3066", np.ones(len(df), bool)),
                   ("학습1924", (df["Split"] == "학습").to_numpy())]:
        fit(Pout[m], T[m], F[m], z[m], f"{tag}/{sub}")

print()
print("[B] 식 (2) 재현 : y = 반응기 DP 후보")
print("    문헌(팀 시뮬)     b0=  -60.3808 bT=  0.08516 bF=  0.21436 bz=     5.6022")
cands = {
    "PT1004-PT1100": P1004 - Pout,
    "PT1004-25-PT1100": P1004 - 25.0 - Pout,   # HX-1005 튜브측 DP 0.25 bar 차감
    "PT1004-50-PT1100": P1004 - 50.0 - Pout,
}
for cname, y in cands.items():
    print(f"  --- 후보 DP = {cname}  (평균 {y.mean():.3f} kPa, 범위 {y.min():.3f}~{y.max():.3f})")
    for tag, F, z in [("r1", F_r1.to_numpy(), z_r1.to_numpy()), ("r0", F_r0.to_numpy(), z_r0.to_numpy())]:
        for sub, m in [("전체3066", np.ones(len(df), bool)),
                       ("학습1924", (df["Split"] == "학습").to_numpy())]:
            fit(y[m], T[m], F[m], z[m], f"{tag}/{sub}")

print()
print("[C] 회귀변수 (T, F, z) 와 ML 입력 (TT-1006, FT-1004, FRC-1004) 의 정보 동등성")
X3 = np.column_stack([np.ones(len(df)), df["TT-1006.PV"], df["FT-1004.PV"], df["FRC-1004.PV"]])
for nm, tgt in [("F_in [kmol/h] (r1)", F_r1.to_numpy()), ("z_BENZENE (r1)", z_r1.to_numpy()),
                ("F_in [kmol/h] (r0)", F_r0.to_numpy()), ("z_BENZENE (r0)", z_r0.to_numpy())]:
    c, *_ = np.linalg.lstsq(X3, tgt, rcond=None)
    pr = X3 @ c
    r2 = 1 - ((tgt - pr) ** 2).sum() / ((tgt - tgt.mean()) ** 2).sum()
    print(f"  {nm:22s} ~ (TT-1006,FT-1004,FRC-1004) 선형 R2 = {r2:.6f}"
          f"   상대오차 max = {np.abs(tgt-pr).max()/np.abs(tgt).mean()*100:.4f} %")

print()
print("[D] 압력 자체를 ML 입력 3종으로 회귀 (CLAUDE.md 기존값 검증)")
for tag in ["PT-1004.PV", "PT-1100.PV"]:
    y = df[tag].to_numpy()
    c, *_ = np.linalg.lstsq(X3, y, rcond=None)
    pr = X3 @ c
    r2 = 1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    vif = 1 / (1 - r2)
    print(f"  {tag}: R2={r2:.4f}  VIF={vif:.1f}  (전체 3066행)")
    m = (df["Split"] == "학습").to_numpy()
    c2, *_ = np.linalg.lstsq(X3[m], y[m], rcond=None)
    pr2 = X3[m] @ c2
    r22 = 1 - ((y[m] - pr2) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
    print(f"           학습셋만 R2={r22:.4f}  VIF={1/(1-r22):.1f}")

print()
print("[E] 압력 변동 폭 (전체 / 캠페인별)")
for tag in ["PT-1004.PV", "PT-1100.PV"]:
    s = df[tag]
    print(f"  {tag}: 전체 {s.min():.2f} ~ {s.max():.2f} kPa (폭 {s.max()-s.min():.2f}, "
          f"평균대비 {(s.max()-s.min())/s.mean()*100:.2f} %)")
dp = P1004 - Pout
print(f"  PT1004-PT1100 : {dp.min():.2f} ~ {dp.max():.2f} kPa")
print(df.groupby("Campaign").apply(
    lambda g: pd.Series({"PT1004_폭": g["PT-1004.PV"].max()-g["PT-1004.PV"].min(),
                         "PT1100_폭": g["PT-1100.PV"].max()-g["PT-1100.PV"].min(),
                         "X_폭": g["X_propene_conversion"].max()-g["X_propene_conversion"].min()}),
    include_groups=False).round(3))
