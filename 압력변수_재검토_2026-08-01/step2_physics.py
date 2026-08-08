# -*- coding: utf-8 -*-
"""step2 — (a) 회귀식 변수 (F, z) 가 ML 입력 (FT-1004, FRC-1004) 의 정확한 대수 함수임을 검증
          (b) 압력이 독립 자유도인지 (종속 상태량인지) 검증
          (c) 기상 PBR 물리: DP/P 크기, 압력이 X 에 주는 영향의 상한
          (d) Ergun 형태 DP 상관식 적합 → 4항 선형회귀와 비교
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/sessions/relaxed-focused-carson/mnt/공정설계대회")
ML = ROOT / "2026-07-21_ML개발"
MW = {"benzene": 78.115, "propene": 42.081, "propane": 44.097, "cumene": 120.196}
WP, WPA = 0.94773, 0.05227   # 프로펜 피드 질량분율 (Appendix B Table 3)

df = pd.read_excel(ML / "data/processed/cumene_ml_training_data.xlsx", sheet_name="Data")
r1 = pd.read_csv(ML / "data/processed/labels_recycle1.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"]); r1["Timestamp"] = pd.to_datetime(r1["Timestamp"])
df = df.merge(r1[["Timestamp", "X_propene_recycle1"]], on="Timestamp")

F1004 = df["FT-1004.PV"].to_numpy(); RATIO = df["FRC-1004.PV"].to_numpy()
T = df["TT-1006.PV"].to_numpy(); P1004 = df["PT-1004.PV"].to_numpy(); P1100 = df["PT-1100.PV"].to_numpy()
X = df["X_propene_conversion"].to_numpy()

print("=" * 92)
print("[A] (F_in, z_BENZENE) 가 (FT-1004, FRC-1004) 의 정확한 대수 함수인가?")
# 비율제어기 정의: 벤젠 질량유량 = FRC-1004 x FT-1004
n_b = RATIO * F1004 / MW["benzene"]
n_p = WP * F1004 / MW["propene"]
n_pa = WPA * F1004 / MW["propane"]
F_alg = n_b + n_p + n_pa
z_alg = n_b / F_alg
F_dat = (df["n_benzene_kmol_h"] + df["n_propene_kmol_h"] + df["n_propane_kmol_h"]).to_numpy()
z_dat = (df["n_benzene_kmol_h"] / F_dat).to_numpy()
print(f"  F_in  대수식 vs 데이터파생: 최대 상대오차 {np.abs(F_alg-F_dat).max()/F_dat.mean()*100:.4f} %"
      f"   상관 {np.corrcoef(F_alg,F_dat)[0,1]:.8f}")
print(f"  z_BEN 대수식 vs 데이터파생: 최대 절대오차 {np.abs(z_alg-z_dat).max():.6f}"
      f"   상관 {np.corrcoef(z_alg,z_dat)[0,1]:.8f}")
print(f"  → z_BENZENE 는 FRC-1004 만의 함수: z = R/MWb / (R/MWb + wp/MWp + wpa/MWpa)")
print(f"     FRC-1004 고유값 개수 {len(np.unique(np.round(RATIO,6)))}, "
      f"z_alg 고유값 개수 {len(np.unique(np.round(z_alg,9)))}")

print()
print("[B] 압력은 독립 자유도인가, 아니면 유량이 정하는 종속 상태량인가?")
tot_mass = F1004 * (1 + RATIO)                      # 반응기 총 질량유량 kg/h
for nm, y in [("PT-1004", P1004), ("PT-1100", P1100)]:
    for lbl, A in [("총질량유량 1차", np.column_stack([np.ones_like(y), tot_mass])),
                   ("총질량유량 2차", np.column_stack([np.ones_like(y), tot_mass, tot_mass**2])),
                   ("총질량유량2차+T", np.column_stack([np.ones_like(y), tot_mass, tot_mass**2, T]))]:
        c, *_ = np.linalg.lstsq(A, y, rcond=None); pr = A @ c
        r2 = 1 - ((y-pr)**2).sum()/((y-y.mean())**2).sum()
        print(f"  {nm} ~ {lbl:16s} R2={r2:.5f}  RMSE={np.sqrt(((y-pr)**2).mean()):8.3f} kPa")

print()
print("[C] 기상 PBR 물리 — 압력강하의 상대 크기와 X 에 대한 감도 상한")
DP = P1004 - 25.0 - P1100
print(f"  반응기 DP(= PT1004-25-PT1100): 평균 {DP.mean():.2f} kPa, 범위 {DP.min():.2f}~{DP.max():.2f}")
print(f"  DP / P_in = {DP.mean()/(P1004.mean()-25):.4f}  → 반응기는 사실상 등압 (압력강하 {DP.mean()/(P1004.mean()-25)*100:.2f}%)")
print(f"  절대압 P_out: {P1100.min():.1f}~{P1100.max():.1f} kPa (폭 {(P1100.max()-P1100.min())/P1100.mean()*100:.1f}%)")
# r ∝ P^2.31 (CLAUDE.md 적합 속도식) 기준, 압력 변동이 만드는 X 변동
n_ord = 2.31
print(f"  속도식 r ∝ P^{n_ord} 가정 시, 전 구간 압력 폭이 만드는 속도 변화 배수: "
      f"{(P1100.max()/P1100.min())**n_ord:.3f}배")
# 10/14 (입력 3종 고정) 구간에서 압력만 움직였을 때의 X 변화
m14 = (df["Campaign"] == "10/14").to_numpy()
print(f"  10/14 (입력3종 고정): P 폭 {P1100[m14].max()-P1100[m14].min():.2f} kPa "
      f"({(P1100[m14].max()-P1100[m14].min())/P1100[m14].mean()*100:.3f}%), "
      f"X 폭 {X[m14].max()-X[m14].min():.5f}")

print()
print("[D] 압력을 넣었을 때 남는 고유 정보 (편상관)")


def partial_corr(a, b, Z):
    A = np.column_stack([np.ones(len(a)), Z])
    ra = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    rb = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    return np.corrcoef(ra, rb)[0, 1]


Z3 = np.column_stack([T, F1004, RATIO])
for nm, p in [("PT-1004", P1004), ("PT-1100", P1100)]:
    pc = partial_corr(X, p, Z3)
    print(f"  편상관(X, {nm} | 입력3종) = {pc:+.4f}  → 잔차 설명력 {pc**2*100:.2f} %")

print()
print("[E] Ergun 형태 DP 상관식 (물리 형태 2모수) vs 4모수 선형회귀")
# G = 질량유량/단면적, rho = 이상기체 근사 (SRK 대신 크기 비교 목적)
Nt, L, D = 234, 2.0, 0.0762
Ac = Nt * np.pi * D**2 / 4
G = (tot_mass / 3600.0) / Ac                       # kg/m2/s
MWavg = tot_mass / (F_alg)                          # kg/kmol
Pm = (P1004 - 25 + P1100) / 2 * 1000.0              # Pa
Tm = (T + df["TT-1100.PV"].to_numpy()) / 2 + 273.15
rho = Pm * MWavg / (8314.46 * Tm)                   # kg/m3
# Ergun: DP = L*[ a*G/rho + b*G^2/rho ]  (a 는 점성항 계수, b 는 관성항 계수; dp, eps 를 흡수)
A_erg = np.column_stack([L * G / rho, L * G**2 / rho])
c_e, *_ = np.linalg.lstsq(A_erg, DP * 1000.0, rcond=None)
pr_e = (A_erg @ c_e) / 1000.0
r2e = 1 - ((DP-pr_e)**2).sum()/((DP-DP.mean())**2).sum()
print(f"  Ergun 2모수: a={c_e[0]:.4e}, b={c_e[1]:.4e}   R2={r2e:.5f} "
      f"RMSE={np.sqrt(((DP-pr_e)**2).mean()):.4f} kPa")
A_lin = np.column_stack([np.ones_like(DP), T, F_alg, z_alg])
c_l, *_ = np.linalg.lstsq(A_lin, DP, rcond=None); pr_l = A_lin @ c_l
r2l = 1 - ((DP-pr_l)**2).sum()/((DP-DP.mean())**2).sum()
print(f"  4모수 선형회귀(팀 식2 형태): R2={r2l:.5f} RMSE={np.sqrt(((DP-pr_l)**2).mean()):.4f} kPa")
# 순수 관성항만 (b 만) — 난류 지배 확인
c_b = np.linalg.lstsq(A_erg[:, [1]], DP*1000.0, rcond=None)[0]
pr_b = (A_erg[:, [1]] @ c_b)/1000.0
print(f"  Ergun 관성항 1모수: b={c_b[0]:.4e}  R2={1-((DP-pr_b)**2).sum()/((DP-DP.mean())**2).sum():.5f} "
      f"RMSE={np.sqrt(((DP-pr_b)**2).mean()):.4f} kPa")
print(f"  점성항/관성항 기여비(평균): {(c_e[0]*L*G/rho).mean()/(c_e[1]*L*G**2/rho).mean():.4f}")

print()
print("[F] 두 식의 동시 사용 = 압력 마디 과잉지정 여부")
eq1 = -226.32 - 0.5904*T + 8.7337*F_alg + 886.994*z_alg
eq2 = -60.3808 + 0.08516*T + 0.21436*F_alg + 5.60222*z_alg
print(f"  식(1) 예측 P_out : 평균 {eq1.mean():.2f} kPa (실측 PT-1100 평균 {P1100.mean():.2f}) "
      f"RMSE {np.sqrt(((eq1-P1100)**2).mean()):.2f}")
print(f"  식(2) 예측 DP    : 평균 {eq2.mean():.2f} kPa (실측 DP 평균 {DP.mean():.2f}) "
      f"RMSE {np.sqrt(((eq2-DP)**2).mean()):.3f}")
Pin_imp = eq1 + eq2
Pin_meas = P1004 - 25.0
print(f"  식(1)+식(2) 가 함축하는 반응기 입구압 : 평균 {Pin_imp.mean():.2f} kPa")
print(f"  실측 기반 반응기 입구압(PT-1004-25)   : 평균 {Pin_meas.mean():.2f} kPa  "
      f"RMSE {np.sqrt(((Pin_imp-Pin_meas)**2).mean()):.2f} kPa")
print("  → 두 식을 동시에 쓰면 반응기 입구압까지 지정된다. 상류 플로우시트가 계산하는 입구압과 충돌.")
