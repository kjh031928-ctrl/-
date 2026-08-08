# -*- coding: utf-8 -*-
"""step2b — [D] 편상관 부분집합별 재확인, [E] Ergun 형태 재검토(비음수 제약·절편)."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import nnls

ROOT = Path("/sessions/relaxed-focused-carson/mnt/공정설계대회"); ML = ROOT / "2026-07-21_ML개발"
MW = {"benzene": 78.115, "propene": 42.081, "propane": 44.097}
WP, WPA = 0.94773, 0.05227

df = pd.read_excel(ML / "data/processed/cumene_ml_training_data.xlsx", sheet_name="Data")
r1 = pd.read_csv(ML / "data/processed/labels_recycle1.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"]); r1["Timestamp"] = pd.to_datetime(r1["Timestamp"])
df = df.merge(r1[["Timestamp", "X_propene_recycle1"]], on="Timestamp")

T = df["TT-1006.PV"].to_numpy(); F1004 = df["FT-1004.PV"].to_numpy(); R = df["FRC-1004.PV"].to_numpy()
P4 = df["PT-1004.PV"].to_numpy(); P0 = df["PT-1100.PV"].to_numpy()
Z3 = np.column_stack([T, F1004, R])


def pcorr(a, b, Z):
    A = np.column_stack([np.ones(len(a)), Z])
    ra = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    rb = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    return np.corrcoef(ra, rb)[0, 1]


print("[D-재확인] 편상관(X, 압력 | 입력3종) — 부분집합·라벨별")
subsets = {"전체3066": np.ones(len(df), bool),
           "학습1924": (df["Split"] == "학습").to_numpy(),
           "검증661": (df["Split"] == "검증").to_numpy()}
for lab in ["X_propene_conversion", "X_propene_recycle1"]:
    y = df[lab].to_numpy()
    for sn, m in subsets.items():
        ok = m & ~np.isnan(y)
        a = pcorr(y[ok], P4[ok], Z3[ok]); b = pcorr(y[ok], P0[ok], Z3[ok])
        print(f"  {lab:22s} {sn:9s} PT-1004 {a:+.4f} ({a**2*100:5.2f}%)  "
              f"PT-1100 {b:+.4f} ({b**2*100:5.2f}%)")

print()
print("[E-재검토] DP 상관식 — 형태별 비교 (전체 3066행)")
tot = F1004 * (1 + R)
n_b = R*F1004/MW["benzene"]; n_p = WP*F1004/MW["propene"]; n_pa = WPA*F1004/MW["propane"]
Fk = n_b+n_p+n_pa; z = n_b/Fk
Nt, L, D = 234, 2.0, 0.0762
Ac = Nt*np.pi*D**2/4
G = (tot/3600.0)/Ac
MWa = tot/Fk
Pm = (P4-25+P0)/2*1000.0
Tm = ((T + df["TT-1100.PV"].to_numpy())/2)+273.15
rho = Pm*MWa/(8314.46*Tm)
DPr = P4 - P0            # 실측 차압 (배관+HX+반응기 전부 포함)
DPa = P4 - 25.0 - P0     # HX-1005 튜브측 사양 0.25 bar 를 뺀 '반응기 DP' (가정)


def rep(nm, y, pred):
    r2 = 1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
    print(f"    {nm:38s} R2={r2:8.5f}  RMSE={np.sqrt(((y-pred)**2).mean()):7.4f} kPa")


for tgt_nm, y in [("실측차압 PT1004-PT1100", DPr)]:
    print(f"  대상 = {tgt_nm}  (평균 {y.mean():.2f}, 폭 {y.max()-y.min():.2f} kPa)")
    v = L*G/rho; i = L*G**2/rho
    # 1) Ergun 2모수 (부호 자유)
    c, *_ = np.linalg.lstsq(np.column_stack([v, i]), y*1000, rcond=None)
    rep(f"Ergun 2모수 자유  a={c[0]:.3e} b={c[1]:.3e}", y, (np.column_stack([v, i])@c)/1000)
    # 2) Ergun 2모수 + 절편 (배관/HX 고정손실)
    A = np.column_stack([np.ones_like(y), v, i])
    c, *_ = np.linalg.lstsq(A, y*1000, rcond=None)
    rep(f"Ergun 2모수+절편 a={c[1]:.3e} b={c[2]:.3e}", y, (A@c)/1000)
    # 3) 비음수 제약 Ergun + 절편
    A2 = np.column_stack([np.ones_like(y), v, i])
    cn, _ = nnls(A2, y*1000)
    rep(f"Ergun NNLS+절편  a={cn[1]:.3e} b={cn[2]:.3e}", y, (A2@cn)/1000)
    # 4) 팀 식 형태 4모수 선형
    A3 = np.column_stack([np.ones_like(y), T, Fk, z])
    c3, *_ = np.linalg.lstsq(A3, y, rcond=None)
    rep("팀 식(2) 형태 4모수 선형(T,F,z)", y, A3@c3)
    # 5) 총질량유량 2차 (2모수)
    A4 = np.column_stack([np.ones_like(y), tot, tot**2])
    c4, *_ = np.linalg.lstsq(A4, y, rcond=None)
    rep("총질량유량 2차 (3모수)", y, A4@c4)
    print(f"    Ergun 점성/관성 항 비 (자유적합, 평균): "
          f"{(c[1]*v).mean()/(c[2]*i).mean():+.4f}")
    print(f"    G 범위 {G.min():.3f}~{G.max():.3f} kg/m2/s, rho 범위 {rho.min():.3f}~{rho.max():.3f} kg/m3")
    print(f"    Re_p 대략(dp=3mm, mu=1.2e-5): {(G*0.003/1.2e-5).min():.0f}~{(G*0.003/1.2e-5).max():.0f} → 관성 지배")
