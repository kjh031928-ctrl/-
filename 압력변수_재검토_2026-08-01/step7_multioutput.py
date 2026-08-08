# -*- coding: utf-8 -*-
"""step7 — 제안안 검증: 입력 (TT-1006, FT-1004, FRC-1004) → 출력 (X, DP_reactor) 다중출력.
DP 를 ML 이 직접 내놓으면 팀의 식(2) 와 성능이 같은가? 식(1) 은 필요 없는가?"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

ML = Path("/sessions/relaxed-focused-carson/mnt/공정설계대회/2026-07-21_ML개발")
MW = {"benzene":78.115,"propene":42.081,"propane":44.097}; WP,WPA = 0.94773,0.05227
df = pd.read_excel(ML/"data/processed/cumene_ml_training_data.xlsx", sheet_name="Data")
df = df[df.Campaign!="10/12"].reset_index(drop=True)
T=df["TT-1006.PV"].to_numpy(); F=df["FT-1004.PV"].to_numpy(); R=df["FRC-1004.PV"].to_numpy()
P4=df["PT-1004.PV"].to_numpy(); P0=df["PT-1100.PV"].to_numpy()
DP = P4 - 25.0 - P0                              # 팀이 식(2)의 타깃으로 쓴 양
n_b=R*F/MW["benzene"]; n_p=WP*F/MW["propene"]; n_pa=WPA*F/MW["propane"]
Fk=n_b+n_p+n_pa; z=n_b/Fk
tr = (df.Split=="학습").to_numpy(); te = (df.Split=="검증").to_numpy()

print("[제안안 검증] 반응기 DP 를 '입력 3종(TT-1006, FT-1004, FRC-1004)' 만으로 예측")
print("  (팀의 식(2)는 (T, F_kmol/h, z_BEN) 을 쓰지만 이 셋은 입력 3종의 정확한 함수)")


def ev(nm, model, Xall):
    model.fit(Xall[tr], DP[tr]); p = model.predict(Xall)
    print(f"  {nm:34s} 학습RMSE={np.sqrt(((DP[tr]-p[tr])**2).mean()):.4f}  "
          f"검증RMSE={np.sqrt(((DP[te]-p[te])**2).mean()):.4f} kPa  "
          f"검증 최대오차={np.abs(DP[te]-p[te]).max():.4f}")


ev("팀 식(2) 형태: 1차 (T,F_k,z)", LinearRegression(), np.column_stack([T,Fk,z]))
ev("입력3종 1차 (T,F,R)", LinearRegression(), np.column_stack([T,F,R]))
ev("입력3종 2차 (T,F,R)", make_pipeline(StandardScaler(),PolynomialFeatures(2),LinearRegression()),
   np.column_stack([T,F,R]))
ev("입력3종 3차 (T,F,R)", make_pipeline(StandardScaler(),PolynomialFeatures(3),LinearRegression()),
   np.column_stack([T,F,R]))
# 팀 식(2) 계수를 그대로 썼을 때 (전체 데이터 적합이므로 참고용)
eq2 = -60.3808 + 0.08516*T + 0.21436*Fk + 5.60222*z
print(f"  {'팀 식(2) 계수 그대로':34s} 학습RMSE={np.sqrt(((DP[tr]-eq2[tr])**2).mean()):.4f}  "
      f"검증RMSE={np.sqrt(((DP[te]-eq2[te])**2).mean()):.4f} kPa  "
      f"검증 최대오차={np.abs(DP[te]-eq2[te]).max():.4f}")

print()
print("[식(1) 이 왜 불필요한가] 식(1) 예측 P_out 과 '상류압력 - DP' 의 차이")
eq1 = -226.32 - 0.5904*T + 8.7337*Fk + 886.994*z
alt = (P4 - 25.0) - eq2                 # 플로우시트가 계산한 입구압 - DP  → 출구압
print(f"  식(1) 로 낸 P_out  : 검증 RMSE(vs PT-1100) = {np.sqrt(((eq1[te]-P0[te])**2).mean()):.3f} kPa")
print(f"  (입구압 - 식(2)) 로 낸 P_out : 검증 RMSE = {np.sqrt(((alt[te]-P0[te])**2).mean()):.3f} kPa")
print("  → 입구압이 상류에서 이미 계산되므로 DP 하나면 출구압이 닫힌다. 식(1)은 중복 지정.")

print()
print("[반응 화학량론 확인] 2 mol → 1 mol 이므로 반응기에서 몰유량 감소")
X = df["X_propene_conversion"].to_numpy()
dn = -n_p*X                                        # 총 몰 변화 (프로펜 1 + 벤젠 1 → 큐멘 1)
print(f"  전환율 평균 {X.mean():.4f} → 총 몰유량 감소율 평균 {(-dn/Fk).mean()*100:.2f} %")
print(f"  즉 출구 몰유량이 줄어 DP 도 줄고 출구압이 올라간다 (PT-1100 은 결과 쪽 변수)")
