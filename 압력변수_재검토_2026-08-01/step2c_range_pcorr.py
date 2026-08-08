# -*- coding: utf-8 -*-
"""step2c — 케이스스터디 범위 vs 학습데이터 범위, 그리고 2차-로짓 기준 편상관 재계산."""
import numpy as np, pandas as pd
from pathlib import Path

ML = Path("/sessions/relaxed-focused-carson/mnt/공정설계대회/2026-07-21_ML개발")
df = pd.read_excel(ML/"data/processed/cumene_ml_training_data.xlsx", sheet_name="Data")
r1 = pd.read_csv(ML/"data/processed/labels_recycle1.csv")
df["Timestamp"]=pd.to_datetime(df["Timestamp"]); r1["Timestamp"]=pd.to_datetime(r1["Timestamp"])
df = df.merge(r1[["Timestamp","X_propene_recycle1"]], on="Timestamp")

T=df["TT-1006.PV"].to_numpy(); F=df["FT-1004.PV"].to_numpy(); R=df["FRC-1004.PV"].to_numpy()
tot = F*(1+R)
print("[G] 케이스스터디 지정 범위 vs 실측 데이터 범위")
print(f"  반응기 온도   문제 335~365 °C   | 데이터 TT-1006 {T.min():.2f}~{T.max():.2f} °C")
print(f"  반응기 총유량 문제 15000~18000 kg/h | 데이터 FT1004*(1+FRC) {tot.min():.1f}~{tot.max():.1f} kg/h")
print(f"  B/P 비율      문제 4~6          | 데이터 FRC-1004 {R.min():.3f}~{R.max():.3f}")
print(f"  (참고) 프로펜 유량 FT-1004 {F.min():.1f}~{F.max():.1f} kg/h")
print(f"  케이스스터디 꼭짓점 8개 중 데이터 볼록포 밖 여부는 아래 주변부 커버리지로 판단")
# 3차원 격자 커버리지
import itertools
pts = list(itertools.product([335,365],[15000,18000],[4,6]))
print("  케이스 꼭짓점(T, 총유량, B/P) 별 최근접 실측점 거리(정규화):")
S = np.column_stack([(T-T.mean())/T.std(), (tot-tot.mean())/tot.std(), (R-R.mean())/R.std()])
for p in pts:
    q = np.array([(p[0]-T.mean())/T.std(), (p[1]-tot.mean())/tot.std(), (p[2]-R.mean())/R.std()])
    d = np.sqrt(((S-q)**2).sum(1)).min()
    print(f"    {p}  최근접거리 {d:.3f}")

print()
print("[D-2] 2차항 기준 편상관 — 최종 레시피(로짓 2차)에 맞춘 통제")
P4=df["PT-1004.PV"].to_numpy(); P0=df["PT-1100.PV"].to_numpy()


def quad(Z):
    cols=[np.ones(len(Z))]+[Z[:,i] for i in range(Z.shape[1])]
    for i in range(Z.shape[1]):
        for j in range(i,Z.shape[1]): cols.append(Z[:,i]*Z[:,j])
    return np.column_stack(cols)


def pc(a,b,A):
    ra=a-A@np.linalg.lstsq(A,a,rcond=None)[0]; rb=b-A@np.linalg.lstsq(A,b,rcond=None)[0]
    return np.corrcoef(ra,rb)[0,1]


Z3=np.column_stack([T,F,R])
for basis_nm, A in [("1차(3항)", np.column_stack([np.ones(len(T)),Z3])), ("2차(10항)", quad(Z3))]:
    for lab in ["X_propene_conversion","X_propene_recycle1"]:
        y=df[lab].to_numpy()
        yl=np.log(np.clip(y,1e-6,1-1e-6)/(1-np.clip(y,1e-6,1-1e-6)))
        for ynm,yy in [("X",y),("logit(X)",yl)]:
            for sn,m in [("전체",np.ones(len(df),bool)),("학습",(df["Split"]=="학습").to_numpy()),
                         ("검증",(df["Split"]=="검증").to_numpy())]:
                ok=m&~np.isnan(yy)
                a=pc(yy[ok],P4[ok],A[ok]); b=pc(yy[ok],P0[ok],A[ok])
                print(f"  {basis_nm:9s} {lab[-10:]:10s} {ynm:8s} {sn:3s}  PT1004 {a:+.4f}({a*a*100:5.2f}%)  PT1100 {b:+.4f}({b*b*100:5.2f}%)")
