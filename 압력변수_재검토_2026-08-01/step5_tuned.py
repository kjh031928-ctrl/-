# -*- coding: utf-8 -*-
"""step5 — (a) 학습/검증 캠페인의 압력 범위 겹침 진단
          (b) 하이퍼파라미터를 세트별로 튜닝(학습 캠페인 내 LOCO)한 뒤 재비교
             → '튜닝을 안 해서 생긴 순위 역전'이라는 반론을 차단."""
import sys, warnings, itertools
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore")
SEED = 42
ML = Path("/sessions/relaxed-focused-carson/mnt/공정설계대회/2026-07-21_ML개발")
CK = Path("/sessions/relaxed-focused-carson/mnt/outputs/ckpt")

df = pd.read_excel(ML/"data/processed/cumene_ml_training_data.xlsx", sheet_name="Data")
r1 = pd.read_csv(ML/"data/processed/labels_recycle1.csv")
df["Timestamp"]=pd.to_datetime(df["Timestamp"]); r1["Timestamp"]=pd.to_datetime(r1["Timestamp"])
df = df.merge(r1[["Timestamp","X_propene_recycle1"]], on="Timestamp")
df = df[df["Campaign"]!="10/12"].reset_index(drop=True)
df["DP_kPa"] = df["PT-1004.PV"]-df["PT-1100.PV"]
df = df.dropna(subset=["X_propene_recycle1"]).reset_index(drop=True)
LABEL="X_propene_recycle1"

if sys.argv[1] == "diag":
    print("[H] 학습 캠페인(10/13~16) vs 검증 캠페인(10/17-18) 변수 범위 겹침")
    tr = df[df["Split"]=="학습"]; te = df[df["Split"]=="검증"]
    for c in ["TT-1006.PV","FT-1004.PV","FRC-1004.PV","PT-1004.PV","PT-1100.PV","DP_kPa"]:
        a,b = tr[c].min(), tr[c].max(); u,v = te[c].min(), te[c].max()
        ov = max(0, min(b,v)-max(a,u)) / (v-u) if v>u else np.nan
        out = ((te[c]<a)|(te[c]>b)).mean()*100
        print(f"  {c:12s} 학습 [{a:9.3f}, {b:9.3f}]  검증 [{u:9.3f}, {v:9.3f}]  "
              f"검증범위 겹침 {ov*100:5.1f}%  학습범위 밖 검증행 {out:5.1f}%")
    print()
    print("  캠페인별 압력 범위:")
    for c, g in df.groupby("Campaign"):
        print(f"    {c:9s} PT-1004 [{g['PT-1004.PV'].min():8.1f},{g['PT-1004.PV'].max():8.1f}] "
              f"PT-1100 [{g['PT-1100.PV'].min():8.1f},{g['PT-1100.PV'].max():8.1f}] "
              f"DP [{g['DP_kPa'].min():6.2f},{g['DP_kPa'].max():6.2f}]")
    sys.exit()

FEATS = {"I1_TFR":["TT-1006.PV","FT-1004.PV","FRC-1004.PV"],
         "I2_+PT1004":["TT-1006.PV","FT-1004.PV","FRC-1004.PV","PT-1004.PV"],
         "I3_+PT1100":["TT-1006.PV","FT-1004.PV","FRC-1004.PV","PT-1100.PV"],
         "I4_+양쪽압력":["TT-1006.PV","FT-1004.PV","FRC-1004.PV","PT-1004.PV","PT-1100.PV"],
         "I5_+DP":["TT-1006.PV","FT-1004.PV","FRC-1004.PV","DP_kPa"]}
GRIDS = {
 "Linear2": [dict()],
 "Ridge2":  [dict(alpha=a) for a in [1e-3,1e-1,1,10,100,1000]],
 "KNN":     [dict(n_neighbors=k, weights=w) for k in [3,5,10,25,50] for w in ["uniform","distance"]],
 "SVR-RBF": [dict(C=c, gamma=g, epsilon=e) for c in [10,300] for g in ["scale",1.0] for e in [0.001]],
 "RF":      [dict(n_estimators=200, max_depth=d, min_samples_leaf=l, random_state=SEED, n_jobs=-1)
             for d in [None,6,12] for l in [1,5,20]],
 "GBR":     [dict(n_estimators=n, learning_rate=lr, max_depth=d, random_state=SEED)
             for n in [300] for lr in [0.03,0.1] for d in [2,3,5]],
 "MLP":     [dict(hidden_layer_sizes=h, alpha=a, max_iter=2000, random_state=SEED)
             for h in [(16,),(64,64),(128,128)] for a in [1e-5,1e-2,1.0]],
}


def build(name, hp):
    if name=="Linear2": return make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression())
    if name=="Ridge2":  return make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(**hp))
    if name=="KNN":     return make_pipeline(StandardScaler(), KNeighborsRegressor(**hp))
    if name=="SVR-RBF": return make_pipeline(StandardScaler(), SVR(**hp))
    if name=="RF":      return RandomForestRegressor(**hp)
    if name=="GBR":     return GradientBoostingRegressor(**hp)
    if name=="MLP":     return make_pipeline(StandardScaler(), MLPRegressor(**hp))


def lg(y): c=np.clip(y,1e-6,1-1e-6); return np.log(c/(1-c))
def il(z): return 1/(1+np.exp(-np.clip(z,-30,30)))


mi = int(sys.argv[1]); mname = list(GRIDS)[mi]
tr_all = df[df["Split"]=="학습"]; te = df[df["Split"]=="검증"]
TRC = ["10/13","10/14","10/15","10/16"]
out=[]
FSEL = list(FEATS) if len(sys.argv)<3 else [list(FEATS)[int(sys.argv[2])]]
for fs in FSEL:
    cols = FEATS[fs]
    best=(1e9,None)
    for hp in GRIDS[mname]:
        errs=[]
        for c in TRC:
            a=tr_all[tr_all.Campaign!=c]; b=tr_all[tr_all.Campaign==c]
            m=build(mname,hp); m.fit(a[cols].to_numpy(float), lg(a[LABEL].to_numpy(float)))
            p=il(m.predict(b[cols].to_numpy(float)))
            errs.append(np.sqrt(((b[LABEL].to_numpy(float)-p)**2).mean()))
        e=float(np.mean(errs))
        if e<best[0]: best=(e,hp)
    m=build(mname,best[1]); m.fit(tr_all[cols].to_numpy(float), lg(tr_all[LABEL].to_numpy(float)))
    p=il(m.predict(te[cols].to_numpy(float)))
    rm=float(np.sqrt(((te[LABEL].to_numpy(float)-p)**2).mean()))
    out.append(dict(model=mname,fset=fs,cv_rmse=best[0],test_rmse=rm,hp=str(best[1])))
    np.save(CK/f"tuned_pred_{mname}_{fs}.npy", p)
    print(f"  {mname:8s} {fs:12s} CV={best[0]:.5f} TEST={rm:.5f}  hp={best[1]}", flush=True)
pd.DataFrame(out).to_csv(CK/f"tuned_{mi}_{sys.argv[2] if len(sys.argv)>2 else 'all'}.csv", index=False, encoding="utf-8-sig")
