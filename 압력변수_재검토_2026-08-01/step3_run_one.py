# -*- coding: utf-8 -*-
"""step3_run_one.py <fset_index> — 입력세트 하나에 대해 full factorial 조각 실행 후 체크포인트 저장."""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore")

SEED = 42
ML = Path("/sessions/relaxed-focused-carson/mnt/공정설계대회/2026-07-21_ML개발")
OUT = Path("/sessions/relaxed-focused-carson/mnt/outputs/ckpt"); OUT.mkdir(exist_ok=True)

df = pd.read_excel(ML/"data/processed/cumene_ml_training_data.xlsx", sheet_name="Data")
r1 = pd.read_csv(ML/"data/processed/labels_recycle1.csv")
df["Timestamp"]=pd.to_datetime(df["Timestamp"]); r1["Timestamp"]=pd.to_datetime(r1["Timestamp"])
df = df.merge(r1[["Timestamp","X_propene_recycle1"]], on="Timestamp")
df = df[df["Campaign"] != "10/12"].reset_index(drop=True)
df["DP_kPa"] = df["PT-1004.PV"] - df["PT-1100.PV"]
df = df.dropna(subset=["X_propene_recycle1"]).reset_index(drop=True)

LABEL = "X_propene_recycle1"
FEATS = {
    "I1_TFR":        ["TT-1006.PV","FT-1004.PV","FRC-1004.PV"],
    "I2_+PT1004":    ["TT-1006.PV","FT-1004.PV","FRC-1004.PV","PT-1004.PV"],
    "I3_+PT1100":    ["TT-1006.PV","FT-1004.PV","FRC-1004.PV","PT-1100.PV"],
    "I4_+양쪽압력":   ["TT-1006.PV","FT-1004.PV","FRC-1004.PV","PT-1004.PV","PT-1100.PV"],
    "I5_+DP":        ["TT-1006.PV","FT-1004.PV","FRC-1004.PV","DP_kPa"],
}
MODELS = {
    "Linear1":   lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "Linear2":   lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()),
    "Ridge2":    lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1.0)),
    "KNN":       lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    "SVR-RBF":   lambda: make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale", epsilon=0.001)),
    "RF":        lambda: RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
    "ExtraTrees":lambda: ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
    "GBR":       lambda: GradientBoostingRegressor(random_state=SEED),
    "HistGB":    lambda: HistGradientBoostingRegressor(random_state=SEED),
    "MLP-64x64": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64,64),
                            max_iter=2000, random_state=SEED)),
    "MLP-16":    lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,),
                            max_iter=3000, random_state=SEED)),
}
CAMPS = ["10/13","10/14","10/15","10/16","10/17-18"]


def to_t(y, tf):
    if tf == "raw": return y
    c = np.clip(y, 1e-6, 1-1e-6); return np.log(c/(1-c))


def from_t(z, tf):
    return z if tf == "raw" else 1/(1+np.exp(-np.clip(z,-30,30)))


fi = int(sys.argv[1]); fname = list(FEATS)[fi]; cols = FEATS[fname]
rows = []; preds = {}
tr = df[df["Split"]=="학습"]; te = df[df["Split"]=="검증"]
for tf in ["raw","logit"]:
    for mn, mk in MODELS.items():
        m = mk(); m.fit(tr[cols].to_numpy(float), to_t(tr[LABEL].to_numpy(float), tf))
        p_te = from_t(m.predict(te[cols].to_numpy(float)), tf)
        p_tr = from_t(m.predict(tr[cols].to_numpy(float)), tf)
        y_te = te[LABEL].to_numpy(float); y_tr = tr[LABEL].to_numpy(float)
        preds[f"{mn}|{tf}"] = p_te
        loco = []
        for c in CAMPS:
            t2 = df[df["Campaign"]!=c]; v2 = df[df["Campaign"]==c]
            m2 = mk(); m2.fit(t2[cols].to_numpy(float), to_t(t2[LABEL].to_numpy(float), tf))
            pp = from_t(m2.predict(v2[cols].to_numpy(float)), tf)
            loco.append(float(np.sqrt(((v2[LABEL].to_numpy(float)-pp)**2).mean())))
        rows.append(dict(model=mn, fset=fname, target=tf,
                         rmse_train=float(np.sqrt(((y_tr-p_tr)**2).mean())),
                         rmse_fixed=float(np.sqrt(((y_te-p_te)**2).mean())),
                         loco_mean=float(np.mean(loco)), loco_max=float(np.max(loco)),
                         **{f"loco_{c}": v for c, v in zip(CAMPS, loco)}))
        print(f"  {fname} {tf} {mn} ok", flush=True)
pd.DataFrame(rows).to_csv(OUT/f"res_{fi}.csv", index=False, encoding="utf-8-sig")
np.savez(OUT/f"preds_{fi}.npz", y=te[LABEL].to_numpy(float), **preds)
print("SAVED", fname)
