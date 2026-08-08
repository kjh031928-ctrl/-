# -*- coding: utf-8 -*-
"""step6 — 세트별 튜닝 후 순위 비교 + 블록 부트스트랩 역전 검정."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

CK = Path("/sessions/relaxed-focused-carson/mnt/outputs/ckpt")
FS = ["I1_TFR","I2_+PT1004","I3_+PT1100","I4_+양쪽압력","I5_+DP"]
MODELS = ["Linear2","Ridge2","KNN","RF","GBR","MLP"]

t = pd.concat([pd.read_csv(f) for f in sorted(CK.glob("tuned_*.csv"))], ignore_index=True)
t = t.drop_duplicates(["model","fset"], keep="last")

# Linear1 (하이퍼파라미터 없음) 은 step3 결과를 그대로 사용
r3 = pd.concat([pd.read_csv(CK/f"res_{i}.csv") for i in range(5)], ignore_index=True)
r3 = r3[(r3.target=="logit") & (r3.model=="Linear1")][["model","fset","rmse_fixed"]]
r3 = r3.rename(columns={"rmse_fixed":"test_rmse"}); r3["cv_rmse"]=np.nan; r3["hp"]="-"
t = pd.concat([t, r3], ignore_index=True)

piv = t.pivot(index="model", columns="fset", values="test_rmse")[FS]
print("="*96)
print("[튜닝 후] 검증 RMSE (학습 10/13~16 → 검증 10/17-18, 타깃 logit)")
print("  하이퍼파라미터는 입력세트마다 따로 튜닝 (학습 캠페인 내 leave-one-campaign-out)")
print(piv.round(5).to_string())
print("\n  세트별 순위 (1위→):")
for c in FS:
    order = piv[c].sort_values().index.tolist()
    print(f"    {c:14s} " + " > ".join(order))
print("\n  I1 기준 순위상관 (Spearman):")
for c in FS:
    a=piv["I1_TFR"].to_numpy(float); b=piv[c].to_numpy(float)
    print(f"    I1 vs {c:14s} {float(spearmanr(a,b).statistic):+.3f}")
print("\n  모델별 순위 변화 (I1 → I4):")
r1_ = piv["I1_TFR"].rank(); r4_ = piv["I4_+양쪽압력"].rank()
for m in piv.index:
    print(f"    {m:9s} {int(r1_[m])}위 → {int(r4_[m])}위   ({piv.loc[m,'I1_TFR']:.5f} → {piv.loc[m,'I4_+양쪽압력']:.5f})")

# --- 블록 부트스트랩 ---
d0 = np.load(CK/"preds_0.npz"); y = d0["y"]; n=len(y); B=60; nb=int(np.ceil(n/B))
rng = np.random.default_rng(42); starts=np.arange(0,n-B+1)
IDX=[np.concatenate([np.arange(s,s+B) for s in rng.choice(starts,nb)])[:n] for _ in range(2000)]
PR={}
for m in MODELS:
    for fs in FS:
        f = CK/f"tuned_pred_{m}_{fs}.npy"
        if f.exists(): PR[(m,fs)] = np.load(f)
d = {i: np.load(CK/f"preds_{i}.npz") for i in range(5)}
for i,fs in enumerate(FS): PR[("Linear1",fs)] = d[i]["Linear1|logit"]


def bs(m,fs):
    e2=(y-PR[(m,fs)])**2
    return np.array([np.sqrt(e2[ix].mean()) for ix in IDX])


print("\n" + "="*96)
print("[튜닝 후] 60분 블록 부트스트랩 2000회 — 모델쌍 승자가 입력세트에 따라 뒤집히는가")
allm = MODELS+["Linear1"]
nflip=0; nsig=0
for i in range(len(allm)):
    for j in range(i+1,len(allm)):
        a,b = allm[i],allm[j]
        rec={}
        for fs in FS:
            if (a,fs) not in PR or (b,fs) not in PR: continue
            dd = bs(a,fs)-bs(b,fs); lo,hi=np.percentile(dd,[2.5,97.5])
            rec[fs]=(a if dd.mean()<0 else b, lo>0 or hi<0, dd.mean(), lo, hi)
        ws={v[0] for v in rec.values()}
        if len(ws)>1:
            nflip+=1
            sigw={v[0] for v in rec.values() if v[1]}
            tag = "★유의역전" if len(sigw)>1 else " 역전(일부 무의미)"
            if len(sigw)>1: nsig+=1
            print(f"  {tag} {a} vs {b}")
            for fs,v in rec.items():
                print(f"      {fs:14s} 승자={v[0]:8s} Δ={v[2]:+.5f} CI[{v[3]:+.5f},{v[4]:+.5f}] {'유의' if v[1] else '무의미'}")
print(f"\n  튜닝 후에도 입력세트에 따라 승자가 뒤집힌 모델쌍: {nflip} / {len(allm)*(len(allm)-1)//2}")
print(f"  그중 양쪽 모두 통계적으로 유의: {nsig}")

print("\n" + "="*96)
print("[참고] 튜닝 선택 기준(CV) 과 최종 검증 성능의 불일치")
tt = t.dropna(subset=["cv_rmse"])
for fs in FS:
    g = tt[tt.fset==fs]
    if len(g)<2: continue
    print(f"  {fs:14s} CV 1위={g.loc[g.cv_rmse.idxmin(),'model']:8s} | "
          f"검증 1위={g.loc[g.test_rmse.idxmin(),'model']:8s} | "
          f"Spearman(CV, 검증)={float(spearmanr(g.cv_rmse.to_numpy(float),g.test_rmse.to_numpy(float)).statistic):+.3f}")
