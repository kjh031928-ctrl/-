# -*- coding: utf-8 -*-
"""step4 — factorial 결과 취합 + 순위 상관 + 60분 블록 부트스트랩 검정."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

OUT = Path("/sessions/relaxed-focused-carson/mnt/outputs"); CK = OUT/"ckpt"
res = pd.concat([pd.read_csv(CK/f"res_{i}.csv") for i in range(5)], ignore_index=True)
res.to_csv(OUT/"factorial_results.csv", index=False, encoding="utf-8-sig")
FS = ["I1_TFR","I2_+PT1004","I3_+PT1100","I4_+양쪽압력","I5_+DP"]

pd.set_option("display.width", 200)
for tf in ["logit","raw"]:
    print("="*104)
    print(f"[1] 고정분할 검증 RMSE (10/13~16 학습 → 10/17-18 검증), 타깃={tf}")
    piv = res[res.target==tf].pivot(index="model", columns="fset", values="rmse_fixed")[FS]
    piv["최선-최악차"] = piv.max(1)-piv.min(1)
    print(piv.round(5).to_string())
    print(f"\n[2] LOCO 평균 RMSE (5캠페인 각각 홀드아웃), 타깃={tf}")
    pl = res[res.target==tf].pivot(index="model", columns="fset", values="loco_mean")[FS]
    print(pl.round(5).to_string())
    print()

print("="*104)
print("[3] 과적합 진단 — 학습 RMSE 대비 검증 RMSE (타깃 logit)")
p_tr = res[res.target=="logit"].pivot(index="model", columns="fset", values="rmse_train")[FS]
p_va = res[res.target=="logit"].pivot(index="model", columns="fset", values="rmse_fixed")[FS]
print("  학습 RMSE:"); print(p_tr.round(5).to_string())
print("  검증/학습 배수:"); print((p_va/p_tr).round(2).to_string())

print()
print("="*104)
print("[4] 입력세트를 바꿨을 때 모델 순위가 유지되는가 (I1 기준 순위상관)")
for tf in ["logit","raw"]:
    piv = res[res.target==tf].pivot(index="model", columns="fset", values="rmse_fixed")[FS]
    pl  = res[res.target==tf].pivot(index="model", columns="fset", values="loco_mean")[FS]
    print(f"  --- 타깃 {tf}")
    for c in FS:
        s1 = spearmanr(piv["I1_TFR"], piv[c]).statistic
        s2 = spearmanr(pl["I1_TFR"], pl[c]).statistic
        print(f"    I1 vs {c:14s} 고정분할 Spearman {s1:+.3f} | LOCO Spearman {s2:+.3f}")
    print(f"    각 세트 1위(고정분할): " + ", ".join(f"{c}:{piv[c].idxmin()}" for c in FS))
    print(f"    각 세트 1위(LOCO)   : " + ", ".join(f"{c}:{pl[c].idxmin()}" for c in FS))

print()
print("="*104)
print("[5] 60분 블록 부트스트랩 (2000회) — 고정분할 검증셋 661행")
d0 = np.load(CK/"preds_0.npz", allow_pickle=False)
y = d0["y"]; n = len(y); B = 60; nb = int(np.ceil(n/B))
rng = np.random.default_rng(42); starts = np.arange(0, n-B+1)
IDX = [np.concatenate([np.arange(s,s+B) for s in rng.choice(starts, nb)])[:n] for _ in range(2000)]
P = {}
for i, fs in enumerate(FS):
    d = np.load(CK/f"preds_{i}.npz", allow_pickle=False)
    for k in d.files:
        if k != "y": P[(fs, k)] = d[k]


def bs(fs, key):
    e2 = (y - P[(fs,key)])**2
    return np.array([np.sqrt(e2[ix].mean()) for ix in IDX])


print("\n  (a) 압력 추가 효과 ΔRMSE = RMSE(입력세트) − RMSE(I1), 모델별 (타깃 logit)")
print(f"    {'모델':12s} {'세트':14s} {'Δ평균':>10s} {'95% CI':>26s}  판정")
for mn in ["Linear1","Linear2","Ridge2","KNN","SVR-RBF","RF","ExtraTrees","GBR","HistGB","MLP-64x64","MLP-16"]:
    base = bs("I1_TFR", f"{mn}|logit")
    for fs in FS[1:]:
        d = bs(fs, f"{mn}|logit") - base
        lo, hi = np.percentile(d, [2.5, 97.5])
        j = "악화(유의)" if lo>0 else ("개선(유의)" if hi<0 else "차이없음")
        print(f"    {mn:12s} {fs:14s} {d.mean():+10.5f} [{lo:+.5f}, {hi:+.5f}]  {j}")

print("\n  (b) 모델쌍 순위 역전 — 같은 쌍이 입력세트에 따라 승자가 바뀌는가 (타깃 logit)")
cand = ["Linear2","Ridge2","GBR","HistGB","RF","ExtraTrees","SVR-RBF","MLP-64x64","MLP-16","KNN","Linear1"]
flips = []
for i in range(len(cand)):
    for j in range(i+1, len(cand)):
        a, b = cand[i], cand[j]
        wins = {}
        for fs in FS:
            d = bs(fs, f"{a}|logit") - bs(fs, f"{b}|logit")
            lo, hi = np.percentile(d, [2.5,97.5])
            wins[fs] = (a if d.mean()<0 else b, lo>0 or hi<0, d.mean(), lo, hi)
        ws = {v[0] for v in wins.values()}
        sig_ws = {v[0] for v in wins.values() if v[1]}
        if len(ws) > 1:
            flips.append((a, b, wins, len(sig_ws) > 1))
print(f"    전체 {len(cand)*(len(cand)-1)//2} 모델쌍 중 입력세트에 따라 승자가 뒤바뀐 쌍: {len(flips)}")
print(f"    그중 '양쪽 모두 통계적으로 유의한' 역전: {sum(1 for f in flips if f[3])}")
for a, b, w, sig in flips:
    if sig:
        print(f"    ★ {a} vs {b} (유의한 역전)")
        for fs, v in w.items():
            print(f"        {fs:14s} 승자={v[0]:10s} Δ={v[2]:+.5f} CI[{v[3]:+.5f},{v[4]:+.5f}] "
                  f"{'유의' if v[1] else '무의미'}")
print("\n    (유의 여부 무관 역전 쌍 목록)")
for a, b, w, sig in flips:
    print(f"      {a:11s} vs {b:11s} : " + " ".join(f"{fs.split('_')[0]}→{v[0][:9]}" for fs, v in w.items()))
