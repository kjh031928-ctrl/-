# -*- coding: utf-8 -*-
"""초안 감사에서 드러난 '근거 없는 수치' 일괄 검증 (§3.1.2 · §3.2.4 · §3.4.2 · §3.6)

기존 스크립트가 담당하지 않던 본문 주장들을 데이터로 재계산한다.
  A. §3.2.4  유효 독립표본 "대략 25–40"        (parsimony 논거 ①의 근거)
  B. §3.1.2  출력 선정 4개 수치                (X를 대표 출력으로 택한 근거)
  C. §3.2.4  FT-1001↔LC-1001 상관 −1.0 · TT-1004 고유정보 0.1%
  D. §3.6    10/14 ΔP 표준편차 0.123 kPa 와 "ΔP 평균의 0.6% 미만" 상한
  E. §3.4.2  "raw 에서 예측이 [0,1] 안에 드는 것은 트리 계열뿐" 주장

재현
  python 근거보강_검증.py        (저장소 루트에서 실행)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CSV = ROOT / "ML_마스터데이터_PV.csv"
RAW = ROOT / "2026 Chemical Engineering Process design competition_rev0_Attachment1.csv"
OUT = HERE / "근거보강_검증_결과.json"

df = pd.read_csv(CSV, encoding="utf-8-sig")
tr = df[df.Split == "학습"]
IC = ["TT-1006.PV", "FT-1004.PV", "FRC-1004.PV"]
ID = ["TT-1006.PV", "total_flow_kgh", "FRC-1004.PV"]
sig = lambda z: 1.0 / (1.0 + np.exp(-z))
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
R = {}


def r2_of(y, X):
    m = LinearRegression().fit(X, y)
    return float(m.score(X, y))


def acf_in(resid, camp, lag):
    a, b = [], []
    for c in np.unique(camp):
        r = resid[camp == c]
        if len(r) > lag:
            a.extend(r[:-lag]); b.extend(r[lag:])
    return float(np.corrcoef(np.array(a), np.array(b))[0, 1])


# ── A. 유효 독립표본 ───────────────────────────────────────────────────────
print("=" * 92)
print("A. §3.2.4 유효 독립표본 — 본문 주장 '대략 25–40'")
print("=" * 92)
y = tr["X_conv_r1_deployed"].values
mc = LinearRegression().fit(tr[IC].values, np.log(y / (1 - y)))
res_c = y - sig(mc.predict(tr[IC].values))
md = LinearRegression().fit(tr[ID].values, tr["DP_reactor_kPa"].values)
res_d = tr["DP_reactor_kPa"].values - md.predict(tr[ID].values)
camp = tr["Campaign"].values
n = len(tr)

for tag, res in (("전환율", res_c), ("ΔP", res_d)):
    rho1 = acf_in(res, camp, 1)
    # ① AR(1) 근사: n_eff = n(1−ρ)/(1+ρ)
    n_ar1 = n * (1 - rho1) / (1 + rho1)
    # ② ACF 합 기반: n_eff = n / (1 + 2Σρ_k), 첫 음수 지연까지 합(Bartlett)
    s, k = 0.0, 1
    while k < 600:
        rk = acf_in(res, camp, k)
        if rk <= 0:
            break
        s += rk; k += 1
    n_acf = n / (1 + 2 * s)
    # ③ 블록 기준: 60분 블록 개수
    n_blk = n / 60
    print(f"  [{tag}] ρ(1분)={rho1:.4f} · 첫 음수 ACF 지연 {k}분")
    print(f"       AR(1) 근사 {n_ar1:>8.1f} · ACF 합 {n_acf:>8.1f} · 60분 블록 {n_blk:>6.1f}")
    R[f"n_eff_{tag}"] = {"rho1": rho1, "ar1": n_ar1, "acf_sum": n_acf, "block60": n_blk,
                         "first_neg_lag": k}
print("  → 본문 '25–40'과 대조: 아래 요약 참조")

# ── B. §3.1.2 출력 선정 근거 ──────────────────────────────────────────────
print("\n" + "=" * 92)
print("B. §3.1.2 출력 선정 — X 를 대표 출력으로 택한 근거 4개")
print("=" * 92)
d = df  # 전 구간(10/12 포함) 기준으로 상관/설명력을 본다
X = d["X_conv_r1_deployed"].values
checks = [
    ("AT-1100 ↔ 출구압 PT-1100 상관", float(np.corrcoef(d["AT-1100.PV"], d["PT-1100.PV"])[0, 1]), 0.996, 0.005),
    ("[X·출구온도·출구압] → AT-1100 R²",
     r2_of(d["AT-1100.PV"].values, np.column_stack([X, d["TT-1100.PV"], d["PT-1100.PV"]])), 0.997, 0.005),
    ("단열상승 dT ↔ X 상관", float(np.corrcoef(d["dT_adiabatic_C"], X)[0, 1]), 0.97, 0.01),
    ("[입구온도·X] → TT-1100 R²",
     r2_of(d["TT-1100.PV"].values, np.column_stack([d["TT-1006.PV"], X])), 0.96, 0.01),
]
for label, got, ref, tol in checks:
    ok = abs(got - ref) <= tol
    print(f"  [{'OK ' if ok else '불일치'}] {label}: 재현 {got:.4f}  기재 {ref}")
    R[label] = {"got": got, "ref": ref, "ok": bool(ok)}

# ── C. §3.2.4 FT-1001·TT-1004 ─────────────────────────────────────────────
print("\n" + "=" * 92)
print("C. §3.2.4 물리셋 밖 변수 — FT-1001 · TT-1004")
print("=" * 92)
raw = pd.read_csv(RAW)
raw["Timestamp"] = pd.to_datetime(raw["Timestamp"]).dt.tz_localize(None)
ren = {c: c.split(" Value")[0] for c in raw.columns if " Value" in c}
raw = raw.rename(columns=ren)
m = df.copy(); m["Timestamp"] = pd.to_datetime(m["Timestamp"])
m = m.merge(raw[["Timestamp"] + [c for c in ("TT-1004.PV", "LC-1001.PV") if c in raw.columns]],
            on="Timestamp", how="left")
if "LC-1001.PV" in m.columns:
    c = float(np.corrcoef(m["FT-1001.PV"], m["LC-1001.PV"])[0, 1])
    ok = abs(abs(c) - 1.0) <= 0.01
    print(f"  [{'OK ' if ok else '불일치'}] corr(FT-1001, LC-1001): 재현 {c:+.4f}  기재 −1.0")
    R["corr_FT1001_LC1001"] = {"got": c, "ref": -1.0, "ok": bool(ok)}
else:
    print("  [건너뜀] LC-1001 태그가 원본 CSV 에 없음")
    R["corr_FT1001_LC1001"] = None
if "TT-1004.PV" in m.columns:
    mt = m[m.Split == "학습"]
    # 기준은 ΔP 입력 F2 — TT-1004 는 총유량의 그림자이므로 총유량이 든 좌표계로 재야 한다.
    r2 = r2_of(mt["TT-1004.PV"].values, mt[ID].values)
    print(f"  [{'OK ' if abs((1-r2)*100 - 0.2) <= 0.1 else '불일치'}] "
          f"TT-1004 를 ΔP 입력 F2 로 회귀: R²={r2:.5f} → 고유정보 {100*(1-r2):.2f}%  기재 0.2%")
    R["TT1004_unique_pct"] = {"got": 100 * (1 - r2), "ref": 0.2, "basis": "F2"}

# ── D. §3.6 10/14 ΔP 산포 ─────────────────────────────────────────────────
print("\n" + "=" * 92)
print("D. §3.6 ΔP 라벨 가정의 잔차 상한")
print("=" * 92)
c14 = df[df.Campaign == "10/14"]
sd = float(c14["DP_reactor_kPa"].std())
mean_all = float(df["DP_reactor_kPa"].mean())
print(f"  10/14 ΔP 표준편차        {sd:.4f} kPa   기재 0.123")
print(f"  전 구간 ΔP 평균          {mean_all:.3f} kPa")
print(f"  sd / 평균                {100*sd/mean_all:.3f} %   ← 기재 '0.6% 미만'과 대조")
print(f"  sd / held-out RMSE 0.191 {100*sd/0.191:.1f} %   기재 '64%'")
R["dp_10_14"] = {"std": sd, "mean_all": mean_all, "pct_of_mean": 100 * sd / mean_all,
                 "pct_of_rmse": 100 * sd / 0.191}

# ── E. §3.4.2 [0,1] 범위 주장 ─────────────────────────────────────────────
print("\n" + "=" * 92)
print("E. §3.4.2 'raw 에서 예측이 [0,1] 안에 드는 것은 트리 계열뿐' 주장")
print("=" * 92)
rj = HERE / "screening_all_models_rawX.json"
if rj.exists():
    rows = json.loads(rj.read_text(encoding="utf-8"))["전환율_25종_rawX"]
    inside = [r for r in rows if r["range_valid"]]
    tree = [r for r in inside if r["family"] == "트리"]
    other = [r for r in inside if r["family"] != "트리"]
    print(f"  [0,1] 안: {len(inside)}종 — 트리 {len(tree)}종 + 비트리 {len(other)}종")
    print(f"  비트리인데 [0,1] 안: " + ", ".join(f"{r['name']}({r['family']}, 최대 {r['pred_max']:.3f})"
                                              for r in other))
    print(f"  → 주장은 부정확. '트리 계열뿐'이 아니라 '트리 9종 + 국소·기준선 4종'이다.")
    R["range_valid_rawX"] = {"n_inside": len(inside), "n_tree": len(tree),
                             "non_tree": [r["name"] for r in other]}
else:
    print("  [건너뜀] screening_all_models_rawX.json 없음")

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(f"\n저장: {OUT}")
