# -*- coding: utf-8 -*-
"""2차항의 외삽 곡률 위험 실측 (§3.5.4 근거 ①)

§3.5.4 는 1차 선형을 택한 이유 ① 로 "2차항은 외삽에서 곡률이 발산할 위험"을 든다.
그 주장을 말로 두지 않고 **기울기 부호가 뒤집히는 지점(꺾임)** 을 축별로 직접 찾는다.

무엇을 재는가
  세 후보(1차 선형 · Poly 2차 · Ridge Poly2)를 학습(10/13–16)으로 적합한 뒤,
  각 입력 축을 조밀하게 스윕하며 예측의 1차 차분 부호가 바뀌는 첫 지점을 찾는다.
  부호가 바뀐다 = 그 지점을 넘어서면 모델이 "온도를 올리면 전환율이 떨어진다"류의
  물리적으로 뒤집힌 응답을 내놓는다는 뜻이다.

  raw-X 와 로짓 두 공간 모두에서 잰다(로짓은 출력을 (0,1)에 가두지만 단조성까지
  보장하지는 않으므로 따로 확인해야 한다).

기준선
  학습 범위       T 329.98–349.98 °C · 프로펜 2499.7–3660.9 kg/h · B/P 4.0–6.0
  케이스 스터디    T 335–365 °C · 총유량 15000–18000 kg/h · B/P 4–6
                 (프로펜 = 총유량/(1+B/P) → 2143–3600 kg/h)

재현
  python 곡률외삽_검증.py      (저장소 루트에서 실행)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CSV = ROOT / "ML_마스터데이터_PV.csv"
if not CSV.exists():
    CSV = HERE.parent / "ML_마스터데이터_PV.csv"
OUT = HERE / "곡률외삽_검증_결과.json"

SEED = 42
RIDGE_ALPHA = 0.01          # 블록20% 로 고른 Ridge Poly2 의 alpha
IC = ["TT-1006.PV", "FT-1004.PV", "FRC-1004.PV"]

df = pd.read_csv(CSV, encoding="utf-8-sig")
tr = df[df.Split == "학습"]
y = tr["X_conv_r1_deployed"].values
ylg = np.log(y / (1 - y))
sig = lambda z: 1.0 / (1.0 + np.exp(-z))


def build(kind, alpha=None):
    if kind == "lin":
        return LinearRegression()
    tail = LinearRegression() if alpha is None else Ridge(alpha=alpha)
    return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), tail)


MODELS = {}
for nm, kind, a in (("1차 선형", "lin", None), ("Poly 2차", "p2", None),
                    ("Ridge Poly2", "rp", RIDGE_ALPHA)):
    for space, t in (("raw", y), ("logit", ylg)):
        est = build(kind, a)
        est.fit(tr[IC].values, t)
        MODELS[(nm, space)] = est


def predict(nm, space, T, ft, bp) -> float:
    v = MODELS[(nm, space)].predict(np.array([[T, ft, bp]]))[0]
    return float(sig(v)) if space == "logit" else float(v)


def first_turn(nm, space, pts, idx):
    """조밀 스윕에서 1차 차분 부호가 처음 바뀌는 축 좌표. 없으면 None."""
    v = np.array([predict(nm, space, *p) for p in pts])
    s = np.sign(np.diff(v))
    flip = np.where(s[1:] != s[:-1])[0]
    return (float(pts[flip[0] + 1][idx]) if len(flip) else None), float(v.max())


AXES = [
    ("온도 [°C]", [(t, 3000.0, 5.0) for t in np.arange(330, 421, 1.0)], 0, (335.0, 365.0)),
    ("프로펜 [kg/h]", [(350.0, f, 5.0) for f in np.arange(2000, 6001, 25.0)], 1, (2143.0, 3600.0)),
    ("B/P [-]", [(350.0, 3000.0, b) for b in np.arange(2.5, 10.01, 0.05)], 2, (4.0, 6.0)),
]

print("=" * 94)
print("2차항 외삽 곡률 검증 — 기울기 부호가 뒤집히는 지점")
print("=" * 94)
print(f"학습 범위  T {tr['TT-1006.PV'].min():.2f}–{tr['TT-1006.PV'].max():.2f} °C · "
      f"프로펜 {tr['FT-1004.PV'].min():.1f}–{tr['FT-1004.PV'].max():.1f} kg/h · "
      f"B/P {tr['FRC-1004.PV'].min():.1f}–{tr['FRC-1004.PV'].max():.1f}")
print("케이스 스터디  T 335–365 °C · 총유량 15000–18000 · B/P 4–6 (프로펜 2143–3600)\n")

results = {}
for space in ("raw", "logit"):
    print(f"[{space}]")
    print(f"  {'축':<14}{'케이스 범위':<14}" + "".join(f"{n:>18}" for n in
          ("1차 선형", "Poly 2차", "Ridge Poly2")))
    for axis, pts, idx, (lo, hi) in AXES:
        cells = []
        for nm in ("1차 선형", "Poly 2차", "Ridge Poly2"):
            t, vmax = first_turn(nm, space, pts, idx)
            results[f"{space}|{axis}|{nm}"] = {"turn_at": t, "max_pred": vmax,
                                               "inside_case_range": bool(t is not None and lo <= t <= hi)}
            if t is None:
                cells.append("반전 없음")
            else:
                cells.append(f"{t:g} 반전" + (" ★범위내" if lo <= t <= hi else ""))
        print(f"  {axis:<14}{f'{lo:g}–{hi:g}':<14}" + "".join(f"{c:>18}" for c in cells))
    print()

# B/P 축 상세 — 케이스 범위 안에서 꺾이는 것을 값으로 보인다
print("=" * 94)
print("B/P 축 상세 (T=350 °C · 프로펜 3000 kg/h) — 케이스 스터디 범위 4–6 안")
print("=" * 94)
bps = [4.0, 4.5, 5.0, 5.5, 5.7, 6.0]
for space in ("raw", "logit"):
    print(f"  [{space}]  B/P " + "".join(f"{b:>9.1f}" for b in bps))
    for nm in ("1차 선형", "Poly 2차", "Ridge Poly2"):
        v = [predict(nm, space, 350.0, 3000.0, b) for b in bps]
        d = np.diff(v)
        mark = "  ← 상승 후 하강(꺾임)" if (d > 0).any() and (d < 0).any() else ""
        print(f"  {nm:<14}" + "".join(f"{x:>9.4f}" for x in v) + mark)
    print()

# 온도 축 상세 — §3.5.4 가 명시한 365 °C 외삽
print("=" * 94)
print("온도 축 상세 (총유량 18000 · B/P 5) — §3.5.4 가 명시한 365 °C 외삽")
print("=" * 94)
Ts = [350.0, 365.0, 380.0, 400.0, 420.0]
ft = 18000.0 / 6.0
for space in ("raw", "logit"):
    print(f"  [{space}]  T " + "".join(f"{t:>9.0f}" for t in Ts))
    for nm in ("1차 선형", "Poly 2차", "Ridge Poly2"):
        v = [predict(nm, space, t, ft, 5.0) for t in Ts]
        print(f"  {nm:<14}" + "".join(f"{x:>9.4f}" for x in v))
    print()

OUT.write_text(json.dumps({
    "설명": "2차항 외삽 곡률(기울기 부호 반전) 실측 — §3.5.4 근거 ①",
    "적합": "학습 10/13–16 · seed 42 · Ridge Poly2 alpha=0.01(블록20% 선택)",
    "학습범위": {"T": [float(tr['TT-1006.PV'].min()), float(tr['TT-1006.PV'].max())],
                "프로펜": [float(tr['FT-1004.PV'].min()), float(tr['FT-1004.PV'].max())],
                "BP": [float(tr['FRC-1004.PV'].min()), float(tr['FRC-1004.PV'].max())]},
    "케이스범위": {"T": [335, 365], "총유량": [15000, 18000], "BP": [4, 6], "프로펜": [2143, 3600]},
    "반전지점": results,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"저장: {OUT}")
