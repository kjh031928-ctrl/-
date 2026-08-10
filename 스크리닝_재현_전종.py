# -*- coding: utf-8 -*-
"""§3.5.1 스크리닝 표(25종) 전체 재현 + 트리 계열 365°C clamp 실측 [부록 A]

무엇을 하는가
  1. 전환율(ONNX1) 후보 25종의 held-out RMSE 를 **전부 다시 계산**한다.
     기존 22종은 `stage1_models.build_specs()` 를 그대로 불러오고, XGBoost·CatBoost·LightGBM 세 종만
     같은 규약으로 덧붙인다. 표에 적힌 값이 인용이 아니라 재현값임을 보장하기 위함이다.
  2. 27점 격자에서 **365°C 예측이 350°C 와 같은지(clamp)** 를 모든 후보에 대해 실측한다.
     보고서 표의 ● 표기(트리 계열 범위보존)를 계열 성질에 대한 유추가 아니라
     모델별 측정으로 뒷받침하기 위함이다.
  3. 압력강하(ONNX2)에서 XGBoost·CatBoost·LightGBM 을 선형과 비교한다(부록 A).

규약을 재구현하지 않는다
  기존 22종의 평가 규약은 `2026-07-21_ML개발/src/evaluation.py` 에 확정돼 있고, 후보 목록은
  같은 폴더 `stage1_models.py` 에 있다. 여기서 다시 구현하면 미세한 차이로 불공정 비교가
  되므로 두 모듈을 **import 만 하고 수정하지 않는다**.
    · 타깃 = logit(X_r1) → 예측을 sigmoid 로 되돌려 전환율 스케일에서 RMSE
    · 튜닝·승자선택 = 블록20%(캠페인 앞 80% 적합 / 뒤 20% 내부검증). 검증셋은 튜닝에 미사용
    · 성능 = 학습 10/13–16 → held-out 10/17–18
    · 물리격자 = 온도 335/350/365 °C × 총유량 3 × B/P 3. 외삽 판정선 349.977 °C

필요 패키지
  pip install xgboost catboost lightgbm   (그 외는 기존 환경: numpy · pandas · scikit-learn)

재현
  python 스크리닝_재현_전종.py      (repo 루트에서 실행. 수 분 소요 — GaussianProcess·SVR 포함)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "2026-07-21_ML개발" / "src"
sys.path.insert(0, str(SRC))

try:
    import evaluation as ev
    import stage1_models as s1
except ImportError as e:  # pragma: no cover
    sys.exit(f"중단 — 스크리닝 모듈을 찾지 못했습니다: {e}\n      기대 경로: {SRC}")

try:
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
except ImportError as e:  # pragma: no cover
    sys.exit(f"중단 — {e}. `pip install xgboost catboost lightgbm` 후 다시 실행하십시오.")

SEED = ev.SEED
OUT_JSON = ROOT / "results" / "screening_all_models.json"

# 보고서 §3.5.1 표 기재값 (재현 대조용)
REF_RMSE = {
    "Dummy(mean)": 0.18552,
    "Linear 1차": 0.02097,
    "Poly 2차": 0.02115,
    "Poly 3차": 0.37704,
    "Ridge 1차": 0.02231,
    "Ridge Poly2": 0.01929,
    "Lasso": 0.03636,
    "ElasticNet": 0.02924,
    "PLS": 0.02097,
    "KNN": 0.07895,
    "SVR(RBF)": 0.05882,
    "SVR(linear)": 0.02349,
    "GaussianProcess": 0.16411,
    "DecisionTree": 0.1008,
    "RandomForest": 0.10014,
    "ExtraTrees": 0.07935,
    "GradientBoosting": 0.0814,
    "HistGradientBoosting": 0.09422,
    "AdaBoost": 0.10159,
    "MLP(16,)": 0.04523,
    "MLP(64, 64)": 0.05182,
    "MLP(128, 128)": 0.03661,
    "XGBoost": 0.08271,
    "CatBoost": 0.09493,
    "LightGBM": 0.08674,
}
TOL = 5e-4
FAILURES: list[str] = []


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


# ---------------------------------------------------------------- 후보 25종
def xgb_factory(p):
    return XGBRegressor(n_estimators=p["n"], max_depth=p["d"], learning_rate=p["lr"],
                        subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                        random_state=SEED, n_jobs=4, verbosity=0, tree_method="hist")


def cat_factory(p):
    return CatBoostRegressor(iterations=p["n"], depth=p["d"], learning_rate=p["lr"],
                             random_seed=SEED, verbose=0, allow_writing_files=False)


def lgb_factory(p):
    # num_leaves 는 max_depth 와 정합하게 2^d−1 로 둔다(XGBoost 격자와 대응시키기 위함).
    return LGBMRegressor(n_estimators=p["n"], max_depth=p["d"], num_leaves=2 ** p["d"] - 1,
                         learning_rate=p["lr"], subsample=0.8, subsample_freq=1,
                         random_state=SEED, n_jobs=4, verbose=-1)


XGB_GRID = [{"n": n, "d": d, "lr": lr}
            for n in (300, 800) for d in (3, 6) for lr in (0.03, 0.1)]
CAT_GRID = [{"n": n, "d": d, "lr": lr}
            for n in (500, 1500) for d in (4, 6) for lr in (0.03, 0.1)]
LGB_GRID = [{"n": n, "d": d, "lr": lr}
            for n in (300, 800) for d in (3, 6) for lr in (0.03, 0.1)]

specs = list(s1.build_specs())                      # 기존 22종 (수정하지 않음)
specs.append(ev.ModelSpec("XGBoost", "트리", xgb_factory, XGB_GRID, onnx_capable=False))
specs.append(ev.ModelSpec("CatBoost", "트리", cat_factory, CAT_GRID, onnx_capable=False))
specs.append(ev.ModelSpec("LightGBM", "트리", lgb_factory, LGB_GRID, onnx_capable=False))
print(f"후보 {len(specs)}종 (기존 {len(specs) - 3} + XGBoost·CatBoost·LightGBM)")

sp = ev.load_splits("I1", "r1")
print(f"입력 {sp.feature_names} · 학습 {len(sp.X_train)}행 → held-out {len(sp.X_valid)}행")
print(f"학습셋 TT-1006 최대 {sp.X_train[:, 0].max():.3f} °C · 케이스 스터디 요구 365 °C\n")


def clamp_count(phys) -> int:
    """27점 격자에서 365°C 예측이 350°C 예측과 완전히 같은 (총유량,B/P) 조합 수 (0~9)."""
    meta, preds = phys["meta"], phys["preds"]
    n = 0
    for mm, pp in zip(meta, preds):
        if mm["T"] != 350.0:
            continue
        p365 = next(p for q, p in zip(meta, preds) if q["T"] == 365.0
                    and q["total"] == mm["total"] and q["bp"] == mm["bp"])
        if abs(p365 - pp) < 1e-12:
            n += 1
    return n


# ---------------------------------------------------------------- 1. 전환율 25종
print("=" * 96)
print("1. 전환율(ONNX1) 25종 — held-out RMSE 재현 + 365°C clamp 실측")
print("=" * 96)
print(f"{'모델':<22}{'계열':<10}{'RMSE':>10}{'기준':>10}{'차':>10}{'온도단조':>9}{'clamp':>8}{'초':>7}")
print("-" * 96)

rows = []
for spec in specs:
    t0 = time.time()
    best, s20 = ev.tune(spec, sp)
    est = ev.fit_on_logit(spec, best, sp.X_train, sp.y_train)
    m = ev.conversion_metrics(sp.y_valid, ev.predict_conv(est, sp.X_valid))
    phys = ev.physical_check(spec, best, sp)
    cl = clamp_count(phys)
    dt = time.time() - t0

    ref = REF_RMSE.get(spec.name)
    dev = (m["rmse"] - ref) if ref is not None else float("nan")
    if ref is not None and abs(dev) > TOL:
        FAILURES.append(f"{spec.name}: 재현 {m['rmse']:.5f} vs 기준 {ref:.5f}")

    if ref is None:
        FAILURES.append(f"{spec.name}: 기준값이 REF_RMSE 에 없어 대조하지 못했습니다")

    rows.append({
        "name": spec.name, "family": spec.family, "rmse": m["rmse"], "r2": m["r2"],
        "ref": ref, "dev": None if ref is None else dev,
        "T_monotone": phys["T_monotone_slices"], "clamp_9": cl,
        "best_params": {k: str(v) for k, v in best.items()}, "block20": s20,
        "seconds": round(dt, 1),
    })
    print(f"{spec.name:<22}{spec.family:<10}{m['rmse']:>10.5f}"
          f"{(f'{ref:.5f}' if ref is not None else '—'):>10}"
          f"{(f'{dev:+.5f}' if ref is not None else '—'):>10}"
          f"{str(phys['T_monotone_slices']) + '/9':>9}{str(cl) + '/9':>8}{dt:>7.1f}")

# ---------------------------------------------------------------- 2. clamp 정리
print("\n" + "=" * 96)
print("2. 365°C clamp 실측 정리 — 학습 최대 349.98°C 밖에서 예측이 얼어붙는가")
print("=" * 96)
clamped = [r for r in rows if r["clamp_9"] == 9]
partial = [r for r in rows if 0 < r["clamp_9"] < 9]
free = [r for r in rows if r["clamp_9"] == 0]
print(f"  9/9 완전 clamp ({len(clamped)}종): " + " · ".join(r["name"] for r in clamped))
if partial:
    print(f"  부분 clamp ({len(partial)}종): "
          + " · ".join(f"{r['name']} {r['clamp_9']}/9" for r in partial))
print(f"  clamp 없음 ({len(free)}종): " + " · ".join(r["name"] for r in free))
tree_rows = [r for r in rows if r["family"] == "트리"]
print(f"\n  트리 계열 {len(tree_rows)}종 중 9/9 clamp: "
      f"{sum(1 for r in tree_rows if r['clamp_9'] == 9)}종 "
      f"→ ● 표기는 계열 유추가 아니라 모델별 측정값이다.")

# ---------------------------------------------------------------- 3. ΔP 비교
print("\n" + "=" * 96)
print("3. 압력강하(ONNX2) — XGBoost·CatBoost·LightGBM 대 선형 (그림 규약과 동일 분할)")
print("=" * 96)

df = pd.read_csv(ROOT / "ML_마스터데이터_PV.csv", encoding="utf-8-sig")
train, valid = df[df.Split == "학습"], df[df.Split == "검증"]
IN_DP, TGT_DP = ["TT-1006.PV", "total_flow_kgh", "FRC-1004.PV"], "DP_reactor_kPa"
Xtr, ytr = train[IN_DP].values, train[TGT_DP].values
Xva, yva = valid[IN_DP].values, valid[TGT_DP].values

m_fit = np.zeros(len(train), bool)
for c in train["Campaign"].unique():
    idx = np.where(train["Campaign"].values == c)[0]
    m_fit[idx[: int(len(idx) * 0.8)]] = True

T_G, BP_G, FL_G = [335.0, 350.0, 365.0], [4.0, 5.0, 6.0], [15000.0, 16500.0, 18000.0]


def dp_monotonicity(est):
    def f(t, fl, bp):
        return float(est.predict(np.array([[t, fl, bp]]))[0])
    flow_ok = sum(f(t, FL_G[0], bp) < f(t, FL_G[1], bp) < f(t, FL_G[2], bp)
                  for t in T_G for bp in BP_G)
    bp_ok = sum(f(t, fl, BP_G[0]) > f(t, fl, BP_G[1]) > f(t, fl, BP_G[2])
                for t in T_G for fl in FL_G)
    return flow_ok, bp_ok


REF_DP = {"XGBoost": (0.3421, 3), "CatBoost": (1.0329, 3), "LightGBM": (0.6253, 6),
          "Linear": (0.1910, 9)}
dp_rows = []
print(f"{'모델':<12}{'held-out RMSE':>15}{'기준':>10}{'유량↑→ΔP↑':>12}{'B/P↑→ΔP↓':>12}")
print("-" * 96)
for key, fac, grid in (("XGBoost", xgb_factory, XGB_GRID),
                       ("CatBoost", cat_factory, CAT_GRID),
                       ("LightGBM", lgb_factory, LGB_GRID),
                       ("Linear", lambda p: LinearRegression(), [{}])):
    best = min(((rmse(fac(p).fit(Xtr[m_fit], ytr[m_fit]).predict(Xtr[~m_fit]), ytr[~m_fit]), i, p)
                for i, p in enumerate(grid)))[2]
    est = fac(best).fit(Xtr, ytr)
    R = rmse(est.predict(Xva), yva)
    fl, bp = dp_monotonicity(est)
    ref_r, ref_bp = REF_DP[key]
    if ref_r is not None and abs(R - ref_r) > 5e-3:
        FAILURES.append(f"ΔP {key}: 재현 {R:.4f} vs 기준 {ref_r:.4f}")
    if ref_bp is not None and bp != ref_bp:
        FAILURES.append(f"ΔP {key} B/P: 재현 {bp}/9 vs 기준 {ref_bp}/9")
    dp_rows.append({"name": key, "rmse": R, "flow_pass": fl, "bp_pass": bp})
    print(f"{key:<12}{R:>15.4f}"
          f"{(f'{ref_r:.4f}' if ref_r is not None else '—'):>10}"
          f"{str(fl) + '/9':>12}{str(bp) + '/9':>12}")

# ---------------------------------------------------------------- 저장 · 판정
OUT_JSON.parent.mkdir(exist_ok=True)
OUT_JSON.write_text(json.dumps({
    "설명": "§3.5.1 스크리닝 25종 전체 재현 + 365°C clamp 실측",
    "규약": {
        "타깃": "logit(X_conv_r1)", "튜닝": "블록20%(캠페인 앞 80%/뒤 20%)",
        "성능": "학습 10/13-16 → held-out 10/17-18", "seed": SEED,
        "격자": "T 335/350/365 °C × 총유량 15000/16500/18000 × B/P 4/5/6",
        "clamp_정의": "365°C 예측이 350°C 예측과 1e-12 이내로 같은 (총유량,B/P) 조합 수 / 9",
    },
    "전환율_전종": rows,
    "압력강하": dp_rows,
    "불일치": FAILURES,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n저장: {OUT_JSON}")

print("=" * 96)
if FAILURES:
    print("기준값 불일치 — 원인을 규명할 것:")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print(f"{len(rows)}종 전부 기준값 재현됨 · clamp 는 모델별 실측값.")
print("=" * 96)
