# -*- coding: utf-8 -*-
"""XGBoost · CatBoost 스크리닝 재현 (§3.5.1 표 · 부록 A)

무엇을 하는가
  표 형태 데이터의 현행 표준으로 통하는 두 부스팅 구현을 **나머지 22종과 똑같은 규약**으로
  적합해 24종 비교표의 16·17위 값을 산출한다. 평판에 기대 유추로 넘기지 않고 실제로 돌린다.

규약을 재구현하지 않는 이유
  기존 22종의 평가 규약은 `2026-07-21_ML개발/src/evaluation.py` 에 이미 확정돼 있다.
  여기서 다시 구현하면 미세한 차이로 불공정 비교가 되므로 **그 모듈을 그대로 import** 한다.
    · 타깃 = logit(X_r1). 모든 모델을 로짓 공간에 적합하고 예측을 sigmoid 로 되돌려
      전환율 스케일에서 RMSE 를 잰다(배포 ONNX 관례와 동일).
    · 하이퍼파라미터 튜닝·승자선택 = 블록20%(각 학습 캠페인의 앞 80% 적합 / 뒤 20% 내부검증).
      검증 캠페인(10/17–18)은 튜닝에 일절 쓰지 않는다.
    · 성능 = 학습 10/13–16 → held-out 10/17–18.
    · 물리검사 = 27점 격자(온도 335/350/365 °C × 총유량 3 × B/P 3). 외삽 판정선 349.977 °C.

  ΔP(ONNX2)는 스크리닝 대상이 아니어서 evaluation.py 에 규약이 없다. 그래서 그림 규약
  (`figures/make_figures.py`)과 같은 분할·입력으로 직접 적합하고, 27점 단조성 검사도
  같은 격자로 수행한다.

필요 패키지
  pip install xgboost catboost      (그 외는 기존 환경: numpy · pandas · scikit-learn)

재현
  python xgboost_catboost_검증.py   (repo 루트에서 실행)
"""

from __future__ import annotations

import sys
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
sys.path.insert(0, str(ROOT / "2026-07-21_ML개발" / "src"))

try:
    import evaluation as ev  # 기존 22종이 쓴 평가 규약 모듈 (수정하지 않음)
except ImportError as e:  # pragma: no cover
    sys.exit(f"중단 — 평가 모듈을 찾지 못했습니다: {e}\n"
             f"      기대 경로: {ROOT / '2026-07-21_ML개발' / 'src' / 'evaluation.py'}")

try:
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
except ImportError as e:  # pragma: no cover
    sys.exit(f"중단 — {e}. `pip install xgboost catboost` 후 다시 실행하십시오.")

SEED = ev.SEED  # 42

# 보고서 기재 기준값 (§3.5.1 표 · 부록 A)
REF = {
    "XGBoost": {"conv_rmse": 0.08269, "dp_rmse": 0.3421, "dp_bp_pass": 3},
    "CatBoost": {"conv_rmse": 0.09267, "dp_rmse": 1.0329, "dp_bp_pass": 3},
    "Linear": {"conv_rmse": 0.02099, "dp_rmse": 0.1910, "dp_bp_pass": 9},
}
TOL_CONV, TOL_DP = 5e-4, 5e-3
FAILURES: list[str] = []


def check(label: str, got: float, ref: float, tol: float) -> None:
    ok = abs(got - ref) <= tol
    print(f"    [{'OK ' if ok else '불일치'}] {label}: 재현 {got:.5f}  기준 {ref:.5f}  "
          f"차 {got - ref:+.5f} (허용 ±{tol:g})")
    if not ok:
        FAILURES.append(f"{label}: 재현 {got:.5f} vs 기준 {ref:.5f}")


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


# =============================================================================
# 1. 전환율 (ONNX1) — 기존 22종과 동일 규약
# =============================================================================
def xgb_factory(p):
    return XGBRegressor(n_estimators=p["n"], max_depth=p["d"], learning_rate=p["lr"],
                        subsample=p.get("sub", 0.8), colsample_bytree=1.0, reg_lambda=1.0,
                        random_state=SEED, n_jobs=4, verbosity=0, tree_method="hist")


def cat_factory(p):
    return CatBoostRegressor(iterations=p["n"], depth=p["d"], learning_rate=p["lr"],
                             random_seed=SEED, verbose=0, allow_writing_files=False)


XGB_GRID = [{"n": n, "d": d, "lr": lr}
            for n in (300, 800) for d in (3, 6) for lr in (0.03, 0.1)]
CAT_GRID = [{"n": n, "d": d, "lr": lr}
            for n in (500, 1500) for d in (4, 6) for lr in (0.03, 0.1)]

print("=" * 84)
print("1. 전환율(ONNX1) — 학습 10/13–16 → held-out 10/17–18, 타깃 logit(X), 튜닝 블록20%")
print("=" * 84)

sp = ev.load_splits("I1", "r1")
print(f"  입력 {sp.feature_names} · 학습 {len(sp.X_train)}행 · held-out {len(sp.X_valid)}행")
print(f"  학습셋 TT-1006 최대 {sp.X_train[:, 0].max():.3f} °C "
      f"(외삽 판정선 {ev.T_MAX_TRAIN} · 케이스 스터디 요구 365 °C)\n")

specs = [
    ("XGBoost", ev.ModelSpec("XGBoost", "트리", xgb_factory, XGB_GRID, onnx_capable=False)),
    ("CatBoost", ev.ModelSpec("CatBoost", "트리", cat_factory, CAT_GRID, onnx_capable=False)),
    ("Linear", ev.ModelSpec("Linear 1차(배포)", "선형", lambda p: LinearRegression(), [{}])),
]

conv_results = {}
for key, spec in specs:
    best, s20 = ev.tune(spec, sp)
    est = ev.fit_on_logit(spec, best, sp.X_train, sp.y_train)
    m = ev.conversion_metrics(sp.y_valid, ev.predict_conv(est, sp.X_valid))
    phys = ev.physical_check(spec, best, sp)

    # 365 °C 예측이 350 °C 와 같은 격자점 수 (clamp 실증)
    meta, preds = phys["meta"], phys["preds"]
    clamped = sum(
        1 for mm, pp in zip(meta, preds) if mm["T"] == 350.0
        and abs(next(p for q, p in zip(meta, preds) if q["T"] == 365.0
                     and q["total"] == mm["total"] and q["bp"] == mm["bp"]) - pp) < 1e-12
    )
    conv_results[key] = dict(rmse=m["rmse"], r2=m["r2"], mono=phys["T_monotone_slices"],
                             clamped=clamped, best=best, s20=s20)
    print(f"  ■ {spec.name}  최적 {best}")
    print(f"     블록20%(순위용) {s20:.5f} · held-out RMSE {m['rmse']:.5f} · R² {m['r2']:.4f}")
    print(f"     온도 단조 {phys['T_monotone_slices']}/{phys['n_slices']} · "
          f"365°C 예측이 350°C 와 동일한 격자점 {clamped}/9")

print("\n  [검증] 보고서 §3.5.1 표 기재값과 대조")
for key in ("XGBoost", "CatBoost", "Linear"):
    check(f"전환율 {key}", conv_results[key]["rmse"], REF[key]["conv_rmse"], TOL_CONV)

# =============================================================================
# 2. 압력강하 (ONNX2) — 그림 규약(figures/make_figures.py)과 동일 분할·입력
# =============================================================================
print("\n" + "=" * 84)
print("2. 압력강하(ONNX2) — 입력 [TT-1006, 총유량, FRC-1004], 변환 없음")
print("=" * 84)

df = pd.read_csv(ROOT / "ML_마스터데이터_PV.csv", encoding="utf-8-sig")
train, valid = df[df.Split == "학습"], df[df.Split == "검증"]
IN_DP, TGT_DP = ["TT-1006.PV", "total_flow_kgh", "FRC-1004.PV"], "DP_reactor_kPa"
Xtr, ytr = train[IN_DP].values, train[TGT_DP].values
Xva, yva = valid[IN_DP].values, valid[TGT_DP].values

# 블록20% 마스크: 캠페인별 앞 80% 적합 / 뒤 20% 내부검증 (전환율과 같은 방식)
m_fit = np.zeros(len(train), bool)
for c in train["Campaign"].unique():
    idx = np.where(train["Campaign"].values == c)[0]
    m_fit[idx[: int(len(idx) * 0.8)]] = True


def tune_dp(factory, grid):
    best = None
    for p in grid:
        s = rmse(factory(p).fit(Xtr[m_fit], ytr[m_fit]).predict(Xtr[~m_fit]), ytr[~m_fit])
        if best is None or s < best[1]:
            best = (p, s)
    return best


T_G, BP_G, FL_G = [335.0, 350.0, 365.0], [4.0, 5.0, 6.0], [15000.0, 16500.0, 18000.0]


def monotonicity(est):
    """27점 격자: 유량↑→ΔP↑ 와 B/P↑→ΔP↓ 각각 9조합 중 몇 개가 부합하는가."""
    def f(t, fl, bp):
        return float(est.predict(np.array([[t, fl, bp]]))[0])
    flow_ok = sum(f(t, FL_G[0], bp) < f(t, FL_G[1], bp) < f(t, FL_G[2], bp)
                  for t in T_G for bp in BP_G)
    bp_ok = sum(f(t, fl, BP_G[0]) > f(t, fl, BP_G[1]) > f(t, fl, BP_G[2])
                for t in T_G for fl in FL_G)
    return flow_ok, bp_ok


dp_results = {}
for key, fac, grid in (("XGBoost", xgb_factory, XGB_GRID),
                       ("CatBoost", cat_factory, CAT_GRID),
                       ("Linear", lambda p: LinearRegression(), [{}])):
    best, _ = tune_dp(fac, grid)
    est = fac(best).fit(Xtr, ytr)
    R = rmse(est.predict(Xva), yva)
    fl, bp = monotonicity(est)
    dp_results[key] = dict(rmse=R, flow=fl, bp=bp)
    print(f"  ■ {key:<10} held-out RMSE {R:>8.4f} kPa · "
          f"유량↑→ΔP↑ {fl}/9 · B/P↑→ΔP↓ {bp}/9")

print("\n  [검증] 부록 A 기재값과 대조")
for key in ("XGBoost", "CatBoost", "Linear"):
    check(f"ΔP {key}", dp_results[key]["rmse"], REF[key]["dp_rmse"], TOL_DP)
    got, ref = dp_results[key]["bp"], REF[key]["dp_bp_pass"]
    print(f"    [{'OK ' if got == ref else '불일치'}] ΔP {key} B/P 방향: "
          f"재현 {got}/9  기준 {ref}/9")
    if got != ref:
        FAILURES.append(f"ΔP {key} B/P 방향: 재현 {got}/9 vs 기준 {ref}/9")

# =============================================================================
# 3. 결론
# =============================================================================
print("\n" + "=" * 84)
print("결론 — 24종 표에 들어가는 값")
print("=" * 84)
print(f"  {'모델':<12}{'전환율 RMSE':>14}{'온도단조':>10}{'365°C clamp':>14}{'ΔP RMSE':>11}{'B/P방향':>10}")
print("  " + "-" * 78)
for key in ("Linear", "XGBoost", "CatBoost"):
    c, d = conv_results[key], dp_results[key]
    print(f"  {key:<12}{c['rmse']:>14.5f}{str(c['mono']) + '/9':>10}"
          f"{str(c['clamped']) + '/9':>14}{d['rmse']:>11.4f}{str(d['bp']) + '/9':>10}")
print("\n  트리 두 종 모두 365°C 예측이 350°C 와 동일(clamp) — 학습 최대 349.98°C 밖으로")
print("  나가지 못하는 범위보존 구조 때문이며, 케이스 스터디가 요구하는 365°C 에서 부적격이다.")

print("=" * 84)
if FAILURES:
    print("기준값 불일치 — 원인을 규명할 것:")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("기준값 전부 재현됨.")
print("=" * 84)
