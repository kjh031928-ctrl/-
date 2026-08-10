# -*- coding: utf-8 -*-
"""전환율(ONNX1) 스크리닝 재현 — **raw-X 기준** (§3.5.1 표 · §3.5.5 로짓 도입 근거)

왜 raw-X 인가
  실제 개발 순서는 (순수 1차 선형 → 케이스 스터디에서 전환율 X>1 발견 → 로짓 도입) 이었다.
  따라서 모델 **비교**는 로짓을 쓰기 전 상태, 즉 원시 X 를 타깃으로 해야 순서가 맞다.
  로짓은 비교가 끝난 뒤 최종 전환율 모델에만 적용된다.
  (배포 모델 `reactor_conversion_r1.onnx` 는 로짓-선형 그대로다 — 이 스크립트는 배포본을 바꾸지 않는다.)

기존 로짓판(`스크리닝_재현_전종.py`)과 **다른 점은 타깃 하나뿐**이다
    logit(X) 에 적합 → 예측을 sigmoid 로 역변환        (기존)
    X 에 그대로 적합 → 예측도 X 스케일 그대로            (이 스크립트)
  규약 모듈(`evaluation.py`)에 이미 있는 `TRANSFORMS["raw"]` 로 변환만 갈아끼운다.
  분할·튜닝기준·seed·후보목록·물리격자는 전부 기존과 동일하다.

규약 (기존과 동일)
  · 분할: 학습 10/13–16 → held-out 10/17–18. 앵커 10/12 는 성능 통계에서 제외. 무작위 분할 금지.
  · 튜닝·승자선택: 블록20%(각 학습 캠페인 앞 80% 적합 / 뒤 20% 내부검증). 검증셋은 튜닝에 미사용.
  · seed = 42.
  · 물리격자: T{335,350,365}°C × 총유량{15000,16500,18000} kg/h × B/P{4,5,6}
              (프로펜유량 = 총유량/(1+B/P)). 외삽 판정선 349.977 °C.

산출물 (모두 이 폴더)
  · screening_all_models_rawX.json  — 25종 × {rmse, r2, T_monotone, clamp_9, pred_max, range_valid, ...}
  · 콘솔: raw-X 순위표 · 로짓판 대비 변화 · §3.5.5 근거 데이터

재현
  python "스크리닝_재현_전종_rawX.py"      (이 파일이 있는 폴더에서 실행)
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # D:/공정설계대회
SRC = ROOT / "2026-07-21_ML개발" / "src"
sys.path.insert(0, str(SRC))

try:
    import evaluation as ev
    import stage1_models as s1
except ImportError as e:  # pragma: no cover
    sys.exit(f"중단 — 스크리닝 모듈을 찾지 못했습니다: {e}\n      기대 경로: {SRC}")

try:
    import sklearn, xgboost, lightgbm, catboost
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
except ImportError as e:  # pragma: no cover
    sys.exit(f"중단 — {e}. `pip install xgboost catboost lightgbm` 후 다시 실행하십시오.")

SEED = ev.SEED
OUT_JSON = HERE / "screening_all_models_rawX.json"
LOGIT_JSON = ROOT / "results" / "screening_all_models.json"   # 대조용(로짓판)

# 사전에 재현해 둔 기대 앵커 — 어긋나면 재현값을 따르고 콘솔에 명시한다
ANCHOR = {
    "linear_raw_rmse": 0.03569,
    "linear_raw_holdout_max": 1.008,
    "linear_raw_grid_max": 1.388,
    "linear_raw_grid_over1": 9,
    "logit_rmse": 0.02097,
}
NOTE: list[str] = []

print("=" * 100)
print("전환율 스크리닝 — raw-X 기준 (타깃 = X 원시값, 로짓 미적용)")
print("=" * 100)
print(f"버전: python {sys.version.split()[0]} · sklearn {sklearn.__version__} · "
      f"numpy {np.__version__} · xgboost {xgboost.__version__} · "
      f"lightgbm {lightgbm.__version__} · catboost {catboost.__version__} · seed={SEED}")


# ---------------------------------------------------------------- 후보 25종
def xgb_factory(p):
    return XGBRegressor(n_estimators=p["n"], max_depth=p["d"], learning_rate=p["lr"],
                        subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                        random_state=SEED, n_jobs=4, verbosity=0, tree_method="hist")


def cat_factory(p):
    return CatBoostRegressor(iterations=p["n"], depth=p["d"], learning_rate=p["lr"],
                             random_seed=SEED, verbose=0, allow_writing_files=False)


def lgb_factory(p):
    return LGBMRegressor(n_estimators=p["n"], max_depth=p["d"], num_leaves=2 ** p["d"] - 1,
                         learning_rate=p["lr"], subsample=0.8, subsample_freq=1,
                         random_state=SEED, n_jobs=4, verbose=-1)


XGB_GRID = [{"n": n, "d": d, "lr": lr} for n in (300, 800) for d in (3, 6) for lr in (0.03, 0.1)]
CAT_GRID = [{"n": n, "d": d, "lr": lr} for n in (500, 1500) for d in (4, 6) for lr in (0.03, 0.1)]
LGB_GRID = [{"n": n, "d": d, "lr": lr} for n in (300, 800) for d in (3, 6) for lr in (0.03, 0.1)]

specs = list(s1.build_specs())
specs.append(ev.ModelSpec("XGBoost", "트리", xgb_factory, XGB_GRID, onnx_capable=False))
specs.append(ev.ModelSpec("CatBoost", "트리", cat_factory, CAT_GRID, onnx_capable=False))
specs.append(ev.ModelSpec("LightGBM", "트리", lgb_factory, LGB_GRID, onnx_capable=False))

# ★ 타깃 변환만 raw 로 교체 — 이 한 줄이 로짓판과의 유일한 차이다.
for sp_ in specs:
    sp_.transform = ev.TRANSFORMS["raw"]
assert all(s.transform.kind == "raw" for s in specs)
print(f"후보 {len(specs)}종 · 타깃 변환 = raw (전부 확인)")

sp = ev.load_splits("I1", "r1")
print(f"입력 {sp.feature_names} · 학습 {len(sp.X_train)}행 → held-out {len(sp.X_valid)}행")
print(f"학습셋 TT-1006 최대 {sp.X_train[:, 0].max():.3f} °C · 케이스 스터디 요구 365 °C\n")


def clamp_count(meta, preds) -> int:
    """365°C 예측이 350°C 예측과 1e-12 이내로 같은 (총유량,B/P) 슬라이스 수 (0~9)."""
    n = 0
    for m, p in zip(meta, preds):
        if m["T"] != 350.0:
            continue
        p365 = next(q for mm, q in zip(meta, preds) if mm["T"] == 365.0
                    and mm["total"] == m["total"] and mm["bp"] == m["bp"])
        if abs(p365 - p) < 1e-12:
            n += 1
    return n


# ---------------------------------------------------------------- 1. 25종 실행
print("=" * 100)
print("1. raw-X 25종 — held-out(10/17–18) RMSE · 물리검사 · 예측범위")
print("=" * 100)
print(f"{'모델':<22}{'계열':<9}{'RMSE':>10}{'R²':>9}{'단조':>7}{'clamp':>7}"
      f"{'예측최대':>10}{'[0,1]':>7}{'초':>7}")
print("-" * 100)

rows = []
for spec in specs:
    t0 = time.time()
    best, s20 = ev.tune(spec, sp)
    est = ev.fit_on_logit(spec, best, sp.X_train, sp.y_train)   # transform=raw 이므로 X 에 직접 적합
    pred_va = ev.predict_conv(est, sp.X_valid)
    m = ev.conversion_metrics(sp.y_valid, pred_va)
    phys = ev.physical_check(spec, best, sp)
    grid_pred = np.asarray(phys["preds"], dtype=float)
    pred_max = float(max(grid_pred.max(), pred_va.max()))
    pred_min = float(min(grid_pred.min(), pred_va.min()))
    range_valid = bool(pred_max <= 1.0 and pred_min >= 0.0)
    cl = clamp_count(phys["meta"], phys["preds"])
    dt = time.time() - t0

    rows.append({
        "name": spec.name, "family": spec.family,
        "rmse": m["rmse"], "r2": m["r2"],
        "T_monotone": phys["T_monotone_slices"], "clamp_9": cl,
        "pred_max": pred_max, "pred_min": pred_min,
        "pred_max_grid": float(grid_pred.max()), "pred_max_holdout": float(pred_va.max()),
        "grid_over1": int(np.sum(grid_pred > 1.0)),
        "range_valid": range_valid,
        "best_params": {k: str(v) for k, v in best.items()},
        "block20": s20, "seconds": round(dt, 1),
    })
    print(f"{spec.name:<22}{spec.family:<9}{m['rmse']:>10.5f}{m['r2']:>9.4f}"
          f"{str(phys['T_monotone_slices']) + '/9':>7}{str(cl) + '/9':>7}"
          f"{pred_max:>10.5f}{('OK' if range_valid else '위반'):>7}{dt:>7.1f}")

rows.sort(key=lambda r: r["rmse"])

# ---------------------------------------------------------------- 2. 순위표
print("\n" + "=" * 100)
print("2. raw-X 순위표 (held-out RMSE 오름차순)")
print("=" * 100)
print(f"{'순위':>4}  {'모델':<22}{'계열':<9}{'RMSE':>10}{'단조':>7}{'clamp':>7}{'[0,1]':>7}")
print("-" * 100)
for i, r in enumerate(rows, 1):
    print(f"{i:>4}  {r['name']:<22}{r['family']:<9}{r['rmse']:>10.5f}"
          f"{str(r['T_monotone']) + '/9':>7}{str(r['clamp_9']) + '/9':>7}"
          f"{('OK' if r['range_valid'] else '위반'):>7}")

# ---------------------------------------------------------------- 3. 로짓판 대비
print("\n" + "=" * 100)
print("3. 로짓판(기존 §3.5.1 표) 대비 변화")
print("=" * 100)
if LOGIT_JSON.exists():
    logit_rows = {r["name"]: r for r in json.loads(LOGIT_JSON.read_text(encoding="utf-8"))["전환율_전종"]}
    print(f"{'모델':<22}{'raw-X':>10}{'로짓':>10}{'차':>10}   비고")
    print("-" * 100)
    for r in rows:
        lg = logit_rows.get(r["name"])
        if lg is None:
            continue
        d = r["rmse"] - lg["rmse"]
        note = ""
        if not r["range_valid"]:
            note = "raw 에서 [0,1] 위반"
        print(f"{r['name']:<22}{r['rmse']:>10.5f}{lg['rmse']:>10.5f}{d:>+10.5f}   {note}")
    same_mono = all(logit_rows[r["name"]]["T_monotone"] == r["T_monotone"]
                    and logit_rows[r["name"]]["clamp_9"] == r["clamp_9"]
                    for r in rows if r["name"] in logit_rows)
    print(f"\n  물리검사(단조·clamp)가 타깃 변환과 무관하게 동일한가: "
          f"{'예 — 25종 전부 일치' if same_mono else '아니오(아래 확인 필요)'}")
    if not same_mono:
        for r in rows:
            lg = logit_rows.get(r["name"])
            if lg and (lg["T_monotone"] != r["T_monotone"] or lg["clamp_9"] != r["clamp_9"]):
                print(f"    - {r['name']}: 단조 raw {r['T_monotone']}/9 vs 로짓 {lg['T_monotone']}/9 · "
                      f"clamp raw {r['clamp_9']}/9 vs 로짓 {lg['clamp_9']}/9")
else:
    print(f"  (로짓판 JSON 없음: {LOGIT_JSON} — 대조 생략)")

# ---------------------------------------------------------------- 4. §3.5.5 근거
print("\n" + "=" * 100)
print("4. §3.5.5 로짓 도입 근거 — 순수 1차 선형이 X>1 을 뱉는다")
print("=" * 100)
lin = next(r for r in rows if r["name"] == "Linear 1차")
print(f"  [순수 1차 선형 · raw-X]")
print(f"    held-out RMSE            {lin['rmse']:.5f}   (앵커 {ANCHOR['linear_raw_rmse']})")
print(f"    held-out 예측 최대        {lin['pred_max_holdout']:.5f}   (앵커 {ANCHOR['linear_raw_holdout_max']})")
print(f"    케이스 격자 예측 최대       {lin['pred_max_grid']:.5f}   (앵커 {ANCHOR['linear_raw_grid_max']})")
print(f"    격자 27점 중 X>1 인 점     {lin['grid_over1']}/27   (앵커 {ANCHOR['linear_raw_grid_over1']})")
print(f"    물리적으로 불가능한 값(X>1)을 내므로 range_valid = {lin['range_valid']}")

# 로짓-선형(참고) — 같은 입력·분할로 재적합
spec_logit = ev.ModelSpec("Linear 1차 (로짓)", "선형", lambda p: LinearRegression(), [{}])
spec_logit.transform = ev.TRANSFORMS["logit"]
est_l = ev.fit_on_logit(spec_logit, {}, sp.X_train, sp.y_train)
pred_l = ev.predict_conv(est_l, sp.X_valid)
m_l = ev.conversion_metrics(sp.y_valid, pred_l)
phys_l = ev.physical_check(spec_logit, {}, sp)
grid_l = np.asarray(phys_l["preds"], dtype=float)
print(f"\n  [로짓-선형 · 참고 — 배포 모델 계열]")
print(f"    held-out RMSE            {m_l['rmse']:.5f}   (앵커 {ANCHOR['logit_rmse']})")
print(f"    held-out 예측 최대        {pred_l.max():.5f}")
print(f"    케이스 격자 예측 최대       {grid_l.max():.5f}   (구조적으로 1 미만)")
print(f"    격자 27점 중 X>1 인 점     {int(np.sum(grid_l > 1.0))}/27")

print(f"\n  [대조] 순수 1차 → 로짓-선형")
print(f"    held-out RMSE  {lin['rmse']:.5f} → {m_l['rmse']:.5f}  "
       f"(개선 {lin['rmse'] - m_l['rmse']:+.5f}, {100*(lin['rmse']-m_l['rmse'])/lin['rmse']:.1f}%)")
print(f"    격자 예측 최대  {lin['pred_max_grid']:.5f} → {grid_l.max():.5f}  "
      f"([0,1] 위반 {lin['grid_over1']}점 → {int(np.sum(grid_l > 1.0))}점)")

# 앵커 대조
print("\n  [앵커 대조] 재현값이 기대와 어긋나면 재현값을 따른다")
for label, got, ref, tol in (("순수1차 RMSE", lin["rmse"], ANCHOR["linear_raw_rmse"], 5e-4),
                             ("순수1차 held-out 최대", lin["pred_max_holdout"], ANCHOR["linear_raw_holdout_max"], 5e-3),
                             ("순수1차 격자 최대", lin["pred_max_grid"], ANCHOR["linear_raw_grid_max"], 5e-2),
                             ("순수1차 격자 X>1 점수", lin["grid_over1"], ANCHOR["linear_raw_grid_over1"], 0),
                             ("로짓-선형 RMSE", m_l["rmse"], ANCHOR["logit_rmse"], 5e-4)):
    ok = abs(got - ref) <= tol
    print(f"    [{'OK ' if ok else '차이'}] {label}: 재현 {got:.5f} · 기대 {ref}")
    if not ok:
        NOTE.append(f"{label}: 재현 {got:.5f} ≠ 기대 {ref} → 재현값 채택")

# ---------------------------------------------------------------- 저장
payload = {
    "설명": "전환율(ONNX1) 스크리닝 25종 — raw-X(로짓 미적용) 기준",
    "규약": {
        "타깃": "X_conv_r1_deployed (원시 X, 변환 없음)",
        "튜닝": "블록20%(캠페인 앞 80%/뒤 20%)",
        "성능": "학습 10/13-16 → held-out 10/17-18 (X 스케일 RMSE)",
        "seed": SEED,
        "격자": "T 335/350/365 °C × 총유량 15000/16500/18000 × B/P 4/5/6",
        "clamp_정의": "365°C 예측이 350°C 예측과 1e-12 이내로 같은 슬라이스 수 / 9",
        "range_valid": "격자 27점과 held-out 예측이 전부 [0,1] 안에 드는가",
    },
    "버전": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
             "numpy": np.__version__, "xgboost": xgboost.__version__,
             "lightgbm": lightgbm.__version__, "catboost": catboost.__version__},
    "전환율_25종_rawX": rows,
    "로짓_참고": {
        "name": "Linear 1차 (로짓-선형, 배포 계열)",
        "rmse": m_l["rmse"], "r2": m_l["r2"],
        "pred_max_holdout": float(pred_l.max()), "pred_max_grid": float(grid_l.max()),
        "grid_over1": int(np.sum(grid_l > 1.0)),
        "T_monotone": phys_l["T_monotone_slices"],
    },
    "앵커_대조_비고": NOTE,
}
OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n저장: {OUT_JSON}")
print("=" * 100)
if NOTE:
    print("기대 앵커와 다른 항목(재현값을 채택함):")
    for n in NOTE:
        print("  -", n)
else:
    print("기대 앵커 전부 일치.")
print("배포 모델(reactor_conversion_r1.onnx = 로짓-선형)은 이 스크립트가 건드리지 않는다.")
print("=" * 100)
