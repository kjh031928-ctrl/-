"""train_extra.py — Phase 10-B 다양한 회귀 모델 1차 스크리닝 (조원 전용 개인 파일).

목적(Phase 9 §3 질문2 후속): 1차 선형(외삽 O·물리위반 X>1)과 GBR(외삽 X·평탄화)
사이의 빈 공간을 메우는 모델이 있는지 탐색한다. **내삽 성능과 외삽 거동을 동시에** 본다.

범위(1차 스크리닝이므로 단순화):
  - 입력세트 I2 고정 / 라벨 r0 고정 / 가중치 없음 고정.
  - 학습 10/13~16, 검증 10/17-18 (Split 컬럼 기준, 무작위 분할 금지 — CLAUDE.md §1-3).
  - 조합 전체 확장은 유망 모델이 나온 뒤에 한다.

원칙:
  - 공용 모듈(data_loader, metrics, interface)은 import 만 하고 수정하지 않는다(CLAUDE.md §3).
  - train_linear.py / train_gbr.py 도 건드리지 않는다.
  - float64 유지(CLAUDE.md §2.4). 시드 42 고정.
  - 문서에 적는 수치는 반드시 저장된 json 값을 쓰고 임의 조정하지 않는다.
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import data_loader
import interface
import metrics

# --- 스크리닝 고정 조건 (본 파일 상단 docstring 참조) ---
FEATURE_SET = "I2"     # PT-1004(라벨 계산에 안 쓰인 압력) — 누수 무관한 최선 입력(Phase 6 §2b)
LABEL = "r0"
WEIGHTED = False

# 결과 파일명 접두어 — metrics.save_result 가 "{model}_{fs}_{label}_{tag}.json" 을 만든다.
# model 이름을 "extra_*" 로 두어 results/extra_*.json 이 되게 한다.
MODEL_PREFIX = "extra"

# --- TT-1006 외삽 스윕 (Phase 7-2 와 동일한 방법론) ---
# 학습셋 TT-1006 상한 349.977 °C (CLAUDE.md §5 기준값). 그 위가 외삽 구간.
SWEEP_T_MIN = 330.0    # Phase 7-2 스윕 범위 하한
SWEEP_T_MAX = 370.0    # Phase 7-2 스윕 범위 상한
SWEEP_T_STEP = 1.0     # 1 °C 간격
TRAIN_T_MAX = 349.977  # 학습셋 TT-1006 최댓값 (CLAUDE.md §5 검증 기준값)

# 전환율의 물리적 상·하한. 전환율 정의상 0~1 (CLAUDE.md §2.1 "출력(라벨) 0~1").
X_PHYS_MIN = 0.0
X_PHYS_MAX = 1.0

# "평탄화" 판정 기준 — 10/14 노이즈 바닥 σ (CLAUDE.md §5 기준값 0.003068).
# 임의 임계값이 아니라 데이터에서 관측된 값을 쓴다: 외삽 구간 예측 변화폭이 이 σ 보다
# 작으면 노이즈와 구별되지 않으므로 사실상 상수로 본다(Phase 4 검증3 은 진단 용도로만 사용).
NOISE_FLOOR_SIGMA = 0.003068


def build_models(seed: int) -> list[tuple[str, object, bool]]:
    """(이름, 추정기, 스케일링여부) 목록. 모두 scikit-learn 기본 하이퍼파라미터.

    random_state 는 지원하는 추정기에만 넘긴다
    (KNeighborsRegressor·SVR 은 random_state 인자가 없다 — 결정론적).

    스케일링: 거리·커널·경사 기반 모델(KNN/SVR/MLP/GPR)은 입력 스케일에 민감하다.
    본 데이터의 입력 스케일은 TT-1006≈340, FT-1004≈3000, FRC-1004≈5, PT-1004≈2576 으로
    3자릿수 차이가 나므로, 스케일링 없이는 사실상 큰 값 축(FT/PT)만 보게 되어 비교 자체가
    무의미해진다. 따라서 해당 4종은 StandardScaler 를 Pipeline 으로 붙인다.
    Pipeline 이므로 스케일러는 **학습셋에만 fit** 되어 검증셋 정보 누수가 없다.
    트리 계열(RF/ET/HGB)은 분할 기준이 순서에만 의존해 스케일 불변이므로 붙이지 않는다.
    """
    return [
        ("rf", RandomForestRegressor(random_state=seed), False),
        ("et", ExtraTreesRegressor(random_state=seed), False),
        ("hgb", HistGradientBoostingRegressor(random_state=seed), False),
        # n_neighbors=5 는 scikit-learn 기본값이자 지시 명시값.
        ("knn", KNeighborsRegressor(n_neighbors=5), True),
        ("svr", SVR(kernel="rbf"), True),
        # 은닉층 (32,16) 은 지시 명시. max_iter 는 기본 200 으로는 수렴 부족이라 넉넉히 2000.
        ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=seed), True),
        ("gpr", GaussianProcessRegressor(random_state=seed), True),
    ]


def _wrap(estimator, scale: bool):
    """스케일링이 필요한 모델은 StandardScaler 와 함께 Pipeline 으로 감싼다."""
    if not scale:
        return estimator
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def sweep_baseline(df, cfg) -> tuple[np.ndarray, list[str], int]:
    """TT-1006 스윕용 고정 입력 1행을 10/12 정상운전 구간에서 만든다.

    Phase 7-2 와 동일: TT-1006 외 입력은 10/12(정상운전검사) 평균으로 고정한다.
    10/12 는 481행 전부 동일 조건이라 평균 = 그 상수값이다(Phase 8 실측 확인).
    하드코딩 대신 데이터에서 직접 계산한다.
    """
    cols = interface.feature_columns(cfg, FEATURE_SET)
    normal_df = data_loader.get_split(df, cfg["split"]["normalcheck_value"])  # 10/12
    base = normal_df[cols].to_numpy(dtype=np.float64).mean(axis=0)
    t_idx = cols.index("TT-1006.PV")
    return base, cols, t_idx


def extrapolation_probe(model, base: np.ndarray, t_idx: int) -> dict:
    """TT-1006 을 330→370 °C 로 스윕한 예측 곡선과 그 요약을 반환.

    학습상한(349.977 °C) 위 구간에서 평탄화되는지 / 계속 증가·발산하는지 /
    물리 상한(0~1)을 위반하는지를 기록한다.
    """
    temps = np.arange(
        SWEEP_T_MIN, SWEEP_T_MAX + SWEEP_T_STEP / 2.0, SWEEP_T_STEP, dtype=np.float64
    )
    grid = np.tile(base, (len(temps), 1))
    grid[:, t_idx] = temps
    preds = np.asarray(model.predict(grid), dtype=np.float64)

    extrap = temps > TRAIN_T_MAX  # 외삽 구간 마스크
    p_ex = preds[extrap]
    t_ex = temps[extrap]

    # 외삽 구간 변화폭 — 첫 외삽점 대비 마지막점(370 °C)
    delta_extrap = float(p_ex[-1] - p_ex[0])
    span_extrap = float(p_ex.max() - p_ex.min())

    # 물리 상한/하한 위반
    viol_mask = (preds > X_PHYS_MAX) | (preds < X_PHYS_MIN)
    n_viol = int(viol_mask.sum())
    first_viol_T = float(temps[viol_mask][0]) if n_viol else None

    # 형태 판정 — 노이즈 바닥 σ 기준(위 상수 주석 참조)
    if span_extrap < NOISE_FLOOR_SIGMA:
        shape = "평탄"
    elif delta_extrap > 0:
        shape = "증가"
    else:
        shape = "감소"

    def at(t: float) -> float:
        return float(preds[int(np.argmin(np.abs(temps - t)))])

    return {
        "sweep_T_min": SWEEP_T_MIN,
        "sweep_T_max": SWEEP_T_MAX,
        "sweep_T_step": SWEEP_T_STEP,
        "train_T_max": TRAIN_T_MAX,
        "shape": shape,
        "violates_physical_bounds": bool(n_viol > 0),
        "n_violation_points": n_viol,
        "first_violation_T": first_viol_T,
        "pred_at_340": at(340.0),
        "pred_at_350": at(350.0),
        "pred_at_360": at(360.0),
        "pred_at_365": at(365.0),
        "pred_at_370": at(370.0),
        "delta_extrap_350_to_370": delta_extrap,
        "span_extrap": span_extrap,
        "pred_min_all": float(preds.min()),
        "pred_max_all": float(preds.max()),
        "curve": [
            {"T": float(t), "pred": float(p)} for t, p in zip(temps, preds)
        ],
    }


def run_one(df, cfg, name: str, estimator, scale: bool) -> dict:
    """모델 1종을 학습·검증하고 외삽 스윕까지 수행해 결과 dict 를 반환."""
    seed = int(cfg["seed"])

    train_df = data_loader.get_split(df, cfg["split"]["train_value"])  # 10/13~16
    valid_df = data_loader.get_split(df, cfg["split"]["valid_value"])  # 10/17~18

    X_train, y_train = data_loader.get_Xy(train_df, FEATURE_SET, LABEL)
    X_valid, y_valid = data_loader.get_Xy(valid_df, FEATURE_SET, LABEL)

    model = _wrap(estimator, scale)

    # 수렴 경고 등은 판단 근거이므로 삼키지 않고 건수를 기록한다.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        model.fit(X_train, y_train)  # 가중치 없음 고정
        fit_seconds = time.time() - t0
    warn_types = sorted({w.category.__name__ for w in caught})

    y_pred = model.predict(X_valid)

    # 검증 지표에 ML_weight 미적용(지시서 §4.5).
    conversion = metrics.conversion_metrics(y_valid, y_pred)
    cumene = metrics.cumene_metrics(y_valid, y_pred, data_loader.get_n_propene(valid_df))

    base, cols, t_idx = sweep_baseline(df, cfg)
    extrap = extrapolation_probe(model, base, t_idx)

    result = interface.make_result(
        model=f"{MODEL_PREFIX}_{name}",
        feature_set=FEATURE_SET,
        label=LABEL,
        weighted=WEIGHTED,
        hyperparams=estimator.get_params(),
        seed=seed,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        n_train=len(train_df),
        n_valid=len(valid_df),
        metrics={"conversion": conversion, "cumene_kgh": cumene},
    )
    # 표준 스키마 밖 추가 정보(공용 interface 는 수정하지 않고 dict 에만 덧붙임).
    result["scaled_inputs"] = bool(scale)
    result["fit_seconds"] = round(float(fit_seconds), 3)
    result["fit_warnings"] = warn_types
    result["sweep_fixed_inputs"] = {
        c: float(v) for c, v in zip(cols, base) if c != "TT-1006.PV"
    }
    result["extrapolation"] = extrap
    return result


def print_summary(results: list[dict]) -> None:
    """내삽 성능 + 외삽 거동 요약 표."""
    print()
    print(f"=== Phase 10-B 모델 스크리닝 ({FEATURE_SET}/{LABEL}/가중치없음) ===")
    print(f"    학습 10/13~16, 검증 10/17-18. 학습 TT-1006 상한 {TRAIN_T_MAX} C")
    print()
    header = (
        f"{'모델':<10}{'RMSE':>10}{'MAE':>10}{'MAPE%':>9}{'R2':>9}"
        f"{'쿠멘%':>9}{'  350C':>9}{'  365C':>9}{'외삽폭':>9}  {'거동':<6}{'물리위반':<9}{'초':>7}"
    )
    print(header)
    print("-" * 112)
    for r in results:
        c = r["metrics"]["conversion"]
        k = r["metrics"]["cumene_kgh"]
        e = r["extrapolation"]
        viol = f"위반({e['first_violation_T']:.0f}C~)" if e["violates_physical_bounds"] else "없음"
        print(
            f"{r['model'].replace(MODEL_PREFIX + '_', ''):<10}"
            f"{c['rmse']:>10.6f}{c['mae']:>10.6f}{c['mape']:>9.4f}{c['r2']:>9.5f}"
            f"{k['pct_of_mean']:>9.4f}"
            f"{e['pred_at_350']:>9.5f}{e['pred_at_365']:>9.5f}"
            f"{e['delta_extrap_350_to_370']:>9.5f}  {e['shape']:<6}{viol:<9}"
            f"{r['fit_seconds']:>7.1f}"
        )
    print()
    print(f"외삽폭 = 예측(370C) - 예측(350C). |폭| < {NOISE_FLOOR_SIGMA} (10/14 노이즈 바닥 s)")
    print("이면 노이즈와 구별 불가로 보아 '평탄' 으로 판정한다.")
    warned = [r for r in results if r["fit_warnings"]]
    if warned:
        print()
        print("학습 경고 발생:")
        for r in warned:
            print(f"  - {r['model']}: {', '.join(r['fit_warnings'])}")


def main() -> list[dict]:
    cfg = data_loader.load_config()
    df = data_loader.load_data(cfg)
    seed = int(cfg["seed"])

    results: list[dict] = []
    for name, estimator, scale in build_models(seed):
        print(f"[학습] {name} ...", flush=True)
        result = run_one(df, cfg, name, estimator, scale)
        path = metrics.save_result(result, cfg["paths"]["results_dir"])
        print(f"[저장] {path.name}  ({result['fit_seconds']:.1f}s)", flush=True)
        results.append(result)

    print_summary(results)
    return results


if __name__ == "__main__":
    main()
