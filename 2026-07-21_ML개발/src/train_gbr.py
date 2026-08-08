"""train_gbr.py — 부스팅 모델 담당자용 (개인 파일 — CLAUDE.md §3).

담당: 부스팅 모델 담당(개인 파일) / 구현 Phase: 5, 튜닝 Phase: 10.

역할: GradientBoostingRegressor 학습. Phase 5 는 scikit-learn 기본값,
Phase 10 은 n_estimators/learning_rate/max_depth/min_samples_leaf 격자 탐색(지시서 §5).
시드 42 고정. 학습범위 밖 평탄화 거동은 Phase 7 에서 별도 시험.

Phase 5 실행 범위: 입력세트(I1/I2/I3) x 라벨(r0/r1) x 가중치(없음/있음) = 12 조합.

원칙:
  - 공용 모듈(data_loader, metrics, interface)은 import 만 하고 수정하지 않는다(CLAUDE.md §3).
  - 분할은 Split 컬럼 기준만 사용. 무작위 분할 금지(CLAUDE.md §1-3).
  - ML_weight 는 학습에만 적용, 검증 지표에는 미적용(지시서 §4.5).
  - float64 유지. float32 는 ONNX export 시점에만(CLAUDE.md §2.4).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

import data_loader
import interface
import metrics

# 모델 이름 — 결과 파일명 접두어(gbr_*)가 된다(metrics.save_result).
MODEL_NAME = "gbr"

# 난수 시드 — config/base.yaml seed 와 동일해야 함(CLAUDE.md §2.3, 전 구간 42 고정).
_SEED_FROM_CONFIG_KEY = "seed"


def _fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
) -> tuple[GradientBoostingRegressor, np.ndarray]:
    """GBR(기본 하이퍼파라미터)을 학습하고 검증셋 예측을 반환.

    Phase 5 는 scikit-learn 기본값 그대로 사용한다(튜닝은 Phase 10 — 지시서 §5).
    random_state 만 시드 고정 목적으로 지정한다.
    """
    model = GradientBoostingRegressor(random_state=seed)
    # sample_weight=None 은 sklearn 에서 '가중치 없음'과 동일하게 처리된다.
    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = model.predict(X_valid)
    return model, y_pred


def run_one(
    df,
    cfg: dict,
    feature_set: str,
    label: str,
    weighted: bool,
) -> dict:
    """조합 1개(입력세트 x 라벨 x 가중치)를 학습·검증하고 결과 dict 를 반환."""
    seed = int(cfg[_SEED_FROM_CONFIG_KEY])

    train_df = data_loader.get_split(df, cfg["split"]["train_value"])  # 10/13~10/16
    valid_df = data_loader.get_split(df, cfg["split"]["valid_value"])  # 10/17~10/18

    X_train, y_train = data_loader.get_Xy(train_df, feature_set, label)
    X_valid, y_valid = data_loader.get_Xy(valid_df, feature_set, label)

    # 라벨 결측 확인. 재순환 1 mol% 라벨 bracket 실패는 0건이 기준값(CLAUDE.md §5)이므로
    # 결측이 있으면 임의 처리(dropna 등) 하지 않고 멈추고 보고한다(CLAUDE.md §1-4).
    for name, y in (("학습", y_train), ("검증", y_valid)):
        n_nan = int(np.isnan(y).sum())
        if n_nan:
            raise RuntimeError(
                f"{feature_set}/{label}: {name}셋 라벨에 결측 {n_nan}건. "
                "기준값(bracket 실패 0건)과 불일치 — 원인 조사 필요."
            )

    sample_weight = data_loader.get_weights(train_df) if weighted else None
    model, y_pred = _fit_predict(X_train, y_train, X_valid, sample_weight, seed)

    # 검증 지표는 가중치 미적용(지시서 §4.5).
    conversion = metrics.conversion_metrics(y_valid, y_pred)
    cumene = metrics.cumene_metrics(y_valid, y_pred, data_loader.get_n_propene(valid_df))

    return interface.make_result(
        model=MODEL_NAME,
        feature_set=feature_set,
        label=label,
        weighted=weighted,
        hyperparams=model.get_params(),
        seed=seed,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        n_train=len(train_df),
        n_valid=len(valid_df),
        metrics={"conversion": conversion, "cumene_kgh": cumene},
    )


def print_summary(results: list[dict]) -> None:
    """12개 조합 요약 표를 콘솔에 출력."""
    header = (
        f"{'입력':<5}{'라벨':<5}{'가중치':<7}"
        f"{'RMSE':>11}{'MAE':>11}{'MAPE(%)':>10}{'R2':>9}"
        f"{'큐멘RMSE':>12}{'평균대비(%)':>11}"
    )
    print()
    print(f"=== GBR (기본 하이퍼파라미터) 검증셋 결과 - {len(results)}개 조합 ===")
    print(header)
    print("-" * 84)
    for r in results:
        c = r["metrics"]["conversion"]
        k = r["metrics"]["cumene_kgh"]
        w = "있음" if r["weighted"] else "없음"
        print(
            f"{r['feature_set']:<5}{r['label']:<5}{w:<7}"
            f"{c['rmse']:>11.6f}{c['mae']:>11.6f}{c['mape']:>10.4f}{c['r2']:>9.5f}"
            f"{k['rmse']:>12.4f}{k['pct_of_mean']:>11.4f}"
        )
    print()
    print("주의: I3 는 PT-1100.PV 가 라벨 계산(SRK 압력)에도 쓰인 변수라 정보 누수 소지가")
    print("      있다(CLAUDE.md 2.1). I1 대비 성능이 크게 좋으면 개선이 아닌 누수를 먼저 의심.")


def main() -> list[dict]:
    cfg = data_loader.load_config()
    df = data_loader.load_data(cfg)

    results: list[dict] = []
    # 3(입력세트) x 2(라벨) x 2(가중치) = 12 조합
    for feature_set in interface.FEATURE_SETS:  # I1, I2, I3
        for label in ("r0", "r1"):
            for weighted in (False, True):
                result = run_one(df, cfg, feature_set, label, weighted)
                path = metrics.save_result(result, cfg["paths"]["results_dir"])
                print(f"[저장] {path.name}")
                results.append(result)

    print_summary(results)
    return results


if __name__ == "__main__":
    main()
