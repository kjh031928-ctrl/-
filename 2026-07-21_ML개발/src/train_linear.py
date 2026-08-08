"""train_linear.py — 선형 모델(LinearRegression) 학습·검증·저장.

담당: 김지훈 전용 (CLAUDE.md §3). 조원은 수정하지 않는다.
공용 모듈(data_loader, metrics, interface)은 import 만 하고 수정하지 않는다.

내용(Phase 5-B):
  입력세트 I1/I2/I3 (3) × 라벨 r0/r1 (2) × 가중치 없음/있음 (2) = 12개 조합을
  sklearn LinearRegression 기본 하이퍼파라미터로 실행한다(튜닝은 이후 Phase).

내용(Phase 10-A) — 선형 계열 개선 격자탐색:
  Phase 7에서 1차 선형이 356°C부터 X>1(물리 위반)을 냈다. 전환율은 정의상 [0,1]이므로
  아래 후보를 12개 조합(I1/I2/I3 × r0/r1 × 가중 유무) 전부에 대해 검증(10/17-18)으로 비교한다.
    (a) 1차 선형 + 하드클립 np.clip(pred,0,1)
    (b) 로짓변환 + 1차 선형   z=log(y/(1-y)), y는 eps=1e-6 클립 → 회귀 → X=sigmoid(z). (0,1) 보장.
    (c) Ridge(alpha∈[0.001,0.01,0.1,1.0]) + 하드클립
    (d) Lasso(alpha∈[0.001,0.01,0.1,1.0]) + 하드클립
  2차 다항은 (일반공간) 제외(외삽 발산). 로짓공간 2차만 1개 참고용으로 넣고 발산하면 제외.
  각 모델의 365°C 외삽 예측값(10/12 조건 FT=3000,FRC=5.0, 압력=10/12평균)을 기록해 상한 1.0 위반 확인.
  결과는 results/ 에 linear2_* 파일명으로 저장(Phase 5-B의 linear_* 와 구분).

규칙:
  - 학습 = get_split(df,"학습"), 검증 = get_split(df,"검증"). 무작위 분할 금지(CLAUDE.md §1-3).
  - 가중치 "있음"이면 fit(..., sample_weight=get_weights(학습)). 검증 지표에는 미적용(§4.5).
  - seed=42 (config). float64 유지(CLAUDE.md §2.4).
  - 결과는 metrics.save_result 로 results/ 에 저장.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

# Windows 콘솔(cp949)에서 한글·기호 출력 시 UnicodeEncodeError 방지. 데이터·계산엔 영향 없음.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
from scipy.special import expit  # 수치 안정 sigmoid: 1/(1+exp(-z))
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

import data_loader as dl
import interface as itf
import metrics as mt

MODEL_NAME = "linear"  # save_result 파일명 접두 → gbr 결과와 비충돌(CLAUDE.md 협업 규칙)

# --- Phase 10-A 상수 ---
LOGIT_EPS = 1e-6  # 로짓변환 전 y 클립 경계(지시서 Phase 10-A). z=log(y/(1-y)) 발산 방지.
RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0)  # 지시서 Phase 10-A (c)
LASSO_ALPHAS = (0.001, 0.01, 0.1, 1.0)  # 지시서 Phase 10-A (d)
EXTRAP_T = 365.0  # 케이스 스터디 상한(°C) — 외삽 예측 확인 지점(지시서 Phase 7/10-A)
EXTRAP_FT = 3000.0  # 10/12 정상운전 FT-1004 (지시서 Phase 10-A)
EXTRAP_FRC = 5.0    # 10/12 정상운전 FRC-1004 (지시서 Phase 10-A)


def run_all() -> list[dict]:
    cfg = dl.load_config()
    seed = cfg["seed"]  # 42 (config/base.yaml, CLAUDE.md §2.3)

    df = dl.load_data(cfg)
    train = dl.get_split(df, cfg["split"]["train_value"])   # 10/13~10/16
    valid = dl.get_split(df, cfg["split"]["valid_value"])   # 10/17~10/18 (최종검증)

    w_train = dl.get_weights(train)          # 학습 가중치 (학습에만 사용)
    npr_valid = dl.get_n_propene(valid)      # 검증 큐멘 환산용

    results = []
    for feature_set in itf.FEATURE_SETS:              # I1, I2, I3
        for label in ("r0", "r1"):                    # 재순환 0/1 mol%
            for weighted in (False, True):            # 가중치 없음/있음
                X_tr, y_tr = dl.get_Xy(train, feature_set, label)
                X_va, y_va = dl.get_Xy(valid, feature_set, label)

                # LinearRegression 은 무작위성이 없어 seed 영향은 없으나 규약상 명시 고정.
                model = LinearRegression()
                if weighted:
                    model.fit(X_tr, y_tr, sample_weight=w_train)
                else:
                    model.fit(X_tr, y_tr)

                y_pred = model.predict(X_va)  # 검증은 가중치 미적용

                conv = mt.conversion_metrics(y_va, y_pred)
                cum = mt.cumene_metrics(y_va, y_pred, npr_valid)

                result = itf.make_result(
                    model=MODEL_NAME,
                    feature_set=feature_set,
                    label=label,
                    weighted=weighted,
                    hyperparams=model.get_params(),  # 기본값 기록(재현성)
                    seed=seed,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    n_train=len(train),
                    n_valid=len(valid),
                    metrics={"conversion": conv, "cumene_kgh": cum},
                )
                path = mt.save_result(result, cfg["paths"]["results_dir"])
                result["_path"] = path.name
                results.append(result)
    return results


def print_summary(results: list[dict]) -> None:
    print(f"\n선형(LinearRegression) — 12개 조합 검증 결과 (검증셋 10/17~18, n={results[0]['n_valid']})")
    header = f"{'입력':<4} {'라벨':<4} {'가중치':<7} {'RMSE':>10} {'R2':>9} {'쿠멘%오차':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        c = r["metrics"]["conversion"]
        k = r["metrics"]["cumene_kgh"]
        wtag = "있음" if r["weighted"] else "없음"
        print(
            f"{r['feature_set']:<4} {r['label']:<4} {wtag:<7} "
            f"{c['rmse']:>10.6f} {c['r2']:>9.5f} {k['pct_of_mean']:>9.4f}%"
        )


# =============================================================================
# Phase 10-A — 선형 계열 개선 격자탐색
# =============================================================================

def _extrap_point(cfg: dict, feature_set: str) -> np.ndarray:
    """365°C 외삽 확인용 입력 1점. 10/12 조건(FT=3000,FRC=5.0), 압력=10/12 평균.

    압력 고정값을 하드코딩하지 않고 10/12 캠페인 실측 평균에서 계산한다(§4.2 출처).
    컬럼 순서는 interface.feature_columns 를 따른다.
    """
    cols = itf.feature_columns(cfg, feature_set)
    df = dl.load_data(cfg)
    n12 = df[df["Campaign"] == "10/12"]
    base = {"TT-1006.PV": EXTRAP_T, "FT-1004.PV": EXTRAP_FT, "FRC-1004.PV": EXTRAP_FRC}
    row = []
    for c in cols:
        if c in base:
            row.append(base[c])
        else:  # 압력 태그(PT-1004/PT-1100) → 10/12 정상운전 평균
            row.append(float(n12[c].mean()))
    return np.array([row], dtype=np.float64)


def _fit_predict(method, alpha, X_tr, y_tr, w, X_va):
    """(method, alpha)에 맞춰 학습하고 (검증 예측(후처리 반영), 원시 예측함수) 반환.

    - hardclip/ridge/lasso: 원공간 회귀 → 검증 예측은 clip(·,0,1). 원시함수는 클립 전 값(상한위반 확인용).
    - logit/logit2: z=log(y/(1-y)) (y∈[eps,1-eps]) 회귀 → 예측은 sigmoid(z)로 (0,1) 보장.
    반환 raw_fn(x)는 '후처리 전' 값(클립계열=선형 원시, 로짓계열=sigmoid 결과 자체).
    """
    if method in ("hardclip", "ridge", "lasso"):
        if method == "hardclip":
            est = LinearRegression()
        elif method == "ridge":
            est = Ridge(alpha=alpha)  # 좌표하강 없음, 결정적
        else:
            est = Lasso(alpha=alpha, max_iter=200000)  # 수렴 여유
        est.fit(X_tr, y_tr, sample_weight=w)
        raw_fn = lambda x: est.predict(x)  # 클립 전 원시(>1 가능)
        y_va = np.clip(est.predict(X_va), 0.0, 1.0)
        return y_va, raw_fn, est

    if method in ("logit", "logit2"):
        yc = np.clip(y_tr, LOGIT_EPS, 1.0 - LOGIT_EPS)
        z = np.log(yc / (1.0 - yc))
        if method == "logit2":  # 로짓공간 2차(참고용)
            poly = PolynomialFeatures(degree=2, include_bias=False)
            Xtr = poly.fit_transform(X_tr)
            est = LinearRegression().fit(Xtr, z, sample_weight=w)
            raw_fn = lambda x: expit(est.predict(poly.transform(x)))
            y_va = expit(est.predict(poly.transform(X_va)))
        else:
            est = LinearRegression().fit(X_tr, z, sample_weight=w)
            raw_fn = lambda x: expit(est.predict(x))
            y_va = expit(est.predict(X_va))
        return y_va, raw_fn, est

    raise ValueError(f"알 수 없는 method: {method}")


def _model_tag(method, alpha):
    """save_result 파일명 접두(linear2_*). alpha 있는 계열은 alpha 포함해 충돌 방지."""
    if method in ("ridge", "lasso"):
        return f"linear2_{method}_a{alpha}"
    return f"linear2_{method}"


def _iter_methods():
    """(method, alpha) 조합. 클립계열 1 + 로짓 1 + Ridge 4 + Lasso 4 = 10개(조합당)."""
    yield ("hardclip", None)
    yield ("logit", None)
    for a in RIDGE_ALPHAS:
        yield ("ridge", a)
    for a in LASSO_ALPHAS:
        yield ("lasso", a)


def run_linear2_grid() -> list[dict]:
    cfg = dl.load_config()
    seed = cfg["seed"]
    df = dl.load_data(cfg)
    train = dl.get_split(df, cfg["split"]["train_value"])
    valid = dl.get_split(df, cfg["split"]["valid_value"])
    w_train = dl.get_weights(train)
    npr_valid = dl.get_n_propene(valid)

    results = []
    for feature_set in itf.FEATURE_SETS:
        x365 = _extrap_point(cfg, feature_set)
        for label in ("r0", "r1"):
            for weighted in (False, True):
                X_tr, y_tr = dl.get_Xy(train, feature_set, label)
                X_va, y_va = dl.get_Xy(valid, feature_set, label)
                w = w_train if weighted else None

                methods = list(_iter_methods())
                # 로짓공간 2차 참고용: 최고 조합(I2/r0/가중)에만 1개 추가
                if feature_set == "I2" and label == "r0" and weighted:
                    methods.append(("logit2", None))

                for method, alpha in methods:
                    y_pred, raw_fn, _est = _fit_predict(method, alpha, X_tr, y_tr, w, X_va)

                    conv = mt.conversion_metrics(y_va, y_pred)
                    cum = mt.cumene_metrics(y_va, y_pred, npr_valid)

                    raw365 = float(raw_fn(x365)[0])
                    final365 = (
                        float(np.clip(raw365, 0.0, 1.0))
                        if method in ("hardclip", "ridge", "lasso")
                        else raw365  # 로짓계열은 이미 (0,1)
                    )
                    extrap = {
                        "T_C": EXTRAP_T,
                        "raw": raw365,          # 후처리 전(클립계열은 선형 원시)
                        "final": final365,      # 모델 실제 출력
                        "raw_exceeds_1": bool(raw365 > 1.0),
                    }

                    hyper = {"method": method, "alpha": alpha,
                             "postprocess": "hardclip" if method in ("hardclip", "ridge", "lasso") else "logit-sigmoid",
                             "logit_eps": LOGIT_EPS if method in ("logit", "logit2") else None}

                    result = itf.make_result(
                        model=_model_tag(method, alpha),
                        feature_set=feature_set,
                        label=label,
                        weighted=weighted,
                        hyperparams=hyper,
                        seed=seed,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        n_train=len(train),
                        n_valid=len(valid),
                        metrics={"conversion": conv, "cumene_kgh": cum, "extrapolation": extrap},
                    )
                    path = mt.save_result(result, cfg["paths"]["results_dir"])
                    result["_path"] = path.name
                    results.append(result)
    return results


def print_linear2_summary(results: list[dict]) -> None:
    rows = sorted(results, key=lambda r: r["metrics"]["conversion"]["rmse"])
    print(f"\n선형 계열 개선 격자탐색 — {len(rows)}개 (검증셋 10/17~18, n={rows[0]['n_valid']}), RMSE 오름차순")
    header = (f"{'method':<16}{'입력':<4}{'라벨':<4}{'가중':<5}"
              f"{'RMSE':>10}{'MAE':>10}{'MAPE':>8}{'R2':>9}{'쿠멘%':>8}{'365raw':>9}{'365out':>9}{'>1':>4}")
    print(header)
    print("-" * len(header))
    for r in rows:
        c = r["metrics"]["conversion"]; k = r["metrics"]["cumene_kgh"]; e = r["metrics"]["extrapolation"]
        print(f"{r['model']:<16}{r['feature_set']:<4}{r['label']:<4}{('있음' if r['weighted'] else '없음'):<5}"
              f"{c['rmse']:>10.6f}{c['mae']:>10.6f}{c['mape']:>8.4f}{c['r2']:>9.5f}"
              f"{k['pct_of_mean']:>7.4f}%{e['raw']:>9.4f}{e['final']:>9.4f}"
              f"{('예' if e['raw_exceeds_1'] else '-'):>4}")


# =============================================================================
# Phase 10-C — 모델 탐색 증거(오차구조 분석 + 물리 입력변환 시험, 실패 포함 기록)
# =============================================================================
import json  # 오차구조 JSON 직접 저장용 (save_result 스키마 밖 payload)

# 물리 변환 상수 (출처 주석 — CLAUDE.md §4.2)
CELSIUS_TO_K = 273.15    # °C → K (표준 절대영도 오프셋)
INV_T_SCALE = 1000.0     # 1000/T_K 의 1000 = 수치 스케일일 뿐. 선형회귀는 스케일 불변이라
                         #   1/T_K 와 결과 동일(RMSE 소수점까지). run 에서 실증 확인.

CUMENE_MW = 120.196      # 큐멘 몰질량 kg/kmol (config properties.cumene, 쿠멘 kg/h 환산용)


def _logit_regress(X_tr, y_tr, w, X_eval):
    """로짓공간 1차 선형 적합 후 X_eval 예측(sigmoid 역변환). (eval예측, est) 반환."""
    yc = np.clip(y_tr, LOGIT_EPS, 1.0 - LOGIT_EPS)
    z = np.log(yc / (1.0 - yc))
    est = LinearRegression().fit(X_tr, z, sample_weight=w)
    return expit(est.predict(X_eval)), est


def _apply_transform(X, kind):
    """I1 입력행렬 [TT-1006, FT-1004, FRC-1004] 에 물리 변환 적용(로짓은 유지).

    kind: none / invT(1000/T_K) / invF(1/FT) / invTF(둘 다). FRC 열은 불변.
    """
    X = X.astype(np.float64).copy()
    if kind in ("invT", "invTF"):
        X[:, 0] = INV_T_SCALE / (X[:, 0] + CELSIUS_TO_K)  # 1000/T_K (Arrhenius 역온도)
    if kind in ("invF", "invTF"):
        X[:, 1] = 1.0 / X[:, 1]                            # 1/FT (체류시간 비례)
    return X


def _bin_stats(mask, cum_pred, cum_true):
    """구간 마스크에 대한 (n, |오차|평균 kg/h, 편향(예측-실제) 평균 kg/h)."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "abs_err_kgh": None, "bias_kgh": None}
    err = cum_pred[mask] - cum_true[mask]
    return {"n": n, "abs_err_kgh": float(np.mean(np.abs(err))), "bias_kgh": float(np.mean(err))}


def run_error_structure() -> dict:
    """[1] 최종후보(로짓 1차 I1 r0, 가중없음)의 검증셋 오차 구조를 구간별로 분석."""
    cfg = dl.load_config()
    df = dl.load_data(cfg)
    train = dl.get_split(df, cfg["split"]["train_value"])
    valid = dl.get_split(df, cfg["split"]["valid_value"])

    X_tr, y_tr = dl.get_Xy(train, "I1", "r0")
    X_va, y_va = dl.get_Xy(valid, "I1", "r0")
    y_pred, _ = _logit_regress(X_tr, y_tr, None, X_va)  # 가중없음

    npr = dl.get_n_propene(valid)
    cum_true = y_va * npr * CUMENE_MW
    cum_pred = y_pred * npr * CUMENE_MW

    TT = X_va[:, 0]; FT = X_va[:, 1]; FRC = X_va[:, 2]
    T_MAX_TRAIN = 349.977  # 학습셋 TT-1006 최댓값(CLAUDE.md §5 기준값)

    groups = {}
    groups["온도(°C)"] = [("335-345", (TT >= 335) & (TT < 345)), ("345-350", (TT >= 345) & (TT < 350)),
                          ("350-355", (TT >= 350) & (TT < 355)), ("355-360", (TT >= 355) & (TT < 360))]
    groups["전환율 X"] = [("0.30-0.50", (y_va >= 0.30) & (y_va < 0.50)), ("0.50-0.70", (y_va >= 0.50) & (y_va < 0.70)),
                          ("0.70-0.85", (y_va >= 0.70) & (y_va < 0.85)), ("0.85-0.95", (y_va >= 0.85) & (y_va < 0.95)),
                          ("0.95-1.00", (y_va >= 0.95) & (y_va <= 1.00))]
    groups["프로펜유량 FT"] = [("2500-2700", (FT >= 2500) & (FT < 2700)), ("2700-2800", (FT >= 2700) & (FT < 2800)),
                              ("2800-3000", (FT >= 2800) & (FT <= 3000))]
    groups["비율 FRC"] = [("4.4-4.6", (FRC >= 4.4) & (FRC < 4.6)), ("4.6-4.8", (FRC >= 4.6) & (FRC < 4.8)),
                          ("4.8-5.1", (FRC >= 4.8) & (FRC <= 5.1))]
    groups["내삽/외삽"] = [("내삽(≤349.977)", TT <= T_MAX_TRAIN), ("외삽(>350)", TT > 350.0)]

    out = {"model": "로짓1차 I1/r0/가중없음", "n_valid": int(len(valid)), "groups": {}}
    print("\n[1] 오차 구조 분석 — 로짓 1차 I1/r0/가중없음 (검증 10/17-18)")
    for gname, bins in groups.items():
        print(f"\n  · {gname}")
        print(f"    {'구간':16}{'n':>5}{'|오차|평균kg/h':>15}{'편향kg/h':>12}")
        out["groups"][gname] = {}
        for label, mask in bins:
            s = _bin_stats(np.asarray(mask), cum_pred, cum_true)
            out["groups"][gname][label] = s
            ae = f"{s['abs_err_kgh']:.2f}" if s['abs_err_kgh'] is not None else "-"
            bi = f"{s['bias_kgh']:+.2f}" if s['bias_kgh'] is not None else "-"
            print(f"    {label:16}{s['n']:>5}{ae:>15}{bi:>12}")

    out_path = _ROOT_RESULTS(cfg) / "linear3_error_structure_I1_r0_unweighted.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {out_path.name}")
    return out


def _ROOT_RESULTS(cfg):
    from pathlib import Path
    d = Path(cfg["paths"]["results_dir"])
    if not d.is_absolute():
        d = Path(__file__).resolve().parents[1] / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_transform_trials() -> list[dict]:
    """[2] 로짓 유지, 물리 입력변환(1/T, 1/F, 둘다) 시험. 실패도 기록. baseline=변환없음."""
    cfg = dl.load_config(); seed = cfg["seed"]
    df = dl.load_data(cfg)
    train = dl.get_split(df, cfg["split"]["train_value"])
    valid = dl.get_split(df, cfg["split"]["valid_value"])
    npr_valid = dl.get_n_propene(valid)

    X_tr0, y_tr = dl.get_Xy(train, "I1", "r0")
    X_va0, y_va = dl.get_Xy(valid, "I1", "r0")

    # 스케일 불변 실증(정정): OLS 는 수학적으로 스케일 불변이나, 원시 1/T_K(≈0.0016)는
    #   특징행렬 조건수가 커(≈1e8) lstsq 수치오차로 붕괴한다. 1000/T_K 또는 표준화로
    #   조건수를 낮추면 두 스케일이 소수점까지 동일 → "1000 은 조건수 보정용, 결과 무관".
    from sklearn.preprocessing import StandardScaler

    def _logit_rmse(scale, standardize):
        Xt = X_tr0.copy(); Xt[:, 0] = scale / (X_tr0[:, 0] + CELSIUS_TO_K)
        Xv = X_va0.copy(); Xv[:, 0] = scale / (X_va0[:, 0] + CELSIUS_TO_K)
        if standardize:
            sc = StandardScaler().fit(Xt); Xt = sc.transform(Xt); Xv = sc.transform(Xv)
        return mt.conversion_metrics(y_va, _logit_regress(Xt, y_tr, None, Xv)[0])["rmse"]

    r1_raw = _logit_rmse(1.0, False); r1k_raw = _logit_rmse(INV_T_SCALE, False)
    r1_std = _logit_rmse(1.0, True); r1k_std = _logit_rmse(INV_T_SCALE, True)
    cond1 = float(np.linalg.cond(np.column_stack([1.0 / (X_tr0[:, 0] + CELSIUS_TO_K), X_tr0[:, 1:].T.T])))
    print(f"\n[2] 스케일/조건수 실증 (1/T 변환, invT):")
    print(f"    원시   1/T_K   RMSE={r1_raw:.10f} (조건수≈1e8, lstsq 붕괴)")
    print(f"    원시 1000/T_K  RMSE={r1k_raw:.10f} (조건수 낮춰 정상)")
    print(f"    표준화 후: 1/T_K={r1_std:.10f}, 1000/T_K={r1k_std:.10f} "
          f"→ 동일={abs(r1_std - r1k_std) < 1e-12} (수학적 불변 확인, 1000은 조건수용)")

    # 공선성 확인
    Tk = X_tr0[:, 0] + CELSIUS_TO_K
    corr = float(np.corrcoef(X_tr0[:, 0], INV_T_SCALE / Tk)[0, 1])
    print(f"    학습 corr(T, 1000/T_K) = {corr:.6f} (≈ -1 → 거의 완전 공선)")

    results = []
    # 365°C 외삽점 (10/12 조건, I1은 압력 없음)
    x365_raw = np.array([[EXTRAP_T, EXTRAP_FT, EXTRAP_FRC]], dtype=np.float64)
    print(f"\n    {'변환':10}{'RMSE':>10}{'MAE':>10}{'MAPE':>8}{'쿠멘%':>8}{'365예측':>9}{'전체편향kg/h':>13}")
    for kind in ("none", "invT", "invF", "invTF"):
        X_tr = _apply_transform(X_tr0, kind)
        X_va = _apply_transform(X_va0, kind)
        y_pred, _ = _logit_regress(X_tr, y_tr, None, X_va)
        conv = mt.conversion_metrics(y_va, y_pred)
        cum = mt.cumene_metrics(y_va, y_pred, npr_valid)
        p365 = float(_logit_regress(X_tr, y_tr, None, _apply_transform(x365_raw, kind))[0][0])
        bias = float(np.mean((y_pred - y_va) * npr_valid * CUMENE_MW))

        result = itf.make_result(
            model=f"linear3_logit_{kind}", feature_set="I1", label="r0", weighted=False,
            hyperparams={"method": "logit", "input_transform": kind,
                         "note_1000_scale": "1000/T_K 스케일은 결과 무관(스케일 불변 실증)",
                         "logit_eps": LOGIT_EPS},
            seed=seed, timestamp=datetime.now(timezone.utc).isoformat(),
            n_train=len(train), n_valid=len(valid),
            metrics={"conversion": conv, "cumene_kgh": cum,
                     "extrapolation": {"T_C": EXTRAP_T, "final": p365, "raw_exceeds_1": bool(p365 > 1.0)},
                     "overall_bias_kgh": bias})
        mt.save_result(result, cfg["paths"]["results_dir"])
        results.append(result)
        print(f"    {kind:10}{conv['rmse']:>10.6f}{conv['mae']:>10.6f}{conv['mape']:>8.4f}"
              f"{cum['pct_of_mean']:>7.4f}%{p365:>9.4f}{bias:>13.2f}")
    return results


def main() -> None:
    # Phase 10-C: 모델 탐색 증거 (results/linear3_*.json)
    run_error_structure()
    run_transform_trials()
    print("\n저장 완료: results/ 에 linear3_*.json (오차구조 1 + 입력변환 4)")
    # 이전 단계: run_all()=Phase 5-B(linear_*), run_linear2_grid()=Phase 10-A(linear2_*)


if __name__ == "__main__":
    main()
