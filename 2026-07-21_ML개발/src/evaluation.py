"""evaluation.py — Stage 0 평가 프로토콜 (모델 비교 공용 모듈, 신설 2026-08-01).

ONNX1(반응기 전환율 예측) 모델족 비교를 위해 모든 단계가 공유하는 평가 규약을 한곳에 모은다.
기존 공용 모듈(data_loader / interface / metrics)을 import 만 하고 수정하지 않는다(CLAUDE.md §3).

핵심 규약(지시서 Stage 0 + 2026-08-01 수정):
  · 주 분할: 학습 10/13~16(1924행) → 검증 10/17-18(661행). 고정. 무작위 분할 금지(CLAUDE.md §1-3).
  · 타깃은 로짓 z=logit(X_r1). 모든 모델을 z 에 적합하고 예측을 sigmoid 로 되돌려 전환율 스케일로 평가.
    (배포 관례 export_onnx.py 와 동일: 회귀기는 z 를 내고 ONNX 에 Sigmoid 노드를 붙여 전환율 출력.)
  · 10/12 는 AT-1100·TT-1100·PT-1100 표준편차 0(정상상태 반복 기록)이라 학습·검증·비교 전부 제외.
  · 검증 캠페인(10/17-18)은 어떤 fit·튜닝·모델선택에도 쓰지 않는다(CLAUDE.md §1-2).

[수정 1] 하이퍼파라미터 튜닝·승자 선택 기준 = **블록20%** (LOCO-4 폐기).
  각 학습 캠페인(10/13,14,15,16)의 앞 80%(시간순) → 학습, 뒤 20% → 내부검증. 풀링해 1개 RMSE.
  근거: LOCO-4 는 "학습에서 안 본 축으로 외삽"을 요구해 홀드아웃 순위와 Spearman −0.714(정반대),
  블록20% 는 +0.810. `score_block20`. 블록20% 절대값은 홀드아웃의 0.13~0.42배로 낙관적 →
  **순위 매기기 전용**, 성능 수치로 보고 금지. 승자 선택은 블록20% 로만(검증셋은 확인용, 추가 1).
  · LOCO-4 는 폐기하지 않고 "왜 바꿨는지" 대조 진단으로만 계산(`score_loco4`).
  · LOCO-5 강건성(`loco5_robustness`): 5개 캠페인 각각 홀드아웃. 단 10/17-18 에는 fit 하지 않는다 —
    10/17-18 홀드아웃 = 주 분할 모델(10/13-16 학습), 나머지 4개는 다른 학습 캠페인으로만 학습.
    이렇게 §1-2 를 지키며 5-캠페인 강건성을 얻는다. 튜닝엔 쓰지 않는다.

[수정 2] 불확실성 주 방법 = **블록 잭나이프(delete-one-block, 결정적)**, 부트스트랩은 교차확인.
  검증 661행을 60분 블록으로 나눠 하나씩 빼고 pseudo-value 로 SE·95%CI. 블록길이 60·120분 둘 다.
  `block_jackknife_ci`. 블록 부트스트랩(60분,2000,seed42)은 교차확인용. 모델쌍 결론이 블록길이
  1~180분에서 동일한지 확인(`BLOCK_LENS`).

금지(CLAUDE.md 절대규칙): ML_weight, TT-1006_rate_C_per_min, τ 는 어디에도 쓰지 않는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import data_loader as dl

# --- 확정 상수 (출처 주석 필수 — CLAUDE.md §4.2) ---
LOGIT_EPS = 1e-6            # 로짓 클립 경계. 지시서 Phase 10-A/11, export_onnx.py 와 동일.
CUMENE_MW = 120.196        # 큐멘 몰질량 kg/kmol. config properties.MW_kg_per_kmol.cumene(AVEVA).
BLOCK_LEN_MIN = 60         # 주 블록 길이(분). 잭나이프·부트스트랩 공용. 잔차 ACF 로 근거(지시서 0-3).
BLOCK_LEN_ALT = 120        # 보조 블록 길이(분). ACF 가 0 에 수렴하는 지점(수정 2).
BLOCK_LENS = (1, 15, 30, 60, 120, 180)  # 블록길이 민감도·모델쌍 결론 안정성 확인(수정 2).
N_BOOT = 2000              # 부트스트랩 반복(지시서 0-3).
SEED = 42                  # config/base.yaml seed. 전 구간 고정.
BLOCK20_FRAC = 0.8         # 블록20% 튜닝: 캠페인당 앞 80%(floor) 학습 / 뒤 20% 내부검증(수정 1).

# 물리 격자 (지시서 0-5) — 케이스 스터디 축. 주어진 값(지시서), 임의 아님.
GRID_T = (335.0, 350.0, 365.0)                 # 반응기 입구온도 °C (TT-1006.PV)
GRID_TOTAL = (15000.0, 16500.0, 18000.0)       # 총 질량유량 kg/h = FT-1004*(1+FRC-1004)
GRID_BP = (4.0, 5.0, 6.0)                       # B/P 질량비 (FRC-1004.PV)
T_MAX_TRAIN = 349.977      # 학습셋 TT-1006 최댓값(CLAUDE.md §5). 외삽 판정 기준선.

_TRAIN_CAMPAIGNS = ("10/13", "10/14", "10/15", "10/16")  # 학습 캠페인(LOCO-4 폴드 단위)


# =============================================================================
# 로짓 변환 / 지표
# =============================================================================
def to_logit(y: np.ndarray) -> np.ndarray:
    """전환율 y(0~1) → 로짓 z. y 는 [eps,1-eps] 로 클립(export_onnx.py 와 동일)."""
    yc = np.clip(np.asarray(y, dtype=np.float64), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(yc / (1.0 - yc))


def from_logit(z: np.ndarray) -> np.ndarray:
    """로짓 z → 전환율(sigmoid). scipy expit 없이 수치 안정 구현."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# =============================================================================
# 타깃 변환 (Stage 2) — 모델을 z=fwd(y) 에 적합, 예측을 inv(z) 로 전환율 복원.
#   ONNX 는 회귀기(z 출력)에 역변환 노드를 append 해 전환율 직접 출력. kind 로 노드 선택.
# =============================================================================
@dataclass
class Transform:
    name: str
    kind: str                     # raw/logit/log/log1m/arcsin — ONNX 역변환 노드 선택 키
    _fwd: "Callable"
    _inv: "Callable"
    onnxable: bool = True         # 역변환 노드를 표준 ONNX 연산으로 붙일 수 있는가

    def fwd(self, y):
        return self._fwd(np.asarray(y, dtype=np.float64))

    def inv(self, z):
        return self._inv(np.asarray(z, dtype=np.float64))


def _build_transforms() -> dict[str, "Transform"]:
    eps = LOGIT_EPS
    return {
        # raw: 변환 없음. 예측이 [0,1] 밖으로 나갈 수 있음(물리검사가 셈).
        "raw":    Transform("raw", "raw", lambda y: y, lambda z: z),
        # logit: 현재 채택안. 구조적으로 (0,1).
        "logit":  Transform("logit", "logit", to_logit, from_logit),
        # log: z=log(y). 예측 exp(z) 는 상한 없음(>1 가능).
        "log":    Transform("log", "log",
                            lambda y: np.log(np.clip(y, eps, None)),
                            lambda z: np.exp(z)),
        # log(1-y): 예측 1-exp(z) 는 하한 없음(<0 가능).
        "log1m":  Transform("log(1-y)", "log1m",
                            lambda y: np.log(np.clip(1.0 - y, eps, None)),
                            lambda z: 1.0 - np.exp(z)),
        # arcsin√y: 예측 sin(z)^2 는 구조적으로 [0,1].
        "arcsin": Transform("arcsin√y", "arcsin",
                            lambda y: np.arcsin(np.sqrt(np.clip(y, 0.0, 1.0))),
                            lambda z: np.sin(z) ** 2),
    }


TRANSFORMS = _build_transforms()


def conversion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """전환율 스케일 지표: rmse, mae, mape(%), r2, max_abs_err."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    nz = y_true != 0.0
    mape = float(np.mean(np.abs(err[nz] / y_true[nz])) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    max_abs = float(np.max(np.abs(err)))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "max_abs_err": max_abs}


def cumene_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_propene: np.ndarray
) -> dict[str, float]:
    """전환율 → 큐멘 생산량[kg/h] 환산 후 RMSE 와 평균 대비 %.

    cumene_kg_h = X × n_propene(kmol/h) × MW_cumene(120.196). (지시서 0-2)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = np.asarray(n_propene, dtype=np.float64)
    ct = y_true * n * CUMENE_MW
    cp = y_pred * n * CUMENE_MW
    err = cp - ct
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mean_t = float(np.mean(ct))
    pct = float(rmse / mean_t * 100.0) if mean_t != 0 else float("nan")
    return {"rmse_kgh": rmse, "pct_of_mean": pct}


# =============================================================================
# 데이터 / 분할
# =============================================================================
@dataclass
class Splits:
    """평가에 필요한 배열 묶음. 전부 float64, 시간순 정렬 유지."""
    X_train: np.ndarray
    y_train: np.ndarray            # 전환율(원 스케일)
    camp_train: np.ndarray         # 학습 각 행의 캠페인 문자열
    X_valid: np.ndarray
    y_valid: np.ndarray
    npr_valid: np.ndarray          # 검증 n_propene_kmol_h (큐멘 환산)
    feature_names: list[str]
    df: Any = field(repr=False, default=None)


def load_splits(feature_set: str = "I1", label: str = "r1") -> Splits:
    """정본 로딩 후 주 분할 배열 생성. 기본 I1·r1(최종 레시피 입력/라벨)."""
    cfg = dl.load_config()
    df = dl.load_data(cfg)
    train = dl.get_split(df, cfg["split"]["train_value"])   # 10/13~16
    valid = dl.get_split(df, cfg["split"]["valid_value"])   # 10/17-18
    Xtr, ytr = dl.get_Xy(train, feature_set, label)
    Xva, yva = dl.get_Xy(valid, feature_set, label)
    import interface as itf
    return Splits(
        X_train=Xtr, y_train=ytr, camp_train=train["Campaign"].to_numpy(),
        X_valid=Xva, y_valid=yva, npr_valid=dl.get_n_propene(valid),
        feature_names=itf.feature_columns(cfg, feature_set), df=df,
    )


# =============================================================================
# 모델 스펙 / 적합-예측
# =============================================================================
Factory = Callable[[dict], Any]   # params -> unfitted sklearn estimator(입력 X → 로짓 z)


@dataclass
class ModelSpec:
    name: str
    family: str
    factory: Factory              # params dict → 미적합 estimator
    param_grid: list[dict]        # 튜닝 후보(빈 dict 1개면 튜닝 없음)
    onnx_capable: bool = True     # skl2onnx 변환 시도 여부(불가 알려진 모델은 False)
    transform: Transform = None   # 타깃 변환(기본 logit). Stage 2 에서 교체.

    def __post_init__(self):
        if self.transform is None:
            self.transform = TRANSFORMS["logit"]


def fit_on_logit(spec: ModelSpec, params: dict, X: np.ndarray, y_conv: np.ndarray):
    """estimator 를 spec.transform 타깃에 적합해 반환. y_conv 는 전환율(원 스케일).

    적합된 est 에 변환을 부착(_eval_transform)해 predict_conv 가 되돌릴 수 있게 한다.
    """
    est = spec.factory(params)
    est.fit(X, spec.transform.fwd(y_conv))
    est._eval_transform = spec.transform
    return est


def predict_conv(est, X: np.ndarray) -> np.ndarray:
    """estimator 예측을 전환율로 되돌린다(부착된 변환 사용). (PLS 등 (n,1) 출력은 ravel)"""
    tr = getattr(est, "_eval_transform", None) or TRANSFORMS["logit"]
    return tr.inv(np.asarray(est.predict(X)).ravel())


# =============================================================================
# 튜닝 기준 — 블록20%(주) / LOCO-4(대조 진단) / LOCO-5(강건성). 전부 10/17-18 미접촉.
# =============================================================================
def _block20_masks(sp: Splits) -> tuple[np.ndarray, np.ndarray]:
    """캠페인당 앞 80%(floor)→학습, 뒤 20%→내부검증. 풀링한 (train_idx, valid_idx).

    행 순서가 캠페인 내 시간순이라는 전제(정본 엑셀은 1분 시계열 정렬). 무작위 아님.
    """
    idx = np.arange(len(sp.X_train))
    tr, va = [], []
    for camp in _TRAIN_CAMPAIGNS:
        ci = idx[sp.camp_train == camp]
        cut = int(BLOCK20_FRAC * len(ci))   # floor. 예: 481→384 학습 / 97 내부검증
        tr.extend(ci[:cut]); va.extend(ci[cut:])
    return np.array(tr), np.array(va)


def score_block20(spec: ModelSpec, params: dict, sp: Splits) -> float:
    """블록20% 내부검증 RMSE(전환율). 튜닝·승자선택 기준. 순위용 값(성능 아님)."""
    tr, va = _block20_masks(sp)
    est = fit_on_logit(spec, params, sp.X_train[tr], sp.y_train[tr])
    return conversion_metrics(sp.y_train[va], predict_conv(est, sp.X_train[va]))["rmse"]


def score_loco4(spec: ModelSpec, params: dict, sp: Splits) -> dict[str, Any]:
    """LOCO-4(학습 4캠페인 홀드아웃, 각 폴드 3캠페인 학습) — 대조 진단용(튜닝 아님)."""
    folds: dict[str, float] = {}
    for camp in _TRAIN_CAMPAIGNS:
        tr = sp.camp_train != camp; va = ~tr
        est = fit_on_logit(spec, params, sp.X_train[tr], sp.y_train[tr])
        folds[camp] = conversion_metrics(sp.y_train[va], predict_conv(est, sp.X_train[va]))["rmse"]
    vals = list(folds.values())
    return {"folds": folds, "mean": float(np.mean(vals)), "worst": float(np.max(vals))}


def loco5_robustness(spec: ModelSpec, params: dict, sp: Splits) -> dict[str, Any]:
    """LOCO-5 강건성. 5개 캠페인 각각 홀드아웃. 단 10/17-18 에는 fit 하지 않는다:
    10/17-18 폴드 = 주 분할 모델(10/13-16 학습), 나머지 4개는 다른 학습 캠페인으로만 학습.
    §1-2(검증에 fit 금지)를 지키며 5-캠페인 강건성을 얻는다. 튜닝엔 쓰지 않는다.
    """
    folds: dict[str, float] = {}
    for camp in _TRAIN_CAMPAIGNS:
        tr = sp.camp_train != camp; va = ~tr
        est = fit_on_logit(spec, params, sp.X_train[tr], sp.y_train[tr])
        folds[camp] = conversion_metrics(sp.y_train[va], predict_conv(est, sp.X_train[va]))["rmse"]
    # 10/17-18 홀드아웃 = 학습 4캠페인으로 학습한 주 분할 모델의 검증 RMSE (fit 은 학습셋만)
    est_full = fit_on_logit(spec, params, sp.X_train, sp.y_train)
    folds["10/17-18"] = conversion_metrics(sp.y_valid, predict_conv(est_full, sp.X_valid))["rmse"]
    vals = list(folds.values())
    return {"folds": folds, "mean": float(np.mean(vals)), "worst": float(np.max(vals))}


def tune(spec: ModelSpec, sp: Splits, criterion: str = "block20") -> tuple[dict, float]:
    """param_grid 를 criterion('block20'|'loco4') 로 평가해 최적 params 선택.

    반환 (best_params, best_score). 동률이면 grid 순서상 먼저. 승자선택은 항상 block20.
    """
    best = None
    for params in spec.param_grid:
        if criterion == "block20":
            score = score_block20(spec, params, sp)
        elif criterion == "loco4":
            score = score_loco4(spec, params, sp)["mean"]
        else:
            raise ValueError(f"알 수 없는 criterion: {criterion!r}")
        if best is None or score < best[1] - 1e-12:
            best = (params, score)
    return best


# =============================================================================
# 블록 부트스트랩 (검증 RMSE 95% CI + 모델쌍 ΔRMSE CI)
# =============================================================================
def block_bootstrap_indices(n: int, block_len: int, n_boot: int, seed: int) -> np.ndarray:
    """이동블록 부트스트랩 인덱스 (n_boot, n). 연속 block_len 행 블록을 복원추출·연결.

    모델쌍 ΔRMSE 를 대응(paired)으로 재기 위해 인덱스 집합을 공유한다.
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    max_start = n - block_len  # 시작점 0..max_start
    idx = np.empty((n_boot, n_blocks * block_len), dtype=np.int64)
    offs = np.arange(block_len)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx[b] = (starts[:, None] + offs[None, :]).ravel()
    return idx[:, :n]


def bootstrap_rmse_ci(
    y_true: np.ndarray, y_pred: np.ndarray, boot_idx: np.ndarray
) -> dict[str, float]:
    """부트스트랩 검증 RMSE 의 점추정과 95% 백분위 CI."""
    err2 = (np.asarray(y_pred) - np.asarray(y_true)) ** 2
    rmses = np.sqrt(err2[boot_idx].mean(axis=1))
    return {
        "rmse": float(np.sqrt(err2.mean())),
        "ci_low": float(np.percentile(rmses, 2.5)),
        "ci_high": float(np.percentile(rmses, 97.5)),
    }


def bootstrap_delta_ci(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, boot_idx: np.ndarray
) -> dict[str, float]:
    """ΔRMSE = RMSE(a) - RMSE(b) 의 대응 부트스트랩 95% CI. 0 포함 시 '차이 없음'."""
    ea = (np.asarray(pred_a) - np.asarray(y_true)) ** 2
    eb = (np.asarray(pred_b) - np.asarray(y_true)) ** 2
    ra = np.sqrt(ea[boot_idx].mean(axis=1))
    rb = np.sqrt(eb[boot_idx].mean(axis=1))
    d = ra - rb
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {
        "delta": float(np.sqrt(ea.mean()) - np.sqrt(eb.mean())),
        "ci_low": lo, "ci_high": hi,
        "indistinguishable": bool(lo <= 0.0 <= hi),
    }


# =============================================================================
# 블록 잭나이프 (delete-one-block) — 주 방법(결정적). 부트스트랩은 위 교차확인용.
# =============================================================================
def _contig_blocks(n: int, block_len: int) -> list[np.ndarray]:
    """검증 시계열을 연속 block_len 행 블록으로 분할(마지막 블록은 짧을 수 있음)."""
    return [np.arange(i, min(i + block_len, n)) for i in range(0, n, block_len)]


def block_jackknife_ci(
    y_true: np.ndarray, y_pred: np.ndarray, block_len: int
) -> dict[str, float]:
    """블록 잭나이프 RMSE 95%CI. pseudo_k = nb*RMSE_full - (nb-1)*RMSE_(-k),
    SE = std(pseudo)/sqrt(nb), CI = mean(pseudo) ± 1.96*SE. 난수 없음(결정적).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    e2 = (np.asarray(y_pred, dtype=np.float64) - y_true) ** 2
    n = len(e2)
    bl = _contig_blocks(n, block_len)
    nb = len(bl)
    full = float(np.sqrt(e2.mean()))
    ps = []
    for b in bl:
        mask = np.ones(n, dtype=bool); mask[b] = False
        ps.append(nb * full - (nb - 1) * float(np.sqrt(e2[mask].mean())))
    ps = np.asarray(ps)
    se = float(ps.std(ddof=1) / np.sqrt(nb))
    m = float(ps.mean())
    return {"rmse": full, "n_block": nb, "mean_pseudo": m, "se": se,
            "ci_low": m - 1.96 * se, "ci_high": m + 1.96 * se,
            "width": 2 * 1.96 * se}


def block_jackknife_delta_ci(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, block_len: int
) -> dict[str, float]:
    """ΔRMSE = RMSE(a)-RMSE(b) 의 블록 잭나이프 95%CI. 0 포함 시 '차이 없음'."""
    y_true = np.asarray(y_true, dtype=np.float64)
    ea = (np.asarray(pred_a, dtype=np.float64) - y_true) ** 2
    eb = (np.asarray(pred_b, dtype=np.float64) - y_true) ** 2
    n = len(ea)
    bl = _contig_blocks(n, block_len)
    nb = len(bl)
    full = float(np.sqrt(ea.mean()) - np.sqrt(eb.mean()))
    ps = []
    for b in bl:
        mask = np.ones(n, dtype=bool); mask[b] = False
        d = float(np.sqrt(ea[mask].mean()) - np.sqrt(eb[mask].mean()))
        ps.append(nb * full - (nb - 1) * d)
    ps = np.asarray(ps)
    se = float(ps.std(ddof=1) / np.sqrt(nb))
    m = float(ps.mean())
    lo, hi = m - 1.96 * se, m + 1.96 * se
    return {"delta": full, "ci_low": lo, "ci_high": hi,
            "indistinguishable": bool(lo <= 0.0 <= hi)}


# =============================================================================
# 잔차 자기상관 (블록 길이 근거)
# =============================================================================
def residual_acf(resid: np.ndarray, lags=(1, 5, 15, 30, 60)) -> dict[int, float]:
    """검증 잔차 시계열의 지연별 자기상관(표본 ACF)."""
    r = np.asarray(resid, dtype=np.float64)
    r = r - r.mean()
    denom = float(np.sum(r * r))
    out = {}
    for k in lags:
        if k < len(r) and denom > 0:
            out[k] = float(np.sum(r[k:] * r[:-k]) / denom)
        else:
            out[k] = float("nan")
    return out


# =============================================================================
# 캠페인 민감도 (학습 캠페인 1개 제외 재학습 → 변화)
# =============================================================================
def campaign_sensitivity(spec: ModelSpec, params: dict, sp: Splits) -> dict[str, Any]:
    """학습 캠페인 하나씩 제외하고 재학습 → 검증 예측 변화와 검증 RMSE 변화.

    선형 계열이면 계수 변화도 함께 담는다(coef_ 존재 시).
    반환: {full_rmse, per_campaign:{camp:{valid_rmse, pred_shift_rms, coef?}}, max_pred_shift_rms}
    """
    full_est = fit_on_logit(spec, params, sp.X_train, sp.y_train)
    full_pred = predict_conv(full_est, sp.X_valid)
    full_rmse = conversion_metrics(sp.y_valid, full_pred)["rmse"]
    full_coef = getattr(full_est, "coef_", None)

    per = {}
    shifts = []
    for camp in _TRAIN_CAMPAIGNS:
        tr = sp.camp_train != camp
        est = fit_on_logit(spec, params, sp.X_train[tr], sp.y_train[tr])
        pred = predict_conv(est, sp.X_valid)
        shift = float(np.sqrt(np.mean((pred - full_pred) ** 2)))
        shifts.append(shift)
        entry = {
            "valid_rmse": conversion_metrics(sp.y_valid, pred)["rmse"],
            "pred_shift_rms": shift,
        }
        coef = getattr(est, "coef_", None)
        if coef is not None and full_coef is not None and np.ndim(coef) == 1:
            entry["coef"] = [float(c) for c in coef]
        per[camp] = entry

    result = {
        "full_valid_rmse": full_rmse,
        "per_campaign": per,
        "max_pred_shift_rms": float(np.max(shifts)),
    }
    if full_coef is not None and np.ndim(full_coef) == 1:
        result["full_coef"] = [float(c) for c in full_coef]
    return result


# =============================================================================
# 물리 타당성 (지시서 0-5) — 단조성 기준은 실측에서 유도한 방향을 쓴다.
#   실측 재확인: T↑→X↑, 총유량↑→X↑ (corr 총유량·PT-1004 = +0.9988). "총유량↑→X↓"는 성립 안 함.
# =============================================================================
def _grid_rows(feature_names: list[str]) -> tuple[np.ndarray, list[dict]]:
    """27점 격자를 feature_names 순서의 입력행렬로. I1(3열) 전제.

    각 격자점: TT-1006=T, FRC-1004=B/P, FT-1004 = total/(1+B/P).
    I2/I3(압력 포함)는 케이스 스터디가 압력을 주지 않으므로 여기선 I1 만 지원.
    """
    if feature_names[:3] != ["TT-1006.PV", "FT-1004.PV", "FRC-1004.PV"] or len(feature_names) != 3:
        raise ValueError(f"물리격자는 I1 전용. feature_names={feature_names}")
    rows, meta = [], []
    for total in GRID_TOTAL:
        for bp in GRID_BP:
            ft = total / (1.0 + bp)      # 프로펜 유량 = 총유량/(1+질량비)
            for T in GRID_T:
                rows.append([T, ft, bp])
                meta.append({"T": T, "total": total, "bp": bp, "ft": ft})
    return np.array(rows, dtype=np.float64), meta


def physical_check(spec: ModelSpec, params: dict, sp: Splits) -> dict[str, Any]:
    """27점 격자 예측으로 물리 합격/불합격 판정.

    반환: pred_in_range(∈[0,1] 개수/27), T_monotone_slices(x/9), T_monotone_pairs(x/18),
          extrap_ok(365°C 예측이 유계·발산 없음), preds(격자 X).
    """
    Xg, meta = _grid_rows(sp.feature_names)
    est = fit_on_logit(spec, params, sp.X_train, sp.y_train)
    preds = predict_conv(est, Xg)

    in_range = int(np.sum((preds >= 0.0) & (preds <= 1.0)))

    # T 단조성: (total,bp) 슬라이스마다 T=335<350<365 순 X 증가 확인
    slices_ok, pairs_ok, pairs_tot = 0, 0, 0
    by_slice: dict[tuple, list] = {}
    for m, p in zip(meta, preds):
        by_slice.setdefault((m["total"], m["bp"]), []).append((m["T"], p))
    for key, seq in by_slice.items():
        seq.sort(key=lambda t: t[0])
        xs = [v for _, v in seq]
        if all(xs[i + 1] > xs[i] for i in range(len(xs) - 1)):
            slices_ok += 1
        for i in range(len(xs) - 1):
            pairs_tot += 1
            if xs[i + 1] > xs[i]:
                pairs_ok += 1

    # 외삽(365°C > 학습 최대 349.977): 유계(0~1)이고 350→365 증가폭이 폭주하지 않는지
    extrap = [(m, p) for m, p in zip(meta, preds) if m["T"] == 365.0]
    extrap_vals = np.array([p for _, p in extrap])
    # 350→365 증가폭을 335→350 증가폭과 비교(발산이면 후자가 급증)
    diverges = False
    for key, seq in by_slice.items():
        seq.sort(key=lambda t: t[0])
        xs = [v for _, v in seq]
        if len(xs) == 3:
            d1, d2 = xs[1] - xs[0], xs[2] - xs[1]
            if d2 > 5.0 * max(d1, 1e-9) and d2 > 0.05:  # 외삽 구간 기울기 폭주
                diverges = True
    extrap_ok = bool(np.all((extrap_vals >= 0.0) & (extrap_vals <= 1.0)) and not diverges)

    return {
        "pred_in_range": in_range,
        "n_grid": len(preds),
        "T_monotone_slices": slices_ok,
        "n_slices": len(by_slice),
        "T_monotone_pairs": pairs_ok,
        "n_pairs": pairs_tot,
        "extrap_ok": extrap_ok,
        "extrap_diverges": diverges,
        "preds": [float(p) for p in preds],
        "meta": meta,
    }


# =============================================================================
# ONNX 변환 + 왕복 검증
# =============================================================================
def _append_inverse_nodes(g, helper, TensorProto, z_out: str, kind: str) -> str:
    """회귀기 출력 z_out 에 변환별 역변환 노드를 붙이고 최종(전환율) 텐서 이름 반환.

    표준 ONNX 연산만 사용. raw=노드없음, logit=Sigmoid, log=Exp, log1m=1-Exp, arcsin=Sin².
    붙일 수 없는 kind 면 NotImplementedError.
    """
    if kind == "raw":
        return z_out
    if kind == "logit":
        g.node.append(helper.make_node("Sigmoid", [z_out], ["conversion"], name="inv_logit"))
        return "conversion"
    if kind == "log":
        g.node.append(helper.make_node("Exp", [z_out], ["conversion"], name="inv_log"))
        return "conversion"
    if kind == "log1m":
        one = helper.make_node("Constant", [], ["one_c"], name="const_one",
                               value=helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]))
        g.node.append(one)
        g.node.append(helper.make_node("Exp", [z_out], ["ez"], name="inv_log1m_exp"))
        g.node.append(helper.make_node("Sub", ["one_c", "ez"], ["conversion"], name="inv_log1m_sub"))
        return "conversion"
    if kind == "arcsin":
        g.node.append(helper.make_node("Sin", [z_out], ["s"], name="inv_arcsin_sin"))
        g.node.append(helper.make_node("Mul", ["s", "s"], ["conversion"], name="inv_arcsin_sq"))
        return "conversion"
    raise NotImplementedError(f"역변환 ONNX 노드 미정의: {kind}")


def onnx_roundtrip(est, sp: Splits) -> dict[str, Any]:
    """estimator(z 출력)를 skl2onnx 로 변환하고 변환별 역변환 노드를 붙여 전환율 ONNX 생성.

    검증셋에서 py 역변환 예측과 ONNX 출력을 비교해 최대 절대차 보고. 실패 시 사유 기록.
    """
    tr = getattr(est, "_eval_transform", None) or TRANSFORMS["logit"]
    try:
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType

        n_feat = sp.X_valid.shape[1]
        onx = to_onnx(est, initial_types=[("X", FloatTensorType([None, n_feat]))],
                      target_opset=None)
        g = onx.graph
        z_out = g.output[0].name
        final = _append_inverse_nodes(g, helper, TensorProto, z_out, tr.kind)
        if final != z_out:
            del g.output[:]
            g.output.append(helper.make_tensor_value_info(final, TensorProto.FLOAT, [None, 1]))
        onnx.checker.check_model(onx)

        sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        ox = np.asarray(sess.run(None, {iname: sp.X_valid.astype(np.float32)})[0]).ravel()
        py = predict_conv(est, sp.X_valid)
        return {"ok": True, "max_abs_diff": float(np.max(np.abs(py - ox))), "error": None}
    except Exception as e:  # noqa: BLE001 — 변환 가능성 진단이 목적, 사유만 기록
        return {"ok": False, "max_abs_diff": None, "error": f"{type(e).__name__}: {e}"}


# =============================================================================
# 파라미터 개수 (계열별 의미가 달라 kind 로 표기)
# =============================================================================
def count_params(est, n_train: int, n_feat: int) -> dict[str, Any]:
    """모델 복잡도 지표. 계열마다 의미가 달라 (value, kind) 로 표기한다."""
    m = type(est).__name__
    # 파이프라인이면 마지막 단계로
    if hasattr(est, "steps"):
        final = est.steps[-1][1]
        m = type(final).__name__
    else:
        final = est

    coef = getattr(final, "coef_", None)
    if coef is not None:
        val = int(np.asarray(coef).size) + int(np.asarray(getattr(final, "intercept_", 0.0)).size)
        return {"value": val, "kind": "계수+절편"}
    # 트리 단일
    if hasattr(final, "tree_"):
        return {"value": int(final.tree_.node_count), "kind": "트리 노드수"}
    # 앙상블(포레스트/부스팅)
    if hasattr(final, "estimators_"):
        try:
            nodes = sum(int(e.tree_.node_count) for e in np.asarray(final.estimators_).ravel())
            return {"value": nodes, "kind": "총 트리 노드수"}
        except Exception:
            return {"value": len(final.estimators_), "kind": "약학습기 수"}
    # HistGBR
    if hasattr(final, "_predictors"):
        n_leaf = sum(p[0].get_n_leaf_nodes() for p in final._predictors)
        return {"value": int(n_leaf), "kind": "총 리프수"}
    # SVR
    if hasattr(final, "support_"):
        return {"value": int(final.support_.size), "kind": "서포트벡터 수"}
    # MLP
    if hasattr(final, "coefs_"):
        val = sum(c.size for c in final.coefs_) + sum(b.size for b in final.intercepts_)
        return {"value": int(val), "kind": "가중치+편향"}
    # GP / KNN → 학습표본 저장형
    if m in ("GaussianProcessRegressor", "KNeighborsRegressor"):
        return {"value": int(n_train), "kind": "저장 학습표본 수"}
    return {"value": -1, "kind": "미정"}


# =============================================================================
# 통합 평가 — 한 모델을 스펙대로 평가해 결과 dict 반환
# =============================================================================
def evaluate_model(
    spec: ModelSpec, sp: Splits, boot_idx: np.ndarray, do_physical: bool = True
) -> dict[str, Any]:
    """블록20% 튜닝(승자선택) → 주분할 검증 → 잭나이프(주)·부트스트랩(교차) CI →
    LOCO-5 강건성 → LOCO-4 대조진단 → 캠페인민감도 → 물리 → ONNX → 시간.

    검증셋 수치는 '확인용'(추가 1). 승자선택 기준은 block20 뿐. 검증 y_pred 도 담아 쌍비교 재사용.
    """
    n_feat = sp.X_valid.shape[1]

    # 1) 튜닝 — 블록20%(주, 승자선택)와 LOCO-4(대조 진단) 각각 최적 params
    t0 = time.perf_counter()
    p_b20, b20 = tune(spec, sp, "block20")
    tune_time = time.perf_counter() - t0
    l4 = tune(spec, sp, "loco4")               # 대조: LOCO-4 로 뽑으면 다른 params/순위
    p_loco, loco4_score = l4[0], l4[1]

    # 선택 모델 = 블록20% params. 주분할 학습/검증 + 시간
    best_params = p_b20
    t0 = time.perf_counter()
    est = fit_on_logit(spec, best_params, sp.X_train, sp.y_train)
    fit_time = time.perf_counter() - t0

    train_rmse = conversion_metrics(sp.y_train, predict_conv(est, sp.X_train))["rmse"]

    t0 = time.perf_counter()
    y_pred = predict_conv(est, sp.X_valid)
    infer_time = (time.perf_counter() - t0) / len(sp.X_valid)

    conv = conversion_metrics(sp.y_valid, y_pred)   # 검증(확인용)
    cum = cumene_metrics(sp.y_valid, y_pred, sp.npr_valid)

    # LOCO-4 로 뽑은 params 의 검증 RMSE(전/후 순위 대조용, 확인만)
    if p_loco == best_params:
        holdout_loco = conv["rmse"]
    else:
        est_l = fit_on_logit(spec, p_loco, sp.X_train, sp.y_train)
        holdout_loco = conversion_metrics(sp.y_valid, predict_conv(est_l, sp.X_valid))["rmse"]

    # 2) 불확실성 — 잭나이프(주) 60·120분 + 부트스트랩(교차) 60분
    jack60 = block_jackknife_ci(sp.y_valid, y_pred, BLOCK_LEN_MIN)
    jack120 = block_jackknife_ci(sp.y_valid, y_pred, BLOCK_LEN_ALT)
    boot = bootstrap_rmse_ci(sp.y_valid, y_pred, boot_idx)

    # 3) LOCO-5 강건성 + 4) 캠페인 민감도
    loco5 = loco5_robustness(spec, best_params, sp)
    sens = campaign_sensitivity(spec, best_params, sp)

    # 5) 물리 6) ONNX 7) 복잡도
    phys = physical_check(spec, best_params, sp) if do_physical else None
    onnx_res = onnx_roundtrip(est, sp) if spec.onnx_capable else {
        "ok": False, "max_abs_diff": None, "error": "onnx_capable=False(사전 제외)"}
    params_info = count_params(est, len(sp.X_train), n_feat)

    return {
        "name": spec.name,
        "family": spec.family,
        "best_params": best_params,          # 블록20% 선택
        "loco4_params": p_loco,              # LOCO-4 선택(대조)
        "params_value": params_info["value"],
        "params_kind": params_info["kind"],
        "train_rmse": train_rmse,
        "block20": b20,                      # 순위용(성능 아님)
        "valid": conv,                       # 확인용
        "cumene": cum,
        "jack60": jack60,                    # 주 CI
        "jack120": jack120,
        "boot": boot,                        # 교차확인
        "loco4_score": loco4_score,          # 대조 진단
        "holdout_loco": holdout_loco,        # LOCO-4 선택 params 의 검증 RMSE
        "loco5_mean": loco5["mean"],         # 강건성
        "loco5_worst": loco5["worst"],
        "loco5_folds": loco5["folds"],
        "sensitivity": sens,
        "physical": phys,
        "onnx": onnx_res,
        "tune_time_s": tune_time,
        "fit_time_s": fit_time,
        "infer_time_us_per_row": infer_time * 1e6,
        "y_pred_valid": [float(v) for v in y_pred],
    }
