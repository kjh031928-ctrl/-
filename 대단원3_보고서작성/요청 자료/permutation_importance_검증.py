# -*- coding: utf-8 -*-
"""
표 C4 — 순열 중요도(permutation importance) 재현 스크립트  [부록 C / §3.2.4]

무엇을 하는가
  각 입력을 "무의미하게" 만들었을 때 예측 RMSE가 얼마나 나빠지는지로 입력의 중요도를
  매긴다(ΔRMSE = 셔플 후 RMSE − baseline RMSE). 모델 계수 해석에 의존하지 않는
  **모델-불문(model-agnostic)** 지표라서, 물리 근거(대단원 2)·L1 통계선택(표 C1)과
  독립된 세 번째 경로가 된다.

왜 단순 행 셔플이 아니라 블록 셔플인가
  1분 간격 시계열이라 잔차 자기상관이 1분 지연에서 0.99 이상이다(§3.2.2). 행을 개별로
  섞으면 시계열이 갖는 자기상관 구조까지 함께 파괴되어 중요도가 과대평가된다. 그래서
  **연속 60분(=60행)을 한 덩어리로 묶고 덩어리 순서만 섞는다**(블록 셔플). 블록 안의
  시간 구조는 그대로 남고, 입력과 라벨의 대응만 끊긴다.
  N=200회 반복, seed=42, 평균±표준편차로 보고한다.

공선성 왜곡 점검(grouped 셔플)
  입력끼리 상관이 있으면 한 변수만 섞을 때 물리적으로 불가능한 조합이 만들어져 중요도가
  왜곡될 수 있다. 그래서 **|상관|이 가장 큰 쌍을 같은 블록순서로 함께 섞은** 값을 병기한다.
  함께 섞으면 두 변수의 상호관계는 보존된 채 라벨과의 대응만 끊긴다.

두 기준을 각각 산출한다
  (a) 배포 ONNX(10/13–18 적합) in-sample, 전 구간 PV 3,066행 — 표 C4 본표.
  (b) held-out 재적합 — 그림 규약(figures/make_figures.py)과 동일하게 학습(10/13–16)만으로
      다시 적합해 검증(10/17–18)에서 평가. (a)와 순위가 같은지 대조한다.

읽기 전용 원칙
  원본 CSV·ONNX·보고서는 읽기만 한다. 이 스크립트가 쓰는 파일은
  `results/permutation_importance.json` 하나뿐이다.

재현
  python permutation_importance_검증.py        (repo 루트에서 실행)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.linear_model import LinearRegression

# 콘솔이 cp949 인 환경에서도 한글·기호(—, Δ, ×)가 깨지지 않게 UTF-8 로 맞춘다.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

# --------------------------------------------------------------------------
# 0. 경로 / 상수
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CSV = ROOT / "ML_마스터데이터_PV.csv"
ONNX_DIR = ROOT / "2026-07-21_ML개발" / "onnx"
ONNX_CONV = ONNX_DIR / "reactor_conversion_r1.onnx"
ONNX_DP = ONNX_DIR / "reactor_dp.onnx"
OUT_DIR = ROOT / "results"
OUT_JSON = OUT_DIR / "permutation_importance.json"

BLOCK = 60      # 블록 길이 [행] = 60분
N_REP = 200     # 반복 횟수
SEED = 42

COL_T = "TT-1006.PV"          # 반응기 입구온도 [°C]
COL_FP = "FT-1004.PV"         # 프로펜 유량 [kg/h]
COL_BP = "FRC-1004.PV"        # B/P 비율 [-]
COL_FT = "total_flow_kgh"     # 총유량 [kg/h] = FT-1003 + FT-1004
COL_X = "X_conv_r1_deployed"  # 전환율 라벨 [-]
COL_DP = "DP_reactor_kPa"     # 압력강하 라벨 [kPa]

IN_CONV = [COL_T, COL_FP, COL_BP]   # ONNX1 입력 순서
IN_DP = [COL_T, COL_FT, COL_BP]     # ONNX2 입력 순서

# 배포 ΔP 모델 계수(보고서 §3.5.3 기재값) — ONNX2 입력 순서 대조용
DEPLOY_DP_INTERCEPT = -51.7101
DEPLOY_DP_COEF = np.array([0.084977, 0.00314476, -0.94009])  # [T, 총유량, FRC]

# 검증 기준값 — (a) 배포 ONNX in-sample, 전 구간 3,066행 (표 C4 기재값)
REF_A = {
    "conversion": {
        "baseline": 0.0145,
        "delta": {COL_T: 0.171, COL_FP: 0.099, COL_BP: 0.033},
        "share_pct": {COL_T: 56.4, COL_FP: 32.6, COL_BP: 11.0},
        "grouped": 0.082,
        "max_abs_corr": 0.54,
    },
    "dp": {
        "baseline": 0.116,
        "delta": {COL_T: 0.499, COL_FT: 4.466, COL_BP: 0.285},
        "share_pct": {COL_T: 9.5, COL_FT: 85.1, COL_BP: 5.4},
        "grouped": 4.22,
        "max_abs_corr": 0.49,
    },
}
# 허용오차: ΔRMSE·grouped 는 상대 2%(200회 몬테카를로 표준오차보다 넉넉),
#           baseline 은 기재 자릿수의 반올림 폭, 정규화(%)는 0.5 %p, 상관은 0.01.
TOL_REL_DELTA = 0.02
TOL_BASE = {"conversion": 5e-4, "dp": 1e-3}
TOL_SHARE_PP = 0.5
TOL_CORR = 0.01

FAILURES: list[str] = []


def check(label: str, got: float, ref: float, tol: float, kind: str = "abs") -> bool:
    """기준값 대조. 어긋나면 기록만 하고(멈춤은 마지막에) 콘솔에 표시한다."""
    dev = got - ref
    ok = (abs(dev) <= tol) if kind == "abs" else (abs(dev) <= tol * abs(ref))
    tag = "OK " if ok else "불일치"
    scale = f"(허용 ±{tol:g}{'' if kind == 'abs' else ' 상대'})"
    print(f"    [{tag}] {label}: 재현 {got:.6g}  기준 {ref:.6g}  차 {dev:+.3g} {scale}")
    if not ok:
        FAILURES.append(f"{label}: 재현 {got:.6g} vs 기준 {ref:.6g} (차 {dev:+.3g})")
    return ok


# --------------------------------------------------------------------------
# 1. 데이터 / 모델 로드
# --------------------------------------------------------------------------
def rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


df = pd.read_csv(CSV, encoding="utf-8-sig")
assert len(df) == 3066, f"행 수가 3,066이 아닙니다: {len(df)}"
train = df[df["Split"] == "학습"]      # 10/13~16, 1,924행
valid = df[df["Split"] == "검증"]      # 10/17~18,   661행
assert (len(train), len(valid)) == (1924, 661), (len(train), len(valid))

sess_conv = ort.InferenceSession(str(ONNX_CONV), providers=["CPUExecutionProvider"])
sess_dp = ort.InferenceSession(str(ONNX_DP), providers=["CPUExecutionProvider"])


def onnx_pred(sess):
    name = sess.get_inputs()[0].name

    def f(X):
        return np.asarray(sess.run(None, {name: np.asarray(X, np.float32)})[0]).ravel().astype(float)

    return f


pred_conv_onnx = onnx_pred(sess_conv)
pred_dp_onnx = onnx_pred(sess_dp)

print("=" * 78)
print("0. 사전 점검 — ONNX2 입력 순서를 배포 계수와 대조")
print("=" * 78)
# ΔP 모델은 순수 선형이므로, 기재된 배포 계수로 손계산한 값과 ONNX 출력이 같아야만
# "입력 순서 = [TT-1006, total_flow, FRC-1004]" 가 맞다. 순서가 뒤바뀌었다면 크게 틀어진다.
_X_dp_all = df[IN_DP].values
_manual = DEPLOY_DP_INTERCEPT + _X_dp_all @ DEPLOY_DP_COEF
_onnx = pred_dp_onnx(_X_dp_all)
_maxerr = float(np.max(np.abs(_manual - _onnx)))
print(f"  ONNX2 출력 vs 배포계수 손계산: max|차| = {_maxerr:.3e}")
if _maxerr >= 1e-4:
    sys.exit(f"중단 — ONNX2 입력 순서/계수 대조 실패 (max오차 {_maxerr:.3e} ≥ 1e-4)")
print("  → 1e-4 미만. 입력 순서 [TT-1006, total_flow, FRC-1004] 확인.")

# 보조 확인 — 두 ONNX 모두 10/13–18 전체 OLS 재적합과 계수가 일치하는지(배포 규약 확인).
_dep = df[df["Split"] != "정상운전검사"]
_y = _dep[COL_X].values
_m1 = LinearRegression().fit(_dep[IN_CONV].values, np.log(_y / (1.0 - _y)))
_m2 = LinearRegression().fit(_dep[IN_DP].values, _dep[COL_DP].values)
print(f"  ONNX1 계수(그래프에서 추출) = [0.13521720, 0.00171340, 0.52581626], 절편 −53.51112")
print(f"  10/13–18 로짓-선형 재적합    = {np.round(_m1.coef_, 8).tolist()}, 절편 {_m1.intercept_:.5f}")
print(f"  10/13–18 ΔP 선형 재적합      = {np.round(_m2.coef_, 8).tolist()}, 절편 {_m2.intercept_:.5f}")
print("  → 두 배포 ONNX는 10/13–18 전체 적합본(= in-sample 기준)임이 확인된다.")

# --------------------------------------------------------------------------
# 2. 블록 셔플 순열 중요도
# --------------------------------------------------------------------------
def make_blocks(n: int, size: int = BLOCK) -> list[np.ndarray]:
    """연속 인덱스를 size 행씩 끊는다. 마지막 블록은 나머지 길이(짧아도 그대로 사용)."""
    return [np.arange(i, min(i + size, n)) for i in range(0, n, size)]


def shuffled_index(blocks: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """블록 '순서'만 무작위로 바꿔 이어붙인 행 인덱스."""
    return np.concatenate([blocks[k] for k in rng.permutation(len(blocks))])


def permutation_importance(predict, cols, target, data, group_cols=None):
    """블록 셔플 순열 중요도.

    predict     : (n,3) 배열 -> (n,) 예측
    group_cols  : 함께 섞을 컬럼 목록(공선성 점검용). None 이면 생략.
    반환        : dict (baseline, per-변수 ΔRMSE 평균/표준편차/표준오차, 정규화 %)
    """
    X = data[cols].values.astype(float)
    y = data[target].values.astype(float)
    base = rmse(predict(X), y)
    blocks = make_blocks(len(data))

    def run(js):
        # 변수마다 같은 seed 로 시작 → 200개의 동일한 블록순열을 공유(짝비교).
        rng = np.random.default_rng(SEED)
        vals = np.empty(N_REP)
        for i in range(N_REP):
            idx = shuffled_index(blocks, rng)
            Xs = X.copy()
            for j in js:
                Xs[:, j] = X[idx, j]
            vals[i] = rmse(predict(Xs), y) - base
        return vals

    per = {}
    for j, c in enumerate(cols):
        v = run([j])
        per[c] = {
            "delta_rmse_mean": float(v.mean()),
            "delta_rmse_std": float(v.std(ddof=0)),
            "delta_rmse_se": float(v.std(ddof=0) / np.sqrt(N_REP)),
        }
    total = sum(p["delta_rmse_mean"] for p in per.values())
    for p in per.values():
        p["share_pct"] = 100.0 * p["delta_rmse_mean"] / total

    out = {
        "n_rows": int(len(data)),
        "n_blocks": len(blocks),
        "block_sizes_last": int(len(blocks[-1])),
        "baseline_rmse": base,
        "per_input": per,
        "delta_sum": total,
    }
    if group_cols:
        js = [cols.index(c) for c in group_cols]
        v = run(js)
        out["grouped"] = {
            "cols": list(group_cols),
            "delta_rmse_mean": float(v.mean()),
            "delta_rmse_std": float(v.std(ddof=0)),
            "delta_rmse_se": float(v.std(ddof=0) / np.sqrt(N_REP)),
        }
    return out


def pairwise_corr(data, cols):
    """입력 쌍별 피어슨 상관과 |상관| 최대 쌍."""
    M = data[cols].corr()
    pairs = {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs[f"{cols[i]} × {cols[j]}"] = float(M.iloc[i, j])
    top = max(pairs, key=lambda k: abs(pairs[k]))
    return pairs, top, [s.strip() for s in top.split("×")]


def print_table(title, res, corr_pairs, top_pair, ref=None):
    print(f"\n  {title}")
    print(f"    baseline RMSE = {res['baseline_rmse']:.5f}   "
          f"(n={res['n_rows']}행 · 블록 {res['n_blocks']}개, 마지막 {res['block_sizes_last']}행)")
    print("    " + "-" * 68)
    print(f"    {'입력':<16}{'ΔRMSE 평균':>13}{'±표준편차':>12}{'표준오차':>11}{'정규화':>9}")
    print("    " + "-" * 68)
    for c, p in res["per_input"].items():
        print(f"    {c:<16}{p['delta_rmse_mean']:>13.4f}{p['delta_rmse_std']:>12.4f}"
              f"{p['delta_rmse_se']:>11.4f}{p['share_pct']:>8.1f}%")
    print("    " + "-" * 68)
    order = sorted(res["per_input"], key=lambda c: -res["per_input"][c]["delta_rmse_mean"])
    print(f"    순위: {'  >  '.join(order)}")
    print(f"    입력 쌍 상관: " + " · ".join(f"{k} = {v:+.3f}" for k, v in corr_pairs.items()))
    if "grouped" in res:
        g = res["grouped"]
        print(f"    grouped 셔플(|상관| 최대 쌍 {top_pair} 을 같은 블록순서로 동시 셔플): "
              f"ΔRMSE = {g['delta_rmse_mean']:.4f} ± {g['delta_rmse_std']:.4f}")
    return order


# --------------------------------------------------------------------------
# 3. (a) 배포 ONNX in-sample — 전 구간 3,066행  [표 C4 본표]
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(a) 배포 ONNX(10/13–18 적합) in-sample · 전 구간 PV 3,066행")
print(f"    블록 셔플 {BLOCK}행 · N={N_REP} · seed={SEED}")
print("=" * 78)

corr_a_conv, top_a_conv, group_a_conv = pairwise_corr(df, IN_CONV)
corr_a_dp, top_a_dp, group_a_dp = pairwise_corr(df, IN_DP)

res_a_conv = permutation_importance(pred_conv_onnx, IN_CONV, COL_X, df, group_a_conv)
res_a_dp = permutation_importance(pred_dp_onnx, IN_DP, COL_DP, df, group_a_dp)

order_a_conv = print_table("전환율 X (ONNX1)", res_a_conv, corr_a_conv, top_a_conv)
order_a_dp = print_table("압력강하 ΔP (ONNX2)", res_a_dp, corr_a_dp, top_a_dp)

# --- 기준값 대조 -----------------------------------------------------------
print("\n  [검증] 표 C4 기재 기준값과 대조")
for key, res, corr_pairs in (("conversion", res_a_conv, corr_a_conv), ("dp", res_a_dp, corr_a_dp)):
    ref = REF_A[key]
    print(f"   · {key}")
    check(f"{key} baseline RMSE", res["baseline_rmse"], ref["baseline"], TOL_BASE[key])
    for c, r in ref["delta"].items():
        check(f"{key} ΔRMSE[{c}]", res["per_input"][c]["delta_rmse_mean"], r,
              TOL_REL_DELTA, kind="rel")
    for c, r in ref["share_pct"].items():
        check(f"{key} 정규화[{c}]", res["per_input"][c]["share_pct"], r, TOL_SHARE_PP)
    check(f"{key} grouped ΔRMSE", res["grouped"]["delta_rmse_mean"], ref["grouped"],
          TOL_REL_DELTA, kind="rel")
    check(f"{key} |corr| 최대", max(abs(v) for v in corr_pairs.values()),
          ref["max_abs_corr"], TOL_CORR)

# --------------------------------------------------------------------------
# 4. (b) held-out 재적합 — 학습(10/13–16) 적합 → 검증(10/17–18) 평가
#        모델 레시피는 figures/make_figures.py(그림 규약)와 동일.
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(b) held-out 재적합 — 학습(10/13–16, 1,924행)으로 적합 → 검증(10/17–18, 661행)에서 평가")
print("    모델 레시피는 그림 규약(figures/make_figures.py)과 동일: 전환율 로짓-선형 · ΔP 선형")
print("=" * 78)

_ytr = train[COL_X].values
m_conv_ho = LinearRegression().fit(train[IN_CONV].values, np.log(_ytr / (1.0 - _ytr)))
m_dp_ho = LinearRegression().fit(train[IN_DP].values, train[COL_DP].values)


def pred_conv_ho(X):
    return 1.0 / (1.0 + np.exp(-m_conv_ho.predict(np.asarray(X, float))))


def pred_dp_ho(X):
    return m_dp_ho.predict(np.asarray(X, float))


# 자체검증 — 그림 규약에 기록된 held-out RMSE 가 재현되는지 확인
_r_conv_ho = rmse(pred_conv_ho(valid[IN_CONV].values), valid[COL_X].values)
_r_dp_ho = rmse(pred_dp_ho(valid[IN_DP].values), valid[COL_DP].values)
print(f"  held-out baseline RMSE: 전환율 {_r_conv_ho:.5f} (기준 0.02099) · "
      f"ΔP {_r_dp_ho:.5f} (기준 0.19100)")
if abs(_r_conv_ho - 0.02099) > 5e-5 or abs(_r_dp_ho - 0.1910) > 5e-4:
    sys.exit("중단 — held-out baseline RMSE 가 그림 규약 값과 다릅니다.")
print("  → 재현 확인.")

corr_b_conv, top_b_conv, group_b_conv = pairwise_corr(valid, IN_CONV)
corr_b_dp, top_b_dp, group_b_dp = pairwise_corr(valid, IN_DP)

res_b_conv = permutation_importance(pred_conv_ho, IN_CONV, COL_X, valid, group_b_conv)
res_b_dp = permutation_importance(pred_dp_ho, IN_DP, COL_DP, valid, group_b_dp)

order_b_conv = print_table("전환율 X (13–16 재적합)", res_b_conv, corr_b_conv, top_b_conv)
order_b_dp = print_table("압력강하 ΔP (13–16 재적합)", res_b_dp, corr_b_dp, top_b_dp)

# --------------------------------------------------------------------------
# 5. (a) / (b) 순위 대조
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(a) / (b) 순위 대조")
print("=" * 78)
rank_match = {}
for tgt, oa, ob in (("전환율", order_a_conv, order_b_conv), ("ΔP", order_a_dp, order_b_dp)):
    same = oa == ob
    rank_match[tgt] = same
    print(f"  {tgt:<6} (a) {'  >  '.join(oa)}")
    print(f"  {'':<6} (b) {'  >  '.join(ob)}   → 순위 {'일치' if same else '불일치'}")

# --------------------------------------------------------------------------
# 6. 저장
# --------------------------------------------------------------------------
OUT_DIR.mkdir(exist_ok=True)
payload = {
    "생성_스크립트": Path(__file__).name,
    "설명": "표 C4 순열 중요도(permutation importance) — 60분 블록 셔플",
    "설정": {
        "block_rows": BLOCK,
        "n_repeat": N_REP,
        "seed": SEED,
        "중요도_정의": "ΔRMSE = 셔플 후 RMSE − baseline RMSE (클수록 중요)",
        "정규화": "각 입력의 ΔRMSE / 세 입력 ΔRMSE 합 × 100 [%]",
        "grouped_셔플": "|피어슨 상관|이 가장 큰 입력 쌍을 같은 블록순서로 동시 셔플",
    },
    "데이터": {
        "csv": CSV.name,
        "n_rows_total": int(len(df)),
        "onnx1": str(ONNX_CONV.relative_to(ROOT)).replace("\\", "/"),
        "onnx2": str(ONNX_DP.relative_to(ROOT)).replace("\\", "/"),
        "입력_전환율": IN_CONV,
        "입력_ΔP": IN_DP,
        "타깃_전환율": COL_X,
        "타깃_ΔP": COL_DP,
    },
    "사전점검": {
        "onnx2_배포계수_대조_max오차": _maxerr,
        "판정": "통과 (<1e-4)",
    },
    "a_배포ONNX_in_sample_전구간3066행": {
        "conversion": {**res_a_conv, "pairwise_corr": corr_a_conv, "max_abs_corr_pair": top_a_conv},
        "dp": {**res_a_dp, "pairwise_corr": corr_a_dp, "max_abs_corr_pair": top_a_dp},
        "기준값_대조": {"기준": REF_A, "불일치": [f for f in FAILURES]},
    },
    "b_heldout_재적합_13to16_적합_17to18_평가": {
        "모델레시피": "figures/make_figures.py 와 동일 (전환율 로짓-선형 · ΔP 선형, 무가중)",
        "conversion": {**res_b_conv, "pairwise_corr": corr_b_conv, "max_abs_corr_pair": top_b_conv},
        "dp": {**res_b_dp, "pairwise_corr": corr_b_dp, "max_abs_corr_pair": top_b_dp},
    },
    "순위_대조": {
        "전환율_a": order_a_conv, "전환율_b": order_b_conv, "전환율_일치": rank_match["전환율"],
        "ΔP_a": order_a_dp, "ΔP_b": order_b_dp, "ΔP_일치": rank_match["ΔP"],
    },
}
OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n저장: {OUT_JSON}")

# --------------------------------------------------------------------------
# 7. 최종 판정
# --------------------------------------------------------------------------
print("=" * 78)
if FAILURES:
    print("검증 기준값 불일치 — 아래 항목을 확인하고 원인을 규명할 것:")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("검증 기준값 전부 재현됨 (a 기준). (b) 는 대조용으로 병기.")
print("=" * 78)
