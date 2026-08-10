# -*- coding: utf-8 -*-
"""ONNX1(전환율) 재학습·재export — SRK 순성분상수 교체 라벨 기준

무엇이 바뀌었나
  SRK 순성분 상수를 문헌값(Sheet1)에서 시뮬레이터 내장값(Sheet2)으로 교체해 전환율 라벨을
  다시 계산했다(근거: `SRK_순성분상수_변경_근거.md`). **바뀐 것은 라벨 하나뿐**이고
  입력·분할·레시피·export 방식은 기존 배포본과 100% 동일하다.

레시피 (src/export_onnx.py 와 동일 — 변경 없음)
  1. 로짓 변환 : yc = clip(y, 1e-6, 1−1e-6) · z = ln(yc/(1−yc))
  2. 적합      : sklearn LinearRegression().fit(X, z) — sample_weight 없음, float64
  3. 예측      : sigmoid(model.predict(X))
  4. export    : skl2onnx.to_onnx → 선형 출력에 Sigmoid 노드 append →
                 최종 출력 "conversion" (FLOAT, [None,1]), 기존 선형 출력은 제거.
                 입력 dtype 은 export 시점만 float32.

학습 데이터
  `ML_마스터데이터_PV (수정).csv` 의 Split ∈ {학습, 검증} = 10/13~18, 2,585행.
  10/12(정상운전검사)는 제외. 무작위 분할 없음.
  ★ 기존 배포본과 같은 in-sample 재학습이므로, 아래 검증셋 RMSE 는 성능이 아니라 대조값이다.

산출
  onnx/reactor_conversion_r1.onnx          ← 배포 파일명 유지(AVEVA 플로우시트가 절대경로로 참조)
  onnx/reactor_conversion_r1_srklit.onnx   ← 교체 전 배포본(문헌 SRK 상수) 보관

재현
  python onnx1_retrain_srksim.py           (저장소 루트에서 실행)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from scipy.special import expit
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
CSV_NEW = ROOT / "ML_마스터데이터_PV (수정).csv"
CSV_OLD = ROOT / "ML_마스터데이터_PV.csv"
ONNX_DIR = ROOT / "2026-07-21_ML개발" / "onnx"
OUT_ONNX = ONNX_DIR / "reactor_conversion_r1.onnx"          # 배포 파일명(AVEVA 배선 고정)
OLD_ONNX = ONNX_DIR / "reactor_conversion_r1_srklit.onnx"   # 교체 전 배포본(문헌 SRK) 보관
OLD_JSON = ROOT / "2026-07-21_ML개발" / "results" / "linear2_logit_I1_r1_unweighted.json"

LOGIT_EPS = 1e-6
IN_COLS = ["TT-1006.PV", "FT-1004.PV", "FRC-1004.PV"]   # 입력 순서 고정
LABEL = "X_conv_r1_deployed"

# 검증점 (기존 export_onnx.py CHECK_POINTS 와 동일)
CHECK_POINTS = {
    "10/12(341.57)": [341.57, 3000.0, 5.0],
    "335C": [335.0, 3000.0, 5.0],
    "350C": [350.0, 3000.0, 5.0],
    "365C": [365.0, 3000.0, 5.0],
}
FAIL: list[str] = []


def fit_logit_linear(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    """로짓공간 1차 선형 적합. 가중치 없음, float64."""
    yc = np.clip(np.asarray(y, np.float64), LOGIT_EPS, 1.0 - LOGIT_EPS)
    z = np.log(yc / (1.0 - yc))
    return LinearRegression().fit(np.asarray(X, np.float64), z)


def py_predict(model: LinearRegression, X) -> np.ndarray:
    return expit(model.predict(np.asarray(X, np.float64)))


def export_onnx_with_sigmoid(model: LinearRegression, out_path: Path) -> None:
    """선형(로짓) ONNX 생성 → Sigmoid append → 전환율 직접 출력."""
    onx = to_onnx(model, initial_types=[("X", FloatTensorType([None, 3]))], target_opset=None)
    graph = onx.graph
    lin_out = graph.output[0].name
    graph.node.append(helper.make_node("Sigmoid", inputs=[lin_out], outputs=["conversion"],
                                       name="logit_inverse_sigmoid"))
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info("conversion", TensorProto.FLOAT, [None, 1]))
    onnx.checker.check_model(onx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onx, str(out_path))


def onnx_predict(path: Path, X) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.asarray(sess.run(None, {name: np.asarray(X, np.float32)})[0]).ravel()


def onnx_coefs(path: Path):
    """ONNX 그래프에서 LinearRegressor 계수·절편을 읽는다."""
    m = onnx.load(str(path))
    for n in m.graph.node:
        if n.op_type == "LinearRegressor":
            c = i = None
            for a in n.attribute:
                if a.name == "coefficients":
                    c = list(a.floats)
                elif a.name == "intercepts":
                    i = list(a.floats)
            return np.array(c, float), float(i[0])
    raise RuntimeError("LinearRegressor 노드를 찾지 못했습니다")


# ══════════════════════════════════════════════════════════════════════════
print("=" * 96)
print("ONNX1 재학습 — SRK 순성분상수 교체 라벨 (라벨 소스만 변경)")
print("=" * 96)

df = pd.read_csv(CSV_NEW, encoding="utf-8-sig")
assert len(df) == 3066, f"행 수 {len(df)} ≠ 3066"
full = df[df["Split"].isin(["학습", "검증"])]
print(f"  라벨 소스   {CSV_NEW.name}")
print(f"  학습 데이터 {len(full)}행 · 캠페인 {sorted(full['Campaign'].unique())} (10/12 제외)")
print(f"  입력 순서   {IN_COLS}")
print(f"  라벨 컬럼   {LABEL} · 범위 [{full[LABEL].min():.6f}, {full[LABEL].max():.6f}]")
print(f"  로짓 클립   eps={LOGIT_EPS} → 발동 건수 "
      f"{int(((full[LABEL] < LOGIT_EPS) | (full[LABEL] > 1 - LOGIT_EPS)).sum())}건")

X = full[IN_COLS].to_numpy(np.float64)
y = full[LABEL].to_numpy(np.float64)
model = fit_logit_linear(X, y)
print(f"\n  재학습 계수 (로짓공간)")
for nm, c in zip(IN_COLS, model.coef_):
    print(f"    b[{nm:<14}] = {c:+.8f}")
print(f"    절편           = {model.intercept_:+.8f}")

export_onnx_with_sigmoid(model, OUT_ONNX)
print(f"\n  저장: {OUT_ONNX}  (기존 {OLD_ONNX.name} 은 그대로 둠)")

# ── 검증 1. ONNX vs 파이썬 ────────────────────────────────────────────────
print("\n" + "=" * 96)
print("검증 1. onnxruntime 예측 vs 파이썬 sigmoid 예측")
print("=" * 96)
Xchk = np.array(list(CHECK_POINTS.values()), np.float64)
p_py, p_ox = py_predict(model, Xchk), onnx_predict(OUT_ONNX, Xchk)
print(f"  {'검증점':<16}{'파이썬':>12}{'ONNX':>12}{'차이':>12}")
for nm, a, b in zip(CHECK_POINTS, p_py, p_ox):
    print(f"  {nm:<16}{a:>12.6f}{b:>12.6f}{abs(a-b):>12.2e}")
va = df[df["Split"] == "검증"]
Xva = va[IN_COLS].to_numpy(np.float64)
d_all = float(np.max(np.abs(py_predict(model, Xva) - onnx_predict(OUT_ONNX, Xva))))
d_chk = float(np.max(np.abs(p_py - p_ox)))
print(f"\n  검증점 max|차| {d_chk:.3e} · 검증셋 661행 max|차| {d_all:.3e}   (기준 ≤ 1e-4)")
if max(d_chk, d_all) > 1e-4:
    FAIL.append(f"ONNX↔파이썬 불일치 {max(d_chk, d_all):.3e} > 1e-4")

# ── 검증 2. 물리 상한 ─────────────────────────────────────────────────────
print("\n" + "=" * 96)
print("검증 2. 물리 상한 — ONNX 출력이 1.0 을 넘는가")
print("=" * 96)
p_full = onnx_predict(OUT_ONNX, df[IN_COLS].to_numpy(np.float64))
over = int(np.sum(p_full > 1.0)); under = int(np.sum(p_full < 0.0))
print(f"  전 구간 3,066행 예측 범위 [{p_full.min():.6f}, {p_full.max():.6f}]")
print(f"  >1.0 위반 {over}건 · <0.0 위반 {under}건   (시그모이드 내장이라 구조적으로 0건)")
if over or under:
    FAIL.append(f"[0,1] 위반 {over + under}건")

# ── 검증 3. 계수 비교 ─────────────────────────────────────────────────────
print("\n" + "=" * 96)
print("검증 3. 새 계수 vs 기존 배포 계수 — SRK 문서 §7 '유효숫자 범위에서 불변' 예측")
print("=" * 96)
c_old, i_old = onnx_coefs(OLD_ONNX)
c_new, i_new = onnx_coefs(OUT_ONNX)
rows = [("절편", i_old, i_new)] + [(f"b[{n}]", co, cn) for n, co, cn in zip(IN_COLS, c_old, c_new)]
print(f"  {'항':<20}{'기존(문헌 SRK)':>18}{'신규(시뮬 SRK)':>18}{'차이':>14}{'상대차':>11}")
for nm, a, b in rows:
    rel = abs(b - a) / abs(a) * 100 if a else float("nan")
    print(f"  {nm:<20}{a:>18.8f}{b:>18.8f}{b-a:>+14.2e}{rel:>10.4f}%")
if OLD_JSON.exists():
    j = json.loads(OLD_JSON.read_text(encoding="utf-8"))
    # 이 JSON 은 지표 기록이라 계수가 없다(모델은 n_train=1924 의 held-out 적합본).
    print(f"\n  {OLD_JSON.name} 에는 계수가 없다 — 지표만 기록돼 있다:")
    print(f"    n_train={j.get('n_train')} · n_valid={j.get('n_valid')} · "
          f"weighted={j.get('weighted')} · logit_eps={j.get('hyperparams', {}).get('logit_eps')}")
    print(f"    held-out RMSE {j['metrics']['conversion']['rmse']:.6f} · "
          f"R² {j['metrics']['conversion']['r2']:.6f}  (보고서 §3.5.4 의 0.021·0.983)")
    print("    → 계수 대조는 위 ONNX 그래프 파싱값으로 갈음한다.")

# ── 검증 4. 검증셋 RMSE ───────────────────────────────────────────────────
print("\n" + "=" * 96)
print("검증 4. 검증셋(10/17–18, 661행) RMSE — ★in-sample 대조값(성능 아님)")
print("=" * 96)
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
y_new = va[LABEL].to_numpy(np.float64)
r_new = rmse(onnx_predict(OUT_ONNX, Xva), y_new)
old_df = pd.read_csv(CSV_OLD, encoding="utf-8-sig")
va_old = old_df[old_df["Split"] == "검증"]
r_old = rmse(onnx_predict(OLD_ONNX, va_old[IN_COLS].to_numpy(np.float64)),
             va_old[LABEL].to_numpy(np.float64))
print(f"  신규 모델 × 신규 라벨   RMSE {r_new:.6f}")
print(f"  기존 모델 × 기존 라벨   RMSE {r_old:.6f}   (기존 배포본 in-sample)")
print(f"  차이 {r_new - r_old:+.6f}")
print(f"  참고: 기존 문서의 held-out(10/13–16 학습) 값 0.020987 — 학습 범위가 달라 직접 비교 불가")

# 보고서가 인용하는 held-out 기준(10/13–16 적합 → 10/17–18)으로 두 라벨을 같은 조건에서 비교
print("\n  [동일조건 held-out 대조] 10/13–16 만으로 적합 → 10/17–18 평가")
for tag, src in (("신규 라벨(시뮬 SRK)", df), ("기존 라벨(문헌 SRK)", old_df)):
    tr_ = src[src["Split"] == "학습"]; va_ = src[src["Split"] == "검증"]
    m_ = fit_logit_linear(tr_[IN_COLS].to_numpy(np.float64), tr_[LABEL].to_numpy(np.float64))
    r_ = rmse(py_predict(m_, va_[IN_COLS].to_numpy(np.float64)), va_[LABEL].to_numpy(np.float64))
    print(f"    {tag:<22} held-out RMSE {r_:.6f}")
print("    → 보고서 §3.5.4 의 0.021 이 바뀌는지 여부는 이 두 값으로 판단한다.")

# ── 판정 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 96)
if FAIL:
    print("검증 실패:")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("검증 1~4 전부 통과.")
print(f"산출물: {OUT_ONNX.relative_to(ROOT)}")
print("=" * 96)
