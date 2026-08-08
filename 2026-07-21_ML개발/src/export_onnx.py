"""export_onnx.py — 최종 모델(로짓 1차 선형) 전체 재학습 + ONNX export(시그모이드 포함).

담당: 김지훈 전용 (CLAUDE.md §3). 공용 모듈(data_loader 등)은 import 만.

Phase 11:
  [단계1] 확정 레시피(로짓변환 + 1차 선형 + I1 + 가중치없음)로 r0/r1 각각 재학습.
          학습 데이터 = 전체 10/13~18 (Split '학습' 또는 '검증'). 10/12 제외.
          ★ 이 재학습본의 성능은 10/17-18로 다시 잴 수 없음(학습에 포함). 보고 성능지표는
            Phase 6/10의 held-out 검증값이며, 재학습본은 '동일 레시피'임(문서 명시).
  [단계2] ONNX export — 방법 A: skl2onnx 로 로짓(선형) 출력 ONNX 생성 후 그래프에 Sigmoid
          노드를 append 하여 ONNX가 직접 전환율(0~1)을 출력. 시뮬레이션 쪽은 로짓 불필요.
          입력 순서 고정 [TT-1006(°C), FT-1004(kg/h), FRC-1004(무차원)], 입력 dtype float32.
          ★ 라벨생성·학습은 float64, ONNX 변환 시점만 float32 (CLAUDE.md §2.4).
  [단계3] onnxruntime 로 sklearn(py) 예측과 ONNX 예측 일치(1e-5 이상) 검증.

물성/상수는 config·interface 에서 읽는다. 로짓 eps 등은 지시서 Phase 10-A/11 명시값.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp949 콘솔 출력 보호
except (AttributeError, ValueError):
    pass

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from scipy.special import expit
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LinearRegression

import data_loader as dl
import interface as itf

_ROOT = Path(__file__).resolve().parents[1]

LOGIT_EPS = 1e-6  # 로짓변환 경계 클립 (지시서 Phase 10-A/11). 데이터 X∈0.29~0.85라 미발동.

# 검증점(재학습 대조용). 10/12 정상운전 조건 + 외삽/저온 확인점.
#   [TT-1006(°C), FT-1004(kg/h), FRC-1004] 순서. FT=3000/FRC=5.0 은 10/12 정상운전값.
CHECK_POINTS = {
    "10/12(341.57)": [341.57, 3000.0, 5.0],
    "335C": [335.0, 3000.0, 5.0],
    "350C": [350.0, 3000.0, 5.0],
    "365C": [365.0, 3000.0, 5.0],
}


def _fit_logit_linear(full, label: str) -> LinearRegression:
    """로짓공간 1차 선형 적합. z=log(y/(1-y)), y는 eps 클립. float64 학습."""
    X, y = dl.get_Xy(full, "I1", label)  # I1 = [TT-1006, FT-1004, FRC-1004]
    yc = np.clip(y, LOGIT_EPS, 1.0 - LOGIT_EPS)
    z = np.log(yc / (1.0 - yc))
    return LinearRegression().fit(X, z)  # 가중치 없음


def _py_predict(model: LinearRegression, X: np.ndarray) -> np.ndarray:
    """파이썬 최종 예측 = sigmoid(선형 로짓). float64."""
    return expit(model.predict(np.asarray(X, dtype=np.float64)))


def _export_onnx_with_sigmoid(model: LinearRegression, out_path: Path) -> None:
    """방법 A: 선형(로짓) ONNX 생성 → Sigmoid 노드 append → 전환율(0~1) 직접 출력.

    입력 dtype float32 (ONNX 변환 시점만). 입력 순서 고정 [TT-1006, FT-1004, FRC-1004].
    """
    # A-1) 로짓(선형) 출력 ONNX. 입력 이름 X, shape [None,3], float32.
    initial_types = [("X", FloatTensorType([None, 3]))]
    onx = to_onnx(model, initial_types=initial_types, target_opset=None)

    graph = onx.graph
    lin_out = graph.output[0].name  # 선형 출력(로짓 z) 텐서 이름

    # A-2) Sigmoid 노드 추가 → 최종 출력 conversion = 1/(1+exp(-z))
    final_name = "conversion"
    sig_node = helper.make_node("Sigmoid", inputs=[lin_out], outputs=[final_name],
                                name="logit_inverse_sigmoid")
    graph.node.append(sig_node)

    # 그래프 출력을 시그모이드 결과로 교체
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info(final_name, TensorProto.FLOAT, [None, 1])
    )

    onnx.checker.check_model(onx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onx, str(out_path))


def _onnx_predict(onnx_path: Path, X: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    out = sess.run(None, {iname: np.asarray(X, dtype=np.float32)})[0]
    return np.asarray(out).ravel()


def run() -> None:
    cfg = dl.load_config()
    itf.validate_against_config(cfg)
    df = dl.load_data(cfg)
    tv, vv = cfg["split"]["train_value"], cfg["split"]["valid_value"]
    full = df[df["Split"].isin([tv, vv])]  # 10/13~18 (10/12 제외)
    print(f"[단계1] 재학습 데이터: {len(full)}행, 캠페인 {sorted(full['Campaign'].unique())}")
    print(f"        레시피: 로짓 + 1차 선형 + I1 + 가중치없음 | 입력순서 {itf.ONNX_INPUT_ORDER}")

    Xchk = np.array(list(CHECK_POINTS.values()), dtype=np.float64)
    onnx_dir = _ROOT / cfg["paths"]["onnx_dir"]

    for label, fname in [("r0", "reactor_conversion_r0.onnx"),
                         ("r1", "reactor_conversion_r1.onnx")]:
        model = _fit_logit_linear(full, label)
        out_path = onnx_dir / fname
        _export_onnx_with_sigmoid(model, out_path)

        py = _py_predict(model, Xchk)
        ox = _onnx_predict(out_path, Xchk)
        maxdiff = float(np.max(np.abs(py - ox)))

        print(f"\n[단계1/2/3] {label} → {fname}")
        print(f"  {'검증점':16}{'sklearn(py)':>14}{'ONNX':>14}{'|차이|':>12}")
        for (name, _), p, o in zip(CHECK_POINTS.items(), py, ox):
            print(f"  {name:16}{p:>14.6f}{o:>14.6f}{abs(p - o):>12.2e}")
        viol = int(np.sum(ox > 1.0))
        print(f"  최대 |py-ONNX| 차이 = {maxdiff:.2e}  (기준 1e-4 이하: "
              f"{'OK' if maxdiff <= 1e-4 else '실패'}) | 물리상한(>1) 위반 {viol}건")

    # 검증셋 몇 행 추가 대조(방법 A 성공 실증)
    valid = dl.get_split(df, vv)
    Xv, _ = dl.get_Xy(valid.head(5), "I1", "r0")
    m_r0 = _fit_logit_linear(full, "r0")
    py = _py_predict(m_r0, Xv); ox = _onnx_predict(onnx_dir / "reactor_conversion_r0.onnx", Xv)
    print(f"\n[단계3] 검증셋 앞 5행(r0) 최대차이 = {np.max(np.abs(py-ox)):.2e}")
    print("\n판정: 방법 A(시그모이드 ONNX 내장) 성공 여부는 위 |차이| 로 확인.")


if __name__ == "__main__":
    run()
