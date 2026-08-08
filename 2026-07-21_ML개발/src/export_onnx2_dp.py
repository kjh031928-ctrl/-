"""export_onnx2_dp.py — ONNX2(반응기 압력강하 ΔP) 제작.

export_onnx.py 를 그대로 본뜨되 **로짓/Sigmoid 제거**(ΔP 무제약 선형). ONNX1 과 유일한 차이.

사양(§0-0 확정):
  모델   LinearRegression, 출력변환 없음, 가중치 없음
  입력   F2 = [TT-1006(°C), 총유량(kg/h)=FT-1003+FT-1004, FRC-1004], export dtype float32, shape [None,3]
  출력   ΔP(kPa) 1개, 이름 "dP", FLOAT [None,1]
  학습   10/13~18 전체(Split '학습'+'검증'), 10/12 제외 (ONNX1 재학습본과 동일 원칙)
  라벨   ΔP = PT-1004 − 25 − PT-1100  (25kPa=HX-1005 튜브측 0.25bar [주어짐] rev2 App B Table 4)

보고 성능지표 = 학습 10/13-16 → 검증 10/17-18 held-out RMSE 0.1910. 재학습본은 동일 레시피.
"""
from __future__ import annotations
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError): pass

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LinearRegression

from onnx2_preanalysis import load, ols, predict, rmse, TRAIN, VALID

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "onnx" / "reactor_dp.onnx"
FEATS = ["TT-1006.PV", "Ftot_b", "FRC-1004.PV"]   # 입력 순서 고정 F2

# 검증점 [TT-1006, 총유량, FRC]
CHECK = {
    "10/12앵커[341.57,18000,5.0]": [341.57, 18000.0, 5.0],   # 학습 held-out, 기대 ΔP≈29.22
    "335C[16500,5.0]":            [335.0, 16500.0, 5.0],
    "350C[16500,5.0]":            [350.0, 16500.0, 5.0],
    "365C외삽[16500,5.0]":         [365.0, 16500.0, 5.0],    # 외삽(선형 추세연장, clamp 없음)
}
ANCHOR_EXPECT = 29.22   # 10/12 정상운전 ΔP [계산, 사전분석]


def main():
    df = load()

    # ── 보고 성능: 13-16 학습 → 17-18 검증 held-out RMSE ──
    tr = df[df["Split"] == TRAIN]; va = df[df["Split"] == VALID]
    Xtr = tr[FEATS].to_numpy(float); ytr = tr["dP"].to_numpy(float)
    Xva = va[FEATS].to_numpy(float); yva = va["dP"].to_numpy(float)
    ch, ic_h = ols(Xtr, ytr)
    held_rmse = rmse(yva, predict(ch, ic_h, Xva))

    # ── 배포본: 10/13-18 전체 재학습(10/12 제외) ──
    full = df[df["Split"].isin([TRAIN, VALID])]
    Xf = full[FEATS].to_numpy(np.float64); yf = full["dP"].to_numpy(np.float64)
    model = LinearRegression().fit(Xf, yf)   # 가중치 없음
    print(f"[학습] 재학습본 데이터 {len(full)}행 (10/13-18, 10/12 제외), "
          f"캠페인 {sorted(full['Campaign'].unique())}")
    print(f"[계수] 절편={model.intercept_:.4f}  b_TT-1006={model.coef_[0]:.6f}  "
          f"b_총유량={model.coef_[1]:.8f}  b_FRC={model.coef_[2]:.5f}")
    print(f"[보고 성능] 학습 10/13-16 → 검증 10/17-18 held-out RMSE = {held_rmse:.4f} kPa "
          f"(재학습본은 동일 레시피)")

    # ── Export: to_onnx (로짓/Sigmoid 없음) → 출력 'dP' ──
    onx = to_onnx(model, initial_types=[("X", FloatTensorType([None, 3]))], target_opset=None)
    g = onx.graph
    lin_out = g.output[0].name
    g.node.append(helper.make_node("Identity", inputs=[lin_out], outputs=["dP"], name="dP_out"))
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("dP", TensorProto.FLOAT, [None, 1]))
    onnx.checker.check_model(onx)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onx, str(_OUT))

    # ── Sanity: onnxruntime vs sklearn(py) ──
    sess = ort.InferenceSession(str(_OUT), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    Xchk = np.array(list(CHECK.values()), dtype=np.float64)
    py = model.predict(Xchk)
    ox = np.asarray(sess.run(None, {iname: Xchk.astype(np.float32)})[0]).ravel()
    maxdiff = float(np.max(np.abs(py - ox)))

    print(f"\n{'검증점':26}{'sklearn(py)':>13}{'ONNX':>12}{'|차이|':>12}")
    for (name, _), p, o in zip(CHECK.items(), py, ox):
        print(f"  {name:24}{p:>13.5f}{o:>12.5f}{abs(p-o):>12.2e}")
    print(f"\n[Sanity] 최대 |py − ONNX| = {maxdiff:.2e}  (기준 ≤1e-4: {'OK' if maxdiff<=1e-4 else '실패'})")
    print(f"[Sigmoid 없음] 그래프 노드: {[n.op_type for n in g.node]}")

    # 10/12 앵커 (학습 미포함) 정합
    anchor = float(py[0])
    print(f"[10/12 앵커] 예측 ΔP={anchor:.4f} kPa vs 기대 {ANCHOR_EXPECT} → 차이 {anchor-ANCHOR_EXPECT:+.4f} kPa (학습 held-out)")
    # 외삽 clamp 확인
    print(f"[외삽] 350C={py[2]:.4f} → 365C={py[3]:.4f} kPa, 증가 {py[3]-py[2]:+.4f} "
          f"(선형 추세연장, clamp 없음)")
    print(f"\n[산출] {_OUT.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
