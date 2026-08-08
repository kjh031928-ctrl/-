"""보고서37_물리보존법칙_검증_재현스크립트.py
=====================================================================
목적: ML 보고서 §3.7 의 검증 수치를 데이터로 재현한다.
      **적합도(RMSE·R²)가 아니라 '물리·보존 법칙을 지키는가'** 를 본다.

  (1) 27점 물리 격자 단조성      — 초안 주장 9/9
  (2) 에너지수지 상관            — 초안 주장 ≥ +0.99
  (3) 큐멘 물질수지 오차%/캠페인  — 초안 주장 ±3% 이내
  (4) 잔차 자기상관 (1분·60분)    — 초안 주장 lag1 ≈ 0.997

답을 강제하지 않는다. 어긋나면 어긋난 대로 출력한다.

절대규칙(루트 CLAUDE.md): 원본 읽기전용 · .xlsx 산출 금지 · seed=42 ·
  폐기컬럼(ML_weight·TT-1006_rate_C_per_min·τ) 일체 미사용(무가중) ·
  [주어짐]/[계산]/[가정] 구분 · 행 수를 신뢰도 근거로 쓰지 말 것(자기상관).
=====================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import onnxruntime as ort
import pandas as pd

SEED = 42                    # [주어짐] CLAUDE.md §2.3 (본 스크립트는 난수 미사용, 고정만 명시)
np.random.seed(SEED)

# --- 상수 (출처 주석 필수) ---
W_PROPENE = 0.94773          # 프로펜 공급 질량분율 [주어짐] CLAUDE.md §2.3
MW_PROPENE = 42.081          # g/mol  [주어짐] 지시서 상수
MW_CUMENE = 120.196          # g/mol  [주어짐] 지시서 상수

# 27점 격자 [주어짐 — 지시서]
GRID_T = [335.0, 350.0, 365.0]          # °C
GRID_F = [15000.0, 16500.0, 18000.0]    # kg/h 총유량
GRID_R = [4.0, 5.0, 6.0]                # B/P 질량비

_ROOT = Path(__file__).resolve().parents[1]
_MASTER = _ROOT / "ML_마스터데이터_PV.csv"
_ONNX1 = _ROOT / "2026-07-21_ML개발" / "onnx" / "reactor_conversion_r1.onnx"
_ONNX2 = _ROOT / "2026-07-21_ML개발" / "onnx" / "reactor_dp.onnx"

CAMPS = ["10/12", "10/13", "10/14", "10/15", "10/16", "10/17-18"]
ANCHOR = "10/12"             # 481행 전체가 동일 정상상태 반복 → 상관·ACF 정의 불가


def hr(t, ch="="):
    print("\n" + ch * 78); print(t); print(ch * 78)


def corr(a, b):
    """상수열(부동소수점 잔차 수준 포함)이면 nan. 10/12 대응."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    for v in (a, b):
        if np.ptp(v) <= 1e-12 * max(abs(np.mean(v)), 1.0):
            return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


# ============================================================ 모델 로드
class Model:
    def __init__(self, path):
        self.s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.iname = self.s.get_inputs()[0].name
        self.oname = self.s.get_outputs()[0].name

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float32).reshape(-1, 3)
        return self.s.run([self.oname], {self.iname: X})[0].ravel().astype(np.float64)


def load():
    df = pd.read_csv(_MASTER, encoding="utf-8-sig")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def print_coefs():
    """배포 ONNX 의 선형 계수를 그대로 꺼내 부호를 확인한다.
    단조성 실패의 원인이 모델 계수 부호인지 확인하기 위함."""
    import onnx
    print("\n배포 ONNX 계수 (LinearRegressor 노드에서 직접 추출) [계산]")
    for path, feats, form in [
            (_ONNX1, ["TT-1006", "FT-1004", "FRC-1004"], "logit(X) = ..., X = sigmoid(·)"),
            (_ONNX2, ["TT-1006", "총유량", "FRC-1004"], "ΔP = ... [kPa]")]:
        g = onnx.load(str(path)).graph
        node = [n for n in g.node if n.op_type == "LinearRegressor"][0]
        att = {a.name: a for a in node.attribute}
        coef = list(att["coefficients"].floats)
        icpt = list(att["intercepts"].floats)
        print(f"  {path.name}   ({form})")
        for f, c in zip(feats, coef):
            print(f"    {f:>10} {c:+.8g}")
        print(f"    {'절편':>10} {icpt[0]:+.8g}")


def sec0b(df):
    """배포 ONNX 가 어느 구간으로 학습됐는지 계수 대조로 역추적한다.
    held-out RMSE 를 배포 ONNX 로 계산해도 되는지가 여기서 갈린다."""
    from sklearn.linear_model import LinearRegression
    hr("(0-보강) 배포 ONNX 는 어느 구간으로 학습됐나 — 계수 대조  [계산]", "-")
    df = df.copy()
    df["logit"] = np.log(df["X_conv_r1_deployed"] / (1 - df["X_conv_r1_deployed"]))
    F1 = ["TT-1006.PV", "FT-1004.PV", "FRC-1004.PV"]
    F2 = ["TT-1006.PV", "total_flow_kgh", "FRC-1004.PV"]
    subs = [("학습 10/13-16", df[df["Campaign"].isin(["10/13", "10/14", "10/15", "10/16"])]),
            ("10/13-18 전체", df[df["Campaign"] != ANCHOR]),
            ("10/12-18 전체", df)]
    for tag, feats, ycol, onnx_c in [
            ("ONNX1 logit(X)", F1, "logit", [0.1352172, 0.0017134006, 0.52581626, -53.51112]),
            ("ONNX2 ΔP", F2, "DP_reactor_kPa", [0.084976934, 0.0031447636, -0.9400878, -51.710121])]:
        print(f"\n  [{tag}] 배포 계수: " +
              " ".join(f"{c:+.8g}" for c in onnx_c[:3]) + f"  절편 {onnx_c[3]:+.8g}")
        for nm, g in subs:
            m = LinearRegression().fit(g[feats].to_numpy(float), g[ycol].to_numpy(float))
            d = max(max(abs(a - b) for a, b in zip(m.coef_, onnx_c[:3])),
                    abs(m.intercept_ - onnx_c[3]))
            print(f"    {nm:<14}" + " ".join(f"{c:+.8g}" for c in m.coef_) +
                  f"  절편 {m.intercept_:+.8g}   최대차 {d:.2e}"
                  f"{'   ← 일치' if d < 1e-5 else ''}")
    print("\n  → 두 배포 ONNX 는 10/13-18 **전체**로 재학습된 모델이다(ONNX2 는 문서에 기록된 결정).")
    print("     따라서 배포 ONNX 로 계산한 10/17-18 RMSE 는 검증치가 아니라 **in-sample** 값이다.")
    print("     §3.7 에 인용할 검증 성능은 10/13-16 학습 모델의 값(0.020987 / 0.1910)이어야 한다.")


def sec1b(df):
    """단조성 실패가 모델 결함인지, 데이터가 원래 그런지 판정한다."""
    from sklearn.linear_model import LinearRegression

    def pcorr(g, x, y, ctrl):
        C = g[ctrl].to_numpy(float)
        rx = g[x] - LinearRegression().fit(C, g[x]).predict(C)
        ry = g[y] - LinearRegression().fit(C, g[y]).predict(C)
        return corr(rx, ry)

    hr("(1-보강) 단조성 실패의 원인 — 데이터 자체는 어느 방향인가  [계산]", "-")
    print("10/13 은 유량만 스윕한 캠페인(온도·비율 거의 고정)이므로 유량 방향을 직접 읽을 수 있다.")
    print(f"  {'캠페인':<10}{'corr(FT-1004, X)':>18}{'corr(총유량, X)':>16}"
          f"{'corr(FRC, X)':>14}{'T 변동폭':>10}{'FRC 변동폭':>11}")
    for c in CAMPS:
        g = df[df["Campaign"] == c]
        print(f"  {c:<10}{corr(g['FT-1004.PV'], g['X_conv_r1_deployed']):>18.4f}"
              f"{corr(g['total_flow_kgh'], g['X_conv_r1_deployed']):>16.4f}"
              f"{corr(g['FRC-1004.PV'], g['X_conv_r1_deployed']):>14.4f}"
              f"{np.ptp(g['TT-1006.PV']):>10.3f}{np.ptp(g['FRC-1004.PV']):>11.3f}")
    g13 = df[df["Campaign"] == "10/13"]
    print(f"\n  10/13 단순회귀 기울기 dX/dFT-1004 = "
          f"{np.polyfit(g13['FT-1004.PV'], g13['X_conv_r1_deployed'], 1)[0]:+.3e} 1/(kg/h)")
    print(f"  10/13 corr(PT-1100, FT-1004) = "
          f"{corr(g13['PT-1100.PV'], g13['FT-1004.PV']):+.4f}  "
          f"(압력이 유량의 그림자 — 루트 CLAUDE.md 기록 +0.9997)")

    print("\n  10/13 은 온도가 9.36°C 움직였으므로 단순상관만으로는 부족하다 → 편상관으로 분리:")
    tr = df[df["Campaign"].isin(["10/13", "10/14", "10/15", "10/16"])]
    for nm, g in [("10/13", g13), ("10/15", df[df["Campaign"] == "10/15"]),
                  ("10/17-18", df[df["Campaign"] == "10/17-18"]), ("학습 13-16", tr)]:
        p1 = pcorr(g, "FT-1004.PV", "X_conv_r1_deployed", ["TT-1006.PV", "FRC-1004.PV"])
        p2 = pcorr(g, "FRC-1004.PV", "X_conv_r1_deployed", ["TT-1006.PV", "FT-1004.PV"])
        print(f"    {nm:<10} 편상관(FT-1004, X | T,FRC) = {p1:+.4f}   "
              f"편상관(FRC, X | T,FT-1004) = {p2:+.4f}")
    print("\n  판정: 온도·비율을 고정해도 **유량↑ → X↑** 가 남는다(학습 전체 편상관 +0.98).")
    print("        즉 모델은 데이터를 충실히 재현한 것이고, 어긋난 것은 '체류시간' 기대 쪽이다.")
    print("        비율 방향(FRC↑ → X↑)은 편상관 +0.89 로 기대와 **일치**한다 —")
    print("        27점 격자의 B/P 축이 실패한 것은 그 축이 총유량을 고정해 프로펜유량을 함께 낮추기 때문이다.")
    print("  [한계] 유량을 압력과 분리해 조작한 캠페인이 없어, 체류시간 효과와 압력 효과를")
    print("         이 데이터로는 식별할 수 없다. 위 판정은 '이 데이터 안에서'로 한정된다.")


# ================================================= 0. 구동 검증(sanity)
def sec0(df, m1, m2):
    hr("0. 모델 구동 검증 — ONNX 를 올바로 호출하고 있는가  [계산]")
    print(f"onnxruntime {ort.__version__} · numpy {np.__version__} · "
          f"pandas {pd.__version__} · seed={SEED}")
    print(f"마스터 {len(df)}행 · 결측 {int(df.isna().sum().sum())} · "
          f"중복 Timestamp {int(df['Timestamp'].duplicated().sum())}")

    x1 = m1(df[["TT-1006.PV", "FT-1004.PV", "FRC-1004.PV"]].to_numpy())
    x2 = m2(df[["TT-1006.PV", "total_flow_kgh", "FRC-1004.PV"]].to_numpy())
    df["X_hat"] = x1
    df["DP_hat"] = x2

    ho = df["Campaign"] == "10/17-18"
    print(f"\nheld-out(10/17-18) RMSE 대조 — 기록값과 맞아야 호출이 옳다:")
    print(f"  ONNX1 전환율 : {rmse(df.loc[ho,'X_conv_r1_deployed'], df.loc[ho,'X_hat']):.6f}"
          f"   (기록 0.020987)")
    print(f"  ONNX2 ΔP     : {rmse(df.loc[ho,'DP_reactor_kPa'], df.loc[ho,'DP_hat']):.6f}"
          f"   (기록 0.1910)")
    a = df[df["Campaign"] == ANCHOR]
    print(f"  10/12 앵커   : X̂={a['X_hat'].mean():.6f} (라벨 {a['X_conv_r1_deployed'].mean():.6f}) · "
          f"ΔP̂={a['DP_hat'].mean():.4f} (라벨 {a['DP_reactor_kPa'].mean():.4f}, 기대 29.22)")
    return df


# ============================================ (1) 27점 물리 격자 단조성
def sec1(m1, m2):
    hr("(1) 27점 물리 격자 단조성   [격자는 주어짐 · 예측은 계산]")
    print(f"격자: T{GRID_T} °C × 총유량{GRID_F} kg/h × B/P{GRID_R}  = 27점")
    print("ONNX1 입력 = [T, 총유량/(1+B/P), B/P]   ONNX2 입력 = [T, 총유량, B/P]")

    pts, X1, X2 = [], [], []
    for T in GRID_T:
        for F in GRID_F:
            for R in GRID_R:
                pts.append((T, F, R))
                X1.append([T, F / (1.0 + R), R])
                X2.append([T, F, R])
    xh = m1(X1)
    dh = m2(X2)
    tab = {p: (xh[i], dh[i]) for i, p in enumerate(pts)}

    # --- 범위 검사 ---
    print(f"\n[범위] X̂ ∈ [{xh.min():.6f}, {xh.max():.6f}]  → [0,1] 안: "
          f"{'예' if (xh > 0).all() and (xh < 1).all() else '아니오'}")
    print(f"[범위] ΔP̂ ∈ [{dh.min():.4f}, {dh.max():.4f}] kPa → 전부 양수: "
          f"{'예' if (dh > 0).all() else '아니오'}")

    # --- 축별 단조성 슬라이스 ---
    def slices(axis, model_idx, expect_up, label):
        """axis ∈ {'T','F','R'}. 나머지 두 축의 9조합에 대해 3점 수열 검사."""
        axes = {"T": GRID_T, "F": GRID_F, "R": GRID_R}
        others = [a for a in ["T", "F", "R"] if a != axis]
        ok, fails, deltas = 0, [], []
        for o1 in axes[others[0]]:
            for o2 in axes[others[1]]:
                v = []
                for a in axes[axis]:
                    d = {axis: a, others[0]: o1, others[1]: o2}
                    v.append(tab[(d["T"], d["F"], d["R"])][model_idx])
                inc = v[0] < v[1] < v[2]
                dec = v[0] > v[1] > v[2]
                good = inc if expect_up else dec
                deltas.append((v[2] - v[0]))
                if good:
                    ok += 1
                else:
                    fails.append((others[0], o1, others[1], o2, tuple(round(x, 6) for x in v)))
        arrow = "↑" if expect_up else "↓"
        print(f"  {label:<34} {ok}/9 통과   (축 끝-끝 변화 {arrow} "
              f"평균 {np.mean(deltas):+.6g})")
        for f in fails:
            print(f"      ✗ 실패: {f[0]}={f[1]}, {f[2]}={f[3]} → 값 {f[4]}")
        return ok

    print("\n[ONNX1 전환율] 기대: 온도↑→X↑ · 총유량↑→X↓ · B/P↑→X↑")
    o1 = slices("T", 0, True, "온도↑ → X↑")
    o2 = slices("F", 0, False, "총유량↑ → X↓")
    o3 = slices("R", 0, True, "B/P↑ → X↑")
    print(f"  → ONNX1 합계 {o1+o2+o3}/27")

    print("\n[ONNX2 ΔP] 기대: 총유량↑→ΔP↑ · B/P↑→ΔP↓")
    p1 = slices("F", 1, True, "총유량↑ → ΔP↑")
    p2 = slices("R", 1, False, "B/P↑ → ΔP↓")
    print(f"  → ONNX2 합계 {p1+p2}/18")
    print("  (참고) 온도축은 초안이 기대방향을 명시하지 않음 — 관측만 보고:")
    slices("T", 1, True, "온도↑ → ΔP↑ (참고)")

    # --- 365 °C 외삽: clamp 없이 추세 연장되는가 ---
    print("\n[외삽] 365°C 에서 clamp 되는가 — 335→350 증분과 350→365 증분 비교")
    print(f"  {'총유량':>8}{'B/P':>6}  {'X̂ 335→350':>12}{'X̂ 350→365':>12}   "
          f"{'ΔP̂ 335→350':>13}{'ΔP̂ 350→365':>13}")
    n_clamp = 0
    for F in GRID_F:
        for R in GRID_R:
            xs = [tab[(T, F, R)][0] for T in GRID_T]
            ds = [tab[(T, F, R)][1] for T in GRID_T]
            d1x, d2x = xs[1] - xs[0], xs[2] - xs[1]
            d1d, d2d = ds[1] - ds[0], ds[2] - ds[1]
            if abs(d2x) < 1e-9 or abs(d2d) < 1e-9:
                n_clamp += 1
            print(f"  {F:>8.0f}{R:>6.1f}  {d1x:>12.6f}{d2x:>12.6f}   "
                  f"{d1d:>13.6f}{d2d:>13.6f}")
    print(f"  → 증분이 0 으로 죽은(clamp) 조합: {n_clamp}/9 "
          f"{'(clamp 없음 — 추세 연장)' if n_clamp == 0 else '(⚠ clamp 발생)'}")
    return tab


# ================================================== (2) 에너지수지 상관
def sec2(df):
    hr("(2) 에너지수지 상관   [계산]")
    print("예측 단열온도상승 ∝ X̂ × n_propene / 총질량유량")
    print("  n_propene = FT-1004 × 0.94773 / 42.081 [kmol/h]  (비례상수 ΔH·cp 는 상관에 무관 → 생략)")
    print("  실측 = dT_adiabatic_C = TT-1100 − TT-1006")

    n_p = df["FT-1004.PV"] * W_PROPENE / MW_PROPENE
    df["dT_pred_prop"] = df["X_hat"] * n_p / df["total_flow_kgh"]

    print(f"\n  {'구간':<12}{'n':>6}{'상관 r':>12}   비고")
    r_all = corr(df["dT_pred_prop"], df["dT_adiabatic_C"])
    print(f"  {'전체':<12}{len(df):>6}{r_all:>12.4f}   (10/12 포함)")
    ex = df[df["Campaign"] != ANCHOR]
    print(f"  {'10/12 제외':<12}{len(ex):>6}"
          f"{corr(ex['dT_pred_prop'], ex['dT_adiabatic_C']):>12.4f}")
    ex2 = df[~df["Campaign"].isin([ANCHOR, "10/14"])]
    print(f"  {'10/12·14 제외':<12}{len(ex2):>6}"
          f"{corr(ex2['dT_pred_prop'], ex2['dT_adiabatic_C']):>12.4f}   "
          f"(10/14 는 입력 고정 캠페인 — 흔든 게 없어 상관 정의가 무의미)")
    for c in CAMPS:
        g = df[df["Campaign"] == c]
        r = corr(g["dT_pred_prop"], g["dT_adiabatic_C"])
        note = "상수열 → 정의 불가" if not np.isfinite(r) else ""
        s = f"{r:>12.4f}" if np.isfinite(r) else f"{'—':>12}"
        print(f"  {c:<12}{len(g):>6}{s}   {note}")
    print("\n  ※ 라벨(전환율)이 AT-1100 밀도에서 역산된 값이고 dT 는 TT-1100 에서 온다.")
    print("     둘은 같은 시뮬레이션의 서로 다른 계기이므로 완전 독립 검증은 아니다 [한계].")
    return r_all


# ================================================ (3) 큐멘 물질수지 %
def sec3(df):
    hr("(3) 큐멘 물질수지 오차 (캠페인별)   [계산]")
    print("예측 큐멘 = X̂ × n_propene × 120.196 [kg/h]   vs 실측 FT-1201.PV")

    n_p = df["FT-1004.PV"] * W_PROPENE / MW_PROPENE
    df["cumene_pred"] = df["X_hat"] * n_p * MW_CUMENE
    df["rel_err"] = (df["cumene_pred"] - df["FT-1201.PV"]) / df["FT-1201.PV"] * 100.0

    print(f"\n  {'캠페인':<10}{'n':>6}{'예측평균':>11}{'실측평균':>11}"
          f"{'평균상대오차%':>14}{'평균절대%':>11}{'평균비%':>10}")
    rows = []
    for c in CAMPS:
        g = df[df["Campaign"] == c]
        m_signed = g["rel_err"].mean()
        m_abs = g["rel_err"].abs().mean()
        m_ratio = (g["cumene_pred"].mean() / g["FT-1201.PV"].mean() - 1) * 100
        rows.append((c, len(g), g["cumene_pred"].mean(), g["FT-1201.PV"].mean(),
                     m_signed, m_abs, m_ratio))
        print(f"  {c:<10}{len(g):>6}{g['cumene_pred'].mean():>11.1f}"
              f"{g['FT-1201.PV'].mean():>11.1f}{m_signed:>14.3f}{m_abs:>11.3f}{m_ratio:>10.3f}")
    allsig = df["rel_err"].mean()
    print(f"  {'전체':<10}{len(df):>6}{df['cumene_pred'].mean():>11.1f}"
          f"{df['FT-1201.PV'].mean():>11.1f}{allsig:>14.3f}"
          f"{df['rel_err'].abs().mean():>11.3f}"
          f"{(df['cumene_pred'].mean()/df['FT-1201.PV'].mean()-1)*100:>10.3f}")
    worst = max(rows, key=lambda r: abs(r[4]))
    print(f"\n  캠페인 평균상대오차의 최대 절대값: {abs(worst[4]):.3f}% ({worst[0]})"
          f"  → ±3% 기준 {'충족' if abs(worst[4]) <= 3 else '초과'}")

    # --- 모델 탓인가, 수지 자체가 안 닫히는가 ---
    print("\n  [분리 검사] 모델 예측 X̂ 대신 **라벨 X 그대로** 넣으면 오차가 사라지는가?")
    df["cumene_label"] = df["X_conv_r1_deployed"] * n_p * MW_CUMENE
    df["rel_err_label"] = (df["cumene_label"] - df["FT-1201.PV"]) / df["FT-1201.PV"] * 100.0
    print(f"    {'캠페인':<10}{'예측 X̂ 기준 %':>16}{'라벨 X 기준 %':>16}{'차이(모델 몫)':>15}")
    for c in CAMPS:
        g = df[df["Campaign"] == c]
        a, b = g["rel_err"].mean(), g["rel_err_label"].mean()
        print(f"    {c:<10}{a:>16.3f}{b:>16.3f}{a-b:>15.3f}")
    a, b = df["rel_err"].mean(), df["rel_err_label"].mean()
    print(f"    {'전체':<10}{a:>16.3f}{b:>16.3f}{a-b:>15.3f}")
    print("    → 라벨을 그대로 써도 오차가 남으면, 그것은 ML 모델 오차가 아니라")
    print("       반응기~FT-1201 사이 수지가 닫히지 않는다는 뜻이다(공정 쪽 원인).")

    # --- 남은 차이를 '공급 큐멘'으로 설명하려면 몇 mol% 여야 하는가 ---
    MW_BENZENE = 78.114   # g/mol [주어짐] 문헌값
    MW_PROPANE = 44.096   # g/mol [주어짐] 문헌값 (프로펜 스트림 잔부를 프로판으로 가정 [가정])
    print("\n  [원인 후보 정량화] 남은 차이를 '공급에 섞여 통과한 큐멘'으로 설명하려면?")
    print(f"    {'캠페인':<10}{'차이 kg/h':>11}{'= kmol/h':>10}{'공급몰유량 kmol/h':>18}{'필요 mol%':>11}")
    for c in CAMPS:
        g = df[df["Campaign"] == c]
        gap = (g["FT-1201.PV"] - g["cumene_label"]).mean()          # 실측 − 라벨기반 생성량
        n_gap = gap / MW_CUMENE
        n_feed = (g["FT-1004.PV"] * W_PROPENE / MW_PROPENE
                  + g["FT-1004.PV"] * (1 - W_PROPENE) / MW_PROPANE
                  + g["FT-1003.PV"] / MW_BENZENE).mean()
        print(f"    {c:<10}{gap:>11.1f}{n_gap:>10.3f}{n_feed:>18.1f}{100*n_gap/n_feed:>11.3f}")
    print("    대조: JM_0728...simx 단일 운전점의 반응기 입구 z[CUMENE] = 0.004306 (=0.431 mol%)")
    print("          [ONNX2_사전분석_DP데이터.md §2-3 기록]. 필요 mol% 가 이 값과 같은 자릿수면")
    print("          '재순환 큐멘 통과'가 유력한 설명이다(정밀 확인은 별도 필요).")

    # 하류 지연 정량화 — 예측이 FT-1201 보다 몇 분 앞서는가
    print("\n  [한계] FT-1201 은 증류탑 하류 계기라 체류시간 지연이 있다. 지연 스캔:")
    print(f"    {'캠페인':<10}{'최대상관 지연 k(분)':>20}{'그때 r':>10}{'k=0 의 r':>10}")
    for c in CAMPS:
        g = df[df["Campaign"] == c]
        p = g["cumene_pred"].to_numpy(float); o = g["FT-1201.PV"].to_numpy(float)
        best_k, best_r = None, -2.0
        r0 = corr(p, o)
        if np.isfinite(r0):
            for k in range(0, 61):
                r = corr(p[:len(p) - k] if k else p, o[k:])
                if np.isfinite(r) and r > best_r:
                    best_r, best_k = r, k
            print(f"    {c:<10}{best_k:>20d}{best_r:>10.4f}{r0:>10.4f}")
        else:
            print(f"    {c:<10}{'—':>20}{'—':>10}{'—':>10}  (상수열)")
    return rows


# ============================================== (4) 잔차 자기상관
def acf_within(res, camp, lags):
    """캠페인 경계를 넘지 않는 쌍만 모아 지연별 Pearson 상관."""
    out = {}
    for k in lags:
        A, B = [], []
        for c in pd.unique(camp):
            r = res[camp == c]
            if np.ptp(r) <= 1e-12 * max(abs(np.mean(r)), 1.0):
                continue                       # 상수 잔차(10/12) 제외
            if len(r) > k:
                A.append(r[:len(r) - k] if k else r); B.append(r[k:])
        if not A:
            out[k] = np.nan; continue
        out[k] = corr(np.concatenate(A), np.concatenate(B))
    return out


def sec4(df):
    hr("(4) 배포 ONNX 잔차의 자기상관 (1분·60분)   [계산]")
    print("잔차 = 실측 − 배포 ONNX 예측. 캠페인 내부에서만 지연쌍을 만든다.")
    df["res_X"] = df["X_conv_r1_deployed"] - df["X_hat"]
    df["res_DP"] = df["DP_reactor_kPa"] - df["DP_hat"]
    lags = [1, 5, 30, 60, 120]

    for col, name in [("res_X", "전환율 잔차"), ("res_DP", "ΔP 잔차")]:
        ex = df[df["Campaign"] != ANCHOR]
        tr = df[df["Campaign"].isin(["10/13", "10/14", "10/15", "10/16"])]
        ho = df[df["Campaign"] == "10/17-18"]
        a_all = acf_within(ex[col].to_numpy(float), ex["Campaign"].to_numpy(), lags)
        a_tr = acf_within(tr[col].to_numpy(float), tr["Campaign"].to_numpy(), lags)
        a_ho = acf_within(ho[col].to_numpy(float), ho["Campaign"].to_numpy(), lags)
        print(f"\n  [{name}]  (10/12 제외 — 잔차가 상수라 정의 불가)")
        print(f"    {'구간':<12}" + "".join(f"{'lag'+str(k):>10}" for k in lags))
        print(f"    {'전체':<12}" + "".join(f"{a_all[k]:>10.4f}" for k in lags))
        print(f"    {'학습 13-16':<12}" + "".join(f"{a_tr[k]:>10.4f}" for k in lags))
        print(f"    {'held-out':<12}" + "".join(f"{a_ho[k]:>10.4f}" for k in lags))
        for c in CAMPS:
            g = df[df["Campaign"] == c]
            a = acf_within(g[col].to_numpy(float), g["Campaign"].to_numpy(), lags)
            row = "".join(f"{a[k]:>10.4f}" if np.isfinite(a[k]) else f"{'—':>10}" for k in lags)
            print(f"    {c:<12}" + row)
        print(f"    잔차 표준편차(10/12 제외): {ex[col].std(ddof=0):.6f}")
    print("\n  ※ 유효 독립표본은 행 수가 아니라 자기상관 감쇠 길이로 정해진다.")
    print("     행 수(3,066)를 신뢰도 근거로 인용하지 말 것.")


def main():
    df = load()
    m1, m2 = Model(_ONNX1), Model(_ONNX2)
    df = sec0(df, m1, m2)
    print_coefs()
    sec0b(df)
    sec1(m1, m2)
    sec1b(df)
    sec2(df)
    sec3(df)
    sec4(df)
    hr("완료 — 주장값 대조는 docs/검증_37_재현.md")


if __name__ == "__main__":
    main()
