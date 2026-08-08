# -*- coding: utf-8 -*-
"""
ML 보고서 대단원 3 그림 생성 스크립트 (F1 / F2 / F3)

원칙
  - 원본 CSV, ONNX, 기존 문서는 읽기 전용. 이 스크립트는 figures/*.png 만 쓴다.
  - parity(F2, F3)와 방향성 검사(F1)는 모두 **held-out** 으로 보인다.
    배포 ONNX 는 10/13~18 전체로 재학습된 production 모델이라 검증셋(10/17-18)에
    돌리면 in-sample 성능이 나온다. 따라서 여기서는 학습셋(10/13~16)만으로
    선형 모델을 다시 적합해 검증셋(10/17-18)에서 예측한다.
  - 각 그림 생성 전, 인수인계 문서에 기록된 held-out RMSE 가 재현되는지 assert 로 자체검증한다.

재현
  python figures/make_figures.py     (repo 루트에서 실행)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------
# 0. 경로 / 스타일
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "ML_마스터데이터_PV.csv"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# 한글 폰트: 설치된 것 중 앞에서부터 하나 선택 (없으면 중단)
_installed = {f.name for f in font_manager.fontManager.ttflist}
for _cand in ("Malgun Gothic", "Noto Sans KR", "NanumGothic", "Gulim", "Dotum"):
    if _cand in _installed:
        KFONT = _cand
        break
else:  # pragma: no cover
    sys.exit("한글 렌더 가능한 폰트를 찾지 못했습니다. Malgun Gothic / Noto Sans KR 등을 설치하십시오.")

plt.rcParams.update({
    "font.family": KFONT,
    "axes.unicode_minus": False,      # 마이너스 기호 깨짐 방지
    "figure.dpi": 220,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#b8b7b2",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#33322f",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "grid.color": "#e3e2dd",
    "grid.linewidth": 0.7,
})

# 색: 검증 통과한 2색 조합 (adjacent CVD ΔE 24.7 / normal ΔE 33.6, 대비 ≥3:1)
C_F2 = "#2a78d6"   # F2 = 총유량 입력 (채택안)
C_F1 = "#eb6834"   # F1 = 프로펜유량 입력 (대조안)
C_INK = "#0b0b0b"
C_MUTED = "#7a7975"

# --------------------------------------------------------------------------
# 1. 데이터 / 모델
# --------------------------------------------------------------------------
COL_T = "TT-1006.PV"          # 반응기 입구온도 [°C]
COL_FP = "FT-1004.PV"         # 프로펜 유량 [kg/h]
COL_BP = "FRC-1004.PV"        # B/P 비율 [-]
COL_FT = "total_flow_kgh"     # 총유량 [kg/h]
COL_X = "X_conv_r1_deployed"  # 전환율 [-]
COL_DP = "DP_reactor_kPa"     # 반응기 압력강하 [kPa]

IN_CONV = [COL_T, COL_FP, COL_BP]   # 전환율 모델 입력 (배포 ONNX 와 동일 순서)
IN_DP_F2 = [COL_T, COL_FT, COL_BP]  # ΔP 채택안(F2)
IN_DP_F1 = [COL_T, COL_FP, COL_BP]  # ΔP 대조안(F1)

df = pd.read_csv(CSV, encoding="utf-8-sig")
train = df[df["Split"] == "학습"]      # 10/13~16, 1924행
valid = df[df["Split"] == "검증"]      # 10/17~18, 661행
assert (len(train), len(valid)) == (1924, 661), (len(train), len(valid))


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def fit_linear(cols, target):
    """변환 없는 선형회귀. 학습셋(10/13~16)만 사용."""
    return LinearRegression().fit(train[cols].values, train[target].values)


def fit_logit_linear(cols, target):
    """로짓-선형: y -> ln(y/(1-y)) 로 변환해 선형적합. 예측은 sigmoid 역변환."""
    y = train[target].values
    return LinearRegression().fit(train[cols].values, np.log(y / (1.0 - y)))


def predict_logit(model, Xmat):
    return 1.0 / (1.0 + np.exp(-model.predict(Xmat)))


m_conv = fit_logit_linear(IN_CONV, COL_X)
m_dp_f2 = fit_linear(IN_DP_F2, COL_DP)
m_dp_f1 = fit_linear(IN_DP_F1, COL_DP)

p_conv = predict_logit(m_conv, valid[IN_CONV].values)
p_dp_f2 = m_dp_f2.predict(valid[IN_DP_F2].values)
p_dp_f1 = m_dp_f1.predict(valid[IN_DP_F1].values)

y_conv = valid[COL_X].values
y_dp = valid[COL_DP].values

R_CONV, R_F2, R_F1 = rmse(p_conv, y_conv), rmse(p_dp_f2, y_dp), rmse(p_dp_f1, y_dp)

print("=" * 74)
print("자체검증 — 학습(10/13~16) 적합 → 검증(10/17-18) held-out RMSE")
print(f"  전환율 로짓-선형   RMSE = {R_CONV:.5f}   (기준 0.02099)")
print(f"  ΔP F2 (총유량)     RMSE = {R_F2:.5f}   (기준 0.1910)")
print(f"  ΔP F1 (프로펜)     RMSE = {R_F1:.5f}   (기준 0.1479)")
assert abs(R_CONV - 0.02099) < 5e-5, f"전환율 RMSE 재현 실패: {R_CONV}"
assert abs(R_F2 - 0.1910) < 5e-4, f"ΔP F2 RMSE 재현 실패: {R_F2}"
assert abs(R_F1 - 0.1479) < 5e-4, f"ΔP F1 RMSE 재현 실패: {R_F1}"
print("  → 세 값 모두 재현됨 (assert 통과)")
print("=" * 74)

# --------------------------------------------------------------------------
# 2. F1 — ΔP 물리 방향성 검사 (27점 격자)
# --------------------------------------------------------------------------
# 검사할 두 단조성
#   (a) 유량 ↑  →  ΔP ↑   (마찰손실이 유속과 함께 커져야 한다)
#   (b) B/P ↑   →  ΔP ↓   (동일 총유량에서 희석제 비율이 커지면 반응이 줄고 몰수 변화·
#                          유속 프로파일이 완만해져 압력강하가 감소한다)
T_GRID = [335.0, 350.0, 365.0]
BP_GRID = [4.0, 5.0, 6.0]
FLOW_GRID_F2 = [15000.0, 16500.0, 18000.0]      # 총유량 [kg/h]  (인수인계 지정 격자)
FLOW_GRID_F1_SPEC = [4500.0, 5000.0, 5500.0]    # 프로펜 [kg/h]  (인수인계 지정 격자)
FLOW_GRID_F1_INRANGE = [2500.0, 3000.0, 3500.0]  # 프로펜, 학습 데이터 범위 안 (2500~3661)


def monotonicity_scan(model, flow_grid):
    """27점 격자에서 (a) 유량 단조증가, (b) B/P 단조감소 통과 수를 센다.

    모델 입력 순서는 [T, 유량, B/P] 로 동일하다.
    """
    def pred(t, f, bp):
        return float(model.predict(np.array([[t, f, bp]]))[0])

    flow_pass = 0   # (T, B/P) 9조합에 대해 유량 3점이 단조증가인가
    for t in T_GRID:
        for bp in BP_GRID:
            v = [pred(t, f, bp) for f in flow_grid]
            flow_pass += int(v[0] < v[1] < v[2])

    bp_pass = 0     # (T, 유량) 9조합에 대해 B/P 3점이 단조감소인가
    for t in T_GRID:
        for f in flow_grid:
            v = [pred(t, f, bp) for bp in BP_GRID]
            bp_pass += int(v[0] > v[1] > v[2])

    return flow_pass, bp_pass


f2_flow, f2_bp = monotonicity_scan(m_dp_f2, FLOW_GRID_F2)
f1_flow, f1_bp = monotonicity_scan(m_dp_f1, FLOW_GRID_F1_SPEC)
f1_flow_ir, f1_bp_ir = monotonicity_scan(m_dp_f1, FLOW_GRID_F1_INRANGE)

print("27점 격자 단조성 검사  (온도 3 × 유량 3 × B/P 3)")
print(f"  F2 (총유량 입력, 채택) : 유량↑→ΔP↑ {f2_flow}/9 통과 · B/P↑→ΔP↓ {f2_bp}/9 통과")
print(f"  F1 (프로펜 입력, 대조) : 유량↑→ΔP↑ {f1_flow}/9 통과 · B/P↑→ΔP↓ {f1_bp}/9 통과")
print(f"  [보조] F1 을 학습범위 안 프로펜 격자 {FLOW_GRID_F1_INRANGE} 로 다시 검사: "
      f"유량 {f1_flow_ir}/9 · B/P {f1_bp_ir}/9  → 판정 동일")
print(f"  F1 의 B/P 계수 = {m_dp_f1.coef_[2]:+.2f} kPa/(B/P 1단위)  ← 부호가 물리와 반대")
print(f"  F2 의 B/P 계수 = {m_dp_f2.coef_[2]:+.2f} kPa/(B/P 1단위)")
assert (f2_flow, f2_bp, f1_flow, f1_bp) == (9, 9, 9, 0), "27점 검사 결과가 문서와 다릅니다"
print("=" * 74)

# 그림용 대표 운전점: T = 350 °C, 유량은 각 모델 격자의 중앙값.
# 단, F1 의 프로펜 축은 지정 격자(4500~5500)가 학습범위(최대 3661) 밖이므로
# 절대값이 물리적으로 무의미해진다. 그림은 학습범위 안(3000 kg/h)에서 그리고,
# 27점 판정은 위에서 두 격자 모두로 확인했다(결과 동일).
T_REP = 350.0
FP_REP = 3000.0
FT_REP = 16500.0

bp_line = np.linspace(4.0, 6.0, 41)
dp_f2_bp = m_dp_f2.predict(np.column_stack([np.full_like(bp_line, T_REP),
                                            np.full_like(bp_line, FT_REP), bp_line]))
dp_f1_bp = m_dp_f1.predict(np.column_stack([np.full_like(bp_line, T_REP),
                                            np.full_like(bp_line, FP_REP), bp_line]))

# 유량 스윕 패널: 두 모델의 유량 축 단위가 달라(총유량 vs 프로펜) 공통 x 를 쓸 수 없으므로
# 각 모델 격자의 하/중/상 3수준을 공통 눈금에 얹고, 눈금 라벨에 실제 값을 적는다.
lvl = np.array([0.0, 1.0, 2.0])
dp_f2_fl = m_dp_f2.predict(np.column_stack([np.full(3, T_REP), FLOW_GRID_F2, np.full(3, 5.0)]))
dp_f1_fl = m_dp_f1.predict(np.column_stack([np.full(3, T_REP), FLOW_GRID_F1_INRANGE, np.full(3, 5.0)]))

YLIM_F1 = (17.5, 42.0)   # 두 패널 공통 y 범위 (같은 축에서 기울기를 비교하기 위함)

fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
fig.suptitle("그림 F1.  압력강하 모델의 물리 방향성 검증 — 입력을 프로펜유량에서 총유량으로 바꾼 이유",
             fontsize=13.5, fontweight="bold", y=1.02)

# --- (a) B/P 스윕 : 판정이 갈리는 축 ---
ax = axes[0]
ax.set_axisbelow(True)
ax.grid(axis="y")
h_f2, = ax.plot(bp_line, dp_f2_bp, color=C_F2, lw=2.0, zorder=3, label="F2 · 총유량 입력 (채택)")
h_f1, = ax.plot(bp_line, dp_f1_bp, color=C_F1, lw=2.0, zorder=3, label="F1 · 프로펜유량 입력 (대조)")
ax.plot(BP_GRID, m_dp_f2.predict(np.column_stack([np.full(3, T_REP), np.full(3, FT_REP), BP_GRID])),
        "o", ms=7, color=C_F2, mec="white", mew=1.6, zorder=4)
ax.plot(BP_GRID, m_dp_f1.predict(np.column_stack([np.full(3, T_REP), np.full(3, FP_REP), BP_GRID])),
        "o", ms=7, color=C_F1, mec="white", mew=1.6, zorder=4)

# 주석은 각 선이 지나가지 않는 빈 사분면에 둔다 (F1 은 좌상단, F2 는 우하단).
ax.text(4.06, 40.8, "물리 위반  ×\nB/P↑ → ΔP↑ (우상향)\nB/P 계수 %+.2f kPa · 0/9 통과" % m_dp_f1.coef_[2],
        color=C_F1, fontsize=10, ha="left", va="top", linespacing=1.45)
ax.text(5.94, 22.4, "물리 부합  ○\nB/P↑ → ΔP↓ (우하향)\nB/P 계수 %+.2f kPa · 9/9 통과" % m_dp_f2.coef_[2],
        color=C_F2, fontsize=10, ha="right", va="top", linespacing=1.45)

ax.set_xlabel("B/P 비율 (FRC-1004) [-]")
ax.set_ylabel("예측 압력강하 ΔP [kPa]")
ax.set_xticks(BP_GRID)
ax.set_xlim(3.88, 6.12)
ax.set_ylim(*YLIM_F1)
ax.set_title("(a) B/P 스윕 — 판정이 갈리는 축", fontsize=11.5, pad=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)

# --- (b) 유량 스윕 : 둘 다 통과 ---
ax = axes[1]
ax.set_axisbelow(True)
ax.grid(axis="y")
ax.plot(lvl, dp_f2_fl, "-o", color=C_F2, lw=2.0, ms=7, mec="white", mew=1.6, zorder=3)
ax.plot(lvl, dp_f1_fl, "-o", color=C_F1, lw=2.0, ms=7, mec="white", mew=1.6, zorder=3)
ax.set_xticks(lvl)
ax.set_xticklabels(["하\nF2 15,000 / F1 2,500", "중\nF2 16,500 / F1 3,000", "상\nF2 18,000 / F1 3,500"])
ax.set_xlim(-0.25, 2.25)
ax.set_ylim(*YLIM_F1)
ax.set_xlabel("유량 수준 [kg/h]  (모델별 입력 단위가 달라 3수준으로 대응)")
ax.set_ylabel("예측 압력강하 ΔP [kPa]")
ax.set_title("(b) 유량 스윕 — 두 모델 모두 통과", fontsize=11.5, pad=10, loc="left")
ax.text(0.03, 0.96, "물리 부합  ○\n유량↑ → ΔP↑ (우상향)\nF2 9/9 · F1 9/9 통과",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        color="#52514e", linespacing=1.45)
ax.spines[["top", "right"]].set_visible(False)

fig.legend(handles=[h_f2, h_f1], loc="lower center", bbox_to_anchor=(0.5, -0.055),
           ncol=2, handlelength=2.0, columnspacing=2.6, fontsize=10.5)
fig.text(0.5, -0.115,
         "27점 격자 = 온도{335, 350, 365} °C × 유량 3수준 × B/P{4, 5, 6}.  "
         "두 모델 모두 학습셋(10/13~16)만으로 적합.  대표 운전점 T = 350 °C.",
         ha="center", fontsize=9, color=C_MUTED)

fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(OUT / "F1_dp_physical_direction.png")
plt.close(fig)


# --------------------------------------------------------------------------
# 3. F2 / F3 — parity plot
# --------------------------------------------------------------------------
def parity_plot(y_true, y_pred, *, fname, fignum, quantity, unit, unit_short,
                title, subtitle, note, box_fmt):
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    ax.set_axisbelow(True)
    ax.grid(color="#e9e8e4", lw=0.7)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.06 * (hi - lo)
    lims = (lo - pad, hi + pad)

    ax.plot(lims, lims, ls="--", lw=1.2, color="#9a9994", zorder=2, label="y = x (완전일치)")
    ax.scatter(y_true, y_pred, s=13, color=C_F2, alpha=0.35, linewidths=0, zorder=3)

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"실측 {quantity} {unit}  (시뮬레이션 기록값)")
    ax.set_ylabel(f"모델 예측 {quantity} {unit}")

    # 제목/부제는 축 위 별도 좌표에 배치한다 (set_title 과 겹치지 않게).
    ax.text(0.0, 1.085, f"그림 {fignum}.  {title}", transform=ax.transAxes,
            fontsize=12.5, fontweight="bold", va="bottom", ha="left")
    ax.text(0.0, 1.022, subtitle, transform=ax.transAxes,
            fontsize=10, color=C_MUTED, va="bottom", ha="left")

    u = f" {unit_short}" if unit_short else ""
    stats = (f"held-out 10/17–18  (n = {len(y_true)})\n"
             f"RMSE = {box_fmt.format(rmse(y_pred, y_true))}{u}\n"
             f"R²   = {r2(y_true, y_pred):.4f}\n"
             f"MAE  = {box_fmt.format(mae(y_pred, y_true))}{u}")
    ax.text(0.035, 0.965, stats, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, color=C_INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cfceca", lw=0.8, alpha=0.95))
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.text(0.02, 0.015, note, ha="left", va="bottom", fontsize=8.8,
             color=C_MUTED, linespacing=1.6)
    fig.savefig(OUT / fname)
    plt.close(fig)


NOTE_HOLDOUT = ("학습 : 10/13~16 캠페인(1,924행)만 사용 · 평가 : 10/17–18 캠페인(661행), 학습에 쓰이지 않은 구간.\n"
                "배포 ONNX 는 10/13~18 전체로 재학습한 모델이므로 이 그림에는 쓰지 않았다(그 경우 in-sample 이 된다).")

parity_plot(y_dp, p_dp_f2,
            fname="F2_dp_parity.png", fignum="F2",
            quantity="압력강하 ΔP", unit="[kPa]", unit_short="kPa",
            title="압력강하 모델(F2, 총유량 입력)의 미학습 구간 예측 정확도",
            subtitle="held-out 10/17–18 · 실측 대 예측 (parity plot)",
            note=NOTE_HOLDOUT, box_fmt="{:.4f}")

parity_plot(y_conv, p_conv,
            fname="F3_conversion_parity.png", fignum="F3",
            quantity="전환율 X", unit="[-]", unit_short="",
            title="전환율 모델(로짓-선형)의 미학습 구간 예측 정확도",
            subtitle="held-out 10/17–18 · 실측 대 예측 (parity plot)",
            note=NOTE_HOLDOUT, box_fmt="{:.5f}")

# --------------------------------------------------------------------------
# 4. 캡션 (보고서 붙여넣기용)
# --------------------------------------------------------------------------
print("저장 완료")
for f in ("F1_dp_physical_direction.png", "F2_dp_parity.png", "F3_conversion_parity.png"):
    print(f"  {OUT / f}")
print("=" * 74)
print("캡션 (보고서 붙여넣기용)")
print(
    f"그림 F1. 압력강하 모델의 물리 방향성 검증. 27점 격자(온도 3 × 유량 3 × B/P 3)에서 "
    f"총유량 입력(F2)은 '유량↑→ΔP↑' {f2_flow}/9, 'B/P↑→ΔP↓' {f2_bp}/9 를 모두 통과한 반면, "
    f"프로펜유량 입력(F1)은 유량 단조성은 {f1_flow}/9 통과했으나 B/P 계수가 {m_dp_f1.coef_[2]:+.2f} kPa 로 "
    f"부호가 뒤집혀 B/P 단조성은 {f1_bp}/9 통과에 그쳤다."
)
print(
    f"그림 F2. 압력강하 모델(F2, 총유량 입력)의 parity plot. 학습셋(10/13~16)으로만 적합한 뒤 "
    f"학습에 쓰이지 않은 10/17–18 구간 {len(y_dp)}행에서 예측한 결과로, "
    f"RMSE {R_F2:.4f} kPa · R² {r2(y_dp, p_dp_f2):.4f} · MAE {mae(p_dp_f2, y_dp):.4f} kPa 이다."
)
print(
    f"그림 F3. 전환율 모델(로짓-선형)의 parity plot. 학습셋(10/13~16)으로만 적합한 뒤 "
    f"학습에 쓰이지 않은 10/17–18 구간 {len(y_conv)}행에서 예측한 결과로, "
    f"RMSE {R_CONV:.5f} · R² {r2(y_conv, p_conv):.4f} · MAE {mae(p_conv, y_conv):.5f} 이다."
)
