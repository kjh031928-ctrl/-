"""stage5_figures.py — 보고서 그림 3종 생성 (신설 2026-08-03).

그림1 예측 vs 실측 산점도(검증셋, 최종 모델)
그림2 캠페인별 시계열 잔차(학습 in-sample + 검증 held-out)
그림3 케이스스터디 27점 반응면(T·총유량·B/P → 큐멘 kg/h)

최종 모델 = 로짓 1차 선형(Linear1/logit, 현 배포 레시피), 학습 10/13-16 로 적합.
색: Okabe-Ito(색맹 안전). 폰트 Malgun Gothic. 얇은 마크·약한 그리드·이중축 없음.
"""
from __future__ import annotations
import sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError): pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluation as ev
import data_loader as dl

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.edgecolor": "#666666", "axes.linewidth": 0.8,
})
OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermil": "#D55E00",
      "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#222222"}
DOC = Path(__file__).resolve().parents[1] / "docs"
MW = 120.196; MFRAC = 0.94773; MWP = 42.081

def _fit_final():
    from sklearn.linear_model import LinearRegression
    spec = ev.ModelSpec("Linear1", "선형", lambda p: LinearRegression(), [{}],
                        transform=ev.TRANSFORMS["logit"])
    sp = ev.load_splits("I1", "r1")
    est = ev.fit_on_logit(spec, {}, sp.X_train, sp.y_train)
    return est, sp


def fig1_pred_actual():
    est, sp = _fit_final()
    yp = ev.predict_conv(est, sp.X_valid); yt = sp.y_valid
    m = ev.conversion_metrics(yt, yp)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo, hi = 0.28, 0.90
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=1.2, ls="--", zorder=1)
    ax.scatter(yt, yp, s=14, color=OK["blue"], alpha=0.55, edgecolor="none", zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("실측 전환율 X (검증셋 10/17–18)")
    ax.set_ylabel("모델 예측 전환율 X")
    ax.set_title("그림1. 예측 vs 실측 (최종 모델, 검증셋)", fontsize=12, pad=10)
    ax.text(0.04, 0.96, f"RMSE = {m['rmse']:.4f}\nR² = {m['r2']:.4f}\nn = {len(yt)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
    ax.text(0.96, 0.06, "점선 = 완벽 예측선(y=x)", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=9, color="#777777")
    fig.tight_layout(); fig.savefig(DOC/"그림1_예측대실측.png", bbox_inches="tight"); plt.close(fig)
    print("그림1 저장 (RMSE %.4f, R2 %.4f)" % (m["rmse"], m["r2"]))


def fig2_campaign_residuals():
    est, sp = _fit_final()
    cfg = dl.load_config(); df = dl.load_data(cfg)
    use = df[df.Campaign != "10/12"].copy().reset_index(drop=True)
    X = use[["TT-1006.PV", "FT-1004.PV", "FRC-1004.PV"]].to_numpy(float)
    yt = use["X_propene_recycle1"].to_numpy(float)
    resid = ev.predict_conv(est, X) - yt
    camps = ["10/13", "10/14", "10/15", "10/16", "10/17", "10/18"]
    use["day"] = np.where(use.Campaign == "10/17-18",
                          np.where(np.arange(len(use)) < 0, "", ""), use.Campaign)
    # Campaign 컬럼은 '10/17-18' 로 합쳐져 있으니 Timestamp 로 10/17·10/18 구분
    import pandas as pd
    day = pd.to_datetime(use["Timestamp"]).dt.strftime("%m/%d")
    colors = {"10/13": OK["blue"], "10/14": OK["sky"], "10/15": OK["green"],
              "10/16": OK["yellow"], "10/17": OK["orange"], "10/18": OK["vermil"]}
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    x = np.arange(len(use))
    for d in ["10/13", "10/14", "10/15", "10/16", "10/17", "10/18"]:
        mask = (day == d).to_numpy()
        if mask.sum() == 0: continue
        ax.scatter(x[mask], resid[mask], s=7, color=colors[d], alpha=0.7,
                   edgecolor="none", label=d)
    # held-out 구간(10/17-18) 음영
    hv = np.where(np.isin(day.to_numpy(), ["10/17", "10/18"]))[0]
    if len(hv): ax.axvspan(hv.min(), hv.max(), color="#f2f2f2", zorder=0)
    ax.axhline(0, color="#999999", lw=1.0)
    ax.set_xlabel("시간 순서 (10/13 → 10/18, 각 캠페인 연속)")
    ax.set_ylabel("잔차 (예측 빼기 실측 X)")
    ax.set_title("그림2. 캠페인별 시계열 잔차  (음영 = 검증 held-out 10/17–18)", fontsize=12, pad=8)
    ax.legend(loc="upper left", ncol=6, fontsize=8.5, frameon=False,
              handletextpad=0.2, columnspacing=0.9, markerscale=1.6)
    ax.margins(x=0.01)
    fig.tight_layout(); fig.savefig(DOC/"그림2_캠페인잔차.png", bbox_inches="tight"); plt.close(fig)
    print("그림2 저장 (잔차 표준편차 %.4f)" % resid.std())


def fig3_response_surface():
    est, _ = _fit_final()
    Ts = [335.0, 350.0, 365.0]; totals = [15000.0, 16500.0, 18000.0]; bps = [4.0, 5.0, 6.0]
    # 온도=순차(연한→진한 파랑), 색 3단계
    tcol = {335.0: "#9ecae1", 350.0: "#4292c6", 365.0: "#08519c"}
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7), sharey=True)
    for ax, bp in zip(axes, bps):
        for T in Ts:
            ys = []
            for tot in totals:
                ft = tot/(1+bp); frc = bp
                xp = ev.predict_conv(est, np.array([[T, ft, frc]]))[0]
                ys.append(xp * ft*MFRAC/MWP * MW)
            ax.plot(totals, ys, "-o", color=tcol[T], lw=2, ms=6, label=f"{T:.0f}°C")
        ax.set_title(f"B/P = {bp:.0f}", fontsize=11)
        ax.set_xlabel("총 질량유량 [kg/h]")
        ax.set_xticks(totals); ax.set_xticklabels(["15000","16500","18000"], fontsize=9)
    axes[0].set_ylabel("큐멘 생산량 [kg/h]")
    axes[-1].legend(title="반응기 입구온도", loc="lower right", fontsize=9, frameon=False)
    # 외삽 경고: 365°C 는 학습 최대 349.977°C 초과
    fig.suptitle("그림3. 케이스스터디 27점 반응면  (365°C는 학습범위 밖 외삽)",
                 fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(DOC/"그림3_반응면.png", bbox_inches="tight"); plt.close(fig)
    print("그림3 저장")


if __name__ == "__main__":
    fig1_pred_actual(); fig2_campaign_residuals(); fig3_response_surface()
    print("완료: docs/그림1~3 .png")
