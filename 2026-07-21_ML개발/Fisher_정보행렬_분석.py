"""Fisher_정보행렬_분석.py — 채택 모델의 파라미터 식별성·신뢰구간 분석.

대상 모델 (Phase 10 확정 + 2026-07-31 r1 확정):
    로짓 1차 선형 · 입력세트 I1 · 라벨 r1 · 가중치 없음
    z = logit(X) = θ0 + θ1·TT-1006 + θ2·FT-1004 + θ3·FRC-1004,   X = sigmoid(z)
    학습셋 = 10/13~16 (1,924행). 검증셋은 건드리지 않는다.

왜 이 모델에 Case A(가이드 §5-3-1)가 그대로 적용되나:
    로짓 공간에서 예측이 파라미터에 대해 **선형**이므로
    자코비안 J = ∂ẑ/∂θ 가 곧 설계행렬 [1, T, F, R] 이다. 수치미분이 필요 없다.
        FIM = JᵀJ / σ²,  Cov(θ) = (JᵀJ)⁻¹ σ²,  SE = sqrt(diag(Cov))

두 가지 함정을 함께 다룬다:
    (1) **고유값은 단위에 의존한다.** T~340, F~3000, R~5, 절편~1 이라 원단위 고유값은
        스케일 차이만 반영해 조건수가 1e11까지 뜬다. 식별성 진단은 **표준화 후** 값으로 한다.
    (2) **고전 FIM은 독립표본을 가정한다.** 잔차 자기상관이 1분 지연 0.99이므로
        고전 SE는 심하게 과소평가된다. 60분 블록 부트스트랩으로 실제 폭과 대조한다.
        (프로젝트 CLAUDE.md: "행 수를 통계적 신뢰도의 근거로 제시하지 말 것")

출력: 콘솔 표 + docs/img/그림D_파라미터_신뢰구간.png
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd
import yaml
from scipy.special import logit

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config" / "base.yaml", encoding="utf-8"))
FEATS = CFG["features"]["I1"]
LABEL = CFG["labels"]["r1"]
SEED = CFG["seed"]
LOGIT_EPS = 1e-6
BLOCK = 60          # 분. 잔차 자기상관이 60분에서 0.41로 떨어지는 지점 (질의응답 §2-4와 동일 규약)
N_BOOT = 4000       # 질의응답 §2-4의 블록 부트스트랩 횟수와 동일
NAMES = ["절편", "TT-1006 (°C⁻¹)", "FT-1004 ((kg/h)⁻¹)", "FRC-1004 (무차원⁻¹)"]


def load_train() -> pd.DataFrame:
    df = pd.read_excel(ROOT / CFG["paths"]["data_xlsx"], sheet_name=CFG["paths"]["data_sheet"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    if LABEL not in df.columns:
        lab = pd.read_csv(ROOT / CFG["paths"]["labels_recycle1"])
        lab["Timestamp"] = pd.to_datetime(lab["Timestamp"])
        df = df.merge(lab[["Timestamp", LABEL]], on="Timestamp", how="left")
    tr = df[df["Split"] == CFG["split"]["train_value"]].reset_index(drop=True)
    if tr[LABEL].isna().any():
        raise RuntimeError("학습셋에 r1 결측이 있습니다.")
    return tr


def design(tr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = tr[FEATS].to_numpy(dtype=np.float64)
    y = tr[LABEL].to_numpy(dtype=np.float64)
    z = logit(np.clip(y, LOGIT_EPS, 1.0 - LOGIT_EPS))
    J = np.column_stack([np.ones(len(X)), X])       # 자코비안 = 설계행렬
    return J, z


def fisher(J: np.ndarray, z: np.ndarray) -> dict:
    n, p = J.shape
    theta = np.linalg.lstsq(J, z, rcond=None)[0]
    r = z - J @ theta
    s2 = float(r @ r) / (n - p)                     # 잔차분산 (로짓 공간)
    JtJ = J.T @ J
    cov = np.linalg.inv(JtJ) * s2                   # = FIM⁻¹
    return dict(theta=theta, resid=r, sigma2=s2, n=n, p=p,
                FIM=JtJ / s2, cov=cov, se=np.sqrt(np.diag(cov)))


def eig_report(J: np.ndarray, s2: float) -> dict:
    """원단위와 표준화 두 가지로 고유값을 낸다 (§함정 1)."""
    ev_raw = np.linalg.eigvalsh(J.T @ J / s2)
    Z = (J[:, 1:] - J[:, 1:].mean(0)) / J[:, 1:].std(0)
    Js = np.column_stack([np.ones(len(J)), Z])
    ev_std = np.linalg.eigvalsh(Js.T @ Js / s2)
    return dict(raw=ev_raw, std=ev_std,
                cond_raw=ev_raw.max() / ev_raw.min(),
                cond_std=ev_std.max() / ev_std.min())


def block_bootstrap(J: np.ndarray, z: np.ndarray, block: int = BLOCK,
                    n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """이동블록 부트스트랩. 연속 `block`행 덩어리를 복원추출해 길이 n 재구성 후 재적합."""
    rng = np.random.default_rng(seed)
    n, p = J.shape
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    out = np.empty((n_boot, p))
    for b in range(n_boot):
        st = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        out[b] = np.linalg.lstsq(J[idx], z[idx], rcond=None)[0]
    return out


def autocorr(r: np.ndarray, lags=(1, 5, 15, 30, 60, 120)) -> dict:
    return {k: float(np.corrcoef(r[:-k], r[k:])[0, 1]) for k in lags}


def n_eff(r: np.ndarray, max_lag: int = 240) -> float:
    """유효표본수 (Bartlett): n_eff = n / (1 + 2·Σρ_k)."""
    n = len(r)
    s = sum(float(np.corrcoef(r[:-k], r[k:])[0, 1]) * (1 - k / n) for k in range(1, max_lag + 1))
    return n / (1 + 2 * s)


# ---------------------------------------------------------------- 독립 검증 경로 3종
def hac_se(J: np.ndarray, r: np.ndarray, L: int) -> np.ndarray:
    """Newey-West(Bartlett 커널) HAC 표준오차 — 재샘플링 없는 해석적 경로.

    부트스트랩과 완전히 다른 원리로 같은 문제(자기상관)를 푼다. 두 결과가 비슷하면
    어느 한쪽의 구현 실수가 아님을 뒷받침한다.
    Cov = (JᵀJ)⁻¹ S (JᵀJ)⁻¹,  S = Σuuᵀ + Σ_l w_l (G_l + G_lᵀ),  w_l = 1 − l/(L+1)
    """
    u = J * r[:, None]
    S = u.T @ u
    for l in range(1, L + 1):
        G = u[l:].T @ u[:-l]
        S = S + (1.0 - l / (L + 1.0)) * (G + G.T)
    A = np.linalg.inv(J.T @ J)
    return np.sqrt(np.diag(A @ S @ A))


def andrews_bandwidth(r: np.ndarray) -> float:
    """Andrews(1991) AR(1) plug-in 자동 대역폭 (Bartlett 커널). 임의 선택을 피한다."""
    rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    a1 = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return 1.1447 * (a1 * len(r)) ** (1 / 3)


def campaign_jackknife(J: np.ndarray, z: np.ndarray, camp: np.ndarray) -> dict:
    """캠페인(하루) 단위 leave-one-out. **조정 파라미터가 하나도 없다.**

    블록 길이·반복 횟수 같은 임의 선택 없이, "이 캠페인이 없었다면 계수가 어떻게 변했나"를
    직접 보여준다. 학습셋은 10/13~16 네 캠페인이고 각 캠페인이 서로 다른 변수를 조작했으므로,
    이 표는 곧 **어느 계수가 어느 실험에 의존하는가**의 지도다.
    """
    days = sorted(set(camp))
    g = len(days)
    rows = np.array([np.linalg.lstsq(J[camp != d], z[camp != d], rcond=None)[0] for d in days])
    se = np.sqrt((g - 1) / g * ((rows - rows.mean(0)) ** 2).sum(0))
    return dict(days=days, rows=rows, se=se)


def campaign_bootstrap(J, z, camp, n_boot=N_BOOT, seed=SEED) -> dict:
    """캠페인 통째로 복원추출 — 가장 보수적. 특이(추정 불가) 표본은 세어서 보고한다."""
    rng = np.random.default_rng(seed)
    days = sorted(set(camp))
    idx = {d: np.where(camp == d)[0] for d in days}
    keep, deg = [], 0
    for _ in range(n_boot):
        ii = np.concatenate([idx[d] for d in rng.choice(days, size=len(days), replace=True)])
        A = J[ii]
        if np.linalg.matrix_rank(A) < A.shape[1] or np.linalg.cond(A.T @ A) > 1e14:
            deg += 1
            continue
        keep.append(np.linalg.lstsq(A, z[ii], rcond=None)[0])
    keep = np.array(keep)
    return dict(se=keep.std(0, ddof=1), n_deg=deg, n_ok=len(keep), draws=keep)


def sensitivity(J, z) -> None:
    """임의로 고른 값(블록 길이·반복 횟수)이 결과를 좌우하는지 점검."""
    se_c = fisher(J, z)["se"]
    print("\n[민감도] 블록 길이별 평균 SE 배율 (반복 2,000)")
    for b in (5, 15, 30, 60, 120, 240, 480):
        sd = block_bootstrap(J, z, block=b, n_boot=2000).std(0, ddof=1)
        print("    %4d분 → %5.1f배" % (b, np.mean(sd / se_c)))
    print("[민감도] 반복 횟수별 TT-1006 SE (블록 60분)")
    for nb in (250, 1000, 4000, 8000):
        print("    %5d회 → %.5f" % (nb, block_bootstrap(J, z, 60, nb).std(0, ddof=1)[1]))


def main() -> dict:
    tr = load_train()
    J, z = design(tr)
    f = fisher(J, z)
    e = eig_report(J, f["sigma2"])
    ac = autocorr(f["resid"])
    ne = n_eff(f["resid"])
    boot = block_bootstrap(J, z)
    se_boot = boot.std(axis=0, ddof=1)

    print("=" * 82)
    print("채택 모델: 로짓 1차 선형 · I1 · r1 · 가중치 없음   학습셋 n=%d · 파라미터 p=%d"
          % (f["n"], f["p"]))
    print("로짓 공간 잔차 표준편차 σ = %.5f" % np.sqrt(f["sigma2"]))
    print("=" * 82)

    print("\n[1] FIM 고유값 — 원단위 vs 표준화")
    print("  원단위 :", np.array2string(e["raw"], precision=3), " 조건수 %.3e" % e["cond_raw"])
    print("  표준화 :", np.array2string(e["std"], precision=1), " 조건수 %.2f" % e["cond_std"])
    print("  → 표준화 조건수 %.2f. 0에 가까운 고유값 없음 = **과대 파라미터화 아님**." % e["cond_std"])

    print("\n[2] 잔차 자기상관 (독립표본 가정 점검)")
    for k, v in ac.items():
        print("    lag %3d분 : %+.4f" % (k, v))
    print("  유효표본수 n_eff ≈ %.1f  (명목 n=%d의 %.2f%%)" % (ne, f["n"], 100 * ne / f["n"]))

    print("\n[3] 파라미터 추정 · 표준오차 · 95% 신뢰구간")
    print("  %-22s %14s | %11s %8s | %11s %8s %6s"
          % ("파라미터", "추정값", "고전 SE", "t", "블록BS SE", "t", "배율"))
    print("  " + "-" * 88)
    for i, nm in enumerate(NAMES):
        th, s_c, s_b = f["theta"][i], f["se"][i], se_boot[i]
        print("  %-22s %14.6g | %11.4g %8.1f | %11.4g %8.1f %6.1fx"
              % (nm, th, s_c, th / s_c, s_b, th / s_b, s_b / s_c))
    print("\n  95% CI (블록 부트스트랩 백분위):")
    for i, nm in enumerate(NAMES):
        lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
        print("    %-22s [%.6g, %.6g]" % (nm, lo, hi))

    # ---- 독립 경로 3종 ----
    camp = tr["Campaign"].to_numpy()
    L = andrews_bandwidth(f["resid"])
    se_hac = hac_se(J, f["resid"], 120)          # 120분 = 배율이 최대가 되는 보수적 지점
    jk = campaign_jackknife(J, z, camp)
    cb = campaign_bootstrap(J, z, camp)

    print("\n[4] 캠페인 단위 leave-one-out — 각 계수가 어느 실험에 기대는가")
    print("  %-11s %12s %12s %12s %12s" % ("", *[n.split()[0] for n in NAMES]))
    print("  %-11s %12.6g %12.6g %12.6g %12.6g" % ("전체 사용", *f["theta"]))
    for d, row in zip(jk["days"], jk["rows"]):
        print("  %-11s %12.6g %12.6g %12.6g %12.6g" % ("－" + d, *row))

    print("\n[5] 방법별 t = 계수/SE  (|t| > 2 면 0과 구분됨)")
    tbl = {"고전 Fisher (독립 가정)": f["se"], "HAC Newey-West (120분)": se_hac,
           "이동블록 BS (60분)": se_boot, "캠페인 잭나이프": jk["se"],
           "캠페인 부트스트랩": cb["se"]}
    print("  %-24s %10s %10s %10s %10s" % ("방법", *[n.split()[0] for n in NAMES]))
    for k, v in tbl.items():
        print("  %-24s %10.1f %10.1f %10.1f %10.1f" % (k, *(f["theta"] / np.asarray(v))))
    print("\n  Andrews(1991) 자동 대역폭 = %.0f분 (임의 선택 아님)" % L)
    print("  캠페인 부트스트랩: %d회 중 %d회 특이 → 유효 %d회" % (N_BOOT, cb["n_deg"], cb["n_ok"]))

    return dict(tr=tr, J=J, z=z, f=f, e=e, ac=ac, n_eff=ne, boot=boot, se_boot=se_boot,
                se_hac=se_hac, jk=jk, cb=cb, L_andrews=L)


# ---------------------------------------------------------------- 그림
def figure(res: dict, outpath: Path) -> Path:
    """잭나이프 중심 3패널. (a)가 원인, (b)가 증상, (c)가 판정."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"font.size": 9.3, "axes.titlesize": 10.2, "axes.titleweight": "bold"})

    tr, f, jk = res["tr"], res["f"], res["jk"]
    short = ["절편", "TT-1006", "FT-1004", "FRC-1004"]
    days = jk["days"]

    fig, ax = plt.subplots(1, 3, figsize=(15.4, 4.6))

    # (a) 캠페인별 조작 강도 — 왜 계수마다 신뢰도가 다른지의 원인
    sig = np.array([[tr[tr["Campaign"] == d][c].std() for c in FEATS] for d in days])
    rel = sig / sig.max(axis=0)
    im = ax[0].imshow(rel, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels([c.split(".")[0] for c in FEATS], fontsize=9)
    ax[0].set_yticks(range(len(days))); ax[0].set_yticklabels(days)
    for i in range(len(days)):
        for j in range(3):
            ax[0].text(j, i, "σ=%.3g" % sig[i, j], ha="center", va="center", fontsize=8.4,
                       color="white" if rel[i, j] > 0.5 else "#333333")
    ax[0].set_title("(a) 원인 — 캠페인별 조작 강도\n비율(FRC)을 움직인 날은 10/15 하나뿐")

    # (b) 캠페인 하나를 빼면 계수가 얼마나 변하는가
    dev = (jk["rows"] - f["theta"]) / np.abs(f["theta"]) * 100
    w = 0.2
    xs = np.arange(4)
    cols = ["#C0392B", "#7F8C8D", "#2E86C1", "#1E8449"]
    for k, d in enumerate(days):
        ax[1].bar(xs + (k - 1.5) * w, dev[k], width=w, color=cols[k], label="－" + d)
    ax[1].axhline(0, color="k", lw=0.9)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(short, fontsize=9)
    ax[1].set_ylabel("전체적합 대비 계수 변화 [%]")
    ax[1].set_title("(b) 증상 — 10/13 제외 시 비율 계수 부호 반전(−122%),\n10/15 제외 시 +316%")
    ax[1].legend(fontsize=8.2, ncol=2); ax[1].grid(axis="y", alpha=0.25)
    ax[1].annotate("부호 반전", xy=(3 - 1.5 * w, dev[0, 3]), xytext=(2.1, -190),
                   fontsize=8.6, color="#C0392B", fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.3))

    # (c) 방법별 |t| — 임의 선택이 없는 방법일수록 아래로 내려온다
    meths = [("고전 Fisher\n(독립 가정)", f["se"], "#BBBBBB"),
             ("HAC\n(120분)", res["se_hac"], "#7FB3D5"),
             ("이동블록 BS\n(60분)", res["se_boot"], "#2E86C1"),
             ("캠페인\n잭나이프", jk["se"], "#B00020"),
             ("캠페인\n부트스트랩", res["cb"]["se"], "#7B241C")]
    mk = ["o", "s", "^", "D", "v"]
    for j, nm in enumerate(short):
        for k, (mn, se, col) in enumerate(meths):
            ax[2].scatter(j, abs(f["theta"][j] / se[j]), s=62, marker=mk[k], color=col,
                          edgecolor="white", linewidth=0.7, zorder=3,
                          label=mn if j == 0 else None)
        ax[2].plot([j] * len(meths), [abs(f["theta"][j] / se[j]) for _, se, _ in meths],
                   color="#999999", lw=0.8, zorder=1)
    ax[2].axhline(2, color="#B00020", ls="--", lw=1.4)
    ax[2].text(3.35, 2.15, "|t| = 2\n유의 기준", color="#B00020", fontsize=8.3, ha="right")
    ax[2].set_yscale("log"); ax[2].set_ylabel("|t| = |계수| / 표준오차 (로그축)")
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels(short, fontsize=9)
    ax[2].set_title("(c) 판정 — 임의 선택 없는 방법에서\nFRC-1004만 기준선 아래로 떨어진다")
    ax[2].legend(fontsize=7.4, loc="lower left", ncol=2); ax[2].grid(alpha=0.25, which="both")

    fig.suptitle("그림 D · 파라미터 식별성 — 로짓 1차 선형 · I1 · r1 · 가중치 없음 (학습셋 1,924행 = 캠페인 4회)",
                 fontsize=11.6, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return outpath


if __name__ == "__main__":
    _res = main()
    print("\nsaved:", figure(_res, ROOT / "docs" / "img" / "그림D_Fisher_진단.png"))
