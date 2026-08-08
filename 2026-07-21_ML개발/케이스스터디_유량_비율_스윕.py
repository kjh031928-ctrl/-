"""케이스스터디_유량_비율_스윕.py — 대회 §3.3-6 케이스 스터디 중 유량·B/P비 스윕.

목적: Phase 7-2가 온도 스윕만 다뤘으므로, 남은 두 축(반응기 공급유량, 벤젠/프로펜 비)을
      같은 방식으로 채운다. 원본·기존 결과 파일은 읽기만 하고 수정하지 않는다.

대회 요구 (rev2 §3.3-6):
  - Reactor feed flow rate : 15,000 – 18,000 kg/h
  - Benzene to propene ratio : 4 – 6
  - 최소 산출: 반응기 출구 큐멘 몰농도 + 큐멘 제품 질량유량

축 환산 (데이터로 검산: FT-1004×FRC-1004 vs 벤젠 실측 FT-1002 최대 상대오차 0.014%):
      총공급[kg/h] = FT-1004[kg/h] × (1 + FRC-1004)

스윕 구성 (OFAT — 한 축을 움직이면 나머지 두 축은 10/12 정상운전 기준값에 고정):
  C) 온도   : 335→365 °C, 총공급 18,000 · 비율 5.0 고정
  A) 유량   : 총공급 15,000→18,000, 341.57 °C · 비율 5.0 고정 → FT-1004 2,500 → 3,000
  B1) 비율  : 비율 4→6, 총공급 18,000 고정               → FT-1004 3,600 → 2,571.4

  대회가 "reactor feed flow rate"를 별개 축으로 명시했으므로, 비율을 스캔할 때
  공급유량은 기준값(18,000)에 고정한다. 이것이 OFAT의 정의다.

모델: Phase 7-2와 동일하게 **학습셋(10/13–16)으로 재학습**한 3종을 비교.
      배포용 ONNX(전체 재학습본)를 쓰면 성능 근거와 예측이 뒤섞이므로 쓰지 않는다.
      전부 입력세트 I1 · 라벨 r0 · 가중치 없음 (Phase 10 확정 레시피 축).
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
from scipy.special import expit, logit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config" / "base.yaml", encoding="utf-8"))

MW = CFG["properties"]["MW_kg_per_kmol"]          # AVEVA 제공값, 대체 금지
W_PROPENE = CFG["propene_feed_mass_fraction"]["propene"]   # 0.94773
W_PROPANE = CFG["propene_feed_mass_fraction"]["propane"]   # 0.05227
SEED = CFG["seed"]                                 # 42
LOGIT_EPS = 1e-6                                   # 지시서 Phase 10-A
FEATS = CFG["features"]["I1"]                      # [TT-1006, FT-1004, FRC-1004]
LABEL = CFG["labels"]["r1"]                        # X_propene_recycle1 (배포 ONNX와 동일 라벨)

# 기준 운전점 = 10/12 정상운전(481행 전부 동일 조건). 학습·검증 어디에도 쓰이지 않은 구간.
BASE_T, BASE_FT, BASE_FRC = 341.57, 3000.0, 5.0
BASE_X_ACTUAL = 0.615885
TRAIN_T_MAX = 349.977                              # 학습셋 TT-1006 최댓값 (CLAUDE.md §5)                           # CLAUDE.md §5 기준값


# ---------------------------------------------------------------- 데이터
def load() -> pd.DataFrame:
    """정본 엑셀 + r1 라벨(별도 CSV) 병합. r1은 Timestamp 기준 병합(config 규약)."""
    df = pd.read_excel(ROOT / CFG["paths"]["data_xlsx"], sheet_name=CFG["paths"]["data_sheet"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    if LABEL not in df.columns:
        lab = pd.read_csv(ROOT / CFG["paths"]["labels_recycle1"])
        lab["Timestamp"] = pd.to_datetime(lab["Timestamp"])
        n0 = len(df)
        df = df.merge(lab[["Timestamp", LABEL]], on="Timestamp", how="left")
        if len(df) != n0 or df[LABEL].isna().all():
            raise RuntimeError("r1 라벨 병합 실패 — Timestamp 키 확인 (data_loader.py 규약과 동일)")
    return df


# ---------------------------------------------------------------- 모델 3종
def fit_models(tr: pd.DataFrame) -> dict:
    Xtr = tr[FEATS].to_numpy(dtype=np.float64)
    ytr = tr[LABEL].to_numpy(dtype=np.float64)

    lin = LinearRegression().fit(Xtr, ytr)
    lgt = LinearRegression().fit(Xtr, logit(np.clip(ytr, LOGIT_EPS, 1 - LOGIT_EPS)))
    gbr = GradientBoostingRegressor(random_state=SEED).fit(Xtr, ytr)

    return {
        "로짓 1차 선형 (채택)": lambda A: expit(lgt.predict(A)),
        "1차 선형 + 하드클립": lambda A: np.clip(lin.predict(A), 0.0, 1.0),
        "GBR": lambda A: gbr.predict(A),
    }


# ---------------------------------------------------------------- 물질수지
def moles(ft: np.ndarray, frc: np.ndarray) -> tuple[np.ndarray, ...]:
    """프로펜 스트림 유량·질량비 → 반응기 입구 몰유량 [kmol/h].

    검산(데이터 1행): nP 67.56468 / nA 3.55604 / nB 192.02458 과 각각 일치.
    """
    nP = ft * W_PROPENE / MW["propene"]
    nA = ft * W_PROPANE / MW["propane"]
    nB = ft * frc / MW["benzene"]
    return nP, nA, nB


def outputs(X: np.ndarray, ft: np.ndarray, frc: np.ndarray) -> dict:
    """전환율 → 큐멘 질량유량 및 출구 큐멘 몰분율.

    반응: 프로펜 + 벤젠 → 큐멘 (1:1 → 1). 출구 총몰수 = 입구 총몰수 − nP·X.
    ※ 몰'농도'[kmol/m³]는 출구 압력이 있어야 계산되는데 케이스 스터디가 압력을
      지정하지 않으므로(압력 제외 근거 ②와 동일 이유) 몰분율로 보고한다.
    """
    nP, nA, nB = moles(ft, frc)
    cumene_kmol = nP * X
    n_out = nB + nP + nA - nP * X
    return {
        "X": X,
        "큐멘_kg_h": cumene_kmol * MW["cumene"],
        "큐멘_몰분율": cumene_kmol / n_out,
        "미반응벤젠_kmol_h": nB - nP * X,
    }


# ---------------------------------------------------------------- 스윕
def sweep(models: dict, T: np.ndarray, ft: np.ndarray, frc: np.ndarray) -> dict:
    A = np.column_stack([T, ft, frc]).astype(np.float64)
    return {name: outputs(f(A), ft, frc) for name, f in models.items()}


def hull_tester(tr: pd.DataFrame):
    """학습셋 I1 3변수의 convex hull 내부 판정 함수를 만든다 (가이드 §6-2).

    변수별 min/max 만으로는 '개별 변수는 범위 안이지만 조합이 학습에 없던 조합'을
    잡지 못한다. 실제로 학습셋의 (FT-1004, FRC-1004)는 총공급이 대략 일정한
    음의 기울기 띠를 이루므로, 이 판정이 필요하다.
    표준화 후 ConvexHull → Delaunay.find_simplex >= 0 이면 내부.
    """
    from scipy.spatial import ConvexHull, Delaunay
    P = tr[FEATS].to_numpy(dtype=np.float64)
    mu, sd = P.mean(axis=0), P.std(axis=0)
    Z = (P - mu) / sd
    tri = Delaunay(Z[ConvexHull(Z).vertices])

    def inside(T, ft, frc):
        Q = (np.column_stack([T, ft, frc]).astype(np.float64) - mu) / sd
        return tri.find_simplex(Q) >= 0

    return inside


def main():
    df = load()
    tr = df[df["Split"] == CFG["split"]["train_value"]]
    models = fit_models(tr)

    rng = {c: (float(tr[c].min()), float(tr[c].max())) for c in FEATS}
    tot_tr = tr["FT-1004.PV"] * (1 + tr["FRC-1004.PV"])
    rng["총공급"] = (float(tot_tr.min()), float(tot_tr.max()))

    n = 61
    tot_A = np.linspace(15000.0, 18000.0, n)
    frc_B = np.linspace(4.0, 6.0, n)

    T_C = np.linspace(335.0, 365.0, n)

    grids = {
        "C": dict(title="스윕 C · 반응기 입구온도 (총공급 18,000 · 비율 5.0 고정)",
                  x=T_C, xlabel="반응기 입구온도 TT-1006 [°C]",
                  T=T_C, ft=np.full(n, BASE_FT), frc=np.full(n, BASE_FRC)),
        "A": dict(title="스윕 A · 반응기 공급유량 (비율 5.0 고정)",
                  x=tot_A, xlabel="반응기 총공급 [kg/h]",
                  T=np.full(n, BASE_T), ft=tot_A / (1 + BASE_FRC), frc=np.full(n, BASE_FRC)),
        "B1": dict(title="스윕 B1 · 벤젠/프로펜 비 (총공급 18,000 kg/h 고정)",
                   x=frc_B, xlabel="벤젠/프로펜 질량비 [-]",
                   T=np.full(n, BASE_T), ft=18000.0 / (1 + frc_B), frc=frc_B),
    }

    ins = hull_tester(tr)
    out = {}
    for k, g in grids.items():
        g["inside"] = ins(g["T"], g["ft"], g["frc"])
        out[k] = dict(grid=g, res=sweep(models, g["T"], g["ft"], g["frc"]))
    return out, rng, models, tr

# ---------------------------------------------------------------- 그림
COLORS = {"로짓 1차 선형 (채택)": "#1F3864", "1차 선형 + 하드클립": "#C77700", "GBR": "#2E8B57"}
STYLES = {"로짓 1차 선형 (채택)": "-", "1차 선형 + 하드클립": "--", "GBR": "-."}


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5,
                         "axes.titleweight": "bold", "figure.dpi": 110})
    return plt


def outside_segments(x: np.ndarray, inside: np.ndarray) -> list[tuple[float, float]]:
    """hull 밖 연속 구간을 [(x시작, x끝), ...] 로."""
    segs, i = [], 0
    while i < len(inside):
        if not inside[i]:
            j = i
            while j + 1 < len(inside) and not inside[j + 1]:
                j += 1
            segs.append((float(x[i]), float(x[j])))
            i = j + 1
        else:
            i += 1
    return segs


def _panel(ax, g, res, field, ylabel, title=None):
    for name, d in res.items():
        ax.plot(g["x"], d[field], STYLES[name], color=COLORS[name], lw=1.9, label=name)
    for a, b in outside_segments(g["x"], g["inside"]):
        ax.axvspan(a, b, color="#B00020", alpha=0.10, zorder=0)
    ax.set_xlabel(g["xlabel"]); ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)


def figures(out, outdir: Path) -> list[Path]:
    plt = _setup_mpl()
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []

    g, res = out["A"]["grid"], out["A"]["res"]
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))
    _panel(ax[0], g, res, "큐멘_kg_h", "큐멘 생산량 [kg/h]")
    _panel(ax[1], g, res, "큐멘_몰분율", "반응기 출구 큐멘 몰분율 [-]")
    ax[0].legend(fontsize=8.4, loc="upper left")
    fig.suptitle("그림 A · 반응기 공급유량 15,000→18,000 kg/h (비율 5.0 · 341.57 °C 고정)"
                 "  —  붉은 음영 = 학습셋 convex hull 밖", fontsize=11.3, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pa = outdir / "그림A_케이스스터디_유량스윕.png"
    fig.savefig(pa, dpi=165, bbox_inches="tight", facecolor="white"); plt.close(fig)
    paths.append(pa)

    g, res = out["C"]["grid"], out["C"]["res"]
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))
    _panel(ax[0], g, res, "큐멘_kg_h", "큐멘 생산량 [kg/h]")
    _panel(ax[1], g, res, "큐멘_몰분율", "반응기 출구 큐멘 몰분율 [-]")
    for a in ax:
        a.axvline(TRAIN_T_MAX, color="#B00020", ls=":", lw=1.4)
    ax[0].legend(fontsize=8.4, loc="upper left")
    fig.suptitle("그림 C · 반응기 입구온도 335→365 °C (총공급 18,000 kg/h · 비율 5.0 고정)"
                 "  —  점선 = 학습 상한 %.3f °C" % TRAIN_T_MAX,
                 fontsize=11.3, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pc = outdir / "그림C_케이스스터디_온도스윕.png"
    fig.savefig(pc, dpi=165, bbox_inches="tight", facecolor="white"); plt.close(fig)
    paths.append(pc)

    g, res = out["B1"]["grid"], out["B1"]["res"]
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))
    _panel(ax[0], g, res, "큐멘_kg_h", "큐멘 생산량 [kg/h]")
    _panel(ax[1], g, res, "큐멘_몰분율", "출구 큐멘 몰분율 [-]")
    ax[0].legend(fontsize=8.4, loc="upper right")
    fig.suptitle("그림 B · 벤젠/프로펜 질량비 4→6 (총공급 18,000 kg/h · 341.57 °C 고정)"
                 "  —  붉은 음영 = 학습셋 convex hull 밖", fontsize=11.3, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pb = outdir / "그림B_케이스스터디_비율스윕.png"
    fig.savefig(pb, dpi=165, bbox_inches="tight", facecolor="white"); plt.close(fig)
    paths.append(pb)
    return paths


if __name__ == "__main__":
    out, rng, models, tr = main()
    for p in figures(out, ROOT / "docs" / "img"):
        print("saved:", p)
