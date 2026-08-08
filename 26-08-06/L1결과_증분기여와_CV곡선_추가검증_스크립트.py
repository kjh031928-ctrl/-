"""L1결과_증분기여와_CV곡선_추가검증_스크립트.py
=====================================================================
목적: 1·2번 스크립트에서 나온 결론을 확증편향 없이 되짚는 세 가지 추가 검사.

  (1) CV 곡선 평탄성 — LassoCV 의 α* 는 최소점 하나다. 곡선이 평탄하면 α* 는
      불안정하고 "선택된 변수 개수"도 불안정하다. 평균±표준오차를 직접 본다.
  (2) 증분 기여 — 물리셋에 후보를 하나씩 더했을 때 held-out RMSE 가 얼마나
      줄어드는가. L1 이 통째로 준 개선을 어떤 변수가 만들었는지 분해한다.
  (3) 캠페인 LOCO CV 로도 같은 순위가 나오는가 — held-out 한 번의 우연이 아닌지.
      (held-out 은 최종 확인용이므로, 순위 판단은 CV 로 한다.)

읽기 전용 · 무가중 · 무작위분할 없음 · seed=42.
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
import pandas as pd
import sklearn
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

SEED = 42
HX = 25.0
_ROOT = Path(__file__).resolve().parents[1]
_ATT1 = _ROOT / "2026 Chemical Engineering Process design competition_rev0_Attachment1.csv"
_DPX = _ROOT / "DP_data.xlsx"
_LAB = _ROOT / "2026-07-21_ML개발" / "data" / "processed" / "labels_recycle1.csv"

FEATS = ["TT-1006", "FT-1004", "총유량", "FRC-1004", "TT-1004", "FT-1001", "FT-1300"]
PHYS = {"X": ["TT-1006", "FT-1004", "FRC-1004"],
        "DP": ["TT-1006", "총유량", "FRC-1004"]}
TRAIN_CAMPS = ["10/13", "10/14", "10/15", "10/16"]


def hr(t, ch="="):
    print("\n" + ch * 78); print(t); print(ch * 78)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def load():
    att = pd.read_csv(_ATT1)
    att = att.rename(columns={att.columns[0]: "Timestamp"})
    att["Timestamp"] = pd.to_datetime(att["Timestamp"], utc=True).dt.tz_localize(None)

    def pick(tag):
        hit = [c for c in att.columns if c.startswith(tag + ".PV")]
        assert len(hit) == 1
        return att[hit[0]].astype(float)

    df = pd.DataFrame({"Timestamp": att["Timestamp"]})
    for tag in ["TT-1006", "FT-1004", "FT-1003", "FRC-1004", "TT-1004",
                "FT-1001", "FT-1300", "PT-1004", "PT-1100"]:
        df[tag] = pick(tag)
    df["총유량"] = df["FT-1003"] + df["FT-1004"]
    lab = pd.read_csv(_LAB, encoding="utf-8-sig")
    lab["Timestamp"] = pd.to_datetime(lab["Timestamp"])
    df = df.merge(lab[["Timestamp", "Campaign", "X_propene_conversion"]], on="Timestamp")
    dpx = pd.read_excel(_DPX)
    dpx = dpx.rename(columns={dpx.columns[0]: "Timestamp"})
    dpx["Timestamp"] = pd.to_datetime(dpx["Timestamp"], utc=True).dt.tz_localize(None)
    df = df.merge(dpx[["Timestamp", "DP_reactor"]], on="Timestamp")
    df["X"] = df["X_propene_conversion"].astype(float)
    df["DP"] = df["DP_reactor"].astype(float)
    return df


# ------------------------------------------------ (1) CV 곡선 평탄성
def cv_curve(tr, target, cols, tag):
    X = tr[cols].to_numpy(float); y = tr[target].to_numpy(float)
    g = tr["Campaign"].to_numpy()
    folds = list(LeaveOneGroupOut().split(X, y, g))
    sc = StandardScaler().fit(X)
    amax = np.max(np.abs(sc.transform(X).T @ (y - y.mean()))) / len(y)
    alphas = np.geomspace(amax, amax * 1e-6, 25)
    print(f"\n[{tag}] 캠페인 LOCO CV 곡선 (α_max={amax:.5g}, 폴드마다 스케일러 재적합)")
    print(f"  {'α':>12} {'평균CV-RMSE':>12} {'±SE':>10} {'선택변수수':>8}  선택 변수")
    rows = []
    for al in alphas:
        errs = []
        for itr, iva in folds:
            s = StandardScaler().fit(X[itr])
            m = Lasso(alpha=al, max_iter=200000, tol=1e-7, random_state=SEED)
            m.fit(s.transform(X[itr]), y[itr])
            errs.append(np.sqrt(np.mean((y[iva] - m.predict(s.transform(X[iva]))) ** 2)))
        s = StandardScaler().fit(X)
        mf = Lasso(alpha=al, max_iter=200000, tol=1e-7, random_state=SEED).fit(s.transform(X), y)
        nz = [cols[j] for j in range(len(cols)) if abs(mf.coef_[j]) > 1e-10]
        e = np.array(errs)
        rows.append((al, e.mean(), e.std(ddof=1) / np.sqrt(len(e)), nz))
        print(f"  {al:12.5g} {e.mean():12.5f} {e.std(ddof=1)/np.sqrt(len(e)):10.5f} "
              f"{len(nz):8d}  {nz}")
    best = min(rows, key=lambda r: r[1])
    thr = best[1] + best[2]
    within = [r for r in rows if r[1] <= thr]
    print(f"  → 최소 CV-RMSE {best[1]:.5f} (α={best[0]:.5g}, 변수 {len(best[3])}개)")
    print(f"  → 1-SE 밴드({thr:.5f}) 안에 드는 α 범위: {min(r[0] for r in within):.5g} "
          f"~ {max(r[0] for r in within):.5g}  → 변수 개수 {min(len(r[3]) for r in within)}"
          f"~{max(len(r[3]) for r in within)}개")
    print("  ※ 밴드 안에서 변수 개수가 넓게 흔들리면 'L1이 고른 개수'는 확정적 결론이 아니다.")


# ------------------------------------------------ (2)(3) 증분 기여
def loco_rmse(tr, target, cols):
    """캠페인 LOCO CV 의 전체 RMSE(폴드 잔차를 모아 계산). OLS."""
    X = tr[cols].to_numpy(float); y = tr[target].to_numpy(float)
    g = tr["Campaign"].to_numpy()
    res = np.empty_like(y)
    per = {}
    for itr, iva in LeaveOneGroupOut().split(X, y, g):
        p = LinearRegression().fit(X[itr], y[itr]).predict(X[iva])
        res[iva] = y[iva] - p
        per[g[iva][0]] = rmse(y[iva], p)
    return float(np.sqrt(np.mean(res ** 2))), per


def incremental(tr, ho, target, tag):
    base = PHYS[target]
    hr(f"(2)(3) 증분 기여 — 타깃 {tag}   물리셋 {base}", "-")
    y_ho = ho[target].to_numpy(float)

    def ho_rmse(cols):
        return rmse(y_ho, LinearRegression()
                    .fit(tr[cols].to_numpy(float), tr[target].to_numpy(float))
                    .predict(ho[cols].to_numpy(float)))

    b_cv, b_per = loco_rmse(tr, target, base)
    b_ho = ho_rmse(base)
    print(f"  {'입력셋':<44} {'LOCO CV':>9} {'held-out':>9}   폴드별 CV-RMSE")
    print(f"  {'물리셋(기준)':<44} {b_cv:9.5f} {b_ho:9.5f}   " +
          " ".join(f"{k}={v:.4f}" for k, v in b_per.items()))
    extras = [c for c in FEATS if c not in base]
    rows = []
    for e in extras:
        cols = base + [e]
        cv, per = loco_rmse(tr, target, cols)
        h = ho_rmse(cols)
        rows.append((e, cv, h, per))
    rows.sort(key=lambda r: r[1])
    for e, cv, h, per in rows:
        print(f"  {'물리셋 + ' + e:<44} {cv:9.5f} {h:9.5f}   " +
              " ".join(f"{k}={v:.4f}" for k, v in per.items()))
    allcv, allper = loco_rmse(tr, target, FEATS)
    print(f"  {'후보 7종 전부':<44} {allcv:9.5f} {ho_rmse(FEATS):9.5f}   " +
          " ".join(f"{k}={v:.4f}" for k, v in allper.items()))
    # 물리셋에서 하나씩 빼기
    print("  --- 물리셋에서 하나씩 제거 (물리 변수가 정말 필요한가) ---")
    for d in base:
        cols = [c for c in base if c != d]
        cv, _ = loco_rmse(tr, target, cols)
        print(f"  {'물리셋 − ' + d:<44} {cv:9.5f} {ho_rmse(cols):9.5f}")


# ------------------------------------------------ (4) 상수열 상관 가드 재확인
def const_guard(df):
    hr("(4) 10/12 상관값 재확인 — 상수열은 상관 정의 불가")
    g = df[df["Campaign"] == "10/12"]
    for c in ["X", "DP", "FT-1001", "FT-1300", "TT-1004", "TT-1006", "FT-1004"]:
        v = g[c].to_numpy(float)
        print(f"  {c:>10} nunique={len(np.unique(v)):3d}  std={v.std(ddof=0):.3e}  "
              f"ptp={np.ptp(v):.3e}")
    print("  → 표준편차가 0 이거나 부동소수점 잔차 수준이면 상관계수는 의미 없다.")
    print("     2번 스크립트 표의 10/12 열(−1.0000 / +1.0000)은 이 수치잡음의 산물이며,")
    print("     '완전 상관'이 아니라 '정의 불가'로 읽어야 한다.")


# ------------------------------------------------ (5) CV 최적 α 에서의 계수
def coef_at(tr, target, cols, alpha, tag):
    X = tr[cols].to_numpy(float); y = tr[target].to_numpy(float)
    sc = StandardScaler().fit(X)
    m = Lasso(alpha=alpha, max_iter=500000, tol=1e-9, random_state=SEED).fit(sc.transform(X), y)
    print(f"\n[{tag}] per-fold 재스케일 CV 최적 α={alpha:.5g} 에서의 표준화 계수")
    for j in np.argsort(-np.abs(m.coef_)):
        print(f"    {cols[j]:>10} {m.coef_[j]:+12.6f}  {'선택' if abs(m.coef_[j])>1e-10 else '탈락'}")
    fl = [c for c in ["FT-1004", "총유량"] if c in cols]
    if len(fl) == 2:
        a, b = (abs(m.coef_[cols.index(c)]) for c in fl)
        big = fl[0] if a > b else fl[1]
        ratio = max(a, b) / min(a, b) if min(a, b) > 0 else np.inf
        print(f"    → 두 유량 중 큰 쪽: {big}  (크기비 {ratio:.2f}배)")


def sec_target_corr(tr):
    hr("(5) 타깃과 후보의 단순 상관 (학습 10/13–16)  [계산]")
    print(f"  {'변수':>10} {'corr(·, X)':>12} {'corr(·, ΔP)':>12}")
    for c in FEATS:
        print(f"  {c:>10} {np.corrcoef(tr[c], tr['X'])[0,1]:>12.4f} "
              f"{np.corrcoef(tr[c], tr['DP'])[0,1]:>12.4f}")


def main():
    print(f"sklearn {sklearn.__version__} · numpy {np.__version__} · seed={SEED}")
    df = load()
    tr = df[df["Campaign"].isin(TRAIN_CAMPS)].reset_index(drop=True)
    ho = df[df["Campaign"] == "10/17-18"].reset_index(drop=True)

    hr("(1) CV 곡선 평탄성 — α* 최소점이 얼마나 믿을 만한가")
    cv_curve(tr, "X", FEATS, "전환율 X")
    cv_curve(tr, "DP", FEATS, "ΔP")

    incremental(tr, ho, "X", "전환율 X")
    incremental(tr, ho, "DP", "ΔP")
    const_guard(df)

    sec_target_corr(tr)
    hr("(6) CV 최적 α 에서 두 유량의 상대 크기 — 물리 기대와 대조")
    coef_at(tr, "X", FEATS, 0.00092646, "전환율 X")    # (1) 곡선 최소점
    coef_at(tr, "DP", FEATS, 0.0070112, "ΔP")          # (1) 곡선 최소점
    print("\n  물리 기대: 전환율=프로펜유량 우세 / ΔP=총유량 우세.")

    # ---------------- (7) 재매개화 등가성 ----------------
    hr("(7) L1 이 FRC-1004 를 버린 이유 — (FT-1004, 총유량) 쌍이 비율을 대신하는가")
    P = tr[["FT-1004", "총유량"]].to_numpy(float)
    f = tr["FRC-1004"].to_numpy(float)
    r2 = LinearRegression().fit(P, f).score(P, f)
    print(f"  FRC-1004 ~ (FT-1004, 총유량) 선형회귀 R² = {r2:.5f}  "
          f"(정의상 FRC = 총유량/FT-1004 − 1, 비선형이므로 완전 1 은 아님)")
    ho2 = df[df["Campaign"] == "10/17-18"]
    for target, tag in [("X", "전환율 X"), ("DP", "ΔP")]:
        print(f"\n  [{tag}]  {'입력셋':<40}{'LOCO CV':>10}{'held-out':>10}")
        for name, cols in [("물리셋", PHYS[target]),
                           ("재매개화 [T, FT-1004, 총유량] (FRC 제거)",
                            ["TT-1006", "FT-1004", "총유량"]),
                           ("[T, FT-1004, 총유량, FRC] (넷 다)",
                            ["TT-1006", "FT-1004", "총유량", "FRC-1004"])]:
            cv, _ = loco_rmse(tr, target, cols)
            h = rmse(ho2[target].to_numpy(float),
                     LinearRegression().fit(tr[cols].to_numpy(float),
                                            tr[target].to_numpy(float))
                     .predict(ho2[cols].to_numpy(float)))
            print(f"        {name:<40}{cv:>10.5f}{h:>10.5f}")


if __name__ == "__main__":
    main()
