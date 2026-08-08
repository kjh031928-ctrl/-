"""L1_변수선택_교차검증_재현스크립트.py
=====================================================================
목적: 물리적 근거로 고른 ML 입력(전환율 I1 / ΔP F2)을
      Lasso·ElasticNet 의 L1 변수선택으로 **독립 교차검증**한다.
      "새 모델 배포"가 아니라 "물리 선택을 통계가 뒷받침하는가" 확인.

절대규칙 (루트 CLAUDE.md + 2026-07-21_ML개발/CLAUDE.md 준수):
  - 원본 대회 파일·DP_data.xlsx 는 **읽기 전용**. 파생변수는 메모리에서만 계산.
  - .xlsx 산출 금지.
  - 폐기 컬럼(ML_weight, TT-1006_rate_C_per_min, τ=3분) 일체 미사용. **전 구간 무가중.**
  - 분할은 날짜 기준만: 학습 10/13–16, held-out 10/17–18, 10/12 는 앵커(통계 미포함).
    무작위 KFold·train_test_split·shuffle 금지.
  - seed = 42 고정. sklearn/numpy 버전 출력. 모든 수치는 이 스크립트가 재계산한 값.

표기: [주어짐]=대회문서/물성 · [계산]=데이터 재계산 · [가정]=이 분석의 가정
=====================================================================
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (ElasticNet, ElasticNetCV, Lasso, LassoCV,
                                  LinearRegression, lasso_path)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ------------------------------------------------------------------ 상수
SEED = 42                      # [주어짐] CLAUDE.md §2.3 난수 시드
HX1005_TUBE_DP = 25.0          # kPa. HX-1005 튜브측 0.25 bar [주어짐] rev2 App B Table 4
N_BOOT = 1000                  # 부트스트랩 반복수 (지시: ≥1000)
ZERO = 1e-10                   # 비영 계수 판정 임계 (수치오차 절단, [가정])

_ROOT = Path(__file__).resolve().parents[1]
_ATT1 = _ROOT / "2026 Chemical Engineering Process design competition_rev0_Attachment1.csv"
_DPX = _ROOT / "DP_data.xlsx"
_LAB = _ROOT / "2026-07-21_ML개발" / "data" / "processed" / "labels_recycle1.csv"

# 후보 입력 superset — 런타임에 얻을 수 있는 상류·측정값만
FEATS = ["TT-1006", "FT-1004", "총유량", "FRC-1004", "TT-1004", "FT-1001", "FT-1300"]
FEAT_DESC = {
    "TT-1006": "반응기 입구온도 °C",
    "FT-1004": "프로펜 스트림 유량 kg/h",
    "총유량": "FT-1003+FT-1004 kg/h [계산]",
    "FRC-1004": "벤젠/프로펜 질량비 (무차원)",
    "TT-1004": "프로펜 공급온도 °C",
    "FT-1001": "신선벤젠 유량 kg/h",
    "FT-1300": "재순환벤젠 유량 kg/h",
}
LEAK = ["PT-1004", "PT-1100"]   # 누출 시연 전용 (§C)

# 물리적으로 고른 기준 입력세트
PHYS = {"X": ["TT-1006", "FT-1004", "FRC-1004"],      # 전환율 I1
        "DP": ["TT-1006", "총유량", "FRC-1004"]}       # ΔP  F2

TRAIN_CAMPS = ["10/13", "10/14", "10/15", "10/16"]
HELD_CAMP = "10/17-18"
ANCHOR_CAMP = "10/12"


def hr(title: str, ch: str = "=") -> None:
    print("\n" + ch * 78)
    print(title)
    print(ch * 78)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


# ============================================================== 0. 데이터
def load() -> pd.DataFrame:
    """원본 3종을 Timestamp 로 조인. 파일은 읽기만 한다."""
    att = pd.read_csv(_ATT1)
    att = att.rename(columns={att.columns[0]: "Timestamp"})
    att["Timestamp"] = pd.to_datetime(att["Timestamp"], utc=True).dt.tz_localize(None)

    def pick(tag: str) -> pd.Series:
        hit = [c for c in att.columns if c.startswith(tag + ".PV")]
        assert len(hit) == 1, f"{tag} 매칭 {hit}"
        return att[hit[0]].astype(float)

    df = pd.DataFrame({"Timestamp": att["Timestamp"]})
    for tag in ["TT-1006", "FT-1004", "FT-1003", "FRC-1004",
                "TT-1004", "FT-1001", "FT-1300", "PT-1004", "PT-1100"]:
        df[tag] = pick(tag)
    # [계산] 총유량 — 데이터에 단일 태그 없음. 정의상 FT-1004×(1+FRC) 와 동일.
    df["총유량"] = df["FT-1003"] + df["FT-1004"]

    # 라벨 1: 전환율 (SRK 역산값, 재유도하지 않고 컬럼 사용)
    lab = pd.read_csv(_LAB, encoding="utf-8-sig")
    lab["Timestamp"] = pd.to_datetime(lab["Timestamp"])
    n0 = len(df)
    df = df.merge(lab[["Timestamp", "Campaign", "X_propene_conversion",
                       "X_propene_recycle1"]], on="Timestamp", how="inner")
    assert len(df) == n0, f"라벨 조인 행수 {n0}->{len(df)}"

    # 라벨 2: ΔP (DP_data.xlsx 읽기 전용)
    dpx = pd.read_excel(_DPX)
    dpx = dpx.rename(columns={dpx.columns[0]: "Timestamp"})
    dpx["Timestamp"] = pd.to_datetime(dpx["Timestamp"], utc=True).dt.tz_localize(None)
    df = df.merge(dpx[["Timestamp", "DP_reactor"]], on="Timestamp", how="inner")
    assert len(df) == n0, f"DP 조인 행수 {n0}->{len(df)}"

    df["X"] = df["X_propene_conversion"].astype(float)
    df["DP"] = df["DP_reactor"].astype(float)
    return df


def verify(df: pd.DataFrame) -> None:
    hr("0. 데이터 조립·검증  [계산]")
    print(f"python {sys.version.split()[0]} · sklearn {sklearn.__version__} · "
          f"numpy {np.__version__} · pandas {pd.__version__} · seed={SEED}")
    print(f"조인 결과 {len(df)}행 · 결측 {int(df.isna().sum().sum())} · "
          f"중복 Timestamp {int(df['Timestamp'].duplicated().sum())}")

    # (1) ΔP 라벨 재계산 대조
    dp_calc = df["PT-1004"] - HX1005_TUBE_DP - df["PT-1100"]
    print(f"ΔP 재계산 최대 절대차 = {float((dp_calc - df['DP']).abs().max()):.3e} kPa  "
          f"(DP_reactor ≡ PT-1004 − 25 − PT-1100 확인)")
    a = df[df["Campaign"] == ANCHOR_CAMP]
    print(f"10/12 앵커  ΔP = {a['DP'].mean():.3f} kPa (기대 29.22) · "
          f"X = {a['X'].mean():.6f} (기대 0.615885)")

    # (2) 총유량 정의 검증
    print(f"10/12 총유량 = FT-1003({a['FT-1003'].mean():.0f}) + FT-1004({a['FT-1004'].mean():.0f})"
          f" = {a['총유량'].mean():.0f} kg/h (기대 18000)")
    alt = df["FT-1004"] * (1.0 + df["FRC-1004"])
    print(f"총유량 두 정의 최대차 = {float((alt - df['총유량']).abs().max()):.4f} kg/h "
          f"(FRC≡FT-1003/FT-1004 이므로 정의상 동일)")

    # (3) 분할
    for camp, g in df.groupby("Campaign"):
        role = ("학습" if camp in TRAIN_CAMPS else
                "held-out" if camp == HELD_CAMP else "앵커(제외)")
        print(f"  {camp:9} n={len(g):4}  {role:9} "
              f"X[{g['X'].min():.3f},{g['X'].max():.3f}]  ΔP[{g['DP'].min():.2f},{g['DP'].max():.2f}]")


# ==================================================== 1. 공선성 진단
def collinearity(tr: pd.DataFrame, cols: list[str]) -> None:
    hr("1. 공선성 진단 (학습 10/13–16)  [계산]")
    C = tr[cols].corr()
    print("상관행렬 r:")
    print("            " + "".join(f"{c:>10}" for c in cols))
    for i, c in enumerate(cols):
        print(f"{c:>11} " + "".join(f"{C.iloc[i, j]:>10.4f}" for j in range(len(cols))))

    Z = StandardScaler().fit_transform(tr[cols].to_numpy(float))
    print("\nVIF (분산팽창인자, 1/(1−R²_j); 10 초과면 강한 공선성):")
    for j, c in enumerate(cols):
        other = [k for k in range(len(cols)) if k != j]
        r2 = LinearRegression().fit(Z[:, other], Z[:, j]).score(Z[:, other], Z[:, j])
        vif = np.inf if r2 >= 1 - 1e-15 else 1.0 / (1.0 - r2)
        print(f"  {c:>10}  R²={r2:8.5f}   VIF={vif:10.2f}")
    print("  ※ 총유량 = FT-1004×(1+FRC-1004) 이므로 세 변수는 결정론적으로 얽혀 있다.")


# ============================== 2. LOCO 교차검증 (per-fold 스케일링)
def loco_curve(Xtr, ytr, groups, alphas, l1_ratio):
    """캠페인 단위 leave-one-campaign-out CV. 폴드마다 StandardScaler 재적합."""
    logo = LeaveOneGroupOut()
    folds = list(logo.split(Xtr, ytr, groups))
    err = np.zeros((len(alphas), len(folds)))
    for fi, (itr, iva) in enumerate(folds):
        sc = StandardScaler().fit(Xtr[itr])
        Za, Zb = sc.transform(Xtr[itr]), sc.transform(Xtr[iva])
        for ai, al in enumerate(alphas):
            m = (Lasso(alpha=al, max_iter=500000, tol=1e-9, random_state=SEED)
                 if l1_ratio >= 1.0 else
                 ElasticNet(alpha=al, l1_ratio=l1_ratio, max_iter=500000,
                            tol=1e-9, random_state=SEED))
            m.fit(Za, ytr[itr])
            err[ai, fi] = np.mean((ytr[iva] - m.predict(Zb)) ** 2)
    return err, [g for g in folds]


def fit_L1(df, tr, target, cols, tag):
    """LassoCV·ElasticNetCV (캠페인 LOCO). 반환: dict"""
    Xtr = tr[cols].to_numpy(float)
    ytr = tr[target].to_numpy(float)
    groups = tr["Campaign"].to_numpy()
    logo = LeaveOneGroupOut()
    cv_splits = list(logo.split(Xtr, ytr, groups))

    # 스케일러는 **학습셋에만** fit (held-out 미사용)
    sc = StandardScaler().fit(Xtr)
    Ztr = sc.transform(Xtr)

    lcv = LassoCV(cv=cv_splits, random_state=SEED, alphas=300,
                  eps=1e-6, max_iter=500000, tol=1e-9).fit(Ztr, ytr)
    ecv = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                       cv=cv_splits, random_state=SEED, alphas=300,
                       eps=1e-6, max_iter=500000, tol=1e-9).fit(Ztr, ytr)

    # per-fold 재스케일 교차확인 + 1-SE 규칙
    err, _ = loco_curve(Xtr, ytr, groups, lcv.alphas_, 1.0)
    m = err.mean(axis=1)
    se = err.std(axis=1, ddof=1) / np.sqrt(err.shape[1])
    i_min = int(np.argmin(m))
    thr = m[i_min] + se[i_min]
    i_1se = int(np.min(np.where(m <= thr)[0]))   # alphas 는 내림차순 → 앞쪽이 큰 α
    a_min, a_1se = lcv.alphas_[i_min], lcv.alphas_[i_1se]

    print(f"\n[{tag}] LassoCV  α*={lcv.alpha_:.6g}   "
          f"(per-fold 재스케일 CV 의 α*={a_min:.6g}, 1-SE α={a_1se:.6g})")
    print(f"        폴드(캠페인)별 CV-RMSE @α*: " +
          " ".join(f"{TRAIN_CAMPS[i]}={np.sqrt(err[i_min, i]):.5f}" for i in range(err.shape[1])))
    print(f"[{tag}] ElasticNetCV  α*={ecv.alpha_:.6g}  l1_ratio*={ecv.l1_ratio_:.3g}")

    def show(name, coef, intercept):
        order = np.argsort(-np.abs(coef))
        nz = [cols[j] for j in order if abs(coef[j]) > ZERO]
        dropped = [cols[j] for j in order if abs(coef[j]) <= ZERO]
        print(f"  {name} 표준화 계수 (크기순):")
        for j in order:
            mark = "선택" if abs(coef[j]) > ZERO else "탈락"
            print(f"    {cols[j]:>10}  {coef[j]:+12.6f}   {mark}")
        print(f"    절편 {intercept:+.6f} | 선택 {len(nz)}개 {nz} | 탈락 {dropped}")
        return nz

    nz_l = show("Lasso   ", lcv.coef_, lcv.intercept_)
    nz_e = show("ElasticNet", ecv.coef_, ecv.intercept_)

    # 1-SE 규칙에서의 선택
    sc1 = StandardScaler().fit(Xtr)
    m1se = Lasso(alpha=a_1se, max_iter=500000, tol=1e-9,
                 random_state=SEED).fit(sc1.transform(Xtr), ytr)
    nz_1se = [cols[j] for j in range(len(cols)) if abs(m1se.coef_[j]) > ZERO]
    print(f"  Lasso 1-SE 규칙(α={a_1se:.6g}) 선택: {nz_1se}")

    return dict(cols=cols, scaler=sc, lasso=lcv, enet=ecv,
                nz_lasso=nz_l, nz_enet=nz_e, nz_1se=nz_1se,
                alpha_min=float(lcv.alpha_), alpha_1se=float(a_1se),
                alpha_perfold=float(a_min))


# ============================================== 3. 정규화 경로 진입 순서
def entry_order(tr, target, cols, tag):
    Xtr = tr[cols].to_numpy(float)
    ytr = tr[target].to_numpy(float)
    Z = StandardScaler().fit_transform(Xtr)
    alphas, coefs, _ = lasso_path(Z, ytr, alphas=400, eps=1e-7)
    print(f"\n[{tag}] Lasso 정규화 경로 — α 감소 시 변수 진입 순서:")
    rows = []
    for j, c in enumerate(cols):
        hit = np.where(np.abs(coefs[j]) > ZERO)[0]
        rows.append((c, alphas[hit[0]] if len(hit) else -np.inf))
    rows.sort(key=lambda r: -r[1])
    for k, (c, a) in enumerate(rows, 1):
        print(f"   {k}. {c:>10}  진입 α = {a:.6g}" if a > 0 else
              f"   -. {c:>10}  경로 끝까지 미진입")
    return [r[0] for r in rows]


# ============================================== 4. held-out 성능 비교
def heldout(tr, ho, target, sets: dict, tag, logit=False):
    print(f"\n[{tag}] held-out(10/17–18) RMSE 비교   ※ held-out 은 여기서 처음 사용")
    out = {}
    for name, cols in sets.items():
        if not cols:
            print(f"  {name:<34} (선택 변수 없음 — 평가 불가)")
            continue
        Xtr, Xho = tr[cols].to_numpy(float), ho[cols].to_numpy(float)
        ytr, yho = tr[target].to_numpy(float), ho[target].to_numpy(float)
        if logit:
            ytr_f = np.log(ytr / (1 - ytr))
            p = LinearRegression().fit(Xtr, ytr_f).predict(Xho)
            pred = 1.0 / (1.0 + np.exp(-p))
        else:
            pred = LinearRegression().fit(Xtr, ytr).predict(Xho)
        r = rmse(yho, pred)
        out[name] = r
        print(f"  {name:<34} RMSE = {r:.6f}   변수 {cols}")
    return out


# ============================== 5. 탈상관 블록길이 L (데이터 추정)
def acf_within(res, groups, maxlag):
    """캠페인 경계를 넘지 않는 ACF. 지연 k 쌍만 모아 Pearson."""
    out = [1.0]
    for k in range(1, maxlag + 1):
        A, B = [], []
        for g in pd.unique(groups):
            r = res[groups == g]
            if len(r) > k:
                A.append(r[:-k]); B.append(r[k:])
        A, B = np.concatenate(A), np.concatenate(B)
        out.append(float(np.corrcoef(A, B)[0, 1]))
    return np.array(out)


def block_length(tr, target, cols, tag, maxlag=240):
    ytr = tr[target].to_numpy(float)
    X = tr[cols].to_numpy(float)
    res = ytr - LinearRegression().fit(X, ytr).predict(X)
    g = tr["Campaign"].to_numpy()
    ac = acf_within(res, g, maxlag)
    n = len(ytr)
    band = 2.0 / np.sqrt(n)
    inv_e = 1.0 / np.e
    L_e = next((k for k in range(1, maxlag + 1) if ac[k] < inv_e), None)
    L_b = next((k for k in range(1, maxlag + 1) if abs(ac[k]) < band), None)
    print(f"\n[{tag}] 물리셋 잔차 ACF (학습 n={n}, 유의밴드 2/√n={band:.4f})")
    print("   지연(분): " + " ".join(f"{k:>7}" for k in [1, 5, 10, 20, 30, 60, 90, 120, 180, 240]))
    print("   ACF     : " + " ".join(f"{ac[k]:>7.3f}" for k in [1, 5, 10, 20, 30, 60, 90, 120, 180, 240]))
    print(f"   ACF<1/e(0.3679) 최초 지연 L = {L_e} 분   |ACF|<2/√n 최초 지연 = {L_b} 분")
    L = L_e if L_e else maxlag
    print(f"   → 이동블록 부트스트랩 블록길이 L = {L} 분 [계산, 데이터 추정]")
    return L, ac


# ================================================ 6. 안정성 선택
def stability(tr, target, cols, alpha, L, tag, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    X = tr[cols].to_numpy(float)
    y = tr[target].to_numpy(float)
    camp = tr["Campaign"].to_numpy()
    idx_by_camp = {c: np.where(camp == c)[0] for c in TRAIN_CAMPS}

    # --- (i) 이동블록 부트스트랩 (캠페인 내부에서만 블록 추출) ---
    cnt = np.zeros(len(cols))
    n_ok = 0
    for _ in range(n_boot):
        take = []
        for c, ids in idx_by_camp.items():
            nc = len(ids)
            starts = np.arange(0, nc - L + 1) if nc >= L else np.array([0])
            nblk = int(np.ceil(nc / L))
            s = rng.choice(starts, size=nblk, replace=True)
            blk = np.concatenate([ids[st:st + L] for st in s])[:nc]
            take.append(blk)
        b = np.concatenate(take)
        Xb, yb = X[b], y[b]
        if np.ptp(yb) == 0:
            continue
        sc = StandardScaler().fit(Xb)
        m = Lasso(alpha=alpha, max_iter=500000, tol=1e-9, random_state=SEED)
        m.fit(sc.transform(Xb), yb)
        cnt += (np.abs(m.coef_) > ZERO)
        n_ok += 1
    freq_mb = 100.0 * cnt / max(n_ok, 1)

    # --- (ii) 캠페인 단위 재표집 (4개 중복허용) ---
    cnt2 = np.zeros(len(cols))
    n_ok2 = 0
    for _ in range(n_boot):
        pick = rng.choice(TRAIN_CAMPS, size=len(TRAIN_CAMPS), replace=True)
        b = np.concatenate([idx_by_camp[c] for c in pick])
        Xb, yb = X[b], y[b]
        if np.ptp(yb) == 0:
            continue
        sd = Xb.std(axis=0)
        if np.any(sd == 0):          # 상수 열 → 스케일 불가, 해당 열은 0 처리
            sd = np.where(sd == 0, 1.0, sd)
        Zb = (Xb - Xb.mean(axis=0)) / sd
        m = Lasso(alpha=alpha, max_iter=500000, tol=1e-9, random_state=SEED)
        m.fit(Zb, yb)
        cnt2 += (np.abs(m.coef_) > ZERO)
        n_ok2 += 1
    freq_cb = 100.0 * cnt2 / max(n_ok2, 1)

    print(f"\n[{tag}] 안정성 선택 (α={alpha:.6g} 고정, {n_boot}회)")
    print(f"  {'변수':>10} {'이동블록 L=%d (%d회 유효)' % (L, n_ok):>28} {'캠페인 재표집 (%d회 유효)' % n_ok2:>28}")
    order = np.argsort(-freq_mb)
    for j in order:
        print(f"  {cols[j]:>10} {freq_mb[j]:>22.1f} %  {freq_cb[j]:>24.1f} %")
    return dict(zip(cols, freq_mb)), dict(zip(cols, freq_cb))


# =========================================================== main
def main():
    df = load()
    verify(df)

    tr = df[df["Campaign"].isin(TRAIN_CAMPS)].reset_index(drop=True)
    ho = df[df["Campaign"] == HELD_CAMP].reset_index(drop=True)
    print(f"\n학습 {len(tr)}행(4캠페인) · held-out {len(ho)}행 · 앵커 10/12 {int((df['Campaign']==ANCHOR_CAMP).sum())}행 제외")

    collinearity(tr, FEATS)

    results = {}
    for target, tname in [("X", "전환율 X"), ("DP", "ΔP")]:
        hr(f"A. L1 변수선택 — 타깃 {tname}   (후보 7종, 캠페인 LOCO CV)")
        R = fit_L1(df, tr, target, FEATS, tname)
        R["entry"] = entry_order(tr, target, FEATS, tname)

        sets = {
            "물리셋 OLS (기준)": PHYS[target],
            "L1(Lasso α*) 선택셋 OLS": R["nz_lasso"],
            "L1(ElasticNet) 선택셋 OLS": R["nz_enet"],
            "L1(1-SE 규칙) 선택셋 OLS": R["nz_1se"],
            "후보 7종 전부 OLS": FEATS,
        }
        R["ho"] = heldout(tr, ho, target, sets, tname)

        # Lasso 자체(축소 계수)의 held-out
        Z = R["scaler"].transform(ho[FEATS].to_numpy(float))
        print(f"  {'Lasso 원모형(축소계수, 7입력)':<34} "
              f"RMSE = {rmse(ho[target].to_numpy(float), R['lasso'].predict(Z)):.6f}")

        hr(f"B. 안정성 선택 — 타깃 {tname}", "-")
        L, _ = block_length(tr, target, PHYS[target], tname)
        R["L"] = L
        # α 두 개에서 각각 본다. LassoCV 의 α* 는 매우 작아 거의 전부를 선택하므로
        # 빈도 100% 가 자동으로 나온다(정보량 없음). per-fold 재스케일 CV 의 α* 는
        # 희소해서 빈도가 실제 판별력을 갖는다. 둘 다 보고한다.
        R["freq_dense"] = stability(tr, target, FEATS, R["alpha_min"], L,
                                    tname + " @LassoCV α*(조밀)")
        R["freq_sparse"] = stability(tr, target, FEATS, R["alpha_perfold"], L,
                                     tname + " @per-fold CV α*(희소)")
        results[target] = R

    # ---------------- 전환율 로짓 공간 강건성 ----------------
    hr("A-보강. 전환율을 로짓 공간에서 다시 선택 (최종 레시피가 로짓 1차이므로)")
    tr2 = tr.copy(); tr2["Xlogit"] = np.log(tr2["X"] / (1 - tr2["X"]))
    R2 = fit_L1(df, tr2, "Xlogit", FEATS, "전환율 logit(X)")
    entry_order(tr2, "Xlogit", FEATS, "전환율 logit(X)")
    heldout(tr, ho, "X", {"물리셋 로짓 OLS": PHYS["X"],
                          "L1 선택셋 로짓 OLS": R2["nz_lasso"]}, "전환율 logit(X)", logit=True)

    # ---------------- r1 라벨 대조 ----------------
    hr("A-보강. r1 라벨(X_propene_recycle1)로 바꿔도 선택이 같은가")
    tr3 = tr.copy(); tr3["Xr1"] = tr3["X_propene_recycle1"].astype(float)
    R3 = fit_L1(df, tr3, "Xr1", FEATS, "전환율 r1")

    # ---------------- C. 누출 시연 ----------------
    hr("C. 누출 시연 (경고 — 권고 아님).  후보에 PT-1004·PT-1100 추가")
    cols_leak = FEATS + LEAK
    for target, tname in [("X", "전환율 X"), ("DP", "ΔP")]:
        RL = fit_L1(df, tr, target, cols_leak, f"{tname}+압력")
        entry_order(tr, target, cols_leak, f"{tname}+압력")
        sets = {"물리셋 OLS (누출 없음)": PHYS[target],
                "L1 선택셋 OLS (압력 포함 가능)": RL["nz_lasso"],
                "후보 9종 전부 OLS": cols_leak}
        heldout(tr, ho, target, sets, f"{tname}+압력")
        results[target + "_leak"] = RL

    hr("완료 — 해석은 docs/L1_변수선택_교차검증.md 참조")


if __name__ == "__main__":
    main()
