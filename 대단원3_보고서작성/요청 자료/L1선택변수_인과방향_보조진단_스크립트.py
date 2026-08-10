"""L1선택변수_인과방향_보조진단_스크립트.py
=====================================================================
목적: L1(Lasso)이 물리셋 밖에서 뽑아온 변수 — FT-1001(신선벤젠),
      FT-1300(재순환벤젠), TT-1004(프로펜 공급온도) — 가
      (a) 정말 독립 정보인지, (b) 결과 쪽(하류) 변수인지,
      (c) 다른 입력의 선형 그림자인지를 데이터로 판정한다.

판정 도구 (전부 데이터에서 직접 계산, 가정 함수형태 없음):
  1. 캠페인별 상관 — 전체 상관은 캠페인 혼합효과로 부풀 수 있다(범위 표기 원칙).
  2. 선행/후행 교차상관 — 타깃이 변수보다 먼저 움직이면 그 변수는 결과 쪽이다.
     (프로젝트가 PT-1100 을 배제할 때 쓴 것과 같은 검사, 루트 CLAUDE.md 참조)
  3. 입력 3종으로의 회귀 R²·VIF — 남은 고유정보 비율.
  4. 편상관 — 물리셋 효과를 양쪽에서 제거한 뒤의 잔여 설명력.

읽기 전용. seed 무관(난수 미사용). 표기 [주어짐]/[계산]/[가정].
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
from sklearn.linear_model import LinearRegression

_ROOT = Path(__file__).resolve().parents[1]
_ATT1 = _ROOT / "2026 Chemical Engineering Process design competition_rev0_Attachment1.csv"
_DPX = _ROOT / "DP_data.xlsx"
_LAB = _ROOT / "2026-07-21_ML개발" / "data" / "processed" / "labels_recycle1.csv"

HX1005_TUBE_DP = 25.0          # kPa [주어짐] rev2 App B Table 4
TRAIN_CAMPS = ["10/13", "10/14", "10/15", "10/16"]
PHYS = {"X": ["TT-1006", "FT-1004", "FRC-1004"],
        "DP": ["TT-1006", "총유량", "FRC-1004"]}
SUSPECT = ["FT-1001", "FT-1300", "TT-1004"]


def hr(t, ch="="):
    print("\n" + ch * 78); print(t); print(ch * 78)


def load():
    att = pd.read_csv(_ATT1).rename(columns=lambda c: c)
    att = att.rename(columns={att.columns[0]: "Timestamp"})
    att["Timestamp"] = pd.to_datetime(att["Timestamp"], utc=True).dt.tz_localize(None)

    def pick(tag):
        hit = [c for c in att.columns if c.startswith(tag + ".PV")]
        assert len(hit) == 1
        return att[hit[0]].astype(float)

    df = pd.DataFrame({"Timestamp": att["Timestamp"]})
    for tag in ["TT-1006", "FT-1004", "FT-1003", "FRC-1004", "TT-1004",
                "FT-1001", "FT-1300", "PT-1004", "PT-1100", "LC-1001"]:
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


def corr(a, b):
    """상수열(또는 부동소수점 잔차 수준의 변동)이면 nan.

    10/12 는 481행이 같은 정상상태 해의 반복이라 std 가 1e-16 수준이다.
    가드를 std==0 으로만 두면 corrcoef 가 ±1 이라는 무의미한 값을 돌려준다.
    → 변동폭(ptp)이 값 크기의 1e-12 미만이면 '정의 불가'로 처리한다.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    for v in (a, b):
        scale = max(abs(np.mean(v)), 1.0)
        if np.ptp(v) <= 1e-12 * scale:
            return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def sec_campaign_corr(df):
    hr("1. 캠페인별 상관 — 전체 상관은 캠페인 혼합으로 부풀 수 있다  [계산]")
    pairs = [("FT-1001", "X"), ("FT-1300", "X"), ("TT-1004", "X"),
             ("FT-1001", "DP"), ("FT-1300", "DP"), ("TT-1004", "DP"),
             ("TT-1004", "총유량"), ("TT-1004", "FT-1004"), ("TT-1004", "FT-1003"),
             ("FT-1001", "FT-1004"), ("FT-1300", "FRC-1004"),
             ("FT-1001", "LC-1001"), ("LC-1001", "X")]
    camps = TRAIN_CAMPS + ["10/17-18", "10/12"]
    print(f"{'쌍':>22} {'학습전체':>9}" + "".join(f"{c:>10}" for c in camps))
    tr = df[df["Campaign"].isin(TRAIN_CAMPS)]
    for a, b in pairs:
        row = f"{a+' vs '+b:>22} {corr(tr[a], tr[b]):>9.4f}"
        for c in camps:
            g = df[df["Campaign"] == c]
            v = corr(g[a], g[b])
            row += f"{v:>10.4f}" if np.isfinite(v) else f"{'—':>10}"
        print(row)
    print("  ※ 10/12 는 전 컬럼 상수(단일 정상상태 반복) → 상관 정의 불가(—).")


def sec_leadlag(df):
    hr("2. 선행/후행 교차상관 — 타깃이 먼저 움직이면 그 변수는 '결과 쪽'  [계산]")
    print("  corr(target[t], var[t+k]) 를 k=−30…+30분에서 계산.")
    print("  k>0 최대  → 타깃이 변수보다 **선행**(변수가 결과 쪽).")
    print("  k<0 최대  → 변수가 타깃보다 선행(변수가 원인 쪽).")
    ks = np.arange(-30, 31)
    for tgt in ["X", "DP"]:
        print(f"\n  [타깃 {tgt}]")
        for v in SUSPECT + PHYS[tgt]:
            best_k, best_r = None, -2
            prof = []
            for k in ks:
                A, B = [], []
                for c in TRAIN_CAMPS:
                    g = df[df["Campaign"] == c]
                    y = g[tgt].to_numpy(float); x = g[v].to_numpy(float)
                    if k >= 0:
                        A.append(y[:len(y) - k] if k else y); B.append(x[k:])
                    else:
                        A.append(y[-k:]); B.append(x[:len(x) + k])
                r = corr(np.concatenate(A), np.concatenate(B))
                prof.append(r)
                if np.isfinite(r) and r > best_r:
                    best_r, best_k = r, int(k)
            r0 = prof[list(ks).index(0)]
            tag = ("타깃 선행 → 결과 쪽 의심" if best_k > 2 else
                   "변수 선행 → 원인 쪽" if best_k < -2 else "동시(±2분 내)")
            print(f"    {v:>10}  k*={best_k:+4d}분  r*={best_r:+.4f}  (k=0 에서 {r0:+.4f})  {tag}")


def sec_unique_info(df):
    hr("3. 물리셋으로 회귀했을 때 남는 고유정보 (학습 10/13–16)  [계산]")
    tr = df[df["Campaign"].isin(TRAIN_CAMPS)]
    for tgt in ["X", "DP"]:
        print(f"\n  [물리셋 {PHYS[tgt]}]  타깃 {tgt}")
        P = tr[PHYS[tgt]].to_numpy(float)
        for v in SUSPECT + [c for c in ["FT-1004", "총유량"] if c not in PHYS[tgt]]:
            y = tr[v].to_numpy(float)
            r2 = LinearRegression().fit(P, y).score(P, y)
            vif = np.inf if r2 >= 1 - 1e-15 else 1 / (1 - r2)
            # 편상관: 물리셋 효과를 양쪽에서 제거
            res_v = y - LinearRegression().fit(P, y).predict(P)
            t = tr[tgt].to_numpy(float)
            res_t = t - LinearRegression().fit(P, t).predict(P)
            pc = corr(res_v, res_t)
            print(f"    {v:>10}  R²(물리셋→변수)={r2:7.5f}  VIF={vif:9.2f}  "
                  f"고유정보={100*(1-r2):5.2f}%   편상관(변수,{tgt})={pc:+.4f}")


def sec_massbalance(df):
    hr("4. FT-1001(신선벤젠)이 전환율의 물질수지 결과인지 직접 확인  [계산]")
    tr = df[df["Campaign"].isin(TRAIN_CAMPS)].copy()
    # 반응 소비 프로펜 질량 ∝ FT-1004 × X (프로펜 질량분율 0.94773 [주어짐])
    W_PROPENE = 0.94773        # [주어짐] CLAUDE.md §2.3 프로펜 질량분율
    M_B, M_P = 78.11, 42.08    # g/mol [주어짐] 벤젠·프로펜 몰질량(문헌)
    tr["벤젠소비_kg_h"] = tr["FT-1004"] * W_PROPENE * tr["X"] / M_P * M_B
    r = corr(tr["벤젠소비_kg_h"], tr["FT-1001"])
    print(f"  벤젠 소비량(=FT-1004·0.94773·X/42.08·78.11) vs FT-1001 상관 = {r:+.4f} (학습 전체)")
    for c in TRAIN_CAMPS:
        g = tr[tr["Campaign"] == c]
        print(f"    {c}: {corr(g['벤젠소비_kg_h'], g['FT-1001']):+.4f}  "
              f"(평균 소비 {g['벤젠소비_kg_h'].mean():8.1f} vs FT-1001 {g['FT-1001'].mean():8.1f} kg/h)")
    print("  ※ FT-1001 은 T-1001 액위제어가 채우는 보충류다. 소비량과 같이 움직이면")
    print("     '전환율의 결과'라는 뜻이며, 입력으로 쓰면 순환 의존이 된다.")


def main():
    df = load()
    print(f"numpy {np.__version__} · pandas {pd.__version__} · 난수 미사용")
    sec_campaign_corr(df)
    sec_leadlag(df)
    sec_unique_info(df)
    sec_massbalance(df)


if __name__ == "__main__":
    main()
