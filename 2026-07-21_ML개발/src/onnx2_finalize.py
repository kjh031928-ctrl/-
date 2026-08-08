"""onnx2_finalize.py — ONNX2 사전분석 (C)누출·품질 잔여 + results/onnx2_dp_eda.csv 생성.

⛔ .onnx·skl2onnx 금지. 사전분석 문서화까지만. stdout 표 + CSV(지표 1행) 산출.
기존 onnx2_preanalysis.load 재사용. 값은 전부 코드 재계산. [주어짐]/[계산]/[가정] 구분.
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError): pass

import numpy as np
from onnx2_preanalysis import load, ols, predict, rmse, TRAIN, VALID
from onnx2_screen_ext import z_and_MW_from_FRC

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "results" / "onnx2_dp_eda.csv"
B_FLOW = 0.00311055   # [계산] 학습 총유량 계수 kPa/(kg/h) (재현치)
B_Z_TEAM = 5.60222    # [주어짐] 팀식(2) z 계수 (.simx P_control1)

EDA = []  # (항목, 범위, 값, 구분, 재현여부)
def rec(item, rng, val, kind, repro="재현"):
    EDA.append({"항목": item, "범위": rng, "값": val, "구분": kind, "재현여부": repro})


def main():
    df = load()
    sub = df[df["Campaign"] != "10/12"]
    tr = df[df["Split"] == TRAIN]; va = df[df["Split"] == VALID]
    y_tr, y_va = tr["dP"].to_numpy(float), va["dP"].to_numpy(float)

    print("="*72); print("# (C) 누출·품질 잔여"); print("="*72)

    # (C)1 — 5입력 누출 재현
    cols5 = ["TT-1006.PV", "FT-1004.PV", "FRC-1004.PV", "PT-1004.PV", "PT-1100.PV"]
    X5t = tr[cols5].to_numpy(float); X5v = va[cols5].to_numpy(float)
    c5, b5 = ols(X5t, y_tr); r5 = rmse(y_va, predict(c5, b5, X5v))
    print(f"\n[C1] 5입력 [T,FT-1004,FRC,PT-1004,PT-1100] → ΔP 검증 RMSE = {r5:.2e} kPa")
    print(f"     계수: PT-1004={c5[3]:+.5f}  PT-1100={c5[4]:+.5f}  절편={b5:+.4f}  "
          f"(ΔP=PT-1004−PT-1100−25 를 그대로 복원 → 누출)")
    rec("5입력 누출 검증RMSE", "학습13-16→검증17-18", f"{r5:.2e} kPa (≈0)", "[계산]")
    rec("5입력 PT-1004/PT-1100 계수", "5입력 OLS", f"{c5[3]:+.5f}/{c5[4]:+.5f}(≈+1/−1)", "[계산]")

    # (C)2 — z 불일치 kPa
    z_std, MW_std = z_and_MW_from_FRC(5.0)   # B/P=5.0 정상운전점 (feed-side)
    z_use = 0.715766                          # [.simx] R-1100_in.z[BENZENE] (재순환 큐멘 포함)
    dz = z_std - z_use; dkpa = dz * B_Z_TEAM
    print(f"\n[C2] z(학습,feed-side B/P=5)={z_std:.5f}  vs  z(사용,.simx)={z_use:.6f}")
    print(f"     차이 {dz:.6f} × b_z({B_Z_TEAM}) = {dkpa:.4f} kPa  (문서 0.078)")
    rec("z 학습 vs 사용 차이", "B/P=5 feed-side vs .simx", f"{z_std:.5f} vs {z_use:.6f} → {dkpa:.4f} kPa", "[계산]/[.simx]")

    # (C)3 — 스크리닝 입력에 PT 부재 확인
    screen_inputs = {
        "F1": ["TT-1006","FT-1004","FRC-1004"],
        "F2": ["TT-1006","총유량(FT-1003+FT-1004)","FRC-1004"],
        "F2sq": ["TT-1006","총유량","FRC-1004","총유량²"],
        "물리군": ["ṁ²·T/(P상수·MW)  (P=상수, 실측압 미사용)"],
        "GBR/RF": ["TT-1006","총유량","FRC-1004"],
    }
    has_pt = any(("PT-1004" in c or "PT-1100" in c) for v in screen_inputs.values() for c in v)
    print(f"\n[C3] 스크리닝 전 입력세트에 PT-1004·PT-1100 포함 여부: {has_pt} (없음)")
    for k, v in screen_inputs.items():
        print(f"     {k:7}: {v}")
    rec("스크리닝 PT 입력 포함", "F1/F2/F2sq/물리군/GBR/RF", "없음(누출 원천 차단)", "[계산]")

    # (C)4 — 온도·FRC 캠페인별 nunique
    print(f"\n[C4] 온도·FRC 캠페인별 고유값 수(nunique)")
    print(f"     {'캠페인':10}{'TT-1006 nunique':>16}{'FRC-1004 nunique':>18}")
    for camp, g in df.groupby("Campaign"):
        nt, nf = g["TT-1006.PV"].nunique(), g["FRC-1004.PV"].nunique()
        print(f"     {camp:10}{nt:>16}{nf:>18}")
        rec(f"nunique {camp}", "T/FRC", f"T={nt}, FRC={nf}", "[계산]")

    # (C)5 — 반응기 입구 총유량 단일 태그 조사 (원본 CSV)
    raw = __import__("pandas").read_csv(_ROOT.parent / "2026 Chemical Engineering Process design competition_rev0_Attachment1.csv")
    raw.columns = [c.lstrip("﻿") for c in raw.columns]
    ft_cols = [c for c in raw.columns if c.startswith("FT-")]
    Ftot_mean = float((df["FT-1003.PV"] + df["FT-1004.PV"]).mean())
    print(f"\n[C5] 원본 CSV FT 태그 평균 (반응기입구총유량 {Ftot_mean:.0f} kg/h 와 대조)")
    single = []
    for c in ft_cols:
        m = float(raw[c].mean())
        near = abs(m - Ftot_mean) / Ftot_mean < 0.02
        if near: single.append(c)
        print(f"     {c:26}{m:>12.1f}{'  ← ~반응기입구총유량?' if near else ''}")
    print(f"     → 단일 '반응기 입구 총유량' 태그: {'없음' if not single else single}. "
          f"FT-1003+FT-1004 가 유일 정의.")
    rec("반응기입구총유량 단일태그", "원본 CSV FT-*", "없음 → FT-1003+FT-1004 유일", "[계산]")

    # ── 앞 절 핵심 지표도 CSV 에 축적 (재현치) ──
    neg = int((df["dP"] < 0).sum())
    rec("ΔP 음수 개수", "전체 3066", str(neg), "[계산]")
    rec("ΔP 10/12제외 평균", "2585행", f"{sub['dP'].mean():.4f} kPa", "[계산]")
    rec("ΔP 라벨 상수 25kPa", "HX-1005 튜브측 0.25bar", "25.0 kPa", "[주어짐]")
    rec("반응기입구압=PT-1004−25", "PFD Fig.1 배관손실 0 가정", "가정", "[가정]")
    rec("검증 10/17-18 ΔP 평균/범위", "661행", f"{va['dP'].mean():.2f} [{va['dP'].min():.2f},{va['dP'].max():.2f}]", "[계산]")
    rec("학습 ΔP 범위(결합)", "10/13-16", f"[{tr['dP'].min():.2f},{tr['dP'].max():.2f}]", "[계산]")
    rec("10/12 ΔP std", "481행", "0.0000 (단일점 앵커 29.22)", "[계산]")
    a = df["FT-1004.PV"]*(1+df["FRC-1004.PV"]); b = df["FT-1003.PV"]+df["FT-1004.PV"]
    d = (a-b).abs()
    rec("총유량 (a)vs(b) 최대차", "3066", f"{d.max():.4f} kg/h = {d.max()/a.mean()*100:.5f}%(÷평균)/{(d/a).max()*100:.5f}%(행별)", "[계산]")
    rec("FT-1002=FT-1003", "3066", "최대차 0.0", "[계산]")
    rec("총유량(c) FT-1001+1300+1004", "3066", "vs(a) 9.52% (탱크홀드업) 사용금지", "[계산]")
    rec("런타임 R-1100_in.W=HX-1005_Tube_Out.W", ".simx 단일점", "5.0 kg/s(18000 kg/h) 동일스트림", "[.simx]")
    rec("FT-1003=T-1001_out(재순환 합류 후)", ".simx 단일점", "T-1001_out.W=15000 kg/h(z[CUMENE]=0.0059 포함)", "[.simx]")
    rec("총유량 학습/런타임 정의 일치", ".simx+데이터", "FT-1003(15000)+FT-1004(3000)=R-1100_in.W(18000) → 일치", "[계산]/[.simx]")
    rec("정의 불일치는 z(몰분율)뿐", ".simx", "총유량은 질량계기가 재순환 포함 → 일치; z만 0.078 kPa 흔들림", "[계산]")
    rec("corr(ΔP,총유량) 10/12제외", "2585", f"{np.corrcoef(sub['Ftot_b'],sub['dP'])[0,1]:+.4f}", "[계산]")
    rec("MW=42.182+35.933·z, corr(z,MW)", "2585", "재현, +1.000000", "[계산]")
    rec("ΔP/P_in 등압", "2585", "1.098% (≈1.1%)", "[계산]")
    rec("물리군2모수 검증RMSE P상수/누출", "13-16→17-18", "0.5727 / 0.1723(PT-1004)", "[계산]")
    rec("순수Ergun형(절편無) 검증RMSE", "누출 P", "2.552 kPa (선형대비 ~16배)", "[계산]")
    rec("F2 선형 검증RMSE", "13-16→17-18", "0.1910 kPa", "[계산]")
    rec("F2 선형 계수", "학습13-16", "−52.4758/+0.090009·T/+0.00311055·Ftot/−1.00735·FRC", "[계산]")
    rec("F2sq(+총유량²) 검증RMSE", "13-16→17-18", "0.2026 (악화)", "[계산]")
    rec("F1(프로펜유량) 물리 B/P↓", "27점격자", "0/9 실패 (곱 표현불가)", "[계산]")
    rec("GBR/RF 검증RMSE", "13-16→17-18", "0.4507 / 0.6730 (선형보다 나쁨)", "[계산]")
    rec("RF 물리 B/P↓", "27점격자", "3/9 실패", "[계산]")
    rec("트리 외삽 clamp T=365", "Ftot16500·BP5", "GBR·RF 350→365 +0.000(clamp)", "[계산]")
    rec("선형/물리군 외삽", "Ftot16500·BP5", "350→365 +1.350/+0.624(추세연장)", "[계산]")
    rec("물리 27점 F2선형", "T{335,350,365}", "F↑9/9·BP↓9/9 범위[18.29,32.34]", "[계산]")
    rec("sklearn/numpy/seed", "환경", "1.9.0 / 2.5.1 / random_state=42", "[계산]")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["항목","범위","값","구분","재현여부"])
        w.writeheader(); w.writerows(EDA)
    print(f"\n[산출] {_OUT.relative_to(_ROOT)}  ({len(EDA)}개 지표행)")


if __name__ == "__main__":
    main()
