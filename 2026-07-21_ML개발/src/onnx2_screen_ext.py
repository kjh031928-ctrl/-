"""onnx2_screen_ext.py — ONNX2 스크리닝 확장 (iii)물리군 + (iv)GBR·RF. stdout 전용.

⛔ 파일·.onnx 생성 금지, skl2onnx 금지, 채택 결론 금지. 선형이 충분한지 데이터로 확인만.
기존 onnx2_preanalysis.py 의 load/ols/predict/rmse/z정의를 재사용(새 정의 금지).
물리 27점 격자는 T 상단을 365 로 올려 외삽까지 시험(학습 최고 T=359.99).
"""
from __future__ import annotations
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError): pass

import numpy as np
from onnx2_preanalysis import load, ols, predict, rmse, TRAIN, VALID

# 물성 [주어짐] AVEVA / rev2 App B
MW_B, MW_P, MW_PA = 78.115, 42.081, 44.097
MFRAC_P, MFRAC_PA = 0.94773, 0.05227   # 프로펜피드 질량분율 [주어짐 App B]

def z_and_MW_from_FRC(FRC):
    """z(벤젠 몰분율)·평균분자량 MW 를 FRC(질량비)로부터 계산. 재순환 미포함(한계 명시).
    n_b = FRC·FT/78.115, n_p=0.94773·FT/42.081, n_pa=0.05227·FT/44.097 (FT 소거)."""
    n_b = FRC / MW_B
    n_p = MFRAC_P / MW_P
    n_pa = MFRAC_PA / MW_PA
    tot = n_b + n_p + n_pa
    z = n_b / tot
    mass = FRC + 1.0                       # (benzene + propene-stream)  ∝ FT
    MW = mass / tot                        # kg/kmol (= Ftot_a/F_kmol, FT 소거)
    return z, MW


def main():
    import sklearn
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    print(f"[env] sklearn {sklearn.__version__}  numpy {np.__version__}")

    df = load()
    sub = df[df["Campaign"] != "10/12"].copy()   # 10/12 제외
    tr = df[df["Split"] == TRAIN].copy()
    va = df[df["Split"] == VALID].copy()

    # 데이터의 몰유량으로 z·MW (기존 데이터 컬럼 사용, 재순환 미포함)
    for g in (df, sub, tr, va):
        nb, npn, npa = g["n_benzene_kmol_h"], g["n_propene_kmol_h"], g["n_propane_kmol_h"]
        Fk = nb + npn + npa
        g["F_kmol"] = Fk
        g["z"] = nb / Fk
        g["MW"] = g["Ftot_a"] / Fk            # 질량/몰 (kg/kmol)
        g["T_K"] = g["TT-1006.PV"] + 273.15

    # ── (iii)-0  MW = a + b·z 재현 + corr(z,MW) ─────────────────────────────
    print("\n" + "="*72)
    print("# (iii) 물리군 — MW=a+b·z 재현, z·MW 상관, 등압 확인")
    print("="*72)
    A = np.column_stack([np.ones(len(sub)), sub["z"]])
    (b0, b1), *_ = np.linalg.lstsq(A, sub["MW"], rcond=None)
    corr_zMW = np.corrcoef(sub["z"], sub["MW"])[0, 1]
    print(f"  MW = {b0:.3f} + {b1:.3f}·z   (문서 42.182 + 35.933·z)")
    print(f"  corr(z, MW) = {corr_zMW:.6f}   (문서 +1.000000)")
    # 등압: ΔP/P_in (P_in ≈ PT-1004)
    ratio = (sub["dP"] / sub["PT-1004.PV"]).mean() * 100
    print(f"  ΔP/P_in 평균 = {ratio:.3f}%  (문서 ≈1.1% → P 상수 흡수 근거)  [한계: z 재순환 미포함]")

    # ── (iii)-1  물리군 두 형태가 대수적으로 동일함 확인 ───────────────────
    def groups(g, P):
        mass = g["Ftot_b"].to_numpy(float)          # ṁ [kg/h]
        Fk = g["F_kmol"].to_numpy(float)            # [kmol/h]
        MW = g["MW"].to_numpy(float); T = g["T_K"].to_numpy(float)
        Pv = P(g)
        g_mass = mass**2 * T / (Pv * MW)            # ṁ²T/(P·MW)
        g_mol = Fk**2 * MW * T / Pv                 # F²·MW·T/P
        return g_mass, g_mol
    # P 상수(=1, 등압 → 흡수) : 누출 없음 / P=PT-1004·PT-1100 : 누출(참고용)
    P_cases = {
        "P=상수(등압,누출없음)": lambda g: np.ones(len(g)),
        "P=PT-1004(누출·참고)": lambda g: g["PT-1004.PV"].to_numpy(float),
        "P=PT-1100(누출·참고)": lambda g: g["PT-1100.PV"].to_numpy(float),
    }
    gm_tr, gmol_tr = groups(tr, P_cases["P=상수(등압,누출없음)"])
    ident = np.corrcoef(gm_tr, gmol_tr)[0, 1]
    print(f"\n  질량형 ṁ²T/(P·MW) ≡ 몰형 F²·MW·T/P (ṁ=F·MW) → corr = {ident:.8f} (동일군)")

    # ── (iii)-2  물리군 2모수 회귀 (절편+물리군1) 학습→검증 ────────────────
    print("\n  [물리군 2모수 회귀]  학습 10/13-16 → 검증 10/17-18")
    print(f"  {'P 처리':22}{'검증RMSE(절편有)':>16}{'검증RMSE(절편無)':>16}{'절편kPa':>12}{'물리평균kPa':>12}")
    y_tr = tr["dP"].to_numpy(float); y_va = va["dP"].to_numpy(float)
    physgroup_pred = None
    for name, Pf in P_cases.items():
        Gtr, _ = groups(tr, Pf); Gva, _ = groups(va, Pf)
        # 절편 있음
        cff, ic = ols(Gtr.reshape(-1, 1), y_tr)
        r_int = rmse(y_va, predict(cff, ic, Gva.reshape(-1, 1)))
        phys_mean = float((cff[0] * Gtr).mean())
        # 절편 없음 (순수 Ergun형)
        b_noint = float(np.linalg.lstsq(Gtr.reshape(-1, 1), y_tr, rcond=None)[0][0])
        r_noint = rmse(y_va, Gva * b_noint)
        print(f"  {name:22}{r_int:>16.4f}{r_noint:>16.4f}{ic:>12.2f}{phys_mean:>12.2f}")
        if name.startswith("P=상수"):
            physgroup_pred = predict(cff, ic, groups(sub, Pf)[0].reshape(-1, 1))

    # 팀식(2) (T,F_kmol,z) 예측과 물리군 예측 상관 (10/12 제외 전체)
    Xteam = sub[["TT-1006.PV", "F_kmol", "z"]].to_numpy(float)
    cft, ict = ols(Xteam, sub["dP"].to_numpy(float))
    team_pred = predict(cft, ict, Xteam)
    corr_pg_team = np.corrcoef(physgroup_pred, team_pred)[0, 1]
    print(f"  물리군(P상수) 예측 vs 팀식(2) 예측 상관 = {corr_pg_team:.6f}  (문서 0.999394)")

    # ── (iv)  비선형 ML — sklearn 기본값 + random_state=42 ────────────────
    print("\n" + "="*72)
    print("# (iv) 비선형 기준선 (sklearn 기본값, random_state=42, 입력 F2)")
    print("="*72)
    Xtr = tr[["TT-1006.PV", "Ftot_b", "FRC-1004.PV"]].to_numpy(float)
    Xva = va[["TT-1006.PV", "Ftot_b", "FRC-1004.PV"]].to_numpy(float)
    models = {
        "GBR": GradientBoostingRegressor(random_state=42),
        "RF":  RandomForestRegressor(random_state=42),
    }
    fitted = {}
    for nm, mdl in models.items():
        mdl.fit(Xtr, y_tr)
        fitted[nm] = mdl
        print(f"  {nm:4} 검증RMSE = {rmse(y_va, mdl.predict(Xva)):.4f}")

    # F2 선형·물리군 예측기 (격자·외삽용)
    coef_lin, ic_lin = ols(Xtr, y_tr)
    def pred_lin(T, F, R): return ic_lin + coef_lin[0]*T + coef_lin[1]*F + coef_lin[2]*R
    Gtr_c, _ = groups(tr, P_cases["P=상수(등압,누출없음)"])
    cff_c, ic_c = ols(Gtr_c.reshape(-1, 1), y_tr)
    def pred_phys(T, F, R):
        z, MW = z_and_MW_from_FRC(R)
        G = F**2 * (T + 273.15) / MW           # P 상수
        return ic_c + cff_c[0]*G
    def pred_tree(mdl, T, F, R):
        return float(mdl.predict(np.array([[T, F, R]]))[0])

    predfun = {
        "F2 선형": lambda T, F, R: float(pred_lin(T, F, R)),
        "물리군":  lambda T, F, R: float(pred_phys(T, F, R)),
        "GBR":    lambda T, F, R: pred_tree(fitted["GBR"], T, F, R),
        "RF":     lambda T, F, R: pred_tree(fitted["RF"], T, F, R),
    }

    # ── 공통 물리 27점 격자 (T 상단 365 = 외삽) ───────────────────────────
    Ts, Fs, Rs = [335.0, 350.0, 365.0], [15000.0, 16500.0, 18000.0], [4.0, 5.0, 6.0]
    print("\n" + "="*72)
    print("# 물리 27점 격자  T{335,350,365} × Ftot{15000,16500,18000} × B/P{4,5,6}")
    print("#   (학습 최고 T=359.99 → 360~365 는 순수 외삽)")
    print("="*72)
    print(f"  {'모델':10}{'총유량↑→ΔP↑':>14}{'B/P↑→ΔP↓':>12}{'예측범위kPa':>18}")
    for nm, f in predfun.items():
        up = sum(1 for T in Ts for R in Rs
                 if f(T, Fs[0], R) < f(T, Fs[1], R) < f(T, Fs[2], R))
        dn = sum(1 for T in Ts for F in Fs
                 if f(T, F, Rs[0]) > f(T, F, Rs[1]) > f(T, F, Rs[2]))
        allv = [f(T, F, R) for T in Ts for F in Fs for R in Rs]
        print(f"  {nm:10}{up:>10}/9{dn:>10}/9   [{min(allv):.2f}, {max(allv):.2f}]")

    # ── 외삽 거동 직접 시험 (Ftot=16500, B/P=5, T=335/350/365) ─────────────
    print("\n" + "="*72)
    print("# 외삽 거동  Ftot=16500, B/P=5 고정, T=335·350·365 예측 ΔP [kPa]")
    print("="*72)
    print(f"  {'모델':10}{'T=335':>10}{'T=350':>10}{'T=365(외삽)':>14}{'350→365 증가':>14}")
    for nm, f in predfun.items():
        v = [f(T, 16500.0, 5.0) for T in [335.0, 350.0, 365.0]]
        print(f"  {nm:10}{v[0]:>10.3f}{v[1]:>10.3f}{v[2]:>14.3f}{v[2]-v[1]:>+14.3f}")


if __name__ == "__main__":
    main()
