"""stage3_features.py — Stage 3 입출력 변수 비교 (신설 2026-08-01).

고정: 타깃 변환 logit(Stage 2 채택), 주 분할, Stage 0 프로토콜. 대상 모델 Ridge Poly2 / Poly2.
3-A 입력 세트 7종(I1/I1b/I1c/I2/I3/I4/I5) — 압력세트는 성능 좋아도 제약으로 채택 불가(제약 심사).
3-B 출력 3종(Y1 전환율→환산 / Y2 몰유량 / Y3 질량유량) — 같은 큐멘 kg/h 상대오차로 통일 비교.
승자선택=블록20%+제약심사. 검증 확인용.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError): pass

import numpy as np
import numpy.linalg as la
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import evaluation as ev
import data_loader as dl

_ROOT = Path(__file__).resolve().parents[1]
MW_CUMENE = 120.196
RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)

# ---- 입력 세트 정의: df → (X ndarray, 이름목록, 물리격자지원, 압력변수목록) ----
def _cols(df, cols): return df[cols].to_numpy(dtype=np.float64)

INPUT_SETS = {
    "I1":  dict(build=lambda d: (_cols(d, ["TT-1006.PV","FT-1004.PV","FRC-1004.PV"]),
                                 ["TT-1006","FT-1004","FRC-1004"]), phys="I1", press=[]),
    "I1b": dict(build=lambda d: (np.column_stack([d["TT-1006.PV"], d["FT-1003.PV"]+d["FT-1004.PV"],
                                 d["FRC-1004.PV"]]).astype(float),
                                 ["TT-1006","총유량(FT1003+FT1004)","FRC-1004"]), phys="I1b", press=[]),
    "I1c": dict(build=lambda d: (np.column_stack([d["TT-1006.PV"], d["FT-1004.PV"], d["FRC-1004.PV"],
                                 d["FT-1004.PV"]*d["FRC-1004.PV"]]).astype(float),
                                 ["TT-1006","FT-1004","FRC-1004","FT1004×FRC"]), phys="I1c", press=[]),
    "I2":  dict(build=lambda d: (_cols(d, ["TT-1006.PV","FT-1004.PV","FRC-1004.PV","PT-1004.PV"]),
                                 ["TT-1006","FT-1004","FRC-1004","PT-1004"]), phys=None, press=["PT-1004.PV"]),
    "I3":  dict(build=lambda d: (_cols(d, ["TT-1006.PV","FT-1004.PV","FRC-1004.PV","PT-1100.PV"]),
                                 ["TT-1006","FT-1004","FRC-1004","PT-1100"]), phys=None, press=["PT-1100.PV"]),
    "I4":  dict(build=lambda d: (_cols(d, ["TT-1006.PV","FT-1004.PV","FRC-1004.PV","PT-1004.PV","PT-1100.PV"]),
                                 ["TT-1006","FT-1004","FRC-1004","PT-1004","PT-1100"]), phys=None,
                press=["PT-1004.PV","PT-1100.PV"]),
    "I5":  dict(build=lambda d: (np.column_stack([d["TT-1006.PV"], d["FT-1004.PV"], d["FRC-1004.PV"],
                                 d["PT-1004.PV"]-d["PT-1100.PV"]]).astype(float),
                                 ["TT-1006","FT-1004","FRC-1004","DP(PT1004-PT1100)"]), phys=None,
                press=["PT-1004.PV","PT-1100.PV"]),
}

MODELS = {
    "Ridge Poly2": lambda: ev.ModelSpec("Ridge Poly2","선형",
        lambda p: make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=p["alpha"])),
        [{"alpha":a} for a in RIDGE_ALPHAS], transform=ev.TRANSFORMS["logit"]),
    "Poly2": lambda: ev.ModelSpec("Poly2","선형",
        lambda p: make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),LinearRegression()),
        [{}], transform=ev.TRANSFORMS["logit"]),
}


def make_splits(df, build, label="X_propene_recycle1"):
    tr = df[df.Split == "학습"]; va = df[df.Split == "검증"]
    Xtr, names = build(tr); Xva, _ = build(va)
    return ev.Splits(X_train=Xtr, y_train=tr[label].to_numpy(float), camp_train=tr["Campaign"].to_numpy(),
                     X_valid=Xva, y_valid=va[label].to_numpy(float),
                     npr_valid=va["n_propene_kmol_h"].to_numpy(float), feature_names=names, df=df)


def phys_grid(setname):
    """물리격자 27점(I1/I1b/I1c 만). 반환 (X, meta) 또는 None(압력세트)."""
    rows, meta = [], []
    for total in ev.GRID_TOTAL:
        for bp in ev.GRID_BP:
            ft = total / (1.0 + bp)
            for T in ev.GRID_T:
                if setname == "I1": rows.append([T, ft, bp])
                elif setname == "I1b": rows.append([T, total, bp])
                elif setname == "I1c": rows.append([T, ft, bp, ft*bp])
                else: return None
                meta.append({"T": T, "total": total, "bp": bp})
    return np.array(rows, float), meta


def phys_check(spec, sp, setname):
    g = phys_grid(setname)
    if g is None:
        return {"applicable": False}
    Xg, meta = g
    est = ev.fit_on_logit(spec, ev.tune(spec, sp, "block20")[0], sp.X_train, sp.y_train)
    preds = ev.predict_conv(est, Xg)
    in_range = int(np.sum((preds >= 0) & (preds <= 1)))
    by = {}
    for m, p in zip(meta, preds): by.setdefault((m["total"], m["bp"]), []).append((m["T"], p))
    slic = sum(1 for s in by.values() if all(x2>x1 for (_,x1),(_,x2) in zip(sorted(s),sorted(s)[1:])))
    return {"applicable": True, "in_range": in_range, "n": len(preds),
            "T_mono_slices": slic, "n_slices": len(by)}


def vif_of(df, target_col, base=("TT-1006.PV","FT-1004.PV","FRC-1004.PV")):
    X = df[list(base)].to_numpy(); y = df[target_col].to_numpy()
    r2 = LinearRegression().fit(X, y).score(X, y)
    return 1.0/(1.0-r2)


def pressure_coef_signs(df, press_col):
    """logit-1차(I1+press) 를 캠페인별 적합해 press 계수 부호(10/14 vs 10/16) 확인."""
    out = {}
    for c in ("10/14", "10/16"):
        sub = df[df.Campaign == c]
        X = sub[["TT-1006.PV","FT-1004.PV","FRC-1004.PV",press_col]].to_numpy()
        z = ev.to_logit(sub["X_propene_recycle1"].to_numpy())
        b = LinearRegression().fit(X, z).coef_[3]
        out[c] = float(b)
    return out


def _merge_ft1003(df):
    """총유량(I1b)용 FT-1003.PV 를 원본 CSV에서 Timestamp 로 병합(정본에 없는 컬럼). 원본 읽기전용."""
    import pandas as pd
    raw = pd.read_csv(_ROOT.parent / "2026 Chemical Engineering Process design competition_rev0_Attachment1.csv")
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    raw["Timestamp"] = raw["Timestamp"].dt.tz_localize(None)   # UTC-aware → naive
    raw = raw.rename(columns={"FT-1003.PV Value kg/h": "FT-1003.PV"})[["Timestamp", "FT-1003.PV"]]
    df = df.copy(); df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)
    m = df.merge(raw, on="Timestamp", how="left")
    if m["FT-1003.PV"].isna().any():
        raise RuntimeError("FT-1003 병합 결측 발생(Timestamp 불일치)")
    return m


def main():
    cfg = dl.load_config(); df = _merge_ft1003(dl.load_data(cfg))
    boot = ev.block_bootstrap_indices(661, ev.BLOCK_LEN_MIN, ev.N_BOOT, ev.SEED)
    allc = df[df.Campaign != "10/12"]; tr = df[df.Split=="학습"]; va = df[df.Split=="검증"]

    rows = []
    # ===== 3-A 입력 세트 (출력 Y1=logit) =====
    print("== 3-A 입력 세트 (Y1 logit) ==")
    for mname, mk in MODELS.items():
        for sname, sdef in INPUT_SETS.items():
            spec = mk()
            sp = make_splits(df, sdef["build"])
            r = ev.evaluate_model(spec, sp, boot, do_physical=False)
            ph = phys_check(spec, sp, sname)
            # 제약 심사
            press = sdef["press"]
            case_specifiable = len(press) == 0            # 압력 미지정 → 지정 불가
            circular = len(press) > 0                     # 반응기 압력=플로우시트 종속량
            vifs = {p: round(vif_of(allc, p), 1) for p in press}
            signs = {}
            for p in press:
                s = pressure_coef_signs(df, p); signs[p] = s
            rows.append({
                "part": "3A_input", "model": mname, "input_set": sname, "output": "Y1(X→kgh)",
                "n_feat": sp.X_train.shape[1],
                "block20": round(r["block20"], 6), "valid_rmse": round(r["valid"]["rmse"], 6),
                "jack60_ci": f"[{r['jack60']['ci_low']:.5f},{r['jack60']['ci_high']:.5f}]",
                "cumene_pct": round(r["cumene"]["pct_of_mean"], 4),
                "loco5_mean": round(r["loco5_mean"], 6), "loco5_worst": round(r["loco5_worst"], 6),
                "sens": round(r["sensitivity"]["max_pred_shift_rms"], 6),
                "phys": ("N/A(압력)" if not ph["applicable"] else
                         f"{ph['in_range']}/{ph['n']}·{ph['T_mono_slices']}/{ph['n_slices']}"),
                "onnx": r["onnx"]["ok"],
                "case_specifiable": case_specifiable, "circular_dep": circular,
                "VIF": (";".join(f"{k.split('.')[0]}={v}" for k,v in vifs.items()) if vifs else "-"),
                "press_coef_sign_flip": (";".join(
                    f"{p.split('.')[0]}:10/14{signs[p]['10/14']:+.5f}/10/16{signs[p]['10/16']:+.5f}"
                    for p in press) if press else "-"),
            })
            print(f"  {mname:12} {sname:4} block20={r['block20']:.5f} valid={r['valid']['rmse']:.5f} "
                  f"cumene%={r['cumene']['pct_of_mean']:.3f} phys={rows[-1]['phys']}")

    # ===== 3-B 출력 (입력 I1) — 같은 큐멘 kg/h 상대오차 =====
    print("\n== 3-B 출력 (I1) 큐멘 kg/h 상대오차 ==")
    npr_tr = tr["n_propene_kmol_h"].to_numpy(float); npr_va = va["n_propene_kmol_h"].to_numpy(float)
    Xtr_i1 = tr[["TT-1006.PV","FT-1004.PV","FRC-1004.PV"]].to_numpy(float)
    Xva_i1 = va[["TT-1006.PV","FT-1004.PV","FRC-1004.PV"]].to_numpy(float)
    Xg = tr["X_propene_recycle1"].to_numpy(float); Xg_va = va["X_propene_recycle1"].to_numpy(float)
    cum_true = Xg_va * npr_va * MW_CUMENE
    def relpct(pred_kgh):
        return float(np.sqrt(np.mean((pred_kgh-cum_true)**2))/np.mean(cum_true)*100)
    for mname, mk in MODELS.items():
        spec = mk()
        # Y1: logit(X) 예측 → 알려진 프로펜몰유량으로 환산 (현재 채택 경로)
        p_best = ev.tune(spec, make_splits(df, INPUT_SETS["I1"]["build"]), "block20")[0]
        estY1 = ev.fit_on_logit(spec, p_best, Xtr_i1, Xg)
        predY1 = ev.predict_conv(estY1, Xva_i1) * npr_va * MW_CUMENE
        # Y2: 큐멘 몰유량[kmol/h] 직접 예측(raw) → ×MW → kg/h
        m2 = _fit_raw(mname, Xtr_i1, Xg*npr_tr);            predY2 = _pred_raw(m2, Xva_i1) * MW_CUMENE
        # Y3: 큐멘 질량유량[kg/h] 직접 예측(raw)
        m3 = _fit_raw(mname, Xtr_i1, Xg*npr_tr*MW_CUMENE);  predY3 = _pred_raw(m3, Xva_i1)
        for oname, pred in [("Y1(X→kgh)", predY1), ("Y2(몰유량→kgh)", predY2), ("Y3(kgh직접)", predY3)]:
            rows.append({"part":"3B_output","model":mname,"input_set":"I1","output":oname,
                         "n_feat":3,"block20":"-","valid_rmse":"-","jack60_ci":"-",
                         "cumene_pct":round(relpct(pred),4),"loco5_mean":"-","loco5_worst":"-",
                         "sens":"-","phys":"-","onnx":"-","case_specifiable":True,"circular_dep":False,
                         "VIF":"-","press_coef_sign_flip":"-"})
            print(f"  {mname:12} {oname:14} 큐멘kg/h 상대RMSE={relpct(pred):.4f}%")

    res = _ROOT/"results"
    with open(res/"stage3_features.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # 재현 대조 출력
    print("\n재현 대조: VIF PT-1004 train/all/valid =",
          round(vif_of(tr,'PT-1004.PV'),1), round(vif_of(allc,'PT-1004.PV'),1), round(vif_of(va,'PT-1004.PV'),1),
          "(ref 19.7/62.6/901.1)")
    print("저장: results/stage3_features.csv")


# --- raw 회귀 헬퍼 (Poly2 구조 유지, 타깃 변환 없음) ---
def _fit_raw(mname, X, y):
    if mname == "Ridge Poly2":
        # alpha 는 큐멘 스케일이 커 block20 없이 기본 1.0 사용(상대비교 목적, 명시)
        est = make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=1.0))
    else:
        est = make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),LinearRegression())
    est.fit(X, y); return est
def _pred_raw(est, X): return np.asarray(est.predict(X)).ravel()


if __name__ == "__main__":
    main()
