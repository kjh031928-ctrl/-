"""stage4_grid.py — Stage 4 확인용 소규모 factorial (상호작용 확인, 신설 2026-08-03).

격자: 3모델(Linear1/Ridge Poly2/Poly2) × 3변환(logit/arcsin/log1m, Stage 2 상위) × 2입력(I1/I1b) = 18칸.
목적: Stage 1~3 의 순차 스크리닝이 가정한 "단계 독립"이 참인지 검증. 입력을 바꾸면 모델 순위가
유지되는지(Spearman), 유의한 역전 쌍이 있는지, 순차 승자가 격자 승자와 같은지, 물리검사가 유지되는지.
승자선택=블록20%, 검증 확인용. Stage 0 프로토콜·evaluation.py 재사용.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError): pass

import numpy as np
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import evaluation as ev
import data_loader as dl
from stage3_features import _merge_ft1003, make_splits, phys_check, INPUT_SETS

_ROOT = Path(__file__).resolve().parents[1]
RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)
TRANSFORMS = ["logit", "arcsin", "log1m"]   # Stage 2 상위 3(범위이탈 최소, logit 포함)
INPUTS = ["I1", "I1b"]

def model_factory(name):
    if name == "Linear1":
        return (lambda p: LinearRegression()), [{}]
    if name == "Ridge Poly2":
        return (lambda p: make_pipeline(PolynomialFeatures(2, include_bias=False),
                StandardScaler(), Ridge(alpha=p["alpha"]))), [{"alpha": a} for a in RIDGE_ALPHAS]
    if name == "Poly2":
        return (lambda p: make_pipeline(PolynomialFeatures(2, include_bias=False),
                StandardScaler(), LinearRegression())), [{}]
MODELS = ["Linear1", "Ridge Poly2", "Poly2"]


def main():
    cfg = dl.load_config(); df = _merge_ft1003(dl.load_data(cfg))
    yv = df[df.Split == "검증"]["X_propene_recycle1"].to_numpy(float)
    boot = ev.block_bootstrap_indices(len(yv), ev.BLOCK_LEN_MIN, ev.N_BOOT, ev.SEED)

    cells = {}   # (model,transform,input) -> result
    rows = []
    print("== Stage 4 격자 18칸 ==")
    for mname in MODELS:
        fac, grid = model_factory(mname)
        for tk in TRANSFORMS:
            for iset in INPUTS:
                spec = ev.ModelSpec(mname, "선형", fac, grid, transform=ev.TRANSFORMS[tk])
                sp = make_splits(df, INPUT_SETS[iset]["build"])
                r = ev.evaluate_model(spec, sp, boot, do_physical=False)
                ph = phys_check(spec, sp, iset)
                cells[(mname, tk, iset)] = r
                rows.append({
                    "model": mname, "transform": tk, "input": iset,
                    "block20": round(r["block20"], 6), "valid_rmse": round(r["valid"]["rmse"], 6),
                    "jack60_ci": f"[{r['jack60']['ci_low']:.5f},{r['jack60']['ci_high']:.5f}]",
                    "jack120_ci": f"[{r['jack120']['ci_low']:.5f},{r['jack120']['ci_high']:.5f}]",
                    "boot60_ci": f"[{r['boot']['ci_low']:.5f},{r['boot']['ci_high']:.5f}]",
                    "cumene_pct": round(r["cumene"]["pct_of_mean"], 4),
                    "loco5_mean": round(r["loco5_mean"], 6),
                    "phys": f"{ph['in_range']}/{ph['n']}·{ph['T_mono_slices']}/{ph['n_slices']}",
                    "onnx": r["onnx"]["ok"],
                })
                print(f"  {mname:12} {tk:7} {iset:4} block20={r['block20']:.5f} "
                      f"valid={r['valid']['rmse']:.5f} phys={rows[-1]['phys']}")

    res = _ROOT / "results"
    with open(res / "stage4_grid.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    combos = [(m, t) for m in MODELS for t in TRANSFORMS]   # 9 model×transform
    def vals(iset, key):
        return [key(cells[(m, t, iset)]) for m, t in combos]

    # Q1: I1→I1b 순위 유지(Spearman), 그리고 배포본(Linear1) I1b 실익
    v_i1 = vals("I1", lambda r: r["valid"]["rmse"]); v_i1b = vals("I1b", lambda r: r["valid"]["rmse"])
    b_i1 = vals("I1", lambda r: r["block20"]); b_i1b = vals("I1b", lambda r: r["block20"])
    sp_valid = float(spearmanr(v_i1, v_i1b).correlation)
    sp_b20 = float(spearmanr(b_i1, b_i1b).correlation)
    print(f"\nQ1 Spearman(I1,I1b) 검증={sp_valid:+.3f} / 블록20%={sp_b20:+.3f}")
    print("   모델×변환별 I1→I1b 검증 변화:")
    deploy_note = {}
    for m in MODELS:
        for t in TRANSFORMS:
            a = cells[(m, t, "I1")]["valid"]["rmse"]; b = cells[(m, t, "I1b")]["valid"]["rmse"]
            arrow = "개선" if b < a else "악화"
            print(f"     {m:12} {t:7} {a:.5f} → {b:.5f} ({arrow})")
            if m == "Linear1" and t == "logit":
                deploy_note = {"i1": a, "i1b": b, "verdict": arrow}

    # Q2: 유의한 역전 쌍 — I1 에서 유의(잭나이프 CI 0배제) + I1b 에서도 유의 + 부호 반전
    def jack_sign(iset, c1, c2):
        d = ev.block_jackknife_delta_ci(yv, np.array(cells[(*c1, iset)]["y_pred_valid"]),
                                        np.array(cells[(*c2, iset)]["y_pred_valid"]), ev.BLOCK_LEN_MIN)
        if d["indistinguishable"]: return 0, d
        return (1 if d["delta"] > 0 else -1), d
    reversals = []
    for c1, c2 in combinations(combos, 2):
        s1, d1 = jack_sign("I1", c1, c2); s2, d2 = jack_sign("I1b", c1, c2)
        if s1 != 0 and s2 != 0 and s1 != s2:
            reversals.append((c1, c2, d1["delta"], d2["delta"]))
    print(f"\nQ2 유의한 순위 역전 쌍(양쪽 CI 0배제·부호반전): {len(reversals)}")
    for c1, c2, d1, d2 in reversals[:20]:
        print(f"   {c1} vs {c2}: I1 Δ={d1:+.5f} / I1b Δ={d2:+.5f}")

    # Q3: 순차 승자 vs 격자 승자(블록20%)
    grid_winner = min(rows, key=lambda x: x["block20"])
    seq = "Ridge Poly2/logit/I1"
    print(f"\nQ3 순차 승자={seq} / 격자 블록20% 승자="
          f"{grid_winner['model']}/{grid_winner['transform']}/{grid_winner['input']} "
          f"(block20 {grid_winner['block20']})")

    # Q4: 물리 27/27·9/9 유지
    phys_all_ok = all(r["phys"] == "27/27·9/9" for r in rows)
    bad = [f"{r['model']}/{r['transform']}/{r['input']}={r['phys']}" for r in rows if r["phys"] != "27/27·9/9"]
    print(f"\nQ4 물리 27/27·9/9 전 칸 유지: {phys_all_ok}  {('예외:'+', '.join(bad)) if bad else ''}")

    with open(res / "stage4_summary.json", "w", encoding="utf-8") as f:
        json.dump({"spearman_valid_I1_I1b": sp_valid, "spearman_block20_I1_I1b": sp_b20,
                   "n_significant_reversals": len(reversals),
                   "reversals": [[list(c1), list(c2), d1, d2] for c1, c2, d1, d2 in reversals],
                   "grid_winner": {"model": grid_winner["model"], "transform": grid_winner["transform"],
                                   "input": grid_winner["input"], "block20": grid_winner["block20"],
                                   "valid": grid_winner["valid_rmse"]},
                   "sequential_winner": seq, "deploy_linear1_i1b": deploy_note,
                   "phys_all_ok": phys_all_ok}, f, ensure_ascii=False, indent=2)
    print("\n저장: results/stage4_grid.csv, stage4_summary.json")


if __name__ == "__main__":
    main()
