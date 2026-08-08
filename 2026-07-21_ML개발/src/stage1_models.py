"""stage1_models.py — Stage 1 모델족 비교 드라이버 (2026-08-01 개편).

조건 고정: 입력 I1, 타깃 logit(X_propene_recycle1), 주 분할(학습 10/13-16 → 검증 10/17-18).
[수정 1] 튜닝·승자선택 = 블록20%(캠페인 내 뒤 20% 시간구간). LOCO-4 는 대조 진단만. LOCO-5 강건성.
[수정 2] 불확실성 주 방법 = 블록 잭나이프(60·120분). 부트스트랩(60분) 교차확인 + 블록길이 안정성.
[추가 1] 승자 결정은 블록20% 로만. 검증셋 수치는 확인용. validation_usage_log.md 누적.

산출물:
  results/stage1_models.csv       — 모델 × 전체 지표(블록20%/LOCO5/검증/잭나이프CI/부트CI/120분CI/ONNX)
  results/stage1_pairwise.csv     — 검증-최우수 대비 ΔRMSE(잭나이프·부트) 95%CI + 동률 판정
  results/stage1_rank_compare.csv — 튜닝기준 전(LOCO4)/후(블록20%)/확인(검증) 순위 대조
  results/stage1_blocklen.csv     — 모델쌍 결론의 블록길이(1~180분) 안정성
  results/stage1_details.json     — 격자예측·LOCO폴드·민감도·ONNX 사유
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import evaluation as ev

_ROOT = Path(__file__).resolve().parents[1]
SEED = ev.SEED


def _grid(*dicts):
    return list(dicts) if dicts else [{}]


def build_specs() -> list[ev.ModelSpec]:
    S = ev.ModelSpec
    specs: list[ev.ModelSpec] = []
    specs.append(S("Dummy(mean)", "기준선",
                   lambda p: DummyRegressor(strategy="mean"), [{}]))
    # 선형
    specs.append(S("Linear 1차", "선형", lambda p: LinearRegression(), [{}]))
    specs.append(S("Poly 2차", "선형",
                   lambda p: make_pipeline(PolynomialFeatures(2, include_bias=False),
                                           StandardScaler(), LinearRegression()), [{}]))
    specs.append(S("Poly 3차", "선형",
                   lambda p: make_pipeline(PolynomialFeatures(3, include_bias=False),
                                           StandardScaler(), LinearRegression()), [{}]))
    specs.append(S("Ridge 1차", "선형",
                   lambda p: make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"])),
                   _grid(*[{"alpha": a} for a in (0.001, 0.01, 0.1, 1.0, 10.0)])))
    specs.append(S("Ridge Poly2", "선형",
                   lambda p: make_pipeline(PolynomialFeatures(2, include_bias=False),
                                           StandardScaler(), Ridge(alpha=p["alpha"])),
                   _grid(*[{"alpha": a} for a in (0.001, 0.01, 0.1, 1.0, 10.0)])))
    specs.append(S("Lasso", "선형",
                   lambda p: make_pipeline(StandardScaler(),
                                           Lasso(alpha=p["alpha"], max_iter=200000)),
                   _grid(*[{"alpha": a} for a in (1e-4, 1e-3, 1e-2, 1e-1)])))
    specs.append(S("ElasticNet", "선형",
                   lambda p: make_pipeline(StandardScaler(),
                                           ElasticNet(alpha=p["alpha"], l1_ratio=p["l1"],
                                                      max_iter=200000)),
                   _grid(*[{"alpha": a, "l1": l} for a in (1e-3, 1e-2, 1e-1)
                           for l in (0.2, 0.5, 0.8)])))
    specs.append(S("PLS", "선형",
                   lambda p: PLSRegression(n_components=p["n"]),
                   _grid(*[{"n": n} for n in (1, 2, 3)])))
    # 거리·커널
    specs.append(S("KNN", "거리·커널",
                   lambda p: make_pipeline(StandardScaler(),
                                           KNeighborsRegressor(n_neighbors=p["k"], weights=p["w"])),
                   _grid(*[{"k": k, "w": w} for k in (3, 5, 10, 20, 50)
                           for w in ("uniform", "distance")])))
    specs.append(S("SVR(RBF)", "거리·커널",
                   lambda p: make_pipeline(StandardScaler(),
                                           SVR(kernel="rbf", C=p["C"], gamma=p["g"], epsilon=0.01)),
                   _grid(*[{"C": C, "g": g} for C in (1.0, 10.0, 100.0)
                           for g in ("scale", 0.1, 1.0)])))
    specs.append(S("SVR(linear)", "거리·커널",
                   lambda p: make_pipeline(StandardScaler(),
                                           SVR(kernel="linear", C=p["C"], epsilon=0.01)),
                   _grid(*[{"C": C} for C in (0.1, 1.0, 10.0)])))
    specs.append(S("GaussianProcess", "거리·커널",
                   lambda p: make_pipeline(
                       StandardScaler(),
                       GaussianProcessRegressor(
                           kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
                           n_restarts_optimizer=1, normalize_y=True, random_state=SEED)),
                   [{}]))
    # 트리
    specs.append(S("DecisionTree", "트리",
                   lambda p: DecisionTreeRegressor(max_depth=p["d"], min_samples_leaf=p["l"],
                                                   random_state=SEED),
                   _grid(*[{"d": d, "l": l} for d in (3, 5, 8, None) for l in (1, 5, 20)])))
    specs.append(S("RandomForest", "트리",
                   lambda p: RandomForestRegressor(n_estimators=p["n"], max_depth=p["d"],
                                                   min_samples_leaf=p["l"], random_state=SEED, n_jobs=-1),
                   _grid(*[{"n": n, "d": d, "l": l} for n in (100, 300)
                           for d in (None, 8) for l in (1, 5)])))
    specs.append(S("ExtraTrees", "트리",
                   lambda p: ExtraTreesRegressor(n_estimators=p["n"], max_depth=p["d"],
                                                 min_samples_leaf=p["l"], random_state=SEED, n_jobs=-1),
                   _grid(*[{"n": n, "d": d, "l": l} for n in (100, 300)
                           for d in (None, 8) for l in (1, 5)])))
    specs.append(S("GradientBoosting", "트리",
                   lambda p: GradientBoostingRegressor(n_estimators=p["n"], learning_rate=p["lr"],
                                                       max_depth=p["d"], random_state=SEED),
                   _grid(*[{"n": n, "lr": lr, "d": d} for n in (100, 300)
                           for lr in (0.05, 0.1) for d in (2, 3)])))
    specs.append(S("HistGradientBoosting", "트리",
                   lambda p: HistGradientBoostingRegressor(learning_rate=p["lr"], max_iter=p["n"],
                                                           max_leaf_nodes=p["leaf"], random_state=SEED),
                   _grid(*[{"lr": lr, "n": n, "leaf": leaf} for lr in (0.05, 0.1)
                           for n in (100, 300) for leaf in (15, 31)])))
    specs.append(S("AdaBoost", "트리",
                   lambda p: AdaBoostRegressor(n_estimators=p["n"], learning_rate=p["lr"],
                                               random_state=SEED),
                   _grid(*[{"n": n, "lr": lr} for n in (50, 100) for lr in (0.5, 1.0)])))
    # 신경망 (alpha 튜닝)
    for arch in [(16,), (64, 64), (128, 128)]:
        specs.append(S(f"MLP{arch}", "신경망",
                       (lambda a: (lambda p: make_pipeline(
                           StandardScaler(),
                           MLPRegressor(hidden_layer_sizes=a, alpha=p["alpha"], max_iter=5000,
                                        random_state=SEED, early_stopping=False))))(arch),
                       _grid(*[{"alpha": al} for al in (1e-4, 1e-3, 1e-2, 1e-1)])))
    return specs


# =============================================================================
def _flatten(r: dict) -> dict:
    v, c, j6, j12, b, s, ph, ox = (r["valid"], r["cumene"], r["jack60"], r["jack120"],
                                   r["boot"], r["sensitivity"], r["physical"], r["onnx"])
    return {
        "model": r["name"], "family": r["family"],
        "best_params_block20": json.dumps(r["best_params"], ensure_ascii=False),
        "params_value": r["params_value"], "params_kind": r["params_kind"],
        # 순위용(성능 아님)
        "block20_rmse_RANK": round(r["block20"], 6),
        # 강건성
        "loco5_mean": round(r["loco5_mean"], 6), "loco5_worst": round(r["loco5_worst"], 6),
        # 대조 진단(폐기 기준)
        "loco4_score": round(r["loco4_score"], 6),
        "holdout_at_loco4_params": round(r["holdout_loco"], 6),
        # 검증(확인용)
        "train_rmse": round(r["train_rmse"], 6),
        "valid_rmse_CONFIRM": round(v["rmse"], 6),
        "jackknife60_ci_low": round(j6["ci_low"], 6), "jackknife60_ci_high": round(j6["ci_high"], 6),
        "jackknife60_se": round(j6["se"], 6),
        "jackknife120_ci_low": round(j12["ci_low"], 6), "jackknife120_ci_high": round(j12["ci_high"], 6),
        "bootstrap60_ci_low": round(b["ci_low"], 6), "bootstrap60_ci_high": round(b["ci_high"], 6),
        "mae": round(v["mae"], 6), "mape_pct": round(v["mape"], 4),
        "r2": round(v["r2"], 5), "max_abs_err": round(v["max_abs_err"], 5),
        "cumene_rmse_kgh": round(c["rmse_kgh"], 3), "cumene_pct_of_mean": round(c["pct_of_mean"], 4),
        "sens_max_pred_shift": round(s["max_pred_shift_rms"], 6),
        # 물리
        "phys_in_range": f"{ph['pred_in_range']}/{ph['n_grid']}",
        "phys_T_monotone_slices": f"{ph['T_monotone_slices']}/{ph['n_slices']}",
        "phys_extrap_ok": ph["extrap_ok"],
        # ONNX (대회 §3.1 요구 — 성능과 별개)
        "onnx_convertible": ox["ok"],
        "onnx_max_abs_diff": (None if ox["max_abs_diff"] is None else float(f"{ox['max_abs_diff']:.3e}")),
        "onnx_fail_reason": (None if ox["ok"] else (ox["error"] or "")[:80]),
        "tune_time_s": round(r["tune_time_s"], 3), "fit_time_s": round(r["fit_time_s"], 4),
        "infer_us_per_row": round(r["infer_time_us_per_row"], 3),
    }


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    sp = ev.load_splits("I1", "r1")
    yv = sp.y_valid
    boot_idx = ev.block_bootstrap_indices(len(yv), ev.BLOCK_LEN_MIN, ev.N_BOOT, ev.SEED)
    specs = build_specs()
    print(f"Stage 1 개편 — {len(specs)}개 모델. 학습 {len(sp.X_train)} → 검증 {len(yv)}. "
          f"타깃 logit(r1), 입력 I1. 승자선택=블록20%.\n")

    results = []
    for spec in specs:
        t0 = time.perf_counter()
        try:
            r = ev.evaluate_model(spec, sp, boot_idx)
        except Exception as e:
            print(f"  [{spec.name:22}] 실패: {type(e).__name__}: {e}")
            continue
        results.append(r)
        print(f"  [{spec.name:22}] block20={r['block20']:.5f} "
              f"valid={r['valid']['rmse']:.5f} LOCO5={r['loco5_mean']:.4f} "
              f"ONNX={'O' if r['onnx']['ok'] else 'X'} ({time.perf_counter()-t0:.1f}s)")

    res_dir = _ROOT / "results"; res_dir.mkdir(exist_ok=True)

    # 정렬: 승자선택 = 블록20% 오름차순(추가 1)
    results.sort(key=lambda r: r["block20"])
    _write_csv(res_dir / "stage1_models.csv", [_flatten(r) for r in results])

    winner = results[0]                                   # 블록20% 최우수 = 공식 선택
    valid_best = min(results, key=lambda r: r["valid"]["rmse"])  # 검증 최우수(확인용)

    # ---- 쌍비교: 검증-최우수 대비 ΔRMSE (잭나이프 주 + 부트스트랩 교차) ----
    pair_rows = []
    for r in results:
        jk = ev.block_jackknife_delta_ci(yv, np.array(r["y_pred_valid"]),
                                         np.array(valid_best["y_pred_valid"]), ev.BLOCK_LEN_MIN)
        bt = ev.bootstrap_delta_ci(yv, np.array(r["y_pred_valid"]),
                                   np.array(valid_best["y_pred_valid"]), boot_idx)
        pair_rows.append({
            "model": r["name"], "valid_rmse": round(r["valid"]["rmse"], 6),
            "jack_delta": round(jk["delta"], 6),
            "jack_ci_low": round(jk["ci_low"], 6), "jack_ci_high": round(jk["ci_high"], 6),
            "jack_indistinguishable": jk["indistinguishable"],
            "boot_ci_low": round(bt["ci_low"], 6), "boot_ci_high": round(bt["ci_high"], 6),
            "boot_indistinguishable": bt["indistinguishable"],
        })
    _write_csv(res_dir / "stage1_pairwise.csv", pair_rows)

    # ---- 전/후 순위 대조: LOCO4(전) vs 블록20%(후) vs 검증(확인) ----
    by_loco4 = sorted(results, key=lambda r: r["loco4_score"])
    by_b20 = sorted(results, key=lambda r: r["block20"])
    by_valid = sorted(results, key=lambda r: r["valid"]["rmse"])
    rank_loco4 = {r["name"]: i + 1 for i, r in enumerate(by_loco4)}
    rank_b20 = {r["name"]: i + 1 for i, r in enumerate(by_b20)}
    rank_valid = {r["name"]: i + 1 for i, r in enumerate(by_valid)}
    rc_rows = []
    for r in results:
        rc_rows.append({
            "model": r["name"],
            "rank_LOCO4_전": rank_loco4[r["name"]], "loco4_score": round(r["loco4_score"], 6),
            "rank_block20_후": rank_b20[r["name"]], "block20_rmse": round(r["block20"], 6),
            "rank_valid_확인": rank_valid[r["name"]], "valid_rmse": round(r["valid"]["rmse"], 6),
        })
    rc_rows.sort(key=lambda x: x["rank_valid_확인"])
    _write_csv(res_dir / "stage1_rank_compare.csv", rc_rows)

    # Spearman(LOCO4,검증) / Spearman(블록20%,검증)
    from scipy.stats import spearmanr
    names = [r["name"] for r in results]
    v_loco4 = [r["loco4_score"] for r in results]
    v_b20 = [r["block20"] for r in results]
    v_val = [r["valid"]["rmse"] for r in results]
    sp_loco4 = float(spearmanr(v_loco4, v_val).correlation)
    sp_b20 = float(spearmanr(v_b20, v_val).correlation)

    # ---- 블록길이 안정성: 검증-최우수 vs 각 모델, BLOCK_LENS 전 구간 결론 동일? ----
    bl_rows = []
    flips = []
    for r in results:
        if r["name"] == valid_best["name"]:
            continue
        decisions = {}
        for L in ev.BLOCK_LENS:
            bi = ev.block_bootstrap_indices(len(yv), L, ev.N_BOOT, ev.SEED)
            d = (np.sqrt(((np.array(r["y_pred_valid"]) - yv) ** 2)[bi].mean(axis=1))
                 - np.sqrt(((np.array(valid_best["y_pred_valid"]) - yv) ** 2)[bi].mean(axis=1)))
            lo, hi = np.percentile(d, [2.5, 97.5])
            decisions[L] = bool(lo <= 0 <= hi)   # True=구분안됨
        stable = len(set(decisions.values())) == 1
        if not stable:
            flips.append(r["name"])
        bl_rows.append({"model": r["name"],
                        **{f"indist_L{L}": decisions[L] for L in ev.BLOCK_LENS},
                        "conclusion_stable": stable})
    _write_csv(res_dir / "stage1_blocklen.csv", bl_rows)

    # ---- 상세 JSON ----
    details = []
    for r in results:
        details.append({
            "name": r["name"], "family": r["family"],
            "best_params_block20": r["best_params"], "loco4_params": r["loco4_params"],
            "loco5_folds": r["loco5_folds"], "sensitivity": r["sensitivity"],
            "physical": {k: r["physical"][k] for k in
                         ("pred_in_range", "T_monotone_slices", "T_monotone_pairs",
                          "extrap_ok", "extrap_diverges", "preds", "meta")},
            "onnx": r["onnx"],
        })
    with open(res_dir / "stage1_details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    tie = [p["model"] for p in pair_rows if p["jack_indistinguishable"]]
    print(f"\n승자(블록20% 최우수): {winner['name']} (block20 {winner['block20']:.6f}, "
          f"검증 {winner['valid']['rmse']:.6f})")
    print(f"검증 최우수(확인용): {valid_best['name']} (검증 {valid_best['valid']['rmse']:.6f})")
    print(f"검증-최우수와 통계적으로 구분되지 않는 모델(잭나이프 {len(tie)}): {tie}")
    print(f"Spearman(LOCO4, 검증)={sp_loco4:+.3f} / Spearman(블록20%, 검증)={sp_b20:+.3f}")
    print(f"블록길이 1~180분에서 결론이 바뀌는 쌍: {flips if flips else '없음'}")
    print("\n저장: stage1_models.csv, stage1_pairwise.csv, stage1_rank_compare.csv, "
          "stage1_blocklen.csv, stage1_details.json")

    # Spearman 값을 렌더러가 읽도록 별도 저장
    with open(res_dir / "stage1_summary.json", "w", encoding="utf-8") as f:
        json.dump({"winner_block20": winner["name"], "valid_best": valid_best["name"],
                   "tie_with_valid_best": tie, "spearman_loco4_valid": sp_loco4,
                   "spearman_block20_valid": sp_b20, "blocklen_flips": flips}, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
