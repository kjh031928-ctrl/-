"""stage2_transforms.py — Stage 2 타깃 변환 비교 (신설 2026-08-01).

고정: 입력 I1, 주 분할, Stage 0 프로토콜(evaluation.py 재사용).
대상 모델: Stage 1 블록20% 상위 4개 + Linear1(필수) = {Ridge Poly2, Poly 2차, ElasticNet, Lasso, Linear 1차}.
변환 5종: raw / logit / log / log(1-y) / arcsin√y.

핵심 질문: RMSE 가 아니라 "예측이 물리적으로 가능한 범위([0,1])에 머무는가". 27점 격자 범위이탈 개수를 센다.
raw/log/log1m 은 외삽에서 [0,1] 이탈 가능, logit·arcsin 은 구조적으로 (0,1). 승자선택은 블록20%, 검증 확인용.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError): pass

import numpy as np
import evaluation as ev
from stage1_models import build_specs

_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODELS = ["Ridge Poly2", "Poly 2차", "ElasticNet", "Lasso", "Linear 1차"]
TRANSFORM_KEYS = ["raw", "logit", "log", "log1m", "arcsin"]


def _row(r, tkey):
    v, j6, j12, b, ph, ox = (r["valid"], r["jack60"], r["jack120"], r["boot"],
                             r["physical"], r["onnx"])
    return {
        "model": r["name"], "transform": ev.TRANSFORMS[tkey].name, "transform_key": tkey,
        "block20_rmse_RANK": round(r["block20"], 6),
        "valid_rmse_CONFIRM": round(v["rmse"], 6),
        "jack60_ci_low": round(j6["ci_low"], 6), "jack60_ci_high": round(j6["ci_high"], 6),
        "jack120_ci_low": round(j12["ci_low"], 6), "jack120_ci_high": round(j12["ci_high"], 6),
        "boot60_ci_low": round(b["ci_low"], 6), "boot60_ci_high": round(b["ci_high"], 6),
        "mae": round(v["mae"], 6), "mape_pct": round(v["mape"], 4), "r2": round(v["r2"], 5),
        "max_abs_err": round(v["max_abs_err"], 5),
        "cumene_pct_of_mean": round(r["cumene"]["pct_of_mean"], 4),
        "loco5_mean": round(r["loco5_mean"], 6), "loco5_worst": round(r["loco5_worst"], 6),
        "sens_max_pred_shift": round(r["sensitivity"]["max_pred_shift_rms"], 6),
        # 물리 — 핵심
        "phys_in_range": ph["pred_in_range"], "phys_out_of_range": ph["n_grid"] - ph["pred_in_range"],
        "phys_T_monotone_slices": f"{ph['T_monotone_slices']}/{ph['n_slices']}",
        "phys_extrap_ok": ph["extrap_ok"],
        "structurally_bounded": tkey in ("logit", "arcsin"),
        # ONNX
        "onnx_convertible": ox["ok"],
        "onnx_max_abs_diff": (None if ox["max_abs_diff"] is None else float(f"{ox['max_abs_diff']:.2e}")),
        "onnx_fail_reason": (None if ox["ok"] else (ox["error"] or "")[:60]),
        "best_params": json.dumps(r["best_params"], ensure_ascii=False),
    }


def main():
    sp = ev.load_splits("I1", "r1")
    boot_idx = ev.block_bootstrap_indices(len(sp.y_valid), ev.BLOCK_LEN_MIN, ev.N_BOOT, ev.SEED)
    all_specs = {s.name: s for s in build_specs()}

    rows, results = [], []
    print(f"Stage 2 — {len(TARGET_MODELS)}모델 × {len(TRANSFORM_KEYS)}변환. 검증 확인용, 선택=블록20%.\n")
    for mname in TARGET_MODELS:
        base = all_specs[mname]
        for tkey in TRANSFORM_KEYS:
            spec = ev.ModelSpec(base.name, base.family, base.factory, base.param_grid,
                                onnx_capable=base.onnx_capable, transform=ev.TRANSFORMS[tkey])
            r = ev.evaluate_model(spec, sp, boot_idx)
            rows.append(_row(r, tkey)); results.append((mname, tkey, r))
            oor = r["physical"]["n_grid"] - r["physical"]["pred_in_range"]
            print(f"  {mname:12} {tkey:7} block20={r['block20']:.5f} valid={r['valid']['rmse']:.5f} "
                  f"범위이탈={oor:>2}/27 ONNX={'O' if r['onnx']['ok'] else 'X'}")

    res_dir = _ROOT / "results"
    with open(res_dir / "stage2_transforms.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # 승자(블록20%), 구조적 안전 변환만 별도
    rows_sorted = sorted(rows, key=lambda x: x["block20_rmse_RANK"])
    winner = rows_sorted[0]
    safe = [x for x in rows_sorted if x["structurally_bounded"]]
    safe_winner = safe[0] if safe else None

    # 변환별 요약(전 모델 평균 범위이탈)
    per_t = {}
    for tk in TRANSFORM_KEYS:
        rs = [x for x in rows if x["transform_key"] == tk]
        per_t[tk] = {"mean_out_of_range": np.mean([x["phys_out_of_range"] for x in rs]),
                     "any_out": any(x["phys_out_of_range"] > 0 for x in rs),
                     "all_onnx": all(x["onnx_convertible"] for x in rs)}

    summary = {"winner_block20": {"model": winner["model"], "transform": winner["transform_key"],
                                  "block20": winner["block20_rmse_RANK"], "valid": winner["valid_rmse_CONFIRM"]},
               "structural_safe_winner": (None if not safe_winner else
                   {"model": safe_winner["model"], "transform": safe_winner["transform_key"],
                    "block20": safe_winner["block20_rmse_RANK"], "valid": safe_winner["valid_rmse_CONFIRM"]}),
               "per_transform": per_t}
    with open(res_dir / "stage2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n블록20% 최우수: {winner['model']} / {winner['transform_key']} "
          f"(block20 {winner['block20_rmse_RANK']}, 검증 {winner['valid_rmse_CONFIRM']}, "
          f"범위이탈 {winner['phys_out_of_range']}/27)")
    if safe_winner:
        print(f"구조적 안전(logit/arcsin) 중 블록20% 최우수: {safe_winner['model']} / "
              f"{safe_winner['transform_key']} (검증 {safe_winner['valid_rmse_CONFIRM']})")
    print("변환별 평균 범위이탈:", {k: round(v["mean_out_of_range"], 1) for k, v in per_t.items()})
    print("저장: results/stage2_transforms.csv, stage2_summary.json")


if __name__ == "__main__":
    main()
