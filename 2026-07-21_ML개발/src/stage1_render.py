"""stage1_render.py — Stage 1 결과 CSV/JSON → docs/Stage1_모델족비교.md (2026-08-01 개편)."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError): pass

_ROOT = Path(__file__).resolve().parents[1]
RES = _ROOT / "results"
csv.field_size_limit(10_000_000)

def rd(name):
    with open(RES / name, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def T(x): return x in ("True", "true")

def main() -> None:
    models = rd("stage1_models.csv")              # block20 오름차순
    pair = {p["model"]: p for p in rd("stage1_pairwise.csv")}
    rc = rd("stage1_rank_compare.csv")            # valid 순
    bl = {b["model"]: b for b in rd("stage1_blocklen.csv")}
    summ = json.load(open(RES / "stage1_summary.json", encoding="utf-8"))
    BL = ["1", "15", "30", "60", "120", "180"]

    L = []; A = L.append
    A("# Stage 1 — 모델족 비교 (ONNX1 반응기 전환율)")
    A("")
    A("입력 I1=[TT-1006, FT-1004, FRC-1004] · 타깃 logit(X_propene_recycle1) · "
      "주 분할(학습 10/13-16 → 검증 10/17-18). 프로토콜 `docs/Stage0_평가프로토콜.md`, 구현 `src/evaluation.py`.")
    A("원자료: `results/stage1_models.csv`, `stage1_pairwise.csv`, `stage1_rank_compare.csv`, "
      "`stage1_blocklen.csv`, `stage1_details.json`.")
    A("")
    A("- **승자 선택 기준 = 블록20%**(캠페인 내 뒤 20% 시간구간). 검증 RMSE 는 확인용(추가 1).")
    A("- 블록20% 절대값은 홀드아웃의 0.13~0.42배로 낙관적 → **순위 매기기 전용, 성능 수치 아님.**")
    A(f"- 승자(블록20%): **{summ['winner_block20']}** / 검증 최우수(확인): **{summ['valid_best']}**.")
    A(f"- Spearman(블록20%, 검증) = **{summ['spearman_block20_valid']:+.3f}** · "
      f"Spearman(LOCO4, 검증) = **{summ['spearman_loco4_valid']:+.3f}** (LOCO4 는 검증 순위와 약상관 → 폐기).")
    A("")

    # ---- (b) 비교표 ----
    A("## (b) 비교표")
    A("")
    A("| 모델 | 계열 | 파라미터수(종류) | 블록20%(순위용) | LOCO5 평균/최악(강건성) | 학습RMSE | "
      "검증RMSE(확인) | 잭나이프60 95%CI | 부트60 95%CI | 블록120 CI | 큐멘%오차 | "
      "물리(범위·T단조·외삽) | 캠페인민감도 | 학습시간s |")
    A("|" + "---|" * 14)
    for m in models:
        phys = f"{m['phys_in_range']}·{m['phys_T_monotone_slices']}·{'OK' if T(m['phys_extrap_ok']) else 'X'}"
        A(f"| {m['model']} | {m['family']} | {m['params_value']}({m['params_kind']}) | "
          f"{m['block20_rmse_RANK']} | {m['loco5_mean']}/{m['loco5_worst']} | {m['train_rmse']} | "
          f"{m['valid_rmse_CONFIRM']} | [{m['jackknife60_ci_low']}, {m['jackknife60_ci_high']}] | "
          f"[{m['bootstrap60_ci_low']}, {m['bootstrap60_ci_high']}] | "
          f"[{m['jackknife120_ci_low']}, {m['jackknife120_ci_high']}] | {m['cumene_pct_of_mean']}% | "
          f"{phys} | {m['sens_max_pred_shift']} | {m['tune_time_s']} |")
    A("")
    A("- 물리 표기: `[0,1]범위개수/27 · T단조슬라이스/9 · 외삽OK`. 캠페인민감도 = 학습캠페인 1개 제외 시 "
      "검증예측 최대이동(RMS, 전환율). 파라미터종류: 선형=계수+절편, 트리=노드수, 앙상블=총노드수, "
      "SVR=서포트벡터수, MLP=가중치+편향, KNN/GP=저장표본수.")
    A("")

    # ---- ONNX 변환(대회 §3.1) ----
    A("## ONNX 변환 가능 여부 (대회 §3.1 요구 — 성능과 별개)")
    A("")
    A("| 모델 | 변환 | 최대절대차 | 실패 사유 |")
    A("|---|---|---|---|")
    for m in models:
        ok = T(m["onnx_convertible"])
        A(f"| {m['model']} | {'O' if ok else '**X**'} | {m['onnx_max_abs_diff'] or '-'} | "
          f"{'-' if ok else (m['onnx_fail_reason'] or '')} |")
    A("")
    A("- 변환 불가는 **성능 문제가 아니라 대회 요구(§3.1: ONNX 배포) 미충족**이므로 배포 후보에서 제외된다.")
    A("")

    # ---- 전/후 순위 대조 ----
    A("## 튜닝 기준 전(LOCO4) / 후(블록20%) / 확인(검증) 순위 대조")
    A("")
    A("| 모델 | LOCO4 순위(전) | 블록20% 순위(후) | 검증 순위(확인) | LOCO4 score | 블록20% | 검증RMSE |")
    A("|---|---|---|---|---|---|---|")
    for r in rc:
        A(f"| {r['model'].strip(chr(34))} | {r['rank_LOCO4_전']} | {r['rank_block20_후']} | "
          f"{r['rank_valid_확인']} | {r['loco4_score']} | {r['block20_rmse']} | {r['valid_rmse']} |")
    A("")
    A("- LOCO4 는 검증 순위와 크게 어긋난다(예: ElasticNet LOCO4 1위→검증 7위, Poly2 LOCO4 22위→검증 4위). "
      "블록20% 는 검증 순위와 잘 맞는다. **단 Poly3 는 블록20% 11위지만 검증 22위(외삽 발산)** — "
      "블록20% 도 캠페인 내부 외삽 붕괴는 못 잡는다(물리 검사가 보완).")
    A("")

    # ---- (c) 동률 목록 ----
    vb = summ["valid_best"]
    A(f"## (c) 검증-최우수({vb})와 통계적으로 구분되지 않는 모델")
    A("")
    A("ΔRMSE = RMSE(모델) − RMSE(검증최우수). 잭나이프(주)·부트스트랩(교차) 95%CI 가 0 을 포함하면 '차이 없음'.")
    A("")
    A("| 모델 | 검증RMSE | ΔRMSE | 잭나이프 95%CI | 부트 95%CI | 구분불가 |")
    A("|---|---|---|---|---|---|")
    for m in models:
        p = pair[m["model"]]
        mark = "○" if T(p["jack_indistinguishable"]) else ""
        A(f"| {m['model']} | {p['valid_rmse']} | {p['jack_delta']} | "
          f"[{p['jack_ci_low']}, {p['jack_ci_high']}] | [{p['boot_ci_low']}, {p['boot_ci_high']}] | {mark} |")
    A("")
    A(f"구분되지 않는 모델(잭나이프 {len(summ['tie_with_valid_best'])}): {', '.join(summ['tie_with_valid_best'])}")
    A("")

    # ---- 블록길이 안정성 ----
    A("## 모델쌍 결론의 블록길이(1~180분) 안정성")
    A("")
    A("검증-최우수 대비 '구분불가' 판정이 블록길이에 따라 바뀌는지(부트스트랩). True=구분안됨.")
    A("")
    A("| 모델 | " + " | ".join(f"L{l}분" for l in BL) + " | 결론일관 |")
    A("|---|" + "---|" * (len(BL) + 1))
    for m in models:
        if m["model"] == vb: continue
        b = bl[m["model"]]
        cells = " | ".join("T" if T(b[f"indist_L{l}"]) else "F" for l in BL)
        A(f"| {m['model']} | {cells} | {'○' if T(b['conclusion_stable']) else '**바뀜**'} |")
    A("")
    flips = summ["blocklen_flips"]
    A(f"- 블록길이에 따라 결론이 바뀌는 쌍: {', '.join(flips) if flips else '없음'}. "
      "이들은 검증-최우수와 경계선에 있는 근접 모델이며, 큰 격차 쌍(트리·GP·Dummy)은 전 구간 일관되게 구분된다.")
    A("")

    # ---- (d) 관찰 요약 ----
    A("## (d) 관찰 사실 요약")
    A("")
    A(f"1. 블록20% 최우수와 검증 최우수가 모두 **{summ['winner_block20']}**(검증 RMSE {models[0]['valid_rmse_CONFIRM']}) "
      "로 일치했다. 검증 상위권은 선형 계열(Ridge Poly2·Linear1·PLS·Poly2·Ridge1)이 차지했다.")
    A("2. 잭나이프 기준 검증-최우수와 구분되지 않는 모델이 "
      f"{len(summ['tie_with_valid_best'])}개로, 상위 선형 모델들은 서로 통계적으로 동률이다.")
    A(f"3. Spearman(블록20%,검증)=+{summ['spearman_block20_valid']:.3f} 로 블록20% 는 검증 순위를 잘 예측하나, "
      f"Spearman(LOCO4,검증)=+{summ['spearman_loco4_valid']:.3f} 로 LOCO4 는 약하다 → 튜닝 기준으로 부적합.")
    A("4. 트리·거리 계열(RF·ET·GBR·KNN·DecisionTree)과 GaussianProcess 는 검증 RMSE 0.06~0.16 으로 "
      "선형 대비 뚜렷이 나빴고, 블록길이 전 구간에서 최우수와 구분됐다.")
    A("5. ONNX 변환은 Dummy·HistGradientBoosting 2종이 실패(대회 §3.1 미충족). 나머지는 최대절대차 ≤ 2e-6 수준.")
    A("")

    out = _ROOT / "docs" / "Stage1_모델족비교.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"저장: {out}")

if __name__ == "__main__":
    main()
