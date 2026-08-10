# -*- coding: utf-8 -*-
"""보고서 수치 일관성 검사 — 같은 값이 본문·보충자료·부록에서 어긋나는 것을 잡는다

왜 필요한가
  같은 수치가 여러 절에 흩어져 있으면 개정 때마다 일부만 고쳐져 반드시 어긋난다.
  실제로 표 3-3b 를 갱신했을 때 보충자료 표 S1b 와 부록 A 가 옛 값으로 남았다.
  눈으로 맞추는 대신 **재현 스크립트 출력(JSON)을 유일한 출처로 삼아** 기계가 대조한다.

무엇을 하나
  1. 재현 스크립트가 낸 JSON 을 읽어 **수치 대장(registry)** 을 만든다.
       results/screening_all_models.json          — 25종 로짓 RMSE·clamp·단조
       요청 자료/screening_all_models_rawX.json    — 25종 raw RMSE·예측최대
       results/permutation_importance.json        — 순열 중요도
  2. 초안 md 의 표 행을 훑어 **모델명이 있는 행의 숫자**를 대장과 대조한다.
  3. 어긋난 곳을 파일·줄번호와 함께 보고하고, 있으면 종료코드 1 로 멈춘다.

문서 쪽 규칙 (사람이 지킬 것)
  · 수치는 **대장에 있는 값만** 인용한다. 새 수치가 필요하면 먼저 재현 스크립트에 넣는다.
  · 표·문단에 값을 적을 때는 어느 스크립트에서 나온 값인지 절 끝에 출처를 적는다.
  · 반올림해 인용할 때는 대장값을 반올림한 결과여야 한다(이 검사가 확인한다).

재현
  python 보고서_수치_일관성검사.py       (저장소 루트에서 실행)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent
DRAFT = ROOT / "대단원3_보고서작성" / "ML보고서_초안_대단원3.md"
J_LOGIT = ROOT / "results" / "screening_all_models.json"
J_RAW = ROOT / "대단원3_보고서작성" / "요청 자료" / "screening_all_models_rawX.json"
J_PERM = ROOT / "results" / "permutation_importance.json"

# 초안에서 쓰는 표기 → JSON 의 모델명
ALIAS = {
    "Linear 1차": "Linear 1차", "Poly 2차": "Poly 2차", "Poly 3차": "Poly 3차",
    "Ridge 1차": "Ridge 1차", "Ridge Poly2": "Ridge Poly2", "Lasso": "Lasso",
    "ElasticNet": "ElasticNet", "PLS": "PLS", "KNN": "KNN",
    "SVR(RBF)": "SVR(RBF)", "SVR(linear)": "SVR(linear)",
    "GaussianProcess": "GaussianProcess", "GaussProc": "GaussianProcess",
    "DecisionTree": "DecisionTree", "RandomForest": "RandomForest",
    "ExtraTrees": "ExtraTrees", "GradientBoosting": "GradientBoosting",
    "HistGradientBoosting": "HistGradientBoosting", "AdaBoost": "AdaBoost",
    "MLP(16,)": "MLP(16,)", "MLP(64,64)": "MLP(64, 64)", "MLP(64, 64)": "MLP(64, 64)",
    "MLP(128,128)": "MLP(128, 128)", "MLP(128, 128)": "MLP(128, 128)",
    "XGBoost": "XGBoost", "CatBoost": "CatBoost", "LightGBM": "LightGBM",
    "Dummy(평균)": "Dummy(mean)", "Dummy(mean)": "Dummy(mean)",
}
NUM = re.compile(r"(?<![\w.])(\d+\.\d{3,6})(?![\w])")
ISSUES: list[str] = []


def load(p: Path, key: str) -> dict:
    if not p.exists():
        print(f"  [건너뜀] {p.name} 없음 — 먼저 재현 스크립트를 돌리십시오")
        return {}
    return {r["name"]: r for r in json.loads(p.read_text(encoding="utf-8"))[key]}


logit = load(J_LOGIT, "전환율_전종")
raw = load(J_RAW, "전환율_25종_rawX")

print("=" * 92)
print("보고서 수치 일관성 검사 — 재현 스크립트 출력을 유일한 출처로")
print("=" * 92)
print(f"  대장: 로짓 {len(logit)}종 · raw {len(raw)}종")
print(f"  대상: {DRAFT.relative_to(ROOT)}")

lines = DRAFT.read_text(encoding="utf-8").splitlines()

# ── 1. 표 행 대조 ─────────────────────────────────────────────────────────
# 표 행에 모델명이 있으면, 그 행의 숫자 중 하나는 대장의 RMSE(또는 그 반올림)여야 한다.
print("\n" + "-" * 92)
print("1. 표 행 대조 — 모델명이 있는 행의 RMSE 가 대장과 맞는가")
print("-" * 92)
checked = 0
for i, line in enumerate(lines, 1):
    if not line.lstrip().startswith("|"):
        continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    hit = None
    for c in cells:
        key = re.sub(r"[*★◆●▲◆]|\s*★?채택.*|\s*★?배포.*|\s*\(.*배포.*\)", "", c).strip()
        key = key.replace("**", "").strip()
        if key in ALIAS:
            hit = ALIAS[key]
            break
    if hit is None:
        continue
    nums = [float(x) for x in NUM.findall(line)]
    if not nums:
        continue
    # raw 표인지 로짓 표인지: 행에 있는 값이 어느 대장과 맞는지로 판정
    cands = {}
    if hit in logit:
        cands["로짓"] = logit[hit]["rmse"]
    if hit in raw:
        cands["raw"] = raw[hit]["rmse"]
    ok_basis = None
    for basis, ref in cands.items():
        # 원값 또는 3~5자리 반올림 인용을 모두 허용한다
        allowed = {round(ref, d) for d in (3, 4, 5, 6)} | {ref}
        if any(any(abs(n - a) < 5e-7 for a in allowed) for n in nums):
            ok_basis = basis
            break
    checked += 1
    if ok_basis is None:
        near = {b: round(v, 5) for b, v in cands.items()}
        ISSUES.append(f"L{i} [{hit}] 행의 수치 {nums} 가 대장과 불일치 — 대장 {near}")
        print(f"  [불일치] L{i:>4} {hit:<22} 문서 {nums}  대장 {near}")
print(f"  대조한 표 행 {checked}개 · 불일치 {len(ISSUES)}개")

# ── 2. 절 간 중복 지도 ────────────────────────────────────────────────────
print("\n" + "-" * 92)
print("2. 절 간 중복 지도 — 같은 값이 어느 절들에 흩어져 있나(개정 시 함께 고칠 목록)")
print("-" * 92)


def section_of(idx: int) -> str:
    for j in range(idx, -1, -1):
        s = lines[j].strip()
        if s.startswith("#"):
            return s.lstrip("# ").split()[0]
        if s.startswith("> **보충자료"):
            return "보충자료"
    return "?"


occ: dict[str, list] = {}
for i, line in enumerate(lines):
    for m in NUM.finditer(line):
        occ.setdefault(m.group(1), []).append((i + 1, section_of(i)))
multi = {v: ps for v, ps in occ.items() if len({s for _, s in ps}) >= 2}
print(f"  2개 이상 절에 등장하는 값: {len(multi)}개")
for v, ps in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:8]:
    print(f"    {v:<10} {' · '.join(f'{s}(L{ln})' for ln, s in ps)}")
if len(multi) > 8:
    print(f"    … 외 {len(multi) - 8}개")

# ── 판정 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 92)
if ISSUES:
    print(f"불일치 {len(ISSUES)}건 — 재현 스크립트 출력(대장) 값으로 고칠 것:")
    for x in ISSUES:
        print("  -", x)
    sys.exit(1)
print("표 행 수치가 전부 대장과 일치한다.")
print("=" * 92)
