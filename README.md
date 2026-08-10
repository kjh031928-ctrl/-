# 공정설계대회 — 큐멘 공정 반응기 대체모델 (ML)

## 1. 개요

2026 화학공학 공정설계 경진대회(문제 rev2) 과제 중 **반응기 R-1100의 대체모델(surrogate model)** 개발 저장소다. 벤젠 + 프로펜 → 큐멘 반응기를 시뮬레이션에서 매번 상세 계산하는 대신, 운전 데이터로 학습한 가벼운 함수로 대신 예측한다. 모델은 두 개이며 각각 ONNX 파일로 배포한다 — **ONNX1은 프로펜 전환율 X**를, **ONNX2는 반응기 압력강하 ΔP**를 예측한다. 둘 다 지도학습·회귀 문제이고, 최종 형태는 입력 3개짜리 1차 선형회귀다(전환율만 [0, 1] 제약을 지키려고 로짓 공간에서 적합 → 예측 시 시그모이드 역변환). 모델의 목적은 계측기 대체가 아니라 **계측기가 없는 시뮬레이션 환경에 값을 공급하는 것**이다(보고서 §3.1.2).

### 처음 오셨다면 이 순서로

1. **완성 원고** — [`대단원3_보고서작성/보고서 본문.docx`](대단원3_보고서작성/) · `Appendix.docx` (제출본)
2. **[정본 초안 `ML보고서_초안_대단원3.md`](대단원3_보고서작성/ML보고서_초안_대단원3.md)** — 문제 정의부터 모델 선정·평가·한계까지 전체 서술. 원고의 근거·수치가 모두 여기서 나온다.
3. **그림 3장** — [F1 ΔP 물리 방향성](figures/F1_dp_physical_direction.png) · [F2 ΔP parity](figures/F2_dp_parity.png) · [F3 전환율 parity](figures/F3_conversion_parity.png)
4. **아래 §4 재현 방법** — 배포 ONNX를 직접 돌려 성능 수치를 재계산해 볼 수 있다.
5. 더 깊이 보려면 → [`2026-07-21_ML개발/docs/`](2026-07-21_ML개발/docs/) 개발 과정 기록(모델 비교·시행착오 전과정), [`2026-07-21_ML개발/src/`](2026-07-21_ML개발/src/) 학습 파이프라인 코드

## 2. 모델 요약

| | ONNX1 (전환율) | ONNX2 (압력강하) |
|---|---|---|
| 출력 | 프로펜 전환율 X [-] | 반응기 압력강하 ΔP [kPa] |
| 입력 (순서 고정) | `TT-1006.PV` 입구온도 [°C]<br>`FT-1004.PV` 프로펜유량 [kg/h]<br>`FRC-1004.PV` B/P 비율 [-] | `TT-1006.PV` 입구온도 [°C]<br>`total_flow_kgh` 총유량 [kg/h]<br>`FRC-1004.PV` B/P 비율 [-] |
| 모델 형태 | 로짓-선형 (LinearRegressor → Sigmoid) | 선형, 출력변환 없음 |
| **held-out RMSE** | **0.021** | **0.191 kPa** |
| **R²** | **0.983** | **0.995** |
| **MAE** | 0.015 | 0.146 kPa |

성능 수치는 보고서 **§3.6.4 성능 종합표** 인용이다. 두 모델은 온도·비율을 공유하고 **유량 자리만** 다르다(반응 성능=프로펜유량, 압력강하=총유량 — §3.1.2).

**데이터 분할** (§3.2.2) — 1분 간격 시계열이라 자기상관이 커서 무작위 분할 대신 **날짜(캠페인) 단위**로만 나눈다.

| 구분 | 캠페인 | 행 수 | 역할 |
|---|---|---|---|
| 학습 | 10/13–16 | 1,924 | 계수 적합 전용 |
| 검증 (held-out) | 10/17–18 | 661 | 위 표의 성능은 이 셋 기준 |
| 앵커 | 10/12 | 481 | 단일 정상상태 반복 기록, 파이프라인 검산용 (성능 근거로 인용하지 않음) |

배포 ONNX는 정확도를 위해 **10/13–18 전체로 재학습**한 것이라, 배포본으로 다시 재면 in-sample 값(0.018 / 0.114)이 나온다. 인용은 held-out 값을 쓴다(§3.2.2 · §3.6.4).

**배포 계수** — ONNX1: `logit(X) = −53.5180 + 0.135237·TT-1006 + 0.0017136·FT-1004 + 0.525359·FRC-1004`. ONNX2: 절편 −51.7101 · b_T 0.084977 · b_총유량 0.00314476 · b_FRC −0.94009.

**라벨의 열역학 기준** — 전환율 라벨은 SRK 순성분 상수를 **시뮬레이터(AVEVA) 내장값**으로 두고 역산한 것이다(근거 `SRK_순성분상수_변경_근거.md`). 학습 타깃과 배포 환경의 열역학 기준을 맞추려는 것이지 정확도 개선이 목적이 아니며, 성능 종합표는 반올림 자리에서 바뀌지 않는다(held-out 0.020987 → 0.020970). ΔP 라벨은 계기값 산술(`PT-1004 − 25 − PT-1100`)이라 영향이 없어 **ONNX2는 재생성하지 않았다**(재적합 시 계수차 9.4e-07).

## 3. 폴더 · 핵심 파일

| 경로 | 내용 |
|---|---|
| `대단원3_보고서작성/보고서 본문.docx` · `Appendix.docx` | **완성 원고**(제출본) |
| `대단원3_보고서작성/ML보고서_초안_대단원3.md` | **정본 초안**. 대단원 3(ML) 본문 + 부록 — 원고의 근거·수치가 모두 여기서 나온다 |
| `대단원3_보고서작성/부록_재현_명세.md` | 부록 표 C1–C4·그림 F1–F3의 재현 명세(어느 스크립트·목표 수치) |
| `2026-07-21_ML개발/onnx/reactor_conversion_r1.onnx`<br>`2026-07-21_ML개발/onnx/reactor_dp.onnx` | **배포 모델 2종**. 보고서에서 `onnx/…`로 줄여 쓰는 파일. 파일명은 AVEVA 플로우시트가 직접 참조하므로 **바꾸지 말 것** |
| `2026-07-21_ML개발/onnx/reactor_conversion_r1_srklit.onnx` | ONNX1의 **교체 전 배포본**(문헌 SRK 라벨) |
| `대단원3_보고서작성/요청 자료/reactor_*.onnx` | 위 두 파일의 사본 (내용 동일, md5 일치 확인) |
| `ML_마스터데이터_PV.csv` | 학습·검증 마스터 데이터(정본). 3,066행 × 20열, 10/12–10/18 1분 간격 |
| `ML_마스터데이터_PV_srklit.csv` | 위 파일의 **교체 전 판**(문헌 SRK 상수 라벨). 비교·이력 보존용 |
| `ML_마스터데이터_PV_컬럼사전.md` | 위 CSV의 컬럼 정의 — 입력 / 라벨 / **입력 금지(누출) 컬럼** 구분 |
| `figures/` | 확정 그림 F1(ΔP 물리 방향성) · F2(ΔP parity) · F3(전환율 parity) + 생성 스크립트 `make_figures.py` |
| `results/permutation_importance.json` | 표 C4 순열 중요도 산출 결과 |
| `results/screening_all_models.json` | 후보 25종의 RMSE·물리검사 실측 원자료(로짓 기준) |
| `대단원3_보고서작성/요청 자료/` | 보고서 표·그림의 **재현 근거 일습**. 무엇이 무엇인지는 그 안의 `_자료목록.md` |
| `대단원3_보고서작성/` | 보고서 작성용 자료 모음(초안·그림·데이터·문제 PDF 사본) |
| `2026-07-21_ML개발/docs/` | 개발 과정 기록 — 모델 비교·시행착오 전과정·질의응답 준비 |
| `2026-07-21_ML개발/src/` · `.../results/` | 학습 파이프라인 코드와 단계별 실험 결과 |
| `CLAUDE.md` | 작업 규칙과 확정 사항 기록 (§5 요약의 원문) |

**재현 스크립트** — 모두 기준값 대조를 내장하고, 어긋나면 종료코드 1로 멈춘다.

| 스크립트 | 무엇을 재현하나 |
|---|---|
| `보고서수치_재현_검증.py` | 본문 수치 일괄(자기상관·외삽·ΔP 대표후보·성능 종합표·삭제검사) |
| `스크리닝_재현_전종.py` | 후보 25종 RMSE + 365°C clamp 실측 (로짓 기준) |
| `대단원3_보고서작성/요청 자료/스크리닝_재현_전종_rawX.py` | 같은 25종을 raw-X 기준으로 — 로짓 도입 근거 |
| `permutation_importance_검증.py` | 순열 중요도(60분 블록셔플 N=200) |
| `대단원3_보고서작성/요청 자료/곡률외삽_검증.py` | 2차항이 탐색 범위 안에서 기울기 부호를 뒤집는 지점 |
| `대단원3_보고서작성/요청 자료/근거보강_검증.py` | 출력 선정 근거·유효 독립표본·예측 범위 주장 |
| `onnx1_retrain_srksim.py` | 배포 ONNX1 재학습·재export(라벨 소스 교체) |
| `figures/make_figures.py` | 그림 F1–F3 재생성(자체 assert 내장) |
| `압력변수_제외근거_검증.py` · `지연효과_검증.py` · `ML_마스터데이터_PV_생성.py` | 압력 제외 근거 · 응답지연 2~3분 · 마스터 데이터 생성 |

`스크리닝_재현_전종*.py` 는 `pip install xgboost catboost lightgbm` 이 추가로 필요하다.

배포 ONNX는 `2026-07-21_ML개발/onnx/`와 `대단원3_보고서작성/요청 자료/` 두 곳에 같은 파일이 있다. 아래 재현 코드는 두 경로 중 있는 쪽을 자동으로 찾는다.

## 4. 재현 방법

필요 패키지:

```bash
pip install onnxruntime numpy pandas
```

저장소 루트에서 실행한다.

```python
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np, pandas as pd, onnxruntime as ort

# ONNX 위치: 저장소에 포함된 사본 우선, 없으면 개발 폴더
BASE = next(p for p in [Path("대단원3_보고서작성/요청 자료"), Path("2026-07-21_ML개발/onnx")] if p.is_dir())
ONNX1, ONNX2 = BASE / "reactor_conversion_r1.onnx", BASE / "reactor_dp.onnx"

def predict(path, X):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.asarray(sess.run(None, {sess.get_inputs()[0].name: np.asarray(X, np.float32)})[0]).ravel()

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))

# 1) 단일 운전점 — 입력 순서를 반드시 지킬 것
print("conversion X :", predict(ONNX1, [[350.0, 3000.0, 5.0]])[0])   # [입구온도, 프로펜유량, B/P]
print("dP [kPa]     :", predict(ONNX2, [[350.0, 16500.0, 5.0]])[0])  # [입구온도, 총유량,   B/P]

# 2) 10/17-18 구간에서 오차 재계산 (배포 ONNX는 10/13-18 적합이므로 이 값은 in-sample)
v = pd.read_csv("ML_마스터데이터_PV.csv", encoding="utf-8-sig").query("Split == '검증'")
p1 = predict(ONNX1, v[["TT-1006.PV", "FT-1004.PV", "FRC-1004.PV"]].values)
p2 = predict(ONNX2, v[["TT-1006.PV", "total_flow_kgh", "FRC-1004.PV"]].values)
print("in-sample RMSE : X", round(rmse(p1, v["X_conv_r1_deployed"]), 4),
      "| dP", round(rmse(p2, v["DP_reactor_kPa"]), 4), "kPa")
```

출력:

```
conversion X : 0.82956916
dP [kPa]     : 25.219963
in-sample RMSE : X 0.0182 | dP 0.1142 kPa
```

마지막 줄이 §3.6.4에 적힌 배포본 in-sample 값(0.018 / 0.114)과 맞으면 모델·데이터·입력 순서가 모두 제대로 연결된 것이다. ONNX 입력은 `(n, 3)` **float32** 배열이고 입력 이름은 `X`, 출력 이름은 각각 `conversion` · `dP`다.

다른 재현 스크립트는 그대로 실행하면 된다(각 스크립트가 기준값 대조를 내장하고 있다).

```bash
python 보고서수치_재현_검증.py            # 본문 수치 일괄
python permutation_importance_검증.py    # 순열 중요도
python 스크리닝_재현_전종.py              # 후보 25종 (pip install xgboost catboost lightgbm 필요)
python figures/make_figures.py           # 그림 F1~F3 재생성
```

> Windows 기본 콘솔(cp949)에서 한글·기호가 깨지면 `set PYTHONIOENCODING=utf-8` 후 실행한다.

## 5. 작업 규칙 (요약 — 원문은 `CLAUDE.md`)

- **원본 대회 파일은 읽기 전용.** 주최측 배포 PDF·CSV는 수정하지 않는다.
- **수치는 반드시 데이터로 재계산해 검증한 뒤 쓴다.** 기존 문서에 적힌 값이라도 재현되지 않으면 **재현된 값을 따르고 그 사실을 명시**한다.
- **근거(ref) 없는 서술 금지.** 확증 편향·임의 판단 금지, 모르는 것은 지어내지 말고 "확인 필요"로 남긴다. 추정과 사실은 구분해 표기한다.
- **산출물은 Word(.docx)와 그림(.png)으로 낸다. 엑셀(.xlsx) 파일은 만들지 않는다.** 그림은 꼭 필요한 것만, 스크립트(.py)는 재현 목적일 때만 남긴다.
