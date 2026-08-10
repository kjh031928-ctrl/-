# -*- coding: utf-8 -*-
"""
시간지연(dead time) 미보정이 ML 모델에 문제가 되는지 독립 재검증.
실행: python 지연효과_검증.py
결과는 10-13_캠페인_심층분석.xlsx 시트 ⑧ 의 수치와 일치해야 함.
"""
import pandas as pd, numpy as np

XLSX = '2026-07-21_ML개발/data/processed/cumene_ml_training_data.xlsx'
IN = ['TT-1006.PV', 'FT-1004.PV', 'FRC-1004.PV']
Y  = 'X_propene_conversion'

m = pd.read_excel(XLSX, sheet_name='Data')
m['ts'] = pd.to_datetime(m['Timestamp'])

logit = lambda p: np.log(p / (1 - p))
A     = lambda d: np.c_[np.ones(len(d)), d[IN].values]
rmse  = lambda e: float(np.sqrt((e ** 2).mean()))

def fit(d, w=None):
    X, y = A(d), logit(d[Y].values)
    if w is None:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        s = np.sqrt(w); b, *_ = np.linalg.lstsq(X * s[:, None], y * s, rcond=None)
    return b

pred = lambda b, d: 1 / (1 + np.exp(-(A(d) @ b)))

def shift_in(df, k, cols):
    """블록(캠페인) 경계를 넘지 않도록 블록 내에서만 k분 시프트."""
    o = df.copy()
    for c in cols:
        o[c] = df.groupby('Block_ID')[c].shift(k)
    return o.dropna(subset=cols)

tr = m[m['Split'] == '학습'].copy()
va = m[m['Split'] == '검증'].copy()
an = m[m['Split'] == '정상운전검사'].copy()
b0 = fit(tr)

print('=== 1. 기준(지연 미보정) ===')
for nm, d in [('학습', tr), ('검증', va), ('10/12 앵커', an)]:
    e = pred(b0, d) - d[Y].values
    print(f'  {nm:10s} RMSE={rmse(e):.5f}  MAE={np.abs(e).mean():.5f}  bias={e.mean():+.5f}')

print('\n=== 2. 입력 3종 동시 시프트 → 검증 RMSE ===')
for k in [0, 1, 2, 3, 5, 10]:
    b = fit(shift_in(tr, k, IN)); d = shift_in(va, k, IN)
    print(f'  lag {k:2d}분  {rmse(pred(b, d) - d[Y].values):.5f}')

print('\n=== 3. 변수별 분해 (지연의 출처) ===')
for c in IN:
    out = []
    for k in [0, 1, 2, 3, 5]:
        b = fit(shift_in(tr, k, [c])); d = shift_in(va, k, [c])
        out.append(f'{k}분:{rmse(pred(b, d) - d[Y].values):.5f}')
    print(f'  {c:14s} ' + '  '.join(out))

print('\n=== 4. 계수 및 케이스스터디 예측 차이 ===')
b2 = fit(shift_in(tr, 2, ['TT-1006.PV']))
print(f'  beta_T  lag0={b0[1]:.6f}  lag2={b2[1]:.6f}  차이={(b2[1]/b0[1]-1)*100:+.2f}%')
for T in [335, 341.57, 350, 355, 360, 365]:
    d = pd.DataFrame({'TT-1006.PV': [T], 'FT-1004.PV': [3000.0], 'FRC-1004.PV': [5.0]})
    p0, p2 = pred(b0, d)[0], pred(b2, d)[0]
    print(f'  T={T:6.2f}°C  lag0 X={p0:.5f}  lag2 X={p2:.5f}  차이={p2-p0:+.5f}')

print('\n=== 5. 과도상태 진단 ===')
m['res'] = pred(b0, m) - m[Y]
for nm, d in [('학습', m[m['Split'] == '학습']), ('검증', m[m['Split'] == '검증'])]:
    print(f'  {nm} corr(잔차, dT/dt) = {np.corrcoef(d["res"], d["TT-1006_rate_C_per_min"])[0,1]:+.4f}')
    a = d['TT-1006_rate_C_per_min'].abs()
    for lo, hi, lab in [(0, .01, '정상'), (.01, .1, '완만'), (.1, 99, '급변')]:
        sub = d[(a >= lo) & (a < hi)]
        print(f'    {lab} n={len(sub):4d}  RMSE={rmse(sub["res"].values):.5f}  bias={sub["res"].mean():+.5f}')
    up = d[d['TT-1006_rate_C_per_min'] > .02]; dn = d[d['TT-1006_rate_C_per_min'] < -.02]
    print(f'    램프방향 편향: 상승 {up["res"].mean():+.5f} vs 하강 {dn["res"].mean():+.5f}')

print('\n=== 6. 잔차 자기상관 (유효 표본수) ===')
for nm, d in [('학습', m[m['Split'] == '학습']), ('검증', m[m['Split'] == '검증'])]:
    r = d['res'].values
    ac = {k: np.corrcoef(r[:-k], r[k:])[0, 1] for k in [1, 10, 60]}
    neff = len(r) * (1 - ac[1]) / (1 + ac[1])
    print(f'  {nm}: lag1={ac[1]:.3f} lag10={ac[10]:.3f} lag60={ac[60]:.3f}  n={len(r)} → 유효표본수≈{neff:.0f}')

print('\n=== 7. ML_weight 적용 시 ===')
bw = fit(tr, tr['ML_weight'].values)
for nm, b in [('미사용(최종안)', b0), ('ML_weight 사용', bw)]:
    e = pred(b, va) - va[Y].values
    st = va[va['TT-1006_rate_C_per_min'].abs() < .01]
    print(f'  {nm:16s} beta_T={b[1]:.6f}  검증={rmse(e):.5f}  정상상태만={rmse(pred(b, st) - st[Y].values):.5f}')
