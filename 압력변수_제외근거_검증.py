import numpy as np, pandas as pd
d=pd.read_excel('data/processed/cumene_ml_training_data.xlsx',sheet_name='Data')
I1=['TT-1006.PV','FT-1004.PV','FRC-1004.PV']
tr=d['Split']=='학습'; va=d['Split']=='검증'

print("="*70); print("[1] 압력이 실제로 얼마나 움직였나")
for c in ['PT-1004.PV','PT-1100.PV']:
    g=d.groupby('Campaign')[c].agg(['min','max','std'])
    g['폭%']=(g['max']-g['min'])/g['min']*100
    print(f"\n--{c}--"); print(g.round(2).to_string())
    print(f"  전체: {d[c].min():.0f}~{d[c].max():.0f} kPa, 폭 {(d[c].max()-d[c].min())/d[c].min()*100:.1f}%")
for c in I1:
    print(f"  (비교) {c}: 폭 {(d[c].max()-d[c].min())/d[c].min()*100:.1f}%")

print("\n"+"="*70); print("[2] 압력이 얼마나 '새로운' 정보인가 (공선성)")
def r2_of(y, Xn, df):
    A=np.column_stack([np.ones(len(df))]+[df[c].values for c in Xn])
    b=np.linalg.lstsq(A,df[y].values,rcond=None)[0]; r=df[y].values-A@b
    return 1-(r**2).sum()/((df[y].values-df[y].values.mean())**2).sum(), r
for c in ['PT-1004.PV','PT-1100.PV']:
    r2,_=r2_of(c,I1,d[tr]); print(f"  {c}: 입력3개로 설명되는 비율 R²={r2:.4f} → VIF={1/(1-r2):.1f}, 고유정보={100*(1-r2):.1f}%")
    print(f"     corr(FT-1004)={np.corrcoef(d[tr][c],d[tr]['FT-1004.PV'])[0,1]:+.4f}  corr(TT-1006)={np.corrcoef(d[tr][c],d[tr]['TT-1006.PV'])[0,1]:+.4f}")

print("\n"+"="*70); print("[3] 남은 정보가 전환율을 설명하나 (편상관)")
for c in ['PT-1004.PV','PT-1100.PV']:
    _,rp=r2_of(c,I1,d[tr])
    z=np.log(d[tr]['X_propene_conversion']/(1-d[tr]['X_propene_conversion']))
    A=np.column_stack([np.ones(tr.sum())]+[d[tr][k].values for k in I1])
    rz=z.values-A@np.linalg.lstsq(A,z.values,rcond=None)[0]
    pc=np.corrcoef(rp,rz)[0,1]
    print(f"  {c}: 편상관 = {pc:+.4f}  (설명력 {pc**2*100:.2f}%)")

print("\n"+"="*70); print("[4] 10/14 (입력 3개 전부 고정) 안에서만 본 압력↔전환율")
s=d['Campaign']=='10/14'
print(f"  TT-1006 폭 {d[s]['TT-1006.PV'].max()-d[s]['TT-1006.PV'].min():.3f}C, FT 폭 {d[s]['FT-1004.PV'].max()-d[s]['FT-1004.PV'].min():.1f}kg/h")
for c in ['PT-1004.PV','PT-1100.PV']:
    print(f"  {c}: 폭 {d[s][c].max()-d[s][c].min():.2f} kPa, corr(X)={np.corrcoef(d[s][c],d[s]['X_propene_conversion'])[0,1]:+.4f}")
print(f"  X 표준편차(노이즈 바닥) = {d[s]['X_propene_conversion'].std():.5f}")
import numpy as np, pandas as pd
from scipy.integrate import solve_ivp
R=8.314462618; Tref=650.
d=pd.read_excel('data/processed/cumene_ml_training_data.xlsx',sheet_name='Data')
W=234*np.pi/4*0.0762**2*2.0*0.5*1600.; lam=161.20
lnk,Ea,a,b=np.log(2.438),117.2,1.740,0.572     # 어제 적합된 속도식
def X_of(Tc,ft,frc,Pk):
    nP=ft*0.94773/42.081;nA=ft*0.05227/44.097;nB=frc*ft/78.115;T0=Tc+273.15;nt0=nP+nA+nB;P=Pk*1000
    def f(w,x):
        x=min(max(x[0],0.),0.9999999);T=T0+lam*x;nt=nt0-nP*x
        yP=max(nP*(1-x)/nt,1e-14);yB=max((nB-nP*x)/nt,1e-14);c=P/(R*T)/1000
        return [np.exp(lnk-Ea*1000/R*(1/T-1/Tref))*(yP*c)**a*(yB*c)**b/nP]
    return min(solve_ivp(f,(0,W),[0.],rtol=1e-8,atol=1e-11,method='LSODA').y[0,-1],1.)

print("="*70);print("[5] 물리로 계산: 압력이 그 정도 흔들리면 전환율이 얼마나 변해야 하나")
s=d['Campaign']=='10/14'
T14=d[s]['TT-1006.PV'].mean(); F14=d[s]['FT-1004.PV'].mean(); R14=d[s]['FRC-1004.PV'].mean()
Plo,Phi=d[s]['PT-1100.PV'].min(),d[s]['PT-1100.PV'].max()
xlo,xhi=X_of(T14,F14,R14,Plo),X_of(T14,F14,R14,Phi)
print(f"  10/14 조건 T={T14:.2f}C F={F14:.0f} FRC={R14:.2f}")
print(f"  압력 {Plo:.1f} -> {Phi:.1f} kPa (+{(Phi/Plo-1)*100:.2f}%)")
print(f"  물리 예측 X: {xlo:.5f} -> {xhi:.5f}   ΔX = {xhi-xlo:+.5f}")
print(f"  실제 관측  X: {d[s]['X_propene_conversion'].min():.5f} -> {d[s]['X_propene_conversion'].max():.5f}   ΔX = {d[s]['X_propene_conversion'].max()-d[s]['X_propene_conversion'].min():+.5f}")
print(f"  속도식이 압력에 의존하는 세기: r ∝ P^{a+b:.2f}")
# 케이스스터디 범위에서의 압력 민감도
print("\n  [참고] 정상운전점에서 압력 100 kPa(약 4%) 변할 때:")
for Tc in [341.57,350,365]:
    x1,x2=X_of(Tc,3000,5.,2450.),X_of(Tc,3000,5.,2550.)
    print(f"    Tin={Tc:6.2f}C: X {x1:.4f} -> {x2:.4f}  (ΔX={x2-x1:+.4f}, 큐멘 {(x2-x1)*3000*0.94773/42.081*120.196:+.0f} kg/h)")

print("\n"+"="*70);print("[6] I1 vs I2/I3 성능차가 통계적으로 의미있나 (블록 부트스트랩, 60분 블록)")
def fit_pred(cols):
    tr=d['Split']=='학습'; va=d['Split']=='검증'
    def A(df): return np.column_stack([np.ones(len(df))]+[df[c].values for c in cols])
    y=d[tr]['X_propene_conversion'].values; z=np.log(y/(1-y))
    bta=np.linalg.lstsq(A(d[tr]),z,rcond=None)[0]
    p=1/(1+np.exp(-A(d[va])@bta))
    return p-d[va]['X_propene_conversion'].values
I1=['TT-1006.PV','FT-1004.PV','FRC-1004.PV']
e={'I1':fit_pred(I1),'I2':fit_pred(I1+['PT-1004.PV']),'I3':fit_pred(I1+['PT-1100.PV'])}
for k,v in e.items(): print(f"  {k} 검증 RMSE = {np.sqrt((v**2).mean()):.5f}")
rng=np.random.default_rng(42); n=len(e['I1']); L=60; nb=n//L
for k in ['I2','I3']:
    ds=[]
    for _ in range(4000):
        st=rng.integers(0,n-L,nb); idx=np.concatenate([np.arange(t,t+L) for t in st])
        ds.append(np.sqrt((e[k][idx]**2).mean())-np.sqrt((e['I1'][idx]**2).mean()))
    ds=np.array(ds)
    print(f"  RMSE({k}) - RMSE(I1) = {np.sqrt((e[k]**2).mean())-np.sqrt((e['I1']**2).mean()):+.5f}, 95% 구간 [{np.percentile(ds,2.5):+.5f}, {np.percentile(ds,97.5):+.5f}]  → {'차이 없음' if np.percentile(ds,2.5)<0<np.percentile(ds,97.5) else '차이 있음'}")
