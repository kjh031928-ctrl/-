import numpy as np, pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
R=8.314462618; Tref=650.0
d=pd.read_excel('data/processed/cumene_ml_training_data.xlsx',sheet_name='Data')
W_cat=234*np.pi/4*0.0762**2*2.0*0.5*1600.0
m=d['Split']!='정상운전검사'
lam=float(np.sum(d.loc[m,'dT_adiabatic_C']*d.loc[m,'X_propene_conversion'])/np.sum(d.loc[m,'X_propene_conversion']**2))
nP0=d['n_propene_kmol_h'].values;nA0=d['n_propane_kmol_h'].values;nB0=d['n_benzene_kmol_h'].values
Tin=d['TT-1006.PV'].values+273.15;Pout=d['PT-1100.PV'].values*1000.;Xobs=d['X_propene_conversion'].values
def sim(th,idx,ab=None):
    if ab is None: lnk,Ea,a,b=th
    else: lnk,Ea=th; a,b=ab
    o=np.empty(len(idx))
    for j,i in enumerate(idx):
        nP,nA,nB,T0,P=nP0[i],nA0[i],nB0[i],Tin[i],Pout[i];nt0=nP+nA+nB
        def f(W,X):
            X=min(max(X[0],0.),0.9999999);T=T0+lam*X;nt=nt0-nP*X
            yP=max(nP*(1-X)/nt,1e-14);yB=max((nB-nP*X)/nt,1e-14);c=P/(R*T)/1000.
            return [np.exp(lnk-Ea*1000/R*(1/T-1/Tref))*(yP*c)**a*(yB*c)**b/nP]
        o[j]=min(solve_ivp(f,(0,W_cat),[0.],rtol=1e-7,atol=1e-10,method='LSODA').y[0,-1],1.0)
    return o
tr=np.where(d['Split'].values=='학습')[0];va=np.where(d['Split'].values=='검증')[0];trs=tr[::16]
def report(nm,th,ab=None):
    p_tr=sim(th,tr[::4],ab);e1=p_tr-Xobs[tr[::4]];p=sim(th,va,ab);e=p-Xobs[va]
    print(f"{nm}: 학습RMSE={np.sqrt((e1**2).mean()):.5f}  검증RMSE={np.sqrt((e**2).mean()):.5f} "
          f"MAE={np.abs(e).mean():.5f} max={np.abs(e).max():.4f} R2={1-(e**2).sum()/((Xobs[va]-Xobs[va].mean())**2).sum():.4f}")
    return p,e
# (A) Turton 형태 고정 a=b=1, 2 파라미터
f2=least_squares(lambda t: sim(t,trs,ab=(1.,1.))-Xobs[trs],[-0.5,100.],bounds=([-20,5],[20,300]),x_scale=[1,50],diff_step=1e-3,max_nfev=200)
print(f"(A) 2파라미터 1차/1차: k(650K)={np.exp(f2.x[0]):.4g}, Ea={f2.x[1]:.1f} kJ/mol")
report("   ",f2.x,ab=(1.,1.))
# (B) 4 파라미터
f4=least_squares(lambda t: sim(t,trs)-Xobs[trs],[-0.5,100.,1.,1.],bounds=([-20,5,.1,0.],[20,300,3,3]),x_scale=[1,50,1,1],diff_step=1e-3,max_nfev=300)
print(f"(B) 4파라미터: k(650K)={np.exp(f4.x[0]):.4g}, Ea={f4.x[1]:.1f}, a={f4.x[2]:.3f}, b={f4.x[3]:.3f}")
p,e=report("   ",f4.x)
# 검증 오차 구간별
Tv=d['TT-1006.PV'].values[va]
for lo,hi in [(320,340),(340,350),(350,355),(355,362)]:
    s=(Tv>=lo)&(Tv<hi)
    if s.sum(): print(f"    Tin {lo}-{hi}C n={s.sum():3d}  RMSE={np.sqrt((e[s]**2).mean()):.5f}  bias={e[s].mean():+.5f}")
