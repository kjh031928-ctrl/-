# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(fp); plt.rcParams['font.family']=fm.FontProperties(fname=fp).get_name()
plt.rcParams['axes.unicode_minus']=False
plt.rcParams.update({'font.size':10.5,'axes.titlesize':12,'axes.titleweight':'bold'})
B="/sessions/festive-inspiring-feynman/mnt/공정설계대회/"
df=pd.read_csv(B+"2026 Chemical Engineering Process design competition_rev0_Attachment1.csv",encoding='utf-8-sig')
df['ts']=pd.to_datetime(df['Timestamp']); df['day']=df['ts'].dt.strftime('%m/%d')
m=pd.read_excel(B+'2026-07-21_ML개발/data/processed/cumene_ml_training_data.xlsx',sheet_name='Data')
m['ts']=pd.to_datetime(m['Timestamp']); m['day']=m['ts'].dt.strftime('%m/%d')
IN=['TT-1006.PV','FT-1004.PV','FRC-1004.PV']; Y='X_propene_conversion'
logit=lambda p: np.log(p/(1-p)); A=lambda d: np.c_[np.ones(len(d)),d[IN].values]
tr=m[m['Split']=='학습']; va=m[m['Split']=='검증']
b0,*_=np.linalg.lstsq(A(tr),logit(tr[Y].values),rcond=None)
m['res']=1/(1+np.exp(-A(m)@b0))-m[Y]

# ── 그림 3 : 지연 진단 ──
fig,ax=plt.subplots(1,3,figsize=(15.6,5.0),gridspec_kw={'wspace':0.42})
h=df[df['day']=='10/17'].reset_index(drop=True)
i0=int(h['TT-1006.PV Value °C'].diff().abs().idxmax())
w=range(i0-4,i0+9); tt=[h['ts'][k].strftime('%H:%M') for k in w]
a=ax[0]; x=np.arange(len(list(w)))
a.plot(x,h['TT-1006.PV Value °C'][w].values,'o-',color='#D62728',lw=2,ms=6,label='TT-1006 입구온도 (좌)')
a.axvline(4,color='k',ls='--',lw=1.2)
a.set_ylabel('입구온도 (°C)',color='#D62728'); a.tick_params(axis='y',labelcolor='#D62728')
a2=a.twinx(); a2.plot(x,h['AT-1100.PV Value kg/m3'][w].values,'s-',color='#1F4E79',lw=2,ms=6,label='AT-1100 출구밀도 (우)')
a2.set_ylabel('출구 밀도 (kg/m³)',color='#1F4E79'); a2.tick_params(axis='y',labelcolor='#1F4E79')
a.set_xticks(x); a.set_xticklabels(tt,rotation=60,fontsize=7.8)
a.set_title('그림 3-a · 10/17 19:00 실제 계단 응답',fontsize=12)
a.text(.03,.94,'같은 분에 온도 +2.631°C,\n밀도 +0.029 kg/m³\n→ 지연 1샘플 미만',transform=a.transAxes,va='top',
  fontsize=9,fontweight='bold',color='#1B5E20',bbox=dict(fc='#E8F5E9',ec='#1B5E20',boxstyle='round,pad=0.4'))
h1,l1=a.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels()
a.legend(h1+h2,l1+l2,fontsize=8.3,loc='lower right'); a.grid(alpha=.25)

g=df[df['day']=='10/13'].reset_index(drop=True)
u=g['FC-1002.PV Value kg/h'].values[180:245]; y=g['AT-1100.PV Value kg/m3'].values[180:245]
cc=[np.corrcoef(u if L==0 else u[:-L], y if L==0 else y[L:])[0,1] for L in range(6)]
a=ax[1]
a.plot(range(6),cc,'o-',color='#1F4E79',lw=2,ms=8)
a.set_xlabel('가정한 지연 (분)'); a.set_ylabel('상관계수')
a.set_title('그림 3-b · 지연 스캔 — 구분 불가',fontsize=12)
for i,v in enumerate(cc): a.annotate(f'{v:.6f}',(i,v),textcoords='offset points',xytext=(0,-16),ha='center',fontsize=7.6)
a.set_ylim(0.999980,1.0000005)
a.set_yticks([0.99998,0.999985,0.99999,0.999995,1.0])
a.set_yticklabels(['0.999980','0.999985','0.999990','0.999995','1.000000'],fontsize=8.2)
a.grid(alpha=.25)
a.text(.97,.55,'0분이 최댓값이고 차이는\n6번째 소수점 → 분해 불가.\n분수 시프트 적합은 τ = −0.86분\n(인과적으로 불가능)',transform=a.transAxes,ha='right',va='top',
  fontsize=8.7,fontweight='bold',color='#B00020',bbox=dict(fc='#FFEBEE',ec='#B00020',boxstyle='round,pad=0.4'))

a=ax[2]
d=m[m['Split']=='검증']; ab=d['TT-1006_rate_C_per_min'].abs()
grp=[('정상\n|dT/dt|<0.01',d[ab<0.01]),('완만\n0.01~0.1',d[(ab>=0.01)&(ab<0.1)]),('급변\n>0.1',d[ab>=0.1])]
v=[np.sqrt((s['res'].values**2).mean()) for _,s in grp]; n=[len(s) for _,s in grp]
bars=a.bar(range(3),v,color=['#2CA02C','#FF9800','#D62728'],ec='k',lw=.7,width=.6)
for i,(vv,nn) in enumerate(zip(v,n)):
    a.text(i,vv+0.0012,f'{vv:.5f}',ha='center',fontweight='bold',fontsize=9.5)
    a.text(i,0.0016,f'n={nn}',ha='center',fontsize=8.5,color='white',fontweight='bold')
a.set_xticks(range(3)); a.set_xticklabels([k for k,_ in grp],fontsize=9)
a.set_ylabel('RMSE (전환율 X)'); a.set_ylim(0,0.049)
a.set_title('그림 3-c · 지연 무시의 대가는 과도상태에만',fontsize=12)
a.grid(alpha=.25,axis='y')
a.text(.5,.62,'케이스 스터디는 정상상태 조건 →\n실제 사용 지점에서는 영향 없음',transform=a.transAxes,ha='center',
  fontsize=9,fontweight='bold',color='#1B5E20',bbox=dict(fc='#E8F5E9',ec='#1B5E20',boxstyle='round,pad=0.4'))
fig.savefig(B+'그림3_지연진단.png',dpi=165,bbox_inches='tight',facecolor='white'); plt.close(fig)

# ── 그림 4 : 압력 (4패널로 축소) ──
tr=tr.copy(); tr['총공급']=tr['FT-1004.PV']*(1+tr['FRC-1004.PV'])
r2=lambda y,yh: 1-((y-yh)**2).sum()/((y-y.mean())**2).sum()
CM={'10/13':'#D62728','10/14':'#2CA02C','10/15':'#1F77B4','10/16':'#FF7F0E'}
LB={'10/13':'10/13 유량 램프','10/14':'10/14 컬럼 램프','10/15':'10/15 비율 램프','10/16':'10/16 온도 램프'}
fig,axg=plt.subplots(2,2,figsize=(12.4,9.2),gridspec_kw={'wspace':0.26,'hspace':0.30})
ax=axg.ravel()
a=ax[0]
for d,gg in tr.groupby('day'): a.scatter(gg['FT-1004.PV'],gg['PT-1100.PV'],s=6,alpha=.55,c=CM[d],label=LB[d],lw=0)
a.set_title('그림 4-a · 단순 상관으로는 공선이라 못 한다',color='#B00020',fontsize=12)
a.set_xlabel('FT-1004 프로펜 유량 (kg/h)'); a.set_ylabel('PT-1100 출구압력 (kPa)')
a.text(.03,.95,f"학습셋 전체 r = {np.corrcoef(tr['FT-1004.PV'],tr['PT-1100.PV'])[0,1]:+.3f}\n(문서의 0.9988 재현 안 됨)",
  transform=a.transAxes,va='top',fontsize=10.5,fontweight='bold',color='#B00020',bbox=dict(fc='#FFEBEE',ec='#B00020',boxstyle='round,pad=0.4'))
a.legend(fontsize=9,loc='lower right',markerscale=2.5); a.grid(alpha=.25)
a=ax[1]
for d,gg in tr.groupby('day'): a.scatter(gg['총공급'],gg['PT-1100.PV'],s=6,alpha=.55,c=CM[d],lw=0)
X=np.c_[np.ones(len(tr)),tr['총공급']]; bb,*_=np.linalg.lstsq(X,tr['PT-1100.PV'].values,rcond=None)
xs=np.linspace(tr['총공급'].min(),tr['총공급'].max(),40); a.plot(xs,bb[0]+bb[1]*xs,'k--',lw=1.8)
a.set_title('그림 4-b · 압력이 따르는 것은 총 공급유량',fontsize=12)
a.set_xlabel('총 공급유량  F × (1+R)  (kg/h)'); a.set_ylabel('PT-1100 (kPa)')
a.text(.03,.95,f"r = {np.corrcoef(tr['총공급'],tr['PT-1100.PV'])[0,1]:+.4f}\n네 캠페인이 한 직선 위",transform=a.transAxes,va='top',
  fontsize=10.5,fontweight='bold',color='#1B5E20',bbox=dict(fc='#E8F5E9',ec='#1B5E20',boxstyle='round,pad=0.4'))
a.grid(alpha=.25)
a=ax[2]
V3=IN; V4=IN+['PT-1100.PV']
def vif(cols):
    o=[]
    for v in cols:
        oth=[x for x in cols if x!=v]; Xv=np.c_[np.ones(len(tr)),tr[oth].values]
        z,*_=np.linalg.lstsq(Xv,tr[v].values,rcond=None); o.append(1/(1-r2(tr[v].values,Xv@z)))
    return o
v3,v4=vif(V3),vif(V4); x=np.arange(4)
a.bar(x-.2,v3+[0],.38,label='입력 3종 (최종)',color='#2CA02C',ec='k',lw=.6)
a.bar(x+.2,v4,.38,label='입력 4종 (압력 추가)',color='#D62728',ec='k',lw=.6)
a.axhline(10,color='k',ls='--',lw=1.3); a.text(-0.45,13.5,'VIF = 10 경고선',fontsize=10)
for i,v in enumerate(v4): a.text(i+.2,v+1.6,f'{v:.1f}',ha='center',fontsize=10,fontweight='bold',color='#B00020')
a.set_xticks(x); a.set_xticklabels(['TT-1006\n온도','FT-1004\n유량','FRC-1004\n비율','PT-1100\n압력'],fontsize=10)
a.set_ylabel('VIF'); a.set_ylim(0,78); a.legend(fontsize=9.5,loc='upper right'); a.grid(alpha=.25,axis='y')
a.set_title('그림 4-c · 압력을 넣으면 공선성 폭발',fontsize=12)
a=ax[3]
an=m[m['Split']=='정상운전검사']
def run(vs):
    Xt=np.c_[np.ones(len(tr)),tr[vs].values]; z,*_=np.linalg.lstsq(Xt,logit(tr[Y].values),rcond=None)
    return [float(np.sqrt(((1/(1+np.exp(-(np.c_[np.ones(len(d)),d[vs].values]@z)))-d[Y].values)**2).mean()))
            for d in [tr,va,an]]
mods=[('T, F, R\n(최종)',V3,'#2CA02C'),('T, F, R, P\n(압력 추가)',V4,'#D62728')]
x=np.arange(3); w=.35
for i,(lb,vs,cc2) in enumerate(mods):
    o=run(vs); a.bar(x+(i-0.5)*w,o,w,label=lb,color=cc2,ec='k',lw=.6)
    for xx,vv in zip(x+(i-0.5)*w,o): a.text(xx,vv+0.0008,f'{vv:.5f}',ha='center',fontsize=9,rotation=90,va='bottom')
a.set_xticks(x); a.set_xticklabels(['학습셋','검증셋','10/12 앵커\n(독립)'],fontsize=10.5)
a.set_ylabel('RMSE (전환율 X)'); a.set_ylim(0,0.032); a.legend(fontsize=9.5,loc='upper left'); a.grid(alpha=.25,axis='y')
a.set_title('그림 4-d · 학습만 좋아지고 밖에서는 나빠진다',fontsize=12)
a.annotate('독립 앵커에서 44% 악화',xy=(2.20,0.0046),xytext=(0.10,0.0292),fontsize=10,fontweight='bold',color='#B00020',
  arrowprops=dict(arrowstyle='->',color='#B00020',lw=1.3,connectionstyle='arc3,rad=-0.25'),bbox=dict(fc='#FFEBEE',ec='#B00020',boxstyle='round,pad=0.35'))
fig.savefig(B+'그림4_압력공선성.png',dpi=165,bbox_inches='tight',facecolor='white'); plt.close(fig)
print('fig3,4 ok')
