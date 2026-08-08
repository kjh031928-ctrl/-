# -*- coding: utf-8 -*-
"""압력(PT-1100)을 ML 입력에서 제외한 근거를 6개 패널로 시각화."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(fp)
plt.rcParams['font.family']=fm.FontProperties(fname=fp).get_name()
plt.rcParams['axes.unicode_minus']=False
plt.rcParams.update({'font.size':9,'axes.titlesize':10.5,'axes.titleweight':'bold','figure.dpi':110})

m=pd.read_excel('2026-07-21_ML개발/data/processed/cumene_ml_training_data.xlsx',sheet_name='Data')
m['ts']=pd.to_datetime(m['Timestamp']); m['day']=m['ts'].dt.strftime('%m/%d')
m['총공급']=m['FT-1004.PV']*(1+m['FRC-1004.PV'])
tr=m[m['Split']=='학습'].copy(); va=m[m['Split']=='검증']; an=m[m['Split']=='정상운전검사']
Y='X_propene_conversion'; logit=lambda p: np.log(p/(1-p))
r2=lambda y,yh: 1-((y-yh)**2).sum()/((y-y.mean())**2).sum()
CMAP={'10/13':'#D62728','10/14':'#2CA02C','10/15':'#1F77B4','10/16':'#FF7F0E'}
LBL={'10/13':'10/13 유량 램프','10/14':'10/14 컬럼 램프','10/15':'10/15 비율 램프','10/16':'10/16 온도 램프'}

fig=plt.figure(figsize=(16.5,10.2))
gs=fig.add_gridspec(2,3,hspace=0.42,wspace=0.28,left=0.055,right=0.985,top=0.885,bottom=0.07)
fig.suptitle('압력(PT-1100)을 ML 입력에서 제외한 근거 — 학습셋 10/13~10/16 (1,924행)',
             fontsize=15,fontweight='bold',y=0.965)
fig.text(0.5,0.925,'단순 상관은 근거가 되지 못한다(패널 1). 진짜 근거는 "압력이 이미 입력의 함수"라는 것이다(패널 2·3) + 넣으면 실제로 나빠진다(패널 4·5·6).',
         ha='center',fontsize=10,color='#444444')

# ── 패널 1 : 단순 산점도 (반증) ──
ax=fig.add_subplot(gs[0,0])
for d,g in tr.groupby('day'):
    ax.scatter(g['FT-1004.PV'],g['PT-1100.PV'],s=6,alpha=.55,c=CMAP[d],label=LBL[d],lw=0)
rp=np.corrcoef(tr['FT-1004.PV'],tr['PT-1100.PV'])[0,1]
ax.set_title('① 단순 상관으로는 "공선"이라 말할 수 없다',color='#B00020')
ax.set_xlabel('FT-1004 프로펜 유량 (kg/h)'); ax.set_ylabel('PT-1100 반응기 출구압력 (kPa)')
ax.legend(fontsize=7.5,loc='lower right',framealpha=.92,markerscale=2)
ax.text(.03,.95,f'학습셋 전체 r = {rp:+.3f}\n(0.99 가 아님)',transform=ax.transAxes,va='top',
        fontsize=10,fontweight='bold',color='#B00020',
        bbox=dict(fc='#FFEBEE',ec='#B00020',boxstyle='round,pad=0.45'))
ax.text(.60,.40,'10/15(파랑)는 유량이 46% 늘어도\n압력은 거의 안 오른다\n→ 유량–압력 관계가 캠페인마다 다르다',
        transform=ax.transAxes,ha='center',fontsize=8,style='italic',color='#333',
        bbox=dict(fc='white',ec='#BBBBBB',boxstyle='round,pad=0.35',alpha=.9))
ax.grid(alpha=.25)

# ── 패널 2 : 총공급 vs 압력 ──
ax=fig.add_subplot(gs[0,1])
for d,g in tr.groupby('day'):
    ax.scatter(g['총공급'],g['PT-1100.PV'],s=6,alpha=.55,c=CMAP[d],lw=0)
X=np.c_[np.ones(len(tr)),tr['총공급']]; b,*_=np.linalg.lstsq(X,tr['PT-1100.PV'].values,rcond=None)
xs=np.linspace(tr['총공급'].min(),tr['총공급'].max(),50)
ax.plot(xs,b[0]+b[1]*xs,'k--',lw=1.8,label='단일 직선 적합')
rt=np.corrcoef(tr['총공급'],tr['PT-1100.PV'])[0,1]
ax.set_title('② 압력이 실제로 따라가는 것은 "총 공급유량"')
ax.set_xlabel('총 공급유량  FT-1004 × (1 + FRC-1004)  (kg/h)'); ax.set_ylabel('PT-1100 (kPa)')
ax.text(.03,.95,f'r = {rt:+.4f}\n네 캠페인이 하나의 직선 위에 놓임',transform=ax.transAxes,va='top',
        fontsize=10,fontweight='bold',color='#1B5E20',
        bbox=dict(fc='#E8F5E9',ec='#1B5E20',boxstyle='round,pad=0.45'))
ax.text(.62,.045,'총공급 = 입력 F × (1+R) — 이미 가진 두 입력의 곱\n→ 압력은 새 정보가 아니라 입력의 함수',
        transform=ax.transAxes,ha='center',fontsize=8,style='italic',color='#333',
        bbox=dict(fc='white',ec='#BBBBBB',boxstyle='round,pad=0.35',alpha=.9))
ax.legend(fontsize=8,loc='center right'); ax.grid(alpha=.25)

# ── 패널 3 : 1:1 예측도 ──
ax=fig.add_subplot(gs[0,2])
Xp=np.c_[np.ones(len(tr)),tr[['총공급','TT-1006.PV']].values]
bp,*_=np.linalg.lstsq(Xp,tr['PT-1100.PV'].values,rcond=None); yh=Xp@bp; y=tr['PT-1100.PV'].values
for d,g in tr.groupby('day'):
    idx=tr['day'].values==d; ax.scatter(yh[idx],y[idx],s=6,alpha=.55,c=CMAP[d],lw=0)
lo,hi=min(y.min(),yh.min())-15,max(y.max(),yh.max())+15
ax.plot([lo,hi],[lo,hi],'k-',lw=1.5); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
ax.set_title('③ 압력은 입력만으로 6 kPa 안에 재현된다')
ax.set_xlabel('예측 압력  =  f(총공급, 입구온도)  (kPa)'); ax.set_ylabel('실측 PT-1100 (kPa)')
ax.text(.03,.95,f'R² = {r2(y,yh):.4f}\nRMSE = {np.sqrt(((y-yh)**2).mean()):.1f} kPa\n(압력 변동폭 {y.max()-y.min():.0f} kPa 의 1.4%)',
        transform=ax.transAxes,va='top',fontsize=10,fontweight='bold',color='#1B5E20',
        bbox=dict(fc='#E8F5E9',ec='#1B5E20',boxstyle='round,pad=0.45'))
ax.grid(alpha=.25)

# ── 패널 4 : VIF ──
ax=fig.add_subplot(gs[1,0])
def vif(cols):
    out=[]
    for v in cols:
        o=[x for x in cols if x!=v]; Xv=np.c_[np.ones(len(tr)),tr[o].values]
        bb,*_=np.linalg.lstsq(Xv,tr[v].values,rcond=None); out.append(1/(1-r2(tr[v].values,Xv@bb)))
    return out
V3=['TT-1006.PV','FT-1004.PV','FRC-1004.PV']; V4=V3+['PT-1100.PV']
v3,v4=vif(V3),vif(V4)
nm=['TT-1006\n온도','FT-1004\n유량','FRC-1004\n비율','PT-1100\n압력']
x=np.arange(4)
ax.bar(x-.2,v3+[0],.38,label='입력 3종 (최종 채택)',color='#2CA02C',ec='k',lw=.6)
ax.bar(x+.2,v4,.38,label='입력 4종 (압력 추가)',color='#D62728',ec='k',lw=.6)
ax.axhline(10,color='k',ls='--',lw=1.3); ax.text(-0.42,12.5,'VIF = 10  공선성 경고선',ha='left',fontsize=8.2,color='#333')
for i,v in enumerate(v4): ax.text(i+.2,v+1.5,f'{v:.1f}',ha='center',fontsize=8.5,fontweight='bold',color='#B00020')
for i,v in enumerate(v3): ax.text(i-.2,v+1.5,f'{v:.1f}',ha='center',fontsize=8.5,color='#1B5E20')
ax.set_xticks(x); ax.set_xticklabels(nm,fontsize=8.5); ax.set_ylabel('VIF (분산팽창인자)')
ax.set_title('④ 압력을 넣는 순간 셋 다 공선성 경고선을 넘는다')
ax.set_ylim(0,72); ax.legend(fontsize=8,loc='upper left'); ax.grid(alpha=.25,axis='y')
ax.text(.98,.68,f'상관행렬 조건수\n3종 {np.linalg.cond(np.corrcoef(((tr[V3]-tr[V3].mean())/tr[V3].std()).values.T)):.0f}  →  4종 {np.linalg.cond(np.corrcoef(((tr[V4]-tr[V4].mean())/tr[V4].std()).values.T)):.0f}',
        transform=ax.transAxes,ha='right',fontsize=8.5,fontweight='bold',
        bbox=dict(fc='#FFF3E0',ec='#E65100',boxstyle='round,pad=0.4'))

# ── 패널 5 : 성능 비교 ──
ax=fig.add_subplot(gs[1,1])
def run(vs):
    Xt=np.c_[np.ones(len(tr)),tr[vs].values]; bb,*_=np.linalg.lstsq(Xt,logit(tr[Y].values),rcond=None)
    o=[]
    for d in [tr,va,an]:
        Xd=np.c_[np.ones(len(d)),d[vs].values]
        o.append(float(np.sqrt(((1/(1+np.exp(-Xd@bb))-d[Y].values)**2).mean())))
    return o
mods=[('T, F, R\n(최종 채택)',V3,'#2CA02C'),('T, F, R, P\n(압력 추가)',V4,'#D62728'),('T, P, R\n(유량 대신 압력)',['TT-1006.PV','PT-1100.PV','FRC-1004.PV'],'#8E24AA')]
sets=['학습셋\n(10/13~16)','검증셋\n(10/17-18)','10/12 앵커\n(독립 정상운전)']
w=.26; x=np.arange(3)
for i,(lab,vs,cc) in enumerate(mods):
    o=run(vs); bars=ax.bar(x+(i-1)*w,o,w,label=lab,color=cc,ec='k',lw=.6)
    for xx,v in zip(x+(i-1)*w,o): ax.text(xx,v+0.0007,f'{v:.5f}',ha='center',fontsize=7.3,rotation=90,va='bottom')
ax.set_xticks(x); ax.set_xticklabels(sets,fontsize=8.5); ax.set_ylabel('RMSE (전환율 X)')
ax.set_title('⑤ 압력을 넣으면 학습만 좋아지고 밖에서는 나빠진다')
ax.set_ylim(0,0.037); ax.legend(fontsize=7.8,loc='upper left'); ax.grid(alpha=.25,axis='y')
ax.annotate('10/12 독립 앵커에서\n0.00282 → 0.00407  (44% 악화)',xy=(2.0,0.0045),xytext=(0.28,0.0322),
            fontsize=8.3,fontweight='bold',color='#B00020',
            arrowprops=dict(arrowstyle='->',color='#B00020',lw=1.4),
            bbox=dict(fc='#FFEBEE',ec='#B00020',boxstyle='round,pad=0.4'))

# ── 패널 6 : LOCO 계수 안정성 ──
ax=fig.add_subplot(gs[1,2])
days=['10/13','10/14','10/15','10/16']
def loco(vs):
    out=[]
    for d in days:
        s=tr[tr['day']!=d]; Xs=np.c_[np.ones(len(s)),s[vs].values]
        bb,*_=np.linalg.lstsq(Xs,logit(s[Y].values),rcond=None); out.append(bb[2])
    return np.array(out)
l3,l4=loco(V3),loco(V4)
x=np.arange(4)
ax.plot(x,l3,'o-',color='#2CA02C',lw=2,ms=8,label='입력 3종 (최종)')
ax.plot(x,l4,'s--',color='#D62728',lw=2,ms=8,label='입력 4종 (압력 추가)')
ax.set_xticks(x); ax.set_xticklabels([f'{d} 제외' for d in days],fontsize=8.5)
ax.set_ylabel('유량 계수 β_F (로짓 공간, 로그축)'); ax.set_yscale('log')
ax.set_yticks([5e-4,1e-3,2e-3,4e-3,8e-3]); ax.set_yticklabels(['0.0005','0.001','0.002','0.004','0.008'],fontsize=8.5)
ax.minorticks_off()
ax.set_title('⑥ 압력을 넣으면 유량 계수가 캠페인마다 요동친다')
ax.legend(fontsize=8,loc='upper left'); ax.grid(alpha=.25,which='both')
ax.text(.97,.10,f'변동폭  3종 {l3.max()-l3.min():.6f}\n         4종 {l4.max()-l4.min():.6f}  ({(l4.max()-l4.min())/(l3.max()-l3.min()):.1f}배)',
        transform=ax.transAxes,ha='right',fontsize=9,fontweight='bold',
        bbox=dict(fc='#FFEBEE',ec='#B00020',boxstyle='round,pad=0.4'))
fig.savefig('압력공선성_진단.png',dpi=170,bbox_inches='tight',facecolor='white')
print('saved')
