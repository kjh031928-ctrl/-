# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(fp); plt.rcParams['font.family']=fm.FontProperties(fname=fp).get_name()
plt.rcParams['axes.unicode_minus']=False
plt.rcParams.update({'font.size':9.5,'axes.titlesize':11,'axes.titleweight':'bold','figure.dpi':110})
B="/sessions/festive-inspiring-feynman/mnt/공정설계대회/"
df=pd.read_csv(B+"2026 Chemical Engineering Process design competition_rev0_Attachment1.csv",encoding='utf-8-sig')
df['ts']=pd.to_datetime(df['Timestamp']); df['day']=df['ts'].dt.strftime('%m/%d')
m=pd.read_excel(B+'2026-07-21_ML개발/data/processed/cumene_ml_training_data.xlsx',sheet_name='Data')
m['ts']=pd.to_datetime(m['Timestamp']); m['day']=m['ts'].dt.strftime('%m/%d')
g=df[df['day']=='10/13'].reset_index(drop=True); c=m[m['day']=='10/13'].reset_index(drop=True)
t=np.arange(len(g)); xt=np.arange(0,len(g),60); xl=[g['ts'][i].strftime('%H:%M') for i in xt]

# ── 그림 1 : 조작변수 프로파일 + 포화 ──
fig,ax=plt.subplots(2,1,figsize=(11,6.4),sharex=True,gridspec_kw={'hspace':0.18,'height_ratios':[1.25,1]})
a=ax[0]
a.plot(t,g['FC-1002.SP Value kg/h'],color='#B00020',lw=2.2,ls='--',label='FC-1002.SP  운전원 지시값')
a.plot(t,g['FC-1002.PV Value kg/h'],color='#1F4E79',lw=2.2,label='FC-1002.PV  실제 유량')
sat=(g['FC-1002.SP Value kg/h']-g['FC-1002.PV Value kg/h'])>50
a.fill_between(t,g['FC-1002.PV Value kg/h'],g['FC-1002.SP Value kg/h'],where=sat,color='#B00020',alpha=.16)
a.axhline(g['FC-1002.PV Value kg/h'].max(),color='#B00020',lw=1.1,ls=':')
a.annotate(f"실제 도달 상한 15,357 kg/h\n(지시 17,500 대비 2,143 미달)",xy=(300,15357),xytext=(196,16620),
  fontsize=9.3,fontweight='bold',color='#B00020',arrowprops=dict(arrowstyle='->',color='#B00020',lw=1.4),
  bbox=dict(fc='#FFEBEE',ec='#B00020',boxstyle='round,pad=0.4'))
a.text(258,13050,'제어 밸브 포화 구간\n19:04~21:50 · 167행(34.7%)',ha='center',fontsize=9,color='#B00020',style='italic')
a.set_ylabel('벤젠 총유량 (kg/h)'); a.legend(fontsize=9.3,loc='lower left'); a.grid(alpha=.25)
a.set_title('그림 1 · 10/13 조작변수 — 지시값은 17,500까지 올렸으나 실제 유량은 15,357에서 멈춘다')
b=ax[1]
b.plot(t,g['FT-1004.PV Value kg/h'],color='#2CA02C',lw=2,label='FT-1004 프로펜 유량 (ML 입력)')
b.set_ylabel('프로펜 유량 (kg/h)',color='#2CA02C'); b.tick_params(axis='y',labelcolor='#2CA02C')
b2=b.twinx(); b2.plot(t,g['FRC-1004.PV Value'],color='#7B1FA2',lw=1.6,label='FRC-1004 벤젠/프로펜 비율')
b2.set_ylabel('벤젠/프로펜 비율 (−)',color='#7B1FA2'); b2.tick_params(axis='y',labelcolor='#7B1FA2'); b2.set_ylim(4.5,5.5)
b.set_xticks(xt); b.set_xticklabels(xl); b.set_xlabel('시각 (UTC)'); b.grid(alpha=.25)
b.set_title('비율이 5.0으로 고정돼 있으므로 프로펜은 벤젠의 1/5로 자동 추종된다 (운전원은 프로펜에 손대지 않았다)',fontsize=9.8)
h1,l1=b.get_legend_handles_labels(); h2,l2=b2.get_legend_handles_labels()
b.legend(h1+h2,l1+l2,fontsize=9,loc='lower right')
fig.savefig(B+'그림1_조작변수와_포화.png',dpi=165,bbox_inches='tight',facecolor='white'); plt.close(fig)

# ── 그림 2 : 체류시간 상쇄 ──
R=8.314462
ntot=(c['n_propene_kmol_h']+c['n_propane_kmol_h']+c['n_benzene_kmol_h']).values
Vd=ntot*1000*R*(c['TT-1006.PV'].values+273.15)/(c['PT-1100.PV'].values*1000)
fig,ax=plt.subplots(1,2,figsize=(12.6,4.9),gridspec_kw={'wspace':0.26})
a=ax[0]
for y,cc,lb in [(ntot,'#1F77B4','총 몰유량 n  (+22.9%)'),(c['PT-1100.PV'].values,'#D62728','압력 P  (+20.4%)'),(Vd,'#2CA02C','부피유량 V̇ = nRT/P  (+2.4%)')]:
    a.plot(t,y/y[0]*100,color=cc,lw=2.2,label=lb)
a.axhline(100,color='#999',lw=1,ls=':')
a.set_xticks(xt); a.set_xticklabels(xl,fontsize=8.5); a.set_xlabel('시각 (UTC)')
a.set_ylabel('15:00 기준 = 100')
a.set_title('그림 2 · 몰유량 증가를 압력 증가가 상쇄한다')
a.legend(fontsize=9.3,loc='upper left'); a.grid(alpha=.25)
a.text(.97,.05,'분자 n 이 22.9% 늘어도 분모 P 가 20.4% 늘어\n부피유량은 2.4%만 변한다 → 체류시간 사실상 불변',
  transform=a.transAxes,ha='right',fontsize=8.8,style='italic',
  bbox=dict(fc='white',ec='#BBB',boxstyle='round,pad=0.4'))
a2=ax[1]
lo=slice(121,180); hi=slice(301,360)
lab=['총 몰유량 n\n(kmol/h)','압력 P\n(kPa)','부피유량 V̇\n(m³/h)','입구온도 T\n(°C)','전환율 X\n(−)']
L=[ntot[lo].mean(),c['PT-1100.PV'].values[lo].mean(),Vd[lo].mean(),c['TT-1006.PV'].values[lo].mean(),c['X_propene_conversion'].values[lo].mean()]
Hh=[ntot[hi].mean(),c['PT-1100.PV'].values[hi].mean(),Vd[hi].mean(),c['TT-1006.PV'].values[hi].mean(),c['X_propene_conversion'].values[hi].mean()]
pct=[(h/l-1)*100 for l,h in zip(L,Hh)]
cols=['#1F77B4','#D62728','#2CA02C','#FF7F0E','#7B1FA2']
bars=a2.barh(range(5),pct,color=cols,ec='k',lw=.6,height=.6)
for i,(p,l,h) in enumerate(zip(pct,L,Hh)):
    a2.text(p+(1.6 if p>0 else -1.6),i,f'{p:+.1f}%',va='center',ha='left' if p>0 else 'right',fontweight='bold',fontsize=10)
    a2.text(0.5,i-0.33,f'{l:,.1f} → {h:,.1f}',fontsize=8,color='#555')
a2.set_yticks(range(5)); a2.set_yticklabels(lab,fontsize=9); a2.invert_yaxis()
a2.axvline(0,color='k',lw=1); a2.set_xlim(-6,62); a2.set_xlabel('저유량 구간(17:01~17:59) → 고유량 구간(20:01~20:59) 변화율')
a2.set_title('입구온도가 같은 두 구간 비교 — 순수 유량 효과')
a2.grid(alpha=.25,axis='x')
fig.savefig(B+'그림2_체류시간_상쇄.png',dpi=165,bbox_inches='tight',facecolor='white'); plt.close(fig)
print('fig1,2 ok')
