import pandas as pd, numpy as np
def tsnaive(s):
    t=pd.to_datetime(s)
    try: t=t.dt.tz_localize(None)
    except: t=t.dt.tz_convert(None)
    return t
lab=pd.read_csv('2026-07-21_ML개발/data/processed/labels_recycle1.csv'); lab['ts']=tsnaive(lab['Timestamp'])
raw=pd.read_csv('2026 Chemical Engineering Process design competition_rev0_Attachment1.csv')
raw.columns=[c.split(' Value')[0].strip() for c in raw.columns]; raw['ts']=tsnaive(raw['Timestamp'])
dp=pd.read_excel('DP_data.xlsx'); dp['ts']=tsnaive(dp['Timestamp'])

rawcols=['TT-1006.PV','FRC-1004.PV','FT-1004.PV','FT-1003.PV','PT-1004.PV','PT-1100.PV',
         'AT-1100.PV','TT-1100.PV','FT-1300.PV','FT-1001.PV','FT-1201.PV','FT-1002.PV']
m=lab[['ts','Campaign','Split','X_propene_recycle1','X_propene_conversion']].merge(
    raw[['ts']+rawcols],on='ts').merge(dp[['ts','DP_reactor']],on='ts')

# 파생변수 [계산]
m['total_flow_kgh']=m['FT-1003.PV']+m['FT-1004.PV']
m['dT_adiabatic_C']=m['TT-1100.PV']-m['TT-1006.PV']

out=pd.DataFrame({
 'Timestamp':m['ts'].dt.strftime('%Y-%m-%d %H:%M:%S'),
 'Campaign':m['Campaign'],'Split':m['Split'],
 # 공유 입력
 'TT-1006.PV':m['TT-1006.PV'],'FRC-1004.PV':m['FRC-1004.PV'],
 # 유량 (파생 total 포함)
 'FT-1004.PV':m['FT-1004.PV'],'FT-1003.PV':m['FT-1003.PV'],'total_flow_kgh':m['total_flow_kgh'],
 # 라벨
 'X_conv_r1_deployed':m['X_propene_recycle1'],'X_conv_r0':m['X_propene_conversion'],'DP_reactor_kPa':m['DP_reactor'],
 # 라벨재료(입력금지=누출)
 'PT-1004.PV':m['PT-1004.PV'],'PT-1100.PV':m['PT-1100.PV'],'AT-1100.PV':m['AT-1100.PV'],'TT-1100.PV':m['TT-1100.PV'],
 # 재순환(라벨 가정)
 'FT-1300.PV':m['FT-1300.PV'],'FT-1001.PV':m['FT-1001.PV'],
 # 검증전용
 'FT-1201.PV':m['FT-1201.PV'],'FT-1002.PV':m['FT-1002.PV'],
 # 에너지수지(파생)
 'dT_adiabatic_C':m['dT_adiabatic_C'],
})
out.to_csv('ML_마스터데이터_PV.csv',index=False,encoding='utf-8-sig')
print('written rows',len(out),'cols',len(out.columns))
print('columns:',list(out.columns))
# sanity
print('총유량 10/12 mean',out[out.Campaign=="10/12"]["total_flow_kgh"].mean())
print('DP 10/12',out[out.Campaign=="10/12"]["DP_reactor_kPa"].iloc[0])
print('결측치 합계',int(out.isna().sum().sum()))
print('Split 분포'); print(out.groupby(['Campaign','Split']).size())
