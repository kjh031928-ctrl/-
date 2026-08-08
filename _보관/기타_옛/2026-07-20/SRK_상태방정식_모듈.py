import numpy as np
R=8.314462618
# ===== AVEVA 제공값 (SRK constant.xlsx) =====
NAMES=['Benzene','Propylene','Propane','Cumene']
Tc=np.array([562.6 , 364.95, 369.82, 631.05])          # K
Pc=np.array([4924.39,4620.42,4249.57,3208.96])*1000.0  # kPa -> Pa
w =np.array([0.209 , 0.1435, 0.1529, 0.3266])
MW=np.array([78.115, 42.081, 44.097,120.196])          # g/mol
KIJ=np.zeros((4,4))
KIJ[0,2]=KIJ[2,0]=0.02      # Benzene-Propane
KIJ[1,2]=KIJ[2,1]=0.0083    # Propene-Propane
m=0.480+1.574*w-0.176*w**2
b_i=0.08664*R*Tc/Pc
def rho_srk(y,T,P,kij=KIJ,mw=MW,tc=Tc,pc=Pc,mm=m,bi=b_i):
    y=np.asarray(y,float); y=y/y.sum()
    alpha=(1+mm*(1-np.sqrt(T/tc)))**2
    a_i=0.42748*R**2*tc**2/pc*alpha
    sq=np.sqrt(np.outer(a_i,a_i))*(1-kij)
    a=float(y@sq@y); b=float(y@bi)
    A=a*P/(R*T)**2; B=b*P/(R*T)
    r=np.roots([1.0,-1.0,A-B-B**2,-A*B])
    r=r[np.abs(r.imag)<1e-8].real; r=r[r>B]
    if len(r)==0: return np.nan
    Z=r.max()
    return P*float(y@mw)/(Z*R*T)/1000.0   # kg/m3
def rho_of_X(X,nin,T,P):
    xi=nin[1]*X
    n=np.array([nin[0]-xi,nin[1]-xi,nin[2],nin[3]+xi])
    if np.any(n<-1e-9): return np.nan
    return rho_srk(np.clip(n,0,None),T,P)
