#!/usr/bin/env python3
from math import atan2,cos,sin
import numpy as np

def cpmg_Baldwin(kex,pb,d_omega,ncyc,T_relax,R2A,R2B):
    # pa=(1-pb)
    keg=kex*(1-pb)
    kge=kex*pb
    deltaR2=R2B-R2A
    nu_cpmg=ncyc/T_relax
    tau_cpmg=T_relax/(4.0*ncyc)  #time for one free precession element

    #########################################################################
    #get the real and imaginary components of the exchange induced shift
    g1=2*d_omega*(deltaR2+keg-kge)                   #same as carver richards zeta
    g2=(deltaR2+keg-kge)**2.0+4*keg*kge-d_omega**2   #same as carver richards psi
    g3=cos(0.5*atan2(g1,g2))*(g1**2.0+g2**2.0)**(1/4.0)   #trig faster than square roots
    g4=sin(0.5*atan2(g1,g2))*(g1**2.0+g2**2.0)**(1/4.0)   #trig faster than square roots
    #########################################################################
    #time independent factors
    N=complex(kge+g3-kge,g4)            #N=oG+oE
    NNc=(g3**2.+g4**2.)
    f0=(d_omega**2.+g3**2.)/(NNc)              #f0
    f2=(d_omega**2.-g4**2.)/(NNc)              #f2
    #t1=(-dw+g4)*(complex(-dw,-g3))/(NNc) #t1
    t2=(d_omega+g4)*(complex(d_omega,-g3))/(NNc) #t2
    t1pt2=complex(2*d_omega**2.,-g1)/(NNc)     #t1+t2
    oGt2=complex((deltaR2+keg-kge-g3),(d_omega-g4))*t2  #-2*oG*t2

    #test for definition used in the appendix of the paper
    #f00=0.5*(complex(deltaR2+kex-g3,dw-g4))
    #oGt2=-2*(kge-f00)*t2

    Rpre=(R2A+R2B+kex)/2.0   #-1/Trel*log(LpreDyn)

    #do calc in np
    E0= 2.0*tau_cpmg*g3  #derived from relaxation       #E0=-2.0*tcp*(f00R-f11R)
    E2= 2.0*tau_cpmg*g4  #derived from chemical shifts  #E2=complex(0,-2.0*tcp*(f00I-f11I))
    E1=(complex(g3,-g4))*tau_cpmg    #mixed term (complex) (E0-iE2)/2
    ex0b=(f0*np.cosh(E0)-f2*np.cos(E2))               #real
    ex0c=(f0*np.sinh(E0)-f2*np.sin(E2)*complex(0,1.)) #complex
    ex1c=(np.sinh(E1))                                   #complex
    v3=np.sqrt(ex0b**2.-1)  #exact result for v2v3
    y=np.power((ex0b-v3)/(ex0b+v3),ncyc)
    v2pPdN=(( complex(deltaR2+kex,d_omega) )*ex0c+(-oGt2-kge*t1pt2)*2*ex1c)        #off diagonal common factor. sinh fuctions

    Tog=(((1+y)/2+(1-y)/(2*v3)*(v2pPdN)/N))     
    Minty=Rpre-ncyc/(T_relax)*np.arccosh((ex0b).real)-1/T_relax*np.log((Tog.real))  #estimate R2eff



    result=[]
    for i in range(len(ncyc)):
        result.append((nu_cpmg[i],Minty[i]))
    return result

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    kex=1000.0
    pb=0.1
    d_omega=100.0
    R2A=10.0
    R2B=10.0
    T_relax=0.1

    ncyc=np.arange(1,201,1)
    result=cpmg_Baldwin(kex,pb,d_omega,ncyc,T_relax,R2A,R2B)
    x=[]
    y=[]
    for point in result:
        x.append(point[0])
        y.append(point[1])

    plt.plot(x,y)
    plt.xlabel('CPMG Frequency (Hz)')
    plt.ylabel('R2eff (s-1)')
    plt.title('CPMG Relaxation Dispersion - Baldwin Method')
    plt.show()