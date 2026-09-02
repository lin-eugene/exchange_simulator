#!/usr/bin/python

#####################################################################
# Simulate exchange in 2D for simulating titrations
# A.Baldwin 27th May 2014
#


import numpy,sys,math,os,copy,sys
from math import cos,sin,atan2,log10,fabs,sqrt
from datetime import datetime,timedelta
# from scipy import mat,zeros,log
from scipy.linalg import expm
from scipy.optimize import leastsq
from scipy.special import betainc

#return the value of a Lorentzian function
def Lorentz(w,dw,R):
    return R/((w-dw)**2.+R**2.)/numpy.pi

#calculate a 1D exchange spectrum
def spec1D(freq,kex,pb,dw,R2g,R2e,outfile,verb='n'):
    pa=(1-pb)
    keg=kex*(1-pb)
    kge=kex*pb
    deltaR2=R2e-R2g
    #########################################################################
    g1=2*dw*(deltaR2+keg-kge)                   #same as carver richards zeta
    g2=(deltaR2+keg-kge)**2.0+4*keg*kge-dw**2   #same as carver richards psi
    g3=cos(0.5*atan2(g1,g2))*(g1**2.0+g2**2.0)**(1/4.0)   #trig faster than square roots
    g4=sin(0.5*atan2(g1,g2))*(g1**2.0+g2**2.0)**(1/4.0)   #trig faster than square roots
    #########################################################################
    N=complex(kge+g3-kge,g4)            #N=oG+oE

    f00=0.5*complex(deltaR2+kex-g3,dw-g4)
    f11=0.5*complex(deltaR2+kex+g3,dw+g4)

    f00R=0.5*(deltaR2+kex-g3)
    f00I=0.5*(dw-g4)
    f11R=0.5*(deltaR2+kex+g3)
    f11I=0.5*(dw+g4)

    sigG=numpy.real((pa*f11+pb*(kex-f00))/N)  * Lorentz(freq,f00I,f00R+R2g)
    sigE=numpy.real((-pa*f00+pb*(f11-kex))/N) * Lorentz(freq,f11I,f11R+R2g)

    if(verb=='y'):
        outy=open(outfile,'w')
        for i in range(len(w)):
            outy.write('%f\t%f\t%f\n' % (w[i],sigG[i],sigE[i]))
        outy.close()
        print(numpy.sum(sigG+sigE)) #print intensity. should be 1
    return sigG,sigE


#calculate the 1D spectrum at the max of the peaks
def spec1Dproj(kex,pb,dw,R2g,R2e):
    pa=(1-pb)
    keg=kex*(1-pb)
    kge=kex*pb
    deltaR2=R2e-R2g
    #########################################################################
    g1=2*dw*(deltaR2+keg-kge)                   #same as carver richards zeta
    g2=(deltaR2+keg-kge)**2.0+4*keg*kge-dw**2   #same as carver richards psi
    g3=cos(0.5*atan2(g1,g2))*(g1**2.0+g2**2.0)**(1/4.0)   #trig faster than square roots
    g4=sin(0.5*atan2(g1,g2))*(g1**2.0+g2**2.0)**(1/4.0)   #trig faster than square roots
    #########################################################################
    N=complex(kge+g3-kge,g4)            #N=oG+oE

    f00=0.5*complex(deltaR2+kex-g3,dw-g4)
    f11=0.5*complex(deltaR2+kex+g3,dw+g4)

    f00R=0.5*(deltaR2+kex-g3)
    f00I=0.5*(dw-g4)
    f11R=0.5*(deltaR2+kex+g3)
    f11I=0.5*(dw+g4)

    #sigG=numpy.real((pa*f11+pb*(kex-f00))/N)  * Lorentz(freq,f00I,f00R+R2g)
    #sigE=numpy.real((-pa*f00+pb*(f11-kex))/N) * Lorentz(freq,f11I,f11R+R2g)

    #evaluate the ground state at two frequencies
    sigG_g=numpy.real((pa*f11+pb*(kex-f00))/N)  * Lorentz(f00I,f00I,f00R+R2g)
    sigG_e=numpy.real((pa*f11+pb*(kex-f00))/N)  * Lorentz(f11I,f00I,f00R+R2g)

    #evaluate the excited state at two frequencies
    sigE_g=numpy.real((-pa*f00+pb*(f11-kex))/N) * Lorentz(f00I,f11I,f11R+R2g)
    sigE_e=numpy.real((-pa*f00+pb*(f11-kex))/N) * Lorentz(f11I,f11I,f11R+R2g)

    Gsum=sigG_g+sigE_g
    Esum=sigG_e+sigE_e

    return Gsum,f00I,Esum,f11I
    


#calculate two 1D spectra, recombine to get 2D
def spec2D(freqH,freqN,sfrq,dfrq,kex,pb,dwH,dwN,R2g,R2e,outfile,verb='n',window='n'):



    sigXG,sigXE=spec1D(freqH,kex,pb,dwH,R2g,R2e,'spec1D.out',verb='n')
    sigYG,sigYE=spec1D(freqN,kex,pb,dwN,R2g,R2e,'spec1D.out',verb='n')

    TH=0.056  #acquisiton time H
    TN=0.047  #acquisition time N


    #this is slowest bit!
    xv,yv=numpy.meshgrid(sigXG+sigXE,sigYG+sigYE)

    if(window=='y'):
        #F(t.g)=F(t)convF(g). g is window. F(t) is known.
        #g=sin(pi t/T)
        #F(g)= (1-e^(iwT))piT/(pi^2-w^2T^2)
        #g=cos(pi t/T)
        #F(g)= i(1-e^(iwT))piT/(pi^2-w^2T^2)


        #g=sin(pi*off+pi*(end-off)* t/T)
        #off=0.5, end=1.0
        #F(g)= (pi e^(iwT)-2iTw)2T/(pi^2-4w^2T^2)

        Hshif=(freqH[0]+freqH[len(freqH)-1])/2.
        Nshif=(freqN[0]+freqN[len(freqN)-1])/2.


        xa=(numpy.pi*numpy.exp(complex(0,1)*(freqH-Hshif)*TH)-2*complex(0,1)*TH*(freqH-Hshif))*2.*TH/(numpy.pi**2.-4*(freqH-Hshif)**2.*TH**2.) #exact FT of window 
        ya=(numpy.pi*numpy.exp(complex(0,1)*(freqN-Nshif)*TN)-2*complex(0,1)*TN*(freqN-Nshif))*2.*TN/(numpy.pi**2.-4*(freqN-Nshif)**2.*TN**2.) #exact FT of window 

        #equivalent
        yv=numpy.apply_along_axis(lambda m: numpy.convolve(m,ya,'same'),axis=0,arr=yv)
        #for i in range(len(freqH)): #convolve nitrogen
        #    yv[:,i]=numpy.convolve(yv[:,i],ya,'same')

        #not equivalent? not sure why. the one we use is prob. slow.
        #xv=numpy.apply_along_axis(lambda m: numpy.convolve(m,xa,'same'),axis=1,arr=xv)
        for i in range(len(freqN)): #convolve proton
            xv[i,:]=numpy.convolve(xv[i,:],xa,'same')



    spec=xv*yv


    #get the max positions
    Hmax=numpy.argmax(numpy.fabs(sigXG+sigXE))
    Nmax=numpy.argmax(numpy.fabs(sigYG+sigYE))
    maxspec= spec[Nmax,Hmax]
    hfreq= freqH[Hmax]/(2.*math.pi*sfrq)
    nfreq= freqN[Nmax]/(2.*math.pi*dfrq)

    #print 'integral:',numpy.sum(spec) #should be 1
    if(verb=='y'):
        outy=open(outfile,'w')
        for i in range(len(freqH)):
            for j in range(len(freqN)):
                outy.write('%f\t%f\t%f\n' % (freqH[i],freqN[j],spec[i,j]))
            outy.write('\n')
        outy.close()


    return hfreq,nfreq,maxspec,spec



#calculate two 1D spectra, recombine to get 2D
def spec2DFull(freqH,freqN,sfrq,dfrq,kex,pb,dwH,dwN,R2g,R2e,outfile,verb='n'):

    sigXG,sigXE=spec1D(freqH,kex,pb,dwH,R2g,R2e,'spec1D.out',verb='n')
    sigYG,sigYE=spec1D(freqN,kex,pb,dwN,R2g,R2e,'spec1D.out',verb='n')

    #this is slowest bit!
    xv,yv=numpy.meshgrid(sigXG+sigXE,sigYG+sigYE)
    spec=xv*yv

    #get the max positions
    Hmax=numpy.argmax(numpy.fabs(sigXG+sigXE))
    Nmax=numpy.argmax(numpy.fabs(sigYG+sigYE))
    maxspec= spec[Nmax,Hmax]
    hfreq= freqH[Hmax]/(2.*math.pi*sfrq)
    nfreq= freqN[Nmax]/(2.*math.pi*dfrq)

    #print 'integral:',numpy.sum(spec) #should be 1
    if(verb=='y'):
        outy=open(outfile,'w')
        for i in range(len(freqH)):
            for j in range(len(freqN)):
                outy.write('%f\t%f\t%f\n' % (freqH[i],freqN[j],spec[i,j]))
            outy.write('\n')
        outy.close()

    return hfreq,nfreq,maxspec,spec


#get max intensities of peaks in 2D spectrum
def spec2Dproj(sfrq,dfrq,kex,pb,dwH,dwN,R2g,R2e):
    GHsum,f00H,EHsum,f11H=spec1Dproj(kex,pb,dwH,R2g,R2e)
    GNsum,f00N,ENsum,f11N=spec1Dproj(kex,pb,dwN,R2g,R2e)
    #get the effective signal from the combined
    Gsum=GHsum*GNsum
    Esum=EHsum*ENsum

    #work out which is biggest.
    #it might be possible to prove that this is never Esum. 
    #in which case half of these evaluations can be discarded.
    if(Gsum>Esum): 
        maxspec=Gsum
        hfreq=f00H/(2.*math.pi*sfrq)
        nfreq=f00N/(2.*math.pi*dfrq)
    if(Gsum<=Esum):
        #rint 'lies!'
        #sys.exit(100)
        maxspec=Gsum
        hfreq=f11H/(2.*math.pi*sfrq)
        nfreq=f11N/(2.*math.pi*dfrq)
    return hfreq,nfreq,maxspec


################################################################

#for population analysis: return the composition concs
#from knowledge of Kds and free concentrations
def GetComponents(kd,pfree,mfree):
    MP=(1/kd)*pfree*mfree
    Mtot=mfree+MP
    Ptot=pfree+MP
    return MP,Mtot,Ptot

#for determination of amount of free monomer/dimer
def MinFunc(x,b):
    mfree=x[0]
    pfree=x[1]
    Mtot=b[0]
    Ptot=b[1]
    kd=b[2]
    MP,MtotCalc,PtotCalc=GetComponents(kd,pfree,mfree)
    return Mtot-MtotCalc,Ptot-PtotCalc

#Miniminse function to get appropriate amounts of components from Kd
def GetComposition(kd,Mtot,Ptot):

    #for a give kd/Mtot/Ptot, get mfree and pfree
    #x0=leastsq(MinFunc,[Mtot/10,Ptot/10],args=[Mtot,Ptot,kd])
    #mfree=x0[0][0]
    #pfree=x0[0][1]
    #print 
    #print mfree,pfree

    #or, slightly more cleverly....
    #Mtot=Mfree+MP = Mfree + 1/kd*pfree*mfree
    #Ptot=Pfree+MP = Pfree + 1/kd*pfree*mfree
    #Pfree= Ptot / (1+ mfree/kd)
    
    #Mtot=Mfree + 1/kd*mfree Ptot / (1+ mfree/kd)
    #0=mfree+ mfree^2/kd + 1/kd*mfree Ptot -Mtot*mfree/kd - Mtot
    a=1/kd
    b=Ptot/kd-Mtot/kd+1
    c=-Mtot
    mfree=( -b + sqrt(b**2.-4*a*c))/(2*a)
    pfree= Ptot / (1.+ mfree/kd)
    #print mfree,pfree

    MP,Mtot,Ptot=GetComponents(kd,pfree,mfree)
    return mfree,pfree,MP

################################################################

#master function. Take parameters and return a spectrum in 1 and 2D 
def RunNMR(sfrq,dfrq,Mtot,Ptot,kd,kplus,R2e,R2g,dwH,dwN,freqH,freqN,tag,verb='n',window='n'):
    mfree,pfree,MP=GetComposition(kd,Mtot,Ptot)

    #recast kex into new variables:
    #kex=k+[Pfree]+k-
    #kd=[M][P]/[MP]=k-/(k+)
    kex=kplus*(pfree+kd)
    pb=MP/Mtot

    hfreq,nfreq,maxspec,spec=spec2D(freqH,freqN,sfrq,dfrq,kex,pb,dwH,dwN,R2g,R2e,'out/spec2D.'+tag+'.out',verb,window=window)
    return hfreq,nfreq,maxspec,spec


#master function. Take parameters and return a spectrum in 1 and 2D 
def RunNMRFull(sfrq,dfrq,Mtot,Ptot,kd,kplus,R2e,R2g,dwH,dwN,freqH,freqN,tag,verb='n',window='n'):
    mfree,pfree,MP=GetComposition(kd,Mtot,Ptot)

    #recast kex into new variables:
    #kex=k+[Pfree]+k-
    #kd=[M][P]/[MP]=k-/(k+)
    kex=kplus*(pfree+kd)
    pb=MP/Mtot

    hfreq,nfreq,maxspec,spec=spec2DFull(freqH,freqN,sfrq,dfrq,kex,pb,dwH,dwN,R2g,R2e,'out/spec2D.'+tag+'.out',verb,window=window)
    return hfreq,nfreq,maxspec,spec

#Take parametrers and return just the proton, indirect and intensity of max peak
def RunNMRproj(sfrq,dfrq,Mtot,Ptot,kd,kplus,R2e,R2g,dwH,dwN,tag,verb='n'):
    mfree,pfree,MP=GetComposition(kd,Mtot,Ptot)
    #recast kex into new variables:
    #kex=k+[Pfree]+k-
    #kd=[M][P]/[MP]=k-/(k+)
    kex=kplus*(pfree+kd)
    pb=MP/Mtot
    hfreq,nfreq,maxspec=spec2Dproj(sfrq,dfrq,kex,pb,dwH,dwN,R2g,R2e,'out/spec2D.'+tag+'.out',verb)
    return hfreq,nfreq,maxspec


################################################################

#gnuplot plotting script for 2D projections of data
def GnuSpec(Mtot,Ptot,Kd,R2g,R2e,specmax):
    gnu=open('gnu.gp','w')
    gnu.write('set term post eps enh color solid\n')
    gnu.write('set size square\n')
    gnu.write('set xlabel \'Hfreq\'\n')
    gnu.write('set ylabel \'Nfreq\'\n')
    gnu.write('set cbrange[0:%f]\n' % (specmax))
    gnu.write('set cntrparam levels 20\n')
    gnu.write('set palette defined(0\'white\',1\'red\',2\'yellow\')\n')
    gnu.write('set ticslevel 0\n')
    gnu.write('set view map\n')
    gnu.write('set contour base\n')
    gnu.write('unset key\n')
    gnu.write('unset surface\n')
    gnu.write('set isosamples 1000,1000\n')
    for i in range(len(Ptot)):
        gnu.write('set label sprintf("Mtot:  %s.2f uM",%f) at graph 0.1,0.9\n' % ('%',Mtot*1E6))
        gnu.write('set label sprintf("Ptot:  %s.2f uM",%f) at graph 0.1,0.85\n' % ('%',Ptot[i]*1E6))
        gnu.write('set label sprintf("kd:    %s.2f uM",%f) at graph 0.1,0.8\n' % ('%',kd*1E6))

        gnu.write('set label sprintf("R2g:   %s.2f s-1",%f) at graph 0.1,0.75\n' % ('%',R2g))
        gnu.write('set label sprintf("R2e:   %s.2f s-1",%f) at graph 0.1,0.7\n' % ('%',R2e))

        gnu.write('set output \'figs/spec.%i.eps\'\n' % (i))
        gnu.write('splot \'out/spec2D.%i.out\' u 1:2:3 w li lc palette\n' % (i))
        gnu.write('unset label\n')
    gnu.close()
    os.system('gnuplot gnu.gp')

#make gnuplot scripts for 1D analysis
def GnuTrace(kd,kplus,dwH,dwN):
    gnu=open('gnu.gp','w')
    gnu.write('set term post eps enh color solid\n')
    gnu.write('set size square\n')
    gnu.write('unset key\n')

    gnu.write('set xlabel \'[P_{Tot}] (uM)\'\n')
 
    gnu.write('set label sprintf("kd:    %s.2f uM",%f) at graph 0.1,0.85\n' % ('%',kd*1E6))
    gnu.write('set label sprintf("kplus: %s.2e uM",%e) at graph 0.1,0.8\n' % ('%',kplus))

    gnu.write('set output \'figs/dat1.eps\'\n')
    gnu.write('set ylabel \'Relative Intensity\'\n')
    gnu.write('set yrange[0:1.1]\n')
    gnu.write('plot \'out/dat.sim\' u 1:4,\'\' u 1:4 w li\n')
    gnu.write('unset label\n')

    gnu.write('set label sprintf("dwH: %s.2f rad s-1",%f) at graph 0.1,0.85\n' % ('%',dwH))
    gnu.write('set output \'figs/dat2.eps\'\n')
    gnu.write('set ylabel \'dwH\'\n')
    gnu.write('set yrange[0:*]\n')
    gnu.write('plot \'out/dat.sim\' u 1:2,\'\' u 1:5 w li\n')
    gnu.write('unset label\n')

    gnu.write('set label sprintf("dwN: %s.2f rad s-1",%f) at graph 0.1,0.85\n' % ('%',dwN))
    gnu.write('set output \'figs/dat3.eps\'\n')
    gnu.write('set ylabel \'dwN\'\n')
    gnu.write('set yrange[0:*]\n')
    gnu.write('plot \'out/dat.sim\' u 1:3,\'\' u 1:6 w li\n')
    gnu.write('unset label\n')
    gnu.close()
    os.system('gnuplot gnu.gp')


################################################################

#auxillary function: find maximum and index of a list
def findmax1(array):
    imax=0
    maxval=array[0]
    for i in range(len(array)):
        if(array[i]>maxval):
            maxval=array[i]
            imax=i
    return maxval,imax

#return chi2 from array of chis
def getChi2(chi):
    return numpy.sum(numpy.power(chi,2))


#find minimum of a specific column
def findmin(array,col):
    imin=0
    min=array[0][col]
    for i in range(len(array)):
        if(array[i][col]<min):
            imin=i
            min=array[i][col]
    return min,imin

#return the Plvl from two chi2 values and dofs
def GetPlvl(chi2Simple,chi2Complex,dofSimple,dofComplex):
    v1=dofSimple*1.
    v2=dofComplex*1.
    Z=v1-v2              #difference in dof
    F=(chi2Simple*1.-chi2Complex*1.)/((v1-v2)*chi2Complex*1./v2*1.) #the F statistic
    ex=v2/(v2+Z*F)
    plvl=betainc(v2/2,Z/2,ex) #integrate to get Plvl
    return plvl


def ppm_to_rads(freq,sfrq):
    return freq*2*math.pi*sfrq

def rads_to_ppm(freq,sfrq):
    return freq/(2*math.pi*sfrq)


################################################################

#return 1D and 2D spectral values
def CalcSpec2D(sfrq,dfrq,freqH,freqN,Mtot,Ptot,kd,kplus,R2e,R2g,dwH,dwN,verb='n',name=0,window='n'):
    hfreq=numpy.zeros(len(Ptot))
    nfreq=numpy.zeros(len(Ptot))
    maxspec=numpy.zeros(len(Ptot))
    specAll=[]
    for i in range(len(Ptot)):#for each concentration
        hfreq[i],nfreq[i],maxspec[i],spec=RunNMR(sfrq,dfrq,Mtot,Ptot[i],kd,kplus,R2e,R2g,dwH,dwN,freqH[i],freqN[i],str(i),verb,window)
        specAll.append(spec)
    vals=[]
    for i in range(len(specAll)):
        vals.append(numpy.max(specAll[i]))
    maxval,imax=findmax1(vals)

    for i in range(len(specAll)):
        specAll[i]=specAll[i]/maxval

    if(verb=='y'):
        GnuSpec(Mtot,Ptot,kd,R2g,R2e,numpy.max(maxspec))
        outy=open('out/dat.sim','w')
        for i in range(len(Ptot)):
            mfree,pfree,MP=GetComposition(kd,Mtot,Ptot[i])
            outy.write('%f\t%f\t%f\t%f\t%f\t%f\n' % (Ptot[i]*1E6,hfreq[i],nfreq[i],maxspec[i]/numpy.max(maxspec),dwH*MP/Mtot,dwN*MP/Mtot))
        outy.close()
        GnuTrace(kd,kplus,dwH,dwN)
        os.system('arraygraph.py 4 6 0 0 0 0 `ls figs/*.eps`')

    return hfreq,nfreq,maxspec,specAll

#return 1D spectral values
def CalcSpec2Dproj(sfrq,dfrq,Mtot,Ptot,kd,kplus,R2e,R2g,dwH,dwN,verb='n'):
    hfreq=numpy.zeros(len(Ptot))
    nfreq=numpy.zeros(len(Ptot))
    maxspec=numpy.zeros(len(Ptot))
    specAll=[]
    for i in range(len(Ptot)):
        hfreq[i],nfreq[i],maxspec[i]=RunNMRproj(sfrq,dfrq,Mtot[i],Ptot[i],kd,kplus,R2e,R2g,dwH,dwN,str(i))
    if(verb=='y'):
        GnuSpec(Mtot,Ptot,kd,R2g,R2e,numpy.max(maxspec))
        outy=open('out/day.sim','w')
        for j in range(len(Ptot)):
            mfree,pfree,MP=GetComposition(kd,Mtot[j],Ptot[j])
            outy.write('%f\t%f\t%f\t%f\t%f\t%f\n' % (Ptot[j]*1E6,hfreq[j],nfreq[j],maxspec[j]/numpy.max(maxspec),dwH*MP/Mtot,dwN*MP/Mtot))
        outy.close()
        GnuTrace(kd,kplus,dwH,dwN)
        os.system('arraygraph.py 4 6 0 0 0 0 `ls figs/*.eps`')
    return hfreq,nfreq,maxspec


################################################################

#functions for fast exchange analysis of chemical shift changes
#analytical solution to Kd equation in fast exchange.
def olyafunc(Ptot, Kd,Mtot,dwLim): 
    return dwLim*(0.5*(Mtot + Ptot + Kd)/Mtot - 0.5*numpy.sqrt((Mtot+Ptot+Kd)**2-4*Mtot*Ptot)/Mtot)

def olyaKdMin(x,b):
    kd=numpy.fabs(x[0])
    dw0a=x[1]
    Mtot=b[0]
    Ptot=b[1]
    ydata=b[2]
    #ydatb=b[3]
    ycalca=olyafunc(Ptot,kd,Mtot,dw0a)
    #ycalcb=olyafunc(Ptot,kd,Mtot,dw0b)
    #chi=[]
    #for i in range(len(ycalca)):
    #    chi.append(ycalca[i]-ydata[i])
    #    chi.append(ycalcb[i]-ydatb[i])
    return ycalca-ydata

def olyaKd(Mtot,Ptot,ydata):
    kd=1/10E-6
    dwa=numpy.max(ydata)
    #dwb=numpy.max(ydatb)
    x0=leastsq(olyaKdMin,[kd,dwa],args=[Mtot,Ptot,ydata])
    return x0[0]


def olyaDw(x,b):
    dw=x[0]
    kd=b[0]
    xdata=b[1] #ptot,mtot
    ydata=b[2]
    ycalc=olyafunc(xdata[1],kd,xdata[0],dw)
    return ycalc-ydata


def olyaKdcalcMin(kd,xdataGlob,ydataGlob,dwlim,dwlimCurr):
    ycalcGlob=[]
    for i in range(len(xdataGlob)):
        x0=leastsq(olyaDw,[dwlim[i],],args=[kd,(xdataGlob[i][0],xdataGlob[i][1]),ydataGlob[i]])
        dwlimCurr[i]=x0[0]
        ycalc=olyafunc(xdataGlob[i][1],kd,xdataGlob[i][0],dwlimCurr[i])
        ycalcGlob.append(ycalc)
    return ycalcGlob

def olyaKdcalc(kd,xdataGlob,ydataGlob,dwlimCurr):
    ycalcGlob=[]
    for i in range(len(xdataGlob)):
        #print len(xdataGlob[i]),(dwlimCurr[i]),kd,Mtot
        ycalc=olyafunc(xdataGlob[i][1],kd,xdataGlob[i][0],dwlimCurr[i])
        ycalcGlob.append(ycalc)
    return ycalcGlob


def getChiGlob(ycalcGlob,ydataGlob):
    chi=[]
    for i in range(len(ycalcGlob)):
        for j in range(len(ycalcGlob[i])):
            chi.append(ycalcGlob[i][j]-ydataGlob[i][j])
    return chi


def olyaKdGlob(x,b):
    kd=numpy.fabs(x[0])
    print(kd)
    Mtot=b[0]
    xdataGlob=b[1]
    ydataGlob=b[2]
    dwlim=b[3]
    dwlimCurr=b[4]
    ycalcGlob=olyaKdcalcMin(kd,xdataGlob,ydataGlob,dwlim,dwlimCurr)
    chi=getChiGlob(ycalcGlob,ydataGlob)
    return chi


def olyaKdGlob2(x,b):
    xdataGlob=b[0]
    ydataGlob=b[1]
    dwlimCurr=b[2]
    kd=numpy.fabs(x[0])
    for i in range(len(x)-1):
        dwlimCurr[i]=x[1+i]
    ycalcGlob=olyaKdcalc(kd,xdataGlob,ydataGlob,dwlimCurr)
    chi=getChiGlob(ycalcGlob,ydataGlob)
    return chi


####################################################################

#find Kd via a grid serach. For error analysis
def GridSearchfastKdGlob(resy,kdmin,kdmax,kdgrid,xdataGlob,ydataGlob,dwlim,dwlimCurr,outfile,verb='n',axis='log'):
    fits=[]
    for i in range(kdgrid):
        chi=[]
        if(axis=='log'):
            kd=kdmin*10**(numpy.log10(kdmax/kdmin)*(i)/(kdgrid-1.))
        elif(axis=='lin'):
            kd=kdmin+i/(kdgrid-1.)*(kdmax-kdmin)
        ycalcGlob=olyaKdcalcMin(kd,xdataGlob,ydataGlob,dwlim,dwlimCurr)
        chi=getChiGlob(ycalcGlob,ydataGlob)
        chi2=getChi2(chi)
        fits.append((kd,chi2))


    mins=findmin(fits,1)
    kdBest=fits[mins[1]][0]
    minchi2=mins[0]

    ycalcGlob=olyaKdcalcMin(kdBest,xdataGlob,ydataGlob,dwlim,dwlimCurr)
    chi=getChiGlob(ycalcGlob,ydataGlob)
    chi2=getChi2(chi)
    if(chi2!=minchi2):
        print('BALLS!')
        sys.exit(100)

    dof=len(chi)-len(resy)-1. #number of degrees of freedom
    sigma=chi2/dof
    #aveErr=numpy.sqrt(sigma)
    #print 'FractionalError:',aveErr/cspMax
    
    if(verb=='y'):
        
        outy=open(outfile,'w')
        ey2=0.0
        ey=0.0
        en=0.0
        for i in range(len(fits)):
            yval=numpy.exp(-(fits[i][1]-chi2)/2/sigma)
            xval=fits[i][0]
            outy.write('%f\t%e\t%e\n' % (xval,fits[i][1],yval))
            ey+=xval*yval
            ey2+=xval*xval*yval
            en+=yval
        outy.write('\n\n')
        kdmean=ey/en
        kdstdev=numpy.sqrt(ey2/en-(ey/en)**2.)
        for i in range(len(fits)):
            xval=fits[i][0]
            yval=numpy.exp(-(kdmean-xval)**2./(2*kdstdev**2.))
            outy.write('%f\t%e\t%e\n' % (xval,fits[i][1],yval))
        outy.close()
        


        gnu=open('gnu/gnu.gp','w')
        gnu.write('set term post eps enh color solid\n')
        gnu.write('set output \'figs/gridfastKd.eps\'\n')
        gnu.write('set title \'grid search global Kd\'\n')
        gnu.write('unset key\n')
#        gnu.write('set size square\n')
        gnu.write('set label sprintf(\'Kd: %s.2f+/-%s.2f uM\',%f,%f) at graph 0.05,0.95\n' % ('%','%',kdmean,kdstdev))
        gnu.write('set xlabel \'Kd (uM)\'\n')
        gnu.write('set ylabel \'Probability\'\n')
        gnu.write('plot \'out/fastKd/grid.out\' i 0 u 1:3 w li,\'\' i 1 u 1:3 w li\n')
        gnu.write('unset label\n')
        gnu.close()
        os.system('gnuplot gnu/gnu.gp')
    

    return chi,chi2,kd,dwlimCurr




################################################################

#fit each residue with local kd (fast)
def localfastKdFits(residues):
    resy=residues.keys()
    chi2Full=0.0
    dofFull=0.0
    for res in resy:
        Ptot=numpy.array(residues[res].conc)
        Mtot=numpy.array(residues[res].Mtot)
        ydata=residues[res].dCSP
        x0=olyaKd(Mtot,Ptot,ydata)
        kd=numpy.fabs(x0[0])
        dwA=x0[1]
        PtotCalc=numpy.linspace(0.1,numpy.max(Ptot)*1.1,20)

        a = 14.3383    
        k = -0.00116997 
        b = 185.605    

        MtotCalc=a*numpy.exp(PtotCalc*k)+b


        ycalc=olyafunc(PtotCalc,kd,MtotCalc,dwA)
        chi=numpy.array(olyaKdMin((kd,dwA),(Mtot,Ptot,ydata)))
        chi2=numpy.sum(chi**2)

        dof=len(chi)-2

        chi2Full+=chi2
        dofFull+=dof

        aveErr=numpy.sqrt(chi2/dof)
        aveErrFrac=aveErr/numpy.max(ydata)
        residues[res].AddFastKdFit(kd,dwA,PtotCalc,ycalc,chi2,dof,aveErrFrac)
        residues[res].PrintFastKdFit()

        
        #os.system('arraygraph.py 4 6 0 0 0 0 `ls figs/fastKd/*.eps`')
    return chi2Full,dofFull


#do global analysis of parameters to get Kds (fast)
def globalfastKdFits(residues):
    resy=residues.keys()
    ydataGlob=[]
    xdataGlob=[]
    dwlim=numpy.zeros(len(resy))
    dwlimCurr=numpy.zeros(len(resy))
    cnt=0

    cspMax=0.
    for res in resy:#now lets run global fit
        Ptot=numpy.array(residues[res].conc)
        Mtot=numpy.array(residues[res].Mtot)
        ydata=residues[res].dCSP
        ydataGlob.append(ydata)
        for i in range(len(ydata)):
            if(ydata[i]>cspMax):
                cspMax=ydata[i]
        xdataGlob.append((Mtot,Ptot))
        dwlim[cnt]=copy.deepcopy(residues[res].dCSPlim)
        dwlimCurr[cnt]=copy.deepcopy(residues[res].dCSPlim)
        cnt+=1


    kdmin=0.1
    kdmax=500
    kdgrid=100
    chi,chi2,kd,dwlimCurr=GridSearchfastKdGlob(resy,kdmin,kdmax,kdgrid,xdataGlob,ydataGlob,dwlim,dwlimCurr,'out/fastKd/grid.out',verb='y',axis='lin')

    dof=len(chi)-len(resy)-1. #number of degrees of freedom
    sigma=chi2/dof
    aveErr=numpy.sqrt(sigma)
    print('FractionalError:',aveErr/cspMax)
    
    #setup and run minimised global fit
    dwTest=numpy.zeros(len(dwlim)+1)
    dwTest[0]=100.
    for i in range(len(dwTest)-1):
        dwTest[i+1]=dwlim[i]
    x0=leastsq(olyaKdGlob2,dwTest,args=[xdataGlob,ydataGlob,dwlimCurr])
    kdBest=x0[0][0]
    for i in range(len(dwlimCurr)):
        dwlimCurr[i]=x0[0][1+i]
    ycalcGlob=olyaKdcalc(kdBest,xdataGlob,ydataGlob,dwlimCurr)
    chi=getChiGlob(ycalcGlob,ydataGlob)
    chi2=getChi2(chi)
    dof=len(chi)-len(resy)-1. #number of degrees of freedom
    sigma=chi2/dof
    aveErr=numpy.sqrt(sigma)
    aveErrFrac=aveErr/cspMax
    print('FractionalError:',aveErrFrac)

    i=0
    for res in resy:#update parameters
        PtotCalc=numpy.linspace(0.1,numpy.max(xdataGlob[i][1])*1.1,20) 

        a = 14.3383    
        k = -0.00116997 
        b = 185.605    

        MtotCalc=a*numpy.exp(PtotCalc*k)+b

        ycalc=olyafunc(PtotCalc,kdBest,MtotCalc,dwlimCurr[i]) #recalculate with new numbers
        residues[res].SetCSPMax(cspMax)

        chiLoc=ydataGlob[i]-ycalcGlob[i]
        chi2Loc=getChi2(chiLoc)
        dofLoc=len(chiLoc)-2. #number of degrees of freedom
        sigmaLoc=chi2Loc/dofLoc
        aveErrLoc=numpy.sqrt(sigmaLoc)
        aveErrFracLoc=aveErrLoc/numpy.max(ydataGlob[i])

        residues[res].AddGlobFastKdFit(kdBest,dwlimCurr[i],PtotCalc,ycalc,chi2Loc,dofLoc,aveErrFracLoc)
        residues[res].PrintGlobFastKdFit()

        i+=1

    return chi2,dof


#############################

#compare local and global analyses
def FastExchangeFits(residues):
    chi2a,dofa=localfastKdFits(residues) #fit each residue for kds (fast exchange)
    chi2b,dofb=globalfastKdFits(residues) #fit global kd (fast exchange)
    print('chi2simple:  ',chi2b)
    print('dofsimple:   ',dofb)
    print( 'chi2complex: ',chi2a)
    print( 'dofcomplex:  ',dofa)
    print( 'Plvl going to complex:',GetPlvl(chi2b,chi2a,dofb,dofa))
    os.system('arraygraph.py 3 7 0 0 0 0 figs/gridfastKd.eps `ls figs/fastKd/*.eps`')
    os.system('mv summary.pdf pdf/fastKd.pdf')

################################################################



#note kd,Mtot,xdata in micromoles, dw in ppm
#do kd analysis in 1D out of fast exchange
def Fit1Dycalc(kd,kplus,xdataGlob,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2g,R2e):
    ycalcHGlob=[]
    ycalcNGlob=[]
    ycalcIGlob=[]

    parallel_flg='n'
    if(parallel_flg!='y'):
        for i in range(len(xdataGlob)):
            xdata=xdataGlob[i][1]
            Mtot=xdataGlob[i][0]
            hfreq,nfreq,maxspec=CalcSpec2Dproj(sfrq,dfrq,Mtot*1E-6,xdata*1E-6,kd*1E-6,kplus,R2e[i],R2g[i],ppm_to_rads(dwlimHCurr[i],sfrq),ppm_to_rads(dwlimNCurr[i],dfrq),'n')
            maxspec=maxspec/(numpy.max(maxspec))
            ycalcHGlob.append(hfreq)
            ycalcNGlob.append(nfreq)
            ycalcIGlob.append(maxspec)

    else:
        import pp,time
        ppservers=("*",) #initialise parallel calc
        job_server = pp.Server(ppservers=ppservers,restart=True)# Creates jobserver with automatically detected number of workers
        print(job_server.get_active_nodes())

        jobs = [(i,job_server.submit(CalcSpec2Dproj,(sfrq,dfrq,xdataGlob[i][0]*1E-6,xdataGlob[i][1]*1E-6,kd*1E-6,kplus,R2e[i],R2g[i],ppm_to_rads(dwlimHCurr[i],sfrq),ppm_to_rads(dwlimNCurr[i],dfrq),'n'),globals=globals())) for i in range(len(xdataGlob))]#loop over the number of specified network building attempts

        for i,job in jobs:
            hfreq,nfreq,maxspec=job()
            maxspec=maxspec/(numpy.max(maxspec))
            ycalcHGlob.append(hfreq)
            ycalcNGlob.append(nfreq)
            ycalcIGlob.append(maxspec)

        job_server.print_stats()
        time.sleep(2)
        job_server.destroy()


    return ycalcHGlob,ycalcNGlob,ycalcIGlob

#determine chi matrix        
def GetChiFit1D(ycalcHGlob,ycalcNGlob,ycalcIGlob,ydataHGlob,ydataNGlob,ydataIGlob,dHmax,dNmax):    
    chi=[]
    for i in range(len(ycalcHGlob)):
        chiI=ycalcIGlob[i]-ydataIGlob[i]
        chiH=(ycalcHGlob[i]-ydataHGlob[i])/dHmax
        chiN=(ycalcNGlob[i]-ydataNGlob[i])/dNmax
        for j in range(len(chiI)):
            chi.append(chiI[j])
            chi.append(chiH[j])
            chi.append(chiN[j])
    return chi


#for minimisation of Kd and Kplus
#using intensities (1D)
def MinFit1D(x,b):

    xdataGlob=b[0]
    ydataHGlob=b[1]
    ydataNGlob=b[2]
    ydataIGlob=b[3]
    dwlimHCurr=b[4]
    dwlimNCurr=b[5]
    sfrq=b[6]
    dfrq=b[7]
    R2g=b[8]
    R2e=b[9]
    dHmax=b[10]
    dNmax=b[11]

    cnt=0
    kd=x[cnt];cnt+=1
    kplus=x[cnt];cnt+=1
    for i in range(len(dwlimHCurr)):
        dwlimHCurr[i]=x[cnt];cnt+=1
    for i in range(len(dwlimHCurr)):
        dwlimNCurr[i]=x[cnt];cnt+=1


    print(kd,kplus)

    ycalcHGlob,ycalcNGlob,ycalcIGlob=Fit1Dycalc(kd,kplus,xdataGlob,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2g,R2e)
    return GetChiFit1D(ycalcHGlob,ycalcNGlob,ycalcIGlob,ydataHGlob,ydataNGlob,ydataIGlob,dHmax,dNmax)




#note kd,Mtot,xdata in micromoles, dw in ppm
#return 2D spectra as function of given Kd/kplus/deltaOmegas
def Fit2Dycalc(kd,kplus,xdataGlob,hfreqGlob,nfreqGlob,Mtot,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2g,R2e,window):
    ycalcGlob=[]
    for i in range(len(xdataGlob)):#for each residue...
        xdata=xdataGlob[i]
        freqH=hfreqGlob[i]#need to be in radians
        freqN=nfreqGlob[i]#need to be in radians

        hfreq,nfreq,maxspec,specAll=CalcSpec2D(sfrq,dfrq,freqH,freqN,Mtot*1E-6,xdata*1E-6,kd*1E-6,kplus,R2e[i],R2g[i],ppm_to_rads(dwlimHCurr[i],sfrq),ppm_to_rads(dwlimNCurr[i],dfrq),'n',name=i,window=window)

        #maxspec=maxspec/(numpy.max(maxspec))
        ycalcGlob.append(specAll)

    return ycalcGlob



###################################################################

#function for creating sparky project file
def MakeSparkyProj(listy):

    outy=open('Sparky/Projects/test.proj','w')
    outy.write('<sparky project file>\n')
    outy.write('<version 3.115>\n')
    outy.write('<savefiles>\n')
    for i in range(len(listy)):
        outy.write('../Save/%s.save\n' % (listy[i]))
    outy.write('<end savefiles>\n')
    outy.write('<options>\n')
    outy.write('<end options>\n')
    outy.write('<overlays>\n')
    outy.write('overlay simulated raw\n')
    outy.write('<end overlays>\n')
    outy.write('<attached data>\n')
    outy.write('<end attached data>\n')
    outy.write('<molecule>\n')
    outy.write('name \n')
    outy.write('<attached data>\n')
    outy.write('<end attached data>\n')
    outy.write('<condition>\n')
    outy.write('name \n')
    outy.write('<resonances>\n')
    outy.write('<end resonances>\n')
    outy.write('<end condition>\n')
    outy.write('<end molecule>\n')
    outy.close()




##################################################################

#note kd,Mtot,xdata in micromoles, dw in ppm
#Calculate entire 2D spectrum
def Sim2Dycalc(spectra,residues,kd,kplus,xdataGlob,Mtot,dwlimHCurr,dwlimNCurr,R2gCurr,R2eCurr):

    print('Calculating 2D spectrum...')


    specy=spectra.keys()
    resy=residues.keys()        

    specVals=[]
    specCalc=[]
    for spectrum in specy: #for each spectrum
        specVl=[]
        specCl=[]
        spectra[spectrum].resetSim() #reset spectrum
        freqH=spectra[spectrum].rads0 #get the spectral width and rads values
        freqN=spectra[spectrum].rads1 #get the spectral width and rads values

        Ptot=float(spectra[spectrum].concStr)

        for i in range(len(xdataGlob)): #for each residue
            dwH=ppm_to_rads(dwlimHCurr[i]+ float(residues[resy[i]].dH[residues[resy[i]].zeroVal]),spectra[spectrum].sfrq)
            dwN=ppm_to_rads(dwlimNCurr[i]+float(residues[resy[i]].dN[residues[resy[i]].zeroVal]),spectra[spectrum].dfrq)

            hfreqTmp,nfreqTmp,specMax,spec=RunNMRFull(spectra[spectrum].dfrq,spectra[spectrum].sfrq,Mtot*1E-6,Ptot*1E-6,kd*1E-6,kplus,R2eCurr[i],R2gCurr[i],dwN,dwH,freqN,freqH,'null','n',window=window) #calc contribution of peak to spectrum

            specVl.append(specMax)
            specCl.append(spec)
        specCalc.append(specCl)
        specVals.append(specVl)



    specCalc=numpy.array(specCalc)
    specVals=numpy.array(specVals)

    #normalise
    maxValPeak=numpy.max(numpy.fabs(specVals),axis=0)#get max intensity for each peak

    cnt=0
    for spectrum in specy: #for each spectrum
        spectra[spectrum].resetSim() #reset spectrum
        for i in range(len(xdataGlob)): #for each residue
            #print residues[resy[i]].sigFac,1/residues[resy[i]].sigFac
            spectra[spectrum].sim+=specCalc[cnt][i]/maxValPeak[i]/residues[resy[i]].sigFac
            spectra[spectrum].getchi()
        cnt+=1



    return






def print2DSpectra(spectra,residues):
    print('printing 2D spectrum:')
    listy=[]
    specy=spectra.keys()
    for spectrum in specy: #for each spectrum
        listy.append('diff.'+spectra[spectrum].concStr+'.ft2')
        listy.append('raw.'+spectra[spectrum].concStr+'.ft2')
        listy.append('test.'+spectra[spectrum].concStr+'.ft2')


    if(os.path.exists('Sparky')!=1):
        os.system('mkdir Sparky')
    if(os.path.exists('Sparky/Lists')!=1):
        os.system('mkdir Sparky/Lists')
    if(os.path.exists('Sparky/Projects')!=1):
        os.system('mkdir Sparky/Projects')
    if(os.path.exists('Sparky/Save')!=1):
        os.system('mkdir Sparky/Save')
    if(os.path.exists('Sparky/Python')!=1):
        os.system('cp -r ~/Sparky/Python ./Sparky')            
    for spectrum in specy: #for each spectrum
        spectra[spectrum].printSpec()
        spectra[spectrum].MakeSparkySaves(residues)

    MakeSparkyProj(listy)
    os.system("export SPARKYHOME=`pwd`/Sparky")



    return






       
def GetChiFit2D(ycalcGlob,ydataGlob):    
    chi=[]
    for i in range(len(ycalcGlob)):
        for j in range(len(ycalcGlob[i])):
            for k in range(len(ycalcGlob[i][j])):
                for l in range(len(ycalcGlob[i][j][k])):
                    chi.append(ycalcGlob[i][j][k][l]-ydataGlob[i][j][k][l])
    return chi


def GetChiFit2DFull(spectra):    
    specy=spectra.keys()
    chi=[]
    for spectrum in specy: #concatenate all the chi
        chi=numpy.concatenate((chi,spectra[spectrum].chi))
    return chi


def MinFit2D(x,b):

    xdataGlob=b[0]
    hfreqGlob=b[1]
    nfreqGlob=b[2]
    ydataGlob=b[3]
    Mtot=b[4]
    dwlimHCurr=b[5]
    dwlimNCurr=b[6]
    sfrq=b[7]
    dfrq=b[8]
    R2g=b[9]
    R2e=b[10]
    window=b[13]
    
    #unpack minimisation variables
    cnt=0
    kd=x[cnt];cnt+=1
    kplus=x[cnt];cnt+=1
    for i in range(len(dwlimNCurr)):
        dwlimNcurr[i]=x[cnt];cnt+=1
    for i in range(len(dwlimHCurr)):
        dwlimHCurr[i]=x[cnt];cnt+=1
    for i in range(len(dwlimNCurr)):
        R2g[i]=x[cnt];cnt+=1
        R2e[i]=x[cnt]

    print(kd,kplus,R2g)

    ycalcGlob=Fit2Dycalc(kd,kplus,xdataGlob,hfreqGlob,nfreqGlob,Mtot,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2g,R2g,window=window)
    return GetChiFit2D(ycalcGlob,ydataGlob)


def MinFit2Drelax(x,b):

    xdataGlob=b[0]
    hfreqGlob=b[1]
    nfreqGlob=b[2]
    ydataGlob=b[3]
    Mtot=b[4]
    dwlimHCurr=b[5]
    dwlimNCurr=b[6]
    sfrq=b[7]
    dfrq=b[8]
    R2g=b[9]
    R2e=b[10]
    kd=b[11]
    kplus=b[12]
    window=b[13]

    cnt=0
    for i in range(len(dwlimHCurr)):
        print('%.0f' % fabs(x[cnt]), end=' ')
        R2g[i]=fabs(x[cnt])
        R2e[i]=fabs(x[cnt])
        cnt+=1

    print
    ycalcGlob=Fit2Dycalc(kd,kplus,xdataGlob,hfreqGlob,nfreqGlob,Mtot,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2g,R2g,window)
    return GetChiFit2D(ycalcGlob,ydataGlob)




def MinFit2DFull(x,b):
    
    spectra=b[0]
    residues=b[1]
    xdataGlob=b[2]
    Mtot=b[3]
    dwlimHCurr=b[4]
    dwlimNCurr=b[5]
    R2gCurr=b[6]
    R2eCurr=b[7]
    window=b[8]

    #unpack the variables
    cnt=0
    kd=x[cnt];cnt+=1
    kplus=x[cnt];cnt+=1
    #for i in range(len(dwlimHCurr)):
    #    dwlimHCurr[i]=x[cnt];cnt+=1
    #for i in range(len(dwlimHCurr)):
    #    dwlimNCurr[i]=x[cnt];cnt+=1
    #for i in range(len(R2gCurr)):
    #    R2gCurr[i]=x[cnt];R2eCurr[i]=x[cnt];cnt+=1

    print(kd,kplus,R2gCurr[0])

    Sim2Dycalc(spectra,residues,kd,kplus,xdataGlob,Mtot,dwlimHCurr,dwlimNCurr,R2gCurr,R2eCurr)
    return GetChiFit2DFull(spectra)  






####################################################################


def ExchangeFits1D(residues,sfrq,dfrq,R2g,R2e):
    resy=residues.keys()
    #take globally fitted Kd value and optimise kplus
    kd=residues[resy[0]].kdFastGlob
    kplus=1E7

    #kd=90.4308823784 
    #kplus=49272.7799232*1000
    #43.4308823784 49272.7799232

    ydataHGlob=[]
    ydataNGlob=[]
    ydataIGlob=[]
    xdataGlob=[]

    dwlimHCurr=numpy.zeros(len(resy))
    dwlimNCurr=numpy.zeros(len(resy))

    R2gCurr=numpy.zeros(len(resy))
    R2eCurr=numpy.zeros(len(resy))

    cnt=0
    dHmax=0.0 #estimate largest dH for plotting
    dNmax=0.0 #estimate largest dN for plotting
    dHmin=0.0 #estimate largest dH for plotting
    dNmin=0.0 #estimate largest dN for plotting
    for res in resy:#now lets run global fit
        Ptot=numpy.array(residues[res].conc)
        Mtot=numpy.array(residues[res].Mtot)
        ydataH=residues[res].dHnorm
        ydataN=residues[res].dNnorm
        ydataI=residues[res].sig

        ydataHGlob.append(ydataH)
        ydataNGlob.append(ydataN)
        ydataIGlob.append(ydataI)
        xdataGlob.append((Mtot,Ptot))

        #estimate the limiting dw values for this given kd
        x0=leastsq(olyaDw,[numpy.max(ydataH),],args=[kd,Ptot,ydataH,Mtot])
        dwlimHCurr[cnt]=x0[0]
        x0=leastsq(olyaDw,[numpy.max(ydataN),],args=[kd,Ptot,ydataN,Mtot])
        dwlimNCurr[cnt]=x0[0]

        R2gCurr[cnt]=R2g
        R2eCurr[cnt]=R2e

        for i in range(len(ydataH)):
            if(ydataH[i]>dHmax):
                dHmax=ydataH[i]
        for i in range(len(ydataN)):
            if(ydataN[i]>dNmax):
                dNmax=ydataN[i]
        for i in range(len(ydataH)):
            if(ydataH[i]<dHmin):
                dHmin=ydataH[i]
        for i in range(len(ydataN)):
            if(ydataN[i]<dNmin):
                dNmin=ydataN[i]
        cnt+=1

    #calculating data for given kd,kplus and dw values

    minny='y'
    if(minny=='y'):

        params=numpy.zeros(len(dwlimNCurr)*2+2)
        #pack up variables
        cnt=0
        params[cnt]=kd;cnt+=1;
        params[cnt]=kplus;cnt+=1;
        for i in range(len(dwlimNCurr)):
            params[cnt]=dwlimHCurr[i];cnt+=1
        for i in range(len(dwlimNCurr)):
            params[cnt]=dwlimNCurr[i];cnt+=1

        x0=leastsq(MinFit1D,params,args=[xdataGlob,ydataHGlob,ydataNGlob,ydataIGlob,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2gCurr,R2eCurr,dHmax,dNmax])

        x=x0[0]

        #unpack minimisation variables
        cnt=0
        kd=x[cnt];cnt+=1
        kplus=x[cnt];cnt+=1
        for i in range(len(dwlimHCurr)):
            dwlimHCurr[i]=x[cnt];cnt+=1
        for i in range(len(dwlimHCurr)):
            dwlimNCurr[i]=x[cnt];cnt+=1


    #recalculate the best fitted values
    ycalcHGlob,ycalcNGlob,ycalcIGlob=Fit1Dycalc(kd,kplus,xdataGlob,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2gCurr,R2eCurr)
    chi=GetChiFit1D(ycalcHGlob,ycalcNGlob,ycalcIGlob,ydataHGlob,ydataNGlob,ydataIGlob,dHmax,dNmax)
    chi2=getChi2(chi)
    print('kinetic chi2:',chi2)
    dof=len(chi)-len(resy)*2-2
    sigma=chi2/dof
    aveErr=numpy.sqrt(sigma)

    #simulate data with these parameters for printing
    xdataCalc=[]
    for i in range(len(xdataGlob)):
        PtotCalc=numpy.linspace(0.1,numpy.max(xdataGlob[i][1])*1.1,20) 

        a = 14.3383    
        k = -0.00116997 
        b = 185.605    
        MtotCalc=a*numpy.exp(PtotCalc*k)+b

        xdataCalc.append((MtotCalc,PtotCalc))
    ycalcHGlob,ycalcNGlob,ycalcIGlob=Fit1Dycalc(kd,kplus,xdataCalc,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2gCurr,R2eCurr)

    #write outputs and update struct
    cnt=0
    for res in resy:
        residues[res].AddGlobFit1D(kd,kplus,dwlimHCurr[cnt],dwlimNCurr[cnt],ycalcHGlob[cnt],ycalcNGlob[cnt],ycalcIGlob[cnt],xdataCalc[cnt][1],chi2,dof,aveErr)
        residues[res].SetdHMax(dHmin,dHmax)
        residues[res].SetdNMax(dNmin,dNmax)
        residues[res].PrintFit1D()
        cnt+=1

    os.system('arraygraph.py 3 7 0 0 0 0 `ls figs/fit1D/*.eps`')
    os.system('mv summary.pdf pdf/fit1D.pdf')


def AddIfNewRev(array,test):
    tick=0
    for i in range(len(array)):
        if(array[i]==(test[0],test[1])):
           tick=1
        if(array[i]==(test[1],test[0])):
           tick=1
    if(tick==0):
        array.append(test)

def AddIfNew(array,test):
    tick=0
    for i in range(len(array)):
        if(array[i]==test):
            tick=1
    if(tick==0):
        array.append(test)
    

def IntersectLists(join):
    go=0
    while(go==0):
        tick=0
        for i in range(len(join)):
            for j in range(len(join[i])):
                test=join[i][j]
                for k in range(len(join)):
                    for l in range(len(join[k])):
                        tast=join[k][l]
                        if(test==tast and k>i and tick==0):
                            #print 'join',i,'and',k
                            list=[]
                            for m in range(len(join[i])):
                                AddIfNew(list,join[i][m])
                            for m in range(len(join[k])):
                                AddIfNew(list,join[k][m])
                            tick=1
                            joinnew=copy.deepcopy(join)
                            joinnew.pop(i)
                            joinnew.pop(k-1)
                            joinnew.append(list)
                            break

        join=copy.deepcopy(joinnew)
        if(tick==0):
            go=1
    return join

def GetRegions(resy,residues,dHrad,dNrad):    
    regions=[]
    for res in resy:
        list=[]
        list.append(res)
        #check to make sure res is not already in the regions list
        tick=0
        for i in range(len(regions)):
            for j in range(len(regions[i])):
                if(regions[i][j]==res):
                    tick=1
        if(tick==1):
            pass
            #print 'Already in a region',res
        else:
            Have=residues[res].Have
            Hsig=residues[res].Hsig
            Nave=residues[res].Nave
            Nsig=residues[res].Nsig
            for ras in resy:
                HaveT=residues[ras].Have
                HsigT=residues[ras].Hsig
                NaveT=residues[ras].Nave
                NsigT=residues[ras].Nsig
                if(ras!=res):
                    if((Have-HaveT)**2./(dHrad+Hsig+HsigT)**2.+(Nave-NaveT)**2./(dNrad+Nsig+NsigT)**2.<1.0):
                        #overlap!
                        list.append(ras)
            regions.append(list)


    regions=IntersectLists(regions)
    

    cnt=0
    for i in range(len(regions)):
        #print regions[i][0],':',
        cnt+=len(regions[i])
        #for j in range(len(regions[i])):
        #    print regions[i][j],
        #print
    print('Number of peaks:',len(resy))
    print('Number of regions:',len(regions))
    if(cnt!=len(resy)):
        print('Problem! Seem to have lost of gained a peak or two')
        sys.exit(100)
    return regions


################################################################

#take the extracted regions and estimate deltaOmega from fast exchange fit
def EstimateDeltaOmega(residues,Mtot,kd):
    resy=residues.keys()
    dwlimHCurr=numpy.zeros(len(resy)) #array of deltaOmegaH (one per residue)
    dwlimNCurr=numpy.zeros(len(resy)) #array of deltaOmegaN (one per residue)
    cnt=0
    for res in resy:
        dwlimHCurr[cnt]=0.005
        dwlimNCurr[cnt]=0.2
        #estimate deltaOmegas from 1D data in fast exchange limit
        Ptot=residues[res].conc
        ydataH=residues[res].dHnorm
        ydataN=residues[res].dNnorm
        #estimate the limiting dw values for this given kd
        x0=leastsq(olyaDw,[numpy.max(ydataH),],args=[kd,Ptot,ydataH,Mtot])
        dwlimHCurr[cnt]=x0[0]
        x0=leastsq(olyaDw,[numpy.max(ydataN),],args=[kd,Ptot,ydataN,Mtot])
        dwlimNCurr[cnt]=x0[0]
        cnt+=1
    return dwlimHCurr,dwlimNCurr

def ExtractRegions(residues,spectra,dHrad,dNrad):
    resy=residues.keys()
    specy=spectra.keys()
    print('Extracting peaks...')
    for res in resy:#read in raw data and extract regions. 
        residues[res].getRadii(spectra,dHrad,dNrad)
    print('Finished extraction.')
    ydataGlob=[]
    xdataGlob=[]
    hfreqGlob=[]
    nfreqGlob=[]
    cnt=0

    

    for res in resy:
        Ptot=residues[res].conc
        ydata=residues[res].yrads #the ydata rectangle
        hdata=residues[res].hrads #the proton shifts rectangle
        ndata=residues[res].nrads #the nitrogen shifts rectangle

        #get frequencies and define them to sit on the 0 conc peak
        hfreq=ppm_to_rads(residues[res].hradsfrq - float(residues[res].dH[residues[res].zeroVal]),spectra[specy[0]].sfrq) #set peakpos to zero
        nfreq=ppm_to_rads(residues[res].nradsfrq - float(residues[res].dN[residues[res].zeroVal]),spectra[specy[0]].dfrq) #set peakpos to zero
        hfreqGlob.append(hfreq) #one entry per spectrum
        nfreqGlob.append(nfreq)
        ydataGlob.append(ydata)
        xdataGlob.append(Ptot)
    return hfreqGlob,nfreqGlob,ydataGlob,xdataGlob




def EstimateR2(residues,R2g,R2e):
    resy=residues.keys()
    R2gCurr=numpy.zeros(len(resy))    #array of R2gs (one per residue)
    R2eCurr=numpy.zeros(len(resy))    #array of R2es (one per residue)
    resy=residues.keys()
    cnt=0
    for res in resy:
        R2gCurr[cnt]=R2g
        R2eCurr[cnt]=R2e
        cnt+=1
    return R2gCurr,R2eCurr



#extract regions and optimise relaxation rates for given dOmegas/kd/kplus
def ExchangeFits2Drelax(residues,spectra,Mtot,sfrq,dfrq,R2gInit,R2eInit,window):
    resy=residues.keys()
    #take globally fitted Kd value and optimise kplus (from fast Ex)
    #kd=residues[resy[0]].kdFastGlob

    R2g=numpy.zeros(len(resy))  #setup relaxtion matrix
    R2e=numpy.zeros(len(resy))  #setup relaxation matrix
    params=numpy.zeros(len(resy)) #setup parameters matrix

    dHrad=0.1 #radius of extraction
    dNrad=1.0 #radius of extraction

    regions=GetRegions(resy,residues,dHrad,dNrad)
    hfreqGlob,nfreqGlob,ydataGlob,xdataGlob=ExtractRegions(residues,spectra,dHrad,dNrad)

    read_flg='y'
    if(read_flg=='y'):
        kd,kplus,dwlimNCurr,dwlimHCurr=GetFit1Dparams(residues)
    else:
        kd=70
        kplus=3.14E6
        dwlimHCurr,dwlimNCurr=EstimateDeltaOmega(residues,Mtot,kd)

    #pack up variables
    cnt=0
    for i in range(len(dwlimNCurr)):
        params[cnt]=R2gInit;
        cnt+=1

    x0=leastsq(MinFit2Drelax,params,args=[xdataGlob,hfreqGlob,nfreqGlob,ydataGlob,Mtot,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2g,R2e,kd,kplus,window])
    x=x0[0]
    #unpack minimisation variables
    cnt=0
    for i in range(len(dwlimHCurr)):
        R2g[i]=fabs(x[i])
        R2e[i]=fabs(x[i])
        cnt+=1
    ycalcGlob=Fit2Dycalc(kd,kplus,xdataGlob,hfreqGlob,nfreqGlob,Mtot,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2g,R2g,window)
    chi=GetChiFit2D(ycalcGlob,ydataGlob)  
    chi2=getChi2(chi)

    dof=len(chi)-len(resy)*2-2
    sigma=chi2/dof
    aveErr=numpy.sqrt(sigma)
    print('2D chi2:',chi2)
    print('2D aveE:',aveErr)

    cnt=0
    for res in resy:
        residues[res].AddGlobFit2D(kd,kplus,dwlimHCurr[cnt],dwlimNCurr[cnt],ycalcGlob[cnt],xdataGlob[cnt],R2g[cnt],R2e[cnt],chi2,dof,aveErr)
        residues[res].printFit2D()
        cnt+=1

    os.system('arraygraph.py 3 7 0 0 0 0 `ls figs/fit2D/*.eps`')
    os.system('mv summary.pdf pdf/fit2D.pdf')



#extract regions and optimise relaxation rates for given dOmegas/kd/kplus
def ExchangeFits2D(residues,spectra,Mtot,sfrq,dfrq,R2gInit,R2eInit,window):
    resy=residues.keys()
    #take globally fitted Kd value and optimise kplus (from fast Ex)
    #kd=residues[resy[0]].kdFastGlob

    R2g=numpy.zeros(len(resy))  #setup relaxtion matrix
    R2e=numpy.zeros(len(resy))  #setup relaxation matrix
    params=numpy.zeros(len(resy)) #setup parameters matrix

    dHrad=0.1 #radius of extraction
    dNrad=1.0 #radius of extraction

    regions=GetRegions(resy,residues,dHrad,dNrad)
    hfreqGlob,nfreqGlob,ydataGlob,xdataGlob=ExtractRegions(residues,spectra,dHrad,dNrad)

    read_flg='y'
    if(read_flg=='y'):
        kd,kplus,dwlimNCurr,dwlimHCurr=GetFit1Dparams(residues)
    else:
        kd=70
        kplus=3.14E6
        dwlimHCurr,dwlimNCurr=EstimateDeltaOmega(residues,Mtot,kd)

    #pack up variables
    params=numpy.zeros(len(resy)*3) #setup parameters matrix

    cnt=0
    params[cnt]=kd;cnt+=1
    params[cnt]=kplus;cnt+=1;
    for i in range(len(dwlimNCurr)):
        params[cnt]=dwlimNcurr[i];cnt+=1;
    for i in range(len(dwlimHCurr)):
        params[cnt]=dwlimHCurr[i];cnt+=1;
    for i in range(len(dwlimNCurr)):
        params[cnt]=R2g[i];cnt+=1

    x0=leastsq(MinFit2D,params,args=[xdataGlob,hfreqGlob,nfreqGlob,ydataGlob,Mtot,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2g,R2e,kd,kplus,window])
    x=x0[0]
    #unpack minimisation variables
    cnt=0
    kd=x[cnt];cnt+=1
    kplus=x[cnt];cnt+=1;
    for i in range(len(dwlimNCurr)):
        dwlimNcurr[i]=x[cnt];cnt+=1;
    for i in range(len(dwlimHCurr)):
        dwlimHCurr[i]=x[cnt];cnt+=1;
    for i in range(len(dwlimNCurr)):
        R2g[i]=x[cnt];cnt+=1
        R2e[i]=x[cnt]


    ycalcGlob=Fit2Dycalc(kd,kplus,xdataGlob,hfreqGlob,nfreqGlob,Mtot,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2g,R2g,window)
    chi=GetChiFit2D(ycalcGlob,ydataGlob)  
    chi2=getChi2(chi)

    dof=len(chi)-len(resy)*2-2
    sigma=chi2/dof
    aveErr=numpy.sqrt(sigma)
    print('2D chi2:',chi2)
    print('2D aveE:',aveErr)


    #write outputs and update struct
    #simulate data with these parameters for printing
    xdataCalc=[]
    for i in range(len(xdataGlob)):
        PtotCalc=numpy.linspace(0.1,numpy.max(xdataGlob[i])*1.1,20) 
        xdataCalc.append(PtotCalc)
    ycalcHGlob,ycalcNGlob,ycalcIGlob=Fit1Dycalc(kd,kplus,xdataCalc,Mtot,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2g,R2e)
    dHmax=0.0 #estimate largest dH for plotting
    dNmax=0.0 #estimate largest dN for plotting
    dHmin=0.0 #estimate largest dH for plotting
    dNmin=0.0 #estimate largest dN for plotting
    for res in resy:#now lets run global fit
        ydataH=residues[res].dHnorm
        ydataN=residues[res].dNnorm
        for i in range(len(ydataH)):
            if(ydataH[i]>dHmax):
                dHmax=ydataH[i]
        for i in range(len(ydataN)):
            if(ydataN[i]>dNmax):
                dNmax=ydataN[i]
        for i in range(len(ydataH)):
            if(ydataH[i]<dHmin):
                dHmin=ydataH[i]
        for i in range(len(ydataN)):
            if(ydataN[i]<dNmin):
                dNmin=ydataN[i]
    cnt=0
    for res in resy:
        residues[res].AddGlobFit1D(kd,kplus,dwlimHCurr[cnt],dwlimNCurr[cnt],ycalcHGlob[cnt],ycalcNGlob[cnt],ycalcIGlob[cnt],xdataCalc[cnt],chi2,dof,aveErr)
        residues[res].SetdHMax(dHmin,dHmax)
        residues[res].SetdNMax(dNmin,dNmax)
        residues[res].PrintFit1D()
        cnt+=1


    #write outputs and update struct for 2D
    cnt=0
    for res in resy:
        residues[res].AddGlobFit2D(kd,kplus,dwlimHCurr[cnt],dwlimNCurr[cnt],ycalcGlob[cnt],xdataGlob[cnt],R2g[cnt],R2e[cnt],chi2,dof,aveErr)
        residues[res].printFit2D()
        cnt+=1

    os.system('arraygraph.py 3 7 0 0 0 0 `ls figs/fit2D/*.eps`')
    os.system('mv summary.pdf pdf/fit2D.pdf')



def ParseParam(res,tag,par):
    infile=open('out/'+tag+'/'+res+'.out','r')
    for line in infile.readlines():
        if(line[0]=='#'):
            test=line.split(':')
            if(len(test)>1):
                if(test[0]==par):
                    return float(test[1])
    print('cannot find:', par)
    sys.exit(100)

def GetFit1Dparams(residues):
    resy=residues.keys()
    dwlimHCurr=numpy.zeros(len(resy)) #array of deltaOmegaH (one per residue)
    dwlimNCurr=numpy.zeros(len(resy)) #array of deltaOmegaN (one per residue)
    for i in range(len(resy)):
        kd=ParseParam(resy[i],'fit1D',"# Global Kd")
        kplus=ParseParam(resy[i],'fit1D',"# Global kplus")
        dwlimHCurr[i]=ParseParam(resy[i],'fit1D',"# dwH")
        dwlimNCurr[i]=ParseParam(resy[i],'fit1D',"# dwN")
    return kd,kplus,dwlimNCurr,dwlimHCurr


def GetFit2Dparams(residues):
    resy=residues.keys()
    dwlimHCurr=numpy.zeros(len(resy)) #array of deltaOmegaH (one per residue)
    dwlimNCurr=numpy.zeros(len(resy)) #array of deltaOmegaN (one per residue)
    R2g=numpy.zeros(len(resy)) #array of deltaOmegaN (one per residue)
    R2e=numpy.zeros(len(resy)) #array of deltaOmegaN (one per residue)
    for i in range(len(resy)):
        kd=ParseParam(resy[i],'rads',"# Global kd")
        kplus=ParseParam(resy[i],'rads',"# Global kplus")
        dwlimHCurr[i]=ParseParam(resy[i],'rads',"# dwH")
        dwlimNCurr[i]=ParseParam(resy[i],'rads',"# dwN")
        R2g[i]=ParseParam(resy[i],'rads',"# R2g")
        R2e[i]=ParseParam(resy[i],'rads',"# R2e")
    return kd,kplus,dwlimNCurr,dwlimHCurr,R2g,R2e


#stitch together the reduced fits in the context of the complete spectrum
def ExchangeFits2DFull(residues,spectra,Mtot,R2g,R2e,window):
    print ('Calculating 2D spectrum...')
    specy=spectra.keys()
    resy=residues.keys()        

    
    newconc=[]
    for i in range(len(specy)):
        newconc.append(float(specy[i]))
    newconc=numpy.array(sorted(newconc))
    for res in resy:
        residues[res].adjustConcs(newconc)

    #MIGHT NEED TO ADD SOMETHING TO MAKE SURE WE HAVE ALL CONCENTRATIONS COVERED
    dHrad=0.1 #radius of extraction (ppm)
    dNrad=1. #radius of extraction (ppm)
    hfreqGlob,nfreqGlob,ydataGlob,xdataGlob=ExtractRegions(residues,spectra,dHrad,dNrad)

    read_flg='y'
    if(read_flg=='y'):
        kd,kplus,dwlimNCurr,dwlimHCurr,R2gCurr,R2eCurr=GetFit2Dparams(residues)
        R2eCurr=R2eCurr*100.

    else:
        kd=70
        kplus=3.1E6
        dwlimHCurr,dwlimNCurr=EstimateDeltaOmega(residues,Mtot,kd)
        R2gCurr,R2eCurr      =EstimateR2(residues,R2g,R2e)

    #this calculates the 2D circles.
    sfrq=600.
    dfrq=60.

    ycalcGlob=Fit2Dycalc(kd,kplus,xdataGlob,hfreqGlob,nfreqGlob,Mtot,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2gCurr,R2eCurr,window)

    #need function to fit the 2D regions back to get a full 2D
    """
    specVals=[]
    specCalc=[]
    for spectrum in specy: #for each spectrum
        specVl=[]
        specCl=[]
        freqH=spectra[spectrum].rads0 #get the spectral width and rads values
        freqN=spectra[spectrum].rads1 #get the spectral width and rads values

        Ptot=float(spectra[spectrum].concStr)

        for i in range(len(xdataGlob)): #for each residue
            dwH=ppm_to_rads(dwlimHCurr[i]+ float(residues[resy[i]].dH[residues[resy[i]].zeroVal]),spectra[spectrum].sfrq)
            dwN=ppm_to_rads(dwlimNCurr[i]+float(residues[resy[i]].dN[residues[resy[i]].zeroVal]),spectra[spectrum].dfrq)

            hfreqTmp,nfreqTmp,specMax,spec=RunNMRFull(spectra[spectrum].dfrq,spectra[spectrum].sfrq,Mtot*1E-6,Ptot*1E-6,kd*1E-6,kplus,R2eCurr[i],R2gCurr[i],dwN,dwH,freqN,freqH,'null','n') #calc contribution of peak to spectrum

            specVl.append(specMax)
            specCl.append(spec)
        specCalc.append(specCl)
        specVals.append(specVl)
    specCalc=numpy.array(specCalc)
    specVals=numpy.array(specVals)
    #normalise
    maxValPeak=numpy.max(numpy.fabs(specVals),axis=0)#get max intensity for each peak
    """

    cnt=0
    for spectrum in specy: #for each spectrum
        spectrum="%.0f" % newconc[cnt] #adjust for shifted concentrations
        spectra[spectrum].resetSim() #reset spectrum
        for i in range(len(ycalcGlob)):#for each residue
            #print len(residues[resy[i]].zidsave[cnt][0])
            #print residues[resy[i]].sigFac,1/residues[resy[i]].sigFac
            #print residues[resy[i]].zidsave[cnt]
            #print             spectra[spectrum].sim[residues[resy[i]].zidsave[cnt]].shape
            #print 'ycalcglob:',ycalcGlob[cnt][i].flatten().shape
            #print spectra[spectrum].sim[residues[resy[i]].zidsave[cnt]].shape
            #print spectra[spectrum].sim[residues[resy[i]].zidsave[cnt]]=ycalcGlob[cnt][i]
            #print 'one:   ',spectra[spectrum].sim[residues[resy[i]].zidsave[cnt]].shape
            #print 'two:   ',numpy.array((ycalcGlob[i][cnt])).shape
            #print 'three: ',len(residues[resy[i]].zidsave[cnt][0])

            #if(resy[i]=="101H-N"):
            spectra[spectrum].sim[residues[resy[i]].zidsave[cnt]]+=((ycalcGlob[i][cnt].transpose()).flatten())/residues[resy[i]].sigFac



            #xvals=residues[resy[i]].hrads #print the file from hrads
            #yvals=residues[resy[i]].nrads
            #outy=open('fart.'+str(i)+'.'+spectrum+'.out','w')
            #outy=open('tattg.out','w')#access the fit. IS GOOD            
            #for k in range(xvals[0]):
            #    outy.write('%f\t%f\t%f\n' % (xvals[0][k],yvals[0][k],((ycalcGlob[i][cnt].transpose()).flatten())[k]/residues[resy[i]].sigFac))
            #outy.close()

            
            #outy=open('fart.'+resy[i]+'.'+spectrum+'.out','w')
            #for ii in range(len(residues[resy[i]].hradsfrq[cnt])):
            #    for jj in range(len(residues[resy[i]].nradsfrq[cnt])):
            #        vi=jj+ii*len(residues[resy[i]].nradsfrq[cnt])
            #        outy.write('%f\t%f\t%e\t%e\n' % (residues[resy[i]].hradsfrq[cnt][ii],residues[resy[i]].nradsfrq[cnt][jj],((ycalcGlob[i][cnt])[jj][ii]),residues[resy[i]].yrads[cnt][jj][ii]))
            #    outy.write('\n')
            #outy.close()
            #sys.exit(100)

        cnt+=1

        #outy=open('test.'+spectrum+'.out','w')
        #for j in range(len(spectra[spectrum].sim)):
        #    for k in range(len(spectra[spectrum].sim[j])):
        #        outy.write('%f\t%f\t%f\t%f\n' % (spectra[spectrum].ppm0[j],spectra[spectrum].ppm1[k],spectra[spectrum].sim[j,k],spectra[spectrum].data[j,k]))
        #    outy.write('\n')
        #outy.close()

    print2DSpectra(spectra,residues) #save ft2s 


    chi=GetChiFit2D(ycalcGlob,ydataGlob)  
    chi2=getChi2(chi)

    dof=len(chi)-len(resy)*2-2
    sigma=chi2/dof
    aveErr=numpy.sqrt(sigma)
    print ('2D chi2:',chi2)
    print ('2D aveE:',aveErr)






    #write outputs and update struct
    cnt=0
    for res in resy:
        residues[res].AddGlobFit2D(kd,kplus,dwlimHCurr[cnt],dwlimNCurr[cnt],ycalcGlob[cnt],xdataGlob[cnt],R2gCurr[cnt],R2eCurr[cnt],chi2,dof,aveErr)
        residues[res].printFit2D()
        cnt+=1



    for res in resy:
        residues[res].resetConcs()
    #write outputs and update struct
    #simulate data with these parameters for printing
    xdataCalc=[]
    for i in range(len(xdataGlob)):
        PtotCalc=numpy.linspace(0.1,numpy.max(xdataGlob[i])*1.1,20) 
        xdataCalc.append(PtotCalc)
    ycalcHGlob,ycalcNGlob,ycalcIGlob=Fit1Dycalc(kd,kplus,xdataCalc,Mtot,dwlimHCurr,dwlimNCurr,sfrq,dfrq,R2gCurr,R2eCurr)
    dHmax=0.0 #estimate largest dH for plotting
    dNmax=0.0 #estimate largest dN for plotting
    dHmin=0.0 #estimate largest dH for plotting
    dNmin=0.0 #estimate largest dN for plotting
    for res in resy:#now lets run global fit
        ydataH=residues[res].dHnorm
        ydataN=residues[res].dNnorm
        for i in range(len(ydataH)):
            if(ydataH[i]>dHmax):
                dHmax=ydataH[i]
        for i in range(len(ydataN)):
            if(ydataN[i]>dNmax):
                dNmax=ydataN[i]
        for i in range(len(ydataH)):
            if(ydataH[i]<dHmin):
                dHmin=ydataH[i]
        for i in range(len(ydataN)):
            if(ydataN[i]<dNmin):
                dNmin=ydataN[i]
    cnt=0
    for res in resy:
        residues[res].AddGlobFit1D(kd,kplus,dwlimHCurr[cnt],dwlimNCurr[cnt],ycalcHGlob[cnt],ycalcNGlob[cnt],ycalcIGlob[cnt],xdataCalc[cnt],chi2,dof,aveErr)
        residues[res].SetdHMax(dHmin,dHmax)
        residues[res].SetdNMax(dNmin,dNmax)
        residues[res].PrintFit1D()
        cnt+=1







    #sys.exit(100)



    #pack up variables
    #paramNo=len(resy)*3+2

    #params=numpy.zeros(paramNo)
    #cnt=0
    #params[cnt]=kd;cnt+=1;
    #params[cnt]=kplus;cnt+=1;
    #for i in range(len(dwlimNCurr)):
    #    params[cnt]=dwlimHCurr[i];cnt+=1
    #for i in range(len(dwlimNCurr)):
    #    params[cnt]=dwlimNCurr[i];cnt+=1
    #for i in range(len(R2gCurr)):
    #    params[cnt]=R2gCurr[i];cnt+=1;

    #x0=leastsq(MinFit2DFull,params,args=[spectra,residues,xdataGlob,Mtot,dwlimHCurr,dwlimNCurr,R2gCurr,R2eCurr,window])

    #x=x0[0]

    #unpack minimisation variables
    #cnt=0
    #kd=x[cnt];cnt+=1
    #kplus=x[cnt];cnt+=1
    #R2g=x[cnt];cnt+=1
    #for i in range(len(dwlimHCurr)):
    #    dwlimHCurr[i]=x[cnt];cnt+=1
    #for i in range(len(dwlimHCurr)):
    #    dwlimNCurr[i]=x[cnt];cnt+=1

    #sys.exit(100)
    #Sim2Dycalc(spectra,residues,kd,kplus,xdataGlob,Mtot,dwlimHCurr,dwlimNCurr,R2g,R2e)
    #chi=GetChiFit2DFull(spectra)  
    #chi2=getChi2(chi)

    #print2DSpectra(spectra,residues)

    #dof=len(chi)-len(resy)*2*2-2
    #sigma=chi2/dof
    #aveErr=numpy.sqrt(sigma)
    #print '2D chi2:',chi2
    #print '2D aveE:',aveErr


    #sys.exit(100)
    #cnt=0
    #for res in resy:
    #    residues[res].AddGlobFit2D(kd,kplus,dwlimHCurr[cnt],dwlimNCurr[cnt],ycalcGlob[cnt],xdataGlob[cnt],chi2,dof,aveErr)
    #    residues[res].printFit2D()
    #    cnt+=1



    #os.system('arraygraph.py 3 7 0 0 0 0 `ls figs/fit2D/*.eps`')
    #os.system('mv summary.pdf pdf/fit2D.pdf')
    


def PrettyPlots(residues,spectra):

    print ('Making pretty plots. Can be slow...')
    specy=spectra.keys()
    resy=residues.keys()        

    for res in resy:
        print(res)
        attstr=''
        attstr+=' figs/fastkd/'+res+'.eps'
        attstr+=' figs/fit1D/'+res+'.a.eps'
        attstr+=' figs/fit1D/'+res+'.b.eps'
        attstr+=' figs/fit1D/'+res+'.c.eps'

        for i in range(len(specy)):
            attstr+=' figs/fit2D/'+res+'.'+str(i)+'.cop.eps'
        os.system('arraygraph.py 4 8 0 0 0 0 '+attstr)
        os.system('mv summary.pdf pdf/'+res+'.pdf')
    return



################################################################

if __name__=="__main__":

    print('Running exchange sim in standalone mode')
    Mtot=200E-6

    Ptot=numpy.linspace(10E-9,3000E-6,20)

    kd=100E-6
    kplus=1E8

    R2e=10.
    R2g=10.
    dwH=50.
    dwN=50.

    #need to array these such that there is one per concentration
    freqH=numpy.linspace(-100.,100,201)
    freqN=numpy.linspace(-100.,100,201)

    verb='n' #if set to yes, will dump files and make plotting scripts

    #read in data, raw and peak lists
    #get appropriate frequency ranges
    #get initial estimates for dwH and dwN, R2e and R2g
    #optimise to get kd in 1D mode
    #run in 2D mode

    sfrq=600.   #Lamor frequency of nuclei one
    dfrq=600.    #Lamor frequency


    #1D version about 8 times faster than 2D.

    hfreq,nfreq,maxspec,specAll=CalcSpec2D(sfrq,dfrq,freqH,freqN,Mtot,Ptot,kd,kplus,R2e,R2g,dwH,dwN,verb,window)
    hfreq,nfreq,maxspec=CalcSpec2Dproj(sfrq,dfrq,Mtot,Ptot,kd,kplus,R2e,R2g,dwH,dwN,verb)


