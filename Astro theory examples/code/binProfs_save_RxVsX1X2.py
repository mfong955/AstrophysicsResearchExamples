#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:33:15 2021

@author: MattFong
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 13:44:21 2021

@author: MattFong
"""

import os, itertools
import time
import numpy as np
import matplotlib.pyplot as plt
import biasProfile as bp
import biasTools_z as bt
from colossus.halo import mass_so
from colossus.utils import constants
from colossus.halo import profile_dk14
from colossus.halo import concentration


from colossus.cosmology import cosmology as ccosmo
my_cosmo={'flat': True, 'H0': 70.4, 'Om0': 0.268, 'Ob0': 0.044, 'sigma8': 0.83, 'ns': 0.968}
cosmo=ccosmo.setCosmology('my_cosmo', my_cosmo)


def make2d(array,nParBin=None):
    if nParBin is None:
        nParBin=nbin-1
    #makes arrays 2d for GPR
    array2d=array.reshape(nParBin, nParBin)
    array2d=array2d.T
    return array2d

tictot=time.time()
##### set up directory names for local

#load directory 
simDir='/Users/MattFong/Desktop/Projects/Data/halo_prof/'

saveDir=simDir+'RxVsX1X2/'

if not os.path.exists(saveDir):
    os.makedirs(saveDir)

print('simDir=%s' %simDir)
print('saveDir=%s' %saveDir)



#[snapshot, redshift]
# redshiftBins=redshiftBins=np.genfromtxt(simDir+'redshiftBins.txt')
# redshiftBins=redshiftBins[-4:]
redshiftBins=np.array([[90.      ,  0.289063],
                       [91.      ,  0.25344 ],
                       [92.      ,  0.218258],
                       [99.      ,  0.      ]])


snap_z_info=np.array([[6.00000000e+01, 2.02254605e+00, 2.00000000e+00],
                       [7.40000000e+01, 1.02790380e+00, 1.00000000e+00],
                       [8.40000000e+01, 5.27653813e-01, 5.10000000e-01],
                       [9.10000000e+01, 2.53440499e-01, 2.50000000e-01],
                       [9.90000000e+01, 1.19209290e-07, 0.00000000e+00]])

# includeSubhalos=True
# includeSubhalos=False
includeSubhaloss=[True,False]

for includeSubhalos in includeSubhaloss: # completed running for [3]
    if includeSubhalos:
        subhaloStr='_includeSubhalos'
    if includeSubhalos is False:
        subhaloStr=''
    
    #only look at snapshot 91, z=0.253 and 99, z=0
    for redbini in [1]: # completed running for [3]
        snapshot=redshiftBins[:,0][redbini]
        z=redshiftBins[:,1][redbini]
        
        print('---------- running code for snapshot (redshift): %d (%.3f)' %(snapshot,z))
        tic_snap=time.time()
        
        Xnames=np.array(['logMvir', 'Vmax_Vvir', 'logahalf'])
        
        
        
        # number of bins=nbin-1
        nbin=11
        
        
        # for i in range(len(Xnames)-1):
        #     for j in range(len(Xnames)):
        for i in [0]: #I HAVE ONLY TESTED FOR Mvir Vs X2
            for j in range(len(Xnames)):
                if j>i:
                    tic=time.time()
                    proxyname1=Xnames[i]
                    proxyname2=Xnames[j]
                    print('=====> # bins for %s vs %s: %d' %(proxyname1, proxyname2, (nbin-1)))
                    
                    # load data
                    # ticload=time.time()
                    fileName='profiles_%s_%s%s_snapshot%d.npz' %(proxyname1,proxyname2,subhaloStr,snapshot)
                    profData=np.load(saveDir+fileName)
                    parBin1=profData['parBin1']
                    parBin2=profData['parBin2']
                    r=profData['r']
                    b=profData['b']
                    eb=profData['eb']
                    rho=profData['rho']
                    erho=profData['erho']
                    numHalosInBin=profData['numHalosInBin']
                    haloParMean=profData['haloParMean']
                    haloParStd=profData['haloParStd']
                    logMvirMeanInBin=profData['logMvirMeanInBin']
                    RvirMeanInBin=profData['RvirMeanInBin']
                    logahalfMeanInBin=profData['logahalfMeanInBin']
                    Vmax_VvirMeanInBin=profData['Vmax_VvirMeanInBin']
                    MARMeanInBin=profData['MARMeanInBin']
                    
                    r_edges=np.load(simDir+'RadialBins.npy')
                    r_middle=np.sqrt(r_edges[:-1]*r_edges[1:])
                    r_middle[0]=r_edges[1]/np.sqrt(r_edges[2]/r_edges[1]) #middle of log bins

                    # tocload=time.time()
                    # print('time to load data (s): %.2f' %(tocload-ticload))
                    X1s=haloParMean[:,0] #halo parameter 1 values (x axis)
                    X2s=haloParMean[:,1] #halo parameter 2 values (y axis)
                    
                    X1Stds=haloParStd[:,0]
                    X2Stds=haloParStd[:,1]
                    
                    X1Bins=[]
                    X2Bins=[]
                    for ii in range(len(parBin1)-1):
                        for jj in range(len(parBin2)-1):
                            X1Bins.append([parBin1[ii], parBin1[ii+1]])
                            X2Bins.append([parBin2[jj], parBin2[jj+1]])
                    X1Bins=np.array(X1Bins)
                    X2Bins=np.array(X2Bins)
                    
                    
                    
                    
                    nHaloLimit=100
                    
                    # Obtain Rx values
                    RcdInBin=np.zeros(len(X1s))
                    McdInBin=np.zeros(len(X1s))
                    RspInBin=np.zeros(len(X1s))
                    MspInBin=np.zeros(len(X1s))
                    
                    krange=np.arange(len(X1s))#[firstind:firstind+10]
                    ColorList=itertools.cycle(plt.cm.jet(np.linspace(0.,1.,len(krange))))
                    for k in krange:
                        color=next(ColorList)
                        if numHalosInBin[k]<=nHaloLimit:
                            Rcd_k=0
                            Mcd_k=0
                            Rsp_k=0
                            Msp_k=0
                        if numHalosInBin[k]>nHaloLimit:
                            Rvir_k=RvirMeanInBin[k]
                            
                            if redbini==1:
                                # works well for both Vmax/Vvir and logahalf (for both subhaloStr)
                                if j==2:
                                    massthresh=13.4
                                    if logMvirMeanInBin[k]<massthresh:
                                        rFitSel=(r>0.7*Rvir_k)&(r<5*Rvir_k) # Mpc/h
                                    if logMvirMeanInBin[k]>=massthresh:
                                        rFitSel=(r>1.3*Rvir_k)&(r<3.8*Rvir_k) # Mpc/h
                                        slopeThreshhold=-0.2
                                if j==1:
                                    massthresh=13.4
                                    if logMvirMeanInBin[k]<massthresh:
                                        rFitSel=(r>0.7*Rvir_k)&(r<4*Rvir_k) # Mpc/h
                                    if logMvirMeanInBin[k]>=massthresh:
                                        rFitSel=(r>1.3*Rvir_k)&(r<3.8*Rvir_k) # Mpc/h
                                        slopeThreshhold=-0.2
                            
                            if redbini==3:
                                # works well for both Vmax/Vvir and logahalf (for both subhaloStr)
                                massthresh=14
                                if logMvirMeanInBin[k]<massthresh:
                                    rFitSel=(r>0.7*Rvir_k)&(r<4*Rvir_k) # Mpc/h
                                if logMvirMeanInBin[k]>=massthresh:
                                    rFitSel=(r>1.3*Rvir_k)&(r<3.8*Rvir_k) # Mpc/h
                            
                            
                            
                            
                            #bias and error
                            rData=r[rFitSel]
                            bData=b[k][rFitSel]
                            ebData=eb[k][rFitSel]
                            
                            fitProf=bp.biasProfile(z=z,cosmo=cosmo)#,log10Mvir=logMvirMeanInBin[k])
                            fitProf.showInputRWarnings=False
                            
                            _,_=fitProf.fit_profile(x=rData,prof=bData,prof_err=ebData,prof_name='b')
                            
                            
                            # pfit, _=bt.fit_bias_profile(rData, bData, prof_err=ebData)#,logMass=logMvirMeanInBin[k]) 
                            # fitProf.updateParams_fromPopt(pfit)
                            
                            bFit=fitProf.bias(rData)
                            
                            RLong=np.logspace(np.log10(rData.min()),np.log10(rData.max()),1000)
                            
                            
                            bFitLong=fitProf.bias(RLong)
                            
                            Rcd_k, Mcd_k = fitProf.getRcdMcd(RLong,slopeThreshhold=slopeThreshhold)
                            # Rsp_k, Msp_k=fitProf.getRspMsp(RLong)
                            
                            
                            
                            
                            ##### fit for Rsp
                            # if j==2: #fitting Vmax/Vvir
                            massthresh=13.5
                            if logMvirMeanInBin[k]<massthresh:
                                rFitSel=(r>0.1*Rvir_k)&(r<3.*Rvir_k) # Mpc/h
                            if logMvirMeanInBin[k]>=massthresh:
                                rFitSel=(r>0.5*Rvir_k)&(r<4*Rvir_k) # Mpc/h
                            
                            #bias and error
                            rData=r[rFitSel]
                            rhoData=rho[k][rFitSel]
                            erhoData=erho[k][rFitSel]
                            
                            Mvir_k=10**logMvirMeanInBin[k]
                            mdef='vir'
                            # noMass_bParLims=np.array([ [noMass_lim_r0[0], noMass_lim_dr1[0], noMass_lim_dr2[0], noMass_lim_alpha[0], noMass_lim_beta[0], noMass_lim_gamma[0], noMass_lim_b0[0]], 
                            #                  [noMass_lim_r0[1], noMass_lim_dr1[1], noMass_lim_dr2[1], noMass_lim_alpha[1], noMass_lim_beta[1], noMass_lim_gamma[1], noMass_lim_b0[1]] ])
                            
                            cvir_k=concentration.concentration(M=Mvir_k, mdef=mdef, z=z) # Here is an example of inteprolating the concntration
                            # below line makes the output data (DeltaSigma=...) include the DEFAULT 2-halo term (see DK14 paper above)
                            
                            fitProf=profile_dk14.getDK14ProfileWithOuterTerms(M=Mvir_k,c=cvir_k,z=z,
                                                                              mdef=mdef)
                            
                            rDataFit=rData*(1e3) #[r]=kpc/h from Mpc/h
                            rhoDataFit=rhoData/(1e3)**3 #[rho]=Msun h /kpc^3 from Msun h /Mpc^3 
                            
                            erhoDataFit=erhoData/(1e3)**3 #[rho]=Msun h /kpc^3 from Msun h /Mpc^3 
                            
                            # ticfit=time.time()
                            fitResults=fitProf.fit(r=rDataFit,q=rhoDataFit,
                                                   q_err=erhoDataFit, quantity='rho',
                                                   verbose=False,maxfev=int(1e6))
                            # tocfit=time.time()
                            # print('total time for 1 fit (s): %.2f' %(tocfit-ticfit)) #~1739.56 s or 30 min
                            
                            # bFit=fitProf.bias(rData)
                            # rhoFit=fitProf.density(rDataFit)*(1e3)**3 #[rho]=Msun h /Mpc^3 from Msun h /kpc^3 
                            
                            RLong=np.logspace(np.log10(rData.min()),np.log10(rData.max()),1000)
                            
                            rhoFitLong=fitProf.density(RLong*1e3)*(1e3)**3
                            
                            Rsp_k,Rsp_k_options=bt.rho2Rsp(RLong,rhoFitLong)#,loLimit=0.2)
                            
                            Msp=bt.enclosedMass_fromRhoBinEdges(rho=rho[k],r_middle=r_middle,
                                                                r_edges=r_edges,rcut=Rsp_k)
                        
                        RcdInBin[k]=Rcd_k
                        McdInBin[k]=Mcd_k
                        RspInBin[k]=Rsp_k
                        MspInBin[k]=Msp_k
                        
                        
                    nParBin=nbin-1
                    
                    
                    ########## Make arrays 2D 
                    #calculate middle of parBin1 and parBin2, should have size=len(parBin1)-1
                    xmid=np.array([(parBin1[p]+parBin1[p+1])/2 for p in range(len(parBin1[:-1]))])
                    ymid=np.array([(parBin2[p]+parBin2[p+1])/2 for p in range(len(parBin2[:-1]))])
                    Xmid, Ymid=np.meshgrid(xmid, ymid)
                    
                    X2d=Xmid #par X1
                    Y2d=Ymid #par X2
                    XStds2d=make2d(X1Stds)
                    YStds2d=make2d(X2Stds)
                    
                    RcdInBin2d=make2d(RcdInBin) 
                    McdInBin2d=make2d(McdInBin)
                    RspInBin2d=make2d(RspInBin)
                    MspInBin2d=make2d(MspInBin)
                    
                    # Rsps2d=make2d(rsps) #Rsp from hanBias of original binned bias (not resampled bias)
                    
                    logMvir2d=make2d(logMvirMeanInBin) #mass in bin
                    logahalf2d=make2d(logahalfMeanInBin) #ahalf in bin
                    Vmax_Vvir2d=make2d(Vmax_VvirMeanInBin)
                    
                    Rvir2d=make2d(RvirMeanInBin)
                    MAR2d=make2d(MARMeanInBin)
                    
                    nn=make2d(numHalosInBin) # number halos in bin
                    
    
                    ##### save data!!!
                    # data=np.vstack([X2d.ravel(), Y2d.ravel(), 
                    #                 XStds2d.ravel(), YStds2d.ravel(), 
                    #                 RcdInBin2d.ravel(), McdInBin2d.ravel(), #Rsps2d.ravel(), 
                    #                 nn.ravel(), 
                    #                 logMvir2d.ravel(), 
                    #                 logahalf2d.ravel(), Vmax_Vvir2d.ravel()])
                    
                    fileName='maps_HaloPar%d%d_nbin%d%s_snapshot%d' %(i, j, nbin,subhaloStr,snapshot)
                    toc=time.time()
                    print(fileName+': %.2f s' %(toc-tic))
                    np.savez(saveDir+fileName, X1=X2d.ravel(),X2=Y2d.ravel(),
                             X1std=XStds2d.ravel(),X2std=YStds2d.ravel(),
                             Rcds=RcdInBin2d.ravel(),Mcds=McdInBin2d.ravel(),
                             nn=nn.ravel(),logMvirs=logMvir2d.ravel(),
                             Rvirs=Rvir2d.ravel(),MARs=MAR2d.ravel(),
                             logahalfs=logahalf2d.ravel(),Vmax_Vvirs=Vmax_Vvir2d.ravel(),
                             Rsps=RspInBin2d.ravel(),Msps=MspInBin2d.ravel()
                             )
                    
        
        toc_snap=time.time()
        print('%s vs %s data, %d nbins (s): %.2f' %(proxyname1, proxyname2, nbin, (toc_snap-tic_snap)))
        
    
    
toctot=time.time()

print('COMPLETED! (s): %.2f' %(toctot-tictot))