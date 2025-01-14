#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 12:20:20 2021

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


from colossus.cosmology import cosmology as ccosmo
my_cosmo = {'flat': True, 'H0': 70.4, 'Om0': 0.268, 'Ob0': 0.044, 'sigma8': 0.83, 'ns': 0.968}
cosmo = ccosmo.setCosmology('my_cosmo', my_cosmo)

def save_R0vsX1X2_binned_bias(proxy1, proxyname1, proxy2, proxyname2, selectedPar=None, nbin=None, bintype='linear', delta_pad=0, norm=True):
    tic = time.time()
    print('===========================================================')
    print('%s vs %s , nbin = %d' %(proxyname1, proxyname2, nbin))
    # fileNames=['Mvir', 'Vmax', 'j', 'e', 'a_half', 'delta_e']
    # fileNames = np.array(fileNames)
    
    if nbin is None:
        nbin=10
    if bintype=='linear': #set up linear bins for proxy1 and 2 (parameter values) in nbin bins
        parBin1=np.linspace(proxy1[selectedPar].min(), proxy1[selectedPar].max(), nbin)
        parBin2=np.linspace(proxy2[selectedPar].min(), proxy2[selectedPar].max(), nbin)
    elif bintype=='percent':
        parBin1=np.percentile(proxy1, np.linspace(0, 100, nbin))
        parBin2=np.percentile(proxy2, np.linspace(0, 100, nbin))
        
    #print('nbin = %d' %nbin)
    #b=np.zeros(shape=(nbin*nbin,3)) #Emtpy matrix, first two columns are 
    b=[]
    rho=[]
    erho=[]
    eb=[]
    haloParMean=[] #mean of parameters in bin
    haloParStd=[]
    numHalosInBin=[]
    logMvirMeanInBin=[]
    RvirMeanInBin=[]
    logahalfMeanInBin=[]
    Vmax_VvirMeanInBin=[]
    MARMeanInBin=[]
    spinMeanInBin=[]
    eMeanInBin=[]
    delta_eMeanInBin=[]
    
    corrfunc=CorrFunc
    for ii in range(len(parBin1)-1):
        
        for jj in range(len(parBin2)-1):
            #Then do the code
            #sel=(proxy>parBin[i])&(proxy<parBin[i+1])&selectedPar #Array of T and F for different proxy (parameter values), T for those within current bin
            sel=(proxy1>parBin1[ii])&(proxy1<=parBin1[ii+1])&(proxy2>parBin2[jj])&(proxy2<=parBin2[jj+1]) #selected halos in bins X1i and X2j
            
            sumsel=sum(sel)
            if sumsel==0:
                par1mean = 0
                par1std = 0
                par2mean = 0
                par2std = 0
                
                bmean = np.zeros(len(densities[0]))
                #bstd = densities[sel].std(0)/corrfunc/np.sum(sel)
                bstd=np.ones(len(densities[0]))
                rhomean = bmean*0.
                #rhostd = densities[sel].std(0)/corrfunc/np.sum(sel)
                rhostd = np.ones(len(densities[0]))
                
                logMvirMean=0
                RvirsMean=0
                logahalfsMean=0
                Vmax_VvirMean=0
                MARsMean=0
                spinMean=0
                eMean=0
                delta_eMean=0
                
            else:
                par1mean = proxy1[sel].mean()
                par1std = proxy1[sel].std()
                par2mean = proxy2[sel].mean()
                par2std = proxy2[sel].std()
                
                bmean=((densities[sel].mean(0)/rhom)-1)/(corrfunc)
                bstd = (((densities[sel]/rhom)-1)/(corrfunc)).std(0)/np.sqrt(sumsel)
                rhomean = densities[sel].mean(0)
                rhostd = densities[sel].std(0)/np.sqrt(sumsel)
                
                logMvirMean=logMvirs[sel].mean()
                RvirsMean=Rvirs[sel].mean()
                logahalfsMean=logahalfs[sel].mean()
                MARsMean=MARs[sel].mean()
                Vmax_VvirMean=Vmax_Vvirs[sel].mean()
                spinMean=spin[sel].mean()
                eMean=e[sel].mean()
                delta_eMean=delta_e[sel].mean()
                
                
                
            b.append(bmean)
            eb.append(bstd)
            rho.append(rhomean)
            erho.append(rhostd)
            
            
            logMvirMeanInBin.append(logMvirMean)
            RvirMeanInBin.append(RvirsMean)
            logahalfMeanInBin.append(logahalfsMean)
            Vmax_VvirMeanInBin.append(Vmax_VvirMean)
            MARMeanInBin.append(MARsMean)
            spinMeanInBin.append(spinMean)
            eMeanInBin.append(eMean)
            delta_eMeanInBin.append(delta_eMean) 
            
            haloParMean.append([par1mean, par2mean])
            haloParStd.append([par1std, par2std])
            numHalosInBin.append(proxy2[sel].size)
            print('[%.2f, %.2f], [%.2f, %.2f]: %d' %(parBin1[ii], parBin1[ii+1], parBin2[jj], parBin2[jj+1], proxy2[sel].size))
            
    b=np.array(b)
    eb=np.array(eb)
    rho=np.array(rho)
    erho=np.array(erho)
    haloParMean=np.array(haloParMean)
    haloParStd=np.array(haloParStd)
    
    numHalosInBin=np.array(numHalosInBin)
    logMvirMeanInBin=np.array(logMvirMeanInBin)
    RvirMeanInBin=np.array(RvirMeanInBin)
    logahalfMeanInBin=np.array(logahalfMeanInBin)
    Vmax_VvirMeanInBin=np.array(Vmax_VvirMeanInBin)
    MARMeanInBin=np.array(MARMeanInBin)
    spinMeanInBin=np.array(spinMeanInBin)
    eMeanInBin=np.array(eMeanInBin)
    delta_eMeanInBin=np.array(delta_eMeanInBin) 
    
    # nameInd1 = xnames_full==proxyname1
    # name1 = fileNames[nameInd1][0]
    # nameInd2 = xnames_full==proxyname2
    # name2 = fileNames[nameInd2][0]
    fileName = 'profiles_%s_%s%s_snapshot%d' %(proxyname1,proxyname2,subhaloStr,snapshot)
    
    np.savez(saveDir+fileName,parBin1=parBin1,parBin2=parBin2,
             r=r,CorrFunc=CorrFunc,
             b=b,eb=eb,rho=rho,erho=erho,
             numHalosInBin=numHalosInBin,
             haloParMean=haloParMean,haloParStd=haloParStd,
             logMvirMeanInBin=logMvirMeanInBin,RvirMeanInBin=RvirMeanInBin,
             logahalfMeanInBin=logahalfMeanInBin,
             Vmax_VvirMeanInBin=Vmax_VvirMeanInBin,
             MARMeanInBin=MARMeanInBin,
             spinMeanInBin=spinMeanInBin,eMeanInBin=eMeanInBin,
             delta_eMeanInBin=delta_eMeanInBin) 
    
    toc = time.time()
    print('%s vs %s data saved , nbin = %d, time: %.2f (s)' %(proxyname1, proxyname2, nbin, (toc-tic)))



tictot=time.time()
##### set up directory names for local

#load directory 
simDir='/Users/MattFong/Desktop/Projects/Data/halo_prof/'

saveDir=simDir+'haloPar_z_X1X2/'

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

# excludeSubhalos=True
# excludeSubhalos=False
# excludeSubhaloss=[True,False]
excludeSubhaloss=[True] #completed running True June 9, 2022, before going to Yosemite

for excludeSubhalos in excludeSubhaloss: #completed running all
    if excludeSubhalos:
        subhaloStr='_excludeSubhalos'
    if excludeSubhalos is False:
        subhaloStr=''
    
    # only look at snapshot 91, z=0.253 and 99, z=0
    # for redbini in [1]: # completed running for [1,3]
    #     snapshot=redshiftBins[:,0][redbini]
    #     z=redshiftBins[:,1][redbini]
    for redbini in np.arange(snap_z_info.shape[0]):
        snapshot=snap_z_info[:,0][redbini]
        z=snap_z_info[:,1][redbini]
        
        print('---------- running code for snapshot (redshift): %d (%.3f)' %(snapshot,z))
        tic_snap=time.time()
          
        # load data
        ticload=time.time()
        
        # marMatchedData.files
        # ['marTrackIDs',
        #  'snapshot',
        #  'z',
        #  'R',
        #  'R_bins',
        #  'CorrFunc',
        #  'densities',
        #  'velocities']
        marMatchedData=np.load(simDir+'profiles_haloIDMatchedData%s_snapshot%d.npz' %(subhaloStr,snapshot))
        r=marMatchedData['R']
        # R_bins=marMatchedData['R_bins']
        CorrFunc=marMatchedData['CorrFunc']
        densities=marMatchedData['densities']
        # velocities=marMatchedData['velocities']
        
        
        marMatchedData=np.load(simDir+'MAR_haloIDMatchedData%s_snapshot%d.npz' %(subhaloStr,snapshot))
        # marTrackIDs=marMatchedData['marTrackIDs']
        logMvirs=marMatchedData['logMvirs']
        Rvirs=marMatchedData['Rvirs'] #[r]=Mpc/h PHYSICAL!!!
        Vmaxs=marMatchedData['Vmaxs'] #[v]=km/s PHYSICAL!!!
        logahalfs=marMatchedData['logahalfs']
        MARs=marMatchedData['MARs']
        # names=marMatchedData['names']
        # data=marMatchedData['data']
        spin=marMatchedData['spin']
        e=marMatchedData['e1']
        delta_e=marMatchedData['delta_e']  
        
        tocload=time.time()
        print('time to load data (s): %.2f' %(tocload-ticload))
        
        
        
        
        
        G=constants.G/(1e3) #[G]=Mpc km2/𝑀⊙/𝑠2 from kpc km2/𝑀⊙/𝑠2 PHYSICAL
        Mvirs=10**logMvirs #[M] = M_odot/h
        Vvirs=np.sqrt(G*Mvirs/Rvirs) #[v]=km/s
        
        Vmax_Vvirs=Vmaxs/Vvirs
        
        rhom=cosmo.rho_m(z)*(1e3)**3 #[rho]=M_sun h^2 / Mpc^3 from M_sun h^2 / kpc^3
        
        
        
        # number of bins = nbin-1
        nbin = 11
        
        #choose fit range for bias, xsel=r>r[4] for this simulation, avoid deviation
        # xsel=r>r[4]
        msel=logMvirs>11.5
        
        Xnames=np.array(['logMvir', 'Vmax_Vvir', 'spin','e','logahalf','delta_e']) 
        Xdata=np.array([logMvirs,Vmax_Vvirs,spin,e,logahalfs,delta_e])
        
        
        # for i in [0]:
        for i in np.arange(len(Xnames)-1): #i>0&i<5, since i finished for logMvir already
            for j in range(len(Xnames)):
            # for j in [5]: #just redo delta_e bins
                if j>i:
                    # save_R0vsX1X2_binned_bias(proxy1, proxyname1, proxy2, proxyname2, 
                    #  selectedPar=None, nbin=None, bintype='linear', delta_pad=0, norm=True):
                    save_R0vsX1X2_binned_bias(proxy1=Xdata[i], proxyname1=Xnames[i], 
                                              proxy2=Xdata[j], proxyname2=Xnames[j], 
                                              selectedPar=msel, nbin=nbin, bintype='linear')
        
toctot=time.time()

print('COMPLETED! (s): %.2f' %(toctot-tictot))