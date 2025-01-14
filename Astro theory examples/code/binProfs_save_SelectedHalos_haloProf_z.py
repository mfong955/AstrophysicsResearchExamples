#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:57:46 2022

Select haloes by a bin and save them in separate file, to then plot in the future.

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


def checkIDsMatch(ID1s,ID2s):
    diff=abs(ID1s-ID2s)
    
    sumdiff=sum(diff)
    
    if sumdiff>0:
        print('if indices are matched, sum(diff)=0:')
        print('sum(diff)=%.3f' %sumdiff)
        raise Exception('halo indices dont match!!!')
    if sumdiff==0:
        print('halo indices successfully matched! ')
        

def saveBinnedData_haloDataVsX1X2(proxy1, proxyname1, proxy2, proxyname2, selPar, selStr, lim_z0,
                              selectedPar=None, nbin=None, bintype='linear', 
                              delta_pad=0, norm=True):
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
    fileName = 'selHaloParBy_%s_%s_z0_profiles_%s_%s%s_snapshot%d' %(selPar,selStr,proxyname1,proxyname2,subhaloStr,snapshot)
    
    np.savez(saveDir+fileName,parBin1=parBin1,parBin2=parBin2,lim_z0=lim_z0,
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


parameterNames=np.array(['logMvir', 'Vmax_Vvir', 'spin','e','logahalf','delta_e']) 



# excludeSubhalos=True
# excludeSubhalos=False
excludeSubhaloss=[True,False]
# excludeSubhaloss=[True]

for excludeSubhalos in excludeSubhaloss: #completed running all
    if excludeSubhalos:
        subhaloStr='_excludeSubhalos'
    if excludeSubhalos is False:
        subhaloStr=''
        
    
    redbini=4 #keep fixed, selecting haloes by z=0 first
    
    snapshot=snap_z_info[:,0][redbini]
    z=snap_z_info[:,1][redbini]
    
    # print('---------- running code for snapshot (redshift): %d (%.3f)' %(snapshot,z))
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
    marTrackIDs_z0=marMatchedData['marTrackIDs']
    logMvirs_z0=marMatchedData['logMvirs']
    Rvirs_z0=marMatchedData['Rvirs'] #[r]=Mpc/h PHYSICAL!!!
    Vmaxs_z0=marMatchedData['Vmaxs'] #[v]=km/s PHYSICAL!!!
    logahalfs_z0=marMatchedData['logahalfs']
    MARs_z0=marMatchedData['MARs']
    # names_z0=marMatchedData['names']
    # data_z0=marMatchedData['data']
    spin_z0=marMatchedData['spin']
    e_z0=marMatchedData['e1']
    delta_e_z0=marMatchedData['delta_e']  
    
    tocload=time.time()
    # print('time to load data (s): %.2f' %(tocload-ticload))
    
    
    
    
    
    
    G=constants.G/(1e3) #[G]=Mpc km2/𝑀⊙/𝑠2 from kpc km2/𝑀⊙/𝑠2 PHYSICAL
    Mvirs_z0=10**logMvirs_z0 #[M] = M_odot/h
    Vvirs_z0=np.sqrt(G*Mvirs_z0/Rvirs_z0) #[v]=km/s
    
    Vmax_Vvirs_z0=Vmaxs_z0/Vvirs_z0
    
    
    # Xnames=np.array(['logMvir', 'Vmax_Vvir', 'spin','e','logahalf','delta_e']) 
    Xdata_z0=np.array([logMvirs_z0,Vmax_Vvirs_z0,spin_z0,e_z0,logahalfs_z0,delta_e_z0])
    
    rhom=cosmo.rho_m(z)*(1e3)**3 #[rho]=M_sun h^2 / Mpc^3 from M_sun h^2 / kpc^3
    
    # parameterNames=np.array(['logMvir', 'Vmax_Vvir', 'spin','e','logahalf','delta_e']) 
    
    ##### select halos
    selParInds=[1] #temporary fix
    selStrs=['lo']
    # selParInds=np.arange(len(parameterNames))
    # selStrs=['hi','lo']
    
    for selParInd in selParInds:
        for selStr in selStrs:
            
            print('==================== (current parameter, selection): (%s, %s)' %(parameterNames[selParInd],selStr))
            
            
            if (selParInd==0): #logMvir
                if (selStr=='hi'):
                    lim_z0=14
                if (selStr=='lo'):
                    lim_z0=11.6
                
            if (selParInd==1): #Vmax_Vvir
                if (selStr=='hi'):
                    lim_z0=1.51
                if (selStr=='lo'):
                    if excludeSubhalos:
                        lim_z0=1
                    if excludeSubhalos is False:
                        lim_z0=1.2
                
            if (selParInd==2): #spin
                if (selStr=='hi'):
                    lim_z0=0.11
                if (selStr=='lo'):
                    lim_z0=0.005
                
            if (selParInd==3): #e
                if (selStr=='hi'):
                    lim_z0=0.55
                if (selStr=='lo'):
                    lim_z0=0.35
                
            if (selParInd==4): #logahalf
                if (selStr=='hi'):
                    lim_z0=-0.08
                if (selStr=='lo'):
                    lim_z0=-0.55
                
            if (selParInd==5): #delta_e
                if (selStr=='hi'):
                    if excludeSubhalos:
                        lim_z0=2
                    if excludeSubhalos is False:
                        lim_z0=4
                if (selStr=='lo'):
                    lim_z0=-0.5
            
            if (selStr=='hi'):
                selection=Xdata_z0[selParInd]>=lim_z0
            if (selStr=='lo'):
                selection=Xdata_z0[selParInd]<=lim_z0
            
            print('%s' %parameterNames[selParInd])
            trackIDs_z0=marTrackIDs_z0[selection]
            print('# haloes in selection: %d' %trackIDs_z0.shape[0])
            
            
            
            for redbini in np.arange(snap_z_info.shape[0]):
                snapshot=snap_z_info[:,0][redbini]
                z=snap_z_info[:,1][redbini]
                
                print('running code for snapshot (redshift): %d (%.3f)' %(snapshot,z))
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
                marTrackIDs=marMatchedData['marTrackIDs']
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
                # print('time to load data (s): %.2f' %(tocload-ticload))
                
                # print('initial sizes:')
                # IDs=marTrackIDs
                # print('IDs=marTrackIDs: IDs.shape[0]=%d' %IDs.shape[0])
                
                
                
                
                G=constants.G/(1e3) #[G]=Mpc km2/𝑀⊙/𝑠2 from kpc km2/𝑀⊙/𝑠2 PHYSICAL
                Mvirs=10**logMvirs #[M] = M_odot/h
                Vvirs=np.sqrt(G*Mvirs/Rvirs) #[v]=km/s
                
                Vmax_Vvirs=Vmaxs/Vvirs
                
                
                sel=np.isin(marTrackIDs,trackIDs_z0)
                marTrackIDs=marTrackIDs[sel]
                # checkIDsMatch(marTrackIDs,trackIDs_z0) #each snapshot has diff # haloes, trackIDs_z0 might not match, but whatever
                # may need to edit data from the beginning. only consider haloes where 
                # trackIDs_z0 all match. this may not be necessary though, and sounds like
                # it may take a while to run. Get this initial step done first then check later
                
                densities=densities[sel]
                
                logMvirs=logMvirs[sel]
                Rvirs=Rvirs[sel]
                Vmax_Vvirs=Vmax_Vvirs[sel]
                logahalfs=logahalfs[sel]
                MARs=MARs[sel]
                spin=spin[sel]
                e=e[sel]
                delta_e=delta_e[sel]
                
                
                # print('final sizes:')
                # IDs=trackIDs
                # print('IDs=trackIDs: IDs.shape[0]=%d' %IDs.shape[0])
                # IDs=marTrackIDs
                # print('IDs=marTrackIDs: IDs.shape[0]=%d' %IDs.shape[0])
                
                np.savez(saveDir+'selHaloParBy_%s_%s_z0%s_snapshot%d' %(parameterNames[selParInd],selStr,subhaloStr,snapshot),
                         trackIDs_z0=trackIDs_z0,lim_z0=lim_z0,trackIDs_z=marTrackIDs,snapshot=snapshot,z=z,
                         logMvirs=logMvirs,Rvirs=Rvirs,Vmax_Vvirs=Vmax_Vvirs,
                         logahalfs=logahalfs,MARs=MARs,js=spin,es=e,deltaes=delta_e)#names=names,data=data)
                
                
                
                # number of bins = nbin-1
                nbin = 9
                
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
                            saveBinnedData_haloDataVsX1X2(proxy1=Xdata[i], proxyname1=Xnames[i], 
                                                          proxy2=Xdata[j], proxyname2=Xnames[j], 
                                                          selPar=parameterNames[selParInd],
                                                          selStr=selStr,lim_z0=lim_z0,
                                                          selectedPar=msel, nbin=nbin, 
                                                          bintype='linear')
        
        
toctot=time.time()
print('COMPLETED! (s): %.2f' %(toctot-tictot))