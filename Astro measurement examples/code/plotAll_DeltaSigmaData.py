#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:12:50 2021

plot all DeltaSigma data together, NO FITS

@author: MattFong
"""

import sys,os,itertools
import h5py
import numpy as np
import matplotlib.pyplot as plt

import matplotlib 
matplotlib.rcParams.update({'font.size': 22,'figure.figsize': (10,10)})
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import matplotlib.ticker as ticker


import glob
import re
import time

from colossus.halo import mass_adv
##### set up data directory, based on current working directory (cwd)
codeDir=os.getcwd()+'/'
#so far acceptable cwds
cwdList=['/Users/MattFong/Desktop/Projects/Code/SJTU/','/home/mfong/Code/halo_prof/']
if codeDir not in cwdList:
    raise Exception('Need to set up work and data directories for biasTools_z!!!')
# if codeDir==cwdList[0]: #for remote gravity server
#     dataDir='/Users/MattFong/Desktop/Projects/Data/halo_prof/'
# if codeDir==cwdList[1]: #for local mac
#     dataDir='/home/mfong/Data/halo_prof/'

sys.path.append(os.path.abspath(codeDir))
import biasTools_z as bt
import biasProfile as bp

from colossus.utils import constants

from colossus.cosmology import cosmology as ccosmo
my_cosmo = {'flat': True, 'H0': 70.4, 'Om0': 0.268, 'Ob0': 0.044, 'sigma8': 0.83, 'ns': 0.968}
cosmo = ccosmo.setCosmology('my_cosmo', my_cosmo)
Om0=cosmo.Om0


cosmoYang2020=ccosmo.setCosmology('planck18-only')
Om0Yang=cosmoYang2020.Om0


##### choose redshift range for data
redshiftRanges=['z_02_03/','z_02_03_binMh/','z_02_03_binMh_richness3/','z_02_03_binMh_richness5/']
# redshiftRanges=['z_02_03_binMh_richness3/','z_02_03_binMh_richness5/']


# col3=True #for old results, without accurate redshift estimation
col3=False
if col3:
    redshiftRanges=['col3_z_02_03/','col3_z_02_03_binMh/']
    
for redshiftRange in redshiftRanges:
    
    
    dataDir='/Users/MattFong/Desktop/Projects/Data/DECaLS/group_cluster/DESI_Yang2020/'
    savePlotDir='/Users/MattFong/Desktop/Projects/Code/SJTU/DECaLS/plots/'
    
    saveDatDir=dataDir+'results/'+redshiftRange
    dataDir=dataDir+redshiftRange
    
    lensCatIDs=np.load(dataDir+'lensCatIDs.npy')
    # lensCatIDs=np.vstack([lensIDs,nLens,
    #                       Lambda_meds,Lambda_los,Lambda_his,
    #                       Mh_meds,Mh_los,Mh_his,
    #                       Lambda_means,Lambda_stds,
    #                       Mh_means,Mh_stds]).T
    
    dataDir=dataDir+'results/'
    savePlotDir=savePlotDir+redshiftRange
    
    
    plot_meanExcessDensity=False #keep FALSE
    print('============== plotting ONLY meanExcessDensity: ', plot_meanExcessDensity)
    if plot_meanExcessDensity:
        savePlotDir=savePlotDir+'meanExcessDensity/'
        saveDatDir=saveDatDir+'meanExcessDensity/'
        
        
    # fitUsingCurvefit=True
    fitUsingCurvefit=False
    print('============== plotting fitUsingCurvefit: ', fitUsingCurvefit)
    if fitUsingCurvefit:
        savePlotDir=savePlotDir+'fitUsingCurveFit/'
        saveDatDir=saveDatDir+'fitUsingCurveFit/'
    
    
    if not os.path.exists(savePlotDir):
        os.makedirs(savePlotDir)
    if not os.path.exists(saveDatDir):
        os.makedirs(saveDatDir)
    
    print('***** REDSHIFT RANGE DIRECTORY: %s' %redshiftRange)
    if redshiftRange=='z_02_03/':
        zRange=[0.2,0.3]
    if redshiftRange=='z_02_06/':
        zRange=[0.2,0.6]
    
    ##### new "corr_jack_signal.dat" from 
    #/lustre/home/acct-phyzj/phyzj-m31/mfong/DECaLS/cross_corr_Mar_DECALS/results/result_lensCat_3
    # columns: rbini, DeltaSigma, DeltaSigma_std, DeltaSigmaCross, DeltaSigmaCross_std, R, ngals
    
    directory_contents = [f.name for f in os.scandir(dataDir) if f.is_dir()] #unsorted
    
    sortedDirContents=np.array([int(re.findall(r'\d+',file)[0]) for file in directory_contents])
    
    sortLensIDs=sortedDirContents.argsort()
    directory_contents=np.array(directory_contents)
    directory_contents=directory_contents[sortLensIDs]
        
    print(directory_contents)
    
    # Rcds=[]
    # pfits_all=np.load(saveDatDir+'pfits_all.npy') #unsorted
    # Rsmins=np.load(saveDatDir+'Rsmins.npy') #unsorted
    # # for file in directory_contents[4:5]: #test
    # Rcds=np.load(saveDatDir+'Rcds.npy')
    
    
    
    titleText=r'%s' %(redshiftRange)
    titleText+=r'$z \in [%.1f,%.1f]$' %(zRange[0],zRange[1]) 
    titleText+='\n'+r'$({\rm log_{10}} M_{\rm vir}^{\rm estimated}: {\rm log_{10}}n_{\rm lens})$'
    
    
    print(titleText)
    # plt.yticks(ticks=None,labels=None)
    #### Delta Sigma plots
    figDeltaSigma = plt.figure(figsize=(10,12))
    
    axp=plt.subplot(2,1,1)
    axp.set_title(titleText,fontsize=11)
    axp.tick_params(right='on', top='on', direction='in', which='both')
    axp.set_ylabel(r'$\Delta \Sigma \, [{\rm M_{\odot}} \, h \, / \, {\rm pc^2}]$')
    axp.tick_params(axis='y')
    plt.setp(axp.get_xticklabels(), visible=False)
    axp.set_xscale('log')
    axp.set_yscale('log')
    axp.axvspan(0,0.06, alpha=0.1, color='grey')
    # axp.grid(b=True, which='both', lw=0.1, color='k', ls='--')
    # axp.text(.5,.9, r'$z= %.3f$' %(z), horizontalalignment='center', 
    #                 verticalalignment='center', fontsize=25, transform=axp.transAxes)
    
    # ratio plots
    axl=plt.subplot(2,1,2, sharex=axp)
    axl.tick_params(right='on', top='on', direction='in', which='both')
    axl.set_xlabel(r'$r\, [{\rm Mpc}/h]$')
    axl.set_ylabel(r'$\Delta \Sigma \, / \, \Delta \Sigma_{\rm 0}$')
    axl.tick_params(axis='x')
    axl.tick_params(axis='y')
    axl.set_xscale('log')
    axl.set_yscale('log')
    # axl.axhline(10,color='k',alpha=0.5,lw=3)
    # axl.axhline(-10,color='k',alpha=0.5,lw=3)
    axl.axvspan(0,0.06, alpha=0.1, color='grey')
    # axl.set_xlim([0.4,20])
    # axl.set_yscale('log')
    # axl.grid(b=True, which='both', lw=0.1, color='k', ls='--')
    
    #original fit range
    # loLim=0.06
    # loLim=0.5
    # hiLim=20
    
    # highMass_biasThreshhold=4
    hmbt=4
    # slopeThreshhold
    st=-0.5
    
    loLimFactor=0.7
    
    ColorList=itertools.cycle(plt.cm.jet(np.linspace(0.,1.,len(directory_contents))))
    
    sortedDirContents=np.array([int(re.findall(r'\d+',file)[0]) for file in directory_contents])
    
    sortLensIDs=sortedDirContents.argsort()
    directory_contents=np.array(directory_contents)
    directory_contents=directory_contents[sortLensIDs]
    # Rsmins=Rsmins[sortLensIDs]
    # pfits_all=pfits_all[sortLensIDs]
    
    DeltaSigma_0=None
    for file in directory_contents:
        lbin=int(re.findall(r'\d+',file)[0]) #get lambda bin ID
        
        z_guess=0.25344
        
        lensData=lensCatIDs[lbin]
        # mixed up the order in the original saving code
        lensID,nLens,Lambda_meds,Lambda_his,Lambda_los,logMh_meds,logMh_his,logMh_los,Lambda_means,Lambda_stds,logMh_means,logMh_stds=lensData
        Mh_meds=10**logMh_meds
        Mh_his=10**logMh_his
        Mh_los=10**logMh_los
        Mh_means=10**logMh_means
        Mh_stds=10**logMh_stds
        #convert masses to our cosmology?
        def convertMass_cosmologies(Mx,Om_original,Om_new):
            return Mx*(Om_new/Om_original)
        
        Mh_meds=convertMass_cosmologies(Mh_meds,Om0Yang,Om0)
        Mh_los=convertMass_cosmologies(Mh_los,Om0Yang,Om0)
        Mh_his=convertMass_cosmologies(Mh_his,Om0Yang,Om0)
        # Mh_means=convertMass_cosmologies(Mh_means,Om0Yang,Om0)
        # Mh_stds=convertMass_cosmologies(Mh_stds,Om0Yang,Om0)
        
        ##### need to convert Mh to Mvir
        Mvir_guess,rvir_guess,cvir_guess=mass_adv.changeMassDefinitionCModel(Mh_meds,z_guess,mdef_in='180m',mdef_out='vir')
        Mvir_lo_guess,_,_=mass_adv.changeMassDefinitionCModel(Mh_los,z_guess,mdef_in='180m',mdef_out='vir')
        Mvir_hi_guess,_,_=mass_adv.changeMassDefinitionCModel(Mh_his,z_guess,mdef_in='180m',mdef_out='vir')
        
        M200m_guess,R200m_guess,c200m_guess=mass_adv.changeMassDefinitionCModel(Mh_meds,z_guess,mdef_in='180m',mdef_out='200m')
        R200m_guess=R200m_guess/1e3 #[r]=Mpc/h from kpc/h
        loLim=loLimFactor*R200m_guess
        
        labelMvir=r'$\log M_{\rm vir}$'
        fileNameMvir='Mvir'
        
        
        clusterPath_i=dataDir+file+'/corr_jack_signal.dat'
        
        clusterData=np.genfromtxt(clusterPath_i)
        
        rbini, DeltaSigma, DeltaSigma_std, DeltaSigmaCross, DeltaSigmaCross_std, R, ngals=clusterData.T
        #[DeltaSigma_j]=M_sun h/pc^2
        
        zL_guess=np.mean(zRange)
        
        hCorrection=cosmo.h/cosmoYang2020.h
        # rsel=(R>loLim)&(R<hiLim)
        DeltaSigma=DeltaSigma*hCorrection*(1+zL_guess) #*** These are temporary corrections!!!
        DeltaSigma_std=DeltaSigma_std*hCorrection*(1+zL_guess) #*** These are temporary corrections!!!
        # DeltaSigmaCross=DeltaSigmaCross[rsel]
        # DeltaSigmaCross_std=DeltaSigmaCross_std[rsel]
        R=R/hCorrection
        
        if lbin==0:
            DeltaSigma_0=DeltaSigma
        
        # posSel=DeltaSigma>0 #only fit over positive values in Delta Sigma
        # R=R[posSel]
        # DeltaSigma_std=DeltaSigma_std[posSel]
        # DeltaSigma=DeltaSigma[posSel]
        
        
        ##### fit over DeltaSigma
        log10Mguess=np.log10(Mvir_guess)
        
        ratio_DeltaSigma=DeltaSigma/DeltaSigma_0
        
        # Rcd=Rcds[:,1][lbin]
        
        # lbintest=Rsmins[:,0][lbin]
        # Rsmin=Rsmins[:,1][lbin]
        # RslopeMin=Rsmins[:,2][lbin]
        
        # if lbintest!=lbin:
        #     print('non-matching richness bins!!!')
        
        #prepare plotting characteristic radii
        # radii=np.array([Rcd,Rsmin,RslopeMin])
        # radiiText=np.array([r'$r_{\rm cd}$',r'$r_{\rm smin}$',r'$r_{\rm smin}^{\Delta \Sigma}$'])
        # markers=np.array(['*','o','X'])
        # mews=np.array([2,2,2])
        # markersize=15
        # alphaplts=[1,0.5,0.5]
        # lrange=range(radii.size)
        
        ########## PLOT DeltaSigma ##########
        lw_originals=2
        alpha_originals=0.3
        color=next(ColorList)
        
        axp.fill_between(x=R,y1=DeltaSigma+DeltaSigma_std,y2=DeltaSigma-DeltaSigma_std,
                         color=color,alpha=alpha_originals,
                         label=r'$%.2f_{%.2f}^{%.2f}:%.1f$' %(np.log10(Mvir_guess),np.log10(Mvir_lo_guess),np.log10(Mvir_hi_guess),np.log10(nLens)))
        
        axl.plot(R,ratio_DeltaSigma,color=color)
        
        # axp.axvline(loLim,ls=':',color=color,alpha=0.3)
        # axl.axvline(loLim,ls=':',color=color,alpha=0.3)
    
        
    axp.legend(loc='best',fontsize=8)
    
    plt.subplots_adjust(wspace=0,hspace=0)
    
    fileName='allTogether_DeltaSigma_dataOnly.png'
    plt.savefig(savePlotDir+fileName)
    
    
    
    # Rcds=np.array(Rcds)
    # np.save(file=saveDatDir+'Rcds',arr=Rcds)
    



