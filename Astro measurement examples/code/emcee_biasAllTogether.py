#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 09:55:27 2021

plot all bias plots together for emcee

@author: MattFong
"""

##### import stuff
import sys,os,time,itertools
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib as mpl

from colossus.halo import mass_adv

# set up cosmology
from colossus.cosmology import cosmology as ccosmo
from colossus.lss import peaks
#convert masses to our cosmology?
def convertMass_cosmologies(Mh_Xiaohu):
    # in this case Mh_Xiaohu MUST be a list/array
    #input [M]=M_sun/h
    #I coded this so it only converts from Xiaohu's cosmology to ours (WMAPs)
    # nus_long and Ms_long in our/Jing's cosmology
    
    cosmoYang2020=ccosmo.setCosmology('planck18-only')
    
    if isinstance(Mh_Xiaohu,np.float64):
        nuh_Xiaohu=peaks.peakHeight(Mh_Xiaohu,z_guess)
        absdiff=abs(nuh_Xiaohu-nus_long)
        M_Jing=Ms_long[absdiff.argmin()]
    else:
        M_Jing=[]
        for Mh_i in Mh_Xiaohu:
            nuh_Xiaohu=peaks.peakHeight(Mh_i,z_guess)
            
            absdiff=abs(nuh_Xiaohu-nus_long)
            M_Jing.append(Ms_long[absdiff.argmin()])
        M_Jing=np.array(M_Jing)
    #set cosmology back to Jing's
    my_cosmo = {'flat': True, 'H0': 70.4, 'Om0': 0.268, 'Ob0': 0.044, 'sigma8': 0.83, 'ns': 0.968}
    cosmo = ccosmo.setCosmology('my_cosmo', my_cosmo)
    return M_Jing

my_cosmo = {'flat': True, 'H0': 70.4, 'Om0': 0.268, 'Ob0': 0.044, 'sigma8': 0.83, 'ns': 0.968}
cosmo = ccosmo.setCosmology('my_cosmo', my_cosmo)
Om0=cosmo.Om0

z_guess=0.25344
Ms_long=10**np.linspace(11,15,50000) #[M]=Msun/h
nus_long=peaks.peakHeight(Ms_long,z_guess) #from sims z_guess=0.25344

cosmoYang2020=ccosmo.setCosmology('planck18-only')
Om0Yang=cosmoYang2020.Om0

tictot=time.time()
##### set up data directory, based on current working directory (cwd)
codeDir=os.getcwd()+'/'
print(codeDir)
#so far acceptable cwds
cwdList=['/Users/MattFong/Desktop/Projects/Code/SJTU/','/home/mfong/Code/DECaLS/',
         '/lustre/home/acct-phyzj/phyzj-m31/mfong/Code/DECaLS/']
if codeDir not in cwdList:
    raise Exception('Need to set up work and data directories for biasTools_z!!!')
if codeDir==cwdList[0]: #for local mac
    print('running on local')
    dataDir='/Users/MattFong/Desktop/Projects/Data/DECaLS/group_cluster/DESI_Yang2020/'
    sys.path.append(os.path.abspath(codeDir))
if codeDir==cwdList[1]: #for remote gravity server
    print('running on gravity server')
    dataDir='/home/mfong/Data/DECaLS/DESI_Yang2020/'
    sys.path.append('/home/mfong/Code')
if codeDir in cwdList[-1:]: #for remote pi2 server
    print('running on pi2 server')
    dataDir='/lustre/home/acct-phyzj/phyzj-m31/mfong/Data/DECaLS/group_cluster/DESI_Yang2020/'
    sys.path.append('/lustre/home/acct-phyzj/phyzj-m31/mfong/Code')



import biasProfile as bp
import biasTools_z as bt


diffLensCatStuff=['z_02_03_binMh_hic/','z_02_03_binMh_loc/']
##### choose redshift range for data
redshiftRanges=['z_02_03_binMh/','z_02_03_binMh_loc/','z_02_03_binMh_hic/']
# redshiftRanges=['z_02_03_binMh/']
cutLoMassBin=True #cuts out lowest mass bin 
cutLoMassBin=False

flattenOuter=False

col3=True #for old results, without accurate redshift estimation
col3=False
if col3:
    redshiftRanges=['col3_z_02_03/','col3_z_02_03_binMh/']
    
# loosenPriors=[False,True]
loosenPriors=[False] #only using False now
for loosenPrior in loosenPriors:
    if loosenPrior:
        addTxt='_loosenPriors'
    else:
        addTxt=''
    print('addTxt: %s ' %addTxt)
    
    if flattenOuter:
        addTxt2='_flattenOuter'
    else:
        addTxt2=''
        
    for redshiftRange in redshiftRanges:
        
        dataDir_lensCatIDs=dataDir+redshiftRange
        
        lensCatIDs=np.load(dataDir_lensCatIDs+'lensCatIDs.npy')
        # lensCatIDs=np.vstack([lensIDs,nLens,
        #                       Lambda_meds,Lambda_los,Lambda_his,
        #                       Mh_meds,Mh_los,Mh_his,
        #                       Lambda_means,Lambda_stds,
        #                       Mh_means,Mh_stds]).T
        
        saveDatDir=dataDir+'results/'+redshiftRange
        
        loadDeltaSigmaDir=dataDir_lensCatIDs+'results/'
        
        ##### set up directory names
        currentCodeDirName='DECaLS/'
        # currentCodeName='emcee_DeltaSigma/'
        
        # codeDir=codeDir+currentCodeDirName
        # loadPlotDir=codeDir+currentCodeDirName+'plots/emcee_DeltaSigma/'+redshiftRange
        savePlotDir=codeDir+currentCodeDirName+'plots/'+redshiftRange
        
        
        #make directories if not made
        if not os.path.exists(savePlotDir):
            os.makedirs(savePlotDir)
        if not os.path.exists(saveDatDir):
            os.makedirs(saveDatDir)
            
        
        print('codeDir=%s' %codeDir)
        print('dataDir=%s' %dataDir)
        print('================================= SAVING DIRECTORIES =================================')
        print('savePlotDir=%s' %savePlotDir)
        print('saveDatDir=%s' %saveDatDir)
        print('======================================================================================')
        
        
        print('***** REDSHIFT RANGE DIRECTORY: %s' %redshiftRange)
        zRange=[0.2,0.3]
        
        
        
        directory_contents = [f.name for f in os.scandir(loadDeltaSigmaDir) if f.is_dir()]
        # directory_contents=['lensCat_0','lensCat_1','lensCat_3','lensCat_4','lensCat_6'] #test
        sortedDirContents=np.array([int(re.findall(r'\d+',file)[0]) for file in directory_contents])
        
        sortLensIDs=sortedDirContents.argsort()
        directory_contents=np.array(directory_contents)
        directory_contents=directory_contents[sortLensIDs]
        
        # directory_contents=directory_contents[:3] #test
        # if (redshiftRange == 'z_02_03_binMh/')&(cutLoMassBin is True):
        if cutLoMassBin is True:
            directory_contents=directory_contents[1:]
        # directory_contents=['lensCat_4']
        
        print(directory_contents)
        
        
        
        
        
        ##### Choose fit limits
        hiLim=30 #Mpc/h
        # loLimFactors=[0.3,0.5,0.7]
        loLimFactors=[0.5]
        
        for loLimFactor in loLimFactors:
            # # loLims=[0.06,0.1,0.3,0.5]
            # loLims=[0.5]
            # for loLim in loLims:
            
            ##### get Rcds from DeltaSigma samples
            RxMxStats=np.load(saveDatDir+'RxMxStats_loLimFactor_%d_hiLim_%d%s.npy' %(int(10*loLimFactor),hiLim,addTxt))
            # RxMxStats.append([Rcds_med, Rcds_lo, Rcds_hi, Rsps_med, Rsps_lo, Rsps_hi, 
            #                     Deltacds_z_med, Deltacds_z_lo, Deltacds_z_hi, 
            #                     Deltasps_z_med, Deltasps_z_lo, Deltasps_z_hi,
            #                     Deltacds_z0_med, Deltacds_z0_lo, Deltacds_z0_hi, 
            #                     Deltasps_z0_med, Deltasps_z0_lo, Deltasps_z0_hi])
            Rcd_meds=RxMxStats[:,0]
            Rcd_los=RxMxStats[:,1]
            Rcd_his=RxMxStats[:,2]
            
            titleText='%s' %(redshiftRange)
            titleText+="\n"+' loLimFactor_%s%s%s' %(loLimFactor,addTxt,addTxt2)
            
            plt.figure()
            # plt.yticks(ticks=None,labels=None)
            ########## PLOT bias ##########
            ax3=plt.subplot(1,1,1)
            ax3.set_title(titleText)
            ax3.tick_params(right='on', top='on', direction='in', which='both')
            ax3.set_xlabel(r'$r\, [{\rm Mpc}/h]$')
            ax3.set_ylabel(r'$b$')
            ax3.tick_params(axis='x')
            ax3.tick_params(axis='y')
            ax3.set_xscale('log')
            ax3.set_yscale('log')
            
            # ax3.set_ylim([1e-2,25])
            
            ColorList=itertools.cycle(plt.cm.jet(np.linspace(0.,1.,len(directory_contents))))
            
            # RcdSamples=[]
            for file in directory_contents:
            # for file in directory_contents[:1]: # test
                print('====================================================')
                tic=time.time()
                lbin=int(re.findall(r'\d+',file)[0]) #get lambda bin ID
                
                # color=next(ColorList)
                
                zL_guess=np.mean(zRange)
                # loLim=loLims[lbin]
                
                ##### load halo data and make init guesses
                lensData=lensCatIDs[lbin]
                # mixed up the order in the original saving code
                if redshiftRange in diffLensCatStuff:
                    lensID,nLens,Lambda_meds,Lambda_los,Lambda_his,logMh_meds,logMh_los,logMh_his,Lambda_means,Lambda_stds,logMh_means,logMh_stds,L_meds,L_los,L_his,L_means,L_stds,Reff_meds,Reff_los,Reff_his,Reff_means,Reff_stds,Leff_meds,Leff_los,Leff_his,Leff_means,Leff_stds,Neff_meds,Neff_los,Neff_his,Neff_means,Neff_stds=lensData
                else:
                    lensID,nLens,Lambda_meds,Lambda_los,Lambda_his,logMh_meds,logMh_los,logMh_his,Lambda_means,Lambda_stds,logMh_means,logMh_stds=lensData
                Mh_meds=10**logMh_meds
                Mh_his=10**logMh_his
                Mh_los=10**logMh_los
                Mh_means=10**logMh_means
                Mh_stds=10**logMh_stds
                #convert masses to our cosmology?
                # def convertMass_cosmologies(Mx,Om_original,Om_new):
                #     #input [M]=M_sun/h
                #     #I coded this so it only converts from Xiaohu's cosmology to WMAPs
                #     hCorrection=cosmo.h/cosmoYang2020.h
                #     return Mx*(Om_new/Om_original)/hCorrection
                
                Mh_meds=convertMass_cosmologies(Mh_meds)#,Om0Yang,Om0)
                Mh_los=convertMass_cosmologies(Mh_los)#,Om0Yang,Om0)
                Mh_his=convertMass_cosmologies(Mh_his)#,Om0Yang,Om0)
                # Mh_means=convertMass_cosmologies(Mh_means,Om0Yang,Om0)
                # Mh_stds=convertMass_cosmologies(Mh_stds,Om0Yang,Om0)
                
                ##### need to convert Mh to Mvir
                Mvir_guess,rvir_guess,cvir_guess=mass_adv.changeMassDefinitionCModel(Mh_meds,zL_guess,mdef_in='180m',mdef_out='vir')
                Mvir_lo_guess,_,_=mass_adv.changeMassDefinitionCModel(Mh_los,zL_guess,mdef_in='180m',mdef_out='vir')
                Mvir_hi_guess,_,_=mass_adv.changeMassDefinitionCModel(Mh_his,zL_guess,mdef_in='180m',mdef_out='vir')
                
                Rvir_guess=rvir_guess/1e3 #[r]=Mpc/h from kpc/h
                
                M200m_guess,R200m_guess,c200m_guess=mass_adv.changeMassDefinitionCModel(Mh_meds,zL_guess,mdef_in='180m',mdef_out='200m')
                R200m_guess=R200m_guess/1e3 #[r]=Mpc/h from kpc/h
                loLim=loLimFactor*R200m_guess
                
                
                print('%s, loLimFactor_%d, %s, %s' %(file, int(10*loLimFactor), addTxt, addTxt2)) #results directory path
                    
                log10Mvir_guess=np.log10(Mvir_guess)
                z_guess=0.25344
                
                
                
                
                
                
                
                
                
                
                
                
                #fit
                fileName='fitProf_%s_loLimFactor_%d_hiLim_%d%s' %(file,int(10*loLimFactor),hiLim,addTxt)
                biasProfArray=np.load(saveDatDir+fileName+'.npy',allow_pickle=True)
                biasProf=biasProfArray.item()
                
                if flattenOuter is True:
                    biasProf.flattenOuter=True
                
                
                if loLim<=0.06:
                    R=np.logspace(np.log10(0.061),np.log10(30),100)
                else:
                    R=np.logspace(np.log10(loLim),np.log10(30),100)
                
                b_j=biasProf.bias(r=R)
                
                
                
                ##### get Rcds from DeltaSigma samples
                Rcd_med=Rcd_meds[lbin]
                Rcd_lo=Rcd_los[lbin]
                Rcd_hi=Rcd_his[lbin]
                residSample=np.vstack([Rcd_med-Rcd_lo,Rcd_hi-Rcd_med])
                # residRcd=max([Rcd-Rcd_lo,Rcd-Rcd_med])
                
                
                
                #prepare plotting characteristic radii
                radii=np.array([Rcd_med])
                radiiText=np.array([r'$r_{\rm cd}^{\rm samples}$'])
                markers=np.array(['*','s'])
                mews=np.array([2,2])
                resids=np.array([residSample])
                markersize=15
                alphaplts=[1,1]
                lrange=np.arange(radii.size)
                
                
                
                
                ########## PLOT DeltaSigma ##########
                color=next(ColorList)
                
                plotX=R
                plotY=b_j
                
                ax3.plot(plotX,plotY,color=color,
                         label=r'$%.2f_{%.2f}^{%.2f}: %.1f$' 
                         %(np.log10(Mvir_guess),np.log10(Mvir_lo_guess),np.log10(Mvir_hi_guess),np.log10(nLens)))
                
                #plot radii
                for l in lrange:
                    radl=radii[l]
                    minInd=bt.getClosestIndex(plotX,radl)
                    resid = resids[l]
                    ax3.errorbar(plotX[minInd], plotY[minInd], xerr=resid, marker=markers[l], 
                                 ms=markersize, color = color, mfc='none',
                                 markeredgewidth=mews[l], alpha=alphaplts[l])
                
            
                
            ax3.legend(loc='best',fontsize=8)
            
            plt.subplots_adjust(wspace=0,hspace=0)
            
            fileName='allBiasFromFit_loLimFactor_%d_hiLim_%d%s%s.png' %(int(10*loLimFactor),hiLim,addTxt,addTxt2)
            plt.savefig(savePlotDir+fileName)
            
            fileName='%s_allBiasFromFit_loLimFactor_%d_hiLim_%d%s%s.png' %(redshiftRange[:-1],int(10*loLimFactor),hiLim,addTxt,addTxt2)
            paperPlotDir='/Users/MattFong/Desktop/Projects/Code/SJTU/DECaLS/plots/_paper/emcee/'
            plt.savefig(paperPlotDir+fileName)
            
    



