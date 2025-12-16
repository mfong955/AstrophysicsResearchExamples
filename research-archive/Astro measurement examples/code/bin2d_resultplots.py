#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:12:40 2021

plot resulting plots

@author: MattFong
"""
import sys,os,itertools
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from scipy.optimize import curve_fit
import matplotlib 
matplotlib.rcParams.update({'font.size': 22,'figure.figsize': (10,10)})

import time

sys.path.append(os.path.abspath('/Users/MattFong/Desktop/Projects/Code/SJTU'))
import biasTools_z as bt

from colossus.utils import constants
from colossus.halo import mass_adv
from colossus.halo import mass_so


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

Ms_long=10**np.linspace(11,15,5000) #[M]=Msun/h
nus_long=peaks.peakHeight(Ms_long,0.25344) #from sims z_guess=0.25344

cosmoYang2020=ccosmo.setCosmology('planck18-only')
Om0Yang=cosmoYang2020.Om0
tictot=time.time()
#data directory



diffLensCatStuff=['z_02_03_binMh_hiReff/','z_02_03_binMh_loReff/']


##### Choose redshift range for data
# redshiftRanges=['z_02_03/','z_02_03_binMh/','z_02_03_binMh_richness3/','z_02_03_binMh_richness5/']
redshiftRanges=['z_02_03_binMh_hiReff/','z_02_03_binMh_loReff/']


col3=True #for old results, without accurate redshift estimation
col3=False
if col3:
    redshiftRanges=['col3_z_02_03/','col3_z_02_03_binMh/']


# redshiftRanges=['z_02_03_binMh/']


# piecewise=True
piecewise=False #keep false
if piecewise is True:
    pwstr='_piecewise'
    print('%s' %pwstr)
else:
    pwstr=''

for redshiftRange in redshiftRanges:
    
    zlo=0.2
    zhi=0.3
    
    #data directory
    dataDir='/Users/MattFong/Desktop/Projects/Data/halo_prof/'
    
    dataDir_DeltaSigma=dataDir+'DeltaSigma/'
    
    dataDir_DECALS='/Users/MattFong/Desktop/Projects/Data/DECaLS/group_cluster/DESI_Yang2020/'
    
    saveDatDir=dataDir_DECALS+'results/'+redshiftRange
    
    # emcee directory 
    Rx_dataDir='/Users/MattFong/Desktop/Projects/Code/SJTU/DECaLS/plots/emcee_DeltaSigma_getRcds/'
    Rx_dataDir=Rx_dataDir+redshiftRange
    
    
    savePlotDir='/Users/MattFong/Desktop/Projects/Code/SJTU/DECaLS/plots/'+redshiftRange
    dataDir_lensIDs=dataDir_DECALS+redshiftRange
    # dataDir_DECALS=dataDir_lensIDs+'results/'
    
    dataDir_nfw=saveDatDir
    
    plot_meanExcessDensity=False #Keep False
    print('============== plotting ONLY meanExcessDensity: ', plot_meanExcessDensity)
    if plot_meanExcessDensity:
        savePlotDir=savePlotDir+'meanExcessDensity/'
        Rx_dataDir=Rx_dataDir+'meanExcessDensity/'
        
    # fitUsingCurvefit=False #keep False for emcee ones. this just changes the saving directory anyway
    # print('============== plotting fitUsingCurvefit: ', fitUsingCurvefit)
    # if fitUsingCurvefit:
    #     savePlotDir=savePlotDir+'fitUsingCurveFit/'
    #     Rx_dataDir=Rx_dataDir+'fitUsingCurveFit/'
        
    if not os.path.exists(savePlotDir):
        os.makedirs(savePlotDir)
    
    print('***** REDSHIFT RANGE DIRECTORY: %s' %redshiftRange)
    zRange=[0.2,0.3]
    
    
    
    # if you want to use the exclusion radius (2.5*Rvir ~ Rcd)
    exclusion=False
    # exclusion=True #uses exclusion radius
    
    #[snapshot, redshift]
    if exclusion:
        exclStr='_exclude'
        print('###***Including halo exclusion!!!***###')
    else:
        exclStr=''
    
    
    Mvir_Xiaohus=[]
    Mvir_WLs=[]
    
    ##### Choose fit limits
    hiLim=30 #Mpc/h
    if redshiftRange in diffLensCatStuff:
        loLimFactors=[0.3,0.5]
    else:
        loLimFactors=[0.3,0.5,0.7]
    for loLimFactor in loLimFactors:
        
        
        ##### Rx and Mx from observation fits!!!
        '''
        RxMxData=np.load(saveDatDir+'RxMxData_loLimFactor_%d_hiLim_%d%s.npy' %(int(10*loLimFactor),hiLim,pwstr))
        # RxMxData.append([lbin,Rcd_med,Rcd_lo,Rcd_hi,Rsp,Mcd_med,Mcd_lo,Mcd_hi,Msp])
        
        RxMxData=RxMxData[RxMxData[:,0].argsort()] #sort out the data according to lbin
        
        lbin=RxMxData[:,0]
        Rcd_med=RxMxData[:,1]
        Rcd_lo=RxMxData[:,2]
        Rcd_hi=RxMxData[:,3]
        Rsp_med=RxMxData[:,4]
        
        Mcd_med=RxMxData[:,5]
        Mcd_lo=RxMxData[:,6]
        Mcd_hi=RxMxData[:,7]
        Msp_med=RxMxData[:,8]
        '''
        RxMxStats=np.load(saveDatDir+'RxMxStats_loLimFactor_%d_hiLim_%d.npy' %(int(10*loLimFactor),hiLim))
        # RxMxStats.append([Rcds_med, Rcds_lo, Rcds_hi, Rsps_med, Rsps_lo, Rsps_hi, 
        #                       Deltacds_z_med, Deltacds_z_lo, Deltacds_z_hi, 
        #                       Deltasps_z_med, Deltasps_z_lo, Deltasps_z_hi,
        #                       Deltacds_z0_med, Deltacds_z0_lo, Deltacds_z0_hi, 
        #                       Deltasps_z0_med, Deltasps_z0_lo, Deltasps_z0_hi])
        Rcd_med=RxMxStats[:,0]
        Rcd_lo=RxMxStats[:,1]
        Rcd_hi=RxMxStats[:,2]
        Rsp_med=RxMxStats[:,3]
        Rsp_lo=RxMxStats[:,4]
        Rsp_hi=RxMxStats[:,5]
        Deltacd_z_med=RxMxStats[:,6]
        Deltacd_z_lo=RxMxStats[:,7]
        Deltacd_z_hi=RxMxStats[:,8]
        Deltasp_z_med=RxMxStats[:,9]
        Deltasp_z_lo=RxMxStats[:,10]
        Deltasp_z_hi=RxMxStats[:,11]
        Deltacd_z0_med=RxMxStats[:,12]
        Deltacd_z0_lo=RxMxStats[:,13]
        Deltacd_z0_hi=RxMxStats[:,14]
        Deltasp_z0_med=RxMxStats[:,15]
        Deltasp_z0_lo=RxMxStats[:,16]
        Deltasp_z0_hi=RxMxStats[:,17]
                              
        ##### Rsmins from catalog!!!!
        lensCatIDs=np.load(dataDir_lensIDs+'lensCatIDs.npy')
        
        lensCatIDs=lensCatIDs[lensCatIDs[:,0].argsort()] #sort out data according to lensIDs
        
        
        if redshiftRange in diffLensCatStuff:
            lensIDs=lensCatIDs[:,0]
            nLens=lensCatIDs[:,1]
            Lambda_meds=lensCatIDs[:,2]
            Lambda_los=lensCatIDs[:,3]
            Lambda_his=lensCatIDs[:,4]
            logMh_meds=lensCatIDs[:,5]
            logMh_los=lensCatIDs[:,6]
            logMh_his=lensCatIDs[:,7]
            Lambda_means=lensCatIDs[:,8]
            Lambda_stds=lensCatIDs[:,9]
            logMh_means=lensCatIDs[:,10]
            logMh_stds=lensCatIDs[:,11]
            L_meds=lensCatIDs[:,12]
            L_los=lensCatIDs[:,13]
            L_his=lensCatIDs[:,14]
            L_means=lensCatIDs[:,15]
            L_stds=lensCatIDs[:,16]
            Reff_meds=lensCatIDs[:,17]
            Reff_los=lensCatIDs[:,18]
            Reff_his=lensCatIDs[:,19]
            Reff_means=lensCatIDs[:,20]
            Reff_stds=lensCatIDs[:,21]
            Leff_meds=lensCatIDs[:,22]
            Leff_los=lensCatIDs[:,23]
            Leff_his=lensCatIDs[:,24]
            Leff_means=lensCatIDs[:,25]
            Leff_stds=lensCatIDs[:,26]
            Neff_meds=lensCatIDs[:,27]
            Neff_los=lensCatIDs[:,28]
            Neff_his=lensCatIDs[:,29]
            Neff_means=lensCatIDs[:,30]
            Neff_stds=lensCatIDs[:,31]
        else:
            lensIDs=lensCatIDs[:,0]
            nLens=lensCatIDs[:,1]
            Lambda_meds=lensCatIDs[:,2]
            Lambda_los=lensCatIDs[:,3]
            Lambda_his=lensCatIDs[:,4]
            logMh_meds=lensCatIDs[:,5]
            logMh_los=lensCatIDs[:,6]
            logMh_his=lensCatIDs[:,7]
            Lambda_means=lensCatIDs[:,8]
            Lambda_stds=lensCatIDs[:,9]
            logMh_means=lensCatIDs[:,10]
            logMh_stds=lensCatIDs[:,11]
        
        # lensID,nLens,Lambda_meds,Lambda_his,Lambda_los,logMh_meds,logMh_his,logMh_los,Lambda_means,Lambda_stds,logMh_means,logMh_stds=lensCatIDs
        Mh_medsi=10**logMh_meds
        Mh_hisi=10**logMh_his
        Mh_losi=10**logMh_los
        Mh_meansi=10**logMh_means
        Mh_stdsi=10**logMh_stds
        
        #convert masses to our cosmology?
        Mh_meds=convertMass_cosmologies(Mh_medsi)
        Mh_los=convertMass_cosmologies(Mh_losi)
        Mh_his=convertMass_cosmologies(Mh_hisi)
        Mh_means=convertMass_cosmologies(Mh_meansi)
        Mh_stds=convertMass_cosmologies(Mh_stdsi)
        
        ##### need to convert Mh to Mvir
        # Mvir_data_lo,rvir_data_lo,cvir_data_lo=mass_adv.changeMassDefinitionCModel(Mh_meds,zlo,mdef_in='180m',mdef_out='vir')
        # Mvir_data_hi,rvir_data_hi,cvir_data_hi=mass_adv.changeMassDefinitionCModel(Mh_meds,zhi,mdef_in='180m',mdef_out='vir')
        z_guess=0.25344
        Mvir_guess,rvir_guess,cvir_guess=mass_adv.changeMassDefinitionCModel(Mh_meds,z_guess,mdef_in='180m',mdef_out='vir')
        Mvir_lo_guess,_,_=mass_adv.changeMassDefinitionCModel(Mh_los,z_guess,mdef_in='180m',mdef_out='vir')
        Mvir_hi_guess,_,_=mass_adv.changeMassDefinitionCModel(Mh_his,z_guess,mdef_in='180m',mdef_out='vir')
        
        #[snapshot, redshift]
        if exclusion:#we only have a few snapshots of this
            redshiftBins=np.array([[91, 0.25344]]) #for excluded
        else:
            redshiftBins=np.array([[90,0.289063],[91, 0.25344], [92, 0.218258]])
        
        zs=redshiftBins[:,1]
        
        
        # loLimFactorNFW=0.3 #Keep fixed! These mass results most match with Xiaohu's estimates
        nfwFitPar=np.load(dataDir_nfw+'my_nfwFitPar_loLimFactor_%d.npy' %(int(10*loLimFactor)))
        # nfwFitPar.append([lbin,logMvir_fit,logMvir_std,Rvir_fit, Rvir_std])
        lbinnfw=nfwFitPar[:,0]
        logMvir_fit=nfwFitPar[:,1]
        logMvir_std=nfwFitPar[:,2]
        Rvir_fit=nfwFitPar[:,3]
        Rvir_std=nfwFitPar[:,4]
        
        Mvir_fit=10**logMvir_fit
        Mvir_std=10**logMvir_std
        
        
        Mvir_Xiaohus.append(Mvir_guess)
        Mvir_WLs.append(Mvir_fit)
        
        titleText=r'%s   loLimFactor=%.1f' %(redshiftRange,loLimFactor) 
        titleText+="\n" +r'$z_{\rm sim} \in (%.2f, %.2f)$' %(zs.min(), zs.max())
        if exclusion:
            titleText=titleText+' using Exclusion Radius'
        print(titleText)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        alphahline=0.1
        
        radiiLabels=np.array([r'$r_{\rm cd}$',r'$r_{\rm sp}$',r'$r_{\rm vir}$'])
        colors=['r','m','b']
        lineStyles=['-','-.',':']
        lws=[2,2,2]
        
        alphas=[1,1,1]

        radii=np.vstack([Rcd_med,Rsp_med,Rvir_fit])
        # masses=np.vstack([Mcd_med,Msp_med,Mvir_fit])
        deltas_z=np.vstack([Deltacd_z_med,Deltasp_z_med])
        deltas_z0=np.vstack([Deltacd_z0_med,Deltasp_z0_med])
        
        radii_lo=np.vstack([Rcd_med-Rcd_lo,Rsp_med-Rsp_lo,Rvir_fit-Rvir_fit]) #set Mvir errors to 0
        radii_hi=np.vstack([Rcd_hi-Rcd_med,Rsp_hi-Rsp_med,Rvir_fit-Rvir_fit])
        
        deltas_z_lo=np.vstack([Deltacd_z_med-Deltacd_z_lo,Deltasp_z_med-Deltasp_z_lo])
        deltas_z_hi=np.vstack([Deltacd_z_hi-Deltacd_z_med,Deltasp_z_hi-Deltasp_z_med])
        
        deltas_z0_lo=np.vstack([Deltacd_z0_med-Deltacd_z0_lo,Deltasp_z0_med-Deltasp_z0_lo])
        deltas_z0_hi=np.vstack([Deltacd_z0_hi-Deltacd_z0_med,Deltasp_z0_hi-Deltasp_z0_med])
        
        
        
        #Rbs,Rsps,Rtas,Rvtot_mins,Rvp_mins,Rvds,Rvirs,RG20s,Rsp1s,Rsp2s,Rmfrs_Mvir
        icds=0
        isps=1
        ivirs=2

        # lbin=RxMxData[:,0]
        # Rcd_med=RxMxData[:,1]
        # Rcd_lo=RxMxData[:,2]
        # Rcd_hi=RxMxData[:,3]
        # Rsp=RxMxData[:,4]
        
        # Mcd_med=RxMxData[:,5]
        # Mcd_lo=RxMxData[:,6]
        # Mcd_hi=RxMxData[:,7]
        # Msp=RxMxData[:,8]
        
        
        
        
        
        
        
        
        
        
        
        ##### plot Rcd vs Rsp
        plt.figure()
        plt.title(titleText)
        
        capsize=3
        
        # Rcd
        Yhi=radii_hi[0]
        Ylo=radii_lo[0]
        Ymed=radii[0]
        Yerr=np.vstack([Ylo,Yhi])
        
        # Rsp
        Xhi=radii_hi[1]
        Xlo=radii_lo[1]
        Xmed=radii[1]
        Xerr=np.vstack([Xlo,Xhi])
        
        # Ymed=Ymed[Ymed>0]
        
        plt.errorbar(Xmed,Ymed,xerr=Xerr,yerr=Yerr,
                     ls='',marker='*',ms=10,color='r',capsize=capsize)
        
        # plot y=2x and y=3x
        plt.plot(Xmed,2*Xmed,color='k',lw=2,ls='-',label=r'$r_{\rm cd}=2.0r_{\rm sp}$')
        plt.plot(Xmed,3*Xmed,color='k',lw=2,ls='--',label=r'$r_{\rm cd}=3.0r_{\rm sp}$')
        
        
        
        
        
        
        
        
        
        
        
        
        
        lineStyles=['-','-.',':']
        for redbini in np.arange(redshiftBins.shape[0]):
            snapshot=redshiftBins[redbini][0]
            z=redshiftBins[redbini][1]
            
            saveBinnedDir=dataDir+'binnedResults/'
            radiisim=np.load(file=saveBinnedDir+'radii_snapshot%d.npy' %snapshot)
            massessim=np.load(file=saveBinnedDir+'masses_snapshot%d.npy' %snapshot)
            # Rcds,Rsps,Rtas,Rvirs=radii
            # Mcds,Msps,Mtas,Mvirs=masses
            icds=0
            isps=1
            # itas=2
            ivirs=2
            # ivirs=3
            
            
            radiiLabels=[r'$r_{\rm cd}$',r'$r_{\rm sp}$',r'$r_{\rm vir}$']#r'$r_{\rm ta}$',r'$r_{\rm vir}$']
            colors=['r','m','b']#'c', 'b']
            # lineStyles=['-','-.',':']
            lws=[2,2,2]
            alphas=[1,1,1]
            
            iplots=[icds,isps,ivirs]#itas,ivirs]
            # plt.figure()
            
            Ys=radiisim[icds]
            Xs=radiisim[isps]
            
            plt.plot(Xs,Ys,color='k',lw=2,alpha=0.1,ls=lineStyles[redbini])#,label=r'z=%.2f' %z)
            
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$r_{\rm sp} \, [{\rm Mpc}/h]$')
        plt.ylabel(r'$r_{\rm cd} \, [{\rm Mpc}/h]$')
        
        plt.legend(loc='best')
        
        fileName='RcdVsRsp_loLimFactor_%d_hiLim_%d' %(int(10*loLimFactor),hiLim)
        fileName=fileName+exclStr+'.png'
        plt.savefig(savePlotDir+fileName, bbox_inches='tight')
        
        fileName='%s_RcdVsRsp_loLimFactor_%d_hiLim_%d.png' %(redshiftRange[:-1],int(10*loLimFactor),hiLim)
        paperPlotDir='/Users/MattFong/Desktop/Projects/Code/SJTU/DECaLS/plots/_paper/emcee/'
        plt.savefig(paperPlotDir+fileName)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ##### plot Deltax vs. Mvir (z)
        

        
        def MxRx2Delta(Mx,Rx,rhox):
            #calculate enclosed density contrast
            #rhox is the density of the Universe, e.g rho_c or rho_m
            #[rhox]=M_dot h^2 / Mpc^3
            return 3.*Mx/(4.*np.pi*(Rx**3))/rhox
        def hline(X,b):
            return b
        def fit_hline(X,Y):
            pfit=curve_fit(f=hline,xdata=X,ydata=Y)[0]
            return pfit
        


        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.set_title(titleText)
        
        
        
        
        
        
        
        
        
        
        
        lineStylesrbin=['-','-.',':']
        for redbini in np.arange(redshiftBins.shape[0]):
            snapshot=redshiftBins[redbini][0]
            z=redshiftBins[redbini][1]
            
            # mean matter density of universe
            rhom = cosmo.rho_m(z)*(1e3)**3. #[rhom] = M_dot h^2 / Mpc^3 from M_dot h^2 / kpc^3
            
            saveBinnedDir=dataDir+'binnedResults/'
            radiisim=np.load(file=saveBinnedDir+'radii_snapshot%d.npy' %snapshot)
            massessim=np.load(file=saveBinnedDir+'masses_snapshot%d.npy' %snapshot)
            # Rcds,Rsps,Rtas,Rvirs=radii
            # Mcds,Msps,Mtas,Mvirs=masses
            icds=0
            isps=1
            # itas=2
            ivirs=2
            # ivirs=3
            
            
            radiiLabels=[r'$r_{\rm cd}$',r'$r_{\rm sp}$',r'$r_{\rm vir}$']#r'$r_{\rm ta}$',r'$r_{\rm vir}$']
            colors=['r','m','b']#'c', 'b']
            # lineStyles=['-','-.',':']
            lws=[2,2,2]
            alphas=[1,1,1]
            
            iplots=[icds,isps]#itas,ivirs]
            # plt.figure()
            
            '''
            for ix in iplots:
                Rx=radiisim[ix]
                Mx=massessim[ix]
                
                RxSel=Rx>0
                MvirsPlot=Mvir_fit[Rx>0]
                
                Mx=Mx[RxSel]
                Rx=Rx[RxSel]
                Deltax=MxRx2Delta(Mx=Mx,Rx=Rx,rhox=rhom)
                
                plt.plot(MvirsPlot, Deltax, color='k', 
                         ls=lineStyles[redbini], alpha=0.1,lw=2)
            '''
                
                
                
        ax.axhline(40,ls=':',color='k',alpha=0.2)
        
        iplots=[icds,isps]
        ifits=[icds,isps]
        
        # Deltacd_lo=MxRx2Delta(Mx=Mcd_lo,Rx=Rcd_lo,rhox=rhom)
        # Deltacd_hi=MxRx2Delta(Mx=Mcd_hi,Rx=Rcd_hi,rhox=rhom)
        
        
        # mean matter density of universe
        rhom = cosmo.rho_m(z_guess)*(1e3)**3. #[rhom] = M_dot h^2 / Mpc^3 from M_dot h^2 / kpc^3
        
        
        for ix in iplots:
            Rx=radii[ix]
            MvirsPlot=Mvir_fit
            MvirsPlot_std=Mvir_std
            '''
            Mx=masses[ix]
            
            RxSel=Rx>0
            MvirsPlot=Mvir_fit[Rx>0]
            
            Mx=Mx[RxSel]
            Rx=Rx[RxSel]
            Deltax=MxRx2Delta(Mx=Mx,Rx=Rx,rhox=rhom)
            '''
            Deltax=deltas_z[ix]
            Deltax_lo=deltas_z_lo[ix]
            Deltax_hi=deltas_z_hi[ix]
            
            dsel=Deltax>0
            MvirsPlot=MvirsPlot[dsel]
            MvirsPlot_std=MvirsPlot_std[dsel]
            Deltax=Deltax[dsel]
            Deltax_lo=Deltax_lo[dsel]
            Deltax_hi=Deltax_hi[dsel]
            
            plt.plot(MvirsPlot, Deltax, color=colors[ix], label=r'%s' %radiiLabels[ix], 
                      ls=lineStyles[ix], alpha=alphas[ix],lw=lws[ix])
            
            MvirsPlot_err=Mvir_std
            Deltax_err=np.vstack([Deltax_lo,Deltax_hi])
            
            # plt.errorbar(MvirsPlot, Deltax, xerr=MvirsPlot_err,yerr=Deltax_err,
            #              color=colors[ix], label=r'%s' %radiiLabels[ix], 
            #              ls=lineStyles[ix], alpha=alphas[ix],lw=lws[ix])
            
            plt.fill_between(x=MvirsPlot,y1=Deltax-Deltax_lo,y2=Deltax_hi+Deltax,color=colors[ix],alpha=0.5)
            
            if ix in ifits:
                # if ix == isps:
                #     ratiofit=fit_hline(X=MvirsPlot[1:],Y=Deltax[1:])
                #     plt.text(x=1.25e15,y=ratiofit*0.95,s='%d' %ratiofit)
                #     plt.axhline(ratiofit,color='k',alpha=alphahline,ls='--')
                # else:
                ratiofit=fit_hline(X=MvirsPlot,Y=Deltax)
                plt.text(x=1.25e15,y=ratiofit*0.95,s='%.1f' %ratiofit)
                plt.axhline(ratiofit,color='k',alpha=alphahline)
            
            print('rhom(z=0.25):')
            print('Deltax=Deltax[2:] : ' +radiiLabels[ix])
            Deltax=Deltax[2:]
            print('(Deltax.mean(),Deltax.std()):(%.2f,%.2f)' %(Deltax.mean(),Deltax.std()))
        # yticks=[0.4,0.6,0.8,1.0,1.2,1.4,1.6]
        # ax.set_yticks(yticks)
        # ax.set_yticklabels(yticks)
        plt.legend(loc='best',fontsize=18)
        plt.ylabel(r'$\Delta=\rho(<r_{X})/\rho_{\rm m}(z)$')
        plt.xlabel(r'$M_{\rm vir} \, [{\rm M_{\odot}}/h]$')
        plt.xscale('log')
        plt.yscale('log')
        yticks=[40,50,100,200,400]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        # plt.grid(b=True, which='both', lw=0.1, color='k', ls='--')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        
        fileName='DeltaVsMvir_loLimFactor_%d_hiLim_%d%s' %(int(10*loLimFactor),hiLim,pwstr)
        fileName=fileName+exclStr+'.png'
        plt.savefig(savePlotDir+fileName, bbox_inches='tight')
        
        fileName='%s_DeltaVsMvir_loLimFactor_%d_hiLim_%d%s.png' %(redshiftRange[:-1],int(10*loLimFactor),hiLim,pwstr)
        paperPlotDir='/Users/MattFong/Desktop/Projects/Code/SJTU/DECaLS/plots/_paper/emcee/'
        plt.savefig(paperPlotDir+fileName)
        
        
        
        
        
        
        
        
        
        
        ##### plot Deltax vs. Mvir (z=0)
        
        # mean matter density of universe
        rhom = cosmo.rho_m(z=0)*(1e3)**3. #[rhom] = M_dot h^2 / Mpc^3 from M_dot h^2 / kpc^3


        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.set_title(titleText)
        
        
        
        ax.axhline(40,ls=':',color='k',alpha=0.2)
        
        
        lineStylesrbin=['-','-.',':']
        for redbini in np.arange(redshiftBins.shape[0]):
            snapshot=redshiftBins[redbini][0]
            z=redshiftBins[redbini][1]
            
            saveBinnedDir=dataDir+'binnedResults/'
            radiisim=np.load(file=saveBinnedDir+'radii_snapshot%d.npy' %snapshot)
            massessim=np.load(file=saveBinnedDir+'masses_snapshot%d.npy' %snapshot)
            # Rcds,Rsps,Rtas,Rvirs=radii
            # Mcds,Msps,Mtas,Mvirs=masses
            icds=0
            isps=1
            # itas=2
            ivirs=2
            # ivirs=3
            
            
            radiiLabels=[r'$r_{\rm cd}$',r'$r_{\rm sp}$',r'$r_{\rm vir}$']#r'$r_{\rm ta}$',r'$r_{\rm vir}$']
            colors=['r','m','b']#'c', 'b']
            # lineStyles=['-','-.',':']
            lws=[2,2,2]
            alphas=[1,1,1]
            
            iplots=[icds,isps]#itas,ivirs]
            # plt.figure()
            
            '''
            for ix in iplots:
                Rx=radiisim[ix]
                Mx=massessim[ix]
                
                RxSel=Rx>0
                MvirsPlot=Mvir_fit[Rx>0]
                
                Mx=Mx[RxSel]
                Rx=Rx[RxSel]
                Deltax=MxRx2Delta(Mx=Mx,Rx=Rx,rhox=rhom)
                
                plt.plot(MvirsPlot, Deltax, color='k', 
                         ls=lineStyles[redbini], alpha=0.1,lw=2)
            '''
                
                
                
        iplots=[icds,isps]
        ifits=[icds,isps]
        
        # Deltacd_lo=MxRx2Delta(Mx=Mcd_lo,Rx=Rcd_lo,rhox=rhom)
        # Deltacd_hi=MxRx2Delta(Mx=Mcd_hi,Rx=Rcd_hi,rhox=rhom)
        # plt.fill_between(x=Mvir_fit,y1=Deltacd_lo,y2=Deltacd_hi,color='r',alpha=0.5)
        
        for ix in iplots:
            Rx=radii[ix]
            MvirsPlot=Mvir_fit
            MvirsPlot_std=Mvir_std
            '''
            Mx=masses[ix]
            
            RxSel=Rx>0
            MvirsPlot=Mvir_fit[Rx>0]
            
            Mx=Mx[RxSel]
            Rx=Rx[RxSel]
            Deltax=MxRx2Delta(Mx=Mx,Rx=Rx,rhox=rhom)
            '''
            Deltax=deltas_z0[ix]
            Deltax_lo=deltas_z0_lo[ix]
            Deltax_hi=deltas_z0_hi[ix]
            
            
            dsel=Deltax>0
            MvirsPlot=MvirsPlot[dsel]
            MvirsPlot_std=MvirsPlot_std[dsel]
            Deltax=Deltax[dsel]
            Deltax_lo=Deltax_lo[dsel]
            Deltax_hi=Deltax_hi[dsel]
            
            plt.plot(MvirsPlot, Deltax, color=colors[ix], label=r'%s' %radiiLabels[ix], 
                      ls=lineStyles[ix], alpha=alphas[ix],lw=lws[ix])
            
            
            Deltax_err=np.vstack([Deltax_lo,Deltax_hi])
            
            # plt.errorbar(MvirsPlot, Deltax, xerr=MvirsPlot_std,yerr=Deltax_err,
            #              color=colors[ix], label=r'%s' %radiiLabels[ix], 
            #              ls=lineStyles[ix], alpha=alphas[ix],lw=lws[ix])
            
            plt.fill_between(x=MvirsPlot,y1=Deltax-Deltax_lo,y2=Deltax_hi+Deltax,color=colors[ix],alpha=0.5)
            
            if ix in ifits:
                # if ix == isps:
                #     ratiofit=fit_hline(X=MvirsPlot[1:],Y=Deltax[1:])
                #     plt.text(x=1.25e15,y=ratiofit*0.95,s='%d' %ratiofit)
                #     plt.axhline(ratiofit,color='k',alpha=alphahline,ls='--')
                # else:
                ratiofit=fit_hline(X=MvirsPlot,Y=Deltax)
                plt.text(x=1.25e15,y=ratiofit*0.95,s='%.1f' %ratiofit)
                plt.axhline(ratiofit,color='k',alpha=alphahline)
            
            print('rhom(z=0):')
            print('Deltax=Deltax[2:] : ' +radiiLabels[ix])
            Deltax=Deltax[2:]
            print('(Deltax.mean(),Deltax.std()):(%.2f,%.2f)' %(Deltax.mean(),Deltax.std()))
        # yticks=[0.4,0.6,0.8,1.0,1.2,1.4,1.6]
        # ax.set_yticks(yticks)
        # ax.set_yticklabels(yticks)
        plt.legend(loc='best',fontsize=18)
        plt.ylabel(r'$\Delta=\rho(<r_{X})/\rho_{\rm m}(z=0)$')
        plt.xlabel(r'$M_{\rm vir} \, [{\rm M_{\odot}}/h]$')
        plt.xscale('log')
        plt.yscale('log')
        yticks=[40,50,100,200,400]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        # plt.grid(b=True, which='both', lw=0.1, color='k', ls='--')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        
        fileName='DeltaVsMvir_z0_loLimFactor_%d_hiLim_%d%s' %(int(10*loLimFactor),hiLim,pwstr)
        fileName=fileName+exclStr+'.png'
        plt.savefig(savePlotDir+fileName, bbox_inches='tight')
        
        fileName='%s_DeltaVsMvir_z0_loLimFactor_%d_hiLim_%d%s.png' %(redshiftRange[:-1],int(10*loLimFactor),hiLim,pwstr)
        paperPlotDir='/Users/MattFong/Desktop/Projects/Code/SJTU/DECaLS/plots/_paper/emcee/'
        plt.savefig(paperPlotDir+fileName)
        
        
        
        
        
        
        
        
        
        
        
        
        ##### plot Rx vs. Mvir
        fig, ax = plt.subplots()
        ax.set_title(titleText)
        
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        
        
        
        
        
        lineStyles=['-','-.',':']
        for redbini in np.arange(redshiftBins.shape[0]):
            snapshot=redshiftBins[redbini][0]
            z=redshiftBins[redbini][1]
            
            saveBinnedDir=dataDir+'binnedResults/'
            radiisim=np.load(file=saveBinnedDir+'radii_snapshot%d.npy' %snapshot)
            massessim=np.load(file=saveBinnedDir+'masses_snapshot%d.npy' %snapshot)
            # Rcds,Rsps,Rtas,Rvirs=radii
            # Mcds,Msps,Mtas,Mvirs=masses
            icds=0
            isps=1
            # itas=2
            ivirs=2
            # ivirs=3
            
            
            radiiLabels=[r'$r_{\rm cd}$',r'$r_{\rm sp}$',r'$r_{\rm vir}$']#r'$r_{\rm ta}$',r'$r_{\rm vir}$']
            colors=['r','m','b']#'c', 'b']
            # lineStyles=['-','-.',':']
            lws=[2,2,2]
            alphas=[1,1,1]
            
            iplots=[icds]#itas,ivirs]
            # plt.figure()
            
            for ix in iplots:
                
                RxSim=radiisim[ix]
                MvirSim=massessim[ivirs]
                
                plt.plot(MvirSim,RxSim,color='k',lw=2,alpha=0.1,ls=lineStyles[redbini])#,label=r'z=%.2f' %z)
            
            
        
        
        
        
        
        
        iplots=[icds,isps,ivirs]
        
        plt.plot(Mvir_fit,2.5*Rvir_fit,ls=':',color='k')
        for ix in iplots:
            Rx=radii[ix]
            # Mx=masses[ix]
            
            MvirsPlot=Mvir_fit[Rx>0]
            MvirsPlot_std=Mvir_std[Rx>0]
            
            Rhi=radii_hi[ix]
            Rlo=radii_lo[ix]
            
            Rerr=np.vstack([Rlo[Rx>0],Rhi[Rx>0]])
            
            Rx=Rx[Rx>0]
            if ix == ivirs:
                plt.errorbar(MvirsPlot,Rx, #no errorbars
                             label=r'%s' %radiiLabels[ix],ls=lineStyles[ix],
                             alpha=alphas[ix],lw=lws[ix],color=colors[ix],capsize=capsize)
            else:
                plt.errorbar(MvirsPlot,Rx,xerr=MvirsPlot_std,yerr=Rerr,
                             label=r'%s' %radiiLabels[ix],ls=lineStyles[ix],
                             alpha=alphas[ix],lw=lws[ix],color=colors[ix],capsize=capsize)
        plt.legend(loc='best',fontsize=20)
        plt.ylabel(r'$r_{\rm X} \, [{\rm Mpc}/h]$')
        plt.xlabel(r'$M_{\rm vir} \, [{\rm M_{\odot}}/h]$')
        plt.xscale('log')
        plt.yscale('log')
        yticks=[0.5,1,5]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        # plt.grid(b=True, which='both', lw=0.1, color='k', ls='--')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        fileName='RxVsMvir_loLimFactor_%d_hiLim_%d' %(int(10*loLimFactor),hiLim)
        fileName=fileName+exclStr+'.png'
        plt.savefig(savePlotDir+fileName, bbox_inches='tight')
        
        fileName='%s_RxVsMvir_loLimFactor_%d_hiLim_%d.png' %(redshiftRange[:-1],int(10*loLimFactor),hiLim)
        paperPlotDir='/Users/MattFong/Desktop/Projects/Code/SJTU/DECaLS/plots/_paper/emcee/'
        plt.savefig(paperPlotDir+fileName, bbox_inches='tight')
        
        
        
        toctot=time.time()
        print('time (s): %.2f' %(toctot-tictot))
        
        
        
        
        
        
        
        
        
        
        
        ##### plot Rcd vs Rsp SHIFT BY 0.87
        plt.figure()
        plt.title(titleText+' Rsps shifted by 0.87')
        
        capsize=3
        
        # Rcd
        Yhi=radii_hi[0]
        Ylo=radii_lo[0]
        Ymed=radii[0]
        Yerr=np.vstack([Ylo,Yhi])
        
        # Rsp
        Xhi=radii_hi[1]
        Xlo=radii_lo[1]
        Xmed=radii[1]
        Xerr=np.vstack([Xlo,Xhi])
        
        # Ymed=Ymed[Ymed>0]
        
        plt.errorbar(Xmed,Ymed,xerr=Xerr,yerr=Yerr,
                     ls='',marker='*',ms=10,color='r',capsize=capsize)
        
        # plot y=2x and y=3x
        plt.plot(Xmed,2*Xmed,color='k',lw=2,ls='-',label=r'$r_{\rm cd}=2.0r_{\rm sp}$')
        plt.plot(Xmed,3*Xmed,color='k',lw=2,ls='--',label=r'$r_{\rm cd}=3.0r_{\rm sp}$')
        
        
        
        
        
        
        
        
        
        
        
        
        
        lineStyles=['-','-.',':']
        for redbini in np.arange(redshiftBins.shape[0]):
            snapshot=redshiftBins[redbini][0]
            z=redshiftBins[redbini][1]
            
            saveBinnedDir=dataDir+'binnedResults/'
            radiisim=np.load(file=saveBinnedDir+'radii_snapshot%d.npy' %snapshot)
            massessim=np.load(file=saveBinnedDir+'masses_snapshot%d.npy' %snapshot)
            # Rcds,Rsps,Rtas,Rvirs=radii
            # Mcds,Msps,Mtas,Mvirs=masses
            icds=0
            isps=1
            # itas=2
            ivirs=2
            # ivirs=3
            
            
            radiiLabels=[r'$r_{\rm cd}$',r'$r_{\rm sp}$',r'$r_{\rm vir}$']#r'$r_{\rm ta}$',r'$r_{\rm vir}$']
            colors=['r','m','b']#'c', 'b']
            # lineStyles=['-','-.',':']
            lws=[2,2,2]
            alphas=[1,1,1]
            
            iplots=[icds,isps,ivirs]#itas,ivirs]
            # plt.figure()
            
            Ys=radiisim[icds]
            Xs=radiisim[isps]*0.87
            
            plt.plot(Xs,Ys,color='k',lw=2,alpha=0.1,ls=lineStyles[redbini])#,label=r'z=%.2f' %z)
            
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$r_{\rm sp} \, [{\rm Mpc}/h]$')
        plt.ylabel(r'$r_{\rm cd} \, [{\rm Mpc}/h]$')
        
        plt.legend(loc='best')
        
        fileName='RcdVsRsp_loLimFactor_%d_hiLim_%d_SHIFTBY077' %(int(10*loLimFactor),hiLim)
        fileName=fileName+exclStr+'.png'
        plt.savefig(savePlotDir+fileName, bbox_inches='tight')
        
        fileName='%s_RcdVsRsp_loLimFactor_%d_hiLim_%d_SHIFTBY077.png' %(redshiftRange[:-1],int(10*loLimFactor),hiLim)
        paperPlotDir='/Users/MattFong/Desktop/Projects/Code/SJTU/DECaLS/plots/_paper/emcee/'
        plt.savefig(paperPlotDir+fileName)
        
        
        
        
        
        
        
        
        
        
    Mvir_Xiaohus=np.array(Mvir_Xiaohus)
    Mvir_WLs=np.array(Mvir_WLs)
    ratio_Mvirs=Mvir_WLs/Mvir_Xiaohus
    
    plt.figure()
    for loLimi in range(Mvir_WLs.shape[0]):
        plt.plot(Mvir_WLs[loLimi],ratio_Mvirs[loLimi],label='%1f' %loLimFactors[loLimi])
    plt.xscale('log')
    plt.legend(loc='best')
    
    ratio_Mvirs05=ratio_Mvirs[1] #only care about 0.05 lower fit factor 
    print(ratio_Mvirs05)
    print('ratio_Mvirs05=Mvir_WLs/Mvir_Xiaohus [0.5 loLimFactor]: (ratio_Mvirs05.mean(),ratio_Mvirs05.std()): (%.2f,%.2f)' %(ratio_Mvirs05.mean(),ratio_Mvirs05.std()))
    




rsp_rvir_meas=radii[isps]/radii[ivirs]
print('(rsp_rvir_meas.mean(),rsp_rvir_meas.std()):(%.2f,%.2f)' %(rsp_rvir_meas.mean(),rsp_rvir_meas.std()))

rsp_rvir_sim_all=[]

for redbini in np.arange(redshiftBins.shape[0]):
    snapshot=redshiftBins[redbini][0]
    z=redshiftBins[redbini][1]
    
    saveBinnedDir=dataDir+'binnedResults/'
    radiisim=np.load(file=saveBinnedDir+'radii_snapshot%d.npy' %snapshot)
    massessim=np.load(file=saveBinnedDir+'masses_snapshot%d.npy' %snapshot)
    # Rcds,Rsps,Rtas,Rvirs=radii
    # Mcds,Msps,Mtas,Mvirs=masses
    icds=0
    isps=1
    # itas=2
    ivirs=2
    # ivirs=3
    
    rsp_rvir_sim=radiisim[isps]/radiisim[ivirs]
    print('(rsp_rvir_sim.mean(),rsp_rvir_sim.std()):(%.2f,%.2f)' %(rsp_rvir_sim.mean(),rsp_rvir_sim.std()))
    
    rsp_rvir_sim_all.append(rsp_rvir_sim)
    
rsp_rvir_sim_all=np.array(rsp_rvir_sim_all)
print('(rsp_rvir_sim_all.mean(),rsp_rvir_sim_all.std()):(%.2f,%.2f)' %(rsp_rvir_sim_all.mean(),rsp_rvir_sim_all.std()))


ratio=rsp_rvir_meas/rsp_rvir_sim_all.mean()
print('(ratio.mean(),ratio.std()):(%.2f,%.2f)' %(ratio.mean(),ratio.std()))






