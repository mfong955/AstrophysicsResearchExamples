#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:16:18 2021

test curve_fit with cov matrix fit
@author: MattFong
"""

import sys,os,itertools
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


##### choose redshift range for data
redshiftRanges=['z_02_03/','z_02_03_binMh/','z_02_03_binMh_richness3/','z_02_03_binMh_richness5/']
# redshiftRanges=['z_02_03/']

# redshiftRange='z_02_03_binMh/'


col3=True #for old results, without accurate redshift estimation
# col3=False #more accurate redshift estimates
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
    # savePlotDir=savePlotDir+'fitUsingCurveFit/'
    # saveDatDir=saveDatDir+'fitUsingCurveFit/'
    
    
    plot_meanExcessDensity=False #keep False!
    print('============== plotting ONLY meanExcessDensity: ', plot_meanExcessDensity)
    if plot_meanExcessDensity:
        savePlotDir=savePlotDir+'meanExcessDensity/'
        saveDatDir=saveDatDir+'meanExcessDensity/'
        
    if not os.path.exists(savePlotDir):
        os.makedirs(savePlotDir)
    if not os.path.exists(saveDatDir):
        os.makedirs(saveDatDir)
    
    print('***** REDSHIFT RANGE DIRECTORY: %s' %redshiftRange)
    zRange=[0.2,0.3]
    # if redshiftRange=='z_02_03/':
    #     zRange=[0.2,0.3]
    # if redshiftRange=='z_02_06/':
    #     zRange=[0.2,0.6]
    
    ##### new "corr_jack_signal.dat" from 
    #/lustre/home/acct-phyzj/phyzj-m31/mfong/DECaLS/cross_corr_Mar_DECALS/results/result_lensCat_3
    # columns: rbini, DeltaSigma, DeltaSigma_err, DeltaSigmaCross, DeltaSigmaCross_std, R, ngals
    
    directory_contents = [f.name for f in os.scandir(dataDir) if f.is_dir()]
    
    sortedDirContents=np.array([int(re.findall(r'\d+',file)[0]) for file in directory_contents])
    
    sortLensIDs=sortedDirContents.argsort()
    directory_contents=np.array(directory_contents)
    directory_contents=directory_contents[sortLensIDs]

    print(directory_contents)
    
    #limits on fit
    # loLim=0.06
    # loLim=0.2
    hiLim=30
    loLimFactors=[0.3,0.5,0.7]
    # loLimFactors=[0.7]
    
    for loLimFactor in loLimFactors:
        
        # loLims=[0.06,0.1,0.3,0.5]
        # for loLim in loLims:
        
        Rsmins=[]
        pfits_all=[]
        
        # for file in directory_contents[4:5]: #test
        for file in directory_contents:
            lbin=int(re.findall(r'\d+',file)[0]) #get lambda bin ID
            
            print('=====================================================')
            print(file) #results directory path
            
            z_guess=0.25344
            # z_guess=np.mean(zRange)
            
            lensData=lensCatIDs[lbin]
            # mixed up the order in the original saving code
            lensID,nLens,Lambda_meds,Lambda_los,Lambda_his,logMh_meds,logMh_los,logMh_his,Lambda_means,Lambda_stds,logMh_means,logMh_stds=lensData
            Mh_meds=10**logMh_meds
            Mh_his=10**logMh_his
            Mh_los=10**logMh_los
            Mh_means=10**logMh_means
            Mh_stds=10**logMh_stds
            #convert masses to our cosmology?
            def convertMass_cosmologies(Mx,Om_original,Om_new):
                #input [M]=M_sun/h
                #I coded this so it only converts from Xiaohu's cosmology to WMAPs
                hCorrection=cosmo.h/cosmoYang2020.h
                return Mx*(Om_new/Om_original)/hCorrection
            
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
            
            print('loLim=%.2g' %loLim)
            titleText=r'$z \in [%.1f,%.1f]$' %(zRange[0],zRange[1])+'\n'+'fit range [Mpc/h]: [%.2f,%d]' %(loLim,hiLim)
            print(titleText)
            
            plt.figure(figsize=(10,12))
            plt.title(titleText)
            plt.yticks(ticks=None,labels=None)
            ########## PLOT DeltaSigma ##########
            ax1=plt.subplot(3,1,1)
            ax1.tick_params(right='on', top='on', direction='in', which='both')
            ax1.set_ylabel(r'$\Delta \Sigma$')
            ax1.tick_params(axis='y')
            plt.setp(ax1.get_xticklabels(minor=True), visible=False)
            plt.setp(ax1.get_yticklabels(minor=True), visible=False)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            # yticks=[1,5,10,50,100]
            # ax1.set_yticks(yticks)
            # ax1.set_yticklabels(yticks)
            # ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            plt.setp(ax1.get_xminorticklabels(), visible=False)
            ax1.axvline(0.02,alpha=0.5,ls=':',color='grey')
            ax1.axvspan(0,0.06, alpha=0.1, color='grey')# ratio plots
            ax1.axvspan(10,50, alpha=0.1, color='grey')
            
            ax1.text(.75,.9, r'$(l_{\rm bin},n_{\rm lens},\lambda_{\rm 180m}^{\rm median}, {\rm log_{10}} M_{\rm vir}^{\rm estimated}):$', horizontalalignment='center', 
                            verticalalignment='center', fontsize=10, transform=ax1.transAxes)
            ax1.text(.75,.8, r'$(%d,%d,%d_{%d}^{%d},%.2f_{%.2f}^{%.2f}):$' 
                     %(lensID,nLens,Lambda_meds,Lambda_los,Lambda_his,np.log10(Mvir_guess),np.log10(Mvir_lo_guess),np.log10(Mvir_hi_guess)), 
                     horizontalalignment='center', verticalalignment='center', fontsize=10, transform=ax1.transAxes)
            
            ########## PLOT R*DeltaSigma ##########
            ax2=plt.subplot(3,1,2, sharex=ax1)
            ax2.tick_params(right='on', top='on', direction='in', which='both')
            ax2.set_ylabel(r'$r \times \Delta \Sigma \, [{\rm Mpc \times M_{\odot}} / {\rm pc^2}]$', fontsize=14)
            plt.setp(ax2.get_xticklabels(minor=True), visible=False)
            plt.setp(ax2.get_yticklabels(minor=True), visible=False)
            ax2.set_yticks([10,50,100])
            ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
            ax2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            # yticks=[1,5,10,50,100]
            # ax2.set_yticks(yticks)
            # ax2.set_yticklabels(yticks)
            # ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            plt.setp(ax2.get_xminorticklabels(), visible=False)
            ax2.axvline(0.02,alpha=0.5,ls=':',color='grey')
            ax2.axvspan(0,0.06, alpha=0.1, color='grey')# ratio plots
            ax2.axvspan(10,50, alpha=0.1, color='grey')
            
            ########## PLOT gamma ##########
            ax3=plt.subplot(3,1,3, sharex=ax1)
            ax3.tick_params(right='on', top='on', direction='in', which='both')
            ax3.set_xlabel(r'$r\, [{\rm Mpc}/h]$')
            ax3.set_ylabel(r'$\gamma(\Delta \Sigma)$')
            ax3.tick_params(axis='x')
            ax3.tick_params(axis='y')
            ax3.set_xscale('log')
            # ax3.set_yscale('log')
            # ax3.grid(b=True, which='both', lw=0.1, color='k', ls='--')
            #plt.ylim([5e10,1e17])
            ax3.axvline(0.02,alpha=0.5,ls=':',color='grey')
            ax3.axvspan(0,0.06, alpha=0.1, color='grey')# ratio plots
            ax3.axvspan(10,50, alpha=0.1, color='grey')
            
            
            clusterPath_i=dataDir+file+'/corr_jack_signal.dat'
            
            labelMvir=r'$\log M_{\rm vir}$'
            fileNameMvir='Mvir'
            
            
            clusterData=np.genfromtxt(clusterPath_i)
            
            rbini, DeltaSigma, DeltaSigma_err, DeltaSigmaCross, DeltaSigmaCross_std, R, ngals=clusterData.T
            
            # zL_guess=np.mean(zRange)
            
            DeltaSigmaPlot=DeltaSigma
            DeltaSigma_errPlot=DeltaSigma_err
            
            ##### covariance matrix
            clusterCovPath_i=clusterPath_i+'.cov'
            cov=np.genfromtxt(clusterCovPath_i)
            cov=cov[:25,:] #get the first 25 rows
            
            # yerr_compare=np.sqrt(np.diag(cov))
            
            # ratio1=yerr_compare/DeltaSigma_err
            # print('R bins with high deviations of sqrt(diag(cov)) to DeltaSigma_err: ', R[ratio1>1.1])
            
            
            ##### remove NaN values found in covariance matrix from arrays and cov
            idarray=sum(np.isnan(cov))
            if sum(idarray)>0:
                indWithLeastNans=idarray.argmin() #chooses ONE row/column with least # nans
                #nans will be symmetric about diag
                boolNanLocations=np.isnan(cov[:,indWithLeastNans])#chooses that particular column and ID nan locations
                DeltaSigma=DeltaSigma[~boolNanLocations]
                DeltaSigma_err=DeltaSigma_err[~boolNanLocations]
                R=R[~boolNanLocations]
                
                cov=cov[~np.isnan(cov)]
                cov=np.reshape(cov,(R.shape[0],R.shape[0]))
            
            ##### Need to get rid of rows/rolumns corresponding to the below code!!!!!
            rsel=(R>loLim)&(R<hiLim)
            posSel=DeltaSigma>0 #only fit over positive values in Delta Sigma
            sel=rsel&posSel
            
            inds2del=np.arange(len(R))
            inds2del=inds2del[~sel]
            inds2del=inds2del[::-1] #reverse order, to delete larger values first
            
            for ind2del in inds2del:
                cov=np.delete(cov,ind2del,axis=0)
                cov=np.delete(cov,ind2del,axis=1)
            
            DeltaSigma=DeltaSigma[sel] 
            DeltaSigma_err=DeltaSigma_err[sel] 
            # DeltaSigmaCross=DeltaSigmaCross[sel]
            # DeltaSigmaCross_err=DeltaSigmaCross_err[sel]
            R=R[sel]
            #
            
            
            is_pos_def(cov)
            
            
            
            hCorrection=cosmo.h/cosmoYang2020.h
            DeltaSigma=DeltaSigma*hCorrection*(1e6)**2 #[DeltaSigma]=M_sun h / Mpc^2 from M_sun / pc^2
            DeltaSigma_err=DeltaSigma_err*hCorrection*(1e6)**2 #[DeltaSigma]=M_sun h / Mpc^2 from M_sun / pc^2
            # DeltaSigmaCross=DeltaSigmaCross*(1e6)**2 #[DeltaSigma]=M_sun h / Mpc^2 from M_sun h / pc^2
            # DeltaSigmaCross_std=DeltaSigmaCross_std*(1e6)**2 #[DeltaSigma]=M_sun h / Mpc^2 from M_sun h / pc^2
            R=R/hCorrection
            
            
            cov=cov*(hCorrection*(1e6)**2)**2
            
            yerr_compare=np.sqrt(np.diag(cov))
            
            # ratio=yerr_compare/DeltaSigma_err
            # print('ratio: ', ratio)
            # print('R bins with high deviations of sqrt(diag(cov)) to DeltaSigma_err: ', R[ratio>1.1])
            
            ##### fit over DeltaSigma
            log10Mvir_guess=np.log10(Mvir_guess)
            biasProf=bp.biasProfile(z=z_guess, cosmo=cosmo, log10Mvir=log10Mvir_guess)
            biasProf.meanExcessDensity=plot_meanExcessDensity
            
            ticfit=time.time()
            popt,pcov=biasProf.fit_profile(x=R,prof=DeltaSigma,
                                            prof_name='DeltaSigma',
                                            prof_cov=cov)#,reportFit=True)
            tocfit=time.time()
            print('time to fit Delta Sigma (s): %.2f' %(tocfit-ticfit))
            
            pfits_all.append(biasProf.pfits)
            # DeltaSigma_fit_comparison=biasProf.excessSurfaceDensity(r=R)
            # DeltaSigma_percentDiff=bt.percentDiff(DeltaSigma_fit_comparison,DeltaSigma)
            
            Rplot=np.logspace(np.log10(0.06),np.log10(30),100)
            DeltaSigma_fit_plot=biasProf.excessSurfaceDensity(r=Rplot)
            
            #Obtain Rsmin and RslopeMin
            Rsmin,Rsmin_options=bt.findRmin(X=Rplot,Y=Rplot*DeltaSigma_fit_plot,Xlo=loLim,Xhi=hiLim)
            
            Rmid,gamma_DeltaSigma=bt.calcGamma(x=Rplot, y=DeltaSigma_fit_plot)
            
            RslopeMin,RslopeMin_options=bt.findRmin(X=Rmid,Y=gamma_DeltaSigma,Xlo=loLim)
            
            
            Rsmins.append([lbin,Rsmin,RslopeMin])
            
            #prepare plotting characteristic radii
            radii=np.array([Rsmin,RslopeMin])
            radiiText=np.array([r'$r_{\rm smin}$',r'$r_{\rm smin}^{\Delta \Sigma}$'])
            markers=np.array(['o','X'])
            mews=np.array([2,2])
            markersize=15
            alphaplts=[1,1]
            lrange=range(radii.size)
            
            ##### convert back to pc
            DeltaSigma=DeltaSigma/(1e6)**2 #[DeltaSigma]=M_sun h / pc^2 from M_sun h/ Mpc^2
            DeltaSigma_err=DeltaSigma_err/(1e6)**2 
            DeltaSigma_fit_plot=DeltaSigma_fit_plot/(1e6)**2 
            
            DeltaSigmaPlot=DeltaSigmaPlot/(1e6)**2
            DeltaSigma_errPlot=DeltaSigma_errPlot/(1e6)**2
            
            ########## PLOT DeltaSigma ##########
            polyfitColor='orange'
            
            plotX=Rplot
            plotY=DeltaSigma_fit_plot
            
            plotYhi=DeltaSigma+DeltaSigma_err
            plotYlo=DeltaSigma-DeltaSigma_err
            
            # ax1.set_ylim([0,1000])
            ax1.fill_between(x=R,y1=plotYhi,y2=plotYlo)
            ax1.plot(plotX,plotY,ls='-',label='fit',color=polyfitColor)
            ax1.axvline(loLim,ls=':',color='k',alpha=0.3)
            
            
            #plot radii
            for l in lrange:
                radl=radii[l]
                minInd=bt.getClosestIndex(plotX,radl)
                resid = plotX[minInd]-radl
                ax1.errorbar(plotX[minInd], plotY[minInd], xerr=resid, marker=markers[l], ms=markersize, color = polyfitColor, 
                             mfc='none',markeredgewidth=mews[l],alpha=alphaplts[l])
            
            
            ########## PLOT R*DeltaSigma ##########
            plotX=Rplot
            plotY=Rplot*DeltaSigma_fit_plot
            
            plotYhi=R*(DeltaSigma+DeltaSigma_err)
            plotYlo=R*(DeltaSigma-DeltaSigma_err)
            
            # ax2.set_ylim([1e11,1e15])
            ax2.fill_between(x=R,y1=plotYhi,y2=plotYlo)
            ax2.plot(plotX,plotY,ls='-',color=polyfitColor)
            ax2.axvline(loLim,ls=':',color='k',alpha=0.3)
            
            
            #plot radii
            for l in lrange:
                radl=radii[l]
                minInd=bt.getClosestIndex(plotX,radl)
                resid = plotX[minInd]-radl
                ax2.errorbar(plotX[minInd], plotY[minInd], xerr=resid, marker=markers[l], ms=markersize, color = polyfitColor, 
                             mfc='none',markeredgewidth=mews[l],alpha=alphaplts[l],label=r'%s' %radiiText[l])
            
            
            ########## PLOT slope DeltaSigma ##########
            Rmid_data,gamma_DeltaSigma_data=bt.calcGamma(x=R, y=DeltaSigma)
            plotX=Rmid
            plotY=gamma_DeltaSigma
            
            ax3.set_ylim([-3,3])
            ax3.plot(Rmid_data,gamma_DeltaSigma_data,lw=5,alpha=0.3)
            ax3.plot(Rmid,gamma_DeltaSigma,ls='-',label='fit',color=polyfitColor)
            ax3.axvline(loLim,ls=':',color='k',alpha=0.3)
            
            
            #plot radii
            for l in lrange:
                radl=radii[l]
                minInd=bt.getClosestIndex(plotX,radl)
                resid = plotX[minInd]-radl
                ax3.errorbar(plotX[minInd], plotY[minInd], xerr=resid, marker=markers[l], ms=markersize, color = polyfitColor, 
                             mfc='none',markeredgewidth=mews[l],alpha=alphaplts[l])
            
            
            
            
            
            # ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax2.legend(loc='best')
            ax3.legend(loc='best')
            
            plt.subplots_adjust(wspace=0,hspace=0)
            
            fileName='DeltaSigma_lbin_%d_loLimFactor_%d.png' %(lbin,int(10*loLimFactor))
            plt.savefig(savePlotDir+fileName)
        
        Rsmins=np.array(Rsmins)
        np.save(file=saveDatDir+'Rsmins_loLimFactor_%d' %int(10*loLimFactor),arr=Rsmins)
        
        pfits_all=np.array(pfits_all)
        np.save(file=saveDatDir+'pfits_all_loLimFactor_%d' %int(10*loLimFactor),arr=pfits_all)
    
    

