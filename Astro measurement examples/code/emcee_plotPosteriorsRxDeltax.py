#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:10:18 2021

show posteriors for Rx, Deltax

@author: MattFong
"""

##### import stuff
import sys,os,time,itertools
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from matplotlib.image import NonUniformImage

from colossus.halo import mass_adv

from multiprocessing import Pool, cpu_count


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

from scipy.interpolate import interp1d




def enclosedMass_fromRhoBinEdges(rho, r_middle, v_bin, r_binRight, rcut):
    #convert rho to rho_cumulative
    #overDensities is the density within (r_right^3 - r_left^3)
    
    
    if rcut==0:
        return 0
    else:
        #the maximum from simulation is rmax_simulation=30 Mpc/h
        mass_profile = np.cumsum(rho * v_bin) #rho is the stacked halo density profile in each bin, rho * vbin is the mass in each radial bins. Then we can get the enclosed mass profile by np.cumsum().
        f_interp = interp1d(np.log10(r_binRight), np.log10(mass_profile), kind="cubic", fill_value="extrapolate")#Interpolate the enclosed mass profilers in log-log space.
        enclosed_mass = 10**f_interp(np.log10(rcut)) # enclosed mass at rd
    return enclosed_mass #[M]=M_sun/h

def MxRx2Delta(Mx,Rx,rhox):
    #calculate enclosed density contrast
    #rhox is the density of the Universe, e.g rho_c or rho_m
    #[rhox]=M_dot h^2 / Mpc^3
    return 3.*Mx/(4.*np.pi*(Rx**3))/rhox


r_edges=np.logspace(np.log10(0.061),np.log10(5),10000) #[r]=Mpc/h from 0.06 to 10 

r_middle=np.sqrt(r_edges[:-1]*r_edges[1:])
r_middle[0]=r_edges[1]/np.sqrt(r_edges[2]/r_edges[1]) #middle of log bins

v_bin=4*np.pi/3*np.diff(r_edges**3.)
r_binRight = r_edges[1:] #right edges


diffLensCatStuff=['z_02_03_binMh_hic/','z_02_03_binMh_loc/']
##### choose redshift range for data
redshiftRanges=['z_02_03_binMh/','z_02_03_binMh_loc/','z_02_03_binMh_hic/']
# redshiftRanges=['z_02_03_binMh/']


# col3=True #for old results, without accurate redshift estimation
col3=False
if col3:
    redshiftRanges=['col3_z_02_03/','col3_z_02_03_binMh/']


# redshiftRanges=['z_02_03_binMh/']


# loosenPriors=[False,True]
# loosenPriors=[True] 
loosenPriors=[False] #only using False now
for loosenPrior in loosenPriors:
    if loosenPrior:
        addTxt='_loosenPriors'
    else:
        addTxt=''
    print('addTxt: %s ' %addTxt)
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
        currentCodeName='emcee_plotPosteriorsRxDeltax/'
        
        # codeDir=codeDir+currentCodeDirName
        # loadPlotDir=codeDir+currentCodeDirName+'plots/emcee_DeltaSigma/'+redshiftRange
        savePlotDir=codeDir+currentCodeDirName+'plots/'+redshiftRange+'allPosteriors/'
        if not os.path.exists(savePlotDir):
            os.makedirs(savePlotDir)
        
        
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
        
        # directory_contents=['lensCat_4']
        
        print(directory_contents)
        
        
        ##### Choose fit limits
        hiLim=30 #Mpc/h
        # loLimFactors=[0.3,0.5,0.7]
        loLimFactors=[0.5]
        
        for loLimFactor in loLimFactors:
            
            
            RxMxStats=[]
            for file in directory_contents:
            # for file in directory_contents[:1]: # test
                print('====================================================')
                lbin=int(re.findall(r'\d+',file)[0]) #get lambda bin ID
                
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
                
                
                log10Mvir_guess=np.log10(Mvir_guess)
                z_guess=0.25344
                
                
                
                rhom_z=cosmo.rho_m(z=z_guess)*(1e3)**3. #[rhom] = M_dot h^2 / Mpc^3 from M_dot h^2 / kpc^3
                rhom_z0=cosmo.rho_m(z=0)*(1e3)**3. #[rhom] = M_dot h^2 / Mpc^3 from M_dot h^2 / kpc^3
                
                titleText='%s %s loLimFactor_%d%s' %(redshiftRange,file, int(10*loLimFactor),addTxt)
                # titleText=r'$z \in [%.1f,%.1f]$' %(zRange[0],zRange[1])+'\n'+'fit range [Mpc/h]: [%.2f,%d]' %(loLim,hiLim)
                print(titleText)
                
                fname='RxMxSample_lbin_%d_loLimFactor_%d_hiLim_%d%s' %(lbin,int(10*loLimFactor),hiLim,addTxt)
                RxMxSample=np.load(saveDatDir+fname+'.npy')
                #[Rcd_si,Rsp_si,Mcd_si,Msp_si]
                Rcds=RxMxSample[:,0]
                Rsps=RxMxSample[:,1]
                Mcds=RxMxSample[:,2]
                Msps=RxMxSample[:,3]
                
                Deltacds_z=MxRx2Delta(Mx=Mcds,Rx=Rcds,rhox=rhom_z)
                Deltasps_z=MxRx2Delta(Mx=Msps,Rx=Rsps,rhox=rhom_z)
                
                Deltacds_z0=MxRx2Delta(Mx=Mcds,Rx=Rcds,rhox=rhom_z0)
                Deltasps_z0=MxRx2Delta(Mx=Msps,Rx=Rsps,rhox=rhom_z0)
                
                Rcds_med, Rcds_lo, Rcds_hi=bt.median_std(Rcds)
                Rsps_med, Rsps_lo, Rsps_hi=bt.median_std(Rsps)
                
                Deltacds_z_med, Deltacds_z_lo, Deltacds_z_hi=bt.median_std(Deltacds_z)
                Deltasps_z_med, Deltasps_z_lo, Deltasps_z_hi=bt.median_std(Deltasps_z)
                
                Deltacds_z0_med, Deltacds_z0_lo, Deltacds_z0_hi=bt.median_std(Deltacds_z0)
                Deltasps_z0_med, Deltasps_z0_lo, Deltasps_z0_hi=bt.median_std(Deltasps_z0)
                
                RxMxStats.append([Rcds_med, Rcds_lo, Rcds_hi, Rsps_med, Rsps_lo, Rsps_hi, 
                                  Deltacds_z_med, Deltacds_z_lo, Deltacds_z_hi, 
                                  Deltasps_z_med, Deltasps_z_lo, Deltasps_z_hi,
                                  Deltacds_z0_med, Deltacds_z0_lo, Deltacds_z0_hi, 
                                  Deltasps_z0_med, Deltasps_z0_lo, Deltasps_z0_hi])
                
                fig= plt.figure()
                ax=fig.add_subplot(1,1,1)
                ax.set_title(titleText)
                x=Rsps
                y=Rcds
                
                plt.scatter(Rsps,Rcds)
                plt.axvline(Rsps_med)
                plt.axhline(Rcds_med)
                
                plt.xlabel(r'$r_{\rm sp}$')
                plt.ylabel(r'$r_{\rm cd}$')
                plt.xscale('log')
                plt.yscale('log')
                # yticks=[0.1,0.5,1]
                # ax.set_yticks(yticks)
                # ax.set_yticklabels(yticks)
                # xticks=[0.1,0.2,0.3]
                # ax.set_xticks(xticks)
                # ax.set_xticklabels(xticks)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.savefig(savePlotDir+'RcdVsRsp_lbin_%d_loLimFactor_%d_hiLim_%d%s.png' %(lbin,int(10*loLimFactor),hiLim,addTxt))
                
                
                
                '''
                fig= plt.figure()
                ax=fig.add_subplot(1,1,1)
                ax.set_title(titleText)
                x=Rsps
                y=Rcds
                
                plt.scatter(Rsps,Rcds)
                plt.axvline(Rsps_med)
                plt.axhline(Rcds_med)
                
                plt.xlabel(r'$r_{\rm sp}$')
                plt.ylabel(r'$r_{\rm cd}$')
                plt.xscale('log')
                plt.yscale('log')
                yticks=[0.1,0.5,1]
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks)
                xticks=[0.1,0.2,0.3]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticks)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.savefig(savePlotDir+'RcdVsRsp_lbin_%d_loLimFactor_%d_hiLim_%d%s.png' %(lbin,int(10*loLimFactor),hiLim,addTxt))
                '''
                
toctot=time.time()
print('TOTAL time (s): %.2f' %(toctot-tictot))






