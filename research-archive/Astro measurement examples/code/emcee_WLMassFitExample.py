#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:42:20 2021

@author: MattFong
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:09:42 2021
 
test Rcd determination for lbin4

@author: MattFong
"""

##### import stuff
import sys,os,time,itertools
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 22,'figure.figsize': (10,10)})
from matplotlib.ticker import FormatStrFormatter

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
redshiftRanges=['z_02_03_binMh/']#,'z_02_03_binMh_loc/','z_02_03_binMh_hic/']
# redshiftRanges=['z_02_03_binMh/']


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
        
        # directory_contents=directory_contents[2:] #test
        
        # directory_contents=['lensCat_4']
        
        print(directory_contents)
        
        
        
        
        
        ##### Choose fit limits
        hiLim=30 #Mpc/h
        # loLimFactors=[0.3,0.5,0.7]
        loLimFactors=[0.5]
        
        for loLimFactor in loLimFactors:
            #these are currently done by eye. Will need to use a more proper method in the future
            #by increasing mass (sorted directory_contents)
            # loLims=[0.5,0.7] #Mpc/h
            
            
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
            titleText+="\n"+' loLimFactor_%s%s' %(loLimFactor,addTxt)
            
            #### Delta Sigma plots
            figDeltaSigma = plt.figure(figsize=(10,12))
            axp=plt.subplot(1,1,1)
            # axp.set_title(titleText)
            axp.tick_params(right='on', top='on', direction='in', which='both')
            axp.set_ylabel(r'$\Delta \Sigma \, [{\rm M_{\odot}} h/{\rm pc^2}]$')
            axp.set_xlabel(r'$R \, [{\rm Mpc} / h]$')
            axp.tick_params(axis='y')
            # plt.setp(axp.get_xticklabels(), visible=False)
            axp.set_xscale('log')
            axp.set_yscale('log')
            # axp.axvspan(0,0.06, alpha=0.1, color='grey')
            
            yticks=[10,100]
            axp.set_yticks(yticks)
            axp.set_yticklabels(yticks)
            axp.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            
            
            
            
            
            
            # RcdSamples=[]
            # for file in directory_contents:
            for file in directory_contents[-1:]: # test
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
                Mh_meds=convertMass_cosmologies(Mh_meds)
                Mh_los=convertMass_cosmologies(Mh_los)
                Mh_his=convertMass_cosmologies(Mh_his)
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
                
                
                ##### load DeltaSigma data
                print('%s, loLimFactor_%d, %s' %(file, int(10*loLimFactor), addTxt)) #results directory path
                clusterPath_i=loadDeltaSigmaDir+file+'/corr_jack_signal.dat'
                
                clusterData=np.genfromtxt(clusterPath_i)
                
                rbini, DeltaSigma, DeltaSigma_std, DeltaSigmaCross, DeltaSigmaCross_std, R, ngals=clusterData.T
                
                
                DeltaSigmaPlot=DeltaSigma
                DeltaSigma_stdPlot=DeltaSigma_std
                
                posSel=DeltaSigma>0 #only fit over positive values in Delta Sigma
                R=R[posSel]
                DeltaSigma_std=DeltaSigma_std[posSel]
                DeltaSigma=DeltaSigma[posSel]
                
                hCorrection=cosmo.h/cosmoYang2020.h
                DeltaSigma=DeltaSigma*hCorrection#*(1e6)**2 #[DeltaSigma]=M_sun h / Mpc^2 from M_sun / pc^2
                DeltaSigma_std=DeltaSigma_std*hCorrection#*(1e6)**2 #[DeltaSigma]=M_sun h / Mpc^2 from M_sun / pc^2
                # DeltaSigmaCross=DeltaSigmaCross*(1e6)**2 #[DeltaSigma]=M_sun h / Mpc^2 from M_sun h / pc^2
                # DeltaSigmaCross_std=DeltaSigmaCross_std*(1e6)**2 #[DeltaSigma]=M_sun h / Mpc^2 from M_sun h / pc^2
                R=R/hCorrection
                
                
                log10Mvir_guess=np.log10(Mvir_guess)
                z_guess=0.25344
                
                
                
                R=R[:-2]
                DeltaSigma=DeltaSigma[:-2]
                DeltaSigma_std=DeltaSigma_std[:-2]
                
                
                
                
                
                
                
                
                ##### fit over DeltaSigma
                #
                fileName='fig_%d_2p_nfw_rayleigh_0_-2.txt' %(lbin)
                fitData=np.loadtxt(saveDatDir+'JiaqiMassFits/'+fileName)
                
                #original fits are in comoving, change to physical
                Rfit=fitData[:,0]*(1+z_guess)
                stellarFit=fitData[:,1]*(1+z_guess)#**2
                centralFit=fitData[:,2]*(1+z_guess)#**2
                satelliteFit=fitData[:,3]*(1+z_guess)#**2
                LSBiasFit=fitData[:,4]*(1+z_guess)#**2
                overallFit=fitData[:,5]*(1+z_guess)#**2
                
                
                
                
                
                
                
                
                
                
                
                ########## PLOT DeltaSigma ##########
                lw_originals=2
                alpha_originals=0.5
                cs=2
                # color=next(ColorList)
                
                
                # axp.fill_between(x=R,y1=DeltaSigma+DeltaSigma_std,y2=DeltaSigma-DeltaSigma_std,
                #                  color=color,alpha=alpha_originals,
                #                  label=r'$%.2f_{%.2f}^{%.2f}:%.1f$' %(logMh_meds,logMh_los,logMh_his,np.log10(nLens)))
                
                axp.errorbar(x=R,y=DeltaSigma,yerr=DeltaSigma_std,capsize=cs,
                             color='k',ls='',marker='o',
                             label=r'$%.2f_{%.2f}^{%.2f}:%.1f$' %(logMh_meds,logMh_los,logMh_his,np.log10(nLens)))
                
                
                
                plotYfits=[overallFit,centralFit,stellarFit,satelliteFit,LSBiasFit]
                lws=[2,2,2,2,2]
                alphafits=[1,0.8,0.8,0.8,0.8]
                fitcolors=['k','b','yellow','purple','green']
                labels=['overall fit','1-halo','stellar','mis-centering','large-scale bias']
                
                for plti in range(len(plotYfits)):
                    axp.errorbar(Rfit,plotYfits[plti],alpha=alphafits[plti],color=fitcolors[plti],
                                 lw=lws[plti],label=labels[plti])
                
                
                
                
                toc = time.time()
                print('time to complete %s (m): %.2f' %(file,(toc-tic)/60))
                
            #RcdSamples.append([lbin,Rcd_med,Rcd_lo,Rcd_hi])
            # RcdSamples=np.array(RcdSamples)
            # np.save(file=saveDatDir+'RcdSamples%s' %(fileTextAppend),arr=RcdSamples)
                
            axp.legend(loc='best',fontsize=12)
            axp.set_ylim([1,400])
            axp.set_xlim([0.03,50])
            
            plt.subplots_adjust(wspace=0,hspace=0)
            
            fileName='WLMassFit_DeltaSigma_lbin_%d.png' %lbin
            plt.savefig(savePlotDir+fileName)
            
            
            toctot=time.time()
            print('TOTAL time (s): %.2f' %(toctot-tictot))



