#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:18:33 2021

@author: MattFong
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:56:08 2021

Plot MAR(z) vs logahalf for different redshift bins

@author: MattFong
"""


import os, itertools
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
import biasProfile as bp
import biasTools_z as bt
from colossus.halo import mass_so
from colossus.utils import constants
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import seaborn as sns
import matplotlib 
matplotlib.rcParams.update({'font.size': 22,'figure.figsize': (10,10)})



from colossus.cosmology import cosmology as ccosmo
from colossus.lss import peaks
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


def cartesian2SphericalCoord(x,y,z):
    # (x,y,z) are the 1D arrays, e.g.: x = np.linspace(-fov, fov, 100)
    
    xy = x**2 + y**2
    r = np.sqrt(xy + z**2)
    phi = np.arctan2(np.sqrt(xy), z) # for elevation angle defined from z-axis down
    # phi = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    theta = np.arctan2(y, x) #on xy-plane, theta=0 starting from y=0
    return r,phi,theta

tictot=time.time()
##### set up directory names for local
#load directory 
dataDir='/Users/MattFong/Desktop/Projects/Data/halo_prof/'

#plot directory
cdw=os.getcwd()+'/'
savePlotDir='/Users/MattFong/Desktop/Projects/Code/SJTU/halo_prof/plots/mah/'

#make directories if not made
if not os.path.exists(savePlotDir):
    os.makedirs(savePlotDir)
    
print('savePlotDir=%s' %savePlotDir)



# snap_z_info (readable):
# [snap,z_find,z_search]
# [60, 2.023, 2.000]
# [74, 1.028, 1.000]
# [84, 0.528, 0.510]
# [91, 0.253, 0.250]
# [99, 0.000, 0.000]
snap_z_info=np.array([[6.00000000e+01, 2.02254605e+00, 2.00000000e+00],
                       [7.40000000e+01, 1.02790380e+00, 1.00000000e+00],
                       [8.40000000e+01, 5.27653813e-01, 5.10000000e-01],
                       [9.10000000e+01, 2.53440499e-01, 2.50000000e-01],
                       [9.90000000e+01, 1.19209290e-07, 0.00000000e+00]])

ColorList=['k','k','k','k','r']
alphas=[0.1,0.2,0.4,0.8,1]


# includeSubhalos=True
# includeSubhalos=False
includeSubhaloss=[True,False]

for includeSubhalos in includeSubhaloss: # completed running for [3]
    if includeSubhalos:
        subhaloStr='_includeSubhalos'
    if includeSubhalos is False:
        subhaloStr=''
    
    fig0,ax0=plt.subplots(figsize=(10,10))
    
    fig1,ax1=plt.subplots(figsize=(10,10))
    ax1.set_xlabel(r'$M_{\rm vir}$')
    ax1.set_ylabel(r'$\log a_{1/2}$')
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
    
    
    for redbini in np.arange(snap_z_info.shape[0]):
        snapshot=snap_z_info[:,0][redbini]
        z=snap_z_info[:,1][redbini]
        
        # fillBetween=False
        fillBetween=True
        
        # haloMatched=False #plots all MAR data, takes an extremely long time
        # haloMatched=True #matches halo par, profile, and MAR data
        
        
        toc_snap=time.time()
        print('====== (snap,z, current time (min)) : (%d,%.3f,%.2f)' %(snapshot,z,((toc_snap-tictot)/60)))
        
        # load data
        ticload=time.time()
        
        
        #marMatchedData.files
        # ['marTrackIDs',
        #  'snapshot',
        #  'z',
        #  'logMvirs',
        #  'Rvirs',
        #  'Vmaxs',
        #  'logahalfs',
        #  'MARs']
        
        # if haloMatched:
        # this data contains the halo par/prof/mar matched data
        marMatchedData=np.load(dataDir+'MAR_haloIDMatchedData%s_snapshot%d.npz' %(subhaloStr,snapshot))
        # marTrackIDs=marMatchedData['marTrackIDs']
        logMvirs=marMatchedData['logMvirs']
        Rvirs=marMatchedData['Rvirs'] #[r]=Mpc/h PHYSICAL!!!
        Vmaxs=marMatchedData['Vmaxs'] #[v]=km/s PHYSICAL!!!
        logahalfs=marMatchedData['logahalfs']
        # MARs=marMatchedData['MARs']
            
        
        Mvirs=10**logMvirs #[M] = M_odot/h
        G=constants.G/(1e3) #[G]=Mpc km2/𝑀⊙/𝑠2 from kpc km2/𝑀⊙/𝑠2 PHYSICAL
        Vvirs=np.sqrt(G*Mvirs/Rvirs) #[v]=km/s
        
        Vmax_Vvirs=Vmaxs/Vvirs
        
        
        
        tocload=time.time()
        print('time to load data (s): %.2f' %(tocload-ticload))
        
        x=logMvirs
        y=logahalfs
        
        if x[np.isnan(x)].shape[0]>0:
            mask=~np.isnan(x)
            x=x[mask]
            y=y[mask]
        if y[np.isnan(y)].shape[0]>0:
            mask=~np.isnan(y)
            x=x[mask]
            y=y[mask]
        
        counts,xbins,ybins,image=ax0.hist2d(x=x, y=y, bins=100,
                                color=ColorList[redbini],alpha=alphas[redbini],
                                norm=LogNorm(),label='z=%.3f' %z)#, levels=[0.2,0.4,0.6,0.8,1])
        ax1.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
                    colors=ColorList[redbini],alpha=alphas[redbini],label='z=%.3f' %z)
        # ax1.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],
        #             linewidths=3, cmap = plt.cm.rainbow, levels = [1,5,10,25,50,70,80,100])
        # sns.kdeplot(x=Vmax_Vvirs, y=logahalfs,alpha=alphas[redbini],
        #             color=ColorList[redbini],
        #             label='z=%.3f' %z, levels=[0.2,0.4,0.6,0.8,1])
    
    
    
    ax1.set_xlim([10.4,12.7])
    ax1.set_ylim([-.55,-0.125])
    # ax1.set_xlim([Vmax_Vvirs[~np.isnan(Vmax_Vvirs)].min(),Vmax_Vvirs[~np.isnan(Vmax_Vvirs)].max()])
    # ax1.set_ylim([logahalfs.min(),logahalfs.max()])
    
    ax1.legend(loc='upper right')
    # ax1.set_xlim([-.55,0.1])
    # ax1.set_ylim([.1,3])
    
    fileName='logahalfVsMvir_z%s' %subhaloStr
    # if fillBetween is True:
    # fileName+='_fillBetween'
    # if M10Interp:
    # fileName+=M10InterpStr
    fig1.savefig(savePlotDir+fileName+'.png')
    
    toctot=time.time()
    print('total time (m) : %.2f' %((toctot-tictot)/60))

