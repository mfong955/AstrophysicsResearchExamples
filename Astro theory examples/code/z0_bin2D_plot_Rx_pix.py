#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:14:58 2022

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

from matplotlib.colors import LogNorm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

from colossus.cosmology import cosmology as ccosmo
my_cosmo = {'flat': True, 'H0': 70.4, 'Om0': 0.268, 'Ob0': 0.044, 'sigma8': 0.83, 'ns': 0.968}
cosmo = ccosmo.setCosmology('my_cosmo', my_cosmo)


def make2d(array,nParBin=None):
    if nParBin is None:
        nParBin=nbin-1
    #makes arrays 2d for GPR
    array2d=array.reshape(nParBin, nParBin)
    array2d=array2d.T
    return array2d



tictot=time.time()
##### set up directory names for local
# simDir='/home/mfong/Data/halo_prof/'
simDir='/Users/MattFong/Desktop/Projects/Data/halo_prof/'
pltDir='/Users/MattFong/Desktop/Projects/Code/SJTU/halo_prof/plots/z0_RxVsX1X2/'
ticload=time.time()

# get all bias data. to list them: alldata.files
loadDir=simDir+'biasData/'
r=np.load(loadDir+'r.npy')
CorrFunc=np.load(loadDir+'CorrFuncExt.npy')
xdata_full=np.load(loadDir+'xdata_full.npy')

tocload=time.time()
print('time to load data (s): %.2f' %((tocload-ticload)))


parameterNames=np.array(['Mvir', 'Vmax', 'j', 'e', 'a_half', 'delta_e'])
parameterLabels=np.array(['$\\log M_{\\rm vir}$', '$V_{\\rm max}/V_{\\rm vir}$', '$j$',
                          '$e$', '$\\log a_{1/2}$', '$\\log(1+\\delta_e)$'])


'''
print('saving bin edges for pcolor plots...')
tictot = time.time()
#run save code
nbin=11
for i in range(len(parameterLabels)):
    for j in range(len(parameterLabels)):
        if j>i:
            tic = time.time()
            fileNames=['Mvir', 'Vmax', 'j', 'e', 'a_half', 'delta_e']
            fileNames = np.array(fileNames)
            xx=np.logspace(-2.,1.5,100)
            parBin1=np.linspace(xdata_full[i].min(), xdata_full[i].max(), nbin)
            parBin2=np.linspace(xdata_full[j].min(), xdata_full[j].max(), nbin)
            
            ########## Make arrays 2D 
            
            Xedges, Yedges = np.meshgrid(parBin1, parBin2)
            
            ##### save data!!!
            data=np.vstack([Xedges.ravel(), Yedges.ravel()])
            fileName='edges_HaloPar%d%d_nbin%d.txt' %(i, j, nbin)
            toc = time.time()
            print(fileName+': %.2f s' %(toc-tic))
            np.savetxt(loadDir+fileName, data)
            
            toc = time.time()
            #print('R0 vs %s vs %s data saved , nbin = %d, time: %.2f (s)' %(name1, name2, nbin, (toc-tic)))
toctot=time.time()
print('COMPLETED! time (s) = %.2f' %(toctot-tictot))
'''

snapshot=99
z=0
nbin=11

rhom=cosmo.rho_m(z=z)*(1e3)**3. #[rhom]= M_sun*h^2/Mpc^3 from  M_sun*h^2/kpc^3

M_ind=0 #Mvir
a_ind=4 #a1/2

scale=xdata_full.std(axis=1)

useKernel=True

cfalpha=0.4 #makes halo num density more transparent (closer to 0)
nHaloPerBinLimit=100


print('---------- running code for snapshot (redshift): %d, %.2f' %(snapshot,z))
irange=np.arange(parameterLabels.shape[0])
# number of bins = nbin-1
nbin = 11

numPar = parameterNames.shape[0]
#fig = plt.figure(figsize=(20,20), constrained_layout=False)
#plt.tick_params(axis=u'both', which=u'both',length=0)
#fig.frameon=False

whatPlots=['Rcd','Rsp','Rcd_Rsp','Rcd_Rvir','Deltacd','Deltasp']
whatPlots=['Rvir']

for whatPlot in whatPlots:
    
    fig, axes=plt.subplots(1,5, figsize=(20,3))
    #plt.tick_params(axis=u'both', which=u'both',length=0)
    #fig.subplots(sharex='col', sharey='row')
    #fig.subplots_adjust(hspace=0, wspace=0)
    #plt.tick_params(axis=u'both', which=u'both',length=0)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    
    #for all parameters 
    gridInds=[1,2,3,4,5]
    
    parInds=[]
    #for i in range(numPar):
    for i in [0]:
        for j in range(numPar):
            if j>i:
                parInds.append([i,j])
    parInds=np.array(parInds)
    
    for p in range(parInds.shape[0]):
    #for p in [0]:
        i,j=parInds[p]
        gridInd=gridInds[p]
        
        print('halo par (i,j)=(%d,%d)' %(i,j))
        ########## Plot the result as an image
        ax = axes[p]
        #ax=fig.subplots(1,5,gridInd,sharex='col', sharey='row')
        
        #tic = time.time()
        #print('nbin: %d' %nbin)
        #Load map data
        #print('%s vs %s' %(parameterLabels[i], parameterLabels[j]))
        tic = time.time()
        
        proxyname1=parameterNames[i]
        proxyname2=parameterNames[j]
        print('# bins for %s vs %s: %d' %(proxyname1, proxyname2, (nbin-1)))
        
        
        fileName = 'profiles_z0_bin2D_%s_%s.npz' %(proxyname1,proxyname2)
        profData=np.load(simDir+fileName)
        parBin1=profData['parBin1']
        parBin2=profData['parBin2']
        r=profData['r']
        
        fileName='z0_map_HaloPar%d%d_nbin%d.npz' %(i, j, nbin)
        binnedData=np.load(simDir+fileName)
        xmid=binnedData['X1']
        ymid=binnedData['X2']
        xstd=binnedData['X1std']
        ystd=binnedData['X2std']
        Rcds=binnedData['Rcds']
        Mcds=binnedData['Mcds']
        Deltacds=binnedData['Deltacds']
        numHalosInBin=binnedData['nn']
        logMvirs=binnedData['logMvirs']
        # Rvirs=binnedData['Rvirs']
        # MARs=binnedData['MARs']
        logahalfs=binnedData['logahalfs']
        Vmax_Vvirs=binnedData['Vmax_Vvirs']
        Rsps=binnedData['Rsps']
        Msps=binnedData['Msps']
        RcdDelta=binnedData['RcdDelta']
        Deltasps=binnedData['Deltasps']
        
        
        Mvirs=10**logMvirs
        Rvirs=mass_so.M_to_R(M=Mvirs, z=z, mdef='vir')
        Rvirs=Rvirs/1e3 #[r]=Mpc/h from kpc/h
        '''
        #Deltas calculations
        # rho_c0=constants.RHO_CRIT_0_MPC3 #[rho]= M_sun h^2 / Mpc^3
        rho_m0=cosmo.rho_m(z)*(1e3)**3 #[rho]= M_sun h^2 / Mpc^3 from M_sun h^2 / kpc^3
        
        volume=(4.*np.pi*Rcds**3.)/3.
        Deltacds=Mcds/volume/rho_m0
        
        volume=(4.*np.pi*Rsps**3.)/3.
        Deltasps=Msps/volume/rho_m0
        '''
        numHalosInBinLimit=100
        
        
        nParBin=nbin-1
        
    
        ########## Choose parameter to plot
        # whatPlots=['Rcd_Rsp','Rcd_Rvir','Deltacd','Deltasp','Rcd','Rsp']
        if whatPlot=='Rcd_Rsp':
            rxs=Rcds/Rsps
            rxLabel=r'$r_{\rm cd}/r_{\rm sp}$'
            
        if whatPlot=='Rcd_Rvir':
            rxs=Rcds/Rvirs
            rxLabel=r'$r_{\rm cd}/r_{\rm vir}$'
            
        if whatPlot=='Deltacd':
            rxs=Deltacds
            rxLabel=r'$\Delta_{\rm cd}$'
            
        if whatPlot=='Deltasp':
            rxs=Deltasps
            rxLabel=r'$\Delta_{\rm sp}$'
            
        if whatPlot=='Rcd':
            rxs=Rcds
            rxLabel=r'$r_{\rm cd}$'
            
        if whatPlot=='Rsp':
            rxs=Rsps
            rxLabel=r'$r_{\rm sp}$'
            
        if whatPlot=='Rvir':
            rxs=Rvirs
            rxLabel=r'$r_{\rm vir}$'
            
        
        if p==0:
            ax.text(0.95,0.85, r'%s' %(rxLabel),ha='right',
                    fontsize=22, transform=ax.transAxes)
    
        rxs[np.isnan(rxs)]=0
        
        
        #rxs[rxs>2]=0
        
        #rxs[np.isinf(rxs)]=0 #set inf (X/0)= 0
        print('halo inds (%d,%d)' %(i,j))
        print('rxs max=%.2f' %rxs.max())
    
        ##### plot n halo/pixel distribution
        nn = numHalosInBin.reshape(nParBin, nParBin)
        #nn = nn.T
    
        #calculate middle of parBin1 and parBin2, should have size = len(parBin1)-1
        #xmid = [(parBin1[p]+parBin1[p+1])/2 for p in range(len(parBin1[:-1]))]
        #ymid = [(parBin2[p]+parBin2[p+1])/2 for p in range(len(parBin2[:-1]))]
    
        # Xmid = xmid.reshape(nParBin, nParBin)
        # Ymid = ymid.reshape(nParBin, nParBin)
        # Xmid=Xmid.T
        # Ymid=Ymid.T
        
        ######################################## REPLACE Xmid with Xedges!!!
        #edgeData=np.vstack([Xedges.ravel(), Yedges.ravel()])
        Xedges, Yedges = np.meshgrid(parBin1, parBin2)
        ##### save data!!!
        edgeData=np.vstack([Xedges.ravel(), Yedges.ravel()])
        Xedges,Yedges=edgeData
        
        
        Xmid = Xedges.reshape(nbin, nbin)
        Ymid = Yedges.reshape(nbin, nbin)
        # Xmid=Xmid.T
        # Ymid=Ymid.T
        ######################################## set up masks
        logMvirs4Mask=logMvirs.reshape(nParBin, nParBin)
        massMask=logMvirs4Mask>13.8
        
        nn = np.log10(nn)
        mask=(nn<np.log10(numHalosInBinLimit))
        mask=np.logical_or(mask,massMask)
        # nplot=np.ma.array(nn,mask=mask)
        #ax.contourf(Xmid, Ymid, nplot, cmap = 'Greys', alpha=cfalpha)
        
        
        ########## plot R0 values in colorbar/histogram, with # halos/bin in values
        zz = rxs.reshape(nParBin, nParBin) #*** b shape should be (len(X1)**2, 22)
        #zz = zz.T #I checked these (comparing xedges, yedges, zz(after transpose) values with X1, X2, r0)
        
        zplot=np.ma.array(zz, mask=mask)
        
        
        ##### plot rx values
        vmin=rxs[rxs>0].min()
        vmax=rxs.max()
        vnum=6
        
        # if whatPlot in ['Deltacd']:
        #     vmax=30
        if whatPlot in ['Deltacd','Deltasp']:
            if j in [1,4,5]:
                vmin=30
                vmax=100
        if whatPlot in ['Rcd']:
            vmin=0.1
            vmax=2
            
        
        '''if i == 0:
            vmax=7'''
        level = np.linspace(vmin, vmax, vnum)
        cfalpha=0.5
        #CS = ax.contourf(Xmid, Ymid, zplot, levels=level,cmap='gray',alpha=cfalpha)
        #CS.clabel(inline = False, fmt='%1.1f', fontsize=14, colors='k')
        
        def show_values(pc, data=None, fmt="%d", **kw):
            # from itertools import izip
            pc.update_scalarmappable()
            ax = pc.axes
            if data is None:
                data=pc.get_array()
            for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), data):
                x, y = p.vertices[:-2, :].mean(0)
                if np.all(color[:3] > 0.5):
                    color = (0.0, 0.0, 0.0)
                else:
                    color = (1.0, 1.0, 1.0)
                ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)
        CS = ax.pcolor(Xmid, Ymid, zplot, cmap='Greys',vmin=vmin,vmax=vmax)
        
        if whatPlot in ['Deltacd','Deltasp']:
            show_values(CS, fmt='%.1f', fontsize=8)
        else:
            show_values(CS, fmt='%.1f', fontsize=10)
        #CS = plt.contour(Xmid, Ymid, zz, colors = 'Black')
        
        ax.set_xlim([11.5,13.8])
        plt.setp(ax.get_yticklabels(), visible=True, fontsize=12)
        plt.setp(ax.get_xticklabels(), visible=True, fontsize=12)
        ##### figure format
        #ax.grid(which='both')
        ax.set_ylabel(r'%s' %parameterLabels[j], fontsize=16)
        ax.set_xlabel(r'%s' %parameterLabels[i], fontsize=16)
        # ax=axes[0]
    
    plt.tight_layout()
    fileName='massMaps_%s_nbin%d_massRow.png' %(whatPlot,nbin)
    toc = time.time()
    print(fileName+': %.2f s' %(toc-tic))
    plt.savefig(pltDir+fileName, bbox_inches='tight')
    plt.show()
    print('saving directory: %s' %pltDir)
    
    toctot = time.time()
    print('Time total: %.2f s' %(toctot-tictot))