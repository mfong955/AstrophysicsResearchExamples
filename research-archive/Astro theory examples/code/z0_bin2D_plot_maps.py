#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:11:30 2022

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




##### FIRST GPR fit!!!

#for nbin in [10]:
M_ind=0 #Mvir
a_ind=4 #a1/2
tic = time.time()

#Load map data
print('%s vs %s' %(parameterLabels[M_ind], parameterLabels[a_ind]))
tic = time.time()
fileName='z0_map_HaloPar%d%d_nbin%d.npz' %(M_ind, a_ind, nbin)
binnedData=np.load(simDir+fileName)
#ouput 
xmid=binnedData['X1']
ymid=binnedData['X2']
xstd=binnedData['X1std']
ystd=binnedData['X2std']
Rcds=binnedData['Rcds']
Mcds=binnedData['Mcds']
numHalosInBin=binnedData['nn']
logMvirs=binnedData['logMvirs']
# Rvirs=binnedData['Rvirs']
# MARs=binnedData['MARs']
logahalfs=binnedData['logahalfs']
Vmax_Vvirs=binnedData['Vmax_Vvirs']
Rsps=binnedData['Rsps']
Msps=binnedData['Msps']
RcdDelta=binnedData['RcdDelta']


selN100=numHalosInBin>nHaloPerBinLimit


rxs=Rcds #Use original Rmins for fits
alphas=abs(RcdDelta)*2

nParBin=nbin-1

########## Plot the result as an image
plt.figure(figsize=(10, 10))

##### plot n halo/pixel distribution
nn = numHalosInBin.reshape(nParBin, nParBin)
#nn = nn.T

#calculate middle of parBin1 and parBin2, should have size = len(parBin1)-1
#xmid = [(parBin1[p]+parBin1[p+1])/2 for p in range(len(parBin1[:-1]))]
#ymid = [(parBin2[p]+parBin2[p+1])/2 for p in range(len(parBin2[:-1]))]

Xmid = xmid.reshape(nParBin, nParBin)
Ymid = ymid.reshape(nParBin, nParBin)

nn = np.log10(nn)
mask=nn<2
nplot=np.ma.array(nn,mask=mask)
plt.contourf(Xmid, Ymid, nplot, cmap = 'Greys', alpha=cfalpha)
#plt.contour(Xmid,Ymid,nn,colors='Black',linestyles=':',linewidths=2,levels=np.array([0,np.log10(nHaloPerBinLimit)]))

##### plot where nn = 3 (nHaloPerBin = 1000)
########## plot R0 values in colorbar/histogram, with # halos/bin in values
zz = rxs.reshape(nParBin, nParBin) #*** b shape should be (len(X1)**2, 22)
#zz = zz.T #I checked these (comparing xedges, yedges, zz(after transpose) values with X1, X2, r0)

##### mask rxs < 0.1?
#mask=zz<0.01
zplot=np.ma.array(zz, mask=mask)

##### plot rx values
vmin=0
vmax=2
vnum=5
'''if i == 0:
    vmax=7'''
level = np.linspace(vmin, vmax, vnum)
CS = plt.contour(Xmid, Ymid, zplot, levels=level, vmin=vmin, vmax=vmax, extend='both')
#CS = plt.contour(Xmid, Ymid, zz, colors = 'Black')
CS.clabel(inline = True, fmt='%1.1f', fontsize=10)

cb = plt.colorbar()
cb.set_label(r'$r_{\rm b}\, [{\rm Mpc}/h]$', fontsize=16)



##### GPR fit
scale=xdata_full.std(axis=1)

xscale=scale[M_ind]
yscale=scale[a_ind]

rxfits = rxs #fit over all

'''nonNanInds=~np.isnan(ymid)
rxfits=rxfits[nonNanInds]
alphas=alphas[nonNanInds]
xmid=xmid[nonNanInds]
ymid=ymid[nonNanInds]'''
#Fit over data with Nhalos>100
rxfits=rxfits[selN100]
alphas=alphas[selN100]
xmid=xmid[selN100]
ymid=ymid[selN100]
xy=np.vstack([xmid/xscale, ymid/yscale]).T
xyfits=xy



normalization=1
#kernel=1*Matern(length_scale=1, nu=nu)
#kernel = 1.0 * RationalQuadratic(length_scale=1., alpha=0.5)
kernel=normalization*Matern(length_scale=1, nu=0.5)


if useKernel:
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alphas).fit(xyfits, rxfits)#[:,np.newaxis])
else: 
    gp = GaussianProcessRegressor(alpha=alphas).fit(xyfits, rxfits)#[:,np.newaxis])

#fit using parameter/variance: x/sigma_x
contourSideDim=100
xfit = np.linspace(xdata_full[M_ind].min(),xdata_full[M_ind].max(),contourSideDim)/xscale
yfit = np.linspace(xdata_full[a_ind].min(),xdata_full[a_ind].max(),contourSideDim)/yscale
Xfit, Yfit = np.meshgrid(xfit, yfit)
XYfit = np.vstack([Xfit.ravel(), Yfit.ravel()]).T
Zpred=gp.predict(XYfit)

#plot using parameter: x
xfit = np.linspace(xdata_full[M_ind].min(),xdata_full[M_ind].max(),contourSideDim)
yfit = np.linspace(xdata_full[a_ind].min(),xdata_full[a_ind].max(),contourSideDim)
Xfit, Yfit = np.meshgrid(xfit, yfit)
#XYfit = np.vstack([Xfit.ravel(), Yfit.ravel()]).T
XYfit=[Xfit.ravel(), Yfit.ravel()]
Zfit = Zpred.reshape(int(np.sqrt(Zpred.shape[0])), int(np.sqrt(Zpred.shape[0]))) 
#Zfit = Zfit.T #I checked these (comparing xedges, yedges, zz(after transpose) values with X1, X2, r0)
plt.contour(Xfit, Yfit, Zfit, levels=level, vmin=vmin, vmax=vmax, 
            extend='both', linewidths=5., alpha=0.3)#, linestyles='dashed')


##### figure format
plt.grid(b=True, which='both', lw=0.1, color='k', ls='--')
plt.tick_params(labelsize=15)
plt.xlabel(r'%s' %parameterLabels[M_ind], fontsize=16)
plt.ylabel(r'%s' %parameterLabels[a_ind], fontsize=16)
plt.title(r'$r_{\rm b}$ vs %s vs %s: nbin=# bins + 1 =%d' %(parameterLabels[M_ind], parameterLabels[a_ind], nbin))








##### THEN make rest of maps!

fig,axes=plt.subplots(figsize=(10,20), constrained_layout=False)


fig.subplots(sharex='col', sharey='row')
fig.subplots_adjust(hspace=0, wspace=0)
#plt.tick_params(axis=u'both', which=u'both',length=0)
plt.xticks(visible=False)
plt.yticks(visible=False)


#for all parameters 
gridInds=[1,6,11,16,21,
            7,12,17,22,
              13,18,23,
                 19,24,
                    25]

parInds=[]
for i in range(numPar):
    for j in range(numPar):
        if j>i:
            parInds.append([i,j])
parInds=np.array(parInds)

for p in range(parInds.shape[0]):
    i,j=parInds[p]
    gridInd=gridInds[p]
    
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
    numHalosInBin=binnedData['nn']
    logMvirs=binnedData['logMvirs']
    # Rvirs=binnedData['Rvirs']
    # MARs=binnedData['MARs']
    logahalfs=binnedData['logahalfs']
    Vmax_Vvirs=binnedData['Vmax_Vvirs']
    Rsps=binnedData['Rsps']
    Msps=binnedData['Msps']
    RcdDelta=binnedData['RcdDelta']
    
    
    Mvirs=10**logMvirs
    Rvirs=mass_so.M_to_R(M=Mvirs, z=z, mdef='vir')
    Rvirs=Rvirs/1e3 #[r]=Mpc/h from kpc/h
    
    numHalosInBinLimit=100
    
    
    
    rxs = Rcds #override rxs. Plot/fit over original rmins, but use rminMed errorbars
    whatPlot='Rcd'
    alphas=abs(RcdDelta)
    
    selN100=numHalosInBin>nHaloPerBinLimit
    
    
    nParBin=nbin-1
    
    
    ########## Plot the result as an image
    ax = fig.add_subplot(numPar-1,numPar-1, gridInd)
    tic = time.time()
    
    ##### plot n halo/pixel distribution
    nn = numHalosInBin.reshape(nParBin, nParBin)
    #nn = nn.T

    #calculate middle of parBin1 and parBin2, should have size = len(parBin1)-1
    #xmid = [(parBin1[p]+parBin1[p+1])/2 for p in range(len(parBin1[:-1]))]
    #ymid = [(parBin2[p]+parBin2[p+1])/2 for p in range(len(parBin2[:-1]))]

    Xmid = xmid.reshape(nParBin, nParBin)
    Ymid = ymid.reshape(nParBin, nParBin)

    nn = np.log10(nn)
    mask=nn<2
    nplot=np.ma.array(nn,mask=mask)
    ax.contourf(Xmid, Ymid, nplot, cmap = 'Greys', alpha=cfalpha)
    #plt.contour(Xmid,Ymid,nn,colors='Black',linestyles=':',linewidths=2,levels=np.array([0,np.log10(nHaloPerBinLimit)]))


    ########## plot Rx values in colorbar/histogram, with # halos/bin in values
    zz = rxs.reshape(nParBin, nParBin) #*** b shape should be (len(X1)**2, 22)
    #zz = zz.T #I checked these (comparing xedges, yedges, zz(after transpose) values with X1, X2, r0)
    ##### mask rxs < 0.1?
    #mask=zz<0.01
    zplot=np.ma.array(zz, mask=mask)
    
    ##### plot rx values
    vmin=0
    vmax=2.5
    vnum=5
    '''if i == 0:
        vmax=7'''
    level = np.linspace(vmin, vmax, vnum)
    
    CS = ax.contour(Xmid, Ymid, zplot, levels=level, vmin=vmin, vmax=vmax, extend='both',linewidths=2)
    #CS = plt.contour(Xmid, Ymid, zz, colors = 'Black')
    CS.clabel(inline = True, fmt='%1.1f', fontsize=10)
    
    
    ##### GPR fit 
    #fit using parameter/variance: x/sigma_x
    Mfit = logMvirs/scale[M_ind]
    afit = logahalfs/scale[a_ind]
    #mask where numHalosInBin==0
    mask=numHalosInBin==0
    #Mfit=np.ma.array(Mfit,mask=mask)
    #afit=np.ma.array(afit,mask=mask)
    Mfit[mask]=0
    afit[mask]=0
    #Xfit, Yfit = np.meshgrid(Mfit, afit)
    #XYfit = np.vstack([Xfit.ravel(), Yfit.ravel()]).T
    XYfit = np.vstack([Mfit, afit]).T
    Zpred=gp.predict(XYfit)
    
    Zfit = Zpred.reshape(int(np.sqrt(Zpred.shape[0])), int(np.sqrt(Zpred.shape[0]))) 
    #Zfit = Zfit.T #I checked these (comparing xedges, yedges, zz(after transpose) values with X1, X2, r0)

    mask=Zfit==Zpred[Mfit==0][0]
    Zfit=np.ma.array(Zfit,mask=mask)


    #Plot RminGPR(Mvir,ahalf) vs X1 X2!!!
    ax.contour(Xmid, Ymid, Zfit, levels=level, vmin=vmin, vmax=vmax, 
                extend='both', linewidths=5., alpha=0.2)#, linestyles='dashed')


    
    # gridInds=[1,6,11,16,21,
    #             7,12,17,22,
    #               13,18,23,
    #                  19,24,
    #                     25]
    ##### figure format
    ax.grid(which='both')
    if gridInd in [7,12,17,13,18,19]:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
    if gridInd in [22,23,24,25]:
        plt.setp(ax.get_yticklabels(), visible=False)
    if gridInd in [1,6,11,16,21]:
        ax.set_ylabel(r'%s' %parameterLabels[j], fontsize=16)
    if gridInd in [21,22,23,24,25]:
        ax.set_xlabel(r'%s' %parameterLabels[i], fontsize=16)
    
axes.axis('off')
fileName='allMaps_GPRRmin_nbin'+str(nbin)+'.png'
toc = time.time()
print(fileName+': %.2f s' %(toc-tic))
plt.savefig(pltDir+fileName, bbox_inches='tight')
plt.show()
print('saving directory: %s' %pltDir)

toctot = time.time()
print('Time total: %.2f s' %(toctot-tictot))
    
    
    
    
    