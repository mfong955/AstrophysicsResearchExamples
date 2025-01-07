#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fong & Han 2020 data use example
arXiv: 2008.03477

Plot examples for Rx versus Mvir, and Delta versus Mvir

Load and plot halo boundaries (characteristic and inner depletion, 
splashback, turnaround, and virial) and the corresponding density contrasts 
(we exclude the constant virial density contrast).

The enclosed masses, Mx=M(<Rx), can be obtained using the halo boundaries 
and density contrasts.

All halos are at z=0 and the radius units are in [r]=Mpc/h.

@author: Matthew Fong
"""

import numpy as np
import matplotlib.pyplot as plt

from os.path import dirname, abspath
parentDir = dirname(dirname(abspath(__file__)))

import h5py

loadDir=parentDir+'/data/'
saveDir=parentDir+'/results/'


#load hdf5 file with radii parameters and Delta (rho_m(z=0)), binned by virial mass
fradii = h5py.File(loadDir+"radii_binnedByMvir.hdf5", "r")


##### plot Rx vs Mvir

#results are binned by virial mass
Mvirs=fradii['Mvir'][()] #[M]=M_sun/h

radiiNames=['turnaround','characteristicDepletion','splashback','innerDepletion','virial'] #radii file names
radii=[]

for i in range(len(radiiNames)):
    radii.append(fradii['Rx/%s' %radiiNames[i]][()]) #[r]=Mpc/h
radiiText=np.array([r'$r_{\rm ta}$',r'$r_{\rm cd}$',r'$r_{\rm sp}$',r'$r_{\rm id}$',r'$r_{\rm vir}$']) #radii labels
radii=np.array(radii)

colors=['c','r','m','k','b']
lineStyles=['--','-','-.',':','--']


plt.figure(figsize=(10,10))
for i in range(len(radii)):
    color=colors[i]

    Yplot=radii[i]
    Xplot=Mvirs 
    if i==3: #exclude the lowest mass bin for the inner depletion radius (please see text)
        Xplot=Xplot[1:]
        Yplot=Yplot[1:]
    plt.plot(Xplot, Yplot, color=color, ls=lineStyles[i], lw=2, label=r'%s' %radiiText[i])

plt.legend(loc='best', fontsize=20)
plt.tick_params(labelsize=24)
plt.ylabel(r'%s' %(r'$r_{\rm X} \, [{\rm Mpc}/h]$'), fontsize=24)
plt.xlabel(r'$M_{\rm vir} \, [{\rm M_{\odot}}/h]$', fontsize=24)
plt.xscale('log')
plt.yscale('log')
plt.savefig(saveDir+'RxVsMvir.png')
plt.show()




##### plot Deltax vs Mvir

radiiNames=radiiNames[:-1] #excluding Delta for virial radius
Deltas=[]
for i in np.arange(len(radiiNames)):
    Deltas.append(fradii['Deltax/%s' %radiiNames[i]][()]) 
Deltas=np.array(Deltas)

fradii.close()

plt.figure(figsize=(10,10))
for i in range(len(radiiNames)):
    color=colors[i]

    Yplot=Deltas[i]
    Xplot=Mvirs
    plt.plot(Xplot, Yplot, color=color, ls=lineStyles[i], lw=2, label=r'%s' %radiiText[i])

plt.legend(loc='best', fontsize=20)
plt.tick_params(labelsize=24)
plt.ylabel(r'%s' %(r'$\Delta = \rho( < r_{X}) / \rho_{\rm m}$'), fontsize=24)
plt.xlabel(r'$M_{\rm vir} \, [{\rm M_{\odot}}/h]$', fontsize=24)
plt.xscale('log')
plt.yscale('log')
plt.savefig(saveDir+'DeltaVsMvir.png')
plt.show()




