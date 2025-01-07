#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fong & Han 2020 data use example
https://arxiv.org/abs/2008.03477
https://doi.org/10.1093/mnras/stab259

Plots for: 
    bias and velocity profiles versus radius
    Radii versus Virial Mass
    Characteristic Overdensities versus Virial Mass

@author: Matthew Fong
"""

import numpy as np
import matplotlib.pyplot as plt

import h5py

from os.path import dirname, abspath
parentDir = dirname(dirname(abspath(__file__)))

# loadDir='../data/'
loadDir=parentDir+'data/'
saveDir=parentDir+'/results/'

"""
Load and plot mean bias and mean velocity profiles
"""

#load hdf5 file with bias and velocity profiles
fprofiles = h5py.File(loadDir+"profiles.hdf5", "r")
fprofiles.keys()

#the radial bins for bias and velocity profiles are different: bias(r) and velocity(r_vr)
r=fprofiles['bias/r'][()] #[r]=Mpc/h; radial bins for bias profiles
r_vr=fprofiles['velocity/r'][()] #[r]=Mpc/h; radial bins for velocity profiles

parameterNames = ['Mvir', 'Vmax', 'j', 'e', 'a_half', 'delta_e'] #parameter file names

parameterLabels=['$\\log M_{\\rm vir}$', '$V_{\\rm max}/V_{\\rm vir}$', '$j$',
                      '$e$', '$\\log a_{1/2}$', '$\\log(1+\\delta_e)$'] #parameter labels

colors=[np.array([0. , 0. , 0.5, 1. ]),
         np.array([0.        , 0.15882353, 1.        , 1.        ]),
         np.array([0.        , 0.83333333, 1.        , 1.        ]),
         np.array([0.49019608, 1.        , 0.47754586, 1.        ]),
         np.array([1.        , 0.90123457, 0.        , 1.        ]),
         np.array([1.        , 0.27668845, 0.        , 1.        ]),
         np.array([0.5, 0. , 0. , 1. ])] #color list for plots

for i in range(len(parameterNames)): #loop over each parameter X_i
    b=fprofiles['bias/%s' %parameterNames[i]][()] #bias; [b]=1
    vr=fprofiles['velocity/%s' %parameterNames[i]][()] #radial velocity (includes Hubble Flow); [v]=km/s
    parameterBinEdges=fprofiles['parameterBinEdges/%s' %parameterNames[i]][()] #bin edges used for parameter X_i
    nHaloPerBin=fprofiles['nHaloPerBin/%s' %parameterNames[i]][()] #number of halos in each parameter bin in X_i
    
    jrange=np.arange(b.shape[0]) #set up parameter bin loop
    
    plt.figure(figsize=(10,14))
    axBias=plt.subplot(2,1,1)
    axVelocity=plt.subplot(2,1,2, sharex=axBias)
    for j in jrange: #loops over parameter bins in X_i
        color=colors[j]
        
        b_j=b[j] #bias data for parameter bin j
        
        v_j=vr[j] #velocity data for parameter bin j
        
        # PLOT BIAS
        axBias.tick_params(right='on', top='on', direction='in', labelsize=10, which='both')
        axBias.set_ylabel(r'$b(r)$', fontsize=24)
        axBias.tick_params(axis='y',labelsize=24)
        axBias.xaxis.set_visible(False)
        axBias.set_xscale('log')
        axBias.set_yscale('log')
        axBias.text(.85,.9, parameterLabels[i], horizontalalignment='center', verticalalignment='center', 
                 fontsize=22, transform=axBias.transAxes)
        axBias.plot(r, b_j, color=color,lw=2)
        
        # PLOT VELOCITY
        axVelocity.tick_params(right='on', top='on', direction='in', labelsize=10, which='both')
        axVelocity.set_xlabel(r'$r\, [{\rm Mpc}/h]$', fontsize=24)
        axVelocity.set_ylabel(r'$v_{\rm r}(r) \, [{\rm km/s}]$', fontsize=24)
        axVelocity.tick_params(axis='y',labelsize=24)
        axVelocity.tick_params(axis='x',labelsize=24)
        axVelocity.set_xscale('log')
        axVelocity.plot(r_vr, v_j, label = r'$[%.2f, %.2f]: %d$' %(parameterBinEdges[j], parameterBinEdges[j+1], nHaloPerBin[j]), color = color)
        
    axVelocity.legend(loc='best', fontsize=12)
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(saveDir+'profiles_%s.png' %parameterNames[i])
    plt.show()
    
fprofiles.close()



"""
Load and plot characteristic radii (characteristic and inner depletion, 
                                    splashback, turnaround, and virial) 
and the density contrasts (excluding virial, which is a constant).
We exclude Delta_virial, because it is just a constant.
Using Deltax and rho_m(z=0), one can obtain the enclosed masses Mx=M(<Rx).
"""
#load hdf5 file with radii parameters and Delta (rho_m(z=0)), binned by virial mass
fradii = h5py.File(loadDir+"radii_binnedByMvir.hdf5", "r")

#results are binned by virial mass
Mvirs=fradii['Mvir'][()] #[M]=M_sun/h

radiiNames=['turnaround','characteristicDepletion','splashback','innerDepletion','virial']
radii=[]

for i in range(len(radiiNames)):
    radii.append(fradii['Rx/%s' %radiiNames[i]][()]) #[r]=Mpc/h
radiiText=np.array([r'$r_{\rm ta}$',r'$r_{\rm cd}$',r'$r_{\rm sp}$',r'$r_{\rm id}$',r'$r_{\rm vir}$']) #radii labels
radii=np.array(radii)

colors=['c','r','m','k','b']
lineStyles=['--','-','-.',':','--']

#plot R_X vs virial mass 
plt.figure(figsize=(10,10))
for i in range(len(radii)):
    color=colors[i]

    Yplot=radii[i]
    Xplot=Mvirs 
    
    if i==3: #for Rid, exclude the lowest mass bin, difficult to determine (please see text)
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





radiiNames=radiiNames[:-1] #excluding Delta for virial radius; not included here
Deltas=[]
for i in np.arange(len(radiiNames)):
    Deltas.append(fradii['Deltax/%s' %radiiNames[i]][()]) 
Deltas=np.array(Deltas)

fradii.close()


#plot Delta_X vs virial mass bins
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




