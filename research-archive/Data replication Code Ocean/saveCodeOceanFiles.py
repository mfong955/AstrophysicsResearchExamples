#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 12:53:53 2021

UPDATE data, since i lost my old code and have the previous code from Code Ocean
Saved old files in ./data_old, updating into./ data

@author: MattFong
"""


import sys,os,itertools
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import matplotlib 
matplotlib.rcParams.update({'font.size': 22,'figure.figsize': (10,10)})

import time
from scipy.optimize import curve_fit

sys.path.append(os.path.abspath('/Users/MattFong/Desktop/Projects/Code/SJTU'))
import biasTools as bt

from colossus.utils import constants
import colossus.halo as Halo
from colossus.halo import mass_so

from colossus.cosmology import cosmology as ccosmo
my_cosmo = {'flat': True, 'H0': 70.4, 'Om0': 0.268, 'Ob0': 0.044, 'sigma8': 0.83, 'ns': 0.968}
cosmo = ccosmo.setCosmology('my_cosmo', my_cosmo)

# mean matter density of universe
mdef = '1m'
z = 0
rhom = mass_so.densityThreshold(z=z, mdef=mdef)*(1e3)**3. #[rhom] = M_dot h^2 / Mpc^3 from M_dot h^2 / kpc^3


tictot=time.time()


#data directory
dataDir='/Users/MattFong/Desktop/Projects/Data/biasData/'

loadDir='./CodeOcean_Fong2020/data_old/'
saveDir='./CodeOcean_Fong2020/data/'

#load hdf5 file with bias and velocity profiles
fprofiles = h5py.File(loadDir+"profiles_old.hdf5", "r")
fprofiles.keys()

r=fprofiles['bias/radii'][()] #[r]=Mpc/h; radial bins for bias profiles
r_vr=fprofiles['velocity/radii'][()] #[r]=Mpc/h; radial bins for velocity profiles

parameterNames = ['Mvir', 'Vmax', 'j', 'e', 'a_half', 'delta_e']

parameterLabels=['$\\log M_{\\rm vir}$', '$V_{\\rm max}/V_{\\rm vir}$', '$j$',
                      '$e$', '$\\log a_{1/2}$', '$\\log(1+\\delta_e)$']

colors=[np.array([0. , 0. , 0.5, 1. ]),
          np.array([0.        , 0.15882353, 1.        , 1.        ]),
          np.array([0.        , 0.83333333, 1.        , 1.        ]),
          np.array([0.49019608, 1.        , 0.47754586, 1.        ]),
          np.array([1.        , 0.90123457, 0.        , 1.        ]),
          np.array([1.        , 0.27668845, 0.        , 1.        ]),
          np.array([0.5, 0. , 0. , 1. ])]


hf = h5py.File(saveDir+'profiles.hdf5', 'w')
group_bias=hf.create_group('bias')
group_bias.create_dataset('r',data=r)#[r]=Mpc/h; radial bins for bias profiles

group_velocity=hf.create_group('velocity') #[vr]=km/s
group_velocity.create_dataset('r',data=r_vr)#[r]=Mpc/h; radial bins for velocity profiles

group_nHaloPerBin=hf.create_group('nHaloPerBin')

group_parameterBinEdges=hf.create_group('parameterBinEdges')


for i in range(len(parameterNames)):
    b=fprofiles['bias/%s' %parameterNames[i]][()]
    # vp=fprofiles['velocity/%s' %parameterNames[i]][()] #[vr]=km/s
    nHaloPerBin=fprofiles['nHaloPerBin/%s' %parameterNames[i]][()]
    parameterBinEdges=fprofiles['parameterBinEdges/%s' %parameterNames[i]][()]
    
    vr=np.load(dataDir+'%s_vrMeans.npy' %parameterNames[i]) #change input velocities to radial (total) velocities
    vr=vr[:,1:] #get rid of first (nan) values, where central particle has 0 velocity
    
    group_bias.create_dataset('%s' %parameterNames[i],data=b)
    group_velocity.create_dataset('%s' %parameterNames[i],data=vr) #[vr]=km/s
    group_nHaloPerBin.create_dataset('%s' %parameterNames[i],data=nHaloPerBin)
    group_parameterBinEdges.create_dataset('%s' %parameterNames[i],data=parameterBinEdges)
    
fprofiles.close()

hf.close()





















#data directory
dataDir='/Users/MattFong/Desktop/Projects/Data/MvirRelations/'
radii=np.genfromtxt(fname=dataDir+'radii.txt')
masses=np.genfromtxt(fname=dataDir+'masses.txt')
Rmfrs_Mvir=np.genfromtxt(fname=dataDir+'Rmfrs_Mvir.txt')
Mmfrs_Mvir=np.genfromtxt(fname=dataDir+'Mmfrs_Mvir.txt')

radii=np.vstack([radii,Rmfrs_Mvir])
masses=np.vstack([masses,Mmfrs_Mvir])
#Rbs,Rsps,Rtas,Rvtot_mins,Rvp_mins,Rvds,Rvirs,RG20s,Rsp1s,Rsp2s,Rmfrs_Mvir
icds=0
isps=1
itas=2
ivrmaxinfall=3
ivpmaxinfall=4
ivrdispersionmin=5
ivirs=6
ixihms=7
ispdk17=8
isp87s=9 #use this only for display purposes
iids=10

#FLAG, or set some smallest Mvir bin values to 0, to exclude from later plots
radii[itas][0]=0
radii[ivrmaxinfall][0]=0
masses[itas][0]=0
masses[ivrmaxinfall][0]=0

#convert to 1.3*Rsp87 to compare with G20 results
radii[isp87s]=1.3*radii[isp87s]

Mvirs1=masses[ivirs]
Rvirs=radii[ivirs]

Mvirs_4Rsp87s=np.logspace(13,15,10) #[M]=M_sun/h
c0s = Halo.concentration.concentration(M=Mvirs_4Rsp87s, mdef='vir', z=0)
Rsp87s,Msp87s,mask=Halo.splashback.splashbackRadius(z=0,M=Mvirs_4Rsp87s,c=c0s,mdef='vir',
                                                    model='diemer17',rspdef='sp-apr-p87')
Rsp87s=1.3*Rsp87s/1e3 #[r]=Mpc/h from kpc/h, also convert to 1.3*Rsp87 to compare with G20 results

# radii[isp87s]=Rsp87s
# masses[isp87s]=Msp87s

##### set up all labels, colors, linestyles, and linewidths corresponding to datasets above

radiiLabels=np.array([r'$r_{\rm cd}$',r'$r_{\rm sp}$',r'$r_{\rm ta}$',
                      r'$r_{(v_{\rm r})^{\rm min}}$',r'$r_{(v_{\rm p})^{\rm min})$',
                      r'$r_{v_{(\sigma_{v_{\rm r}}})^{\rm min}}$',r'$r_{\rm vir}$',
                      r'$r_{(r^2 \xi_{\rm hm})^{\rm min}}$',r'$r_{\rm sp}^{\rm est}$',
                      r'$1.3 \times r_{\rm sp87}$',r'$r_{\rm id}$'])
colors=['r','m','c', 
        'k', 'r',
        'r','b',
        'y', 'r',
        'grey', 'k']
lineStyles=['-','-.','--',
            '-.', ':',
            ':',':',
            ':', ':',
            '-.', ':']
lws=[2,2,2,
      2,2,
      2,2,
      2,2,
      5,2]

alphas=[1,1,1,
        1,1,
        1,1,
        1,1,
        0.5,1]

Rids=radii[iids]
Mids=masses[iids]


"""
Load and plot characteristic radii (depletion, splashback, turnaround, and virial) 
and the characteristic overdensities (depletion, splashback, turnaround)
"""
#load hdf5 file with radii parameters and enclosed masses, binned by virial mass
fradii = h5py.File(loadDir+"radii_binnedByMvir_old.hdf5", "r")

Mvirs=fradii['Mvir'][()] #[M]=M_sun/h the virial mass bins 


hf = h5py.File(saveDir+'radii_binnedByMvir.hdf5', 'w')
hf.create_dataset('Mvir',data=Mvirs)

group_radii=hf.create_group('Rx')
group_deltas=hf.create_group('Deltax')

radiiNames=['characteristicDepletion','splashback','turnaround','virial']
#enclosedMasses=[] #for enclosed mass
for i in range(len(radiiNames)):
    if radiiNames[i]=='characteristicDepletion':
        Rx=fradii['radii/depletion'][()]
    else:
        Rx=fradii['radii/%s' %radiiNames[i]][()]
    
    #enclosedMasses.append(fradii['enclosedMasses/%s' %radiiNames[i]][()]) #[M]=M_sun/h  #for enclosed mass
    group_radii.create_dataset('%s' %radiiNames[i],data=Rx)

group_radii.create_dataset('innerDepletion',data=radii[iids])


# mean matter density of universe
rhom=cosmo.rho_m(z=0)*(1e3)**3. #[rhom] = M_dot h^2 / Mpc^3 from M_dot h^2 / kpc^3
rhoc=cosmo.rho_c(z=0)*(1e3)**3. #[rhom] = M_dot h^2 / Mpc^3 from M_dot h^2 / kpc^3

ratio_changeCharDensity=rhoc/rhom

radiiNames=['characteristicDepletion','splashback','turnaround']
Deltas=[]
for i in np.arange(len(radiiNames)):
    if radiiNames[i]=='characteristicDepletion':
        Deltax=fradii['deltas/depletion'][()]*ratio_changeCharDensity
    else:
        Deltax=fradii['deltas/%s' %radiiNames[i]][()]*ratio_changeCharDensity
    group_deltas.create_dataset('%s' %radiiNames[i],data=Deltax)


def MxRx2Delta(Mx,Rx,rhox):
    #calculate enclosed density contrast
    #rhox is the density of the Universe, e.g rho_c or rho_m
    #[rhox]=M_dot h^2 / Mpc^3
    return 3.*Mx/(4.*np.pi*(Rx**3))/rhox

Deltaids=MxRx2Delta(Mx=Mids,Rx=Rids,rhox=rhom)
group_deltas.create_dataset('innerDepletion',data=Deltaids)

fradii.close()
hf.close()






