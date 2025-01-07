#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:55:02 2020

@author: MattFong
"""

import numpy as np
import os, sys
import scipy
import random
import collections


import corner
import matplotlib.pyplot as plt

import biasTools_z as bt
from scipy.interpolate import interp1d



##### set up data directory, based on current 
codeDir=os.getcwd()+'/'
#data directory
cwdList=['/Users/MattFong/Desktop/Projects/Code/SJTU/','/home/mfong/Code/halo_prof/',
         '/home/mfong/Code/DECaLS/','/lustre/home/acct-phyzj/phyzj-m31/mfong/Code/DECaLS/',
         '/lustre/home/acct-phyzj/phyzj-m31/mfong/Code/']
# print('cwd: '+os.getcwd())
if codeDir not in cwdList:
    raise Exception('Need to set up work and data directories for biasTools_z!!!')
if codeDir==cwdList[0]: #for local mac
    dataDir='/Users/MattFong/Desktop/Projects/Data/halo_prof/'
    sys.path.append(os.path.abspath(codeDir))
if codeDir in cwdList[1:3]: #for remote gravity server
    dataDir='/home/mfong/Data/halo_prof/'
    sys.path.append('/home/mfong/Code')
if codeDir in cwdList[-2:]: #for remote pi2 server
    dataDir='/lustre/home/acct-phyzj/phyzj-m31/mfong/Data/halo_prof/'
    sys.path.append('/lustre/home/acct-phyzj/phyzj-m31/mfong/Code')



# import biasTools_z as bt

# print('biasProfile dataDir: %s' %dataDir)

#class biasProfile(profile_base.HaloDensityProfile):
class biasProfile():

    def __init__(self, z, cosmo, initPar=None, log10Mvir=None, mdef=None, 
                 ncpus=None, nsteps=50000):#, xi_mm_prof=False):
        
        # NOTE: there are some radii restrictions on this code.
        # values given in log10(Mpc/h), since interpolation for the profiles
        # are log(y(log(x))), and used in integrations for computation speed
        
        # The bias inner fit radius must be larger than 0.06 Mpc/h (Fong2020)
        # to overcome this we need to alter the fitting function such that the 
        # inner profile is flattened 
        self.log_r_min=np.log10(0.06) #log10(Mpc/h)
        # This can be corrected by using a NFW inner profile to fit!
        # I think this can be done by either fitting NFW profile to r<0.06
        # or testing to see if the linear bias can reliably obtain M_Delta
        # as an input for NFW profile, then a concentration can be assumed
        
        # Maximum integration for Sigma
        # The calculation for Sigma is significantly sensitive to the maximum
        # integration limit of rho(R,z)dz. However, because this is NOT an 
        # observable anyway, and the constant factor being added (rho_m*dz) 
        # to Sigma cancels out in DeltaSigma, we will not optimize the max
        # integration limit for Sigma. However, if you wish to improve Sigma
        # for any reason, you need to consider this parameter carefully.
        # If using simulations, the integration limit should match the radius
        # cut of the halo. The Delta Sigma profile pinches at r_max_integrate
        # which is not a natural feature. Integrating to infinity isn't 
        # practical either, due to interpolation possibly failing at very large 
        # radii, which we haven't tested. Furthermore rho at large r will have 
        # negligible additions to Sigma. We vary log_r_max_integrate to find 
        # where the DeltaSigma (the observable we care about) profiles MOSTLY 
        # converge.
        self.log_r_max_integrate=np.log10(200) #log10(Mpc/h)
        # log_r_max_integrate impacts on the very inner DeltaSigma profile, 
        # but on scales too small, which we exclude from our fit range anyway.
        # higher values don't seem to impact the integration time
        # note to self: our simulation cut is 15 Mpc/h, which works supagud 
        # with np.log10(15)
        
        
        # Lower integration for Delta Sigma 
        # Sigma interpolation looks reliable down to 0.02 Mpc/h in most cases,
        # but we optimize our code to reproduce DeltaSigma, the observable we 
        # care about. Inner region of DeltaSigma is sensitive to this parameter,
        # and though we haven't completely optimize this value, we did find a 
        # roughly good value to use. This is extremely far into the halo, where
        # baryonic effects are expected to have an impact, while we used a DM
        # simulation. In the end we integrate to match DeltaSigma roughly out 
        # to self.log_r_min=np.log10(0.06), then cut the region within.
        self.log_r_min_integrate=np.log10(0.007) #log10(Mpc/h)
        
        # Current matter-matter non-linear auto-correlation function limits
        self.xi_mm_x_lims=[2e-2,200] #Mpc/h
        
        # If you want to ignore the radii limit warnings, set to False
        self.showInputRWarnings=True
        
        # Test DeltaSigma integration calculation time
        # integrating using simps method gives the same results
        # simps: (times_all.mean(),times_all.std()): (1.35,0.14)
        # quad: (times_all.mean(),times_all.std()): (1.39,0.18)
        self.integrateQuad=False
        
        # Cosmology from Colossus package
        self.cosmo=cosmo
        
        # Redshift of lens
        self.z=z
        
        # For testing purposes, but maybe useful if you want to pull out meanSigma values as well
        self.meanSigma=None
        
        # for paper purposes, make DeltaSigma profile based on xi_mm
        # self.xi_mm_prof=xi_mm_prof
        
        # Parameter names and math text
        self.par_names = ['r0', 'dr1', 'dr2', 'alpha', 'beta', 'gamma', 'b0']
        self.par_str_math = [r'$r_{\rm 0}$', r'${\rm d}r_{\rm 1}$', r'${\rm d}r_{\rm 2}$',
                            r'$\alpha$', r'$\beta$', r'$\gamma$', r'$b_{\rm 0}$']
        
        # This changes wrt the most recent profile you wish to calculate. 
        # Used for raising exceptions for radii inputs.
        self.current_prof_name=None
        
        ##### set up initial guess, if initGuess is not given
        if log10Mvir is None:
            self.log10Mvir=0
        else:
            self.log10Mvir=log10Mvir
        
        if initPar is None:
            if self.log10Mvir>=14:
                self.i_guess=np.array(hiMass_init_guess)
            if (self.log10Mvir<14)&(self.log10Mvir>0):
                self.i_guess=np.array(loMass_init_guess)
            if self.log10Mvir==0:
                self.i_guess=np.array(noMass_init_guess)
        else:
            self.i_guess=np.array(initPar) #Initial guesses, will NOT be updated from fits!
        
        if mdef is None:
            self.mdef='vir' #default input for research
        else:
            self.mdef=mdef
        
        
        # This sets up easy-access parameter arrays/dictionaries 
        # self.par, self.pfits, and self.fit_params will be updated after any 
        # fitting method is used
        
        # Simple dictionary with current parameter names and values
        # [r_x] = Mpc/h, should be same units as input r
        self.par = collections.OrderedDict()
        for pari in range(len(self.par_names)):
            self.par[self.par_names[pari]]=self.i_guess[pari] 
        
        # Simple array with ONLY parameter values
        self.pfits=np.array([self.par[parname] for parname in self.par_names])
        
        # Limits (inclusive bounds) of flat priors (defined outside of class)
        # To update limits use self.lim_r0=[...,...], etc., then self.updateLimits()
        # These limits are optimized for bias binned by virial mass!!!
        if self.log10Mvir >= 14:
            self.lim_r0=hiMass_lim_r0 
            self.lim_dr1=hiMass_lim_dr1 
            self.lim_dr2=hiMass_lim_dr2 
            self.lim_alpha=hiMass_lim_alpha  
            self.lim_beta=hiMass_lim_beta 
            self.lim_gamma=hiMass_lim_gamma 
            self.lim_b0=hiMass_lim_b0 
        if (self.log10Mvir<14)&(self.log10Mvir>0):
            if self.z==0:
                self.lim_r0=loMass_lim_r0
            if self.z>0:
                self.lim_r0=loMass_lim_r0_z
            self.lim_dr1=loMass_lim_dr1  
            self.lim_dr2=loMass_lim_dr2  
            self.lim_alpha=loMass_lim_alpha  
            self.lim_beta=loMass_lim_beta
            self.lim_gamma=loMass_lim_gamma  
            self.lim_b0=loMass_lim_b0   
        if self.log10Mvir ==0:
            self.lim_r0=noMass_lim_r0
            self.lim_dr1=noMass_lim_dr1  
            self.lim_dr2=noMass_lim_dr2  
            self.lim_alpha=noMass_lim_alpha  
            self.lim_beta=noMass_lim_beta
            self.lim_gamma=noMass_lim_gamma  
            self.lim_b0=noMass_lim_b0   
            
        
        self.allParBounds=np.array([
                                    [self.lim_r0[0], self.lim_dr1[0], self.lim_dr2[0], 
                                      self.lim_alpha[0], self.lim_beta[0], self.lim_gamma[0], 
                                      self.lim_b0[0]], #lower bounds
                                    [self.lim_r0[1], self.lim_dr1[1], self.lim_dr2[1], 
                                      self.lim_alpha[1], self.lim_beta[1], self.lim_gamma[1], 
                                      self.lim_b0[1]] #upper bounds
                                    ])
        
        # # Sets up Parameter object, which includes bounds, for profile fitting
        # self.fit_params=Parameters()
        # for pi in range(len(self.par)):
        #     self.fit_params.add(self.par_names[pi], value = self.par[self.par_names[pi]], 
        #                         vary=True, max=self.allParBounds[1][pi], min=self.allParBounds[0][pi])
        
        # Number of parameters, used for fits
        self.ndim=len(self.par_names)
        
        # Interpolate for Sigma and DeltaSigma calculations. The interpolation
        # works for all cases studied and saves a significant amount of time.
        # Set to False if you want to test or whatever
        self.interpolate=True
        
        
        # Sometimes the bias fit profiles increase with r on large scales.
        # For testing: since we know they should be flat, I'm going to use 
        # this option to flatten bias(r>10). This does look a little weird
        # and shows up as a kink around r>10 Mpc/h
        self.flattenOuter=False
        
        
        # Obtain redshift related data. CURRENTLY LIMITED to specific redshift 
        # bins determined from simulation snapshots!!
        # If we want to generalize this, we need matter-matter non-linear
        # correlation functions as a function of redshift (xi_mm(r,z))
        redshiftBins=np.genfromtxt(dataDir+'redshiftBins.txt')
        
        self.currentAcceptable_snapshots=redshiftBins[:,0]
        self.currentAcceptable_z=redshiftBins[:,1]
        
        if self.z not in self.currentAcceptable_z:
            message='Input redshift must be in currentAcceptable_z:',self.currentAcceptable_z
            # message=message+'I need to generalize this later!!!'
            raise Exception(message)
        
        # Reference snapshot, for loading xi_mm(r,z)
        self.snapshot=self.currentAcceptable_snapshots[self.currentAcceptable_z==self.z][0]
        
        # load xi_mm interpolation stuff
        interpArray=np.load(dataDir+'corrFitFunc_%d.npy' %self.snapshot,allow_pickle=True)
        self.log_xi_mm_interp=interpArray.item()
        
        
        # matter density of Universe
        self.rhom=self.cosmo.rho_m(z=self.z)*(1e3)**3. #[rhom] = M_dot h^2 / Mpc^3 from M_dot h^2 / kpc^3
        
        # for testing, use this factor to calculate Sigma and Delta Sigma
        # from Delta rho(r) = rho(r) - rhom, or the meanExcessDensity.
        # self.rhom is constant, but may have radial impact on Sigma, DeltaSigma
        self.meanExcessDensity=False
        
        
        
        ##### Set up for fitting functions!!!
        # this generalizes the self.fit_profile(...) fitting function 
        # Function names and strings for easy plotting
        self.quantity_names=['b','rho','Sigma','DeltaSigma']
        self.quantity_str_math=['b','\rho \, [{\rm M_{\odot}} h^2 / {\rm Mpc^3}]','\Sigma \, [{\rm M_{\odot}} h / {\rm Mpc^2}]',
                                '\Delta \Sigma \, [{\rm M_{\odot}} h / {\rm Mpc^2}]']
        
        self.quantities = {}
        self.quantities['b'] = self._bias
        self.quantities['rho'] = self._density
        #self.quantities['M'] = self.enclosedMass
        self.quantities['Sigma'] = self._surfaceDensity
        self.quantities['DeltaSigma'] = self._excessSurfaceDensity
        
        
        ##### curve_fit setup
        self.pcov=None
        
        ##### lmfit fitting setup
        # tolerances for fit (using for all xtol, ftol, and gtol for now see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
        self.tolerances=1e-15
        # self.tolerances=3e-16
        # max number of fits
        self.max_nfev=int(1e5)
        # self.max_nfev=int(5e5)
        # for brute force method, finding global minimum. takes long and doesn't work well
        # self.brutediv=10
        
        # if self.out is None, self.fit_profile(...) has not been run
        # I may use this as a flag, since emcee requires optimized position for walkers
        # however, this limits the usage of biasProfile, like if someone is fitting
        # different profiles using different fitting methods
        self.out=None
        
        
        ##### for MCMC fitting. 
        # if sampler is None, then need to run_emcee(...)
        self.icov=None
        self.cov_diag=None
        self.sampler=None
        self.flat_samples=None
        # self.emcee_fit_par=None
        self.fit_name=None # profile name, to be updated when emcee is used
        self.nsteps=nsteps # default ?, 50*max(autocorr times [tau]), tau \in (~150-900)
        self.burn_in_steps=int(self.nsteps/100)
        if ncpus is None:
            if codeDir in cwdList[1:3]: #for remote gravity server, 72cpus/node
                self.ncpus=72
                self.nwalkers=int(self.ncpus) #144 walkers, 2 loops
            if codeDir in cwdList[-2:]: #for remote pi2 server, 40 cpus/node
                self.ncpus=40
                self.nwalkers=int(2*self.ncpus) #80 walkers, 2 loops
                # self.nsteps=nsteps #double the steps to make up for ~1/2 steps in gravity server
        else:
            self.ncpus=ncpus #this would be for 2 nodes with 72 cores/node for example
            self.nwalkers=int(self.ncpus)
            
        
        # self.updateLimits()
        # self.updateParams_forTestingFits()
        return
    
    # this raises an error if input r is outside of recommended limits
    def raiseExceptionOnInputR(self,r,whichProfile):
        if self.showInputRWarnings is True:
            if hasattr(r, "__len__"): # does input r have length (array or list)?
                r=np.array(r)
            else:
                r=np.array([r])
            
            #lower limit
            if whichProfile in ['b', 'rho']:
                r_min=10**self.log_r_min
            if whichProfile in ['Sigma','DeltaSigma']:
                r_min=10**self.log_r_min_integrate
            
            if r.min()<r_min:
                if whichProfile in ['b', 'rho']:
                    message='Due to the inner-powerlaw of the bias, the inner profile of the density cannot be reliably calculated. '
                if whichProfile in ['Sigma','DeltaSigma']:
                    message='The interpolation of Sigma (used to integrate for the mean enclosed surface density) is only reliable down to a certain radius. '
                message=message+'The minimum input r for this code (currently) has to be larger than %.2f Mpc/h. ' %r_min
                message=message+'If you want to ignore these messages use self.showInputRWarnings=False.'
                raise Exception(message)
            #upper limit
            if whichProfile in ['b', 'rho']:
                r_max=200
            if whichProfile in ['Sigma','DeltaSigma']:
                r_max=round(10**self.log_r_max_integrate)
            if r.max()>r_max:
                if whichProfile in ['b', 'rho']:
                    message='xi_hm(r) is determined from Colossus package, which has an upper limit on r. '
                if whichProfile in ['Sigma','DeltaSigma']:
                    message='I put an upper limit for r for integrating rho(r). To change the upper limit, use self.log_r_max_integrate. '
                message=message+'The maximum input r for this code (currently) has to be less than %d Mpc/h. ' %r_max
                message=message+'If you want to ignore these messages use self.showInputRWarnings=False.'
                raise Exception(message)
        
    # to update limits for testing fit boundaries (need to change self.lim_{par}, first)
    def updateLimits(self):
        self.allParBounds=np.array([ [self.lim_r0[0], self.lim_dr1[0], self.lim_dr2[0], 
                                 self.lim_alpha[0], self.lim_beta[0], self.lim_gamma[0], 
                                 self.lim_b0[0]], 
                               [self.lim_r0[1], self.lim_dr1[1], self.lim_dr2[1], 
                                self.lim_alpha[1], self.lim_beta[1], self.lim_gamma[1], 
                                self.lim_b0[1]] ])
        
        # for pi in range(len(self.par_names)):
        #     # sets up fit parameters to the initialized parameters, including bounds
        #     self.fit_params[self.par_names[pi]].max=self.allParBounds[1][pi]
        #     self.fit_params[self.par_names[pi]].min=self.allParBounds[0][pi]
    
    # to update parameters for testing fits (need to change self.i_guess, first)
    def updateParams_forTestingFits(self):
        #updates individual values in self.par and self.fit_params
        for i in range(len(self.par)):
            self.par[self.par_names[i]] = self.i_guess[i]
            # self.fit_params[self.par_names[i]].value = self.i_guess[i]
        #updates pfits array from self.par
        self.pfits=np.array([self.par[parname] for parname in self.par_names])
    
    '''
    # for updating parameters after fitting, from Parameters object
    def updateParams_fromParametersObject(self, newParameters):
        #newParameters must be a Parameters() object
        self.fit_params=newParameters
        #update dictionary
        for i in range(len(self.par)):
            self.par[self.par_names[i]] = newParameters[self.par_names[i]].value
        #update simple array from self.par
        self.pfits=np.array([self.par[parname] for parname in self.par_names])
    '''
    
    # for updating parameters after fitting, using curve_fit
    def updateParams_fromPopt(self, popt):
        self.pfits=popt
        #update dictionary
        for i in range(len(self.par)):
            self.par[self.par_names[i]] = popt[i]
        # #newParameters must be a Parameters() object
        # self.fit_params=Parameters()
        # for pi in range(len(self.par)):
        #     self.fit_params.add(self.par_names[pi], value = self.par[self.par_names[pi]], 
        #                         vary=True, max=self.allParBounds[1][pi], min=self.allParBounds[0][pi])
    
    def enclMassProf(self, r, interp=False):
        
        #bias profile is valid from 0.06
        r_edges=np.logspace(np.log10(0.06),1,1000) #[r]=Mpc/h from 0.06 to 10 
        
        r_middle=np.sqrt(r_edges[:-1]*r_edges[1:])
        r_middle[0]=r_edges[1]/np.sqrt(r_edges[2]/r_edges[1]) #middle of log bins
        
        rho=self.density(r_middle)
        
        v_bin=4*np.pi/3*np.diff(r_edges**3.)
        r_binRight = r_edges[1:] #right edges
        
        #the maximum from simulation is rmax_simulation=30 Mpc/h
        enclosed_mass = np.cumsum(rho * v_bin) #rho is the stacked halo density profile in each bin, rho * vbin is the mass in each radial bins. Then we can get the enclosed mass profile by np.cumsum().
        if interp:
            f_interp = interp1d(np.log10(r_binRight), np.log10(enclosed_mass), kind="cubic", fill_value="extrapolate")#Interpolate the enclosed mass profilers in log-log space.
            enclosed_mass = 10**f_interp(np.log10(r)) # enclosed mass at rd 
        return r_middle, enclosed_mass #[M]=M_sun/h
        
    def enclosedMass(self, rcut,interp = False):
        #convert rho to rho_cumulative
        #overDensities is the density within (r_right^3 - r_left^3)
        
        #bias profile is valid from 0.06
        r_edges=np.logspace(np.log10(0.06),1,1000) #[r]=Mpc/h from 0.06 to 10 
        
        r_middle=np.sqrt(r_edges[:-1]*r_edges[1:])
        r_middle[0]=r_edges[1]/np.sqrt(r_edges[2]/r_edges[1]) #middle of log bins
        
        rho=self.density(r_middle)
        
        v_bin=4*np.pi/3*np.diff(r_edges**3.)
        r_binRight = r_edges[1:] #right edges
        
        if rcut==0:
            return 0
        else:
            #the maximum from simulation is rmax_simulation=30 Mpc/h
            enclosed_mass_prof = np.cumsum(rho * v_bin) #rho is the stacked halo density profile in each bin, rho * vbin is the mass in each radial bins. Then we can get the enclosed mass profile by np.cumsum().
            if interp:
                if enclosed_mass_prof[enclosed_mass_prof<0].shape[0]==0:
                    f_interp = interp1d(np.log10(r_binRight), np.log10(enclosed_mass_prof), kind="cubic", fill_value="extrapolate")#Interpolate the enclosed mass profilers in log-log space.
                    enclosed_mass = 10**f_interp(np.log10(rcut)) # enclosed mass at rd
                else:
                    minInd=bt.getClosestIndex(r_middle,rcut)
                    enclosed_mass=enclosed_mass_prof[minInd]
            else:
                minInd=bt.getClosestIndex(r_middle,rcut)
                enclosed_mass=enclosed_mass_prof[minInd]
        return enclosed_mass #[M]=M_sun/h


    def getRcd(self,Rsearch,Xlo=None,highMass_biasThreshhold=None,slopeThreshhold=None):
        # This has mostly only been used for OBSERVATION DATA!!!
        # The reason for this function is to fine tune the Rcd estimation for the
        # highest mass bins, as their biases are not as perfect as simulation fits 
        # (originating from DeltaSigma fits)
        # works best if len(R) is large
        # used for accurately determining Rcd, but it requires more inputs
        # this needs to be adjusted depending on the dataset
        # highMass_biasThreshhold: bias.min() > highMass_biasThreshhold, means high mass
        # highMass_lowLimRcd: Rcd MUST be > 0.5 Mpc/h
        
        
        # For current observations z_02_03_binMh:
        # highMass_biasThreshhold=2
        # slopeThreshhold=-0.7
        if highMass_biasThreshhold is None:
            highMass_biasThreshhold=2
        if slopeThreshhold is None:
            slopeThreshhold=-0.7
        
        if self.showInputRWarnings is True:
            showInputRWarnings=True
            self.showInputRWarnings=False
        else: 
            showInputRWarnings=False
        bias=self.bias(Rsearch)
        Rcd,Rcd_options=bt.getRcd(R=Rsearch,biasFit=bias,Xlo=Xlo,
                                  highMass_biasThreshhold=highMass_biasThreshhold,
                                  slopeThreshhold=slopeThreshhold)
        # rho=self.density(r_middle)
        # Mcd=bt.enclosedMass_fromRhoBinEdges(rho=rho,r_middle=r_middle,
        #                                     r_edges=r_edges,rcut=Rcd)
        # Mcd=self.enclosedMass(rcut=Rcd) #this doesn't work too well unless fitting full dataset?
        
        if showInputRWarnings:
            self.showInputRWarnings=True
        return Rcd#, Mcd
    
    # get Rsp, Msp
    def getRsp(self,Rsearch):
        # This has mostly only been used for OBSERVATION DATA!!!
        # I think this should be accurate, since the process of finding Rsp is
        # a little easier. There always seems to be a dip in slope(rho)
        if self.showInputRWarnings is True:
            showInputRWarnings=True
            self.showInputRWarnings=False
        else: 
            showInputRWarnings=False
        rho=self.density(Rsearch)
        Rsp, Rsp_options = bt.rho2Rsp(R=Rsearch, rho=rho)
        # rho=self.density(r_middle)
        # Msp=bt.enclosedMass_fromRhoBinEdges(rho=rho,r_middle=r_middle,
        #                                     r_edges=r_edges,rcut=Rsp)
        # Msp=self.enclosedMass(rcut=Rsp) #this doesn't work too well unless fitting full dataset?
        if showInputRWarnings:
            self.showInputRWarnings=True
        return Rsp#, Msp
    
    def MxRx2Delta(Mx,Rx,rhox):
        #calculate enclosed density contrast
        #rhox is the density of the Universe, e.g rho_c or rho_m
        #[rhox]=M_dot h^2 / Mpc^3
        return 3.*Mx/(4.*np.pi*(Rx**3))/rhox
    
    def xi_mm(self,r):
        #[r]=Mpc/h
        return 10**self.log_xi_mm_interp(np.log10(r)) #[xi_mm]=1
    
    
    # converts input self.bias(...) parameters to simply r0, r1, r2, ...
    # this can be used for studying bias parameter dependencies, but not really used here
    def convertParams(self):
        r0=self.par['r0']
        dr1=self.par['dr1']
        dr2=self.par['dr2']
        alpha=self.par['alpha']
        beta=self.par['beta']
        gamma=self.par['gamma']
        b0=self.par['b0']
        return r0, r0+dr1, r0+dr1+dr2, alpha, beta, gamma, b0 #r0 < r1 < r2
    
    # # calculates Rcd,Mcd, and give rough estimates for Rvir,Mvir for testing
    # def get_Rcd(self,highMass_biasThreshhold=2,highMass_lowLimRcd=0.5,slopeThreshhold=-0.5, Xlo=None, Xhi=None, Ylo=None, Yhi=None):
    #     # this does not use bt.calcRmin(), which figures out Rcd from fit
    #     # this just estimates from the minimum of the CURRENT parameter set bias
    #     # as the bias can be wonky, we use bt.getRcd(), with the requirement
    #     # that it needs highMass_biasThreshhold
    #     # highMass_biasThreshhold: the bias minimum threshhold that classifies 
    #     # the bias bin as high or low mass (only used when binned by mass)
    #     # used for accurately determining Rcd, but it requires more inputs
    #     # this needs to be adjusted depending on the dataset
    #     # highMass_biasThreshhold: bias.min() > highMass_biasThreshhold, means high mass
    #     # highMass_lowLimRcd: Rcd MUST be > 0.5 Mpc/h
    #     R=np.logspace(np.log10(0.06),np.log10(10),1000)
    #     Rcd, _ = bt.getRcd(R,self.bias(R),highMass_biasThreshhold)
    #     Mcd=bt.enclosedMass(rho=self.density(R),rcut=Rcd,z=self.z)
    #     Rvir=Rcd/2.5
    #     Mvir=bt.enclosedMass(rho=self.density(R),rcut=Rvir,z=self.z)
    #     return Rcd,Mcd,Rvir,Mvir
    
    # original bias function
    def _bias(self,r,r0,dr1,dr2,alpha,beta,gamma,b0):
        if self.showInputRWarnings is True:
            self.raiseExceptionOnInputR(r,self.current_prof_name)
        # r_eps=1e-10 #makes it so brute force fitting doesn't run into infinities
        #r0 < r1 < r2
        
        r1=r0+dr1
        r2=r0+dr1+dr2
        bias=(1+(r/r0)**-(alpha+beta))/(1+(r/r1)**-(beta+gamma))*(b0+(r/r2)**-gamma)
        if self.flattenOuter is True: # sets the outer bias profile to a constant
            if hasattr(bias, "__len__"): #only if bias is an array or list with a length (integration requires scalar bias output)
                if r[-1]>10: #only if the input radial ARRAY goes beyond 10 Mpc/h
                    bias[r>10]=bias[(r>=5)&(r<=10)].mean()
        return bias
        # return (1+(r/r0)**(-(alpha)))/(1+(r/r1)**(-(beta)))*(1+(r/r2)**(-gamma))+b0
    
    def bias(self,r):
        #[r]=Mpc/h
        '''alpha>0, beta>=0?, gamma>=0: innerslope, void slope, outer slope.
        cluster: gamma~0
        void: b0<0, beta~0
        r2>r1>r0
        '''
        
        self.current_prof_name='b'
        
        r0=self.par['r0']
        dr1=self.par['dr1']
        dr2=self.par['dr2']
        alpha=self.par['alpha']
        beta=self.par['beta']
        gamma=self.par['gamma']
        b0=self.par['b0']
        
        if r.min() < 10**self.log_r_min:
            self.message='Please note that the bias is only reliable beyond r > %.2f.' %(10**self.log_r_min)
        
        return self._bias(r,r0,dr1,dr2,alpha,beta,gamma,b0)
    
    def _density(self,r,r0,dr1,dr2,alpha,beta,gamma,b0):
        if self.showInputRWarnings is True:
            self.raiseExceptionOnInputR(r,self.current_prof_name)
        #[r]=Mpc/h
        #bias=self.bias_rfunc(r)
        #corrFunc=xi_mm(r)
        #density=self.rhom*(1+bias*corrFunc)
        ret = self.rhom*(1.+self._bias(r,r0,dr1,dr2,alpha,beta,gamma,b0)*self.xi_mm(r)) #[rhom] = M_dot h^2 / Mpc^3
        if self.meanExcessDensity: #if only inner density (taking)
            ret=ret-self.rhom
        return ret
    
    def density(self, r):
        self.current_prof_name='rho'
        
        r0=self.par['r0']
        dr1=self.par['dr1']
        dr2=self.par['dr2']
        alpha=self.par['alpha']
        beta=self.par['beta']
        gamma=self.par['gamma']
        b0=self.par['b0']
        return self._density(r,r0,dr1,dr2,alpha,beta,gamma,b0) #[rhom] = M_dot h^2 / Mpc^3
    
    def _surfaceDensity(self,r,r0,dr1,dr2,alpha,beta,gamma,b0):
        if self.showInputRWarnings is True:
            self.raiseExceptionOnInputR(r,self.current_prof_name)
        #[r]=Mpc/h
        # accuracy=1E-2
        
        # integrating in log space, gives factor of r in integrand
        def integrand_interpolate(logr, R2):
            r2 = (10**(logr))**2 #bc integrating in log space
            # interpolationFunction: log10( rho( log10( r ) ) )
            ret = r2 * (10**(interpFunction(logr))) / np.sqrt(r2 - R2)
            return ret *np.log(10) #this factor corrects for integrating in log space, with x=log10(r), dx/dr=(1/r)log10(e)!!
        def integrand_direct(logr, R2):
        # def integrand(logr, R2):
            r_i=10**logr
            r2 = (r_i)**2 #bc integrating in log space
            ret = r2 * self._density(r_i,r0,dr1,dr2,alpha,beta,gamma,b0) / np.sqrt(r2 - R2)
            return ret *np.log(10) #this factor corrects for integrating in log space, with x=log10(r), dx/dr=(1/r)log10(e)!!
        
        if self.interpolate:
            # log_r_table=np.log10(np.logspace(np.log10(r.min()*(0.99)),np.log10(r.max()*(1.01)),1000))
            log_r_table=np.log10(np.logspace(self.log_r_min_integrate,self.log_r_max_integrate-0.001,50))
            
            rho_interp=self._density(10**log_r_table,r0,dr1,dr2,alpha,beta,gamma,b0)
            
            log_r_table=log_r_table[rho_interp>0] #gets rid of negative densities
            rho_interp=rho_interp[rho_interp>0]
            
            # if np.min(rho_interp)<0:
            #     raise Exception('negative in density and cannot make interpolation table. self.par: ',self.par)
            # interpolationFunction: log10( rho( log10( r ) ) )
            interpFunction=scipy.interpolate.InterpolatedUnivariateSpline(log_r_table, np.log10(rho_interp))
            integrand=integrand_interpolate
            
        if self.interpolate is False:
            integrand=integrand_direct
        
        if hasattr(r, "__len__"): # does input r have length (array or list)?
            r_use = np.array(r)
        else:
            r_use=np.array([r]) #for integrating when calculating DeltaSigma
        surfaceDensity = np.zeros_like(r_use)
        log_r_use = np.log10(r_use)
        for i in range(len(r_use)):
        	surfaceDensity[i], _ = scipy.integrate.quad(integrand, log_r_use[i], self.log_r_max_integrate, 
        								args = (r_use[i]**2))#, epsrel = self.accuracy, limit = 1000)
        	# surfaceDensity[i] *= 2.0 #put it at the end
        return surfaceDensity*2 #[Sigma] = M_dot h / Mpc^2 
    
    def surfaceDensity(self,r):
        self.current_prof_name='Sigma'
        r0=self.par['r0']
        dr1=self.par['dr1']
        dr2=self.par['dr2']
        alpha=self.par['alpha']
        beta=self.par['beta']
        gamma=self.par['gamma']
        b0=self.par['b0']
        return self._surfaceDensity(r,r0,dr1,dr2,alpha,beta,gamma,b0)
        
    def _excessSurfaceDensity(self,r,r0,dr1,dr2,alpha,beta,gamma,b0):
        if self.showInputRWarnings is True:
            self.raiseExceptionOnInputR(r,self.current_prof_name)
        #[r]=Mpc/h
        
        # accuracy=1E-4
        
        # the excess surface density should not depend on mass sheets (mass sheet degeneracy)
        # so this acts as a switch to exclude self.rhom from the density before
        # any delta sigma calculations are done
        # self.meanExcessDensity=True
        
        # integrating in log space, gives factor of r in integrand
        def integrand_interpolate(log10_r):
            r2 = (10**(log10_r))**2 #bc integrating in log space
            # interpolationFunction: log10( Sigma( log10( r ) ) )
            ret = r2 * (10**(interpFunction(log10_r)))
            return ret *np.log(10) #this factor corrects for integrating in log space, with x=log10(r), dx/dr=(1/r)log10(e)!!
        
        def integrand_direct(log10_r):
            r_i = 10**(log10_r)
            r2 = r_i**2 #bc integrating in log space #this is due to change of units from x=log10_r => dx=rdx
            ret = (r2) * self._surfaceDensity(r_i,r0,dr1,dr2,alpha,beta,gamma,b0) #[Sigma] = M_dot h / Mpc^2
            return ret *np.log(10) #this factor corrects for integrating in log space, with x=log10(r), dx/dr=(1/r)log10(e)!!
        
        
        # def integrand_interpolate(r_i):
        #     # interpolationFunction: log10( Sigma( log10( r ) ) )
        #     ret = r_i * (10**(interpFunction(np.log10(r_i))))
        #     return ret
        
        # def integrand_direct(r_i):
        #     ret = (r_i) * self._surfaceDensity(r_i,r0,dr1,dr2,alpha,beta,gamma,b0) #[Sigma] = M_dot h / Mpc^2
        #     return ret 
        
        # using interpolation of Sigma (100 R bins) seems slower than direct integration method
        if self.interpolate:
            r_table=np.logspace(self.log_r_min_integrate,self.log_r_max_integrate-0.001,100)
            log_r_table=np.log10(r_table)
            sigma_interp=self._surfaceDensity(r_table,r0,dr1,dr2,alpha,beta,gamma,b0)
            if np.min(sigma_interp)<0:
                raise Exception('negative in density and cannot make interpolation table. self.par: ',self.par)
            # interpolationFunction: log10( Sigma( log10( r ) ) )
            interpFunction=scipy.interpolate.InterpolatedUnivariateSpline(log_r_table, np.log10(sigma_interp))
            # interpFunction=scipy.interpolate.interp1d(log_r_table,np.log10(sigma_interp),fill_value='extrapolate')
            integrand=integrand_interpolate
        if self.interpolate is False:
            integrand=integrand_direct
        
        # integrand=integrand_direct
        
        if hasattr(r, "__len__"): # does input r have length (array or list)?
            r_use = np.array(r)
        else:
            r_use=np.array([r])
        
        # this extends the inner profile for integration, to avoid differing 
        # DeltaSigma inner amplitudes due to integration
        # this will be thrown away at the end
        nbin_innerExtend=7
        if r.min()>self.log_r_min_integrate:
            r_inner_extend=np.logspace(self.log_r_min_integrate,np.log10(r.min()),nbin_innerExtend+1)
            r_use=np.concatenate((r_inner_extend[:-1],r_use))
        log_r_use=np.log10(r_use)
        
        
        
        if self.interpolate:
            Sigma=10**(interpFunction(log_r_use))
        else:
            Sigma = self._surfaceDensity(r_use,r0,dr1,dr2,alpha,beta,gamma,b0)
        
        '''
        # THIS METHOD WORKS WELL!!! But if there is a different integration method 
        # that is faster, switch to that
        # r_long=np.logspace(self.log_r_min_integrate,np.log10(r_use[-1]),100000)
        r_long=np.linspace(10**self.log_r_min_integrate,r_use[-1],100000)
        sigmaLong=10**(interpFunction(np.log10(r_long)))
        
        meanSigma = np.zeros_like(r_use) #actually mean enclosed density within r_use
        
        for i in range(len(r_use)):
            r_table_sel=(r_long>10**self.log_r_min_integrate)&(r_long<=r_use[i])
            if len(r_long[r_table_sel])==0: #flag the first bin if no values in bin
                meanSigma[i]=0 #this can occur for r[i]<10**self.log_r_min_integrate
            else:
                meanSigma[i]=np.average(a=sigmaLong[r_table_sel],weights=r_long[r_table_sel])
        '''
        
        if self.integrateQuad:
            # This method WORKS NOW. integrate.quad
            meanSigma = np.zeros_like(r_use) #actually mean enclosed density within r_use
            meanSigma_0,_=scipy.integrate.quad(integrand, self.log_r_min_integrate, log_r_use[0], args = ())#, epsrel = accuracy, limit = 1000)
            meanSigma[0]=meanSigma_0
            for i in range(len(r_use))[1:]: 
                meanSigma_i, _ = scipy.integrate.quad(integrand, log_r_use[i-1], log_r_use[i], 
        				 						     args = ())#, epsrel = accuracy, limit = 1000)
                meanSigma[i] = meanSigma[i-1]+meanSigma_i
            meanSigma=meanSigma * 2.0 / r_use**2
        
        
        if self.integrateQuad is False:
            # This seems to work well, and compared to np.average method, it SEEMS a TINY
            # bit faster, and the inner amplitudes seem to change a bit
            rSigma=r_use*Sigma
            
            sum_SD=np.zeros(len(r_use))
            for riii in range(len(r_use)): 
                rsel=r_use<=r_use[riii]
                # sum_SD[riii] = np.sum(Sigma[rsel]*R[rsel]*dr[rsel])
                # r_uptoRlim=r_use[rsel] #current integration radius
                # SD_uptoRlim=Sigma[rsel] #selected surface density OUT TO radius
                rSigma_rsel=rSigma[rsel]
                
                sum_SD[riii]=scipy.integrate.simps(y=rSigma_rsel, x=r_use[rsel]) #integration out to radius for mean
            
            meanSigma=2*sum_SD/(r_use**2) #mult by 2 and divide out area to get mean
        
        
        meanSigma=meanSigma[nbin_innerExtend:]
        Sigma=Sigma[nbin_innerExtend:]
        
        self.meanSigma=meanSigma
        
        
        deltaSigma = meanSigma - Sigma
		
        # self.meanExcessDensity=False #turn off switch
        return deltaSigma
        # burninoutinds=10
        # r_use=np.array(r)
        # if r.min()>10**self.log_min_r:
        #     r_use=np.concatenate((np.logspace(self.log_min_r,np.log10(r.min()),burninoutinds)[:-1],r_use)) #add smaller radii for integration
        # if r.max()<10**self.log_max_r:
        #     r_use=np.concatenate((r_use,np.logspace(np.log10(r.max()),self.log_max_r,burninoutinds)[1:]))
        # Sigma = self._surfaceDensity(r_use,r0,dr1,dr2,alpha,beta,gamma,b0)
        
        # #integration using this method is faster
        # meanSurfaceDensity=np.zeros(len(r_use))
        # for riii in range(len(r_use)): 
        #     rsel=r_use<=r_use[riii]
        #     # sum_SD[riii] = np.sum(Sigma[rsel]*R[rsel]*dr[rsel])
        #     r_uptoRlim=r_use[rsel] #current integration radius
        #     SD_uptoRlim=Sigma[rsel] #selected surface density OUT TO radius
            
        #     meanSurfaceDensity[riii]=integrate.simps(y=SD_uptoRlim*r_uptoRlim, x=r_uptoRlim) #integration out to radius for mean
            
        # meanSurfaceDensity=2*meanSurfaceDensity/(r_use**2) #mult by 2 and divide out area to get mean
        
        # DeltaSigma = meanSurfaceDensity - Sigma
        # if r.min()>10**self.log_min_r: #get rid of the beginning of the array that was added
        #     DeltaSigma=DeltaSigma[(burninoutinds-1):] 
        # if r.max()<10**self.log_max_r: #get rid of the end of the array that was added
        #     DeltaSigma=DeltaSigma[:-(burninoutinds-1)]
        # return DeltaSigma #[Delta Sigma] = M_dot h / Mpc^2
    
    def excessSurfaceDensity(self,r):
        self.current_prof_name='DeltaSigma'
        r0=self.par['r0']
        dr1=self.par['dr1']
        dr2=self.par['dr2']
        alpha=self.par['alpha']
        beta=self.par['beta']
        gamma=self.par['gamma']
        b0=self.par['b0']
        return self._excessSurfaceDensity(r, r0, dr1, dr2, alpha, beta, gamma, b0)
        
    
    
    
    # for minimization fitting!
    def functionToMinimize(self, par, x, prof, model, err = None):
        resid = prof - model(x,par['r0'],par['dr1'],par['dr2'],par['alpha'],
                             par['beta'],par['gamma'],par['b0'])
        if err is not None:
            resid = resid / err
        return resid
    
    # fitting function
    def fit_profile(self, x, prof, prof_name, prof_cov=None, prof_err=None, 
                    xLo=None, xHi=None, reportFit=False):
        # This method performs as well as LMFit and is faster. Though less 
        # information in output
        
        # # for simulations this works best when halos are binned by 
        # # mass and logahalf or Vmax/Vvir, and log10Mvir=0!!!!
        # Rvir=mass_so.M_to_R(10**self.log10Mvir,self.z,self.mdef)
        # Rsearch=np.logspace(np.log10(Rsearch.min()),np.log10(Rsearch.max()),5000)
        # if self.log10Mvir<14:
        #     Rsel = (Rsearch>0.7*Rvir)&(Rsearch<4*Rvir) # Mpc/h
        # if self.log10Mvir>=14:
        #     Rsel = (Rsearch>1.3*Rvir)&(Rsearch<3.8*Rvir) # Mpc/h
        # Rsearch=Rsearch[Rsel]
        
        #reportFit = True: reports fit 
        self.updateLimits()
        
        if prof_name not in self.quantity_names:
            raise Exception("prof_name needs to be in: ", self.quantity_names)
        
        if prof_name in ['Sigma']:
            print('WARNING: Please note that we optimize this code to fit the observable bias, rho, and DeltaSigma. To appropriately fit Sigma choose self.log_r_max_integrate=np.log10(simulation_radius_cut).')
        # Set the correct function to evaluate during the fitting process. We could just pass
		# quantity, but that would mean many evaluations of the dictionary entry.
        f = self.quantities[prof_name]
        
        # changes corrent profile name, carries into quantity function, to inform
        # about limits on r
        self.current_prof_name=prof_name
        
        
        popt,pcov=scipy.optimize.curve_fit(f,x,prof,sigma=prof_cov,
                                           p0=self.i_guess,bounds=self.allParBounds,
                                           maxfev=int(1e8),xtol=1e-10)
        
        self.updateParams_fromPopt(popt)
        self.pcov=pcov
        return popt,pcov
    '''
    # fitting function
    def fit_profile_using_LMFit(self, x, prof, prof_name, prof_cov=None, prof_err=None, 
                    xLo=None, xHi=None, reportFit=False):
        from lmfit import Parameters, Minimizer
        from lmfit.printfuncs import report_fit
        #reportFit = True: reports fit 
        self.updateLimits()
        
        if prof_name not in self.quantity_names:
            raise Exception("prof_name needs to be in: ", self.quantity_names)
        
        if prof_name in ['Sigma']:
            print('WARNING: Please note that we optimize this code to fit the observable bias, rho, and DeltaSigma. To appropriately fit Sigma choose self.log_r_max_integrate=np.log10(simulation_radius_cut).')
        
        # if (prof_name in ['Sigma','DeltaSigma'])&(self.interpolate is False):
        #     print('WARNING: fitting method while interpolation False takes forever! Recommended use self.interpolate=True')
        # Set the correct function to evaluate during the fitting process. We could just pass
		# quantity, but that would mean many evaluations of the dictionary entry.
        f = self.quantities[prof_name]
        
        # changes corrent profile name, carries into quantity function, to inform
        # about limits on r
        self.current_prof_name=prof_name
        
        #x=(RadialBinsExt_xmid/1e3)
        #if prof_err is None:
        #    prof_err=np.max([prof*1, siglin/CorrFuncExt], axis=0) #Always use prof_err = eb[i] (error in bin i)
        if xLo is None:
            xLo=0.06#Fong2020 to fit only outer power-law
        if xHi is None:
            xHi=20 #Sigma drops off at the end, which I am uncertain about
        
        xsel=(x>=xLo)&(x<=xHi)
        
        x=x[xsel]
        prof=prof[xsel]
        if prof_err is not None:
            prof_err=prof_err[xsel]
            
        # fit using minimizer
        #def functionToMinimize(self, par, x, prof, model, err = None):
        # out = minimize(self.functionToMinimize, self.fit_params, args=(x, prof, f, prof_err),
        #                nan_policy='omit') #this method is default, and only finds LOCAL minimum
        fitter = Minimizer(self.functionToMinimize, self.fit_params, 
                           fcn_args=(x, prof, f, prof_err), 
                           nan_policy='omit')
        # out=fitter.minimize(method='ampgo')
        ## global minimizers
        # out=fitter.ampgo()
        # out=fitter.brute(Ns=100,keep=20)
        # out=fitter.shgo()
        # out=fitter.basinhopping()
        # out=fitter.dual_annealing()
        # out = fitter.minimize(method='brute')#, Ns=25, keep=25)
        
        ## local minimizers
        
        out=fitter.least_squares(max_nfev=self.max_nfev,xtol=self.tolerances,ftol=self.tolerances,
                                 gtol=self.tolerances) #I think this one is the best
        # out=fitter.leastsq()
        
        # self.out is not None, and fitting has occurred
        self.out = out
        
        if reportFit:
            report_fit(out, show_correl=True, modelpars=self.fit_params)
        
        #update parameters
        self.updateParams_fromParametersObject(out.params)
        return out
    '''
    
    # fitting function
    def emcee_fit_profile(self, x, prof, prof_name, prof_cov, xLo=None, 
                          saveBackendFilePath=None, randomInit=False):
        
        #if saveBackendFilePath is given, will save backend (save as go), 
        # and also fits until 100*tau(autocorr time) OR estimate changes by less than 1%
        
        # if self.emcee_fit_par is None:
        import emcee
        import time
        # import random
        
        self.updateLimits()
        
        # Created 02/23/21: This will start with flat priors, maybe update in the future
        # I will need to study the prior distributions for the bias binned by parameter X,
        # with emphasis on X=Mvir
        # Following steps from emcee documentation https://emcee.readthedocs.io/en/stable/tutorials/line/
        
        self.icov=np.linalg.pinv(prof_cov)
        
        # self.cov_diag=prof_cov.diagonal()
        
        if prof_name not in self.quantity_names:
            raise Exception("prof_name needs to be in: ", self.quantity_names)
        
        # Set the correct function to evaluate during the fitting process. We could just pass
  		# quantity, but that would mean many evaluations of the dictionary entry.
        f = self.quantities[prof_name]
        
        # changes corrent profile name, carries into quantity function, to inform
        # about limits on r
        self.current_prof_name=prof_name
        
        # updates latest profile name that was used in fits
        self.fit_name=prof_name
        
        # should I write code to get rid of nan values here or keep it outside of code?
        # i'll just let user do that
            
        # def run_emcee(f,x,y,icov,saveFilePath,nwalkers=self.nwalkers, 
        #               burn_in_steps=int(self.nsteps/10),nsteps=self.nsteps,
        #               randomInit=randomInit):
        print('running emcee...')
        tic_emcee=time.time()
        
        # set up initial walkers (pos)
        def randomInRange(limits):
            return random.random()*abs(limits[1]-limits[0])+limits[0]
        
        if randomInit:
            pos = []
            for i in range(self.nwalkers):
                pos.append([randomInRange(self.lim_r0),randomInRange(self.lim_dr1),randomInRange(self.lim_dr2),
                            randomInRange(self.lim_alpha),randomInRange(self.lim_beta),randomInRange(self.lim_gamma),
                            randomInRange(self.lim_b0)])
        else:
            # for simple minimizer
            out,_=self.fit_profile(x=x, prof=prof, #prof_err=self.cov_diag, \
                                   prof_name=self.fit_name,prof_cov=prof_cov)
            self.updateParams_fromPopt(out)
            pos0 = self.pfits.tolist()
            pos0=np.array([pos0,]*self.nwalkers)
            pos = pos0*(1 + 1e-2 * np.random.randn(self.nwalkers, self.ndim))
            for pi in np.arange(self.ndim): #adjust pos values such that they remain within limits
                pos[pi,:][pos[pi,:]>self.allParBounds[1][pi]]=self.allParBounds[1][pi] #for values too large, set to max in prior
                pos[pi,:][pos[pi,:]<self.allParBounds[0][pi]]=self.allParBounds[0][pi] #for values too small, set to min in prior
            
        
        # using multiprocessing, WORKS!
        print('using multiprocessing')
        from multiprocessing import Pool
        
        # if backend=None, then it won't save iteratively
        backend=None
        
        if saveBackendFilePath is not None:
            # Set up the backend
            # Don't forget to clear it in case the file already exists
            
            if os.path.isfile(saveBackendFilePath):
                backend = emcee.backends.HDFBackend(saveBackendFilePath)
                print("Initial size: {0}".format(backend.iteration))
                with Pool(processes=self.ncpus) as pool:
                    # initialize sampler with backend
                    sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, log_probability, 
                                                    args=(f, x, prof, self.icov, self.lim_r0, self.lim_dr1, self.lim_dr2, self.lim_alpha, self.lim_beta, self.lim_gamma, self.lim_b0), 
                                                    pool=pool,backend=backend)
                    
                    samples=backend.get_chain(flat=True)
                    n_new = int(abs(self.nsteps-samples.shape[0]))
                    print('n_new steps: %d' %n_new)
                    
                    start = time.time()
                    sampler.run_mcmc(None, n_new,store = True)
                    end = time.time()
        
                    multi_time = (end - start)/60
                    print("Multiprocessing took {0:.1f} min".format(multi_time))
                    
                pool.close()
            else:
                backend = emcee.backends.HDFBackend(saveBackendFilePath)
                backend.reset(self.nwalkers, self.ndim)
                with Pool(processes=self.ncpus) as pool:
                    # initialize sampler with backend
                    sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, log_probability, 
                                                    args=(f, x, prof, self.icov, self.lim_r0, self.lim_dr1, self.lim_dr2, self.lim_alpha, self.lim_beta, self.lim_gamma, self.lim_b0), 
                                                    pool=pool,backend=backend)
                    start = time.time()
                    sampler.run_mcmc(pos, self.nsteps,store = True)#, progress=True)
                    end = time.time()
        
                    multi_time = (end - start)/60
                    print("Multiprocessing took {0:.1f} min".format(multi_time))
                    
                pool.close()
        
        
        
        
        
        
        
        self.sampler = sampler
        
        self.flat_samples = sampler.get_chain(discard=self.burn_in_steps,flat=True)
        
        fit_par=[]
        for i in range(self.ndim):
            sigma_1=0.68269/2*100 #1 sigma
            mcmc = np.percentile(self.flat_samples[:, i], [50-sigma_1, 50, 50+sigma_1])
            q = np.diff(mcmc)
            fit_par.append([mcmc[1], q[0], q[1]]) #median, minus, plus
        fit_par=np.array(fit_par)
        self.emcee_fit_par=fit_par
        
        print('fit_par: ',fit_par)
        fits=[]
        for pi in range(len(self.par)):
            # sets up fit parameters to the initialized parameters, including bounds
            fits.append(self.emcee_fit_par[pi,0])
        self.updateParams_fromPopt(fits)
        
        toc_emcee=time.time()
        print('self.flat_samples.shape: ', self.flat_samples.shape)
        print('time to run emcee (m): %.2f' %((toc_emcee-tic_emcee)/60))
        
        '''
        # using mpi4py #takes way too long? i must be coding this incorrectly
        from mpi4py import MPI
        print('using mpi4py')
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if rank==0:
            print('comm.Get_size()=%d' %size)
        
        tic_rank=time.time()
        # use backends option of emcee to store chains 
        
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, log_probability, 
                                        args=(f, x, prof, self.icov, self.lim_r0, self.lim_dr1, self.lim_dr2, self.lim_alpha, self.lim_beta, self.lim_gamma, self.lim_b0), 
                                        )
        
        #start the actual mcmc process
        sampler.run_mcmc(pos, self.nsteps)#,progress=True)
        
        toc_rank=time.time()
        print('time for %d steps for rank %d (s): %.2f' %(self.nsteps,rank,(toc_rank-tic_rank)))
        
        '''
        '''
        # using schwimmbad MPIPool, doesn't work for me
        print('using MPIPool from schwimmbad')
        from schwimmbad import MPIPool
        with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, 
                                            log_probability, args=(f, x, prof, self.icov, self.lim_r0, self.lim_dr1, self.lim_dr2, self.lim_alpha, self.lim_beta, self.lim_gamma, self.lim_b0), 
                                            pool=pool)
            start = time.time()
            sampler.run_mcmc(pos, self.nsteps)#, progress=True)
            end = time.time()
            
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))
            pool.close()
        '''
        '''
        # Dont use this anymore! MPI no longer in emcee
        print('using MPIPool from emcee.utils')
        from emcee.utils import MPIPool
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, 
                                        log_probability, args=(f, x, prof, self.icov, self.lim_r0, self.lim_dr1, self.lim_dr2, self.lim_alpha, self.lim_beta, self.lim_gamma, self.lim_b0), 
                                        pool=pool)
        
        start = time.time()
        sampler.run_mcmc(pos, self.nsteps)#, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        pool.close()
        '''
        
        
        
        
        # commenting out because tau raises error for short chains
        # tau = sampler.get_autocorr_time() 
        # burnin = int(2 * np.max(tau)) # take out burn-in so chain "forget" where they started
        # thin = int(0.5 * np.min(tau)) # thin because posteriors are correlated
        # flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        # self.flat_samples=flat_samples
        
        
        # ##### lmfit fitting setup
        # fits=Parameters()
        # for pi in range(len(self.par)):
        #     # sets up fit parameters to the initialized parameters, including bounds
        #     fits.add(self.par_names[pi], value = self.emcee_fit_par[pi,0], vary=True,
        #               max=self.allParBounds[1][pi], min=self.allParBounds[0][pi])
        # self.updateParams_fromParametersObject(fits)
        
        # ##### run emcee
        # run_emcee(f=f,x=x,y=prof,icov=self.icov,saveFilePath=saveFilePath,
        #           nwalkers=self.nwalkers, burn_in_steps=int(self.nsteps/10),
        #           nsteps=self.nsteps,randomInit=randomInit)
          
        
        
        #update parameters
    def plot_walkerPositions(self,savePath=None):
        #plot walker positions (value of each parameter versus walker step)
        if self.sampler is None or self.flat_samples is None:
            raise Exception('need sampler from run_emcee() first!!')
        
        fig, axes = plt.subplots(self.ndim, figsize=(10, 18), sharex=True)
        samples = self.flat_samples
        labels = self.par_str_math
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        
        axes[-1].set_xlabel("step number");
        
        if savePath:
            fig.savefig(savePath)
    
    
    
    def plot_corner(self,savePath=None,truths=None,closefig=True):
        #plot corner plot of results
        # if true values are known, format is list 
        # (e.g. [r0, r1, r2, alpha, beta, gamma, b0])
        
        if self.sampler is None or self.flat_samples is None:
            raise Exception('need sampler from run_emcee() first!!')
            
        labels = self.par_str_math
        
        # fig=plt.figure()
        if truths is None:
            fig = corner.corner(self.flat_samples, labels=labels);
        else:
            #plot truth values following: https://corner.readthedocs.io/en/latest/pages/custom.html?highlight=truths%20color
            fig = corner.corner(self.flat_samples, labels=labels);#, truths=truths);
            # Extract the axes
            axes = np.array(fig.axes).reshape((self.ndim, self.ndim))
            # Loop over the diagonal
            for i in range(self.ndim):
                ax = axes[i, i]
                ax.axvline(truths[i], color="r", lw=2)
            # Loop over the histograms
            for yi in range(self.ndim):
                for xi in range(yi):
                    ax = axes[yi, xi]
                    ax.axvline(truths[xi], color="r")
                    ax.axhline(truths[yi], color="r")
                    ax.plot(truths[xi], truths[yi], "sr")
            
        
        if savePath:
            fig.savefig(savePath)
        if closefig:
            plt.close(fig)
            
    # This module might not be great to use, especially if plotting DeltaSigma
    # since it plots many fit examples, which ends up being a bit time consuming
    # def plot_individualFits(self, x, y, yerr, savePath=None):
    #     print('This code is not working properly yet!!!')
    #     x0=np.logspace(np.log10(x.min()),np.log10(x.max()),500)
        
    #     fig=plt.figure()
    #     if self.sampler is None or self.flat_samples is None:
    #         raise Exception('need sampler from run_emcee() first!!')
    #     inds = np.random.randint(len(self.flat_samples), size=100)
    #     for ind in inds:
    #         r0,dr1,dr2,alpha,beta,gamma,b0 = self.flat_samples[ind]
    #         plt.plot(x0, self.quantity[self.fit_name](x0,r0,dr1,dr2,alpha,beta,gamma,b0), "C1", alpha=0.1)
    #     # plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0, label="data")
    #     plt.fill_between(x, y+yerr, y-yerr, label="data",alpha=0.5)
    #     plt.legend(fontsize=14)
    #     plt.xlabel(r'$r \, [{\rm Mpc}/h]$')
    #     plt.ylabel(r'$%s$' %self.quantity_str_math[self.fit_name]);
    #     if savePath:
    #         fig.savefig(savePath)
    
    
    def _xi_mm_density(self,r):
        ret = self.rhom*(1.+self.xi_mm(r)) #[rhom] = M_dot h^2 / Mpc^3
        if self.meanExcessDensity: #if only inner density (taking)
            ret=ret-self.rhom
        return ret
    def xi_mm_density(self, r):
        return self._xi_mm_density(r) #[rhom] = M_dot h^2 / Mpc^3
    
    
    def _xi_mm_surfaceDensity(self,r):
        if self.showInputRWarnings is True:
            self.raiseExceptionOnInputR(r,self.current_prof_name)
        #[r]=Mpc/h
        # accuracy=1E-2
        
        # integrating in log space, gives factor of r in integrand
        def integrand_interpolate(logr, R2):
            r2 = (10**(logr))**2 #bc integrating in log space
            # interpolationFunction: log10( rho( log10( r ) ) )
            ret = r2 * (10**(interpFunction(logr))) / np.sqrt(r2 - R2)
            return ret *np.log(10) #this factor corrects for integrating in log space, with x=log10(r), dx/dr=(1/r)log10(e)!!
        def integrand_direct(logr, R2):
        # def integrand(logr, R2):
            r_i=10**logr
            r2 = (r_i)**2 #bc integrating in log space
            ret = r2 * self._xi_mm_density(r_i) / np.sqrt(r2 - R2)
            return ret *np.log(10) #this factor corrects for integrating in log space, with x=log10(r), dx/dr=(1/r)log10(e)!!
        
        if self.interpolate:
            # log_r_table=np.log10(np.logspace(np.log10(r.min()*(0.99)),np.log10(r.max()*(1.01)),1000))
            log_r_table=np.log10(np.logspace(self.log_r_min_integrate,self.log_r_max_integrate-0.001,50))
            
            rho_interp=self._xi_mm_density(10**log_r_table)
            
            log_r_table=log_r_table[rho_interp>0] #gets rid of negative densities
            rho_interp=rho_interp[rho_interp>0]
            
            # if np.min(rho_interp)<0:
            #     raise Exception('negative in density and cannot make interpolation table. self.par: ',self.par)
            # interpolationFunction: log10( rho( log10( r ) ) )
            interpFunction=scipy.interpolate.InterpolatedUnivariateSpline(log_r_table, np.log10(rho_interp))
            integrand=integrand_interpolate
            
        if self.interpolate is False:
            integrand=integrand_direct
        
        if hasattr(r, "__len__"): # does input r have length (array or list)?
            r_use = np.array(r)
        else:
            r_use=np.array([r]) #for integrating when calculating DeltaSigma
        surfaceDensity = np.zeros_like(r_use)
        log_r_use = np.log10(r_use)
        for i in range(len(r_use)):
        	surfaceDensity[i], _ = scipy.integrate.quad(integrand, log_r_use[i], self.log_r_max_integrate, 
        								args = (r_use[i]**2))#, epsrel = self.accuracy, limit = 1000)
        	# surfaceDensity[i] *= 2.0 #put it at the end
        return surfaceDensity*2 #[Sigma] = M_dot h / Mpc^2 
    
    def xi_mm_surfaceDensity(self,r):
        return self._xi_mm_surfaceDensity(r)
        
    def _xi_mm_excessSurfaceDensity(self,r):
        if self.showInputRWarnings is True:
            self.raiseExceptionOnInputR(r,self.current_prof_name)
        #[r]=Mpc/h
        
        # accuracy=1E-4
        
        # the excess surface density should not depend on mass sheets (mass sheet degeneracy)
        # so this acts as a switch to exclude self.rhom from the density before
        # any delta sigma calculations are done
        # self.meanExcessDensity=True
        
        # integrating in log space, gives factor of r in integrand
        def integrand_interpolate(log10_r):
            r2 = (10**(log10_r))**2 #bc integrating in log space
            # interpolationFunction: log10( Sigma( log10( r ) ) )
            ret = r2 * (10**(interpFunction(log10_r)))
            return ret *np.log(10) #this factor corrects for integrating in log space, with x=log10(r), dx/dr=(1/r)log10(e)!!
        
        def integrand_direct(log10_r):
            r_i = 10**(log10_r)
            r2 = r_i**2 #bc integrating in log space #this is due to change of units from x=log10_r => dx=rdx
            ret = (r2) * self._xi_mm_surfaceDensity(r_i) #[Sigma] = M_dot h / Mpc^2
            return ret *np.log(10) #this factor corrects for integrating in log space, with x=log10(r), dx/dr=(1/r)log10(e)!!
        
        
        # def integrand_interpolate(r_i):
        #     # interpolationFunction: log10( Sigma( log10( r ) ) )
        #     ret = r_i * (10**(interpFunction(np.log10(r_i))))
        #     return ret
        
        # def integrand_direct(r_i):
        #     ret = (r_i) * self._surfaceDensity(r_i,r0,dr1,dr2,alpha,beta,gamma,b0) #[Sigma] = M_dot h / Mpc^2
        #     return ret 
        
        # using interpolation of Sigma (100 R bins) seems slower than direct integration method
        if self.interpolate:
            r_table=np.logspace(self.log_r_min_integrate,self.log_r_max_integrate-0.001,100)
            log_r_table=np.log10(r_table)
            sigma_interp=self._xi_mm_surfaceDensity(r_table)
            if np.min(sigma_interp)<0:
                raise Exception('negative in density and cannot make interpolation table. self.par: ',self.par)
            # interpolationFunction: log10( Sigma( log10( r ) ) )
            interpFunction=scipy.interpolate.InterpolatedUnivariateSpline(log_r_table, np.log10(sigma_interp))
            # interpFunction=scipy.interpolate.interp1d(log_r_table,np.log10(sigma_interp),fill_value='extrapolate')
            integrand=integrand_interpolate
        if self.interpolate is False:
            integrand=integrand_direct
        
        # integrand=integrand_direct
        
        if hasattr(r, "__len__"): # does input r have length (array or list)?
            r_use = np.array(r)
        else:
            r_use=np.array([r])
        
        # this extends the inner profile for integration, to avoid differing 
        # DeltaSigma inner amplitudes due to integration
        # this will be thrown away at the end
        nbin_innerExtend=7
        if r.min()>self.log_r_min_integrate:
            r_inner_extend=np.logspace(self.log_r_min_integrate,np.log10(r.min()),nbin_innerExtend+1)
            r_use=np.concatenate((r_inner_extend[:-1],r_use))
        log_r_use=np.log10(r_use)
        
        
        
        if self.interpolate:
            Sigma=10**(interpFunction(log_r_use))
        else:
            Sigma = self._xi_mm_surfaceDensity(r_use)
        
        if self.integrateQuad:
            # This method WORKS NOW. integrate.quad
            meanSigma = np.zeros_like(r_use) #actually mean enclosed density within r_use
            meanSigma_0,_=scipy.integrate.quad(integrand, self.log_r_min_integrate, log_r_use[0], args = ())#, epsrel = accuracy, limit = 1000)
            meanSigma[0]=meanSigma_0
            for i in range(len(r_use))[1:]: 
                meanSigma_i, _ = scipy.integrate.quad(integrand, log_r_use[i-1], log_r_use[i], 
        				 						     args = ())#, epsrel = accuracy, limit = 1000)
                meanSigma[i] = meanSigma[i-1]+meanSigma_i
            meanSigma=meanSigma * 2.0 / r_use**2
        
        
        if self.integrateQuad is False:
            # This seems to work well, and compared to np.average method, it SEEMS a TINY
            # bit faster, and the inner amplitudes seem to change a bit
            rSigma=r_use*Sigma
            
            sum_SD=np.zeros(len(r_use))
            for riii in range(len(r_use)): 
                rsel=r_use<=r_use[riii]
                # sum_SD[riii] = np.sum(Sigma[rsel]*R[rsel]*dr[rsel])
                # r_uptoRlim=r_use[rsel] #current integration radius
                # SD_uptoRlim=Sigma[rsel] #selected surface density OUT TO radius
                rSigma_rsel=rSigma[rsel]
                
                sum_SD[riii]=scipy.integrate.simps(y=rSigma_rsel, x=r_use[rsel]) #integration out to radius for mean
            
            meanSigma=2*sum_SD/(r_use**2) #mult by 2 and divide out area to get mean
        
        
        meanSigma=meanSigma[nbin_innerExtend:]
        Sigma=Sigma[nbin_innerExtend:]
        
        self.meanSigma=meanSigma
        
        
        deltaSigma = meanSigma - Sigma
		
        # self.meanExcessDensity=False #turn off switch
        return deltaSigma
    
    def xi_mm_excessSurfaceDensity(self,r):
        return self._xi_mm_excessSurfaceDensity(r)
        
    

##### Make sure to match the radial bins from simulations!!! 
# Unfortunately the choice of radial bins impacts the enclosed densities
# In simulations using either the true simulation densities OR the bias fit densities
# the enclosed overdensities are still the same!
# so to match as closely to simulations make sure to use the same radial bins. 
#
r_edges=np.load(dataDir+'RadialBins.npy')
r_middle=np.sqrt(r_edges[:-1]*r_edges[1:])
r_middle[0]=r_edges[1]/np.sqrt(r_edges[2]/r_edges[1]) #middle of log bins



##### Change fit parameters and limits here!!!!
minEps=1e-10

# if no mass given
noMass_r0g=1 #1
noMass_dr1g=1 #1
noMass_dr2g=2.5 #2.5 or 2 or 3 most recently (04/23/20)
noMass_alphag=2 #2
noMass_betag=2 #2
noMass_gammag=0 #0
noMass_b0g=1 #1
noMass_init_guess=np.array([noMass_r0g, noMass_dr1g, noMass_dr2g, noMass_alphag, 
                            noMass_betag, noMass_gammag, noMass_b0g])
# noMass_lim_r0=[0.0,10] #[0.00,5] These are good 04/02/2020 @ 5:50 pm
# noMass_lim_dr1=[0.001,10] #[0.20,15]
# noMass_lim_dr2=[0.001,10] #[0.50,30]
# noMass_lim_alpha=[-20,30] #[0,50] 
# noMass_lim_beta=[-20,30] #[0,50]
# noMass_lim_gamma=[-20,30] #[0,70], or [0,20] most recently (04/23/20)
# noMass_lim_b0=[-20,20] #[-10,10] 
noMass_lim_r0=[0.0,5] #[0.00,5] These are good 04/02/2020 @ 5:50 pm
noMass_lim_dr1=[0.05,15] #[0.1,15]
noMass_lim_dr2=[0.05,30] #[0.1,30]
noMass_lim_alpha=[-10,30] #[0,50] 
noMass_lim_beta=[-10,30] #[0,50]
noMass_lim_gamma=[-20,30] #[0,70], or [0,20] most recently (04/23/20)
noMass_lim_b0=[-10,10] #[-10,10] 
#alllims = [lim_r0, lim_dr1, lim_dr2, lim_alpha, lim_beta, lim_gamma, lim_C0]
noMass_bParLims=np.array([ [noMass_lim_r0[0], noMass_lim_dr1[0], noMass_lim_dr2[0], noMass_lim_alpha[0], noMass_lim_beta[0], noMass_lim_gamma[0], noMass_lim_b0[0]], 
                 [noMass_lim_r0[1], noMass_lim_dr1[1], noMass_lim_dr2[1], noMass_lim_alpha[1], noMass_lim_beta[1], noMass_lim_gamma[1], noMass_lim_b0[1]] ])

# low mass limit log10Mvir<14
loMass_r0g=1 
loMass_dr1g=0.3 
loMass_dr2g=0.2
loMass_alphag=1.25 
loMass_betag=5 
loMass_gammag=0 #minEps #just in case, to avoid 0 in denominator
loMass_b0g=1 
loMass_init_guess=[loMass_r0g, loMass_dr1g, loMass_dr2g, loMass_alphag, 
                   loMass_betag, loMass_gammag, loMass_b0g]
#commented are those that worked before strict parameter limits were set; strict limits did not impact on the fits (i don't think)
loMass_lim_r0=[0.2,2.3] #[0,2.3] 04/09/21
loMass_lim_r0_z=[0.2,1.9] #for z>0, all r0<the higher limit here
loMass_lim_dr1=[0.2,1.5] #[0.2,2] 
loMass_lim_dr2=[0,2.5] #[0.2,1.5]
loMass_lim_alpha=[0.5,2] #[-1,2] 
loMass_lim_beta=[1,7] #[1,12.5] 
loMass_lim_gamma=[-.1,1] #[-.1,1] 
loMass_lim_b0=[-1,3] #[-1,7] 
loMass_bParLims=[loMass_lim_r0,loMass_lim_dr1,loMass_lim_dr2,loMass_lim_alpha,loMass_lim_beta,
                 loMass_lim_gamma,loMass_lim_b0]

# high mass limit log10Mvir>=14
hiMass_r0g=0.4 
hiMass_dr1g=0.5
hiMass_dr2g=0.5
hiMass_alphag=0.6
hiMass_betag=0 #minEps #just in case, to avoid 0 in denominator
hiMass_gammag=5
hiMass_b0g=4
hiMass_init_guess=[hiMass_r0g, hiMass_dr1g, hiMass_dr2g, hiMass_alphag, 
                   hiMass_betag, hiMass_gammag, hiMass_b0g]
#commented are those that worked before strict parameter limits were set; strict limits did not impact on the fits (i don't think)
hiMass_lim_r0=[0.3,1.5] #[0.3,1.5] 04/09/21
hiMass_lim_dr1=[0.5,2] #[0.5,2]
hiMass_lim_dr2=[0.25,2] #[0.25,2] 
hiMass_lim_alpha=[0.3,0.7] #[-1,0.7]
hiMass_lim_beta=[-1,1] #[-1,1] 
hiMass_lim_gamma=[4,6] #[-1,6] 
hiMass_lim_b0=[1,6] #[-1,7] 
hiMass_bParLims=[hiMass_lim_r0,hiMass_lim_dr1,hiMass_lim_dr2,hiMass_lim_alpha,hiMass_lim_beta,
                 hiMass_lim_gamma,hiMass_lim_b0]

# # for general fitting; used in Fong2020 paper, check biasTools.py


#commented are those that worked before strict parameter limits were set; strict limits did not impact on the fits (i don't think)
loosenPriors_lim_r0=[0,7] #[0.3,1.5] 04/09/21
loosenPriors_lim_dr1=[-1,10] #[0.5,2]
loosenPriors_lim_dr2=[0,10] #[0.25,2] 
loosenPriors_lim_alpha=[-1,5] #[-1,0.7]
loosenPriors_lim_beta=[-1,15] #[-1,1] 
loosenPriors_lim_gamma=[-3,7] #[-1,6] 
loosenPriors_lim_b0=[-1,10] #[-1,7] 
loosenPriors_bParLims=[loosenPriors_lim_r0,loosenPriors_lim_dr1,loosenPriors_lim_dr2,loosenPriors_lim_alpha,loosenPriors_lim_beta,
                       loosenPriors_lim_gamma,loosenPriors_lim_b0]

##### For emcee fitting!!!


# FLAT priors, will need to update in the future, muchos muchos tests required
def log_prior(theta, lim_r0, lim_dr1, lim_dr2, lim_alpha, lim_beta, lim_gamma, lim_b0):
    # theta is the INPUT varying parameter array. 
    r0, dr1, dr2, alpha, beta, gamma, b0 = theta
    r1=r0+dr1
    r2=r0+dr1+dr2
    if lim_r0[0] <= r0 <= lim_r0[1] and lim_dr1[0] <= dr1 <= lim_dr1[1] and lim_dr2[0] <= dr2 <= lim_dr2[1] and lim_alpha[0] <= alpha <= lim_alpha[1] and lim_beta[0] <= beta <= lim_beta[1] and lim_gamma[0] <= gamma <= lim_gamma[1] and lim_b0[0] <= b0 <= lim_b0[1]:
        #if theta is within limits, log_prior = 0 or prior probability=1
        if ( (r0==0)&((alpha+beta)<0) ) or ( (r1==0)&(beta+gamma<0) ) or ( (r2==0)&(gamma<0) ) or ( r1<0 ) or ( r2<0 ):
            #if model will give infinity, set prior probability=0
            return -np.inf 
        return 0
    return -np.inf #outside of limits, log_prior = - inf, or prior probability=0


def log_likelihood(theta, model, y, icov):
# def log_likelihood(theta): #emcee performs more quickly if x, y, err are global params
    r0, dr1, dr2, alpha, beta, gamma, b0 = theta
    
    # model = f(x, r0, dr1, dr2, alpha, beta, gamma, b0) #from given prof_name
    
    deltaFit=y-model
    
    return - 0.5 * np.dot( deltaFit, np.dot(icov,deltaFit) )
    # diff = y-model
    # return - 0.5 * np.dot( diff, np.dot(np.linalg.pinv(cov),diff) ) #when cov given instead of yerr
    

# def log_probability(theta): #emcee performs more quickly if x, y, err are global params
def log_probability(theta, f, x, y, icov, lim_r0, lim_dr1, lim_dr2, lim_alpha, lim_beta, lim_gamma, lim_b0):
    # FIRST check if current parameters within priors
    lp = log_prior(theta, lim_r0, lim_dr1, lim_dr2, lim_alpha, lim_beta, lim_gamma, lim_b0)
    if not np.isfinite(lp):
        return -np.inf # if prior probability = 0 (log_prior = - inf), set probability to 0 before calculations
    
    # SECOND make sure model does not contain NaN or inf
    r0, dr1, dr2, alpha, beta, gamma, b0 = theta
    model = f(x, r0, dr1, dr2, alpha, beta, gamma, b0)
    if (model[np.isnan(model)].shape[0]>0):
        return -np.inf
    if (model[np.isinf(model)].shape[0]>0):
        return -np.inf
    
    # for some reason there are still NaNs? I don't understand why, but try this last filter
    ret = lp + log_likelihood(theta, model, y, icov)
    if np.isnan(ret):
        return -np.inf
    
    # if above pass, then continue
    return ret
    # return lp + log_likelihood(theta)
    