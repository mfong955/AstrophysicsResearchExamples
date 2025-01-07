#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:23:51 2021

Bias Tools

@author: MattFong
"""
##### Bias tools
import os, itertools, sys
import time
import numpy as np
from scipy.optimize import curve_fit
import scipy.integrate as integrate
from scipy.interpolate import interp1d

#import Colossus modules and input cosmology 
from colossus.utils import constants

from colossus.cosmology import cosmology as ccosmo
my_cosmo = {'flat': True, 'H0': 70.4, 'Om0': 0.268, 'Ob0': 0.044, 'sigma8': 0.83, 'ns': 0.968}
cosmo = ccosmo.setCosmology('my_cosmo', my_cosmo)



##### set up data directory, based on current 
codeDir=os.getcwd()+'/'
#data directory
cwdList=['/Users/MattFong/Desktop/Projects/Code/SJTU/','/home/mfong/Code/halo_prof/',
         '/home/mfong/Code/DECaLS/','/lustre/home/acct-phyzj/phyzj-m31/mfong/Code/DECaLS/',
         '/lustre/home/acct-phyzj/phyzj-m31/mfong/Code/']
print('cwd: '+os.getcwd())
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





#middle bin in log-space
def midBinLogSpace(r_rightEdge):
    r_rightEdge=np.concatenate(([0],r_rightEdge))
    r_mid=np.sqrt(r_rightEdge[:-1]*r_rightEdge[1:])
    r_mid[0]=r_rightEdge[1]/np.sqrt(r_rightEdge[2]/r_rightEdge[1])
    return r_mid


#Take derivative of F with respect to x for dlogF/dlogx

def derivative(F, x):
    #returns array dF/dx with length len(F)-1
    dF = []
    dx = []
    xmid=[]
    for i in np.arange(len(F)-1):
        dF.append(F[i+1] - F[i])
        dx.append(x[i+1] - x[i])
        xmid.append((x[i+1] + x[i])/2.)
    dF = np.array(dF)
    dx = np.array(dx)
    xmid = np.array(xmid)
    return dF/dx, xmid





def bias_func(r, r0, r1, r2, alpha, beta, gamma, b0):
    '''alpha>0, beta>=0?, gamma>=0: innerslope, void slope, outer slope.
    cluster: gamma~0
    void: b0<0, beta~0
    r2>r1>r0
    '''
    return (1+(r/r0)**-(alpha+beta))/(1+(r/r1)**-(beta+gamma))*(b0+(r/r2)**-gamma)


def bias_rfunc(r, r0, dr1, dr2, alpha, beta, gamma, b0):
    return bias_func(r, r0, r0+dr1, r0+dr1+dr2, alpha, beta, gamma, b0)

def convertParams(r0, dr1, dr2, alpha, beta, gamma, b0):
    return r0, r0+dr1, r0+dr1+dr2, alpha, beta, gamma, b0 #r0 < r1 < r2


#def redHan_rbias(r, r0, dr1, dr2, alpha, beta, gamma, C0):
def fit_bias_profile(x, prof, prof_err=None, xsel=None, logMass=None):
    #x=(RadialBinsExt_xmid/1e3)
    #if prof_err is None:
    #    prof_err=np.max([prof*1, siglin/CorrFuncExt], axis=0) #Always use prof_err = eb[i] (error in bin i)
    if xsel is None:
        xsel=x>0.06 #Mpc/h
    
    #we want to fit the highest mass bins better, so we add a modification on the fit boundaries
    #if logMass>14:
    #    bounds=...
    
    #get rid of negative bias values
    ridInd=np.where(prof[xsel]<0)[0]
    x=np.delete(x[xsel],ridInd)
    prof=np.delete(prof[xsel],ridInd)
    if prof_err is not None:
        prof_err=np.delete(prof_err[xsel],ridInd)
    
    if logMass is None:
        #pfit, pcov = curve_fit(f=bias_rfunc, xdata=x, ydata=prof, sigma=None,
        pfit, pcov = curve_fit(f=bias_rfunc, xdata=x, ydata=prof, sigma=prof_err,
                              p0=allinitguesses, maxfev=int(1e8), bounds=alllims)#, ftol=1e-20, xtol=1e-10)
    else:
        if logMass>14:
            allinitguesses4TwoMostMassiveBins, alllims4TwoMostMassiveBins=Lims4TwoMostMassiveBins()
            pfit, pcov = curve_fit(f=bias_rfunc, xdata=x, ydata=prof, sigma=prof_err,
                                  p0=allinitguesses4TwoMostMassiveBins, bounds=alllims4TwoMostMassiveBins, max_nfev=1e8)#, ftol=1e-20, xtol=1e-10)
        else:
            #pfit, pcov = curve_fit(f=bias_rfunc, xdata=x, ydata=prof, sigma=None,
            pfit, pcov = curve_fit(f=bias_rfunc, xdata=x, ydata=prof, sigma=prof_err,
                                  p0=allinitguesses, maxfev=int(1e8), bounds=alllims)#, 
    result=pcov
    return pfit, result



####################################    Set Limits!    ####################################
#***limits

r0g=1 #1
dr1g=1 #1
dr2g=2.5 #2.5 or 2 or 3 most recently (04/23/20)
alphag=2 #2
betag=2 #2
gammag=0 #0
b0g=1 #1
allinitguesses=np.array([r0g, dr1g, dr2g, alphag, betag, gammag, b0g])
lim_r0=[0.0,5] #[0.00,5] These are good 04/02/2020 @ 5:50 pm
lim_dr1=[0.2,15.0] #[0.20,15]
lim_dr2=[0.5,30.0] #[0.50,30]
lim_alpha=[0,50.0] #[0,50] 
lim_beta=[0,50.0] #[0,50]
lim_gamma=[0,70.0] #[0,70], or [0,20] most recently (04/23/20)
lim_b0=[-10.,10.0] #[-10,10] 
#alllims = [lim_r0, lim_dr1, lim_dr2, lim_alpha, lim_beta, lim_gamma, lim_C0]
alllims=np.array([ [lim_r0[0], lim_dr1[0], lim_dr2[0], lim_alpha[0], lim_beta[0], lim_gamma[0], lim_b0[0]], 
                 [lim_r0[1], lim_dr1[1], lim_dr2[1], lim_alpha[1], lim_beta[1], lim_gamma[1], lim_b0[1]] ])

def Lims4TwoMostMassiveBins():
    r0g=.4 #1
    dr1g=0.5 #1
    dr2g=1 #3 06/08/20; #2.5 #2.5 or 2 or 3 most recently (04/23/20)
    alphag=0.5 #2
    betag=0 #2
    gammag=1 #0
    b0g=1 #1
    allinitguesses=np.array([r0g, dr1g, dr2g, alphag, betag, gammag, b0g])
    lim_r0=[0.4,1.5] #[0.0,10] 06/08/20; #[0.00,5] These are good 04/02/2020 @ 5:50 pm
    lim_dr1=[0.5,5] #[0.1,15.0] 06/08/20; #[0.20,15]
    lim_dr2=[0.5,10] #[0.50,30]
    lim_alpha=[0,5.0] #[0,50] 
    lim_beta=[-2,5.0] #[0,50]
    lim_gamma=[0,10] #[0,70], or [0,20] most recently (04/23/20)
    lim_b0=[-1.,7.0] #[-10,10] 
    #alllims = [lim_r0, lim_dr1, lim_dr2, lim_alpha, lim_beta, lim_gamma, lim_C0]
    alllims=np.array([ [lim_r0[0], lim_dr1[0], lim_dr2[0], lim_alpha[0], lim_beta[0], lim_gamma[0], lim_b0[0]], 
                     [lim_r0[1], lim_dr1[1], lim_dr2[1], lim_alpha[1], lim_beta[1], lim_gamma[1], lim_b0[1]] ])
    return allinitguesses, alllims

################################################################################

# Define linear*step function here
def inverseHeavy(x, x0):
    #ret = 1 for x<=x0
    #ret = 0 for x>x0
    ret = np.zeros(x.shape[0])
    ret[x<=x0] = 1.
    return ret
    

def linStep(x, rfit, mfit, bfit):
    linFn = mfit*x+bfit
    stepFn = inverseHeavy(x, rfit)
    return linFn*stepFn


def fit_linStep(x, prof, prof_err=None):
    #x=(RadialBinsExt_xmid/1e3)
    if prof_err is None:
        prof_err = np.ones(x.shape[0])
    pfit, pcov = curve_fit(f=linStep, xdata=x, ydata=prof, sigma=prof_err,
                           bounds=linStepBounds, maxfev=1e6)
    return pfit, pcov
#linStepInitGuess=[4., 10, -20]
linStepBounds=[ [1, -200, -2000], [10, 200, 2000] ]


def getClosestIndex(R, rx):
    diff=abs(R-rx)
    return diff.argmin()

def calcRsp_min(Rmid, gamma, loLimit=None, hiLimit=None):
    ########## find rmin, where slope=0
    if loLimit is None:
        loLimit=3e-2
    if hiLimit is None:
        hiLimit = 7
    limXs = Rmid
    limYs = gamma

    rangeInds = np.where((limXs>=loLimit)&(limXs<=hiLimit))[0]
    if sum(rangeInds)==0:
        Rsp_min=0 #flag if nothing in this range or more likely nans
        gamma_min=0
    else:
        limXs=limXs[rangeInds]
        limYs=limYs[rangeInds]

        rminInds = np.r_[True, limYs[1:] < limYs[:-1]] & np.r_[limYs[:-1] < limYs[1:], True]
        Rsp_min = limXs[rminInds][0]
        gamma_min=limYs[rminInds][0]
    return Rsp_min, gamma_min

def bias2density(b, corrFunc, rhom):
    #b and corrFunc has same dim
    delta = b*corrFunc
    density = rhom*(delta+1.)
    return density
    

def calcGamma(x, y):
    #calculates exponential slope of y (dlogy/dlogx)
    logy=np.log10(y)
    logx = np.log10(x)
    gammay, xmid = derivative(logy, logx)
    xmid = 10**xmid
    return xmid, gammay

def bias2Rsp(R, bFit, corrFuncFit, rhom, Rsel=None, loLimit=None, hiLimit=None):
    if Rsel is None:
        Rsel=R>0.06 #from Fong2020, to fit only outer power-law
    R=R[Rsel]
    bFit=bFit[Rsel]
    corrFuncFit=corrFuncFit[Rsel]
    densityFit=bias2density(b=bFit,corrFunc=corrFuncFit, rhom=rhom)
    Rmid, gamma = calcGamma(x=R, y=densityFit)
    Rsp_min, gamma_min = calcRsp_min(Rmid=Rmid, gamma=gamma, loLimit=loLimit, hiLimit=hiLimit)
    return Rsp_min, gamma_min
    
def rho2Rsp(R, rho, Rsel=None, loLimit=None, hiLimit=None):
    if Rsel is None:
        Rsel=R>0.01 #from Fong2020, to fit only outer power-law
    R=R[Rsel]
    rho=rho[Rsel]
    Rmid, gamma = calcGamma(x=R, y=rho)
    X_min,X_minOptions=findRmin(X=Rmid,Y=gamma,Xlo=loLimit, Xhi=hiLimit)
    return X_min, X_minOptions
    

def findMin(X,Y,loLimit=None, hiLimit=None):
    ########## find rmin, where slope=0
    if loLimit is None:
        loLimit=3e-2
    if hiLimit is None:
        hiLimit = 5
    limXs = X
    limYs = Y

    rangeInds = np.where((limXs>=loLimit)&(limXs<=hiLimit))[0]
    limXs=limXs[rangeInds]
    limYs=limYs[rangeInds]
    X_min=limXs[limYs.argmin()]
    Y_min=limYs[limYs.argmin()]
    #rminInds = np.r_[True, limYs[1:] < limYs[:-1]] & np.r_[limYs[:-1] < limYs[1:], True]
    #X_min = limXs[rminInds][-1]
    #Y_min=limYs[rminInds][-1]
    return X_min, Y_min

def getRcd(R,biasFit,highMass_biasThreshhold, highMass_lowLimRcd=1,slopeThreshhold=-0.5, Xlo=None, Xhi=None, Ylo=None, Yhi=None):
    # THIS SHOULD BE USED FOR BIAS PRODUCED FROM DELTASIGMA FITS TO OBSERVATION DATA!!!
    # The reason for this function is to fine tune the Rcd estimation for the
    # highest mass bins, as their biases are not as perfect as simulation fits 
    # (originating from DeltaSigma fits)
    # works best if len(R) is large
    # used for accurately determining Rcd, but it requires more inputs
    # this needs to be adjusted depending on the dataset
    # highMass_biasThreshhold: bias.min() > highMass_biasThreshhold, means high mass
    # highMass_lowLimRcd: Rcd MUST be > 0.5 Mpc/h
    ########## find rmin, where slope=0
    if Xlo is None:
        Xlo = 0.06
    if Xhi is None:
        Xhi = 20
    if Ylo is None:
        Ylo=-np.inf
    if Yhi is None:
        Yhi=np.inf
    limXs = R 
    limYs = biasFit

    rangeInds = np.where((limXs>=Xlo)&(limXs<=Xhi)&(limYs>=Ylo)&(limYs<=Yhi))[0]
    limXs=limXs[rangeInds]
    limYs=limYs[rangeInds]
    
    X_min=limXs[limYs.argmin()] #this is here so X_min isn't referenced before assignment
    # Y_min=limYs[limYs.argmin()]
    rminInds = np.r_[True, limYs[1:] < limYs[:-1]] & np.r_[limYs[:-1] < limYs[1:], True]
    X_minOptions = limXs[rminInds]
    Y_minOptions = limYs[rminInds]
    
    if X_minOptions.shape[0]==0: #if there is no local minimum
        #I think this is just in case, since the edges or somewhere between will have one
        X_min=0
    
    # get rid of first and last options
    if X_minOptions.shape[0]>0:
        #delete these one at a time, since sometimes it'll make options size=0
        if X_minOptions[0] == limXs[0]: #if it is the first bin, delete
            X_minOptions=np.delete(arr=X_minOptions,obj=0)
            Y_minOptions=np.delete(arr=Y_minOptions,obj=0)
        if X_minOptions.shape[0]>0:
            if X_minOptions[-1] == limXs[-1]: #if it is the last bin, delete
                X_minOptions=np.delete(arr=X_minOptions,obj=-1)
                Y_minOptions=np.delete(arr=Y_minOptions,obj=-1)
    
    if (X_minOptions.shape[0]>0)&(limYs.min()<highMass_biasThreshhold): 
        # if there are still local minima
        # and if the bias is low (low mass) 
        # then the job is done!
        X_min=X_minOptions[Y_minOptions.argmin()] #choose minimum from Y_minOptions
    
   
        
    if limYs.min()>=highMass_biasThreshhold: #If the bias is high, then likely high mass bin!
        # these also should have no local minima. However some of them do due to all shapes
        # so get rid of values with low X_min options <= 0.5 Mpc/h
        Y_minOptions=Y_minOptions[X_minOptions>highMass_lowLimRcd]
        X_minOptions=X_minOptions[X_minOptions>highMass_lowLimRcd]
        if X_minOptions.shape[0]>0: #if there is still a minimum, maybe it is a true minimum!
            X_min=X_minOptions[Y_minOptions.argmin()]
            
        else:#cases where there are no local minima left
            X_min=0 #flagged. but this can be changed if this is a high mass bin
            # in this case, use slope of the bias and find where the slope approaches 0
            # but first limit range where X>=0.5, since these are cluster masses and Rcd > 0.5
            limYs=limYs[limXs>=highMass_lowLimRcd]
            limXs=limXs[limXs>=highMass_lowLimRcd]
            dF_dX, Xmid=derivative(F=limYs,x=limXs)
            newXlo_whenNoMinimum,_=findRmin(X=Xmid,Y=dF_dX, Xlo=Xlo, Xhi=Xhi, Ylo=Ylo, Yhi=Yhi)
            # change new lower limit to minimum of derivative of the bias (since derivative can vary wildly)
            # then 
            rangeInds=(Xmid>=newXlo_whenNoMinimum)&(dF_dX>=slopeThreshhold)
            X_minOptions=Xmid[rangeInds]
            if X_minOptions.shape[0]>0:
                X_min=X_minOptions.min()
    
    return X_min,X_minOptions


def findRmin(X,Y,Xlo=None, Xhi=None, Ylo=None, Yhi=None):
    # THIS WORKS BEST FOR SIMULATIONS!!! 
    # The binned bias in simulations is much cleaner!
    #this doesn't work as well for very complicated looking bias profiles
    #this is what i used for Fong2020 paper. I will use this for the new updated one
    
    ########## find rmin, where slope=0
    if Xlo is None:
        Xlo=3e-2
    if Xhi is None:
        Xhi = 20
    if Ylo is None:
        Ylo=-np.inf
    if Yhi is None:
        Yhi=np.inf
    limXs = X
    limYs = Y

    rangeInds = np.where((limXs>=Xlo)&(limXs<=Xhi)&(limYs>=Ylo)&(limYs<=Yhi))[0]
    limXs=limXs[rangeInds]
    limYs=limYs[rangeInds]
    X_min=limXs[limYs.argmin()]
    # Y_min=limYs[limYs.argmin()]
    rminInds = np.r_[True, limYs[1:] < limYs[:-1]] & np.r_[limYs[:-1] < limYs[1:], True]
    X_minOptions = limXs[rminInds]
    Y_minOptions = limYs[rminInds]
    
    if X_minOptions.shape[0]==0: #if there are no local minima
        X_min=0
    if X_minOptions.shape[0]>0:
        if X_minOptions[0] == limXs[0]: #if it is the first bin, delete
            X_minOptions=np.delete(arr=X_minOptions,obj=0)
            Y_minOptions=np.delete(arr=Y_minOptions,obj=0)
        if X_minOptions.shape[0]>0:
            if X_minOptions[-1] == limXs[-1]: #if it is the last bin, delete
                X_minOptions=np.delete(arr=X_minOptions,obj=-1)
                Y_minOptions=np.delete(arr=Y_minOptions,obj=-1)
        
        if sum(X_minOptions)==0: #if there are no local minima left
            X_min=0
        else:
            X_min=X_minOptions[Y_minOptions.argmin()] #choose minimum from Y_minOptions
    
    return X_min,X_minOptions
    # return X_min, Y_min

##### calc Rcd
def calcRmin(R, bFit, r, bData, plotExamples=None, logMass=None):
    #get rid of negative bias values
    ridInd=np.where(bFit<0)[0]
    R=np.delete(R,ridInd)
    bFit=np.delete(bFit,ridInd)

    ridInd=np.where(bData<0)[0]
    r=np.delete(r,ridInd)
    bData=np.delete(bData,ridInd)

    #slopes (exponential behavior of F) dlog(F)/dlog(x)
    logr=np.log10(r)
    logR=np.log10(R)

    logbData=np.log10(bData) #r
    logbFit=np.log10(bFit) #R

    slopebData, logrmid = derivative(F=logbData, x=logr)
    slopebFit, logRmid = derivative(F=logbFit, x=logR)

    DbData, logrmid = derivative(F=bData,x=logr)
    DbFit, logRmid, = derivative(F=bFit,x=logR)


    # rmid=10**logrmid
    Rmid=10**logRmid

    ########## find rmin, where slope=0
    limXs=R
    limYs = bFit
    #limYs[np.isnan(limYs)]=-np.inf
    loLimit=1e-1
    hiLimit=20
    #hiLimit = 2.

    rangeInds = np.where((limXs>=loLimit)&(limXs<=hiLimit)&(limYs<=10))[0]
    if sum(rangeInds)==0:
        rangeInds = np.where((limXs>=loLimit)&(limXs<=hiLimit)&(limYs<=20))[0]
        if sum(rangeInds)==0:
            rangeInds = np.where((limXs>=loLimit)&(limXs<=hiLimit))[0]
    if sum(rangeInds)==0:
        rmin=0. #flag if nothing in this range or more likely nans
        # rminstd=0
    else:
        rInRange=limXs[rangeInds]
        bInRange=limYs[rangeInds]

        rminInds = np.r_[True, bInRange[1:] < bInRange[:-1]] & np.r_[bInRange[:-1] < bInRange[1:], True]
        rmin1 = rInRange[rminInds]
        rmin1=rmin1[0] #ideal case, but there are cases where fit should flatten but doesn't

        #Collect possible Rmin values for ideal case, but will be taken out if below is satisfied:
        rminOptions=[]
        rminOptions.append(rmin1)

        ##### the following two cases are for the highest mass bins. 
        # 2nd highest mass bin choose min of data
        if logMass>14: #at masses above this threshhold there are no obvious minima, need to consider this as option
            rminOptions.append(r[bData[r<5].argmin()])
        # highest mass bin use linear fit to get rmin
        #add line fit to linear bias, and capture the location where profile deviates from linear bias
        linFitRangeInds=np.where((rInRange)>10)
        if len(linFitRangeInds[0])>2:
            linFitVal=np.polyfit(x=rInRange[linFitRangeInds],y=bInRange[linFitRangeInds],deg=0)
            bufferLinFit=.4
            linFitRangeInds=np.where((rInRange)>4)
            bInLinFitRange=bInRange[linFitRangeInds]
            rInLinFitRange=rInRange[linFitRangeInds]
            minBInRange=abs(bInLinFitRange-(linFitVal[0]+bufferLinFit)).argmin()
            rminLin=rInLinFitRange[minBInRange]
            rminOptions.append(rminLin)


        if rmin1 == rInRange[-1]: #when there is no local minimum above method chooses min of array, needs corrections

            if sum(np.isnan(slopebData))>0: #if there are nans in slopebData (cases where bias behaves weird, i.e. low # halos)
                rmin3 = 0
                #print('picking rmin=0')
            if sum(np.isnan(slopebData))==0: 
                #Collect possible Rmin values in case where min is last point in search range
                arrayRmins=[]
                arrayRmins.append(r[bData.argmin()]) #rmin2, minimum from data. most important for 2nd highest mass bin

                ##### other options for slopebFit
                #First rmin option is where the fitted linear step function of slope reaches 0
                #slopebDataFitRange= slopebData[slopebData.argmin():] #range where slope is linear step function, 0 after Rmin
                #rDataFitRange= rmid[slopebData.argmin():]
                #pfit4Rmin, pcov4Rmin = fit_linStep(rDataFitRange, slopebDataFitRange)
                #pfit4Rmin, pcov4Rmin = fit_linStep(rDataFitRange, slopebDataFitRange)
                slopebFitFitRange= slopebFit[slopebFit.argmin():] #range where slope is linear step function, 0 after Rmin
                rFitFitRange= Rmid[slopebFit.argmin():]
                pfit4Rmin, pcov4Rmin = fit_linStep(rFitFitRange, slopebFitFitRange)
                rmin=pfit4Rmin[0]
                rfit, mfit, bfit =pfit4Rmin
                yfit = mfit*Rmid+bfit
                absyfit = abs(yfit)
                arrayRmins.append(Rmid[absyfit.argmin()]) 

                #Second rmin option is where fitting linear function of slope reaches 0
                #linFitPar = np.polyfit(x=rDataFitRange,y=slopebDataFitRange,deg=1)
                #linFit=np.poly1d(linFitPar)
                #absLinFit=abs(linFit(Rmid))
                #arrayRmins.append(Rmid[absLinFit.argmin()])
                linFitPar = np.polyfit(x=rFitFitRange,y=slopebFitFitRange,deg=1)
                linFit=np.poly1d(linFitPar)
                absLinFit=abs(linFit(Rmid))
                arrayRmins.append(Rmid[absLinFit.argmin()])

                #Third rmin option is where slopebFit goes from negative to positive
                slopes=np.array(slopebFitFitRange)
                maxSumSlopes=abs(slopes).sum()
                if abs(sum(slopes))==maxSumSlopes: #All slopes are negative or positive
                    arrayRmins.append(0.) #flagged
                else:
                    slopesPositiveBool=slopes>0
                    if sum(slopesPositiveBool)>0:
                        #Choose where slope crosses from negative to positive
                        slopes[slopes>0]=0
                        arrayRmins.append(rFitFitRange[slopes==0.][0])
                    else:
                        arrayRmins.append(0.) #flagged


                #Pick lowest value of the options, with rmin > loLimit
                arrayRmins=np.array(arrayRmins)
                arrayRmins[arrayRmins<=loLimit]=0
                if sum(arrayRmins)==0.:
                    #rmin=rInRange[-1]
                    rmin3=0 #flagged rmins==0, i.e. no valid rmin values found using slopebData
                else:
                    rmin3=arrayRmins[arrayRmins>loLimit].min()


                
            rminOptions.append(rmin3)
        rminOptions=np.array(rminOptions)
        rmin = rminOptions[rminOptions>loLimit].min()
        # rminInd = getClosestIndex(limXs, rmin)
        # rminstd = (limXs[rminInd]-limXs[rminInd-1])/2. #not really std but nearest index
        
        if plotExamples is True:
            import matplotlib.pyplot as plt
            #shows test plots for rmin calculations, ONLY IF rmin == rInRange[-1] and sum(np.isnan(slopebData))==0, 
            #in other words, if bias has no minimum within range (chooses last value in range) and if there are no nans
            print('arrayRmins:')
            print(arrayRmins)
            plt.figure()
            plt.plot(rFitFitRange,slopebFitFitRange, label='orig slope')
            plt.plot(rFitFitRange,slopes, label='all positive slope')
            #def linStep(x, rfit, mfit, bfit):return linFn*stepFn
            linStepFit=linStep(Rmid,rfit,mfit,bfit)
            plt.plot(Rmid,linStepFit, ls='--', label='lin-step fit onto orig')
            plt.plot(Rmid,linFit(Rmid),ls=':', label='lin fit onto orig')
            #plt.plot(rDataFitRange[slopebDataFitRange<0],slopebDataFitRange[slopebDataFitRange<0],lw=3)
            # rmin_ind=getClosestIndex(Rmid,rmin)
            plt.plot(rmin3,0,marker='*',ms=20,mew=5,mfc='None',label=r'$r_{min}=%.2f$' %rmin)
            plt.axhline(0,ls=':')
            #plt.xscale('log')
            plt.legend(loc='best')
            plt.show()
    return rmin, rminOptions


def median_std(distribution, percent=None):
    if percent is None:
        percent=0.68269 #1 sigma
    
    Q=np.quantile(distribution,q=[0.5-percent/2,0.5,0.5+percent/2])
    
    low=Q[0]
    median=Q[1]
    high=Q[2]
    
    '''#Give array (distribution) and returns median (50%), median+sigma (84%), median-sigma (15%)
    sortedDistribution = np.array(np.sort(distribution))
    Ninds=sortedDistribution.size
    if Ninds % 2 == 0:
        perc50=sum(sortedDistribution[Ninds/2:Ninds/2+1])/2.
        perc16=sortedDistribution[int((Ninds-1)*.16)]
        perc84=sortedDistribution[int((Ninds-1)*.84)]
    if Ninds % 2 == 1:
        perc50=sortedDistribution[Ninds/2]
        perc16=sortedDistribution[int((Ninds)*.16)]
        perc84=sortedDistribution[int((Ninds)*.84)]'''
    return median, low, high

def resampleRminX1X2(parBin1,parBin2, proxy1=None, proxy2=None, Nresamples = None, rLo=None, displayTime=None,logMass=None):
    ticResRmin=time.time()
    ##### Resample Rmin Nresamples times
    if proxy1 is None:
        proxy1=xdata_full[0]
    if proxy2 is None:
        proxy2=xdata_full[4]
    if Nresamples is None:
        Nresamples = 100 #resampling rmin 1000 times
    if rLo is None:
        rLo = 0.06 #from Fong2020, to fit only outer power-law
    
    import random
    #parBin1=X1Bins[p] #e.g. array([12.89847244, 13.24801497])
    #parBin2=X2Bins[p] #e.g. array([-0.39697284, -0.31924556])
    
    bsRmins=[]
    
    sel=(proxy1>parBin1[0])&(proxy1<parBin1[1])&(proxy2>parBin2[0])&(proxy2<parBin2[1]) #selected halos in bins X1i and X2j
    sumsel=sum(sel)
    
    if sumsel==0.:
        print('Skipping bin p=%d, no halos in bin' %p)
    else:
        #Resample the selection, length of selected halos with repetitions
        allHaloInds=np.array(range(len(sel)))
        currentHaloInds=allHaloInds[sel]
        
        if sumsel<100:
            resSel=sel #don't resample selection
            b_sel=(overDensities[resSel])/(corrfunc)
            
            bmean=b_sel.mean(0)
            bstd =b_sel.std(0)/np.sqrt(sumsel)
            
            if sum(bstd)==0:
                bstd=None

            #get rid of negative bias values
            ridInd=np.where(bmean<0)[0]
            rPos=np.delete(r,ridInd)
            bPos=np.delete(bmean,ridInd)
            ebPos=np.delete(bstd,ridInd)

            #fit over 
            rFitSel=rPos>r[rInd]
            pfit=fit_bias_profile(rPos, bPos, prof_err=ebPos,xsel=rFitSel)[0]
            #pfit=fit_bias_profile(rPos, bPos, prof_err=ebPos,xsel=rFitSel)

            '''if fitAll is True:
                pfit=fit_bias_profile(r, bs[p], prof_err=ebs[p],xsel=rFitSel)[0] 
            if fitAll is False:
                rFitSel=rPlot>r[rInd]
                pfit=fit_bias_profile(rPlot, bData, prof_err=ebData,xsel=rFitSel)[0] '''

            r0, r1, r2, alpha, beta, gamma, C0 = convertParams(*pfit)

            #bias fit profile
            bFit=bias_func(R, r0, r1, r2, alpha, beta, gamma, C0)

            bsRmin, _ = calcRmin(R, bFit, rPos, bPos, plotExamples=False,logMass=logMass)
            '''if fitAll is True:
                bsRmin, _ = calcRmin(R, bFit, r, bmean, plotExamples=False)
            if fitAll is False:
                bsRmin, _ = calcRmin(R, bFit, rPos, bPos, plotExamples=False)'''
            bsRmins.append(bsRmin)
        else:
            for Nres_ind in range(Nresamples):
                resSel=np.array([random.choice(currentHaloInds) for r_ind in range(len(currentHaloInds))]) #resampled selection
                
                b_sel=(overDensities[resSel])/(corrfunc)
                
                bmean=b_sel.mean(0)
                bstd =b_sel.std(0)/np.sqrt(sumsel)
                
                if sum(bstd)==0:
                    bstd=None

                #get rid of negative bias values
                ridInd=np.where(bmean<0)[0]
                rPos=np.delete(r,ridInd)
                bPos=np.delete(bmean,ridInd)
                ebPos=np.delete(bstd,ridInd)

                #fit over 
                rFitSel=rPos>r[rInd]
                pfit=fit_bias_profile(rPos, bPos, prof_err=ebPos,xsel=rFitSel)[0]
                #pfit=fit_bias_profile(rPos, bPos, prof_err=ebPos,xsel=rFitSel)

                '''if fitAll is True:
                    pfit=fit_bias_profile(r, bs[p], prof_err=ebs[p],xsel=rFitSel)[0] 
                if fitAll is False:
                    rFitSel=rPlot>r[rInd]
                    pfit=fit_bias_profile(rPlot, bData, prof_err=ebData,xsel=rFitSel)[0] '''

                r0, r1, r2, alpha, beta, gamma, C0 = convertParams(*pfit)

                #bias fit profile
                bFit=bias_func(R, r0, r1, r2, alpha, beta, gamma, C0)

                bsRmin, _ = calcRmin(R, bFit, rPos, bPos, plotExamples=False,logMass=logMass)
                '''if fitAll is True:
                    bsRmin, _ = calcRmin(R, bFit, r, bmean, plotExamples=False)
                if fitAll is False:
                    bsRmin, _ = calcRmin(R, bFit, rPos, bPos, plotExamples=False)'''
                bsRmins.append(bsRmin)
        bsRmins=np.array(bsRmins)
        tocResRmin=time.time()
        if displayTime is True:
            print('Time resampling %d halos in calculation for Rmin %d times = %.2f s ' %(sumsel,Nresamples, (tocResRmin-ticResRmin)))
    return bsRmins


def resampleRminX(parBin1, proxy1, Nresamples = None, rLo=None, displayTime=None):
    ticResRmin=time.time()
    ##### Resample Rmin Nresamples times
    if Nresamples is None:
        Nresamples = 100 #resampling rmin 1000 times
    if rLo is None:
        rLo = 0.06 #from Fong2020, to fit only outer power-law
    
    import random
    #parBin1=X1Bins[p] #e.g. array([12.89847244, 13.24801497])
    
    bsRmins=[]
    
    sel=(proxy1>parBin1[0])&(proxy1<parBin1[1]) #selected halos in bins X1i
    sumsel=sum(sel)
    #if sum(sel)==0.:
    if sumsel<10:
        print('#halos < 10, using original Rmin')
        b_sel=(overDensities[sel])/(corrfunc)
        
        
        bmean=b_sel.mean(0)
        bstd =b_sel.std(0)/np.sqrt(sumsel)

        #get rid of negative bias values
        ridInd=np.where(bmean<0)[0]
        rPos=np.delete(r,ridInd)
        bPos=np.delete(bmean,ridInd)
        ebPos=np.delete(bstd,ridInd)

        if sum(ebPos)==0:
            ebPos=None

        #fit over 
        rFitSel=rPos> rLo
        pfit=fit_bias_profile(rPos, bPos, prof_err=ebPos,xsel=rFitSel)[0]
        #pfit=fit_bias_profile(rPos, bPos, prof_err=ebPos,xsel=rFitSel)

        r0, r1, r2, alpha, beta, gamma, C0 = convertParams(*pfit)

        #bias fit profile
        bFit=bias_func(R, r0, r1, r2, alpha, beta, gamma, C0)

        bsRmin, _ = calcRmin(R, bFit, rPos, bPos, plotExamples=False)
        bsRmins=np.array([bsRmin])
    else:
        #Resample the selection, length of selected halos with repetitions
        allHaloInds=np.array(range(len(sel)))
        currentHaloInds=allHaloInds[sel]
        
        for Nres_ind in range(Nresamples):
            resSel=np.array([random.choice(currentHaloInds) for r_ind in range(len(currentHaloInds))]) #resampled selection
            
            b_sel=(overDensities[sel])/(corrfunc)
            
            bmean=b_sel.mean(0)
            bstd = b_sel.std(0)/sum(resSel)
            
            #get rid of negative bias values
            ridInd=np.where(bmean<0)[0]
            rPos=np.delete(r,ridInd)
            bPos=np.delete(bmean,ridInd)
            ebPos=np.delete(bstd,ridInd)
            
            if sum(ebPos)==0:
                ebPos=None
                
            #fit over 
            rFitSel=rPos>r[rInd]
            pfit=fit_bias_profile(rPos, bPos, prof_err=ebPos,xsel=rFitSel)[0]
            #pfit=fit_bias_profile(rPos, bPos, prof_err=ebPos,xsel=rFitSel)
            
            r0, r1, r2, alpha, beta, gamma, C0 = convertParams(*pfit)

            #bias fit profile
            bFit=bias_func(R, r0, r1, r2, alpha, beta, gamma, C0)
            
            bsRmin, _ = calcRmin(R, bFit, rPos, bPos, plotExamples=False)
            bsRmins.append(bsRmin)
        bsRmins=np.array(bsRmins)
        tocResRmin=time.time()
        if displayTime is True:
            print('Time resampling %d halos in calculation for Rmin %d times = %.2f s ' %(sumsel,Nresamples, (tocResRmin-ticResRmin)))
    return bsRmins






def spherical2Cartesian(r,theta,phi):
    # spherical coordinates to cartesian
    #r, theta (RA [0,2pi]), phi (dec [pi/2,-pi/2])
    #input theta and phi in radians
    x=r*np.cos(theta)*np.cos(phi)
    y=r*np.sin(theta)*np.cos(phi)
    z=r*np.sin(phi)
    return x, y, z

def separation_cartesian(x1,y1,z1,x2,y2,z2):
    return np.sqrt( (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 )

#For fitting Rsp vs Rmin relation
def mxFit(x, m):
    return m*x

def fit_RspVsRmin(Rmins, Rsps):
    #[R_x]=Mpc/h
    pfit, _ = curve_fit(f=mxFit, xdata=Rmins, ydata=Rsps)
    return pfit


def enclosedMass_integrate(R, rhoFit, Rmin):
    Rintegrand=R[R<=Rmin]
    integrand = 4*np.pi*(Rintegrand**2.)*rhoFit[R<=Rmin]
    result = integrate.trapz(integrand, x=Rintegrand)
    return result

def enclosedMass(rho, rcut, z, rmax_simulation=30):
    #convert rho to rho_cumulative
    #overDensities is the density within (r_right^3 - r_left^3)
    #SPECIFIC FOR THIS SIMULATION at z=0
    if z==0:
        r_edges=np.load(file=dataDir+'RadialBins_z0.npy') #[r]=Mpc/h from kpc/h
    #local data dir FOR z>0
    if z>0:
        r_edges=np.load(file=dataDir+'RadialBins.npy')
        
    v_bin=4*np.pi/3*np.diff(r_edges**3.)
    r_binRight = r_edges[1:] #right edges
    
    if rcut==0:
        return 0
    else:
        #the maximum from simulation is rmax_simulation=30 Mpc/h
        mass_profile = np.cumsum(rho * v_bin) #rho is the stacked halo density profile in each bin, rho * vbin is the mass in each radial bins. Then we can get the enclosed mass profile by np.cumsum().
        f_interp = interp1d(np.log10(r_binRight), np.log10(mass_profile), kind="linear", fill_value="extrapolate")#Interpolate the enclosed mass profilers in log-log space.
        enclosed_mass = 10**f_interp(np.log10(rcut)) # enclosed mass at rd
    return enclosed_mass #[M]=M_sun/h


def enclosedMass_fromBiasProfile(biasProf, rcut):
    #convert rho to rho_cumulative
    #overDensities is the density within (r_right^3 - r_left^3)
    
    #bias profile is valid from 0.06
    r_edges=np.logspace(np.log10(0.06),1,1000) #[r]=Mpc/h from 0.06 to 10 
    
    r_middle=np.sqrt(r_edges[:-1]*r_edges[1:])
    r_middle[0]=r_edges[1]/np.sqrt(r_edges[2]/r_edges[1]) #middle of log bins
    
    rho=biasProf.density(r_middle)
    
    v_bin=4*np.pi/3*np.diff(r_edges**3.)
    r_binRight = r_edges[1:] #right edges
    
    if rcut==0:
        return 0
    else:
        #the maximum from simulation is rmax_simulation=30 Mpc/h
        mass_profile = np.cumsum(rho * v_bin) #rho is the stacked halo density profile in each bin, rho * vbin is the mass in each radial bins. Then we can get the enclosed mass profile by np.cumsum().
        f_interp = interp1d(np.log10(r_binRight), np.log10(mass_profile), kind="cubic", fill_value="extrapolate")#Interpolate the enclosed mass profilers in log-log space.
        enclosed_mass = 10**f_interp(np.log10(rcut)) # enclosed mass at rd
    return enclosed_mass #[M]=M_sun/h


def enclosedMass_fromRhoBinEdges(rho, r_middle, r_edges, rcut):
    #convert rho to rho_cumulative
    #overDensities is the density within (r_right^3 - r_left^3)
    
    
    v_bin=4*np.pi/3*np.diff(r_edges**3.)
    r_binRight = r_edges[1:] #right edges
    
    if rcut==0:
        return 0
    else:
        #the maximum from simulation is rmax_simulation=30 Mpc/h
        mass_profile = np.cumsum(rho * v_bin) #rho is the stacked halo density profile in each bin, rho * vbin is the mass in each radial bins. Then we can get the enclosed mass profile by np.cumsum().
        f_interp = interp1d(np.log10(r_binRight), np.log10(mass_profile), kind="linear", fill_value="extrapolate")#Interpolate the enclosed mass profilers in log-log space.
        enclosed_mass = 10**f_interp(np.log10(rcut)) # enclosed mass at rd
    return enclosed_mass #[M]=M_sun/h

def MxRx2Delta(Mx,Rx,rhox):
    #calculate enclosed density contrast
    #rhox is the density of the Universe, e.g rho_c or rho_m
    #[rhox]=M_dot h^2 / Mpc^3
    return 3.*Mx/(4.*np.pi*(Rx**3))/rhox




def v2mfr(r,v,rho):#differential mrf! because of how rho was calculated from simulation
    return rho*v*4*np.pi*r**2







def mmCorrFunc_z(R,snapshot):
    interpArray=np.load(dataDir+'corrFitFunc_%d.npy' %snapshot,allow_pickle=True)
    interp=interpArray.item()
    return 10**interp(np.log10(R)) #[xi_mm]=1
    # fileName='corrFitFunc_%d.npy' %snapshot
    # log10mmCorrFunc_z_coeffs=np.load(file=dataDir+fileName)
    # log10mmCorrFunc_z=np.poly1d(log10mmCorrFunc_z_coeffs)
    # return 10**log10mmCorrFunc_z(np.log10(R))

##### binned_bias
def binned_bias(proxy, z, x, corrfunc, densities, selectedPar=None, nbin=8, bintype='linear', delta_pad=0, norm=True, xsel=None):
    
    #x=r #[r]=Mpc/h
    
    #select redshift from snapshot
    # snapshot=redshiftBins[redbini][0]
    # z=redshiftBins[:,1][redshiftBins[:,0]==snapshot][0]
    # CorrFuncExt=np.load(file=dataDir+'CorrFuncExt_snapshot%d.npy' %snapshot)
    
    # mean matter density of universe
    rhom = cosmo.rho_m(z=z) #𝑀_sun h^2/kpc^3 ; 
    rhom = rhom*(1e3)**3. #[rhom] = M_dot h^2 / Mpc^3 from M_dot h^2 / kpc^3
    
    if selectedPar is None:
        selectedPar=~np.isnan(proxy)
    if bintype=='linear':
        parBin=np.linspace(proxy[selectedPar].min(), proxy[selectedPar].max(), nbin)
    elif bintype=='percent':
        parBin=np.percentile(proxy, np.linspace(0, 100, nbin))
    if xsel is None:
        xsel=x>0.06 #from Fong2020, to fit only outer power-law
    b=[]
    eb=[]
    rho=[]
    erho=[]
    parMean=[] #mean of parameters in bin
    pfits=[]
    massMeans=[]
    nPerBin=[]
    
    overDensities=densities/rhom-1
    #corrfunc=ps.CorrFunc(x)
    for i in range(len(parBin)-1):
        sel=(proxy>parBin[i])&(proxy<=parBin[i+1])&selectedPar
        sumsel=sum(sel)
        
        
        if sumsel==0:
            b.append(np.zeros(22))
            eb.append(np.zeros(22))
            rho.append(np.zeros(22))
            erho.append(np.zeros(22))
            parMean.append(np.zeros(22))
            massMeans.append(0)
            nPerBin.append(0)

        else:
            b_sel=(delta_pad+overDensities[sel])/(delta_pad+corrfunc)
            
            #bmean/=bmean[-1]
            b.append(b_sel.mean(0))
            eb.append(b_sel.std(0)/np.sqrt(sumsel))
            ### get density 1h term
            # delta=overDensities[sel].mean(0)
            # edelta=overDensities[sel].std(0)/np.sqrt(sum(sel))
            # rho.append(rhom*(delta+1))
            # erho.append(rhom*(edelta+1))
            rho_sel=rhom*(overDensities[sel]+1)
            rho.append(rho_sel.mean(0))
            erho.append(rho_sel.std(0)/np.sqrt(sumsel))
            
            parMean.append(proxy[sel].mean())
            #massMean = np.log10((10**xdata_full[0][sel]).mean()) #[massMean]=M_sun/h
            massMean = proxy[sel].mean() #[massMean]=log10(M_sun)
            massMeans.append(massMean)
            nPerBin.append(sumsel)

    b=np.array(b)
    eb=np.array(eb)
    
    # for i in np.arange(b.shape[0]):
    #     pfit=fit_bias_profile(x=x, prof=b[i], prof_err=eb[i],xsel=xsel,logMass=massMeans[i])[0] #***fitRangeMin 1e-1 Mpc/h
    #     #pfit=fit_bias_profile(x, b[i], prof_err=eb[i],xsel=xsel)
    #     pfits.append(pfit)
    
    rho=np.array(rho)
    erho=np.array(erho)
    pfits = np.array(pfits)
    parMean = np.array(parMean)
    massMeans=np.array(massMeans)
    parBin = np.array(parBin)
    nPerBin=np.array(nPerBin)
    return b, eb, rho, erho, parBin, massMeans, nPerBin #[massMean]=log10(M_sun/h)


##### binned 2d profiles
def binned_sigma(proxy, z, x, Sigmas_all, selectedPar=None, nbin=8, bintype='linear', norm=True, xsel=None):
    
    #x=r #[r]=Mpc/h
    
    if selectedPar is None:
        selectedPar=~np.isnan(proxy)
    if bintype=='linear':
        parBin=np.linspace(proxy[selectedPar].min(), proxy[selectedPar].max(), nbin)
    elif bintype=='percent':
        parBin=np.percentile(proxy, np.linspace(0, 100, nbin))
    if xsel is None:
        xsel=x>0.06 #from Fong2020, to fit only outer power-law
    
    binnedSigmas=[]
    binnedSigmaErr=[]
    binnedDeltaSigmas=[]
    binnedDeltaSigmaErr=[]
    
    
    for i in range(len(parBin)-1):
        tic_bin=time.time()
        sel=(proxy>parBin[i])&(proxy<=parBin[i+1])&selectedPar
        sumsel=sum(sel)
        
        print('starting bin i=%d, %d halos' %(i,sumsel))
        
        if sumsel==0:
            binnedSigmas.append(len(x))
            binnedSigmaErr.append(len(x))
            binnedDeltaSigmas.append(len(x))
            binnedDeltaSigmaErr.append(len(x))

        else:
            sigma_sel=Sigmas_all[sel]
            
            binnedSigmas.append(sigma_sel.mean(0))
            binnedSigmaErr.append(sigma_sel.std(0)/np.sqrt(sumsel))
            
            xSigma=x*sigma_sel
            
            meanSurfaceDensity_sel=np.empty(shape=xSigma.shape)
            for halo_i in range(xSigma.shape[0]):
                
                # print('starting bin halo_i=%d' %(halo_i))
                # tic_halo_i=time.time()
                xSigma_halo_i=xSigma[halo_i]
                
                for riii in range(len(x)): 
                    rsel=x<=x[riii]
                    meanSurfaceDensity_sel[halo_i,riii]=integrate.simps(y=xSigma_halo_i[rsel], x=x[rsel]) #integration out to radius for mean
                # toc_halo_i=time.time()
                # print('time for integration len(x)=%d times: %.2f min' %(len(x),(toc_halo_i-tic_halo_i)/60))
            
            meanSurfaceDensityInBin=2*meanSurfaceDensity_sel/(x**2) #mult by 2 and divide out area to get mean
            
            DeltaSigma_sel = meanSurfaceDensityInBin - sigma_sel
            binnedDeltaSigmas.append(DeltaSigma_sel.mean(0))
            binnedDeltaSigmaErr.append(DeltaSigma_sel.std(0)/np.sqrt(sumsel))
        
        toc_bin=time.time()
        print('time to bin i=%d: %.2f min' %(i,(toc_bin-tic_bin)/60))
        
    binnedSigmas=np.array(binnedSigmas)
    binnedSigmaErr=np.array(binnedSigmaErr)
    binnedDeltaSigmas=np.array(binnedDeltaSigmas)
    binnedDeltaSigmaErr=np.array(binnedDeltaSigmaErr)
    
    return binnedSigmas, binnedSigmaErr, binnedDeltaSigmas, binnedDeltaSigmaErr






# turn-around radius calculations

def calc_Rta_a(cosmo, a_z, mass_range=None, plotTest=False):
    ###########Calculates Rta(a_z) from theory, Tanoglidis2016 (arXiv:1601.03740)
    #cosmo is colossus cosmology object
    #a_z is scale factor at given redshift a(z)
    if mass_range is None:
        mass_range=M_plot
    ########### From Tanoglidis2016 (arXiv:1601.03740 eq. 2.19)
    #From Tanoglidis2016 (arXiv:1601.03740 eq. 2.7)
    #w=.705/0.295 #Korkidis2019
    w=cosmo.Ode0/cosmo.Om0

    xi=(1./cosmo.Om0)-1.-w

    #Need to solve for delta_ta(a). There are a few proposed ways:
    def RHSIntegrand(y):
        integrand=(np.sqrt(y))/np.sqrt(w*(y**3.) + xi*y/(a_z**2.) + 1./a_z**3.)
        return (w**(1/2.))*integrand
    RHSresult, _ = integrate.quad(func=RHSIntegrand,a=0.,b=1.)

    delta_ta_as=np.linspace(0,100,1000)
    def LHSIntegrand(u,delta_ta_a):
        integrand=(np.sqrt(u))/np.sqrt( (1.-u)*( (1.+delta_ta_a)/(w*(a_z**3.)) -u*(u+1.)  ) )
        return integrand
    LHSresults=[]
    for delta_ta_a in delta_ta_as:
        LHSresult,_ = integrate.quad(func=LHSIntegrand,a=0.,b=1.,args=(delta_ta_a))
        LHSresults.append(LHSresult)
    LHSresults=np.array(LHSresults)

    delta_ta_as=delta_ta_as[~np.isnan(LHSresults)]
    LHSresults=LHSresults[~np.isnan(LHSresults)]
    diffs=abs(LHSresults-RHSresult)
    LHSeqRHSIndex=diffs.argmin()
    delta_ta_a=delta_ta_as[LHSeqRHSIndex]
    #delta_ta_a=11 #from Korkidis2019
    #delta_ta_a=4.55 #critical Universe
    
    if plotTest is True:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,10))
        plt.plot(delta_ta_as, LHSresults, label = 'LHS results, a=%.1f' %a_z)
        plt.axhline(RHSresult, label='RHS result, a=%.1f' %a_z)
        plt.plot(delta_ta_a,LHSresults[LHSeqRHSIndex],marker='o',ms=10)
        plt.xlabel(r'$\delta_{\rm ta}(a=%.1f)$' %a_z)
        plt.ylabel('results')
    
    #From Tanoglidis2016 (Rta eq 2.7):
    M_star=(10**13.) #[M]= M_sun
    rho_c0=constants.RHO_CRIT_0_MPC3*(cosmo.h)**2. #[rho]=M_sun / Mpc^3 from M_sun h^2 / Mpc^3
    R_star=(3.*M_star/(4.*np.pi*rho_c0))**(1./3.) #[R]=Mpc

    #R_star=2.05*(cosmo.h)**(-2./3.) #from Tanoglidis2016, This is very close to R_star above; [R]=Mpc
    
    #Tanoglidis2016
    Rtas=R_star*(cosmo.Om0*(1.+delta_ta_a))**(-1./3.)*a_z*((mass_range)/M_star)**(1./3.) #[R]=Mpc
    
    return Rtas, delta_ta_a

def calc_Rcrit_a(cosmo, a_z, mass_range=None, plotTest=False):
    #critical overdensity that collapses at a_z=1
    ########### From Tanoglidis2015 eq. 2.7: THIS IS THE THRESHHOLD MINIMUM OVERDENSITY
    w=cosmo.Ode0/cosmo.Om0
    if mass_range is None:
        mass_range=M_plot
    #solve for mu_c
    def Nu_1_integ(x, nu):
        return (3./2.)*np.sqrt(x)/np.sqrt( (1.-x)*(-x**2. -x + nu) )
    nus=np.linspace(0,10,100)
    Nu_1=[]
    for nu in nus:
        Nu_i_int,_=integrate.quad(func=Nu_1_integ,a=0,b=1,args=(nu))
        Nu_1.append(Nu_i_int)
    Nu_1=np.array(Nu_1)

    a_eq28=(w**(-1./3.))*(np.sinh(Nu_1))**(2./3.)

    nus=nus[~np.isnan(a_eq28)]
    a_eq28=a_eq28[~np.isnan(a_eq28)]
    absdiff=abs(a_eq28-a_z)
    nu_c=nus[absdiff.argmin()]
    
    if plotTest is True:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(nus, a_eq28)
        plt.axhline(a_z)
        plt.plot(nu_c,a_eq28[absdiff.argmin()],marker='o',color='r',markersize=5)
        plt.xlabel('\nu_{\rm c}', fontsize=24)
        plt.ylabel('Equation 2.8', fontsize=24)

    def A_eq25_integrand(u):
        return ( u/( u**3. + 2.) )**(3./2.)
    def A_eq25(x):
        ret = ((x**3. + 2.)**(1./2.))/(x**(3./2.))
        Integral,_ = integrate.quad(func=A_eq25_integrand, a=0, b=x)
        return ret*Integral
    delta0_crit_a=( 3.*A_eq25( (2.*w)**(1./3.) )/(2.**(1./3.)) )*( (1.+nu_c)/(nu_c**(2./3.)) )
    # delta0_crit_a_min_present=(9./2.)*(A_eq25( (2.*w)**(1./3.) ))
    #print('delta_ta_a/delta_ta_a_min_present = %.2f' %(delta0_crit_a/delta0_crit_a_min_present))
    
    #From Tanoglidis2016 (Rta eq 2.7):
    M_star=(10**13.) #[M]= M_sun
    rho_c0=constants.RHO_CRIT_0_MPC3*(cosmo.h)**2. #[rho]=M_sun / Mpc^3 from M_sun h^2 / Mpc^3
    R_star=(3.*M_star/(4.*np.pi*rho_c0))**(1./3.) #[R]=Mpc
    
    R0_crit_a=R_star*(cosmo.Om0*(1.+delta0_crit_a))**(-1./3.)*a_z*((mass_range)/M_star)**(1./3.) #[R]=Mpc
    return R0_crit_a, delta0_crit_a #[R]=Mpc

def vfitfunc(R, a,b,c):
    #Rvir and Vvir are the virial radius and velocities
    #Rvir in units of R
    #Vvir in units of output
    v0=a*Vvir_i
    #test: bad test but worked! global values are working here
    #print('(Vvir_i,Rvir_i): (%.2f,%.2f)' %(Vvir_i, Rvir_i))
    return -v0*(R/Rvir_i)**(-b) + R*HUBBLE*(1e3)+c
    
    
    #return a*np.arctan(R/c)+b #best one so far
    #this makes a very messy non-reliable v_r: return (Vinfall + R*HUBBLE*(1e3))/V200ms +a#from Ceccarelli2005: cites Peebles1980
    #return (-a*(R/Rvirs)**(-b)+R*HUBBLE*(1e3))/V200ms #Albaek2017
    #not sure how I feel about this fit:return R*HUBBLE*(1e3)/V200ms+a #[H0]=km/s/(Mpc/h) from km/s/(kpc/h)
    #does not work well: return a*scipy.special.erf(z)+b


def Rta_searcher(r, v, r_loLim, vmax=None,testPlot=False,logMass=None):
    #v is the total velocity (NOT scaled by V200m): v_particle + v_hubble
    #output units same as input, but usually [v]=km/s
    #vmax is the maximum v_r(R) where Rta will be searched. This gets rid of weird v_r shapes, not approaching 0 for R>r_loLim
    if vmax is None:
        vmax=20.
    #get rid of nan values
    r=r[~np.isnan(v)]
    v=v[~np.isnan(v)]
    #chose lower bound such that R>r_loLom
    vfull=v
    rfull=r
    v=v[r>r_loLim]
    #delta_current=delta_i[r>r_loLim]
    r=r[r>r_loLim]
    
    r_ta=0.
    #reverse arrays, searching from highest r bins to lower ones
    v_search=np.array(v[::-1])
    # r_search=np.array(r[::-1])
    v_0crossInds=np.where((v_search[:-1] * v_search[1:])<=0)[0]
    if sum(v_0crossInds)==0:
        ##### Non-ideal case, where radial velocity does not go negative in range r>r_loLim
        #Can do this without reversed bins
        
        
        #smooth out bins
        numBins=int(sum(r>r_loLim)/3)
        r_leftEdgeSmooth = np.logspace(np.log10(r.min()), np.log10(r.max()), numBins) 
        v_smooth=[]
        r_smooth=[]
        v_std=[]
        #delta_smooth=[]
        for rbin in range(numBins-1):
            rselbin=(r>r_leftEdgeSmooth[rbin])&(r<r_leftEdgeSmooth[rbin+1])
            v_smooth.append(v[rselbin].mean())
            r_smooth.append(r[rselbin].mean())
            v_std.append(v[rselbin].std())
            #delta_smooth.append(delta_current[(r<r_leftEdgeSmooth[rbin+1])&(r>r_leftEdgeSmooth[rbin])].mean())
        v_smooth=np.array(v_smooth)
        r_smooth=np.array(r_smooth)
        v_std=np.array(v_std)
        
        #delta_smooth=np.array(delta_smooth)
        #find minimum of smoothed radial velocity to then fit over, to eventually find the root (where v_r \approx 0)
        if logMass<11.8:
            rSmoothMin=0.4
            v_smooth=v_smooth[r_smooth>rSmoothMin]
            v_std=v_std[r_smooth>rSmoothMin]
            r_smooth=r_smooth[r_smooth>rSmoothMin]
        v_min=v_smooth.argmin()
        v_fit=v_smooth[v_min:]
        r_fit=r_smooth[v_min:]
        v_fitstd=v_std[v_min:]
        if len(v_fit)<5:
            v_fit=v_smooth[v_min-10:]
            r_fit=r_smooth[v_min-10:]
            v_fitstd=v_std[v_min-10:]
        #delta_smooth=delta_smooth[v_min:]
        #***create fitting function here for this weird shape
        
        #print('len(r,v,delta)(%d,%d,%d)' %(r_smooth.shape[0],v_smooth.shape[0],delta_smooth.shape[0]))
        #Omega_0=cosmo.Ob0+cosmo.Ode0+cosmo.Ogamma0+cosmo.Ok0+cosmo.Om0+cosmo.Onu0+cosmo.Or0
        #delta_c=50
        #global Vinfall
        #Vinfall=-(1./3.)*Omega_0**(0.6)*HUBBLE*(1e3)*r_smooth*(delta_smooth*exp(-delta_smooth/delta_c)/(1.+delta_smooth)**(0.25)) #https://arxiv.org/pdf/astro-ph/0512160v2.pdf
        popt, pcov = curve_fit(f=vfitfunc, xdata=r_fit, ydata=v_fit,sigma=v_fitstd,maxfev=10000000)
        #print(*popt)
        r_vrFit=np.linspace(r_loLim, r.max(), 1000)
        fitvr=vfitfunc(r_vrFit,*popt)
        #fitvr=vfitfuncPlot(rfull,*popt)
        r_ta=r_vrFit[abs(fitvr).argmin()]
        v_ta=fitvr[abs(fitvr).argmin()]
    
    if sum(v_0crossInds)>0:
        ##### Ideal case, where radial velocity goes negative
        v_0FirstCross=v_0crossInds[0]
        #v[v.shape[0]-v_0FirstCross-1]-v_search[v_0FirstCross] #=0
        #v[v.shape[0]-v_0FirstCross-2]-v_search[v_0FirstCross+1] #=0
        #print('[v_cross],[v_rev_cross]:[%.2g,%.2g],[%.2g,%.2g]' 
        #      %(v[v.shape[0]-v_0FirstCross-1], v[v.shape[0]-v_0FirstCross-2], 
        #        v_search[v_0FirstCross], v_search[v_0FirstCross+1])) #same
        bracketOriginalInds=[v.shape[0]-v_0FirstCross-2, v.shape[0]-v_0FirstCross-1]
        
        rLineFit=r[bracketOriginalInds]
        vLineFit=v[bracketOriginalInds]
        lineFunction=np.poly1d(np.polyfit(x=rLineFit,y=vLineFit,deg=1))
        rLine=np.linspace(r[bracketOriginalInds[0]],r[bracketOriginalInds[1]],100)
        
        
        #rLineFit=r_smooth[bracketOriginalInds]
        #vLineFit=v_smooth[bracketOriginalInds]
        #lineFunction=np.poly1d(np.polyfit(x=rLineFit,y=vLineFit,deg=1))
        #rLine=np.linspace(r_smooth[bracketOriginalInds[0]],r_smooth[bracketOriginalInds[1]],100)
        vLineFunc=lineFunction(rLine)
        r_ta=rLine[abs(vLineFunc).argmin()]
        v_ta=vLineFunc[abs(vLineFunc).argmin()]
        
    if (r_ta/r_loLim < 1.2): #***add vmax limit here:
        r_ta=0.
        if testPlot:
            print('r_ta/r_loLim<1.2, flagged: rta=0')
    if (v_ta/vmax>1):
        r_ta=0.
        if testPlot:
            print('v_ta/vmax>1, flagged: rta=0')
    
    
    if testPlot:
        import matplotlib.pyplot as plt
        print('(r_ta/r_loLim,v_ta/v_max):(%.2f,%.2f)' %(r_ta/r_loLim,v_ta/vmax))
        
        plt.figure()
        #plt.title('Mvir=%.2g [M_sun/h]' %(Mvir_i))
        plt.axhline(0, ls=':',color='k')
        plt.plot(rfull, vfull, ls=':', color='grey', label=r'original $v_{\rm r}(r > r_{\rm 200m})$')
        plt.axvline(r_loLim, ls=':', color='k', lw=0.5)
        plt.axhline(vmax,ls=':',color='k',lw=0.5)
        plt.ylim([vfull.min(),vfull.max()])
        #plt.xscale('log')
        
        plt.plot(r_ta,v_ta, marker='o', markersize=10, label=r'$r_{\rm ta}= %.2f$' %r_ta)
        if sum(v_0crossInds)==0:
            plt.plot(r_vrFit, vfitfunc(r_vrFit,*popt), label='vfitfunc fit')
            plt.errorbar(r_smooth,v_smooth,yerr=v_std,lw=2,label=r'smoothed ydata fit, # bins = %d' %(numBins-1))
            #plt.plot(r_ta,vfitfunc(r_ta,*popt), marker='o', markersize=10, label=r'$r_{\rm ta}= %.2f$' %r_ta)
            #plt.plot(rfull, vfitfuncPlot(rfull,*popt), label='vfitfunc fit')
            #plt.axvline(r_ta,ls=':', color='r',label=r'$r_{\rm ta}= %.2f$' %r_ta)
            #print('r_ta/r_loLim=%.5f'%(r_ta/r_loLim))
        else:
            plt.plot(rLine,vLineFunc, label='linear fit between straddling points')
        plt.xlabel('r [Mpc/h]', fontsize=24)
        plt.ylabel('v [km/s]', fontsize=24)
        plt.legend(loc='best', fontsize=16)
        #plt.xlim([.2,.7])
        #plt.ylim([-100,100])
        plt.show()
    return r_ta



def sortR(R):
    #this sorts array R, but preserves the indices, so that the reults for 
    #bias, density, etc. can be properly calculated, 
    ii=R.argsort().argsort() #indices to sort R such that R_sorted = R[i]
    #example usage:
    # c=function(sort(R)) #function calculated with sorted (increasing) R
    # res=c[ii] #unsorted results function(R)
    return np.sort(R),ii


def percentDiff(guess,true):
    #negative is under predicting
    return 100*(guess-true)/true


def z2dz_logspaceBins(z,interp=False,testPlot=False):
    #finds dz around z positions for z[0]>0
    #so far only tested for z=np.logspace(...), not irregular binning
    #interp=True fits a polynomial to dz vs z, then interprets dz[-1]
    dz=np.zeros(z.shape[0])
    for zi in np.arange(z.shape[0]-1):
        dz[zi]=(z[zi+1]-z[zi-1])/2.
    dz[0]=(z[0]+z[1])/2.
    if interp:
        coeffs=np.polyfit(x=z[:-1],y=dz[:-1],deg=2)
        fitFunc=np.poly1d(coeffs)
        polyFit=fitFunc(z)
        dz[-1]=polyFit[-1]
        if testPlot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(z[:-1],dz[:-1],lw=3,color='b')
            plt.plot(z,dz,color='r')
            plt.xlabel('z')
            plt.ylabel('dz')
            plt.xscale('log')
            plt.yscale('log')
    else:
        dz[-1]=dz[-2]
    return dz


def cartesian2SphericalCoord(x,y,z):
    # (x,y,z) are the 1D arrays, e.g.: x = np.linspace(-fov, fov, 100)
    
    xy = x**2 + y**2
    r = np.sqrt(xy + z**2)
    phi = np.arctan2(np.sqrt(xy), z) # for elevation angle defined from z-axis down
    # phi = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    theta = np.arctan2(y, x) #on xy-plane, theta=0 starting from y=0
    return r,phi,theta


def getArrayFreqRepeats(X):
    Y = np.array([(x, len(list(y))) for x, y in itertools.groupby(X)])
    return Y

def getUniqueOfArray(X):
    Y = np.array([(x) for x, y in itertools.groupby(X)])
    return Y


def separation_angular(ra1,dec1,ra2,dec2):
    #inputs in degrees
    #input np.sin() is in radians
    ra1=np.deg2rad(ra1)
    dec1=np.deg2rad(dec1)
    ra2=np.deg2rad(ra2)
    dec2=np.deg2rad(dec2)
    
    cos_sep=np.sin(dec1)*np.sin(dec2)+np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    return np.arccos(cos_sep)*(180/np.pi) #[res]=deg from radians






##### Save data for binned bias by proxy1 and +/- quartiles of proxy2
#example usage save_binnedBiasData_splitC.py

def save_X1_X2quartiles_binned_bias(proxy1, proxy2, 
                                    z, x, corrfunc, densities,
                                    selectedPar=None, nbin=None, bintype='linear', 
                                    delta_pad=0, norm=True):
    # proxy1 needs to be one of these:
    # fileNames=['Mvir', 'Vmax', 'j', 'e', 'a_half', 'delta_e']
    # fileNames = np.array(fileNames)
    
    tic=time.time()
    
    # mean matter density of universe
    rhom = cosmo.rho_m(z=z) #𝑀_sun h^2/kpc^3 ; 
    rhom = rhom*(1e3)**3. #[rhom] = M_dot h^2 / Mpc^3 from M_dot h^2 / kpc^3
    
    ##### first set up first proxy bin
    if nbin is None:
        nbin=6
    if selectedPar is None: #preset limits on parameters, if NOT None
        selectedPar=(~np.isnan(proxy1))#&(~np.isnan(proxy2))
    if bintype=='linear': #set up linear bins for proxy1 and 2 (parameter values) in nbin bins
        parBin1=np.linspace(proxy1[selectedPar].min(), proxy1[selectedPar].max(), nbin)
        # parBin2=np.linspace(proxy2[selectedPar].min(), proxy2[selectedPar].max(), nbin)
    elif bintype=='percent':
        parBin1=np.percentile(proxy1, np.linspace(0, 100, nbin))
        # parBin2=np.percentile(proxy2, np.linspace(0, 100, nbin))
    
    b_hi=[]
    eb_hi=[]
    rho_hi=[]
    erho_hi=[]
    parMean_hi=[] #mean of parameters in bin
    pfits_hi=[]
    massMeans_hi=[]
    nPerBin_hi=[]
    
    b_lo=[]
    eb_lo=[]
    rho_lo=[]
    erho_lo=[]
    parMean_lo=[] #mean of parameters in bin
    pfits_lo=[]
    massMeans_lo=[]
    nPerBin_lo=[]
    
    overDensities=densities/rhom-1
    #corrfunc=ps.CorrFunc(x)
    for i in range(len(parBin1)-1):
        sel=(proxy1>parBin1[i])&(proxy1<=parBin1[i+1])&selectedPar
        sumsel=sum(sel)
        
        #skip the bin
        if sumsel==0:
            b_hi.append(np.zeros(22))
            eb_hi.append(np.zeros(22))
            rho_hi.append(np.zeros(22))
            erho_hi.append(np.zeros(22))
            parMean_hi.append(np.zeros(22))
            massMeans_hi.append(0)
            nPerBin_hi.append(0)
            
            b_lo.append(np.zeros(22))
            eb_lo.append(np.zeros(22))
            rho_lo.append(np.zeros(22))
            erho_lo.append(np.zeros(22))
            parMean_lo.append(np.zeros(22))
            massMeans_lo.append(0)
            nPerBin_lo.append(0)

        else:
            
            # get the quartile values
            percentiles=[0.25,0.5,0.75]
            
            # first pick out proxy2 from proxy1 bin!!!
            p2_lo,p2_med,p2_hi=np.quantile(proxy2[sel],q=percentiles)
            
            # now that we have the percentiles, choose quartiles AND those in same mass bin
            # high and low selections. do one at a time!
            sel_hi=(proxy2>=p2_hi)&sel
            sel_lo=(proxy2<=p2_lo)&sel
            
            ##### first use sel_hi !
            selp2=sel_hi
            sumselp2=sum(selp2)
            
            # bias
            b_sel=(delta_pad+overDensities[selp2])/(delta_pad+corrfunc)
            b_hi.append(b_sel.mean(0))
            eb_hi.append(b_sel.std(0)/np.sqrt(sumselp2))
            # density
            rho_sel=rhom*(overDensities[selp2]+1)
            rho_hi.append(rho_sel.mean(0))
            erho_hi.append(rho_sel.std(0)/np.sqrt(sumselp2))
            # mass
            parMean_hi.append(proxy1[selp2].mean())
            massMean = proxy1[selp2].mean() #[massMean]=log10(M_sun)
            massMeans_hi.append(massMean)
            nPerBin_hi.append(sumselp2)
            
            
            ##### then use sel_lo !
            selp2=sel_lo
            sumselp2=sum(selp2)
            
            # bias
            b_sel=(delta_pad+overDensities[selp2])/(delta_pad+corrfunc)
            b_lo.append(b_sel.mean(0))
            eb_lo.append(b_sel.std(0)/np.sqrt(sumselp2))
            # density
            rho_sel=rhom*(overDensities[selp2]+1)
            rho_lo.append(rho_sel.mean(0))
            erho_lo.append(rho_sel.std(0)/np.sqrt(sumselp2))
            # mass
            parMean_lo.append(proxy1[selp2].mean())
            massMean = proxy1[selp2].mean() #[massMean]=log10(M_sun)
            massMeans_lo.append(massMean)
            nPerBin_lo.append(sumselp2)
            
    # start with hi first
    b_hi=np.array(b_hi)
    eb_hi=np.array(eb_hi)
    
    for i in np.arange(b_hi.shape[0]):
        xsel=x>0.06 #Mpc/h
        bsel=(~np.isnan(b_hi[i]))&xsel
        pfit=bt.fit_bias_profile(x=x[bsel], prof=b_hi[i][bsel], prof_err=eb_hi[i][bsel],logMass=massMeans_hi[i])[0] #***fitRangeMin 1e-1 Mpc/h
        #pfit=fit_bias_profile(x, b[i], prof_err=eb[i],xsel=xsel)
        pfits_hi.append(pfit)
        
    rho_hi=np.array(rho_hi)
    erho_hi=np.array(erho_hi)
    pfits_hi=np.array(pfits_hi)
    parMean_hi=np.array(parMean_hi)
    massMeans_hi=np.array(massMeans_hi)
    parBin1=np.array(parBin1) #this one uses proxy1 par bins
    nPerBin_hi=np.array(nPerBin_hi)
    
    # then with lo
    b_lo=np.array(b_lo)
    eb_lo=np.array(eb_lo)
    
    for i in np.arange(b_lo.shape[0]):
        xsel=x>0.06 #Mpc/h
        bsel=(~np.isnan(b_lo[i]))&xsel
        pfit=bt.fit_bias_profile(x=x[bsel], prof=b_lo[i][bsel], prof_err=eb_lo[i][bsel],logMass=massMeans_lo[i])[0] #***fitRangeMin 1e-1 Mpc/h
        #pfit=fit_bias_profile(x, b[i], prof_err=eb[i],xsel=xsel)
        pfits_lo.append(pfit)
        
    rho_lo=np.array(rho_lo)
    erho_lo=np.array(erho_lo)
    pfits_lo=np.array(pfits_lo)
    parMean_lo=np.array(parMean_lo)
    massMeans_lo=np.array(massMeans_lo)
    parBin2=np.array(percentiles) #this one is just going to say the percentiles [25,75]
    nPerBin_lo=np.array(nPerBin_lo)
    
    np.savez(saveDataDir+'binnedBiasData_splitC_snapshot%d' %snapshot,
             b_lo=b_lo, eb_lo=eb_lo, rho_lo=rho_lo, erho_lo=erho_lo, pfits_lo=pfits_lo, 
             parBin2=parBin2, massMeans_lo=massMeans_lo, nPerBin_lo=nPerBin_lo,
             b_hi=b_hi, eb_hi=eb_hi, rho_hi=rho_hi, erho_hi=erho_hi, pfits_hi=pfits_hi, 
             parBin1=parBin1, massMeans_hi=massMeans_hi, nPerBin_hi=nPerBin_hi)
    
    toc=time.time()
    print('time to bin (s): %.2f' %((toc-tic)))
