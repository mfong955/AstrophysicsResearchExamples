
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


from scipy.optimize import curve_fit
#def redHan_rbias(r, r0, dr1, dr2, alpha, beta, gamma, C0):
def fit_bias_profile(x, prof, prof_err=None, xsel=None, logMass=None):
    #x=(RadialBinsExt_xmid/1e3)
    #if prof_err is None:
    #    prof_err=np.max([prof*1, siglin/CorrFuncExt], axis=0) #Always use prof_err = eb[i] (error in bin i)
    if xsel is None:
        xsel=x>x[4] #***May need to change the fit inner radius?
    
    #we want to fit the highest mass bins better, so we add a modification on the fit boundaries
    #if logMass>14:
    #    bounds=...
    
    #get rid of negative bias values
    ridInd=np.where(prof[xsel]<0)[0]
    x=np.delete(x[xsel],ridInd)
    prof=np.delete(prof[xsel],ridInd)
    if prof_err is not None:
        prof_err=np.delete(prof_err[xsel],ridInd)
    
    if logMass>14:
        allinitguesses4TwoMostMassiveBins, alllims4TwoMostMassiveBins=Lims4TwoMostMassiveBins()
        pfit, pcov = curve_fit(f=bias_rfunc, xdata=x, ydata=prof, sigma=prof_err,
                              p0=allinitguesses4TwoMostMassiveBins, bounds=alllims4TwoMostMassiveBins, max_nfev=1e8)#, ftol=1e-20, xtol=1e-10)
    else:
        #pfit, pcov = curve_fit(f=bias_rfunc, xdata=x, ydata=prof, sigma=None,
        pfit, pcov = curve_fit(f=bias_rfunc, xdata=x, ydata=prof, sigma=prof_err,
                              p0=allinitguesses, bounds=alllims, max_nfev=1e8)#, ftol=1e-20, xtol=1e-10)
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

#def bias_rfunc(r, r0, dr1, dr2, alpha, beta, gamma, b0):
#    return bias_func(r, r0, r0+dr1, r0+dr1+dr2, alpha, beta, gamma, b0)
