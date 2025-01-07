#Example usage

import ./bFitFunc

#if i is looping over your bias profiles:
#b[i] is one bias profile in one mass bin
#eb[i] is the error in the bias profile in one mass bin
#xsel is your fit range. You will likely need to change this. For example if you want to fit 
#   your entire range in x (and your range is greater than 0) then use xsel=x>0 
#   (boolean array of True values)
#massMeans[i] is the mean log10(Mvir [M_sun/h]) in one mass bin

#this returns the fit parameters (r0, dr1, dr2, alpha, beta, gamma, b0)
pfit=fit_bias_profile(x=x, prof=b[i], prof_err=eb[i],xsel=xsel,logMass=massMeans[i])[0]

#convert (r0, dr1, dr2, alpha, beta, gamma, b0) to (r0, r1, alpha, beta, C0)
r0, r1, r2, alpha, beta, gamma, C0 = convertParams(*pfit)

#bias fit profile and corresponding density
bFit=bias_func(x, r0, r1, r2, alpha, beta, gamma, C0)

plt.figure()
plt.plot(x,bFit)

