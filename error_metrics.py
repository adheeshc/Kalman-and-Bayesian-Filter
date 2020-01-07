import numpy as np
from scipy.stats import multivariate_normal

import sys
sys.path.insert(2, './Multivariate Kalman Filter')

from KF_Comparisons import *
from aux_classes import *

def NEES(xs,est_xs,P): #NORMALIZED ESTIMATED ERROR SQUARED
	est_err=xs-est_xs
	err=[]
	for x,p in zip(est_err,P):
		err.append(np.dot(np.dot(x.T,np.linalg.inv(p)),x))
	return err

def log_likelihood(): #Returns log-likelihood of the measurement  
    # hx = np.dot(kf2.H, x).flatten()
    # S = np.dot(H, P).dot(H.T) + R
    # likelihood = multivariate_normal.logpdf(z.flatten(), mean=hx, cov=S)
    # return likelihood
    likelihood=[]
    S=np.array(kf2.ss).flatten()
    y=np.array(kf2.yy).flatten()
    for i in range(0,len(S)):
        likelihood.append(multivariate_normal.logpdf(x=y[i],cov=S[i]))
    return likelihood


if __name__=="__main__":
    R=6
    Q=0.02
    dt=1

    xs,zs=simulate_acc_system(R,Q,count=80)
    kf2=SecondOrderKF(R,Q,dt,zs)
    #kf2.toPlot()
    #kf2.plotResiduals(xs, 3)
    est_xs,covs=kf2.out()

    nees=NEES(xs,est_xs,covs)
    eps=np.mean(nees)
    print(f'mean NEES is: {eps}')
    print(f'dim_x is: {kf2.dim_x}')
    if eps < kf2.dim_x:
        print('Good Filter!')
    else:
        print('FAIL!')

    loglike=log_likelihood()
    plt.plot(loglike)   
    plt.show()
        
    