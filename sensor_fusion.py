import numpy as np
import sys
sys.path.insert(2, './Multivariate Kalman Filter')
from kalman_filter_m import *
from aux_classes import *

def FirstOrderKF(R_in, Q_in, dt,measurements):
    x=np.zeros(2)
    P=np.array([100,1])

    X=Gaussian(x,P)
    F=np.eye(2)
    F[0][1]=dt
    H=np.array([[1,0]])
    B=[]
    u=[]
    Q=generateWhiteNoise(2,dt,Q_in)
    R=R_in
    kfm=Kalman_Filter_multi(X,F,H,B,Q,R, measurements,u)
    return kfm

def fusion():
	dt=
	

if __name__=="__main__":
