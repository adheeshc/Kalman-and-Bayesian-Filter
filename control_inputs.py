import numpy as np
import sys
sys.path.insert(2, './Multivariate Kalman Filter')
from kalman_filter_m import *
from aux_classes import *

def FirstOrderKF(R_in, Q_in, dt,measurements,B_in=[],u_in=[],P=100):
    x=np.zeros(2)
    P=np.array([P,1])
    X=Gaussian(x,P)
    F=np.zeros((2,2))
    F[0][0]=dt
    H=np.array([[1,0]])
    B=B_in
    u=u_in
    Q=generateWhiteNoise(2,dt,Q_in)
    R=R_in
    kfm=Kalman_Filter_multi(X,F,H,B,Q,R, measurements,u)
    return kfm
    return kfm

if __name__=="__main__":
    R=3
    dt=1
    P=10
    Q=0.1
    B=np.array([[dt],[1]])
    u=np.array([1])
    #xs,zs=simulate_vel_system(Q,count=80)
    zs = np.array([i + np.random.randn()*R for i in range(1, 100)])
    kf2=FirstOrderKF(R,Q,dt,zs,B,u,P)
    kf2.toPlot()
    #kf2.plotResiduals(xs, 3)
