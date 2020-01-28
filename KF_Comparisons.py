import numpy as np
from scipy.linalg import block_diag
import sys
sys.path.insert(2, './Multivariate Kalman Filter')
from kalman_filter_m import *
from aux_classes import *
"""

A system to be filtered is designed.We simulate an object with constant velocity. 
Essentially no physical system has a truly constant velocity, so on each update we alter the velocity by a small amount. 

Now we compare the behaviors of 0 order, 1st order and 2nd order kalman filters for this problem
"""
def ZeroOrderKF(R_in, Q_in,measurements):
    x=np.array([0.])
    P=np.array([1.])
    X=Gaussian(x,P)
    F=np.eye(1)
    H=np.eye(1)
    B=[]
    u=[]
    Q=Q_in
    R=R_in
    kfm=Kalman_Filter_multi(X,F,H,B,Q,R, measurements,u)
    return kfm

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

def SecondOrderKF(R_in, Q_in, dt,measurements,P=100):
    x=np.zeros(3)
    P=np.array([P,1,1])
    X=Gaussian(x,P)
    F=np.eye(3)
    F[0][1]=dt
    F[1][2]=dt
    F[0][2]=0.5*dt**2
    H=np.array([[1,0,0]])
    B=[]
    u=[]
    Q=generateWhiteNoise(3,dt,Q_in)
    R=R_in**2
    kfm=Kalman_Filter_multi(X,F,H,B,Q,R, measurements,u)
    return kfm

if __name__=="__main__":
	R, Q = 1, 0.03
	xs, zs = simulate_vel_system(Q=Q, count=50)
	xs500, zs500 = simulate_vel_system(Q=Q, count=500)
	dt=1
	
	# kf0=ZeroOrderKF(R,Q,zs)
	# kf0.filter_details()
	# kf0.toPlot()
	# kf0.plotResiduals(xs,3)
	# print("\n======================================================================================================\n")
	# kf1=FirstOrderKF(R,Q,dt,zs)
	# kf1.filter_details()
	# kf1.toPlot()
	# kf1.plotResiduals(xs,1)
	print("\n======================================================================================================\n")
	kf2=SecondOrderKF(R,Q,dt,zs)
	kf2.filter_details()
	kf2.toPlot()
	#kf2.plotResiduals(xs,1)
	print("\n======================================================================================================\n")
	# IF WE SET PROCESS NOISE TO 0
	# kf2=SecondOrderKF(R,0,dt,zs500)
	# kf2.filter_details()
	# kf2.toPlot()
	# kf2.plotResiduals(xs500,1)
	# print("Although the filter fits measurements perfectly, the residual shows us its a bad fit for the problem at hand")

