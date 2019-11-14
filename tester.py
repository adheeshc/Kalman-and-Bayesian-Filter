from gh_filter import gH_filter
import numpy as np

def main():
	weights=np.array([158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 169.6, 167.4, 166.4, 171.0, 171.2, 172.6])
	init_est=160.0
	gh=gH_filter(weights,init_est,1.0,6./10,2./3,1.,plot=True)
	gh.toString()

if __name__=="__main__":
	main()