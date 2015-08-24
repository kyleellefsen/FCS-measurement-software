# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:46:43 2015

@author: Kyle Ellefsen
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from leastsqbound import leastsqbound





def exp_decay_1st_order(x, A1, t1):
    return A1 * np.exp(-x/t1) 
def exp_decay_2nd_order(x, A1, t1, A2, t2):
    return A1*np.exp(-x/t1) + A2*np.exp(-x/t2)
def exp_decay_3rd_order(x, A1, t1, A2, t2, A3, t3):
    return A1*np.exp(-x/t1) + A2*np.exp(-x/t2) + A3*np.exp(-x/t3)
    
def err(p, y, x):
    ''' 
    p is a tuple contatining the initial parameters.  p=(A1, t1)
    y is the data we are fitting to (the dependent variable)
    x is the independent variable
    '''
    order=len(p)/2
    if order==1:
        return y - exp_decay_1st_order(x,*p)
    elif order==2:
        return y - exp_decay_2nd_order(x,*p)
    elif order==3:
        return y - exp_decay_3rd_order(x,*p)

    
    



def fit_exponential(x,y,order=1):
    p0=(1,.0005)*order
    popt, cov_x, infodic, mesg, ier = leastsqbound(err, p0, args=(y,x), full_output=True)
    if order==1:
        yfit=exp_decay_1st_order(x, *popt)
    elif order==2:
        yfit=exp_decay_2nd_order(x, *popt)
    elif order==3:
        yfit=exp_decay_3rd_order(x, *popt)
    return popt,yfit
    
if __name__=='__main__':
    x, y = np.loadtxt(r'Z:\tmp\exponential_fitting.txt', unpack=True)
    x=x[1:1000]
    y=y[1:1000]
    popt,yfit=fit_exponential(x,y,3)
    from pyqtgraph import plot
    p=plot(x,y)
    p.plot(x, yfit,pen=(255,0,0))


