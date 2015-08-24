# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:29:12 2015

@author: Kyle Ellefsen
"""

from __future__ import division
import scipy.io
import numpy as np
from pyqtgraph import plot

filename=r'Z:\2015_08_19_FCS_in_hela_cell_cal520\trial4_20usinterval_cell3.mat'
f=scipy.io.loadmat(filename)
plot(np.squeeze(f['FCS_data']))

averaged_corr=np.squeeze(f['averaged_corr'].T)
autocorrcurve2 = self.autocorrplt.plot(self.x[1:], averaged_corr[1:],  pen=(255,0,255))
self.autocorrplt.removeItem(autocorrcurve2)

a # high laser, high aperature
b # low laser, high aperature
c # high laser, low aperature 

a=self.previous_averaged_corrs[-1]
b=self.previous_averaged_corrs[-2]
autocorrcurve2 = self.autocorrplt.plot(self.x[1:], a[1:],  pen=(255,0,255))
autocorrcurve3 = self.autocorrplt.plot(self.x[1:], b[1:],  pen=(0,255,255))