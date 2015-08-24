# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:39:28 2015
@author: Kyle Ellefsen
Fluorescence correlation spectroscopy measurement software

When this program is run, 
Make sure that the National Instruments DAC PCI-6132 is "Dev1" with the National Instruments Measurement and Automation tool.  

- Dev1/ai1 is the photodiode values


"""
from __future__ import division
import sys, os
import numpy as np
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyDAQmx import *
from PyDAQmx.DAQmxCallBack import *
from numpy import zeros
import pyqtgraph as pg
from pyqtgraph import plot
from pyqtgraph.dockarea import *
import random
import scipy.io
import time
from scipy.fftpack import fft, fftfreq, fftshift
from exponential_fitting import fit_exponential


sampleInterval=20 # microseconds
# To get all the data collected so far, call
#data=np.array(self.analog_data).ravel()

def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

class DiodeValueWidget(QWidget) :
    """
    This widget is what actually displays the circle.
    """
    def __init__(self, parent = None) :
        QWidget.__init__(self, parent)
        self.setGeometry(9, 32, 1268, 600)
        self.setWindowTitle( "FCS Measurement software" )
        
        ## set layout ##
        self.l = QVBoxLayout()
        self.setLayout(self.l)
        self.area = DockArea()
        self.l.addWidget(self.area)
        
        ## plotting live input
        self.d1 = Dock("Live input",size=(1164, 437))
        self.area.addDock(self.d1)
        self.plt=pg.PlotWidget()
        self.d1.addWidget(self.plt)
        self.plt.showGrid(x=True, y=True)
        self.x=np.arange(100,dtype=np.float64)+1
        self.y=np.zeros(100,dtype=np.float64)+1
        self.curve = self.plt.plot(self.x, self.y, pen=(255,0,0))
        self.plt.setYRange(0,70)
        self.plt.setLabel('bottom','Time',units='s')
        self.plt.setLabel('left','Photon counts')
        
        ## plotting histogram ##
        self.d2 = Dock("Histogram")
        self.area.addDock(self.d2,'right',size=(382, 216))
        self.histplt=pg.PlotWidget()
        self.d2.addWidget(self.histplt)
        self.histplt.showGrid(x=True, y=True)
        self.hist = self.histplt.plot(self.x, self.y[:-1],  fillLevel=0, brush=(0,0,255,150), stepMode=True) #,
        self.histplt.setXRange(-1,100)
        self.histplt.setLabel('bottom','Number of Photons')
        self.histplt.setLabel('left','Counts')
        
        ## plotting fft ##
        self.d3 = Dock("FFT")
        self.area.addDock(self.d3,'bottom',self.d2,size=(382, 216))
        self.fftplt=pg.PlotWidget()
        self.d3.addWidget(self.fftplt)
        self.fftplt.showGrid(x=True, y=True)
        self.fftplt.setLogMode(x=True,y=True)
        self.fftcurve = self.fftplt.plot(self.x, self.y,  pen=(0,225,0))
        self.fftplt.setLabel('bottom','Frequency')
        self.fftplt.setLabel('left','Absolute Value of Amplitude ')
        
        ## plotting autocorr ##
        self.d4 = Dock("Auto Correlation")
        self.area.addDock(self.d4,'bottom',self.d3,size=(382, 216))
        self.autocorrplt=pg.PlotWidget()
        self.d4.addWidget(self.autocorrplt)
        self.autocorrplt.showGrid(x=True, y=True)
        self.autocorrcurve = self.autocorrplt.plot(self.x, self.y,  pen=(255,225,0))
        #self.autocorrplt.setXRange(0,100)
        self.autocorrplt.setYRange(-.1,.1)
        self.autocorrplt.setXRange(0,.1)
        self.autocorrplt.setLabel('bottom','Lag Times',units='s')
        self.autocorrplt.setLogMode(x=False,y=False)
        
        
        ## buttons ##
        self.control_panel=QGridLayout()
        self.sampleInterval_spinbox =QDoubleSpinBox()
        self.sampleInterval_spinbox.setSingleStep(.01)
        self.sampleInterval_spinbox.setDecimals(4)
        self.sampleInterval_spinbox.setMinimum(1)
        self.sampleInterval_spinbox.setMaximum(10000)
        self.sampleInterval_spinbox.setValue(sampleInterval)
        self.sampleInterval_spinbox.valueChanged.connect(self.setSampleInterval)
        self.stop=False
        self.start_stop_button=QPushButton('Stop')
        self.start_stop_button.pressed.connect(self.start_stop)
        self.averageOn=False
        self.nAveraged=0
        self.nAveraged_corr=0
        self.averaged_corr=None
        self.previous_averaged_corrs=[]
        self.averageButton=QPushButton('Average')
        self.averageButton.pressed.connect(self.altAverage)
        self.exportButton=QPushButton('Export Raw Trace')
        self.exportButton.pressed.connect(self.export_gui)
        self.exportCorrButton=QPushButton('Export Correlation')
        self.exportCorrButton.pressed.connect(self.export_corr_gui)
        self.control_panel.addWidget(QLabel('Sample Interval (microseconds)'),0,0)
        self.control_panel.addWidget(self.sampleInterval_spinbox,0,1)
        self.control_panel.addWidget(self.start_stop_button,1,1)
        self.control_panel.addWidget(self.exportButton,2,0)
        self.control_panel.addWidget(self.exportCorrButton,2,1)
        self.samplesAveragedLabel=QLabel('Samples Averaged: 0')
        self.control_panel.addWidget(self.samplesAveragedLabel,4,0)
        self.control_panel.addWidget(self.averageButton,4,1)
        
        self.control_panel.addWidget(QLabel('Order of Exponential Fit'),5,0)
        self.exp_order=QComboBox()
        self.exp_order.addItem('1')
        self.exp_order.addItem('2')
        self.exp_order.addItem('3')
        self.control_panel.addWidget(self.exp_order,5,1)
        self.fitButton=QPushButton('Fit')
        self.fitButton.pressed.connect(self.fit_exponential)
        self.control_panel.addWidget(self.fitButton,5,2)
        self.exponential_label=QLabel('Parameters:')
        self.control_panel.addWidget(self.exponential_label,5,3)
        
        self.control_panelWidget=QWidget()
        self.control_panelWidget.setLayout(self.control_panel)
        self.control_panelWidget.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
        self.l.addWidget(self.control_panelWidget)
        
        ## acquire data from analog input thread##
        self.analog_data=[] #np.zeros(100,dtype=np.float64)
        self.timer=QTimer()
        self.timer_slow=QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.timeout.connect(self.updateHistogram)
        self.timer_slow.timeout.connect(self.updatefftplt)
        self.timer_slow.timeout.connect(self.updateAutoCorr)
        self.timer.start(50)
        self.timer_slow.start(80)
        
        self.analogInputThread=AnalogInputThread(self)
        self.analogInputThread.start()
    def paintEvent( self, event ) :
        if len(self.analog_data)==0:
            pass
        else:
            y=np.array(self.analog_data[-10:])
            self.y=y.ravel()
            self.x = np.linspace(0.0, len(self.y)*sampleInterval, len(self.y))/1000000. #microseconds to seconds
            #self.y=self.analog_data[-1]
        self.curve.setData(self.x, self.y)
        pass

    def updateHistogram(self):
        #unique,counts=np.unique(np.array(self.analog_data[-100:]),return_counts=True)
        if len(self.analog_data)>1:
            a=np.array(self.analog_data[-10:])
            counts,bin_edges=np.histogram(a,bins=np.arange(0,200))
            self.hist.setData(bin_edges,counts)

    def updatefftplt(self):
        a=np.array(self.analog_data[-10:]).ravel()
        N = 8192
        sampleInterval_S=sampleInterval/1000000  #microseconds to seconds
        if len(a)>N:
            # sample spacing
            x = np.linspace(0.0, N*sampleInterval_S, N)
            yf = fft(a[-N:])
            xf = fftfreq(N, sampleInterval_S)
            xf=xf[1:N/2]
            yf=np.abs(yf[1:N/2])
            if self.averageOn==True:
                self.averaged_fft=(self.averaged_fft*self.nAveraged+yf)/(self.nAveraged+1)
                self.nAveraged+=1
                self.fftcurve.setData(xf, self.averaged_fft)
                self.samplesAveragedLabel.setText('Samples Averaged: {}'.format(self.nAveraged))
            else:
                self.fftcurve.setData(xf, yf)    
    def updateAutoCorr(self):
        a=np.array(self.analog_data[-10:]).ravel()
        if len(a)>1000:
            corr=estimated_autocorrelation(a)
            x = np.linspace(0.0, len(a)*sampleInterval, len(a))/1000000. #microseconds to seconds
            if self.averageOn==True and len(a)==10000:
                self.averaged_corr=(self.averaged_corr*self.nAveraged_corr+corr)/(self.nAveraged_corr+1)
                self.nAveraged_corr+=1
                self.autocorrcurve.setData(x[1:],self.averaged_corr[1:])
            else:
                self.autocorrcurve.setData(x[1:],corr[1:])
    def fit_exponential(self):
        if self.averageOn:
            y=self.averaged_corr
            x=np.linspace(0.0, len(y)*sampleInterval/1000000., len(y), endpoint=False) #microseconds to seconds
            y=y[1:]
            x=x[1:]        
            order=int(self.exp_order.currentText())
            popt,yfit=fit_exponential(x,y,order)
            self.autocorrfitcurve = self.autocorrplt.plot(x, yfit,  pen=(0,225,255))
            if order==1:
                text="A1*exp(x/t1), A1={}, t1={}".format(*popt)
            elif order==2:
                text="A1*exp(x/t1)+A2*exp(x/t2), A1={}, t1={}, A2={}, t2={}".format(*popt)
            elif order==3:
                text="A1*exp(x/t1)+A2*exp(x/t2)+A3*exp(x/t3), A1={}, t1={}, A2={}, t2={},A3={}, t3={}".format(*popt)
        else:
            text="You must be averaging the autocorrelation in order to fit it"
        self.exponential_label.setText(text)
    def altAverage(self):
        if self.averageOn:
            self.averageButton.setText('Average')
            self.samplesAveragedLabel.setText('Samples Averaged: 0')
            self.averageOn=False
        else:
            self.averageButton.setText('Stop Averaging')
            self.nAveraged=0
            self.nAveraged_corr=0
            N=8192
            self.averaged_fft=np.zeros(N/2-1)
            N=10000
            if self.averaged_corr is not None:
                self.previous_averaged_corrs.append(np.copy(self.averaged_corr))
            self.averaged_corr=np.zeros(N)
            self.averageOn=True

    def start_stop(self):
        if self.stop==False:
            self.stop_input_thread()
        else:
            self.start_input_thread()
            
    def stop_input_thread(self):
        if self.stop==False:
            self.analogInputThread.stop=True
            self.timer.stop()
            self.timer_slow.stop()
            self.start_stop_button.setText('Start')
            self.stop=True
            print('Input thread stopped')
        
    def start_input_thread(self):
        if self.stop==True:
            self.analog_data=[]
            self.timer.start(20)
            self.timer_slow.start(50)
            self.analogInputThread=AnalogInputThread(self)
            self.analogInputThread.start()
            self.start_stop_button.setText('Stop')
            self.stop=False
            print('Input thread restarted')
        
    def setSampleInterval(self,value):
        global sampleInterval
        self.stop_input_thread()
        sampleInterval=value
        time.sleep(.1)
        self.start_input_thread()


    def export_gui(self):
        if self.stop==False:
            self.stop_input_thread()
        filename= QFileDialog.getSaveFileName(self, 'Export data', r'C:/Users/Admin/Desktop/','*.mat')
        filename=str(filename)
        if filename=='':
            return False
        else:
            self.export(filename)
    def export(self,filename=None):
        data=np.array(self.analog_data).ravel()
        #output_file = open(filename, 'wb')
        #data.tofile(output_file)
        #output_file.close()
        scipy.io.savemat(filename, {'FCS_data':data, 'averaged_corr':self.averaged_corr})
        print('Exported data')

    def export_corr_gui(self):
        if self.stop==False:
            self.stop_input_thread()
        filename= QFileDialog.getSaveFileName(self, 'Export Correlation', r'C:/Users/Admin/Desktop/','*.txt')
        filename=str(filename)
        if filename=='':
            return False
        else:
            self.export_corr(filename)
    def export_corr(self,filename=None):
        y=self.averaged_corr
        x=np.linspace(0.0, len(y)*sampleInterval/1000000., len(y), endpoint=False) #microseconds to seconds
        data=np.array([x,y]).T
        np.savetxt(filename, data,fmt='%10.6f')
        print('Exported Correlation Data')
        
class MyList(list):
    pass

class AnalogInputThread(QThread):
    def __init__(self,diodeValueWidget):
        super(AnalogInputThread , self).__init__()
        self.diodeValueWidget=diodeValueWidget
        self.data=MyList()
        self.id_a=create_callbackdata_id(self.data)
        self.EveryNCallback = DAQmxEveryNSamplesEventCallbackPtr(self.EveryNCallback_py)
        self.DoneCallback = DAQmxDoneEventCallbackPtr(self.DoneCallback_py)
        self.stop=False
        self.nSamples=1000
        self.previousValue=0

    def run(self):
        global sampleInterval
        capture_rate=int(np.round(1./(sampleInterval/1000000.)))
        taskHandle=TaskHandle(0)
        self.th=taskHandle
        DAQmxCreateTask("",byref(taskHandle))
        DAQmxCreateCICountEdgesChan(taskHandle,"Dev1/ctr0","",DAQmx_Val_Rising,0,DAQmx_Val_CountUp) # ctr0 gets its input from PFI8
        DAQmxCfgSampClkTiming(taskHandle,"/Dev1/PFI0",capture_rate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,self.nSamples) #PFI0 sets the timing. 
        
        DAQmxRegisterEveryNSamplesEvent(taskHandle,DAQmx_Val_Acquired_Into_Buffer,self.nSamples,0,self.EveryNCallback,self.id_a)
        DAQmxRegisterDoneEvent(taskHandle,0,self.DoneCallback,None)
        

        
        DAQmxStartTask(taskHandle)
        
    def EveryNCallback_py(self, taskHandle, everyNsamplesEventType, nSamples, callbackData_ptr):
        #callbackdata = get_callbackdata_from_id(callbackData_ptr)
        read = int32()
        data = zeros(self.nSamples,dtype=np.uint32) #1 channel times 100 samples per channel = 100

        

        DAQmxReadCounterU32(taskHandle,self.nSamples,-1,data,self.nSamples,byref(read),None)
        #callbackdata.extend(data.tolist())
        #print "Acquired total %d samples"%len(data)
        
        data2=np.append([self.previousValue],data[:-1])
        diff=data-data2
        
        #if np.any(data<data2): #when we overflow, we need to figure out what the real number was
        #    idx=np.where(data<data2)[0][0]
        #    diff[idx]=(data[idx]+2**32-1)-data2[idx]
        self.diodeValueWidget.analog_data.append(diff)
        self.previousValue=data[-1]
        if self.stop:
            DAQmxStopTask(taskHandle)
            DAQmxClearTask(taskHandle)
        #self.diodeValueWidget.update()
        #QApplication.processEvents()
        return 0 # The function should return an integer

    def DoneCallback_py(self,taskHandle, status, callbackData):
        #print "Status",status.value
        return 0 # The function should return an integer
    

        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    self = DiodeValueWidget()
    self.show()
    insideSpyder='SPYDER_SHELL_ID' in os.environ
    if not insideSpyder: #if we are running outside of Spyder
        sys.exit(app.exec_()) #This is required to run outside of Spyder
    
    
