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
import random
import scipy.io

samplePeriod=100 # microseconds


class DiodeValueWidget(QWidget) :
    """
    This widget is what actually displays the circle.
    """
    def __init__(self, parent = None) :
        QWidget.__init__(self, parent)
        self.setGeometry(100, 100, 600, 600)
        self.setWindowTitle( "FCS Measurement software" )
        
        ## plotting ##
        self.l = QVBoxLayout()
        self.setLayout(self.l)
        self.plt=pg.PlotWidget()
        self.plt.showGrid(x=True, y=True)
        self.l.addWidget(self.plt)
        self.x=np.arange(100,dtype=np.float64)
        self.y=np.zeros(100,dtype=np.float64)
        self.curve = self.plt.plot(self.x, self.y, pen=(255,0,0))
        self.plt.setYRange(-2,10.2)
        
        ## buttons ##
        
        self.control_panel=QGridLayout()
        self.stop=False
        self.start_stop_button=QPushButton('Stop')
        self.start_stop_button.pressed.connect(self.start_stop)
        self.exportButton=QPushButton('Export')
        self.exportButton.pressed.connect(self.export_gui)
        self.control_panel.addWidget(self.start_stop_button,0,0)
        self.control_panel.addWidget(self.exportButton,1,0)
        self.control_panelWidget=QWidget()
        self.control_panelWidget.setLayout(self.control_panel)
        self.l.addWidget(self.control_panelWidget)
        
        ## acquire data from analog input thread##
        self.analog_data=[] #np.zeros(100,dtype=np.float64)
        self.timer=QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)
        self.analogInputThread=AnalogInputThread(self)
        #self.connect(thread , QtCore.SIGNAL('update(QString)') , self.change)
        self.analogInputThread.start()
    def paintEvent( self, event ) :
        #self.analogInputThread.stop=True
        if len(self.analog_data)==0:
            pass
        else:
            y=np.array(self.analog_data[-100:])
            self.y=y.reshape((y.shape[0]*y.shape[1]))
            self.x=np.arange(len(self.y))
            #self.y=self.analog_data[-1]
        self.curve.setData(self.x, self.y)
        pass
        #print(self.analog_data)
        #qp = QPainter()
        #qp.begin( self )
        #self.drawArcs(qp)
        #self.drawSpots(qp)
        #qp.end()
    def start_stop(self):
        if self.stop==False:
            self.stop_input_thread()
        else:
            self.start_input_thread()
            
    def stop_input_thread(self):
        self.analogInputThread.stop=True
        self.timer.stop()
        self.start_stop_button.setText('Start')
        self.stop=True
        print('Input thread stopped')
        
    def start_input_thread(self):
        self.analog_data=[]
        self.timer.start(10)
        self.analogInputThread=AnalogInputThread(self)
        self.analogInputThread.start()
        self.start_stop_button.setText('Stop')
        self.stop=False
        print('Input thread restarted')


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
        data=np.array(self.analog_data)
        data=data.reshape((data.shape[0]*data.shape[1]))
        #output_file = open(filename, 'wb')
        #data.tofile(output_file)
        #output_file.close()
        scipy.io.savemat(filename, {'FCS_data':data})
        print('Exported data')

        
        
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

    def run(self):
        global samplePeriod
        capture_rate=1/(samplePeriod/1000000)
        taskHandle=TaskHandle()
        self.th=taskHandle
        DAQmxCreateTask("",byref(taskHandle))
        DAQmxCreateAIVoltageChan(taskHandle,"Dev1/ai0","",DAQmx_Val_Cfg_Default,-10.0,10.0,DAQmx_Val_Volts,None)
        DAQmxCfgSampClkTiming(taskHandle,"",capture_rate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,100)
        
        DAQmxRegisterEveryNSamplesEvent(taskHandle,DAQmx_Val_Acquired_Into_Buffer,100,0,self.EveryNCallback,self.id_a)
        DAQmxRegisterDoneEvent(taskHandle,0,self.DoneCallback,None)
        
        DAQmxStartTask(taskHandle)
    
        #raw_input('Acquiring samples continuously. Press Enter to interrupt\n')
        #DAQmxStopTask(taskHandle)
        #DAQmxClearTask(taskHandle)
        
    def EveryNCallback_py(self, taskHandle, everyNsamplesEventType, nSamples, callbackData_ptr):
        #callbackdata = get_callbackdata_from_id(callbackData_ptr)
        read = int32()
        data = zeros(100) #1 channel times 100 samples per channel = 100
        DAQmxReadAnalogF64(taskHandle,100,10.0,DAQmx_Val_GroupByChannel,data,100,byref(read),None)
        #callbackdata.extend(data.tolist())
        #print "Acquired total %d samples"%len(data)
        self.diodeValueWidget.analog_data.append(data)
        if self.stop:
            DAQmxStopTask(taskHandle)
            DAQmxClearTask(taskHandle)
        #self.diodeValueWidget.update()
        #QApplication.processEvents()
        return 0 # The function should return an integer

    def DoneCallback_py(self,taskHandle, status, callbackData):
        print "Status",status.value
        return 0 # The function should return an integer
    

        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = DiodeValueWidget()
    w.show()
    insideSpyder='SPYDER_SHELL_ID' in os.environ
    if not insideSpyder: #if we are running outside of Spyder
        sys.exit(app.exec_()) #This is required to run outside of Spyder
