#!/usr/bin/env python
import roslib; roslib.load_manifest('Kinefly')
import rospy

import time

import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.animation as animation
import numpy as np

from Kinefly.srv import SrvTipdata

class TipPlotter:
    
    def __init__(self):
        
        rospy.init_node('tipplotter', anonymous=True)
        
        # Attach to services.
        service_name = 'tipdata_head'
        rospy.wait_for_service(service_name)
        self.get_tipdata = rospy.ServiceProxy(service_name, SrvTipdata)
        
        self.timePrev = rospy.Time.now().to_sec()
        
        tipdata = self.get_tipdata(0)
        
        intensities = np.zeros(len(tipdata.abscissa))
        diffs = np.zeros(len(tipdata.abscissa))

        self.intensities_hi = -np.inf
        self.intensities_lo = np.inf
        self.diffs_hi = -np.inf
        self.diffs_lo = np.inf
        detectionH = 0
        detectionV = 0
                
        color = (1,0,0,1)#'red'
        
        # Open a figure window with subplots.  top:intensity, bottom:diff
        self.fig = plt.figure(service_name)
        self.sub1 = plt.subplot(2,1,1)
        self.sub2 = plt.subplot(2,1,2)
        self.sub1.set_xlim(0, len(tipdata.abscissa))
        self.sub2.set_xlim(0, len(tipdata.abscissa))
        self.sub1.set_title('Intensity')
        self.sub2.set_title('Intensity Gradient')


        self.plot1_intensities,    = self.sub1.plot(tipdata.abscissa, intensities, '.', color=tipdata.color)
        self.plot1_detectionH,      = self.sub1.plot([detectionH, detectionH], [0,1], color=tipdata.color,   linewidth=1)
        self.plot1_detectionV,      = self.sub1.plot([0,1], [detectionV, detectionV], color=tipdata.color,   linewidth=1)

        self.plot2_diffs,          = self.sub2.plot(tipdata.abscissa, diffs, '.', color=tipdata.color)
        self.plot2_detectionH,      = self.sub2.plot([detectionH, detectionH], [0,1], color=tipdata.color,   linewidth=1)
        #self.plot2_detectionV,      = self.sub2.plot([0,1], [detectionV, detectionV], color=color,   linewidth=1)

                
        #self.image_animation = animation.FuncAnimation(self.fig, self.update_plots, self.abscissa, init_func=self.init_plot, interval=50, blit=True)
        self.image_animation = animation.FuncAnimation(self.fig, self.update_plots, init_func=self.init_plot, interval=50, blit=True)
        
        
    def update_plots(self, i):
        rv = (self.plot1_detectionH,
              self.plot1_detectionV,
              self.plot2_detectionH,
              #self.plot2_detectionV,
              )

        # Get the tipdata.
        try:
            tipdata = self.get_tipdata(0)
        except rospy.ServiceException:
            pass
        else:
            decay = 1.0#0.97
    
            intensities = np.array([tipdata.intensities])
            self.intensities_hi = np.max(np.hstack(np.append(intensities, self.intensities_hi*decay).flat))
            self.intensities_lo = np.min(np.hstack(np.append(intensities, self.intensities_lo*decay).flat))
    
            diffs = np.array([tipdata.diffs])
            self.diffs_hi = np.max(np.hstack(np.append(diffs, self.diffs_hi*decay).flat))
            self.diffs_lo = np.min(np.hstack(np.append(diffs, self.diffs_lo*decay).flat))
    
            # Set axis limits.
            self.sub1.set_xlim(0, len(tipdata.abscissa))
            self.sub2.set_xlim(0, len(tipdata.abscissa))
            self.sub1.set_ylim(self.intensities_lo, self.intensities_hi)
            self.sub2.set_ylim(self.diffs_lo, self.diffs_hi)
            self.fig.show()
    
            
            # Plot the intensities.    
            if (tipdata.intensities is not None) and (tipdata.diffs is not None):
                self.plot1_intensities.set_data(tipdata.abscissa, tipdata.intensities)
                self.plot2_diffs.set_data(tipdata.abscissa, tipdata.diffs)
            
            
                self.plot1_detectionH.set_data([tipdata.detectionH, tipdata.detectionH], self.sub1.get_ylim())
                self.plot1_detectionV.set_data(self.sub1.get_xlim(), [tipdata.detectionV, tipdata.detectionV])
                self.plot2_detectionH.set_data([tipdata.detectionH, tipdata.detectionH], self.sub2.get_ylim())
                #self.plot2_detectionV.set_data(self.sub2.get_xlim(), [tipdata.detectionV, tipdata.detectionV])
    
            
            if (len(tipdata.abscissa)==len(tipdata.intensities)) and (len(tipdata.abscissa)==len(tipdata.diffs)):        
                rv = (self.plot1_intensities, 
                      self.plot1_detectionH,
                      self.plot1_detectionV,
                      self.plot2_diffs, 
                      self.plot2_detectionH, 
                      #self.plot2_detectionV,
                      )
        
        return rv
        
        
    def init_plot(self): # required to start with clean slate
        self.plot1_intensities.set_data([],[])
        self.plot1_detectionH.set_data([],[])
        self.plot1_detectionV.set_data([],[])
        self.plot2_diffs.set_data([],[])
        self.plot2_detectionH.set_data([],[])
        #self.plot2_detectionV.set_data([],[])
        
        return (self.plot1_intensities, 
                self.plot1_detectionH,
                self.plot1_detectionV,
                self.plot2_diffs, 
                self.plot2_detectionH, 
                #self.plot2_detectionV,
                )


    def main(self):
        plt.show()
        
        

if __name__ == '__main__':
    wingplotter = TipPlotter()
    wingplotter.main()
    
