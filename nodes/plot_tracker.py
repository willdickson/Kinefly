#!/usr/bin/env python
import roslib; roslib.load_manifest('Kinefly')
import rospy

import time

import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.animation as animation
import numpy as np

from Kinefly.srv import SrvTrackerdata

class TrackerPlotter:
    
    def __init__(self):
        
        rospy.init_node('trackerplotter', anonymous=True)
        
        # Attach to services.
        service_name = 'trackerdata_abdomen'
        rospy.wait_for_service(service_name)
        self.get_trackerdata = rospy.ServiceProxy(service_name, SrvTrackerdata)
        rospy.sleep(1)
        trackerdata = self.get_trackerdata(0)
        
        self.intensities_hi = -np.inf
        self.intensities_lo = np.inf
        self.diffs_hi = -np.inf
        self.diffs_lo = np.inf
                
        # Open a figure window with subplots.  top:intensity, bottom:diff
        self.fig = plt.figure(service_name)
        self.sub1 = plt.subplot(2,1,1)
        self.sub2 = plt.subplot(2,1,2)
        #rospy.logwarn(trackerdata)
        self.sub1.set_xlim(np.min(trackerdata.abscissa), np.max(trackerdata.abscissa))
        self.sub2.set_xlim(np.min(trackerdata.abscissa), np.max(trackerdata.abscissa))
        self.sub1.set_title('plot1')
        self.sub2.set_title('plot2')
        
        self.sub1.hold(False)
        self.sub1.plot(trackerdata.abscissa, np.zeros(len(trackerdata.abscissa)), '.', color=trackerdata.color)
        self.sub1.hold(True)

        #self.plot1_markers = []
        for marker in trackerdata.markersH:
            self.sub1.plot([marker, marker], [0,1], color=trackerdata.color,   linewidth=1)
        for marker in trackerdata.markersV:
            self.sub1.plot([0,1], [marker, marker], color=trackerdata.color,   linewidth=1)

        self.sub2.hold(False)
        self.sub2.plot(trackerdata.abscissa, np.zeros(len(trackerdata.abscissa)), '.', color=trackerdata.color)
        self.sub2.hold(True)
        for marker in trackerdata.markersH:
            self.sub2.plot([marker, marker], [0,1], color=trackerdata.color,   linewidth=1)
        for marker in trackerdata.markersV:
            self.sub2.plot([0,1], [marker, marker], color='red',   linewidth=1)

        self.fig.show()
        #self.image_animation = animation.FuncAnimation(self.fig, self.update_plots, init_func=self.init_plot, interval=50, blit=True)
        
        
    def update_plots(self):
        # Get the trackerdata.
        try:
            trackerdata = self.get_trackerdata(0)
        except rospy.ServiceException:
            pass
        else:
            decay = 1.0#0.97
            intensities = np.array([trackerdata.intensities])
            self.intensities_hi = np.max(np.hstack(np.append(intensities, self.intensities_hi*decay).flat))
            self.intensities_lo = np.min(np.hstack(np.append(intensities, self.intensities_lo*decay).flat))
    
            diffs = np.array([trackerdata.diffs])
            self.diffs_hi = np.max(np.hstack(np.append(diffs, self.diffs_hi*decay).flat))
            self.diffs_lo = np.min(np.hstack(np.append(diffs, self.diffs_lo*decay).flat))
    
            # Set axis limits.
            self.sub1.set_xlim(np.min(trackerdata.abscissa), np.max(trackerdata.abscissa))
            self.sub1.set_ylim(self.intensities_lo, self.intensities_hi)
            self.sub2.set_xlim(np.min(trackerdata.abscissa), np.max(trackerdata.abscissa))
            self.sub2.set_ylim(self.diffs_lo, self.diffs_hi)
    
            
            # Plot the intensities.    
            if (trackerdata.intensities is not None) and (trackerdata.diffs is not None):
                self.sub1.hold(False)
                self.sub2.hold(False)

                self.sub1.plot(trackerdata.abscissa, trackerdata.intensities)
                self.sub2.plot(trackerdata.abscissa, trackerdata.diffs)

                self.sub1.hold(True)
                self.sub2.hold(True)
            
                self.sub1.set_title(trackerdata.title1)
                self.sub2.set_title(trackerdata.title2)

                for marker in trackerdata.markersH:
                    self.sub2.plot([marker, marker], self.sub2.get_ylim())
                    self.sub1.plot([marker, marker], self.sub1.get_ylim())
                for marker in trackerdata.markersV:
                    self.sub1.plot(self.sub1.get_xlim(), [marker, marker])
                    self.sub2.plot(self.sub2.get_xlim(), [marker, marker])
    
            self.fig.canvas.draw()

        
        
    def init_plot(self): # required to start with clean slate
        self.sub1.plot([],[])
        self.sub2.plot([],[])
                


    def run(self):
        rosrate = rospy.Rate(20)
        while (not rospy.is_shutdown()):
            self.update_plots()
            rosrate.sleep()
        
        

if __name__ == '__main__':
    main = TrackerPlotter()
    main.run()
    
