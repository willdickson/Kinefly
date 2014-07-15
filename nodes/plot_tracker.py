#!/usr/bin/env python


##############################################################################
# plot_tracker
#    For assistance with correctly setting bodypart tracking parameters, run this 
#    node alongside Kinefly to show various internal signals related to the 
#    tracking.  This may help you in setting parameters such as 'threshold'.
#
#    Each Kinefly tracker makes its data available via a service name such
#    as 'trackerdata_head', 'trackerdata_left', 'trackerdata_right', or
#    'trackerdata_abdomen'.  
#
#    Prior to running this, set the 'trackerdata_name' parameter to the 
#    signal of interest.  For example:
#
#    rosparam set trackername trackerdata_abdomen
#    rosrun Kinefly plot_tracker.py
#
#
##############################################################################


import roslib; roslib.load_manifest('Kinefly')
import rospy

import time

import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.animation as animation
import numpy as np

from Kinefly.srv import SrvTrackerdata

class PlotTracker:
    
    def __init__(self):
        
        rospy.init_node('plottracker')
        
        # Attach to services.
        service_name = rospy.get_param('trackername','YOU_NEED_TO_SET_TRACKERNAME_VIA_ROSPARAM')
        rospy.logwarn('Waiting for Kinefly service: %s' % service_name)
        rospy.wait_for_service(service_name)
        self.get_trackerdata = rospy.ServiceProxy(service_name, SrvTrackerdata)
        rospy.logwarn('Connected to Kinefly.')
        rospy.sleep(1)
        trackerdata = self.get_trackerdata(0)
        
        self.intensities_hi = -np.inf
        self.intensities_lo = np.inf
        self.diffs_hi = -np.inf
        self.diffs_lo = np.inf
                
        # Open a figure window with subplots.
        self.fig = plt.figure(service_name)

        self.sub1 = self.fig.add_subplot(2,1,1)
        self.sub2 = self.fig.add_subplot(2,1,2)

        self.sub1.set_xlim(np.min(trackerdata.abscissa), np.max(trackerdata.abscissa))
        self.sub2.set_xlim(np.min(trackerdata.abscissa), np.max(trackerdata.abscissa))
        self.sub1.set_title('plot1')
        self.sub2.set_title('plot2')
        
        self.sub1.hold(False)
        self.sub1.plot(trackerdata.abscissa, np.zeros(len(trackerdata.abscissa)), '.', color=trackerdata.color)
        self.sub1.hold(True)

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

        self.cid = self.fig.canvas.mpl_connect('close_event', self.onClose)
        self.fig.show()
        #self.image_animation = animation.FuncAnimation(self.fig, self.update_plots, init_func=self.init_plot, interval=50, blit=True)
        

    def onClose(self, guiEvent=None):
        self.fig.canvas.mpl_disconnect(self.cid)
        self.fig.clear()
        rospy.signal_shutdown('User requested exit.')

        
    def update_plots(self):
        # Get the trackerdata.
        try:
            trackerdata = self.get_trackerdata(0)
        except rospy.ServiceException:
            self.onClose()
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
    
        plt.draw()
            
            
        
        
    def init_plot(self): # required to start with clean slate
        self.sub1.plot([],[])
        self.sub2.plot([],[])
                


    def run(self):
        rosrate = rospy.Rate(20)
        try:
            while (not rospy.is_shutdown()):
                self.fig.canvas.flush_events()
                self.update_plots()
                rosrate.sleep()
        except rospy.exceptions.ROSInterruptException:
            pass
        
        

if __name__ == '__main__':
    main = PlotTracker()
    main.run()
    
