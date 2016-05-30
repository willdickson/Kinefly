#!/usr/bin/env python
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
        service_name = 'trackerdata_left'
        rospy.wait_for_service(service_name)
        self.get_trackerdata = rospy.ServiceProxy(service_name, SrvTrackerdata)
        
        self.timePrev = rospy.Time.now().to_sec()
        
        trackerdata = self.get_trackerdata(0)
        
        intensities = np.zeros(len(trackerdata.abscissa))
        diffs = np.zeros(len(trackerdata.abscissa))

        self.intensities_hi = -np.inf
        self.intensities_lo = np.inf
        self.diffs_hi = -np.inf
        self.diffs_lo = np.inf
        markersH = []
        markersV = []
                
        color = (1,0,0,1)#'red'
        
        # Open a figure window with subplots.  top:intensity, bottom:diff
        self.fig = plt.figure(service_name)
        self.sub1 = plt.subplot(2,1,1)
        self.sub2 = plt.subplot(2,1,2)
        self.sub1.set_xlim(0, len(trackerdata.abscissa))
        self.sub2.set_xlim(0, len(trackerdata.abscissa))
        self.sub1.set_title('Intensity')
        self.sub2.set_title('Intensity Gradient')


        self.plot1_intensities,    = self.sub1.plot(trackerdata.abscissa, intensities, '.', color=trackerdata.color)
        
        self.plot1_markers = []
        for threshold in markersH:
            plot,      = self.sub1.plot([threshold, threshold], [0,1], color=trackerdata.color,   linewidth=1)
            self.plot1_markers.append(plot)
        for threshold in markersV:
            plot,      = self.sub1.plot([0,1], [threshold, threshold], color=trackerdata.color,   linewidth=1)
            self.plot1_markers.append(plot)

        self.plot2_diffs,          = self.sub2.plot(trackerdata.abscissa, diffs, '.', color=trackerdata.color)
        self.plot2_markers = []
        for threshold in markersH:
            plot,      = self.sub2.plot([threshold, threshold], [0,1], color=trackerdata.color,   linewidth=1)
            self.plot2_markers.append(plot)
        #for threshold in markersV:
        #    plot,      = self.sub2.plot([0,1], [threshold, threshold], color=color,   linewidth=1)
        #    self.plot2_markers.append(plot)

                
        #self.image_animation = animation.FuncAnimation(self.fig, self.update_plots, self.abscissa, init_func=self.init_plot, interval=50, blit=True)
        self.image_animation = animation.FuncAnimation(self.fig, self.update_plots, init_func=self.init_plot, interval=50, blit=True)
        
        
    def update_plots(self, i):
        rv = [self.plot1_markers]

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
            self.sub1.set_xlim(0, len(trackerdata.abscissa))
            self.sub2.set_xlim(0, len(trackerdata.abscissa))
            self.sub1.set_ylim(self.intensities_lo, self.intensities_hi)
            self.sub2.set_ylim(self.diffs_lo, self.diffs_hi)
            self.fig.show()
    
            
            # Plot the intensities.    
            if (trackerdata.intensities is not None) and (trackerdata.diffs is not None):
                self.plot1_intensities.set_data(trackerdata.abscissa, trackerdata.intensities)
                self.plot2_diffs.set_data(trackerdata.abscissa, trackerdata.diffs)
            
                for plot in self.plot1_markers:
                    for threshold in trackerdata.markersH:
                        plot.set_data([threshold, threshold], self.sub1.get_ylim())
                    for threshold in trackerdata.markersV:
                        plot.set_data(self.sub1.get_xlim(), [threshold, threshold])
                for plot in self.plot2_markers:
                    for threshold in trackerdata.markersH:
                        plot.set_data([threshold, threshold], self.sub2.get_ylim())
                    #for threshold in trackerdata.markersV:
                    #    plot.set_data(self.sub2.get_xlim(), [threshold, threshold])
    
            
            if (len(trackerdata.abscissa)==len(trackerdata.intensities)) and (len(trackerdata.abscissa)==len(trackerdata.diffs)):        
                rv = []
                rv.append(self.plot1_intensities) 
                rv.extend(self.plot1_markers)
                rv.append(self.plot2_diffs)
                rv.extend(self.plot2_markers)
        
        return rv
        
        
    def init_plot(self): # required to start with clean slate
        self.plot1_intensities.set_data([],[])
        self.plot2_diffs.set_data([],[])
        for plot in self.plot1_markers:
            plot.set_data([],[])

        for plot in self.plot2_markers:
            plot.set_data([],[])
        
        rv = []
        rv.append(self.plot1_intensities) 
        rv.extend(self.plot1_markers)
        rv.append(self.plot2_diffs)
        rv.extend(self.plot2_markers)
                
        return rv


    def run(self):
        plt.show()
        
        

if __name__ == '__main__':
    main = TrackerPlotter()
    main.run()
    
