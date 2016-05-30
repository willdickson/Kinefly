#!/usr/bin/env python
import rospy
import time
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.animation as animation
import numpy as np
from Kinefly.srv import SrvTrackerdata

class WingPlotter:
    
    def __init__(self):
        
        rospy.init_node('wingplotter', anonymous=True)
        
        # Attach to services.
        service_name = "trackerdata_right"
        rospy.wait_for_service(service_name)
        self.get_trackerdata_right = rospy.ServiceProxy(service_name, SrvTrackerdata)
        
        service_name = "trackerdata_left"
        rospy.wait_for_service(service_name)
        self.get_trackerdata_left = rospy.ServiceProxy(service_name, SrvTrackerdata)

        
        self.timePrev = rospy.Time.now().to_sec()
        
        # Open a figure window with subplots.  top:intensity, bottom:diff
        self.fig = plt.figure()
        self.sub1 = plt.subplot(2,1,1)
        self.sub2 = plt.subplot(2,1,2)
        self.sub1.set_xlim(-3*np.pi/2, 3*np.pi/2)
        self.sub2.set_xlim(-3*np.pi/2, 3*np.pi/2)
        self.sub1.set_title('Intensity')
        self.sub2.set_title('Intensity Gradient')


        trackerdata_right = self.get_trackerdata_right(0)
        trackerdata_left  = self.get_trackerdata_left(0)
        
        self.limits_right = [-np.pi, np.pi]
        self.limits_left  = [-np.pi, np.pi]
        edges_right = [-np.pi, np.pi]
        edges_left  = [-np.pi, np.pi]
        
        intensities_right = np.zeros(len(trackerdata_right.abscissa))
        intensities_left  = np.zeros(len(trackerdata_left.abscissa))
        diffs_right = np.zeros(len(trackerdata_right.abscissa))
        diffs_left  = np.zeros(len(trackerdata_left.abscissa))

        self.intensities_hi = -np.inf
        self.intensities_lo = np.inf
        self.diffs_hi = -np.inf
        self.diffs_lo = np.inf
                
        colorR = (1,0,0,1)#'red'
        colorL = (0,1,0,1)#'green'
        colorRdim = (0.5,0,0,1)#'red'
        colorLdim = (0,0.5,0,1)#'green'
        
        
        self.plot1_intensities_right,    = self.sub1.plot(trackerdata_right.abscissa, intensities_right, '.', color=colorR)
        self.plot1_limits_lo_right,      = self.sub1.plot([self.limits_right[0], self.limits_right[0]], [0,1], color='black', linewidth=1)
        self.plot1_limits_hi_right,      = self.sub1.plot([self.limits_right[1], self.limits_right[1]], [0,1], color='black', linewidth=1)
        self.plot1_edges_major_right,    = self.sub1.plot([edges_right[0], edges_right[0]], [0,1], color=colorR, linewidth=1)
        self.plot1_edges_minor_right,    = self.sub1.plot([edges_right[1], edges_right[1]], [0,1], color=colorRdim, linewidth=1)

        self.plot1_intensities_left,     = self.sub1.plot(trackerdata_left.abscissa, intensities_left, '.', color=colorL)
        self.plot1_limits_lo_left,       = self.sub1.plot([self.limits_left[0], self.limits_left[0]], [0,1], color='black', linewidth=1)
        self.plot1_limits_hi_left,       = self.sub1.plot([self.limits_left[1], self.limits_left[1]], [0,1], color='black', linewidth=1)
        self.plot1_edges_major_left,     = self.sub1.plot([edges_left[0], edges_left[0]], [0,1], color=colorL, linewidth=1)
        self.plot1_edges_minor_left,     = self.sub1.plot([edges_left[1], edges_left[1]], [0,1], color=colorLdim, linewidth=1)
        
        self.plot2_diffs_right,          = self.sub2.plot(trackerdata_right.abscissa, diffs_right, '.', color=colorR)
        self.plot2_limits_lo_right,      = self.sub2.plot([self.limits_right[0], self.limits_right[0]], [0,1], color='black', linewidth=1)
        self.plot2_limits_hi_right,      = self.sub2.plot([self.limits_right[1], self.limits_right[1]], [0,1], color='black', linewidth=1)
        self.plot2_edges_major_right,    = self.sub2.plot([edges_right[0], edges_right[0]], [0,1], color=colorR, linewidth=1)
        self.plot2_edges_minor_right,    = self.sub2.plot([edges_right[1], edges_right[1]], [0,1], color=colorRdim, linewidth=1)

        self.plot2_diffs_left,           = self.sub2.plot(trackerdata_left.abscissa, diffs_left, '.', color=colorL)
        self.plot2_limits_lo_left,       = self.sub2.plot([self.limits_left[0], self.limits_left[0]], [0,1], color='black', linewidth=1)
        self.plot2_limits_hi_left,       = self.sub2.plot([self.limits_left[1], self.limits_left[1]], [0,1], color='black', linewidth=1)
        self.plot2_edges_major_left,     = self.sub2.plot([edges_left[0], edges_left[0]], [0,1], color=colorL, linewidth=1)
        self.plot2_edges_minor_left,     = self.sub2.plot([edges_left[1], edges_left[1]], [0,1], color=colorLdim, linewidth=1)
        
                
        #self.image_animation = animation.FuncAnimation(self.fig, self.update_plots, self.angles, init_func=self.init_plot, interval=50, blit=True)
        self.image_animation = animation.FuncAnimation(self.fig, self.update_plots, init_func=self.init_plot, interval=50, blit=True)
        
        
    def update_plots(self, i):
        try:
            rv = (self.plot1_limits_lo_right, 
                  self.plot1_limits_hi_right, 
                  self.plot1_limits_lo_left, 
                  self.plot1_limits_hi_left, 
                  self.plot1_edges_minor_right, 
                  self.plot1_edges_major_right, 
                  self.plot1_edges_minor_left, 
                  self.plot1_edges_major_left,
                  self.plot2_limits_lo_right, 
                  self.plot2_limits_hi_right, 
                  self.plot2_limits_lo_left, 
                  self.plot2_limits_hi_left, 
                  self.plot2_edges_minor_right, 
                  self.plot2_edges_major_right, 
                  self.plot2_edges_minor_left, 
                  self.plot2_edges_major_left)

            # Get the trackerdata.
            trackerdata_right = self.get_trackerdata_right(0)
            trackerdata_left = self.get_trackerdata_left(0)
            
            
            decay = 1.0#0.97

            intensities = np.array([trackerdata_right.intensities, trackerdata_left.intensities])
            self.intensities_hi = np.max(np.hstack(np.append(intensities, self.intensities_hi*decay).flat))
            self.intensities_lo = np.min(np.hstack(np.append(intensities, self.intensities_lo*decay).flat))

            diffs = np.array([trackerdata_right.diffs, trackerdata_left.diffs])
            self.diffs_hi = np.max(np.hstack(np.append(diffs, self.diffs_hi*decay).flat))
            self.diffs_lo = np.min(np.hstack(np.append(diffs, self.diffs_lo*decay).flat))

            # Set the figure y-limits.
            self.sub1.set_ylim(self.intensities_lo, self.intensities_hi)
            self.sub2.set_ylim(self.diffs_lo, self.diffs_hi)
            self.fig.show()

            # Read the parameters once per sec.            
            timeNow = rospy.Time.now().to_sec()
            if (timeNow - self.timePrev > 1):
                self.limits_right = [rospy.get_param('kinefly/right/angle_lo', -np.pi), rospy.get_param('kinefly/right/angle_hi', np.pi)]
                self.limits_left  = [rospy.get_param('kinefly/left/angle_lo', -np.pi),  rospy.get_param('kinefly/left/angle_hi', np.pi)]
                self.timePrev = timeNow
            
            
            # Plot the intensities.    
            if (trackerdata_right.intensities is not None):
                self.plot1_intensities_right.set_data(trackerdata_right.abscissa, trackerdata_right.intensities)
            if (trackerdata_left.intensities is not None):
                self.plot1_intensities_left.set_data(trackerdata_left.abscissa, trackerdata_left.intensities)
            
            
            # Plot the intensity diffs.
            if (trackerdata_right.diffs is not None):
                self.plot2_diffs_right.set_data(trackerdata_right.abscissa, trackerdata_right.diffs)
            if (trackerdata_left.diffs is not None):
                self.plot2_diffs_left.set_data(trackerdata_left.abscissa, trackerdata_left.diffs)
            
            
            # Plot the right minor/major edge bars on plot1.
            self.plot1_limits_lo_right.set_data([self.limits_right[0], self.limits_right[0]], [self.intensities_lo, self.intensities_hi])
            self.plot1_limits_hi_right.set_data([self.limits_right[1], self.limits_right[1]], [self.intensities_lo, self.intensities_hi])
            if (len(trackerdata_right.anglesMinor)>0):
                self.plot1_edges_minor_right.set_data([trackerdata_right.anglesMinor[0], trackerdata_right.anglesMinor[0]], [self.intensities_lo, self.intensities_hi])
            else:
                self.plot1_edges_minor_right.set_data([], [])
            if (len(trackerdata_right.anglesMajor)>0):
                self.plot1_edges_major_right.set_data([trackerdata_right.anglesMajor[0], trackerdata_right.anglesMajor[0]], [self.intensities_lo, self.intensities_hi])
            else:
                self.plot1_edges_major_right.set_data([], [])
            
            
            # Plot the left minor/major edge bars on plot1.
            self.plot1_limits_lo_left.set_data([self.limits_left[0], self.limits_left[0]], [self.intensities_lo, self.intensities_hi])
            self.plot1_limits_hi_left.set_data([self.limits_left[1], self.limits_left[1]], [self.intensities_lo, self.intensities_hi])
            if (len(trackerdata_left.anglesMinor)>0):
                self.plot1_edges_minor_left.set_data([trackerdata_left.anglesMinor[0], trackerdata_left.anglesMinor[0]], [self.intensities_lo, self.intensities_hi])
            else:
                self.plot1_edges_minor_left.set_data([], [])
            if (len(trackerdata_left.anglesMajor)>0):
                self.plot1_edges_major_left.set_data([trackerdata_left.anglesMajor[0], trackerdata_left.anglesMajor[0]], [self.intensities_lo, self.intensities_hi])
            else:
                self.plot1_edges_major_left.set_data([], [])
    
    
            # Plot the right minor/major edge bars on plot2.
            self.plot2_limits_lo_right.set_data([self.limits_right[0], self.limits_right[0]], [self.diffs_lo, self.diffs_hi])
            self.plot2_limits_hi_right.set_data([self.limits_right[1], self.limits_right[1]], [self.diffs_lo, self.diffs_hi])
            if (len(trackerdata_right.anglesMinor)>0):
                self.plot2_edges_minor_right.set_data([trackerdata_right.anglesMinor[0], trackerdata_right.anglesMinor[0]], [self.diffs_lo, self.diffs_hi])
            else:
                self.plot2_edges_minor_right.set_data([], [])
            if (len(trackerdata_right.anglesMajor)>0):
                self.plot2_edges_major_right.set_data([trackerdata_right.anglesMajor[0], trackerdata_right.anglesMajor[0]], [self.diffs_lo, self.diffs_hi])
            else:
                self.plot2_edges_major_right.set_data([], [])
            
            
            # Plot the left minor/major edge bars on plot2.
            self.plot2_limits_lo_left.set_data([self.limits_left[0], self.limits_left[0]], [self.diffs_lo, self.diffs_hi])
            self.plot2_limits_hi_left.set_data([self.limits_left[1], self.limits_left[1]], [self.diffs_lo, self.diffs_hi])
            if (len(trackerdata_left.anglesMinor)>0):
                self.plot2_edges_minor_left.set_data([trackerdata_left.anglesMinor[0], trackerdata_left.anglesMinor[0]], [self.diffs_lo, self.diffs_hi])
            else:
                self.plot2_edges_minor_left.set_data([], [])
            if (len(trackerdata_left.anglesMajor)>0):
                self.plot2_edges_major_left.set_data([trackerdata_left.anglesMajor[0], trackerdata_left.anglesMajor[0]], [self.diffs_lo, self.diffs_hi])
            else:
                self.plot2_edges_major_left.set_data([], [])
    
    
            if (len(trackerdata_right.abscissa)==len(trackerdata_right.intensities)) and (len(trackerdata_left.abscissa)==len(trackerdata_left.intensities)) and (len(trackerdata_right.abscissa)==len(trackerdata_right.diffs)) and (len(trackerdata_left.abscissa)==len(trackerdata_left.diffs)):        
                rv = (self.plot1_intensities_right, 
                      self.plot1_intensities_left, 
                      self.plot1_limits_lo_right, 
                      self.plot1_limits_hi_right, 
                      self.plot1_limits_lo_left, 
                      self.plot1_limits_hi_left, 
                      self.plot1_edges_minor_right, 
                      self.plot1_edges_major_right, 
                      self.plot1_edges_minor_left, 
                      self.plot1_edges_major_left, 
                      self.plot2_diffs_right, 
                      self.plot2_diffs_left, 
                      self.plot2_limits_lo_right, 
                      self.plot2_limits_hi_right, 
                      self.plot2_limits_lo_left, 
                      self.plot2_limits_hi_left, 
                      self.plot2_edges_minor_right, 
                      self.plot2_edges_major_right, 
                      self.plot2_edges_minor_left, 
                      self.plot2_edges_major_left)
        
        except rospy.ServiceException:
            pass
        
        except Exception,e:
            rospy.logwarn('Exception in plot_trackerdata.update_plots() %s' % e)

        return rv
        
        
    def init_plot(self): # required to start with clean slate
        self.plot1_intensities_right.set_data([],[])
        self.plot1_intensities_left.set_data([],[])

        self.plot2_diffs_right.set_data([],[])
        self.plot2_diffs_left.set_data([],[])

        self.plot1_limits_hi_right.set_data([],[])
        self.plot1_limits_lo_right.set_data([],[])
        self.plot1_edges_minor_right.set_data([],[])
        self.plot1_edges_major_right.set_data([],[])

        self.plot1_limits_hi_left.set_data([],[])
        self.plot1_limits_lo_left.set_data([],[])
        self.plot1_edges_minor_left.set_data([],[])
        self.plot1_edges_major_left.set_data([],[])
        
        self.plot2_limits_hi_right.set_data([],[])
        self.plot2_limits_lo_right.set_data([],[])
        self.plot2_edges_minor_right.set_data([],[])
        self.plot2_edges_major_right.set_data([],[])

        self.plot2_limits_hi_left.set_data([],[])
        self.plot2_limits_lo_left.set_data([],[])
        self.plot2_edges_minor_left.set_data([],[])
        self.plot2_edges_major_left.set_data([],[])
        
        
        return (self.plot1_intensities_right, 
                self.plot1_intensities_left, 
                self.plot2_diffs_right, 
                self.plot2_diffs_left, 
                self.plot1_limits_lo_right, 
                self.plot1_limits_hi_right, 
                self.plot1_limits_lo_left, 
                self.plot1_limits_hi_left, 
                self.plot1_edges_minor_right, 
                self.plot1_edges_major_right, 
                self.plot1_edges_minor_left, 
                self.plot1_edges_major_left, 
                self.plot2_limits_lo_right, 
                self.plot2_limits_hi_right, 
                self.plot2_limits_lo_left, 
                self.plot2_limits_hi_left, 
                self.plot2_edges_minor_right, 
                self.plot2_edges_major_right, 
                self.plot2_edges_minor_left, 
                self.plot2_edges_major_left)


    def main(self):
        plt.show()
        
        

if __name__ == '__main__':
    wingplotter = WingPlotter()
    wingplotter.main()
    
