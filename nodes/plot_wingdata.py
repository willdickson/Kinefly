#!/usr/bin/env python
import roslib; roslib.load_manifest('Kinefly')
import rospy

import time

import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.animation as animation
import numpy as np

from Kinefly.srv import SrvWingdata

class WingPlotter:
    
    def __init__(self):
        
        rospy.init_node('wingplotter', anonymous=True)
        
        # Attach to services.
        service_name = "wingdata_right"
        rospy.wait_for_service(service_name)
        self.get_wingdata_right = rospy.ServiceProxy(service_name, SrvWingdata)
        
        service_name = "wingdata_left"
        rospy.wait_for_service(service_name)
        self.get_wingdata_left = rospy.ServiceProxy(service_name, SrvWingdata)

        
        self.timePrev = rospy.Time.now().to_sec()
        
        # Open a figure window with subplots.  top:intensity, bottom:diff
        self.fig = plt.figure()
        self.sub1 = plt.subplot(2,1,1)
        self.sub2 = plt.subplot(2,1,2)
        self.sub1.set_xlim(-3*np.pi/2, 3*np.pi/2)
        self.sub2.set_xlim(-3*np.pi/2, 3*np.pi/2)
        self.sub1.set_title('Intensity')
        self.sub2.set_title('Intensity Gradient')


        wingdata_right = self.get_wingdata_right(0)
        wingdata_left  = self.get_wingdata_left(0)
        
        self.limits_right = [-np.pi, np.pi]
        self.limits_left  = [-np.pi, np.pi]
        edges_right = [-np.pi, np.pi]
        edges_left  = [-np.pi, np.pi]
        
        intensities_right = np.zeros(len(wingdata_right.angles))
        intensities_left  = np.zeros(len(wingdata_left.angles))
        diffs_right = np.zeros(len(wingdata_right.angles))
        diffs_left  = np.zeros(len(wingdata_left.angles))

        self.intensities_hi = -np.inf
        self.intensities_lo = np.inf
        self.diffs_hi = -np.inf
        self.diffs_lo = np.inf
                
        colorR = (1,0,0,1)#'red'
        colorL = (0,1,0,1)#'green'
        colorRdim = (0.5,0,0,1)#'red'
        colorLdim = (0,0.5,0,1)#'green'
        
        
        self.plot1_intensities_right,    = self.sub1.plot(wingdata_right.angles, intensities_right, '.', color=colorR)
        self.plot1_limits_lo_right,      = self.sub1.plot([self.limits_right[0], self.limits_right[0]], [0,1], color='black', linewidth=1)
        self.plot1_limits_hi_right,      = self.sub1.plot([self.limits_right[1], self.limits_right[1]], [0,1], color='black', linewidth=1)
        self.plot1_edges_major_right,    = self.sub1.plot([edges_right[0], edges_right[0]], [0,1], color=colorR, linewidth=1)
        self.plot1_edges_minor_right,    = self.sub1.plot([edges_right[1], edges_right[1]], [0,1], color=colorRdim, linewidth=1)

        self.plot1_intensities_left,     = self.sub1.plot(wingdata_left.angles, intensities_left, '.', color=colorL)
        self.plot1_limits_lo_left,       = self.sub1.plot([self.limits_left[0], self.limits_left[0]], [0,1], color='black', linewidth=1)
        self.plot1_limits_hi_left,       = self.sub1.plot([self.limits_left[1], self.limits_left[1]], [0,1], color='black', linewidth=1)
        self.plot1_edges_major_left,     = self.sub1.plot([edges_left[0], edges_left[0]], [0,1], color=colorL, linewidth=1)
        self.plot1_edges_minor_left,     = self.sub1.plot([edges_left[1], edges_left[1]], [0,1], color=colorLdim, linewidth=1)
        
        self.plot2_diffs_right,          = self.sub2.plot(wingdata_right.angles, diffs_right, '.', color=colorR)
        self.plot2_limits_lo_right,      = self.sub2.plot([self.limits_right[0], self.limits_right[0]], [0,1], color='black', linewidth=1)
        self.plot2_limits_hi_right,      = self.sub2.plot([self.limits_right[1], self.limits_right[1]], [0,1], color='black', linewidth=1)
        self.plot2_edges_major_right,    = self.sub2.plot([edges_right[0], edges_right[0]], [0,1], color=colorR, linewidth=1)
        self.plot2_edges_minor_right,    = self.sub2.plot([edges_right[1], edges_right[1]], [0,1], color=colorRdim, linewidth=1)

        self.plot2_diffs_left,           = self.sub2.plot(wingdata_left.angles, diffs_left, '.', color=colorL)
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

            # Get the wingdata.
            wingdata_right = self.get_wingdata_right(0)
            wingdata_left = self.get_wingdata_left(0)
            
            
            decay = 1.0#0.97

            intensities = np.array([wingdata_right.intensities, wingdata_left.intensities])
            self.intensities_hi = np.max(np.hstack(np.append(intensities, self.intensities_hi*decay).flat))
            self.intensities_lo = np.min(np.hstack(np.append(intensities, self.intensities_lo*decay).flat))

            diffs = np.array([wingdata_right.diffs, wingdata_left.diffs])
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
            if (wingdata_right.intensities is not None):
                self.plot1_intensities_right.set_data(wingdata_right.angles, wingdata_right.intensities)
            if (wingdata_left.intensities is not None):
                self.plot1_intensities_left.set_data(wingdata_left.angles, wingdata_left.intensities)
            
            
            # Plot the intensity diffs.
            if (wingdata_right.diffs is not None):
                self.plot2_diffs_right.set_data(wingdata_right.angles, wingdata_right.diffs)
            if (wingdata_left.diffs is not None):
                self.plot2_diffs_left.set_data(wingdata_left.angles, wingdata_left.diffs)
            
            
            # Plot the right minor/major edge bars on plot1.
            self.plot1_limits_lo_right.set_data([self.limits_right[0], self.limits_right[0]], [self.intensities_lo, self.intensities_hi])
            self.plot1_limits_hi_right.set_data([self.limits_right[1], self.limits_right[1]], [self.intensities_lo, self.intensities_hi])
            if (len(wingdata_right.anglesMinor)>0):
                self.plot1_edges_minor_right.set_data([wingdata_right.anglesMinor[0], wingdata_right.anglesMinor[0]], [self.intensities_lo, self.intensities_hi])
            else:
                self.plot1_edges_minor_right.set_data([], [])
            if (len(wingdata_right.anglesMajor)>0):
                self.plot1_edges_major_right.set_data([wingdata_right.anglesMajor[0], wingdata_right.anglesMajor[0]], [self.intensities_lo, self.intensities_hi])
            else:
                self.plot1_edges_major_right.set_data([], [])
            
            
            # Plot the left minor/major edge bars on plot1.
            self.plot1_limits_lo_left.set_data([self.limits_left[0], self.limits_left[0]], [self.intensities_lo, self.intensities_hi])
            self.plot1_limits_hi_left.set_data([self.limits_left[1], self.limits_left[1]], [self.intensities_lo, self.intensities_hi])
            if (len(wingdata_left.anglesMinor)>0):
                self.plot1_edges_minor_left.set_data([wingdata_left.anglesMinor[0], wingdata_left.anglesMinor[0]], [self.intensities_lo, self.intensities_hi])
            else:
                self.plot1_edges_minor_left.set_data([], [])
            if (len(wingdata_left.anglesMajor)>0):
                self.plot1_edges_major_left.set_data([wingdata_left.anglesMajor[0], wingdata_left.anglesMajor[0]], [self.intensities_lo, self.intensities_hi])
            else:
                self.plot1_edges_major_left.set_data([], [])
    
    
            # Plot the right minor/major edge bars on plot2.
            self.plot2_limits_lo_right.set_data([self.limits_right[0], self.limits_right[0]], [self.diffs_lo, self.diffs_hi])
            self.plot2_limits_hi_right.set_data([self.limits_right[1], self.limits_right[1]], [self.diffs_lo, self.diffs_hi])
            if (len(wingdata_right.anglesMinor)>0):
                self.plot2_edges_minor_right.set_data([wingdata_right.anglesMinor[0], wingdata_right.anglesMinor[0]], [self.diffs_lo, self.diffs_hi])
            else:
                self.plot2_edges_minor_right.set_data([], [])
            if (len(wingdata_right.anglesMajor)>0):
                self.plot2_edges_major_right.set_data([wingdata_right.anglesMajor[0], wingdata_right.anglesMajor[0]], [self.diffs_lo, self.diffs_hi])
            else:
                self.plot2_edges_major_right.set_data([], [])
            
            
            # Plot the left minor/major edge bars on plot2.
            self.plot2_limits_lo_left.set_data([self.limits_left[0], self.limits_left[0]], [self.diffs_lo, self.diffs_hi])
            self.plot2_limits_hi_left.set_data([self.limits_left[1], self.limits_left[1]], [self.diffs_lo, self.diffs_hi])
            if (len(wingdata_left.anglesMinor)>0):
                self.plot2_edges_minor_left.set_data([wingdata_left.anglesMinor[0], wingdata_left.anglesMinor[0]], [self.diffs_lo, self.diffs_hi])
            else:
                self.plot2_edges_minor_left.set_data([], [])
            if (len(wingdata_left.anglesMajor)>0):
                self.plot2_edges_major_left.set_data([wingdata_left.anglesMajor[0], wingdata_left.anglesMajor[0]], [self.diffs_lo, self.diffs_hi])
            else:
                self.plot2_edges_major_left.set_data([], [])
    
    
            if (len(wingdata_right.angles)==len(wingdata_right.intensities)) and (len(wingdata_left.angles)==len(wingdata_left.intensities)) and (len(wingdata_right.angles)==len(wingdata_right.diffs)) and (len(wingdata_left.angles)==len(wingdata_left.diffs)):        
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
            rospy.logwarn('Exception in plot_wingdata.update_plots() %s' % e)

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
    
