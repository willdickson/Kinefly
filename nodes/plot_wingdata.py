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
        self.sub1.set_xlim(-np.pi,np.pi)
        self.sub2.set_xlim(-np.pi,np.pi)
        self.sub1.set_title('Intensity')
        self.sub2.set_title('Intensity Gradient')


        wingdata_right = self.get_wingdata_right(0)
        wingdata_left  = self.get_wingdata_left(0)
        
        angles_right = wingdata_right.angles
        angles_left  = wingdata_left.angles

        self.limits_right = [-np.pi, np.pi]
        self.limits_left  = [-np.pi, np.pi]
        edges_right = [-np.pi, np.pi]
        edges_left  = [-np.pi, np.pi]
        
        intensities_right = np.zeros(len(angles_right))
        intensities_left  = np.zeros(len(angles_left))
        diffs_right = np.zeros(len(angles_right))
        diffs_left  = np.zeros(len(angles_left))

        self.intensities_hi = -np.inf
        self.intensities_lo = np.inf
        self.diffs_hi = -np.inf
        self.diffs_lo = np.inf
                
        colorRight = 'red'
        colorLeft = 'green'
        
        
        self.plot1_intensities_right,    = self.sub1.plot(angles_right, intensities_right, '.', color=colorRight)
        self.plot1_limits_lo_right,      = self.sub1.plot([self.limits_right[0], self.limits_right[0]], [0,1], color='black', linewidth=1)
        self.plot1_limits_hi_right,      = self.sub1.plot([self.limits_right[1], self.limits_right[1]], [0,1], color='black', linewidth=1)
        self.plot1_edges_minor_right,    = self.sub1.plot([edges_right[0], edges_right[0]], [0,1], color=colorRight, linewidth=1)
        self.plot1_edges_major_right,    = self.sub1.plot([edges_right[1], edges_right[1]], [0,1], color=colorRight, linewidth=1)

        self.plot1_intensities_left,     = self.sub1.plot(angles_left, intensities_left, '.', color=colorLeft)
        self.plot1_limits_lo_left,       = self.sub1.plot([self.limits_left[0], self.limits_left[0]], [0,1], color='black', linewidth=1)
        self.plot1_limits_hi_left,       = self.sub1.plot([self.limits_left[1], self.limits_left[1]], [0,1], color='black', linewidth=1)
        self.plot1_edges_minor_left,     = self.sub1.plot([edges_left[0], edges_left[0]], [0,1], color=colorLeft, linewidth=1)
        self.plot1_edges_major_left,     = self.sub1.plot([edges_left[1], edges_left[1]], [0,1], color=colorLeft, linewidth=1)
        
        self.plot2_diffs_right,          = self.sub2.plot(angles_right, diffs_right, '.', color=colorRight)
        self.plot2_limits_lo_right,      = self.sub2.plot([self.limits_right[0], self.limits_right[0]], [0,1], color='black', linewidth=1)
        self.plot2_limits_hi_right,      = self.sub2.plot([self.limits_right[1], self.limits_right[1]], [0,1], color='black', linewidth=1)
        self.plot2_edges_minor_right,    = self.sub2.plot([edges_right[0], edges_right[0]], [0,1], color=colorRight, linewidth=1)
        self.plot2_edges_major_right,    = self.sub2.plot([edges_right[1], edges_right[1]], [0,1], color=colorRight, linewidth=1)

        self.plot2_diffs_left,           = self.sub2.plot(angles_left, diffs_left, '.', color=colorLeft)
        self.plot2_limits_lo_left,       = self.sub2.plot([self.limits_left[0], self.limits_left[0]], [0,1], color='black', linewidth=1)
        self.plot2_limits_hi_left,       = self.sub2.plot([self.limits_left[1], self.limits_left[1]], [0,1], color='black', linewidth=1)
        self.plot2_edges_minor_left,     = self.sub2.plot([edges_left[0], edges_left[0]], [0,1], color=colorLeft, linewidth=1)
        self.plot2_edges_major_left,     = self.sub2.plot([edges_left[1], edges_left[1]], [0,1], color=colorLeft, linewidth=1)
        
                
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
            angles_right = wingdata_right.angles
            intensities_right = wingdata_right.intensities
            diffs_right = wingdata_right.diffs
            edges_right = wingdata_right.edges
            
            wingdata_left = self.get_wingdata_left(0)
            angles_left = wingdata_left.angles
            intensities_left = wingdata_left.intensities
            diffs_left = wingdata_left.diffs
            edges_left = wingdata_left.edges
            
            
            intensities = np.array([intensities_right, intensities_left])
            diffs = np.array([diffs_right, diffs_left])

            decay = 0.97
            self.intensities_hi = np.max(np.hstack(np.append(intensities, self.intensities_hi*decay).flat))
            self.intensities_lo = np.min(np.hstack(np.append(intensities, self.intensities_lo*decay).flat))
            self.diffs_hi = np.max(np.hstack(np.append(diffs, self.diffs_hi*decay).flat))
            self.diffs_lo = np.min(np.hstack(np.append(diffs, self.diffs_lo*decay).flat))

            self.sub1.set_ylim(self.intensities_lo, self.intensities_hi)
            self.sub2.set_ylim(self.diffs_lo, self.diffs_hi)
            self.fig.show()
            
            timeNow = rospy.Time.now().to_sec()
            if (timeNow - self.timePrev > 1):
                self.limits_right = [rospy.get_param('kinefly/right/angle_lo', -np.pi), rospy.get_param('kinefly/right/angle_hi', np.pi)]
                self.limits_left  = [rospy.get_param('kinefly/left/angle_lo', -np.pi),  rospy.get_param('kinefly/left/angle_hi', np.pi)]
                self.timePrev = timeNow
                
            if intensities_right is not None:
                self.plot1_intensities_right.set_data(angles_right, intensities_right)
            if intensities_left is not None:
                self.plot1_intensities_left.set_data(angles_left, intensities_left)
            
            if diffs_right is not None:
                self.plot2_diffs_right.set_data(angles_right, diffs_right)
            if diffs_left is not None:
                self.plot2_diffs_left.set_data(angles_left, diffs_left)
            
            self.plot1_limits_lo_right.set_data([self.limits_right[0], self.limits_right[0]], [self.intensities_lo, self.intensities_hi])
            self.plot1_limits_hi_right.set_data([self.limits_right[1], self.limits_right[1]], [self.intensities_lo, self.intensities_hi])
            self.plot1_edges_minor_right.set_data([edges_right[0], edges_right[0]], [self.intensities_lo, self.intensities_hi])
            self.plot1_edges_major_right.set_data([edges_right[1], edges_right[1]], [self.intensities_lo, self.intensities_hi])
            
            self.plot1_limits_lo_left.set_data([self.limits_left[0], self.limits_left[0]], [self.intensities_lo, self.intensities_hi])
            self.plot1_limits_hi_left.set_data([self.limits_left[1], self.limits_left[1]], [self.intensities_lo, self.intensities_hi])
            self.plot1_edges_minor_left.set_data([edges_left[0], edges_left[0]], [self.intensities_lo, self.intensities_hi])
            self.plot1_edges_major_left.set_data([edges_left[1], edges_left[1]], [self.intensities_lo, self.intensities_hi])
    
            self.plot2_limits_lo_right.set_data([self.limits_right[0], self.limits_right[0]], [self.diffs_lo, self.diffs_hi])
            self.plot2_limits_hi_right.set_data([self.limits_right[1], self.limits_right[1]], [self.diffs_lo, self.diffs_hi])
            self.plot2_edges_minor_right.set_data([edges_right[0], edges_right[0]], [self.diffs_lo, self.diffs_hi])
            self.plot2_edges_major_right.set_data([edges_right[1], edges_right[1]], [self.diffs_lo, self.diffs_hi])
            
            self.plot2_limits_lo_left.set_data([self.limits_left[0], self.limits_left[0]], [self.diffs_lo, self.diffs_hi])
            self.plot2_limits_hi_left.set_data([self.limits_left[1], self.limits_left[1]], [self.diffs_lo, self.diffs_hi])
            self.plot2_edges_minor_left.set_data([edges_left[0], edges_left[0]], [self.diffs_lo, self.diffs_hi])
            self.plot2_edges_major_left.set_data([edges_left[1], edges_left[1]], [self.diffs_lo, self.diffs_hi])
    
            if (len(angles_right)==len(intensities_right)) and (len(angles_left)==len(intensities_left)) and (len(angles_right)==len(diffs_right)) and (len(angles_left)==len(diffs_left)):        
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
    
