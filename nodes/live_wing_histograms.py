#!/usr/bin/env python
import roslib; roslib.load_manifest('StrokelitudeROS')
import rospy

import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from StrokelitudeROS.srv import SrvFloat32List

class Live_Plotter:
    
    def __init__(self):
        
        rospy.init_node('liveplotter', anonymous=True)
        
        # right wing
        service_name = "wing_intensities_right"
        rospy.wait_for_service(service_name)
        self.get_wing_intensities_right = rospy.ServiceProxy(service_name, SrvFloat32List)
        
        service_name = "wing_angles_right"
        rospy.wait_for_service(service_name)
        self.get_wing_angles_right = rospy.ServiceProxy(service_name, SrvFloat32List)
        
        service_name = "wing_edges_right"
        rospy.wait_for_service(service_name)
        self.get_wing_edges_right = rospy.ServiceProxy(service_name, SrvFloat32List)
        
        # left wing
        service_name = "wing_intensities_left"
        rospy.wait_for_service(service_name)
        self.get_wing_intensities_left = rospy.ServiceProxy(service_name, SrvFloat32List)
        
        service_name = "wing_angles_left"
        rospy.wait_for_service(service_name)
        self.get_wing_angles_left = rospy.ServiceProxy(service_name, SrvFloat32List)
        
        service_name = "wing_edges_left"
        rospy.wait_for_service(service_name)
        self.get_wing_edges_left = rospy.ServiceProxy(service_name, SrvFloat32List)
        
        #
        
        self.last_update = time.time()
        
        # live plot
        self.fig = plt.figure()
        self.angles_right = self.get_wing_angles_right(1).data
        self.angles_left = self.get_wing_angles_left(1).data

        self.limits_right = [-np.pi, np.pi]
        self.limits_left = [-np.pi, np.pi]
        self.edges_right = [-np.pi, np.pi]
        self.edges_left = [-np.pi, np.pi]
        
        data_right = np.random.random(len(self.angles_right))
        data_left = np.random.random(len(self.angles_left))
        
        colorRight = 'red'
        colorLeft = 'green'
        
        
        self.line_right,          = plt.plot(self.angles_right, data_right, '.', color=colorRight)
        self.line_lo_right,       = plt.plot([self.limits_right[0], self.limits_right[0]], [0,1], color='black', linewidth=1)
        self.line_hi_right,       = plt.plot([self.limits_right[1], self.limits_right[1]], [0,1], color='black', linewidth=1)
        self.line_trailing_right, = plt.plot([self.edges_right[0], self.edges_right[0]], [0,1], color=colorRight, linewidth=1)
        self.line_leading_right,  = plt.plot([self.edges_right[1], self.edges_right[1]], [0,1], color=colorRight, linewidth=1)

        self.line_left,           = plt.plot(self.angles_left, data_left, '.', color=colorLeft)
        self.line_lo_left,        = plt.plot([self.limits_right[0], self.limits_right[0]], [0,1], color='black', linewidth=1)
        self.line_hi_left,        = plt.plot([self.limits_right[1], self.limits_right[1]], [0,1], color='black', linewidth=1)
        self.line_trailing_left,  = plt.plot([self.edges_left[0], self.edges_right[0]], [0,1], color=colorLeft, linewidth=1)
        self.line_leading_left,   = plt.plot([self.edges_left[1], self.edges_left[1]], [0,1], color=colorLeft, linewidth=1)
        
        #plt.ylim(0.1, 0.1)
        plt.xlim(-np.pi,np.pi)
        plt.ylim(0,1)
        #plt.autoscale(True)
                
        #self.image_animation = animation.FuncAnimation(self.fig, self.update_line, self.angles, init_func=self.init_plot, interval=50, blit=True)
        self.image_animation = animation.FuncAnimation(self.fig, self.update_line, init_func=self.init_plot, interval=50, blit=True)
        
        plt.show()
        
        
    def update_line(self, i):
        try:
            rv = (self.line_lo_right, 
                  self.line_hi_right, 
                  self.line_lo_left, 
                  self.line_hi_left, 
                  self.line_trailing_right, 
                  self.line_leading_right, 
                  self.line_trailing_left, 
                  self.line_leading_left)

            if time.time() - self.last_update > 1:
                self.angles_right = self.get_wing_angles_right(1).data
                self.angles_left = self.get_wing_angles_left(1).data

                self.limits_right = [rospy.get_param('strokelitude/right/angle_lo'), rospy.get_param('strokelitude/right/angle_hi')]
                self.limits_left = [rospy.get_param('strokelitude/left/angle_lo'), rospy.get_param('strokelitude/left/angle_hi')]
                self.last_update = time.time()
                
            data_right = self.get_wing_intensities_right(1).data
            data_left = self.get_wing_intensities_left(1).data
            
            self.edges_right = self.get_wing_edges_right(1).data
            self.edges_left = self.get_wing_edges_left(1).data
            
            if data_right is not None:
                self.line_right.set_data(self.angles_right, data_right)
            if data_left is not None:
                self.line_left.set_data(self.angles_left, data_left)
            
            self.line_lo_right.set_data([self.limits_right[0], self.limits_right[0]], [0,1])
            self.line_hi_right.set_data([self.limits_right[1], self.limits_right[1]], [0,1])
            self.line_trailing_right.set_data([self.edges_right[0], self.edges_right[0]], [0,1])
            self.line_leading_right.set_data([self.edges_right[1], self.edges_right[1]], [0,1])
            
            self.line_lo_left.set_data([self.limits_left[0], self.limits_left[0]], [0,1])
            self.line_hi_left.set_data([self.limits_left[1], self.limits_left[1]], [0,1])
            self.line_trailing_left.set_data([self.edges_left[0], self.edges_left[0]], [0,1])
            self.line_leading_left.set_data([self.edges_left[1], self.edges_left[1]], [0,1])
    
            if (len(self.angles_right)==len(data_right)) and (len(self.angles_left)==len(data_left)):        
                rv = (self.line_right, 
                      self.line_left, 
                      self.line_lo_right, 
                      self.line_hi_right, 
                      self.line_lo_left, 
                      self.line_hi_left, 
                      self.line_trailing_right, 
                      self.line_leading_right, 
                      self.line_trailing_left, 
                      self.line_leading_left)
        
        except:
            pass

        return rv
        
        
    def init_plot(self): # required to start with clean slate
        self.line_right.set_data([],[])
        self.line_left.set_data([],[])

        self.line_hi_right.set_data([],[])
        self.line_lo_right.set_data([],[])
        self.line_trailing_right.set_data([],[])
        self.line_leading_right.set_data([],[])

        self.line_hi_left.set_data([],[])
        self.line_lo_left.set_data([],[])
        self.line_trailing_left.set_data([],[])
        self.line_leading_left.set_data([],[])
        
        return (self.line_right, 
                self.line_left, 
                self.line_lo_right, 
                self.line_hi_right, 
                self.line_lo_left, 
                self.line_hi_left, 
                self.line_trailing_right, 
                self.line_leading_right, 
                self.line_trailing_left, 
                self.line_leading_left)
        

if __name__ == '__main__':
    live_plotter = Live_Plotter()
