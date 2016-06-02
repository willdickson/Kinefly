#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import ui
import imageprocessing


###############################################################################
###############################################################################
# AxisTracker()
# Track the body axis.
#
class AxisTracker:
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        self.name           = name
        self.bEqualizeHist  = bEqualizeHist
        self.color          = color
        self.bgra           = ui.bgra_dict[color]
        self.bgra_dim       = tuple(0.5*np.array(ui.bgra_dict[color]))
        self.bgra_state     = ui.bgra_dict[color]
        self.pixelmax       = 255.0
        self.iCount         = 0
        self.bValidMask     = False
  
        self.params = {}
        self.pt1_i = np.array([0,0])
        self.pt2_i = np.array([0,0])
        self.handles = {'pt1':ui.Handle(self.pt1_i, self.bgra, name='pt1'),
                        'pt2':ui.Handle(self.pt2_i, self.bgra, name='pt2'),
                        }
        

    
    # set_params()
    # Set the given params dict into this object, and cache a few values.
    #
    def set_params(self, params):
        self.params = params

        self.angleBody_i = self.get_bodyangle_i()
        self.cosAngleBody_i = np.cos(self.angleBody_i)
        self.sinAngleBody_i = np.sin(self.angleBody_i)

        self.pt1_i        = np.array([self.params['gui'][self.name]['pt1']['x'], self.params['gui'][self.name]['pt1']['y']], dtype=np.int16)
        self.pt2_i        = np.array([self.params['gui'][self.name]['pt2']['x'], self.params['gui'][self.name]['pt2']['y']], dtype=np.int16)

        
        # Refresh the handle points.
        self.update_handle_points()


    def set_background(self, image):
        pass


    def create_mask(self, shape):
        self.bValidMask = True


    def get_bodyangle_i(self):
        angle_i = imageprocessing.get_angle_from_points_i(self.pt1_i, self.pt2_i)
        #angleBody_i  = (angle_i + np.pi) % (2.0*np.pi) - np.pi
        angleBody_i  = float(angle_i)
        return angleBody_i 
        
                
    # update_handle_points()
    # Update the dictionary of handle point names and locations.
    # Compute the various handle points.
    #
    def update_handle_points (self):
        self.handles['pt1'].pt  = self.pt1_i
        self.handles['pt2'].pt  = self.pt2_i

        
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, dt, image, bInvertColor):
        self.iCount += 1
        self.dt = dt
                


    # hit_object()
    # Get the UI object, if any, that the mouse is on.    
    def hit_object(self, ptMouse):
        tag = None
        
        # Check for handle hits.
        for tagHandle,handle in self.handles.iteritems():
            if (handle.hit_test(ptMouse)):
                tag = tagHandle
                break
                
        return (self.name, tag)
    

    def draw_handles(self, image):
        # Draw all handle points.
        for tagHandle,handle in self.handles.iteritems():
            handle.draw(image)

    
    # draw()
    # Draw the outline.
    #
    def draw(self, image):
        if (self.params['gui'][self.name]['track']):
            # Draw line to indicate the body axis.
            cv2.line(image, (self.pt1_i[0],self.pt1_i[1]), (self.pt2_i[0],self.pt2_i[1]), self.bgra, 1)

            self.draw_handles(image)
    
                
# end class AxisTracker

    


