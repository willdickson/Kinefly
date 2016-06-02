#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from bodypart_intensity import IntensityTrackedBodypart
from Kinefly.srv import SrvTrackerdata, SrvTrackerdataResponse
from Kinefly.msg import MsgFlystate, MsgState
import ui
from wingbeatdetector import WingbeatDetector



###############################################################################
###############################################################################
# The 'aux' area to track intensity.
class IntensityTracker(IntensityTrackedBodypart):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        IntensityTrackedBodypart.__init__(self, name, params, color, bEqualizeHist)
        
        self.state = MsgState()
        self.state.intensity = 0.0
        self.state.freq = 0.0
        
        self.wingbeat = WingbeatDetector(0, 1000)

        self.set_params(params)

        
    
    # set_params()
    # Set the given params dict into this object.
    #
    def set_params(self, params):
        IntensityTrackedBodypart.set_params(self, params)

        self.imgRoiBackground = None
        self.iCount = 0
        self.state.intensity = 0.0

        self.wingbeat.set(self.params['wingbeat_min'], 
                          self.params['wingbeat_max'])

        
    # update_state()
    #
    def update_state(self):
        #f = 175         # Simulate this freq.
        #t = gImageTime 
        #self.state.intensity = np.cos(2*np.pi*f*t)
        self.state.intensity = np.sum(self.imgRoiFgMasked).astype(np.float32) / self.mask.sum
        self.state.freq = self.wingbeat.freq_from_intensity(self.state.intensity, 1.0/self.dt)
    
        
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, dt, image, bInvertColor):
        IntensityTrackedBodypart.update(self, dt, image, bInvertColor)

        if (self.params['gui'][self.name]['track']):
            self.update_state()
    
            


    # draw()
    # Draw the outline.
    #
    def draw(self, image):
        IntensityTrackedBodypart.draw(self, image)

# end class IntensityTracker

    


