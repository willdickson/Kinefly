#!/usr/bin/env python

import rospy
import cv
import cv2
import numpy as np
from imagewindow import ImageWindow
import imageprocessing
from bodypart_motion import MotionTrackedBodypart, MotionTrackedBodypartPolar
from Kinefly.srv import SrvTrackerdata, SrvTrackerdataResponse
from Kinefly.msg import MsgFlystate, MsgState
import ui



###############################################################################
###############################################################################
# Find the lowest image point where image intensity exceeds a threshold.
#
class TipDetector(object):
    def __init__(self, threshold=0.0, sense=1):
        self.intensities = []
        self.diff = []
        self.set_params(threshold, sense)
        self.imgThreshold = None


    def set_params(self, threshold, sense):
        self.threshold = threshold
        self.sense = sense


    # detect()
    # Get the pixel position in the given image of the bottommost thresholded value.
    #
    def detect(self, image):
        #(threshold, img) = cv2.threshold(image, int(self.threshold*255), 255, cv2.THRESH_TOZERO) # BINARY)#
        (threshold, img) = (0, image)
        
        axis = 1
        self.intensities = np.sum(img, axis).astype(np.float32)
        self.intensities /= (255.0 * img.shape[axis]) # Put into range [0,1]

        # Compute the intensity gradient. 
        n = 2
        a = np.append(self.intensities[n:], self.intensities[-n]*np.ones(n))
        b = np.append(self.intensities[n]*np.ones(n), self.intensities[:-n])
        self.diff = b-a
        
        
        
        i = np.where(self.diff > self.threshold)[0]
        if (len(i)>0):
            yTip = i[-1]
        else:
            yTip = None
        
        if (yTip is not None):
            line = img[yTip,:]
            xTip = line.argmax()
        else:
            xTip = None
        
        return (xTip, yTip)
        
           
    
# End class TipDetector
    
    
###############################################################################
###############################################################################
# TipTracker()
# Track the position of the tip of a bodypart, e.g. a wingtip.  
# Finds the point at the greatest distance from the hinge.
#
class TipTracker(MotionTrackedBodypartPolar):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        MotionTrackedBodypartPolar.__init__(self, name, params, color, bEqualizeHist)
        
        self.name     = name
        self.detector = TipDetector()
        self.state    = MsgState()
        self.set_params(params)
        self.windowTip = ImageWindow(False, self.name+'Tip')
        self.iAngle = 0
        self.iRadius = 0

        # Services, for live intensities plots via live_wing_histograms.py
        self.service_tipdata    = rospy.Service('trackerdata_'+self.name, SrvTrackerdata, self.serve_tipdata_callback)

    
    # set_params()
    # Set the given params dict into this object.
    #
    def set_params(self, params):
        MotionTrackedBodypartPolar.set_params(self, params)
        
        self.imgRoiBackground = None
        self.iCount = 0
        self.state.intensity = 0.0
        self.state.angles = []
        self.state.gradients = []

        # Compute the 'handedness' of the head/abdomen and wing/wing axes.  self.sense specifies the direction of positive angles.
        matAxes = np.array([[self.params['gui']['head']['hinge']['x']-self.params['gui']['abdomen']['hinge']['x'], self.params['gui']['head']['hinge']['y']-self.params['gui']['abdomen']['hinge']['y']],
                            [self.params['gui']['right']['hinge']['x']-self.params['gui']['left']['hinge']['x'], self.params['gui']['right']['hinge']['y']-self.params['gui']['left']['hinge']['y']]])
        if (self.name in ['left','right']):
            self.senseAxes = np.sign(np.linalg.det(matAxes))
            a = -1 if (self.name=='right') else 1
            self.sense = a*self.senseAxes
        else:
            self.sense = 1  

        self.detector.set_params(params[self.name]['threshold'], self.sense)
    
        
    # update_state()
    #
    def update_state(self):
        imgNow = self.imgRoiFgMaskedPolarCropped
        
        # Get the rotation & expansion between images.
        if (imgNow is not None):
            # Pixel position and strength of the edges.
            (self.iAngle, self.iRadius) = self.detector.detect(imgNow)

            if (self.iAngle is not None) and (self.iRadius is not None):
                # Convert pixel to angle units, and put angle into the wing frame.
                anglePerPixel = (self.params['gui'][self.name]['angle_hi']-self.params['gui'][self.name]['angle_lo']) / float(imgNow.shape[1])
                angle_b = self.params['gui'][self.name]['angle_lo'] + self.iAngle * anglePerPixel
                angle_p = (self.transform_angle_p_from_b(angle_b) + np.pi) % (2*np.pi) - np.pi
                self.state.angles = [angle_p]
                self.state.gradients = self.detector.diff
                
                radius = self.params['gui'][self.name]['radius_inner'] + self.iRadius
                self.state.radii = [radius]
                self.state.intensity = np.mean(imgNow)/255.0
            else:
                self.state.angles = []
                self.state.radii = []
                self.state.gradients = self.detector.diff
                self.state.intensity = np.mean(imgNow)/255.0
                

        
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, dt, image, bInvertColor):
        MotionTrackedBodypartPolar.update(self, dt, image, bInvertColor)

        if (self.params['gui'][self.name]['track']):
            self.update_state()
            self.windowTip.set_image(self.detector.imgThreshold)
            
            
    
    
    # draw()
    # Draw the state.
    #
    def draw(self, image):
        MotionTrackedBodypartPolar.draw(self, image)

        if (self.params['gui'][self.name]['track']) and (len(self.state.angles)>0):
            # Draw the bodypart state position.
            angle = self.state.angles[0] * self.sense
            pt = self.R.dot([self.state.radii[0]*np.cos(angle), 
                             self.state.radii[0]*np.sin(angle)]) 
            ptState_i = imageprocessing.clip_pt((int(pt[0]+self.params['gui'][self.name]['hinge']['x']), 
                                                 int(pt[1]+self.params['gui'][self.name]['hinge']['y'])), image.shape) 
            
            cv2.ellipse(image,
                        ptState_i,
                        (2,2),
                        0,
                        0,
                        360,
                        self.bgra_state, 
                        1)
            
            # Draw line from hinge to state point.        
            ptHinge_i = imageprocessing.clip_pt((int(self.ptHinge_i[0]), int(self.ptHinge_i[1])), image.shape) 
            cv2.line(image, ptHinge_i, ptState_i, self.bgra_state, 1)
            self.windowTip.show()
        
        
    def serve_tipdata_callback(self, request):
        intensities = self.detector.intensities
        diffs = self.detector.diff
        abscissa = range(len(intensities))
        if (self.iRadius is not None):
            markersH = self.iRadius
        else:
            markersH = 0
            
        markersV = self.params[self.name]['threshold']
        
            
        return SrvTrackerdataResponse(self.color, abscissa, intensities, diffs, markersH, markersV)
        
        
# end class TipTracker

    

