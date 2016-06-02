#!/usr/bin/env python
import rospy
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
# AreaTracker()
# Track a bodypart using the image correlation technique.  i.e. track the
# motion based on the movement of the pixels in the area of the image.
# Usually used for Head or Abdomen.
#
class AreaTracker(MotionTrackedBodypartPolar):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        MotionTrackedBodypartPolar.__init__(self, name, params, color, bEqualizeHist)
        
        self.phasecorr          = imageprocessing.PhaseCorrelation()
        self.state              = MsgState() # In the bodypart frame.
        self.stateOrigin_p      = MsgState() # In the bodypart frame.
        self.stateLo_p          = MsgState() # In the bodypart frame.
        self.stateHi_p          = MsgState() # In the bodypart frame.

        self.imgComparison      = None
        
        self.windowStabilized   = ImageWindow(False, self.name+'Stable')
        self.windowComparison   = ImageWindow(True, self.name+'Comparison')
        self.set_params(params)

    
    # set_params()
    # Set the given params dict into this object.
    #
    def set_params(self, params):
        MotionTrackedBodypartPolar.set_params(self, params)
        
        self.imgRoiBackground = None
        self.imgComparison = None
        self.windowComparison.set_image(self.imgComparison)
        
        self.iCount = 0

        # Compute the 'handedness' of the head/abdomen and wing/wing axes.  self.sense specifies the direction of positive angles.
        matAxes = np.array([[self.params['gui']['head']['hinge']['x']-self.params['gui']['abdomen']['hinge']['x'], self.params['gui']['head']['hinge']['y']-self.params['gui']['abdomen']['hinge']['y']],
                            [self.params['gui']['right']['hinge']['x']-self.params['gui']['left']['hinge']['x'], self.params['gui']['right']['hinge']['y']-self.params['gui']['left']['hinge']['y']]])
        if (self.name in ['left','right']):
            self.senseAxes = np.sign(np.linalg.det(matAxes))
            a = -1 if (self.name=='right') else 1
            self.sense = a*self.senseAxes
        else:
            self.sense = 1  


        self.stateOrigin_p.intensity = 0.0
        self.stateOrigin_p.angles    = [self.transform_angle_p_from_b((self.params['gui'][self.name]['angle_hi']+self.params['gui'][self.name]['angle_lo'])/2.0)] # In the bodypart frame.
        self.stateOrigin_p.angles[0] = ((self.stateOrigin_p.angles[0] + np.pi) % (2*np.pi)) - np.pi

        self.stateOrigin_p.radii     = [(self.params['gui'][self.name]['radius_outer']+self.params['gui'][self.name]['radius_inner'])/2.0]
        
        self.state.intensity = 0.0
        self.state.angles    = [0.0]
        self.state.radii     = [0.0]

        self.stateLo_p.intensity = np.inf
        self.stateLo_p.angles    = [4.0*np.pi]
        self.stateLo_p.radii     = [np.inf]

        self.stateHi_p.intensity = -np.inf
        self.stateHi_p.angles    = [-4.0*np.pi]
        self.stateHi_p.radii     = [-np.inf]
        
        
        self.windowStabilized.set_enable(self.params['gui']['windows'] and self.params['gui'][self.name]['track'] and self.params['gui'][self.name]['stabilize'])
        self.windowComparison.set_enable(self.params['gui']['windows'] and self.params['gui'][self.name]['track'])


        
    # update_state()
    # Update the bodypart translation & rotation.
    #
    def update_state(self):
        imgNow = self.imgRoiFgMaskedPolarCroppedWindowed

        
        if (imgNow is not None) and (self.imgComparison is not None):
            # Get the rotation & expansion between images.
            (rShift, aShift) = self.phasecorr.get_shift(imgNow, self.imgComparison)
            
            # Convert polar pixel shifts to radians rotation & pixel expansion.
            angleOffset = self.sense * aShift * (self.params['gui'][self.name]['angle_hi']-self.params['gui'][self.name]['angle_lo']) / float(imgNow.shape[1])
            radiusOffset = rShift
            self.state.angles  = [(self.stateOrigin_p.angles[0] + angleOffset)]
            self.state.angles  = [((self.state.angles[0] + np.pi) % (2*np.pi)) - np.pi]
            self.state.radii   = [self.stateOrigin_p.radii[0] + radiusOffset]

            
            # Get min,max's
            self.stateLo_p.angles  = [min(self.stateLo_p.angles[0], self.state.angles[0])]
            self.stateHi_p.angles  = [max(self.stateHi_p.angles[0], self.state.angles[0])]
            self.stateLo_p.radii   = [min(self.stateLo_p.radii[0], self.state.radii[0])]
            self.stateHi_p.radii   = [max(self.stateHi_p.radii[0], self.state.radii[0])]
            
            # Control the (angle,radius) origin to be at the midpoint of loangle, hiangle
            # Whenever an image appears that is closer to the midpoint, then
            # take that image as the new origin image.  Thus driving the origin image 
            # toward the midpoint image over time.
            if (self.params[self.name]['autozero']) and (self.iCount>100):
                # If we see a current angle that is closer to the mask midpoint than the ref angle
                # (defined as the mipoint of the movement range), then take a new comparison image, and 
                # shift the movement range (so that the ref angle becomes the current angle).
                angleRef_p = (self.stateHi_p.angles[0] + self.stateLo_p.angles[0])/2.0

                if (angleRef_p < self.state.angles[0] < self.stateOrigin_p.angles[0]) or (self.stateOrigin_p.angles[0] < self.state.angles[0] < angleRef_p):
                    self.imgComparison = imgNow
                    self.windowComparison.set_image(self.imgComparison)

                    # Converge the origin to zero.
                    self.stateLo_p.angles[0] -= angleOffset
                    self.stateHi_p.angles[0] -= angleOffset
                    self.state.angles[0] = self.stateOrigin_p.angles[0]
                    

            # Stabilized image.
            if (self.params['gui'][self.name]['stabilize']):
                # Stabilize the polar image.
                #size = (self.imgRoiFgMaskedPolar.shape[1],
                #        self.imgRoiFgMaskedPolar.shape[0])
                #center = (self.imgRoiFgMaskedPolar.shape[1]/2.0+aShift, 
                #          self.imgRoiFgMaskedPolar.shape[0]/2.0+rShift)
                #self.imgStabilized = cv2.getRectSubPix(self.imgRoiFgMaskedPolar, size, center)
                
                # Stabilize the bodypart in the entire camera image.
                center = (self.params['gui'][self.name]['hinge']['x'], self.params['gui'][self.name]['hinge']['y'])
                size = (self.image.shape[1], self.image.shape[0])
                
                # Stabilize the rotation. 
                T = cv2.getRotationMatrix2D(center, np.rad2deg(self.state.angles[0]), 1.0)
                
                # Stabilize the expansion.
                T[0,2] -= rShift * self.cosAngleBody_i
                T[1,2] -= rShift * self.sinAngleBody_i 
                
                self.imgStabilized = cv2.warpAffine(self.image, T, size)
                self.windowStabilized.set_image(self.imgStabilized)

            
        if (self.imgRoiFgMasked is not None):
            self.state.intensity = float(np.sum(self.imgRoiFgMasked) / self.mask.sum)
        else:
            self.state.intensity = 0.0            
        
        
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, dt, image, bInvertColor):
        MotionTrackedBodypartPolar.update(self, dt, image, bInvertColor)

        if (self.imgComparison is None) and (self.iCount>1):
            self.imgComparison = self.imgRoiFgMaskedPolarCroppedWindowed
            self.windowComparison.set_image(self.imgComparison)
        
        if (self.params['gui'][self.name]['track']):
            self.update_state()
            
    
    # draw()
    # Draw the outline.
    #
    def draw(self, image):
        MotionTrackedBodypartPolar.draw(self, image)

        if (self.params['gui'][self.name]['track']):
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
            
            # Set a pixel at the min/max state positions.
            try:
                angle = self.stateLo_p.angles[0] * self.sense
                pt = self.R.dot([self.state.radii[0]*np.cos(angle), 
                                 self.state.radii[0]*np.sin(angle)]) 
                ptStateLo_i = imageprocessing.clip_pt((int(pt[0]+self.params['gui'][self.name]['hinge']['x']), 
                                       int(pt[1]+self.params['gui'][self.name]['hinge']['y'])), image.shape) 
                
                
                angle = self.stateHi_p.angles[0] * self.sense
                pt = self.R.dot([self.state.radii[0]*np.cos(angle), 
                                 self.state.radii[0]*np.sin(angle)]) 
                ptStateHi_i = imageprocessing.clip_pt((int(pt[0]+self.params['gui'][self.name]['hinge']['x']), 
                                       int(pt[1]+self.params['gui'][self.name]['hinge']['y'])), image.shape) 
                
                # Set the pixels.
                image[ptStateLo_i[1]][ptStateLo_i[0]] = np.array([255,255,255]) - image[ptStateLo_i[1]][ptStateLo_i[0]]
                image[ptStateHi_i[1]][ptStateHi_i[0]] = np.array([255,255,255]) - image[ptStateHi_i[1]][ptStateHi_i[0]]
                
            except ValueError:
                pass

            
            # Draw line from hinge to state point.        
            ptHinge_i = imageprocessing.clip_pt((int(self.ptHinge_i[0]), int(self.ptHinge_i[1])), image.shape) 
            cv2.line(image, ptHinge_i, ptState_i, self.bgra_state, 1)
            
            self.windowStabilized.show()
            self.windowComparison.show()
        
# end class AreaTracker

    


