#!/usr/bin/env python

import rospy
import copy
import cv
import cv2
import numpy as np
import threading
from imagewindow import ImageWindow
import imageprocessing
import ui

class Struct:
    pass

gbShowMasks = False

###############################################################################
###############################################################################
# Contains the common behavior for tracking the motion of a bodypart via a polar 
# coordinates transform, e.g. Head, Abdomen, or Wing.
#
class MotionTrackedBodypart(object):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        self.name           = name
        self.bEqualizeHist  = bEqualizeHist

        self.bValidMask     = False

        self.color          = color
        self.bgra           = ui.bgra_dict[color]
        self.bgra_dim       = tuple(0.5*np.array(ui.bgra_dict[color]))
        self.bgra_state     = ui.bgra_dict[color]#ui.bgra_dict['red']
        self.pixelmax       = 255.0
                        
        self.shape          = (np.inf, np.inf)
        self.ptHinge_i      = np.array([0,0])
        self.roi            = None
        
        self.mask       = Struct()
        self.mask.img   = None
        self.mask.sum   = 1.0
        self.mask.xMin = None
        self.mask.yMin = None
        self.mask.xMax = None
        self.mask.yMax = None
        
        self.dt = np.inf

        self.params = {}
        self.handles = {'hinge':ui.Handle(np.array([0,0]), self.bgra, name='hinge'),
                        'angle_hi':ui.Handle(np.array([0,0]), self.bgra, name='angle_hi'),
                        'angle_lo':ui.Handle(np.array([0,0]), self.bgra, name='angle_lo'),
                        'radius_inner':ui.Handle(np.array([0,0]), self.bgra, name='radius_inner')
                        }
        
        self.lockBackground = threading.Lock()

        # Region of interest images.
        self.imgFullBackground                  = None
        self.imgRoiBackground                   = None
        self.imgRoi                             = None # Untouched roi image.
        self.imgRoiFg                           = None # Background subtracted.
        self.imgRoiFgMasked                     = None
        self.imgFinal                           = None # The image to use for tracking.

        # Extra windows.
        self.windowBG         = ImageWindow(False, self.name+'BG')
        self.windowFG         = ImageWindow(False, self.name+'FG')
        self.windowMask       = ImageWindow(gbShowMasks, self.name+'Mask')


    
    # set_params()
    # Set the given params dict into this object, and cache a few values.
    #
    def set_params(self, params):
        self.params = params

        self.rc_background = self.params['rc_background']
        self.angleBody_i = self.get_bodyangle_i()
        self.cosAngleBody_i = np.cos(self.angleBody_i)
        self.sinAngleBody_i = np.sin(self.angleBody_i)

        self.ptHinge_i        = np.array([self.params['gui'][self.name]['hinge']['x'], self.params['gui'][self.name]['hinge']['y']])
        self.ptHingeHead_i    = np.array([self.params['gui']['head']['hinge']['x'], self.params['gui']['head']['hinge']['y']])
        self.ptHingeAbdomen_i = np.array([self.params['gui']['abdomen']['hinge']['x'], self.params['gui']['abdomen']['hinge']['y']])

        
        # Compute the body-outward-facing angle, which is the angle from the body center to the bodypart hinge.
#         pt1 = [params['head']['hinge']['x'], params['head']['hinge']['y']]
#         pt2 = [params['abdomen']['hinge']['x'], params['abdomen']['hinge']['y']]
#         pt3 = [params['gui']['left']['hinge']['x'], params['gui']['left']['hinge']['y']]
#         pt4 = [params['right']['hinge']['x'], params['right']['hinge']['y']]
#         ptBodyCenter_i = get_intersection(pt1,pt2,pt3,pt4)
#         self.angleBodypart_i = float(np.arctan2(self.ptHinge_i[1]-ptBodyCenter_i[1], self.ptHinge_i[0]-ptBodyCenter_i[0]))

        # Compute the body-outward-facing angle, which is the angle to the current point from the forward body axis.
        if (self.name in ['head','abdomen']):
            nameRelative = {'head':'abdomen', 
                            'abdomen':'head', 
                            'left':'right', 
                            'right':'left'}
            self.angleBodypart_i = float(np.arctan2(self.params['gui'][self.name]['hinge']['y']-self.params['gui'][nameRelative[self.name]]['hinge']['y'], 
                                                   self.params['gui'][self.name]['hinge']['x']-self.params['gui'][nameRelative[self.name]]['hinge']['x']))
        else:
            ptBodyaxis_i = self.get_projection_onto_bodyaxis(self.ptHinge_i)
            self.angleBodypart_i = float(np.arctan2(self.params['gui'][self.name]['hinge']['y']-ptBodyaxis_i[1], 
                                                   self.params['gui'][self.name]['hinge']['x']-ptBodyaxis_i[0]))

        self.angleBodypart_b = self.angleBodypart_i - self.angleBody_i
        #rospy.logwarn('%s: %0.2f, %0.2f' % (self.name, self.angleBodypart_i, self.angleBodypart_b))

        
        cosAngleBodypart_i = np.cos(self.angleBodypart_i)
        sinAngleBodypart_i = np.sin(self.angleBodypart_i)
        self.R = np.array([[cosAngleBodypart_i, -sinAngleBodypart_i], 
                           [sinAngleBodypart_i, cosAngleBodypart_i]])

        # Turn on/off the extra windows.
        self.windowBG.set_enable(self.params['gui']['windows'] and self.params['gui'][self.name]['track'] and self.params['gui'][self.name]['subtract_bg'])
        self.windowFG.set_enable(self.params['gui']['windows'] and self.params['gui'][self.name]['track'])

        self.angle_hi_i = self.transform_angle_i_from_b(self.params['gui'][self.name]['angle_hi'])
        self.angle_lo_i = self.transform_angle_i_from_b(self.params['gui'][self.name]['angle_lo'])
        
        # Refresh the handle points.
        self.update_handle_points()


    # get_projection_onto_bodyaxis()
    # Project the given point onto the body axis.
    #
    def get_projection_onto_bodyaxis(self, ptAnywhere):
        # Project the point onto the body axis.
        ptB = self.ptHingeHead_i - self.ptHingeAbdomen_i
        ptM = ptAnywhere - self.ptHingeAbdomen_i
        ptAxis = np.dot(ptB,ptM) / np.dot(ptB,ptB) * ptB + self.ptHingeAbdomen_i
            
        return ptAxis
        
                
    # transform_angle_b_from_p()
    # Transform an angle from the bodypart frame to the fly body frame.
    #
    def transform_angle_b_from_p(self, angle_p):
        angle_b =  self.sense * angle_p + self.angleBodypart_b
        return angle_b


    # transform_angle_p_from_b()
    # Transform an angle from the fly body frame to the bodypart frame.
    #
    def transform_angle_p_from_b(self, angle_b):
        angle_p =  self.sense * (angle_b - self.angleBodypart_b)
        return angle_p


    # transform_angle_i_from_b()
    # Transform an angle from the fly body frame to the camera image frame.
    #
    def transform_angle_i_from_b(self, angle_b):
        angle_i = angle_b + self.angleBody_i 
        return angle_i
        

    # transform_angle_b_from_i()
    # Transform an angle from the camera image frame to the fly frame: longitudinal axis head is 0, CW positive.
    #
    def transform_angle_b_from_i(self, angle_i):
        angle_b = angle_i - self.angleBody_i
        angle_b = ((angle_b+np.pi) % (2.0*np.pi)) - np.pi

        return angle_b
         

    def get_bodyangle_i(self):
        angle_i = imageprocessing.get_angle_from_points_i(np.array([self.params['gui']['abdomen']['hinge']['x'], self.params['gui']['abdomen']['hinge']['y']]), 
                                          np.array([self.params['gui']['head']['hinge']['x'], self.params['gui']['head']['hinge']['y']]))
        #angleBody_i  = (angle_i + np.pi) % (2.0*np.pi) - np.pi
        angleBody_i  = float(angle_i)
        return angleBody_i 
        
                
    # create_mask()
    # Create elliptical wedge masks, and window functions.
    #
    def create_mask(self, shape):
        # Create the mask.
        img = np.zeros(shape, dtype=np.uint8)
        
        # Args for the ellipse calls.
        x = int(self.params['gui'][self.name]['hinge']['x'])
        y = int(self.params['gui'][self.name]['hinge']['y'])
        r_outer = int(np.ceil(self.params['gui'][self.name]['radius_outer']))
        r_inner = int(np.floor(self.params['gui'][self.name]['radius_inner']))-1
        hi = int(np.ceil(np.rad2deg(self.angle_hi_i)))
        lo = int(np.floor(np.rad2deg(self.angle_lo_i)))
        
        # Draw the mask.
        cv2.ellipse(img, (x, y), (r_outer, r_outer), 0, hi, lo, ui.bgra_dict['white'], cv.CV_FILLED)
        cv2.ellipse(img, (x, y), (r_inner, r_inner), 0, 0, 360, ui.bgra_dict['black'], cv.CV_FILLED)
        img = cv2.dilate(img, np.ones([3,3])) # Make the mask one pixel bigger to account for pixel aliasing.
        self.windowMask.set_image(img)
        
        
        # Find the ROI of the mask.
        xSum = np.sum(img, 0)
        ySum = np.sum(img, 1)
        x_list = np.where(xSum>0)[0]
        y_list = np.where(ySum>0)[0]
        
        if (len(x_list)>0) and (len(y_list)>0): 
            # Get the extents.
            xMin = np.where(xSum>0)[0][0]  
            xMax = np.where(xSum>0)[0][-1] + 1
            yMin = np.where(ySum>0)[0][0]  
            yMax = np.where(ySum>0)[0][-1] + 1
            
            # Clip border to image edges.
            self.mask.xMin = np.max([0,xMin])
            self.mask.yMin = np.max([0,yMin])
            self.mask.xMax = np.min([xMax, shape[1]-1])
            self.mask.yMax = np.min([yMax, shape[0]-1])
            
            self.roi = np.array([self.mask.xMin, self.mask.yMin, self.mask.xMax, self.mask.yMax])
            self.mask.img = img[self.mask.yMin:self.mask.yMax, self.mask.xMin:self.mask.xMax]
            self.mask.sum = np.sum(self.mask.img).astype(np.float32)
            self.bValidMask = True  

        else:
            self.mask.xMin = None
            self.mask.yMin = None
            self.mask.xMax = None
            self.mask.yMax = None

            rospy.logwarn('%s: Empty mask.' % self.name)      
            

        
    # set_background()
    # Set the given image as the background image.
    #                
    def set_background(self, image):
        with self.lockBackground:
            self.imgFullBackground = image.astype(np.float32)
            self.imgRoiBackground = None
        
        
    # invert_background()
    # Invert the color of the background image.
    #                
    def invert_background(self):
        with self.lockBackground:
            if (self.imgRoiBackground is not None):
                self.imgRoiBackground = 255-self.imgRoiBackground
        
        
    def update_background(self):
        alphaBackground = 1.0 - np.exp(-self.dt / self.rc_background)

        with self.lockBackground:
            if (self.imgRoiBackground is not None):
                if (self.imgRoi is not None):
                    if (self.imgRoiBackground.size==self.imgRoi.size):
                        cv2.accumulateWeighted(self.imgRoi.astype(np.float32), self.imgRoiBackground, alphaBackground)
                    else:
                        self.imgRoiBackground = None
                        self.imgRoi = None
            else:
                if (self.imgFullBackground is not None) and (self.roi is not None):
                    self.imgRoiBackground = copy.deepcopy(self.imgFullBackground[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]])
                    
            if (self.imgRoiBackground is not None):
                self.imgHeadroom = (255 - self.imgRoiBackground).astype(np.float32) / 255.0
            else:
                self.imgHeadroom = None
                
            self.windowBG.set_image(self.imgRoiBackground)
        

    def update_roi(self, image, bInvertColor):
        self.image = image
        self.shape = image.shape
        
        # Extract the ROI images.
        if (self.roi is not None):
            self.imgRoi = copy.deepcopy(image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]])

            # Background Subtraction.
            if (bInvertColor):
                self.imgRoiFg = 255-self.imgRoi
            else:
                self.imgRoiFg = self.imgRoi

            if (self.params['gui'][self.name]['subtract_bg']):
                with self.lockBackground:
                    if (self.imgRoiBackground is not None):
                        if (self.imgRoiBackground.shape==self.imgRoiFg.shape):
                            if (bInvertColor):
                                self.imgRoiFg = cv2.absdiff(self.imgRoiFg, 255-self.imgRoiBackground.astype(np.uint8))
                            else:
                                self.imgRoiFg = cv2.absdiff(self.imgRoiFg, self.imgRoiBackground.astype(np.uint8))
                        
                
            # Equalize the brightness/contrast.
            if (self.bEqualizeHist):
                if (self.imgRoiFg is not None):
                    self.imgRoiFg -= np.min(self.imgRoiFg)
                    max2 = np.max([1.0, np.max(self.imgRoiFg)])
    
                    self.imgRoiFg *= (255.0/float(max2))
                
            self.windowFG.set_image(self.imgRoiFg) 
        

            
    # update_handle_points()
    # Update the dictionary of handle point names and locations.
    # Compute the various handle points.
    #
    def update_handle_points (self):
        x = self.params['gui'][self.name]['hinge']['x']
        y = self.params['gui'][self.name]['hinge']['y']
        radius_outer = self.params['gui'][self.name]['radius_outer']
        radius_inner = self.params['gui'][self.name]['radius_inner']
        angle = (self.angle_hi_i+self.angle_lo_i)/2.0
        
        
        self.handles['hinge'].pt        = np.array([x, y])
        self.handles['radius_inner'].pt = np.array([x, y]) + ((radius_inner) * np.array([np.cos(angle),np.sin(angle)]))
        self.handles['angle_hi'].pt     = np.array([x, y]) + np.array([(radius_outer)*np.cos(self.angle_hi_i), (radius_outer)*np.sin(self.angle_hi_i)])
        self.handles['angle_lo'].pt     = np.array([x, y]) + np.array([(radius_outer)*np.cos(self.angle_lo_i), (radius_outer)*np.sin(self.angle_lo_i)])

        self.ptWedgeHi_outer = tuple((np.array([x, y]) + np.array([radius_outer*np.cos(self.angle_hi_i), (radius_outer)*np.sin(self.angle_hi_i)])).astype(int))
        self.ptWedgeHi_inner = tuple((np.array([x, y]) + np.array([radius_inner*np.cos(self.angle_hi_i), (radius_inner)*np.sin(self.angle_hi_i)])).astype(int))
        self.ptWedgeLo_outer = tuple((np.array([x, y]) + np.array([radius_outer*np.cos(self.angle_lo_i), (radius_outer)*np.sin(self.angle_lo_i)])).astype(int))
        self.ptWedgeLo_inner = tuple((np.array([x, y]) + np.array([radius_inner*np.cos(self.angle_lo_i), (radius_inner)*np.sin(self.angle_lo_i)])).astype(int))
        
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, dt, image, bInvertColor):
        self.iCount += 1
        
        self.dt = dt
                
        if (self.params['gui'][self.name]['track']):
            if (not self.bValidMask):
                self.create_mask(image.shape)
                self.bValidMask = True
                
            self.update_background()
            self.update_roi(image, bInvertColor)

            # Apply the mask.
            if (self.imgRoiFg is not None) and (self.imgHeadroom is not None) and (self.mask.img is not None):
#                self.imgRoiFg  = cv2.multiply(self.imgRoiFg.astype(np.float32), self.imgHeadroom)
                self.imgRoiFgMasked = cv2.bitwise_and(self.imgRoiFg.astype(self.mask.img.dtype), self.mask.img)
                #self.imgRoiFgMasked  = cv2.multiply(self.imgRoiFg.astype(self.wfnRoi.dtype), self.wfnRoi)
                
                self.imgFinal = self.imgRoiFgMasked 
            else:
                self.imgFinal = None
                


    # hit_object()
    # Get the UI object, if any, that the mouse is on.    
    def hit_object(self, ptMouse):
        tag = None
        
        # Check for handle hits.
        if (self.params['gui'][self.name]['track']):
            for tagHandle,handle in self.handles.iteritems():
                if (handle.hit_test(ptMouse)):
                    tag = tagHandle
                    break
        else:
            tagHandle,handle = ('hinge',self.handles['hinge'])
            if (handle.hit_test(ptMouse)):
                tag = tagHandle
            
                
        return (self.name, tag)
    

    def draw_handles(self, image):
        # Draw all handle points, or only just the hinge handle.
        if (self.params['gui'][self.name]['track']):
            for tagHandle,handle in self.handles.iteritems():
                handle.draw(image)
        else:
            tagHandle,handle = ('hinge',self.handles['hinge'])
            handle.draw(image)

    
    # draw()
    # Draw the outline.
    #
    def draw(self, image):
        self.draw_handles(image)

        if (self.params['gui'][self.name]['track']):
            x = int(self.params['gui'][self.name]['hinge']['x'])
            y = int(self.params['gui'][self.name]['hinge']['y'])
            radius_outer = int(self.params['gui'][self.name]['radius_outer'])
            radius_inner = int(self.params['gui'][self.name]['radius_inner'])
            radius_mid = int((self.params['gui'][self.name]['radius_outer']+self.params['gui'][self.name]['radius_inner'])/2.0)

#             if ()
#             angle1 = self.angle_lo_i
#             angle2 = self.angle_hi_i
            
            # Draw the mid arc.
            cv2.ellipse(image,
                        (x, y),
                        (radius_mid, radius_mid),
                        np.rad2deg(0.0),
                        np.rad2deg(self.angle_hi_i),
                        np.rad2deg(self.angle_lo_i),
                        self.bgra_dim, 
                        1)

            # Draw the outer arc.
            cv2.ellipse(image,
                        (x, y),
                        (radius_outer, radius_outer),
                        np.rad2deg(0.0),
                        np.rad2deg(self.angle_hi_i),
                        np.rad2deg(self.angle_lo_i),
                        self.bgra_dim, 
                        1)
    
            # Draw the inner arc.
            cv2.ellipse(image,
                        (x, y),
                        (radius_inner, radius_inner),
                        np.rad2deg(0.0),
                        np.rad2deg(self.angle_hi_i),
                        np.rad2deg(self.angle_lo_i),
                        self.bgra_dim, 
                        1)
    
            # Draw wedge lines.        
            cv2.line(image, self.ptWedgeHi_inner, self.ptWedgeHi_outer, self.bgra_dim, 1)
            cv2.line(image, self.ptWedgeLo_inner, self.ptWedgeLo_outer, self.bgra_dim, 1)

    
            # Show the extra windows.
            self.windowBG.show()
            self.windowFG.show()
            self.windowMask.show()
                
# end class MotionTrackedBodypart



###############################################################################
###############################################################################
class MotionTrackedBodypartPolar(MotionTrackedBodypart):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        MotionTrackedBodypart.__init__(self, name, params, color, bEqualizeHist)

        self.polartransforms                    = imageprocessing.PolarTransforms()
        self.create_wfn                         = imageprocessing.WindowFunctions().create_tukey
        self.wfnRoiMaskedPolarCropped           = None
        
        self.imgRoiFgMaskedPolar                = None
        self.imgRoiFgMaskedPolarCropped         = None
        self.imgRoiFgMaskedPolarCroppedWindowed = None
        self.imgFinal                           = None # The image to use for tracking.

        self.windowPolar                        = ImageWindow(False, self.name+'Polar')
        
        
    def set_params(self, params):
        MotionTrackedBodypart.set_params(self, params)
        self.windowPolar.set_enable(self.params['gui']['windows'] and self.params['gui'][self.name]['track'])

        
    def create_mask(self, shape):
        MotionTrackedBodypart.create_mask(self, shape)

        self.wfnRoiMaskedPolarCropped = None

        if (self.mask.xMin is not None): 
            self.i_0 = self.params['gui'][self.name]['hinge']['y'] - self.roi[1]
            self.j_0 = self.params['gui'][self.name]['hinge']['x'] - self.roi[0]

    
            # Args for the ellipse calls.
            x = int(self.params['gui'][self.name]['hinge']['x'])
            y = int(self.params['gui'][self.name]['hinge']['y'])
            r_outer = int(np.ceil(self.params['gui'][self.name]['radius_outer']))
            r_inner = int(np.floor(self.params['gui'][self.name]['radius_inner']))-1
            hi = int(np.ceil(np.rad2deg(self.angle_hi_i)))
            lo = int(np.floor(np.rad2deg(self.angle_lo_i)))

    
            # Find where the mask might be clipped.  First, draw an unclipped ellipse.        
            delta = 1
            ptsUnclipped =                         cv2.ellipse2Poly((x, y), (r_outer, r_outer), 0, hi, lo, delta)
            ptsUnclipped = np.append(ptsUnclipped, cv2.ellipse2Poly((x, y), (r_inner, r_inner), 0, hi, lo, delta), 0)
            #ptsUnclipped = np.append(ptsUnclipped, [list of line1 pixels connecting the two arcs], 0) 
            #ptsUnclipped = np.append(ptsUnclipped, [list of line2 pixels connecting the two arcs], 0) 
    
    
            # These are the unclipped locations.        
            minUnclipped = ptsUnclipped.min(0)
            maxUnclipped = ptsUnclipped.max(0)
            xMinUnclipped = minUnclipped[0]
            yMinUnclipped = minUnclipped[1]
            xMaxUnclipped = maxUnclipped[0]+1
            yMaxUnclipped = maxUnclipped[1]+1

            # Compare unclipped with the as-drawn locations.
            xClip0 = self.mask.xMin - xMinUnclipped
            yClip0 = self.mask.yMin - yMinUnclipped
            xClip1 = xMaxUnclipped - self.mask.xMax
            yClip1 = yMaxUnclipped - self.mask.yMax
            roiClipped = np.array([xClip0, yClip0, xClip1, yClip1])
    
            (i_n, j_n) = shape[:2]
                    
            # Determine how much of the bottom of the polar image to trim off (i.e. rClip) based on if the ellipse is partially offimage.
            (rClip0, rClip1, rClip2, rClip3) = (1.0, 1.0, 1.0, 1.0)
            if (roiClipped[0]>0): # Left
                rClip0 = 1.0 - (float(roiClipped[0])/float(r_outer-r_inner))#self.j_0))
            if (roiClipped[1]>0): # Top
                rClip1 = 1.0 - (float(roiClipped[1])/float(r_outer-r_inner))#self.i_0))
            if (roiClipped[2]>0): # Right
                rClip2 = 1.0 - (float(roiClipped[2])/float(r_outer-r_inner))#j_n-self.j_0))
            if (roiClipped[3]>0): # Bottom
                rClip3 = 1.0 - (float(roiClipped[3])/float(r_outer-r_inner))#i_n-self.i_0))
    
            self.rClip = np.min([rClip0, rClip1, rClip2, rClip3])

        
    def update_polarimage(self):
        if (self.imgRoiFgMasked is not None):
            theta_0a = self.angle_lo_i - self.angleBodypart_i
            theta_1a = self.angle_hi_i - self.angleBodypart_i
            
            radius_mid = (self.params['gui'][self.name]['radius_outer'] + self.params['gui'][self.name]['radius_inner'])/2.0
            dr         = (self.params['gui'][self.name]['radius_outer'] - self.params['gui'][self.name]['radius_inner'])/2.0

            self.imgRoiFgMaskedPolar  = self.polartransforms.transform_polar_elliptical(self.imgRoiFgMasked, 
                                                     self.i_0, 
                                                     self.j_0, 
                                                     raxial=radius_mid, 
                                                     rortho=radius_mid,
                                                     dradiusStrip=int(dr),
                                                     amplifyRho = 1.0,
                                                     rClip = self.rClip,
                                                     angleEllipse=self.angleBodypart_i,
                                                     theta_0 = min(theta_0a,theta_1a), 
                                                     theta_1 = max(theta_0a,theta_1a),
                                                     amplifyTheta = 1.0)
                
            # Find the y value where the black band should be cropped out (but leave at least one raster if image is all-black).
            sumY = np.sum(self.imgRoiFgMaskedPolar,1)
            iSumY = np.where(sumY==0)[0]
            if (len(iSumY)>0):
                iMinY = np.max([1,np.min(iSumY)])
            else:
                iMinY = self.imgRoiFgMaskedPolar.shape[0]

            self.imgRoiFgMaskedPolarCropped = self.imgRoiFgMaskedPolar[0:iMinY]
    
             
            if (self.bValidMask):
                if (self.wfnRoiMaskedPolarCropped is None) or (self.imgRoiFgMaskedPolarCropped.shape != self.wfnRoiMaskedPolarCropped.shape):
                    self.wfnRoiMaskedPolarCropped = self.create_wfn(self.imgRoiFgMaskedPolarCropped.shape)
                
                self.imgRoiFgMaskedPolarCroppedWindowed = cv2.multiply(self.imgRoiFgMaskedPolarCropped.astype(np.float32), self.wfnRoiMaskedPolarCropped)

            
        # Show the image.
        img = self.imgRoiFgMaskedPolarCroppedWindowed
        #img = self.imgRoiFgMaskedPolarCropped
        self.windowPolar.set_image(img)
        

    def update(self, dt, image, bInvertColor):
        MotionTrackedBodypart.update(self, dt, image, bInvertColor)
        self.update_polarimage()
        self.imgFinal = self.imgRoiFgMaskedPolarCroppedWindowed
        

        
    def draw(self, image):
        MotionTrackedBodypart.draw(self, image)
        self.windowPolar.show()
        
    
# end class MotionTrackedBodypartPolar




