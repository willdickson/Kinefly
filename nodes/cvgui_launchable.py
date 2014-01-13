#!/usr/bin/env python
from __future__ import division
import roslib; roslib.load_manifest('StrokelitudeROS')
import rospy

import sys

import time
import numpy as np
import cv
import cv2

import copy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from optparse import OptionParser

from StrokelitudeROS.srv import *
from StrokelitudeROS.msg import float32list as float32list_msg
from std_msgs.msg import *



def get_angle_from_points(pt1, pt2):
    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]
    return np.rad2deg(np.arctan2(y,x))



##########################################################################
class Wing(object):
    def __init__(self, side='right', params=None):
        self.side = side
        
        if side == 'right':
            self.color ='red'
            self.sense = 1
        elif side == 'left':
            self.color ='green'
            self.sense = -1
        else:
            rospy.logwarn('Wing side must be ''right'' or ''left''.')
            
            
        self.imgMask               = None
        self.imgTest               = None
        self.imgMasks              = None
        self.ravelMask             = None
        self.ravelAngleInBodyFrame = None

        self.bins                  = None
        self.intensities           = None
        self.binsValid             = None
        self.intensitiesValid      = None

        
        # Bodyframe angles have zero degrees point orthogonal to body axis: 
        # If body axis is north/south, then 0-deg is east for right wing, west for left wing.
        
        # Imageframe angles are oriented to the image.  0-deg is east, +-180 is west, -90 is north, +90 is south.
        
        
        self.angleTrailingEdge = None
        self.angleLeadingEdge  = None
        self.amp               = None
        self.msa               = None
        self.bFlying           = False
        
        # drawing parameters
        self.numeric_color_dict = { 'green'         : cv.Scalar(0,255,0,0), 
                                    'light_green'   : cv.Scalar(175,255,175,0),
                                    'red'           : cv.Scalar(0,0,255,0),
                                    'blue'          : cv.Scalar(255,0,0,0),
                                    'purple'        : cv.Scalar(255,0,255,0),
                                    'white'         : cv.Scalar(255,255,255,0),
                                    'black'         : cv.Scalar(0,0,0,0),
                                   }
        self.numeric_color      = self.numeric_color_dict[self.color]
        self.thickness_inner    = 1
        self.thickness_outer    = 2
        self.thickness_wing     = 2
        
        self.set_params(params)

        # services, for live histograms
        name                = 'wing_histogram_' + side
        self.service_histogram  = rospy.Service(name, float32list, self.serve_histogram)
        name                = 'wing_bins_' + side
        self.service_bins       = rospy.Service(name, float32list, self.serve_bins)
        name                = 'wing_edges_' + side
        self.service_edges       = rospy.Service(name, float32list, self.serve_edges)
        
        
        # publishers
        name                = 'strokelitude/wba/' + self.side + '/sum'
        self.pubWingSum           = rospy.Publisher(name, Float32)
        name                = 'strokelitude/wba/' + self.side + '/contrast'
        self.pubWingContrast      = rospy.Publisher(name, Float32)
        name                = 'strokelitude/wba/' + self.side + '/amplitude'
        self.pubWingAmp           = rospy.Publisher(name, Float32)
        name                = 'strokelitude/wba/' + self.side + '/mean_stroke_angle'
        self.pubWingMsa           = rospy.Publisher(name, Float32)
        name                = 'strokelitude/wba/' + self.side + '/leading_edge'
        self.pubWingLeading       = rospy.Publisher(name, Float32)
        name                = 'strokelitude/wba/' + self.side + '/trailing_edge'
        self.pubWingTrailing      = rospy.Publisher(name, Float32)
        
    
    # get_angles_i_from_b()
    # Return angle1 and angle2 oriented to the image rather than the fly.
    # * corrected for left/right full-circle angle, i.e. east is 0-deg, west is 270-deg.
    # * corrected for wrapping at delta>180.
    #
    def get_angles_i_from_b(self, angle_lo_b, angle_hi_b):
        angle_lo_i = self.transform_angle_i_from_b(angle_lo_b)
        angle_hi_i = self.transform_angle_i_from_b(angle_hi_b)
        
        if (angle_hi_i-angle_lo_i > 180):
            angle_hi_i -= 360
        if (angle_lo_i-angle_hi_i > 180):
            angle_lo_i -= 360
            
        return (angle_lo_i, angle_hi_i)
    
    
    # transform_angle_i_from_b()
    # Transform an angle from the fly frame to the camera image frame.
    #
    def transform_angle_i_from_b(self, angle_b):
        if self.side == 'right':
            angle_i  = int(self.get_bodyangle_i()) + angle_b
        else: # left
            angle_i  = int(self.get_bodyangle_i()) - angle_b + 180
             
        angle_i = (angle_i+180) % 360 - 180
        return angle_i
        

    # transform_angle_b_from_i()
    # Transform an angle from the camera image frame to the fly frame.
    #
    def transform_angle_b_from_i(self, angle_i):
        if self.side == 'right':
            angle_b  = angle_i - int(self.get_bodyangle_i())
        else:  
            angle_b  = int(self.get_bodyangle_i()) - angle_i + 180 

        angle_b = (angle_b+180) % 360 - 180
        return angle_b
        

    def set_params(self, params):
        self.params  = params
        self.ptHinge = np.array([self.params[self.side]['hinge']['x'], self.params[self.side]['hinge']['y']])
        self.bins    = np.linspace(-180, 180, self.params['nbins'])
        
        
        
    def get_bodyangle_i(self):
        return get_angle_from_points(np.array([self.params['left']['hinge']['x'], self.params['left']['hinge']['y']]), 
                                     np.array([self.params['right']['hinge']['x'], self.params['right']['hinge']['y']]))
        
                
    def get_hinge(self):
        return np.array([self.params[self.side]['hinge']['x'], self.params[self.side]['hinge']['y']]) 
        
        
    # draw()
    # Draw the wing envelope, and leading and trailing edges, onto the given image.
    #
    def draw(self, image):
        (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi'])

        # inner circle
        cv2.ellipse(image, 
                    tuple(self.ptHinge),  
                    (self.params[self.side]['radius_inner'], self.params[self.side]['radius_inner']),
                    0,
                    angle_lo_i,
                    angle_hi_i,
                    color=self.numeric_color,
                    thickness=self.thickness_inner,
                    )
        
        # outer circle         
        cv2.ellipse(image, 
                    tuple(self.ptHinge),  
                    (self.params[self.side]['radius_outer'], self.params[self.side]['radius_outer']),
                    0,
                    angle_lo_i,
                    angle_hi_i,
                    color=self.numeric_color,
                    thickness=self.thickness_outer,
                    )
        
        
        # wing leading and trailing edges
        if self.angleTrailingEdge is not None:
            (angle_leading, angle_trailing) = self.get_angles_i_from_b(self.angleLeadingEdge, self.angleTrailingEdge)

            x = int(self.ptHinge[0] + self.params[self.side]['radius_outer'] * np.cos( np.deg2rad(angle_trailing) ))
            y = int(self.ptHinge[1] + self.params[self.side]['radius_outer'] * np.sin( np.deg2rad(angle_trailing) ))
            cv2.line(image, tuple(self.ptHinge), (x,y), self.numeric_color, self.thickness_wing)
            
            x = int(self.ptHinge[0] + self.params[self.side]['radius_outer'] * np.cos( np.deg2rad(angle_leading) ))
            y = int(self.ptHinge[1] + self.params[self.side]['radius_outer'] * np.sin( np.deg2rad(angle_leading) ))
            cv2.line(image, tuple(self.ptHinge), (x,y), self.numeric_color, self.thickness_wing)

                        
    # create_angle_mask1()
    # Create an image where each pixel value is the angle from the hinge.                    
    # The slow way.  But it give nice smooth pixels.                        
    def create_angle_mask1(self, shape):
        # Calculate the angle at each pixel.
        self.imgAngleInBodyFrame = np.zeros(shape)
        for y in range(shape[0]):
            for x in range(shape[1]):
                angle_i = get_angle_from_points(self.ptHinge, (x,y))
                angle_b  = self.transform_angle_b_from_i(angle_i)

                self.imgAngleInBodyFrame[y,x] = angle_b
        
        self.imgAngleInBodyFrame = (self.imgAngleInBodyFrame+180) % 360 - 180
        self.ravelAngleInBodyFrame = np.ravel(self.imgAngleInBodyFrame)
#        rospy.logwarn('%s: angles on range [%s, %s]' % (self.side, self.imgAngleInBodyFrame.min(), self.imgAngleInBodyFrame.max()))
                   

    # create_angle_mask2()
    # Create an image where each pixel value is the angle from the hinge.                    
    # The fast way.  But the ellipse drawing has glitches when the ellipse is partially offscreen left or right.
    def create_angle_mask2(self, shape):
        (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi'])
        degPerBin = (angle_hi_i-angle_lo_i)/self.params['nbins']
        
        angle_i = angle_lo_i
        self.imgAngleInBodyFrame = np.zeros(shape)

        # Draw a wedge for each bin.
        for k in range(self.params['nbins']):
            cv2.ellipse(self.imgAngleInBodyFrame,
                        tuple(self.ptHinge),
                        (self.params[self.side]['radius_outer'], self.params[self.side]['radius_outer']),
                        0,
                        angle_i,
                        angle_i+degPerBin,
                        float(self.transform_angle_b_from_i(angle_i)), 
                        cv.CV_FILLED)
            cv2.ellipse(self.imgAngleInBodyFrame,
                        tuple(self.ptHinge),
                        (self.params[self.side]['radius_inner'], self.params[self.side]['radius_inner']),
                        0,
                        angle_i-10,
                        angle_i+degPerBin+10,
                        0, 
                        cv.CV_FILLED)

            angle_i += degPerBin
            
        self.ravelAngleInBodyFrame = np.ravel(self.imgAngleInBodyFrame)

        
    def create_stroke_mask(self, shape):
        # Create the wing mask.
        self.imgMask = np.zeros(shape)
        (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi'])

        cv2.ellipse(self.imgMask,
                    tuple(self.ptHinge),
                    (self.params[self.side]['radius_outer'], self.params[self.side]['radius_outer']),
                    0,
                    angle_lo_i,
                    angle_hi_i,
                    1, 
                    cv.CV_FILLED)
        cv2.ellipse(self.imgMask,
                    tuple(self.ptHinge),
                    (self.params[self.side]['radius_inner'], self.params[self.side]['radius_inner']),
                    0,
                    angle_lo_i,
                    angle_hi_i,
                    0, 
                    cv.CV_FILLED)
        
        self.ravelMask = np.ravel(self.imgMask)
        
        
    # assign_pixels_to_bins()
    # Create two dictionaries of bins, one containing the pixel indices, and the other containing the mean pixel values.
    #
    def assign_pixels_to_bins(self):
        # Create empty bins.
        self.pixels = [[] for i in range(len(self.bins))]
        self.intensities = np.zeros(len(self.bins))

        # Put each pixel into an appropriate bin.            
        for iPixel, angle in enumerate(self.ravelAngleInBodyFrame):
            if self.ravelMask[iPixel]:
                iBest = np.argmin(np.abs(self.bins - angle))
                self.pixels[iBest].append(iPixel)
                
        self.imgTest = self.imgAngleInBodyFrame #- np.min(self.imgAngleInBodyFrame)
        #self.imgTest /= np.max(self.imgAngleInBodyFrame)
                
    '''   
    def publish_wing_sum(self):
        if self.intensities is not None:
            s = np.sum(self.intensities[self.angle_indices_ok])
            if s > self.params['flight_threshold']:
                self.bFlying = True
            else:
                self.bFlying = False
            self.pubWingSum.publish(s)
    '''
         
    # updateIntensities()
    # The "histogram" is a dictionary of angles and their corresponding mean pixel intensity in the image.
    #            
    def updateIntensities(self, image):
        if (self.ravelMask is not None) and (self.intensities is not None):
            ravelImageMasked = np.ravel(image).astype(float)/255. * self.ravelMask.astype(float)
            for iBin in range(len(self.bins)):
                iPixels = self.pixels[iBin]
                if len(iPixels) > 0:
                    self.intensities[iBin] = np.sum(ravelImageMasked[iPixels]) / float(len(iPixels))
                else:
                    self.intensities[iBin] = 0.0
             
            angle_lo_b = min([self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi']])
            angle_hi_b = max([self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi']])
 
            iValid                = np.where((angle_lo_b < self.bins) * (self.bins < angle_hi_b))[0]
            self.binsValid        = self.bins[iValid]
            self.intensitiesValid = self.intensities[iValid]

                        
    # calc_wing_edges()
    # Calculate the leading and trailing edge angles,
    # and the amplitude & mean stroke angle.
    #                       
    def calc_wing_edges(self):                
        if self.intensities is not None:
            self.angleLeadingEdge = None
            self.angleTrailingEdge = None
            self.amp = None
            self.msa = None
            
            if self.bFlying:
                if (len(self.intensitiesValid)>1):
                    (angle, threshold) = self.get_threshold()
                    iEdges = np.where(self.intensitiesValid < threshold)[0]
                    
#                     if (self.side=='right'):
#                         rospy.logwarn('threshold = %s, %s' % (angle, threshold))
                    
                    if (len(iEdges)>0):
                        iLeading = iEdges[0]
                        iTrailing = iEdges[-1]
                
                        self.angleLeadingEdge = float(self.binsValid[iLeading])
                        self.angleTrailingEdge = float(self.binsValid[iTrailing])
                
                        self.amp = np.abs(self.angleLeadingEdge - self.angleTrailingEdge)
                        self.msa = np.mean([self.angleLeadingEdge, self.angleTrailingEdge])
            
            else: # not flying
                self.angleLeadingEdge = -180.
                self.angleTrailingEdge = -180.
                self.amp = 0
                self.msa = 0


    # get_threshold()
    # Calculate and return the best pixel intensity threshold to use for wing angle detection.
    #                    
    def get_threshold(self):
        iMin = np.argmin(np.diff(self.intensitiesValid))
        threshold = self.intensitiesValid[iMin]
        angle = self.binsValid[iMin]
        
        return (angle, threshold)
        

    def publish_wing_contrast(self):
        if self.intensities is not None:
            max_contrast = np.abs( np.max(self.intensities) - np.min(self.intensities) )
            if max_contrast > self.params['flight_threshold']:
                self.bFlying = True
            else:
                self.bFlying = False
            self.pubWingContrast.publish(max_contrast)
                

    def publish_wing_edges(self):
        if self.intensities is not None:
            if (self.angleLeadingEdge is not None):                
                self.pubWingMsa.publish(self.msa)
                self.pubWingAmp.publish(self.amp)
                self.pubWingLeading.publish(self.angleLeadingEdge)
                self.pubWingTrailing.publish(self.angleTrailingEdge)
                
                    
    def serve_bins(self, request):
        if self.binsValid is not None:
            #return float32listResponse(self.bins)
            return float32listResponse(self.binsValid[:-1])
            
    def serve_histogram(self, request):
        if self.intensitiesValid is not None:
            #return float32listResponse(self.intensities)
            return float32listResponse(np.diff(self.intensitiesValid))
            
    def serve_edges(self, request):
        if self.angleTrailingEdge is not None:            
            return float32listResponse([self.angleTrailingEdge, self.angleLeadingEdge])
            
            
            
###############################################################
class ImageDisplay:

    def __init__(self):
        # initialize display
        self.display_name = "Display"
        cv.NamedWindow(self.display_name,1)
        self.cvbridge = CvBridge()
        
        # Get parameters from parameter server
        self.params = rospy.get_param('strokelitude')
        defaults = {'nbins': 50,
                    'black_on_white':True,
                    'flight_threshold':0.2,
                    'right':{'hinge':{'x':300,
                                      'y':100},
                             'radius_outer':30,
                             'radius_inner':10,
                             'angle_hi':45, 
                             'angle_lo':-45
                             },

                    'left':{'hinge':{'x':100,
                                     'y':100},
                            'radius_outer':30,
                            'radius_inner':10,
                            'angle_hi':45, 
                            'angle_lo':-45
                            }
                    }
        self.SetDictWithPreserve(self.params, defaults)
        
        
        # initialize wings and body
        self.wing_r                 = Wing('right', self.params)
        self.wing_l                 = Wing('left', self.params)
        self.bInitWings = False
        self.bSettingControls= False

        self.controlselected = None
        self.controlpts = {}
        self.update_control_points()

        # initialize
        rospy.init_node('strokelitude', anonymous=True)

        # publishers
        self.pubWingLeftMinusRight = rospy.Publisher('strokelitude/wba/LeftMinusRight', Float32)
        self.pubWingLeftPlusRight  = rospy.Publisher('strokelitude/wba/LeftPlusRight', Float32)
        self.pubFlightStatus       = rospy.Publisher('strokelitude/wba/flight_status', Float32)
        self.pubMask               = rospy.Publisher('strokelitude/wba/image_mask', Image)
        self.pubTestR              = rospy.Publisher('strokelitude/wba/image_test_r', Image)
        self.pubTestL              = rospy.Publisher('strokelitude/wba/image_test_l', Image)

        # subscribe to images        
        self.subImageRaw           = rospy.Subscriber('/camera/image_raw',Image,self.image_callback)

        # user callbacks
        cv.SetMouseCallback(self.display_name, self.onMouse, param=None)
        
        
    # SetDict(self, dTarget, dSource, bPreserve)
    # Takes a target dictionary, and enters values from the source dictionary, overwriting or not, as asked.
    # For example,
    #    dT={'a':1, 'b':2}
    #    dS={'a':0, 'c':0}
    #    Set(dT, dS, True)
    #    dT is {'a':1, 'b':2, 'c':0}
    #
    #    dT={'a':1, 'b':2}
    #    dS={'a':0, 'c':0}
    #    Set(dT, dS, False)
    #    dT is {'a':0, 'b':2, 'c':0}
    #
    def SetDict(self, dTarget, dSource, bPreserve):
        for k,v in dSource.iteritems():
            bKeyExists = (k in dTarget)
            if (not bKeyExists) and type(v)==type({}):
                dTarget[k] = {}
            if ((not bKeyExists) or not bPreserve) and (type(v)!=type({})):
                dTarget[k] = v
                    
            if type(v)==type({}):
                self.SetDict(dTarget[k], v, bPreserve)
    
    
    def SetDictWithPreserve(self, dTarget, dSource):
        self.SetDict(dTarget, dSource, True)
    
    def SetDictWithOverwrite(self, dTarget, dSource):
        self.SetDict(dTarget, dSource, False)


    def create_masks(self):
        rospy.logwarn('Creating wing masks...')
        self.wing_r.create_stroke_mask(self.shapeImage)
        self.wing_r.create_angle_mask1(self.shapeImage)
        self.wing_r.assign_pixels_to_bins()
        self.wing_l.create_stroke_mask(self.shapeImage)
        self.wing_l.create_angle_mask1(self.shapeImage)
        self.wing_l.assign_pixels_to_bins()
        rospy.logwarn('Creating wing masks... done.')

        
    def image_callback(self, rosimage):
        # Receive an image:
        try:
            imgCamera = np.uint8(cv.GetMat(self.cvbridge.imgmsg_to_cv(rosimage, 'passthrough')))
        except CvBridgeError, e:
            rospy.logwarn ('Exception converting background image from ROS to opencv:  %s' % e)
            imgCamera = None
            
        if (imgCamera is not None):
            self.shapeImage = imgCamera.shape # (height,width)
            imgOutput = cv2.cvtColor(imgCamera, cv.CV_GRAY2RGB)

            if (not self.bInitWings):
                self.create_masks()
                self.bInitWings = True
                                
            left_pixel   = 10
            bottom_pixel = imgOutput.shape[0]-10 
            right_pixel  = imgOutput.shape[1]-10
                
            # draw line from one hinge to the other.
            if (self.wing_r.get_hinge() is not None) and (self.wing_l.get_hinge() is not None):
                cv2.line(imgOutput, tuple(self.wing_r.get_hinge()), tuple(self.wing_l.get_hinge()), cv.Scalar(255,0,0,0), 2)
            
            # draw wings
            if self.wing_r.get_hinge() is not None:
                self.wing_r.draw(imgOutput)
            if self.wing_l.get_hinge() is not None:
                self.wing_l.draw(imgOutput)
            
            if not self.bSettingControls:
                # calculate wing beat analyzer stats
                self.wing_r.updateIntensities(imgCamera)
                self.wing_r.calc_wing_edges()
                self.wing_r.publish_wing_contrast()
                self.wing_r.publish_wing_edges()
                
                self.wing_l.updateIntensities(imgCamera)
                self.wing_l.calc_wing_edges()
                self.wing_l.publish_wing_contrast()
                self.wing_l.publish_wing_edges()
                
                # publish difference in WBA
                if self.wing_l.amp is not None and self.wing_r.amp is not None:
                    LeftMinusRight = self.wing_l.amp - self.wing_r.amp
                    self.pubWingLeftMinusRight.publish(LeftMinusRight)
                    s = 'L-R WBA: ' + str(LeftMinusRight)
                    cv2.putText(imgOutput, s, (right_pixel-100, bottom_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cv.Scalar(255,50,50,0) )
                    
                # publish sum of WBA
                if self.wing_l.amp is not None and self.wing_r.amp is not None:
                    LeftPlusRight = self.wing_l.amp + self.wing_r.amp
                    self.pubWingLeftPlusRight.publish(LeftPlusRight)
                    s = 'L+R WBA: ' + str(LeftPlusRight)
                    cv2.putText(imgOutput, s, (right_pixel-200, bottom_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cv.Scalar(255,50,50,0) )
                    
                # publish flight status
                if self.wing_l.bFlying and self.wing_r.bFlying:
                    s = 'FLIGHT'
                    self.pubFlightStatus.publish(1)
                else:
                    s = 'no flight'
                    self.pubFlightStatus.publish(0)
                cv2.putText(imgOutput, s, (right_pixel-300, bottom_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cv.Scalar(255,50,255,0) )
                
                if self.wing_r.amp is not None:
                    s = 'WBA R: ' + str(self.wing_r.amp)
                    cv2.putText(imgOutput, s, (left_pixel,bottom_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.wing_r.numeric_color)
                if self.wing_l.amp is not None:
                    s = 'WBA L: ' + str(self.wing_l.amp)
                    cv2.putText(imgOutput, s, (left_pixel+100, bottom_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.wing_l.numeric_color)
                
    
                if (self.wing_r.imgMask is not None) and (self.wing_l.imgMask is not None):
                    self.pubMask.publish(self.cvbridge.cv_to_imgmsg(cv.fromarray(self.wing_r.imgMask + self.wing_l.imgMask), 'passthrough'))
    
                if (self.wing_r.imgTest is not None):
                    self.pubTestR.publish(self.cvbridge.cv_to_imgmsg(cv.fromarray(self.wing_r.imgTest), 'passthrough'))
                if (self.wing_l.imgTest is not None):
                    self.pubTestL.publish(self.cvbridge.cv_to_imgmsg(cv.fromarray(self.wing_l.imgTest), 'passthrough'))

            
            # Draw the control points.
            for controlname,controlpt in self.controlpts.iteritems():
                cv2.circle(imgOutput, tuple(controlpt),  2, cv.Scalar(255,255,255,0), 2)         


            # display image
            cv2.imshow("Display", imgOutput)
            cv2.waitKey(1)


    # hit_test()
    # Get the name of the nearest control point to the mouse point.
    # ptMouse    = [x,y]
    #
    def hit_test(self, ptMouse):
        names = self.controlpts.keys()
        pts = np.array(self.controlpts.values())

        dx = np.subtract.outer(ptMouse[0], pts[:,0])
        dy = np.subtract.outer(ptMouse[1], pts[:,1])
        d = np.hypot(dx, dy)
        controlname = names[np.argmin(d)]
        
        return controlname
        
        
    # update_control_points()
    # Update the dictionary of control point names and locations.
    # Compute the various control points.
    #
    def update_control_points (self):
        # Hinge Points.
        self.controlpts['hinge_l'] = self.wing_l.get_hinge()
        self.controlpts['hinge_r'] = self.wing_r.get_hinge()
        
        # Left High & Low Angles.
        (angle_lo_i, angle_hi_i) = self.wing_l.get_angles_i_from_b(self.wing_l.params['left']['angle_lo'], self.wing_l.params['left']['angle_hi'])
        self.controlpts['hi_l'] = (self.wing_l.get_hinge() + self.wing_l.params['left']['radius_outer'] * np.array([np.cos(np.deg2rad(angle_hi_i)), 
                                                                                                                    np.sin(np.deg2rad(angle_hi_i))])).astype(int)
        self.controlpts['lo_l'] = (self.wing_l.get_hinge() + self.wing_l.params['left']['radius_outer'] * np.array([np.cos(np.deg2rad(angle_lo_i)), 
                                                                                                                    np.sin(np.deg2rad(angle_lo_i))])).astype(int)

        # Left Inner Radius.
        angle = (angle_hi_i - angle_lo_i)/2 + angle_lo_i 
        self.controlpts['inner_l'] = (self.wing_l.get_hinge() + self.wing_l.params['left']['radius_inner'] * np.array([np.cos(np.deg2rad(angle)), 
                                                                                                                       np.sin(np.deg2rad(angle))])).astype(int)

        # Right High & Low Angles.
        (angle_lo_i, angle_hi_i) = self.wing_r.get_angles_i_from_b(self.wing_r.params['right']['angle_lo'], self.wing_r.params['right']['angle_hi'])
        self.controlpts['hi_r'] = (self.wing_r.get_hinge() + self.wing_r.params['right']['radius_outer'] * np.array([np.cos(np.deg2rad(angle_hi_i)), 
                                                                                                                     np.sin(np.deg2rad(angle_hi_i))])).astype(int)
        self.controlpts['lo_r'] = (self.wing_r.get_hinge() + self.wing_r.params['right']['radius_outer'] * np.array([np.cos(np.deg2rad(angle_lo_i)), 
                                                                                                                     np.sin(np.deg2rad(angle_lo_i))])).astype(int)

        # Right Inner Radius.
        angle = (angle_hi_i - angle_lo_i)/2 + angle_lo_i 
        self.controlpts['inner_r'] = (self.wing_r.get_hinge() + self.wing_r.params['right']['radius_inner'] * np.array([np.cos(np.deg2rad(angle)), 
                                                                                                                        np.sin(np.deg2rad(angle))])).astype(int)

    
    def clip(self, x, lo, hi):
        return max(min(x,hi),lo)
    

    def onMouse(self, event, x, y, flags, param):
        ptMouse = np.array([x, y])

        # Keep track of the mouse button state.        
        if (event==cv.CV_EVENT_LBUTTONDOWN):
            self.controlselected = self.hit_test(ptMouse)
        if (event==cv.CV_EVENT_LBUTTONUP):
            self.controlselected = None
            self.create_masks()
            rospy.set_param('strokelitude', self.params)



        # When the left button is down, adjust the control points.
        if (flags & cv.CV_EVENT_FLAG_LBUTTON):
            self.bSettingControls = True
            
            # Set the new point.             
            if (self.controlselected=='hinge_l'): # Left hinge point.
                self.params['left']['hinge']['x'] = int(self.clip(ptMouse[0], 0+self.params['left']['radius_outer'], self.shapeImage[1]-self.params['left']['radius_outer'])) # Keep the mask onscreen left & right.
                self.params['left']['hinge']['y'] = int(ptMouse[1])

            elif (self.controlselected=='hinge_r'): # Right hinge point.
                self.params['right']['hinge']['x'] = int(self.clip(ptMouse[0], 0+self.params['right']['radius_outer'], self.shapeImage[1]-self.params['right']['radius_outer'])) # Keep the mask onscreen left & right.
                self.params['right']['hinge']['y'] = int(ptMouse[1])
        
            elif (self.controlselected=='hi_l'): # Left high angle.
                pt = ptMouse - self.wing_l.get_hinge()
                self.params['left']['angle_hi'] = int(self.wing_l.transform_angle_b_from_i(np.rad2deg(np.arctan2(pt[1], pt[0]))))
                #radius = int(max(self.wing_l.params['left']['radius_inner']+2, int(np.linalg.norm(self.wing_l.get_hinge() - ptMouse)))) # Outer radius > inner radius.
                self.params['left']['radius_outer'] = int(self.clip(np.linalg.norm(self.wing_l.get_hinge() - ptMouse), 
                                                                    self.wing_l.params['left']['radius_inner']+2,               # Outer radius > inner radius.
                                                                    min(self.shapeImage[1]-self.params['left']['hinge']['x'],   # Outer radius < right edge
                                                                        0+self.params['left']['hinge']['x'])))                  # Outer radius > left edge
                  
            elif (self.controlselected=='hi_r'): # Right high angle.
                pt = ptMouse - self.wing_r.get_hinge()
                self.params['right']['angle_hi'] = int(self.wing_r.transform_angle_b_from_i(np.rad2deg(np.arctan2(pt[1], pt[0]))))
                #self.params['right']['radius_outer'] = int(max(self.wing_r.params['right']['radius_inner']+2, int(np.linalg.norm(self.wing_r.get_hinge() - ptMouse))))
                self.params['right']['radius_outer'] = int(self.clip(np.linalg.norm(self.wing_r.get_hinge() - ptMouse), 
                                                                    self.wing_l.params['right']['radius_inner']+2,               # Outer radius > inner radius.
                                                                    min(self.shapeImage[1]-self.params['right']['hinge']['x'],   # Outer radius < right edge
                                                                        0+self.params['right']['hinge']['x'])))                  # Outer radius > left edge
                  
            elif (self.controlselected=='lo_l'): # Left low angle.
                pt = ptMouse - self.wing_l.get_hinge()
                self.params['left']['angle_lo'] = int(self.wing_l.transform_angle_b_from_i(np.rad2deg(np.arctan2(pt[1], pt[0]))))
                #self.params['left']['radius_outer'] = int(max(self.wing_l.params['left']['radius_inner']+2, int(np.linalg.norm(self.wing_l.get_hinge() - ptMouse))))
                self.params['left']['radius_outer'] = int(self.clip(np.linalg.norm(self.wing_l.get_hinge() - ptMouse), 
                                                                    self.wing_l.params['left']['radius_inner']+2,               # Outer radius > inner radius.
                                                                    min(self.shapeImage[1]-self.params['left']['hinge']['x'],   # Outer radius < right edge
                                                                        0+self.params['left']['hinge']['x'])))                  # Outer radius > left edge
                  
            elif (self.controlselected=='lo_r'): # Right low angle.
                pt = ptMouse - self.wing_r.get_hinge()
                self.params['right']['angle_lo'] = int(self.wing_r.transform_angle_b_from_i(np.rad2deg(np.arctan2(pt[1], pt[0]))))
                #self.params['right']['radius_outer'] = int(max(self.wing_r.params['right']['radius_inner']+2, int(np.linalg.norm(self.wing_r.get_hinge() - ptMouse))))
                self.params['right']['radius_outer'] = int(self.clip(np.linalg.norm(self.wing_r.get_hinge() - ptMouse), 
                                                                    self.wing_l.params['right']['radius_inner']+2,               # Outer radius > inner radius.
                                                                    min(self.shapeImage[1]-self.params['right']['hinge']['x'],   # Outer radius < right edge
                                                                        0+self.params['right']['hinge']['x'])))                  # Outer radius > left edge
                  
            elif (self.controlselected=='inner_l'): # Left inner radius.
                self.params['left']['radius_inner'] = int(min(int(np.linalg.norm(self.wing_l.get_hinge() - ptMouse)), self.wing_l.params['left']['radius_outer']-2))
                
            elif (self.controlselected=='inner_r'): # Right inner radius.
                self.params['right']['radius_inner'] = int(min(int(np.linalg.norm(self.wing_r.get_hinge() - ptMouse)), self.wing_r.params['right']['radius_outer']-2))
                
            self.wing_l.set_params(self.params)
            self.wing_r.set_params(self.params)
            self.update_control_points()
        else:
            self.bSettingControls = False
            
            
                
    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    #parser = OptionParser()
    #parser.add_option("--color", action="store_true", dest="color", default=True)
    #(options, args) = parser.parse_args()


    rospy.logwarn('')
    rospy.logwarn('')
    rospy.logwarn('**************************************************************************')
    rospy.logwarn('*                                                                        *')  
    rospy.logwarn('*    StrokelitudeROS: Camera based wingbeat analyzer software for ROS    *')
    rospy.logwarn('*        by Floris van Breugel, Steve Safarik, (c) 2013                  *')
    rospy.logwarn('*                                                                        *')
    rospy.logwarn('*    Left click+drag to move any control points.                         *')
    rospy.logwarn('*                                                                        *')  
    rospy.logwarn('*                                                                        *') 
    rospy.logwarn('**************************************************************************')
    rospy.logwarn('')
    rospy.logwarn('')

    im = ImageDisplay()
    im.run()
