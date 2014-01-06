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


# remap +/- 180 to 0, and +/- 0 to +/- 180
def flip_angle(angle):
    if angle > 0:
        return angle -180
    else:
        return angle +180
#     return angle


##########################################################################
class Wing(object):
    def __init__(self, side='right', config=None):
        self.side = side
        
        if side == 'right':
            self.color ='red'
            self.sign  = 1
        elif side == 'left':
            self.color ='green'
            self.sign  = -1
        else:
            rospy.logwarn('Wing side must be ''right'' or ''left''.')
            
            
        self.nbins              = 50
        self.degPerBin          = 361/float(self.nbins)
        self.bins               = np.linspace(-180, 180, self.nbins)
        self.imgMask            = None
        self.mask               = None
        self.intensityByAngle             = None
        self.ravelAngleInBodyFrame        = None
        self.intensityByAngle_angles      = None
        self.intensityByAngle_intensities = None
        
        # Bodyframe angles define the maximum wing extents, where zero degrees points orthogonal to body axis: 
        # If body axis is north/south, then 0-deg is east for right wing, west for left wing.
        self.angle1_bodyframe   = 0
        self.angle2_bodyframe   = 0
        
        self.extract_parameters(config)
        
        self.angleTrailingEdge= None
        self.angleLeadingEdge = None
        self.amp                = None
        self.msa                = None
        self.bFlying            = False
        
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
        self.thickness_outer    = 3
        
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
        
    
    # get_angles_imageframe_from_bodyframe()
    # Return angle1 and angle2 oriented to the image rather than the fly.
    # * corrected for left/right full-circle angle, i.e. east is 0-deg, west is 270-deg.
    # * corrected for wrapping at delta>180.
    #
    def get_angles_imageframe_from_bodyframe(self, angle1_bodyframe, angle2_bodyframe):
        if self.side == 'left':
            angle1A_bodyframe = -angle2_bodyframe - 180
            angle2A_bodyframe = -angle1_bodyframe - 180
        else:
            angle1A_bodyframe = angle1_bodyframe
            angle2A_bodyframe = angle2_bodyframe

        angle1_imageframe = self.transform_angle_imageframe_from_bodyframe(angle1A_bodyframe)
        angle2_imageframe = self.transform_angle_imageframe_from_bodyframe(angle2A_bodyframe)
        
        if (angle2_imageframe-angle1_imageframe > 180):
            angle2_imageframe -= 360
        if (angle1_imageframe-angle2_imageframe > 180):
            angle1_imageframe -= 360
            
        return (angle1_imageframe, angle2_imageframe)
    
    
    # transform_angle_imageframe_from_bodyframe()
    # Transform an angle from the fly frame to the camera image frame.
    #
    def transform_angle_imageframe_from_bodyframe(self, angle_bodyframe):
        angle_imageframe  = angle_bodyframe + self.bodyangle 

        return angle_imageframe
        

    # transform_angle_bodyframe_from_imageframe()
    # Transform an angle from the camera image frame to the fly frame.
    #
    def transform_angle_bodyframe_from_imageframe(self, angle_imageframe):
        angle_bodyframe  = angle_imageframe - self.bodyangle 

        return angle_bodyframe
        

    # set_angles_bodyframe()
    # Set the angles into the member vars.  Enforces the range [-180,+180].
    #
    def set_angles_bodyframe(self, angle1_bodyframe, angle2_bodyframe):
        self.angle1_bodyframe = (angle1_bodyframe+180) % 360 - 180
        self.angle2_bodyframe = (angle2_bodyframe+180) % 360 - 180
        
        
    # extract parameters from parameter server, given "strokelitude" parameter
    def extract_parameters(self, config):
        if config is not None:
            self.ptHinge        = (config['wing'][self.side]['hinge']['x'], config['wing'][self.side]['hinge']['y'])
            
            self.radius_outer   = config['wing']['radius_outer']
            self.radius_inner   = config['wing']['radius_inner']
            self.bodyangle      = config['bodyangle']
            self.threshold      = config['wing'][self.side]['threshold']
            self.flight_threshold=config['wing'][self.side]['flight_threshold']

            self.set_angles_bodyframe(config['wing']['angle1'], config['wing']['angle2'])
            
            
            if config['wing']['nbins'] != self.nbins:
                self.nbins          = config['wing']['nbins']
                self.degPerBin      = 361/float(self.nbins)
                self.bins           = np.linspace(-180, 180, self.nbins)
                if self.ravelAngleInBodyFrame is not None:
                    self.assign_pixels_to_bins()
                
                
    # draw()
    # Draw the wing envelope, and leading and trailing edges, onto the given image.
    #
    def draw(self, image):
        (angle1_imageframe, angle2_imageframe) = self.get_angles_imageframe_from_bodyframe(self.angle1_bodyframe, self.angle2_bodyframe)

        # inner circle
        cv2.ellipse(image, 
                    self.ptHinge,  
                    (self.radius_inner, self.radius_inner),
                    0,
                    angle1_imageframe,
                    angle2_imageframe,
                    color=self.numeric_color,
                    thickness=self.thickness_inner,
                    )
        
        # outer circle         
        cv2.ellipse(image, 
                    self.ptHinge,  
                    (self.radius_outer, self.radius_outer),
                    0,
                    angle1_imageframe,
                    angle2_imageframe,
                    color=self.numeric_color,
                    thickness=self.thickness_outer,
                    )
        
        
        # wing leading and trailing edges
        if self.angleTrailingEdge is not None:
            (angle_leading, angle_trailing) = self.get_angles_imageframe_from_bodyframe(self.angleLeadingEdge, self.angleTrailingEdge)

#             if (self.side=='left'):
#                 rospy.logwarn('L,T: %s,   l,t: %s' % ((self.angleLeadingEdge, self.angleTrailingEdge), (angle_leading, angle_trailing)))
            p1 = int(self.ptHinge[0] + np.cos( np.deg2rad(angle_trailing) )*self.radius_outer)
            p2 = int(self.ptHinge[1] + np.sin( np.deg2rad(angle_trailing) )*self.radius_outer)
            cv2.line(image, self.ptHinge, (p1,p2), self.numeric_color, 2)
            
            p1 = int(self.ptHinge[0] + np.cos( np.deg2rad(angle_leading) )*self.radius_outer)
            p2 = int(self.ptHinge[1] + np.sin( np.deg2rad(angle_leading) )*self.radius_outer)
            cv2.line(image, self.ptHinge, (p1,p2), self.numeric_color, 2)

                        
    def calc_mask(self, shape):
        # Calculate the angle at each pixel.
        self.imgAngleInBodyFrame = np.zeros(shape)
        for y in range(shape[0]):
            for x in range(shape[1]):
                angle_imageframe = get_angle_from_points(self.ptHinge, (x,y))
                angle_bodyframe  = self.transform_angle_bodyframe_from_imageframe(angle_imageframe)

                if self.side == 'left':
                    angle_bodyframe = -angle_bodyframe - 180

                self.imgAngleInBodyFrame[y,x] = angle_bodyframe
        
        self.imgAngleInBodyFrame = (self.imgAngleInBodyFrame+180) % 360 - 180
        self.ravelAngleInBodyFrame = np.ravel(self.imgAngleInBodyFrame)
#        rospy.logwarn('%s: angles on range [%s, %s]' % (self.side, self.imgAngleInBodyFrame.min(), self.imgAngleInBodyFrame.max()))
                     
        # Create the wing mask.
        self.imgMask = np.zeros(shape)
        (angle1_imageframe, angle2_imageframe) = self.get_angles_imageframe_from_bodyframe(self.angle1_bodyframe, self.angle2_bodyframe)

        cv2.ellipse(self.imgMask,
                    self.ptHinge,
                    (self.radius_outer, self.radius_outer),
                    0,
                    angle1_imageframe,
                    angle2_imageframe,
                    1, 
                    cv.CV_FILLED)
        cv2.ellipse(self.imgMask,
                    self.ptHinge,
                    (self.radius_inner, self.radius_inner),
                    0,
                    angle1_imageframe,
                    angle2_imageframe,
                    0, 
                    cv.CV_FILLED)
        
        self.mask = np.ravel(self.imgMask)
        
        self.assign_pixels_to_bins()
        
        
    # assign_pixels_to_bins()
    # Create two dictionaries of bins, one containing the pixel indices, and the other containing the mean pixel values.
    #
    def assign_pixels_to_bins(self):
        rospy.logwarn ('Calculating wing mask: %s' % self.side)
        rospy.logwarn ('This could take a minute...')
        rospy.logwarn ('')
        
        # Create empty dicts of bins with each bin empty.
        self.bin_indices = {}
        self.intensityByAngle = {}
        for bin in self.bins:
            self.bin_indices.setdefault(bin, [])    # List of pixel indices in the bin.
            self.intensityByAngle.setdefault(bin, 0.0)    # Mean pixel value in the bin.

        # Put each pixel into an appropriate bin.            
        for k, angle in enumerate(self.ravelAngleInBodyFrame):
            if self.mask[k]:
                bind = np.argmin(np.abs(self.bins - angle))
                best_bin = self.bins[bind]
                self.bin_indices[best_bin].append(k)
                
                
    '''   
    def publish_wing_sum(self):
        if self.intensityByAngle is not None:
            s = np.sum(self.intensityByAngle_intensities[self.angle_indices_ok])
            if s > self.flight_threshold:
                self.bFlying = True
            else:
                self.bFlying = False
            self.pubWingSum.publish(s)
    '''
         
    # updateIntensityByAngle()
    # The "histogram" is a dictionary of angles and their corresponding mean pixel intensity in the image.
    #            
    def updateIntensityByAngle(self, image):
        if (self.mask is not None):
            image_raveled = np.ravel(image)/255. * self.mask
            for angle, indices in self.bin_indices.items():
                if len(indices) > 0:
                    self.intensityByAngle[angle] = np.sum(image_raveled[indices]) / float(len(indices))
                else:
                    self.intensityByAngle[angle] = 0.0
            
            # rearranging histogram
            angles                            = np.array(self.intensityByAngle.keys())
            intensities                       = np.array(self.intensityByAngle.values())
            iSorted                           = np.argsort(angles)
            self.intensityByAngle_angles      = angles[iSorted]
            self.intensityByAngle_intensities = intensities[iSorted]
#             rospy.logwarn(self.intensityByAngle_intensities)
            
            angleLo = np.min([self.angle1_bodyframe, self.angle2_bodyframe])
            angleHi = np.max([self.angle1_bodyframe, self.angle2_bodyframe])

            #rospy.logwarn('%s  anglesLo,Hi: %s, %s' % (self.side, angleLo, angleHi))
            
            iValidAngles     = np.where((angleLo < self.intensityByAngle_angles) * (self.intensityByAngle_angles < angleHi))[0]
            self.angles      = self.intensityByAngle_angles[iValidAngles]
            self.intensities = self.intensityByAngle_intensities[iValidAngles]
            
                        
    # calc_wing_angles()
    # Calculate the leading and trailing edge angles,
    # and the amplitude & mean stroke angle.
    #                       
    def calc_wing_angles(self):                
        if self.intensityByAngle is not None:
            self.angleLeadingEdge = None
            self.angleTrailingEdge = None
            self.amp = None
            self.msa = None
            
            if self.bFlying:
#                 if (self.side=='left'):
#                     #rospy.logwarn('angle_indices_ok: %s' % self.angle_indices_ok)
#                     rospy.logwarn('angles: %s' % self.angles)
#                     rospy.logwarn('intensities: %s' % self.intensities)
                
                if (len(self.intensities)>0):
                    # auto rescale
                    self.intensities -= np.min(self.intensities)
                    self.intensities /= np.max(self.intensities)
                    
                    iWing = np.where(self.intensities<self.get_threshold())[0]
                    
                    if (len(iWing)>0):
                        leading_edge_index = iWing[0]
                        trailing_edge_index = iWing[-1]
                
                        self.angleLeadingEdge = int(self.angles[leading_edge_index])
                        self.angleTrailingEdge = int(self.angles[trailing_edge_index])
                
                        self.amp = np.abs(self.angleLeadingEdge - self.angleTrailingEdge)
                        self.msa = np.mean([self.angleLeadingEdge, self.angleTrailingEdge])
            
            else: # not flying
                self.angleLeadingEdge = -180
                self.angleTrailingEdge = -180
                self.amp = 0
                self.msa = 0


    # get_threshold()
    # Calculate and return the best pixel intensity threshold to use for wing angle detection.
    #                    
    def get_threshold(self):
        threshold = self.intensities[np.argmin(np.diff(self.intensities))]
        
        return threshold
        

    def publish_wing_contrast(self):
        if self.intensityByAngle is not None:
            max_contrast = np.abs( np.max(self.intensityByAngle_intensities) - np.min(self.intensityByAngle_intensities) ) 
            if max_contrast > self.flight_threshold:
                self.bFlying = True
            else:
                self.bFlying = False
            self.pubWingContrast.publish(max_contrast)
                

    def publish_wing_edges(self):
        if self.intensityByAngle is not None:
            if (self.angleLeadingEdge is not None):                
                self.pubWingMsa.publish(self.msa)
                self.pubWingAmp.publish(self.amp)
                self.pubWingLeading.publish(self.angleLeadingEdge)
                self.pubWingTrailing.publish(self.angleTrailingEdge)
                
                    
#     def serve_histogram(self, request):
#         if self.intensityByAngle_angles is not None:
#             return float32listResponse(self.intensityByAngle_angles)
#             
#             
#     def serve_bins(self, request):
#         if self.intensityByAngle_intensities is not None:
#             return float32listResponse(self.intensityByAngle_intensities)

    def serve_histogram(self, request):
        if self.intensityByAngle is not None:
            return float32listResponse(self.intensityByAngle.values())
            
    def serve_bins(self, request):
        if self.intensityByAngle is not None:
            return float32listResponse(self.intensityByAngle.keys())
            
            
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
        
        # get parameters from parameter server
        config = rospy.get_param('strokelitude')
        
        # initialize wings and body
        self.wing_r                 = Wing('right', config)
        self.wing_l                 = Wing('left', config)
        self.bodyangle              = config['bodyangle']
        self.bSettingHinges = False
        self.bSettingWings= False
        self.angletext              = 'none' # for debugging

        # initialize
        node_name                   = 'strokelitude'
        rospy.init_node(node_name, anonymous=True)

        # publishers
        name                     = 'strokelitude/wba/' + 'LeftMinusRight'
        self.pubWingLeftMinusRight = rospy.Publisher(name, Float32)
        name                     = 'strokelitude/wba/' + 'LeftPlusRight'
        self.pubWingLeftPlusRight  = rospy.Publisher(name, Float32)
        name                     = 'strokelitude/wba/' + 'flight_status'
        self.pubFlightStatus     = rospy.Publisher(name, Float32)

        name                     = 'strokelitude/wba/' + 'image_mask'
        self.pubMask             = rospy.Publisher(name, Image)
        
        # subscribe to images        
        self.subImageRaw         = rospy.Subscriber('/camera/image_raw',Image,self.image_callback)

        # user callbacks
        cv.SetMouseCallback(self.display_name, self.mouse, param=None)
        
        
    def get_configuration(self, config=None):
        if config is None:
            config = rospy.get_param('strokelitude')
        self.bodyangle = config['bodyangle']
        self.wing_r.extract_parameters(config)
        self.wing_l.extract_parameters(config)
        
        
    def image_callback(self, rosimage):
        # Receive an image:
        try:
            imgCamera = np.uint8(cv.GetMat(self.cvbridge.imgmsg_to_cv(rosimage, 'passthrough')))
        except CvBridgeError, e:
            rospy.logwarn ('Exception converting background image from ROS to opencv:  %s' % e)
            imgCamera = None
            
        if (imgCamera is not None):
            self.shapeImage = imgCamera.shape
            imgOutput = cv2.cvtColor(imgCamera, cv.CV_GRAY2RGB)
                
            left_pixel   = 10
            bottom_pixel = imgOutput.shape[0]-10 
            right_pixel  = imgOutput.shape[1]-10
                
            # draw line from one hinge to the other.
            if (self.wing_r.ptHinge is not None) and (self.wing_l.ptHinge is not None):
                cv2.line(imgOutput, self.wing_r.ptHinge, self.wing_l.ptHinge, cv.Scalar(255,0,0,0), 2)
                #body_center_x = int( (self.wing_r.ptHinge[0] + self.wing_l.ptHinge[0])/2. ) 
                #body_center_y = int( (self.wing_r.ptHinge[1] + self.wing_l.ptHinge[1])/2. ) 
            
            # draw wings
            if self.wing_r.ptHinge is not None:
                self.wing_r.draw(imgOutput)
            if self.wing_l.ptHinge is not None:
                self.wing_l.draw(imgOutput)
            
            if not self.bSettingWings:
                # calculate wing beat analyzer stats
                self.wing_r.updateIntensityByAngle(imgCamera)
                self.wing_r.calc_wing_angles()
                self.wing_r.publish_wing_contrast()
                self.wing_r.publish_wing_edges()
                
                self.wing_l.updateIntensityByAngle(imgCamera)
                self.wing_l.calc_wing_angles()
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
            

            if (self.wing_r.amp is not None) and (self.wing_l.amp is not None):
                self.pubMask.publish(self.cvbridge.cv_to_imgmsg(cv.fromarray(self.wing_r.imgMask + self.wing_l.imgMask), 'passthrough'))
            
            # display image
            cv2.imshow("Display", imgOutput)
            cv2.waitKey(1)

#             # Some debugging stuff.
#             angle1R_imageframe = self.wing_r.transform_angle_imageframe_from_bodyframe(self.wing_r.angle1_bodyframe)
#             angle2R_imageframe = self.wing_r.transform_angle_imageframe_from_bodyframe(self.wing_r.angle2_bodyframe)
#             angle1L_imageframe = self.wing_l.transform_angle_imageframe_from_bodyframe(self.wing_l.angle1_bodyframe)
#             angle2L_imageframe = self.wing_l.transform_angle_imageframe_from_bodyframe(self.wing_l.angle2_bodyframe)
#             rospy.logwarn('angle1,2Lb: %s,   angle1,2Rb: %s' % ((self.wing_l.angle1_bodyframe, self.wing_l.angle2_bodyframe), (self.wing_r.angle1_bodyframe, self.wing_r.angle2_bodyframe)))
#             rospy.logwarn('angle1,2Li: %s,   angle1,2Ri: %s' % ((angle1L_imageframe, angle2L_imageframe), (angle1R_imageframe, angle2R_imageframe)))
#             rospy.logwarn('bodyangle: %s, %s, %s' % (self.bodyangle, self.wing_l.bodyangle, self.wing_r.bodyangle))


    def mouse(self, event, x, y, flags, param):
        ptMouse = (x, y)
        
        # FETCH PARAMETERS : left button double click
        if (event == cv.CV_EVENT_LBUTTONDBLCLK):
            rospy.logwarn('Fetching parameters from parameter server.')
            rospy.logwarn('')
            self.get_configuration()
            self.wing_r.calc_mask(self.shapeImage)
            self.wing_l.calc_mask(self.shapeImage)

            
        # BODY : MBUTTON
        if (event == cv.CV_EVENT_MBUTTONDOWN):
            if (not self.bSettingHinges):
                self.bSettingHinges = True
                if (not flags & cv.CV_EVENT_FLAG_SHIFTKEY): 
                    self.wing_r.ptHinge      = ptMouse
                else: # <Shift> means do the left wing first.                                   
                    self.wing_l.ptHinge      = ptMouse
            else:
                self.bSettingHinges = False
                if (not flags & cv.CV_EVENT_FLAG_SHIFTKEY):
                    self.wing_l.ptHinge      = ptMouse
                else: # <Shift> means do the right wing second.
                    self.wing_r.ptHinge      = ptMouse
                    
                self.bodyangle              = int(get_angle_from_points(self.wing_l.ptHinge, self.wing_r.ptHinge))
                self.wing_r.bodyangle       = self.bodyangle
                self.wing_l.bodyangle       = self.bodyangle
                
                rospy.set_param('strokelitude/wing/right/hinge/x', self.wing_r.ptHinge[0])
                rospy.set_param('strokelitude/wing/right/hinge/y', self.wing_r.ptHinge[1])
                rospy.set_param('strokelitude/wing/left/hinge/x', self.wing_l.ptHinge[0])
                rospy.set_param('strokelitude/wing/left/hinge/y', self.wing_l.ptHinge[1])
                rospy.set_param('strokelitude/bodyangle', self.bodyangle)
            
            
        # During setting of hinge points, update the hinge points and bodyangle.
        if (self.bSettingHinges):
            if (flags & cv.CV_EVENT_FLAG_SHIFTKEY):
                self.wing_r.ptHinge      = ptMouse
            else:
                self.wing_l.ptHinge      = ptMouse

            self.bodyangle              = int(get_angle_from_points(self.wing_l.ptHinge, self.wing_r.ptHinge))
            self.wing_r.bodyangle       = self.bodyangle
            self.wing_l.bodyangle       = self.bodyangle
            
            
        if (self.wing_r.ptHinge is not None) and (self.wing_l.ptHinge is not None):
            if (flags & cv.CV_EVENT_FLAG_SHIFTKEY):
                wing = self.wing_l
            else:
                wing = self.wing_r
                
            radius           = int(np.sqrt( (ptMouse[1] - wing.ptHinge[1])**2 + (ptMouse[0] - wing.ptHinge[0])**2 ))
            angleMouse_imageframe = int(get_angle_from_points(wing.ptHinge, ptMouse))
            angleMouse_bodyframe  = wing.transform_angle_bodyframe_from_imageframe(angleMouse_imageframe)
                
            if self.wing_l.ravelAngleInBodyFrame is not None:
                self.angletext          = str(wing.imgAngleInBodyFrame[y,x])
            

        # WINGS : RBUTTON         
        if (event == cv.CV_EVENT_RBUTTONDOWN) and (not self.bSettingWings):
            self.bSettingWings= True
            self.wing_r.set_angles_bodyframe(angleMouse_bodyframe, self.wing_r.angle2_bodyframe)
            self.wing_l.set_angles_bodyframe(angleMouse_bodyframe, self.wing_l.angle2_bodyframe)
            
            if (flags & cv.CV_EVENT_FLAG_CTRLKEY):
                self.wing_r.radius_inner    = radius
                self.wing_l.radius_inner    = radius
            else:          
                self.wing_l.radius_outer    = radius
                self.wing_r.radius_outer    = radius
            
            
        elif (event == cv.CV_EVENT_RBUTTONDOWN) and (self.bSettingWings):
            self.bSettingWings= False
            self.wing_r.set_angles_bodyframe(self.wing_r.angle1_bodyframe, angleMouse_bodyframe)
            self.wing_l.set_angles_bodyframe(self.wing_l.angle1_bodyframe, angleMouse_bodyframe)
            
            rospy.set_param('strokelitude/wing/angle1', self.wing_r.angle1_bodyframe)
            rospy.set_param('strokelitude/wing/angle2', self.wing_r.angle2_bodyframe)
            rospy.set_param('strokelitude/wing/radius_outer', self.wing_r.radius_outer)
            rospy.set_param('strokelitude/wing/radius_inner', self.wing_r.radius_inner)
            
        
        # During setting of wings, update the wing angles.
        if (self.bSettingWings):
            self.wing_r.set_angles_bodyframe(self.wing_r.angle1_bodyframe, angleMouse_bodyframe)
            self.wing_l.set_angles_bodyframe(self.wing_l.angle1_bodyframe, angleMouse_bodyframe)


    def run(self):
        try:
            rospy.spin()
        except:
            rospy.logwarn('Shutting down')
        
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
    rospy.logwarn('*    Controls:                                                           *')
    rospy.logwarn('*                                                                        *')  
    rospy.logwarn('*    Middle Mouse (single click): select wing hinge                      *')
    rospy.logwarn('*        Default   : Right wing (green)                                  *')
    rospy.logwarn('*        Shift key : Left wing (red)                                     *')
    rospy.logwarn('*    Right Mouse (single click) : draw radius mask                       *')
    rospy.logwarn('*        Default   : Right wing (green), Outer radius (thick)            *')
    rospy.logwarn('*        Shift key : Left wing (red)                                     *')
    rospy.logwarn('*        Ctrl  key : Inner radius (thin)                                 *')
    rospy.logwarn('*    Left Mouse (double click)  : apply parameters, calculate WBA mask   *')
    rospy.logwarn('*                                                                        *') 
    rospy.logwarn('**************************************************************************')
    rospy.logwarn('')
    rospy.logwarn('')

    im = ImageDisplay()
    im.run()
