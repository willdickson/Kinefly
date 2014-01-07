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
    def __init__(self, side='right', config=None):
        self.side = side
        
        if side == 'right':
            self.color ='red'
        elif side == 'left':
            self.color ='green'
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

        self.extract_parameters(config)
        
        
        # Bodyframe angles have zero degrees point orthogonal to body axis: 
        # If body axis is north/south, then 0-deg is east for right wing, west for left wing.
        
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
            angle_i  = self.angleBody + angle_b
        else: # left
            angle_i  = self.angleBody + angle_b + 180
             

        return angle_i
        

    # transform_angle_b_from_i()
    # Transform an angle from the camera image frame to the fly frame.
    #
    def transform_angle_b_from_i(self, angle_i):
        if self.side == 'right':
            angle_b  = angle_i - self.angleBody
        else:  
            angle_b  = -(self.angleBody - angle_i + 180) 

        return angle_b
        

    # set_angles_b()
    # Set the angles into the member vars.  Enforces the range [-180,+180].
    #
    def set_angles_b(self, angle_lo_b, angle_hi_b):
        self.angle_lo_b = (angle_lo_b+180) % 360 - 180
        self.angle_hi_b = (angle_hi_b+180) % 360 - 180
        
        
    # extract parameters from parameter server, given "strokelitude" parameter
    def extract_parameters(self, config):
        if config is not None:
            self.ptHinge        = np.array([config['wing'][self.side]['hinge']['x'], config['wing'][self.side]['hinge']['y']])
            
            self.radius_outer   = config['wing']['radius_outer']
            self.radius_inner   = config['wing']['radius_inner']
            self.angleBody      = config['bodyangle']
            self.threshold      = config['wing'][self.side]['threshold']
            self.flight_threshold=config['wing'][self.side]['flight_threshold']

            self.set_angles_b(config['wing']['angle_lo_b'], config['wing']['angle_hi_b'])
            
            
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
        (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.angle_lo_b, self.angle_hi_b)

        # inner circle
        cv2.ellipse(image, 
                    tuple(self.ptHinge),  
                    (self.radius_inner, self.radius_inner),
                    0,
                    angle_lo_i,
                    angle_hi_i,
                    color=self.numeric_color,
                    thickness=self.thickness_inner,
                    )
        
        # outer circle         
        cv2.ellipse(image, 
                    tuple(self.ptHinge),  
                    (self.radius_outer, self.radius_outer),
                    0,
                    angle_lo_i,
                    angle_hi_i,
                    color=self.numeric_color,
                    thickness=self.thickness_outer,
                    )
        
        
        # wing leading and trailing edges
        if self.angleTrailingEdge is not None:
            (angle_leading, angle_trailing) = self.get_angles_i_from_b(self.angleLeadingEdge, self.angleTrailingEdge)

#             if (self.side=='left'):
#                 rospy.logwarn('L,T: %s,   l,t: %s' % ((self.angleLeadingEdge, self.angleTrailingEdge), (angle_leading, angle_trailing)))
            x = int(self.ptHinge[0] + self.radius_outer * np.cos( np.deg2rad(angle_trailing) ))
            y = int(self.ptHinge[1] + self.radius_outer * np.sin( np.deg2rad(angle_trailing) ))
            cv2.line(image, tuple(self.ptHinge), (x,y), self.numeric_color, self.thickness_wing)
            
            x = int(self.ptHinge[0] + self.radius_outer * np.cos( np.deg2rad(angle_leading) ))
            y = int(self.ptHinge[1] + self.radius_outer * np.sin( np.deg2rad(angle_leading) ))
            cv2.line(image, tuple(self.ptHinge), (x,y), self.numeric_color, self.thickness_wing)

                        
    def calc_mask(self, shape):
        # Calculate the angle at each pixel.
        self.imgAngleInBodyFrame = np.zeros(shape)
        for y in range(shape[0]):
            for x in range(shape[1]):
                angle_i = get_angle_from_points(self.ptHinge, (x,y))
                angle_b  = self.transform_angle_b_from_i(angle_i)

                if self.side == 'left':
                    angle_b = -angle_b - 180

                self.imgAngleInBodyFrame[y,x] = angle_b
        
        self.imgAngleInBodyFrame = (self.imgAngleInBodyFrame+180) % 360 - 180
        self.ravelAngleInBodyFrame = np.ravel(self.imgAngleInBodyFrame)
#        rospy.logwarn('%s: angles on range [%s, %s]' % (self.side, self.imgAngleInBodyFrame.min(), self.imgAngleInBodyFrame.max()))
                     
        # Create the wing mask.
        self.imgMask = np.zeros(shape)
        (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.angle_lo_b, self.angle_hi_b)

        cv2.ellipse(self.imgMask,
                    tuple(self.ptHinge),
                    (self.radius_outer, self.radius_outer),
                    0,
                    angle_lo_i,
                    angle_hi_i,
                    1, 
                    cv.CV_FILLED)
        cv2.ellipse(self.imgMask,
                    tuple(self.ptHinge),
                    (self.radius_inner, self.radius_inner),
                    0,
                    angle_lo_i,
                    angle_hi_i,
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
            
            angle_lo_b = np.min([self.angle_lo_b, self.angle_hi_b])
            angle_hi_b = np.max([self.angle_lo_b, self.angle_hi_b])

            #rospy.logwarn('%s  anglesLo,Hi: %s, %s' % (self.side, angle_lo_b, angle_hi_b))
            
            iValidAngles     = np.where((angle_lo_b < self.intensityByAngle_angles) * (self.intensityByAngle_angles < angle_hi_b))[0]
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
                
                if (len(self.intensities)>1):
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
        self.angleBody              = config['bodyangle']
        self.bSettingHinges = False
        self.bSettingWings= False
        self.angletext              = 'none' # for debugging

        self.controlpts = {}
        self.update_control_points()
        
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
        cv.SetMouseCallback(self.display_name, self.onMouse, param=None)
        
        
    def get_configuration(self, config=None):
        if config is None:
            config = rospy.get_param('strokelitude')
        self.angleBody = config['bodyangle']
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
                cv2.line(imgOutput, tuple(self.wing_r.ptHinge), tuple(self.wing_l.ptHinge), cv.Scalar(255,0,0,0), 2)
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
    #
    def update_control_points (self):
        # Compute the various control points.
        self.controlpts['hinge_l'] = self.wing_l.ptHinge
        self.controlpts['hinge_r'] = self.wing_r.ptHinge
        
        (angle_lo_i, angle_hi_i) = self.wing_l.get_angles_i_from_b(self.wing_l.angle_lo_b, self.wing_l.angle_hi_b)
        self.controlpts['hi_l'] = (self.wing_l.ptHinge + self.wing_l.radius_outer * np.array([np.cos(np.deg2rad(angle_hi_i)), 
                                                                                              np.sin(np.deg2rad(angle_hi_i))])).astype(int)
        self.controlpts['lo_l'] = (self.wing_l.ptHinge + self.wing_l.radius_outer * np.array([np.cos(np.deg2rad(angle_lo_i)), 
                                                                                              np.sin(np.deg2rad(angle_lo_i))])).astype(int)

        (angle_lo_i, angle_hi_i) = self.wing_r.get_angles_i_from_b(self.wing_r.angle_lo_b, self.wing_r.angle_hi_b)
        self.controlpts['hi_r'] = (self.wing_r.ptHinge + self.wing_r.radius_outer * np.array([np.cos(np.deg2rad(angle_hi_i)), 
                                                                                              np.sin(np.deg2rad(angle_hi_i))])).astype(int)
        self.controlpts['lo_r'] = (self.wing_r.ptHinge + self.wing_r.radius_outer * np.array([np.cos(np.deg2rad(angle_lo_i)), 
                                                                                              np.sin(np.deg2rad(angle_lo_i))])).astype(int)

        angle_lo = self.wing_l.angle_lo_b
        angle_hi = self.wing_l.angle_hi_b
        if (angle_hi-angle_lo > 180):
            angle_hi -= 360
        if (angle_lo-angle_hi > 180):
            angle_lo -= 360
        angle = self.wing_l.angleBody + (angle_lo + angle_hi)/2 + 180
        self.controlpts['inner_l'] = (self.wing_l.ptHinge + self.wing_l.radius_inner * np.array([np.cos(np.deg2rad(angle)), 
                                                                                                 np.sin(np.deg2rad(angle))])).astype(int)

        angle_lo = self.wing_r.angle_lo_b
        angle_hi = self.wing_r.angle_hi_b
        if (angle_hi-angle_lo > 180):
            angle_hi -= 360
        if (angle_lo-angle_hi > 180):
            angle_lo -= 360
        angle = self.wing_r.angleBody + (angle_lo + angle_hi)/2
        self.controlpts['inner_r'] = (self.wing_r.ptHinge + self.wing_r.radius_inner * np.array([np.cos(np.deg2rad(angle)), 
                                                                                                 np.sin(np.deg2rad(angle))])).astype(int)


        
    def onMouse(self, event, x, y, flags, param):
        ptMouse = np.array([x, y])

        # Keep track of the mouse button state.        
        if (event==cv.CV_EVENT_LBUTTONDOWN):
            self.controlselected = self.hit_test(ptMouse)
        if (event==cv.CV_EVENT_LBUTTONUP):
            self.controlselected = None
#             self.get_configuration()
            self.wing_r.calc_mask(self.shapeImage)
            self.wing_l.calc_mask(self.shapeImage)



        # When the left button is down, adjust the control points.
        if (flags & cv.CV_EVENT_FLAG_LBUTTON):
            # Set the new point.             
            if (self.controlselected=='hinge_l'): # Left hinge point.
                self.wing_l.ptHinge   = ptMouse
                self.angleBody        = int(get_angle_from_points(self.wing_l.ptHinge, self.wing_r.ptHinge))
                self.wing_r.angleBody = self.angleBody
                self.wing_l.angleBody = self.angleBody

            elif (self.controlselected=='hinge_r'): # Right hinge point.
                self.wing_r.ptHinge   = ptMouse
                self.angleBody        = int(get_angle_from_points(self.wing_l.ptHinge, self.wing_r.ptHinge))
                self.wing_r.angleBody = self.angleBody
                self.wing_l.angleBody = self.angleBody
        
            elif (self.controlselected=='hi_l'): # Left high angle.
                pt = ptMouse - self.wing_l.ptHinge
                self.wing_l.angle_hi_b = self.wing_l.transform_angle_b_from_i(np.rad2deg(np.arctan2(pt[1], pt[0])))
                self.wing_l.radius_outer = max(self.wing_l.radius_inner+2, int(np.linalg.norm(self.wing_l.ptHinge - ptMouse)))
                  
            elif (self.controlselected=='hi_r'): # Right high angle.
                pt = ptMouse - self.wing_r.ptHinge
                self.wing_r.angle_hi_b = self.wing_r.transform_angle_b_from_i(np.rad2deg(np.arctan2(pt[1], pt[0])))
                self.wing_r.radius_outer = max(self.wing_r.radius_inner+2, int(np.linalg.norm(self.wing_r.ptHinge - ptMouse)))
                  
            elif (self.controlselected=='lo_l'): # Left low angle.
                pt = ptMouse - self.wing_l.ptHinge
                self.wing_l.angle_lo_b = self.wing_l.transform_angle_b_from_i(np.rad2deg(np.arctan2(pt[1], pt[0])))
                self.wing_l.radius_outer = max(self.wing_l.radius_inner+2, int(np.linalg.norm(self.wing_l.ptHinge - ptMouse)))
                  
            elif (self.controlselected=='lo_r'): # Right low angle.
                pt = ptMouse - self.wing_r.ptHinge
                self.wing_r.angle_lo_b = self.wing_r.transform_angle_b_from_i(np.rad2deg(np.arctan2(pt[1], pt[0])))
                self.wing_r.radius_outer = max(self.wing_r.radius_inner+2, int(np.linalg.norm(self.wing_r.ptHinge - ptMouse)))
                  
            elif (self.controlselected=='inner_l'): # Left inner radius.
                self.wing_l.radius_inner = min(int(np.linalg.norm(self.wing_l.ptHinge - ptMouse)), self.wing_l.radius_outer-2)
                
            elif (self.controlselected=='inner_r'): # Right inner radius.
                self.wing_r.radius_inner = min(int(np.linalg.norm(self.wing_r.ptHinge - ptMouse)), self.wing_r.radius_outer-2)
                
            self.update_control_points()
            
                
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
