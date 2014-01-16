#!/usr/bin/env python
from __future__ import division
import roslib; roslib.load_manifest('StrokelitudeROS')
import rospy

import copy
import cv
import cv2
import numpy as np
import os
import sys
import time
import dynamic_reconfigure.server

from cv_bridge import CvBridge, CvBridgeError
from optparse import OptionParser
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from StrokelitudeROS.srv import *
from StrokelitudeROS.msg import float32list as float32list_msg
from StrokelitudeROS.cfg import strokelitudeConfig




###############################################################################
###############################################################################
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
        self.ravelMask             = None
        self.ravel_angle_b         = None

        self.bins                  = None
        self.intensities           = None
        self.binsValid             = None
        self.intensitiesValid      = None

        
        # Bodyframe angles have zero degrees point orthogonal to body axis: 
        # If body axis is north/south, then 0-deg is east for right wing, west for left wing.
        
        # Imageframe angles are oriented to the image.  0-deg is east, +-180 is west, -90 is north, +90 is south.
        
        
        self.angle_trailing_b = None
        self.angle_leading_b  = None
        self.angle_amplitude  = None
        self.angle_mean       = None
        self.contrast         = 0.0
        self.bFlying          = False
        
        # drawing parameters
        self.color_bgra_dict = {'green'         : cv.Scalar(0,255,0,0), 
                                'light_green'   : cv.Scalar(175,255,175,0),
                                'red'           : cv.Scalar(0,0,255,0),
                                'blue'          : cv.Scalar(255,0,0,0),
                                'purple'        : cv.Scalar(255,0,255,0),
                                'white'         : cv.Scalar(255,255,255,0),
                                'black'         : cv.Scalar(0,0,0,0),
                                }
        self.color_bgra         = self.color_bgra_dict[self.color]
        self.thickness_inner    = 1
        self.thickness_outer    = 2
        self.thickness_wing     = 2
        
        self.set_params(params)

        # services, for live histograms
        self.service_histogram = rospy.Service('wing_histogram_'+side, float32list, self.serve_histogram)
        self.service_bins      = rospy.Service('wing_bins_'+side, float32list, self.serve_bins)
        self.service_edges     = rospy.Service('wing_edges_'+side, float32list, self.serve_edges)
        
        
        # Publishers
        #self.pubSum           = rospy.Publisher('strokelitude/' + self.side + '/sum', Float32)
        self.pubContrast       = rospy.Publisher('strokelitude/' + self.side + '/contrast', Float32)
        self.pubAngleAmplitude = rospy.Publisher('strokelitude/' + self.side + '/angle_amplitude', Float32)
        self.pubAngleMean      = rospy.Publisher('strokelitude/' + self.side + '/angle_mean', Float32)
        self.pubAngleLeading   = rospy.Publisher('strokelitude/' + self.side + '/angle_leading', Float32)
        self.pubAngleTrailing  = rospy.Publisher('strokelitude/' + self.side + '/angle_trailing', Float32)
        
    
    def get_angle_from_points(self, pt1, pt2):
        x = pt2[0] - pt1[0]
        y = pt2[1] - pt1[1]
        return np.rad2deg(np.arctan2(y,x))


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
        

    # set_params()
    # Set the given params dict into this object.
    #
    def set_params(self, params):
        self.params  = params
        self.ptHinge = np.array([self.params[self.side]['hinge']['x'], self.params[self.side]['hinge']['y']])
        self.bins    = np.linspace(-180, 180, self.params['nbins'])
        
        
        
    def get_bodyangle_i(self):
        return self.get_angle_from_points(np.array([self.params['left']['hinge']['x'], self.params['left']['hinge']['y']]), 
                                          np.array([self.params['right']['hinge']['x'], self.params['right']['hinge']['y']]))
        
                
    def get_hinge(self):
        return np.array([self.params[self.side]['hinge']['x'], self.params[self.side]['hinge']['y']]) 
        
        
    # draw()
    # Draw the wing envelope and leading and trailing edges, onto the given image.
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
                    color=self.color_bgra,
                    thickness=self.thickness_inner,
                    )
        
        # outer circle         
        cv2.ellipse(image, 
                    tuple(self.ptHinge),  
                    (self.params[self.side]['radius_outer'], self.params[self.side]['radius_outer']),
                    0,
                    angle_lo_i,
                    angle_hi_i,
                    color=self.color_bgra,
                    thickness=self.thickness_outer,
                    )
        
        
        # wing leading and trailing edges
        if (self.angle_leading_b is not None):
            (angle_leading_i, angle_trailing_i) = self.get_angles_i_from_b(self.angle_leading_b, self.angle_trailing_b)

            x = int(self.ptHinge[0] + self.params[self.side]['radius_outer'] * np.cos( np.deg2rad(angle_trailing_i) ))
            y = int(self.ptHinge[1] + self.params[self.side]['radius_outer'] * np.sin( np.deg2rad(angle_trailing_i) ))
            cv2.line(image, tuple(self.ptHinge), (x,y), self.color_bgra, self.thickness_wing)
            
            x = int(self.ptHinge[0] + self.params[self.side]['radius_outer'] * np.cos( np.deg2rad(angle_leading_i) ))
            y = int(self.ptHinge[1] + self.params[self.side]['radius_outer'] * np.sin( np.deg2rad(angle_leading_i) ))
            cv2.line(image, tuple(self.ptHinge), (x,y), self.color_bgra, self.thickness_wing)

                        
    # create_angle_mask()
    # Create an image where each pixel value is the angle from the hinge.                    
    # 
    def create_angle_mask(self, shape):
        # Set up matrices of x and y coordinates.
        x = np.tile(np.array([range(shape[1])])   - self.ptHinge[0], (shape[0], 1))
        y = np.tile(np.array([range(shape[0])]).T - self.ptHinge[1], (1, shape[1]))
        
        # Calc their angles.
        angle_i = np.rad2deg(np.arctan2(y,x))
        self.img_angle_b  = self.transform_angle_b_from_i(angle_i)

        self.ravel_angle_b = np.ravel(self.img_angle_b)
                   

    # create_stroke_mask()
    # Create a mask of valid wingstroke areas.
    #
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
    # Create two lists, one containing the pixel indices for each bin, and the other containing the intensities (mean pixel values).
    #
    def assign_pixels_to_bins(self):
        # Create empty bins.
        self.pixels = [[] for i in range(len(self.bins))]
        self.intensities = np.zeros(len(self.bins))

        # Put each pixel into an appropriate bin.            
        for iPixel, angle in enumerate(self.ravel_angle_b):
            if self.ravelMask[iPixel]:
                iBest = np.argmin(np.abs(self.bins - angle))
                self.pixels[iBest].append(iPixel)
                
         
    def filter_median(self, data):
        data2 = copy.copy(data)
        q = 1
        for i in range(q,len(data)-q):
            data2[i] = np.median(data[i-q:i+q]) # Median filter of window.

        return data2
        
        
    # get_edges()
    # Get the angles of the two wing edges.
    #                    
    def get_edges(self):
        intensities = self.intensitiesValid
        diff = self.filter_median(np.diff(intensities))
            
        iPeak = np.argmax(intensities)
        iMax = np.argmax(diff)
        iMin = np.argmin(diff)
        (iMajor,iMinor) = (iMax,iMin) if (np.abs(diff[iMax]) > np.abs(diff[iMin])) else (iMin,iMax) # Major and minor edges.

        iEdge1 = iMajor
        
        # The minor edge must be at least 3/4 the strength of the major edge to be used, else use the end of the array.
        if (3*np.abs(diff[iMajor])/4 < np.abs(diff[iMinor])):
            iEdge2 = iMinor
        else:
            if (self.params['invert']):
                iEdge2 = 0              # Front edge.
            else:
                iEdge2 = -1             # Back edge.
        
        angle1 = float(self.binsValid[iEdge1])
        angle2 = float(self.binsValid[iEdge2])
        
        return (angle1, angle2)
        

    # update_intensities()
    # Update the list of intensities corresponding to the bin angles.
    #            
    def update_intensities(self, image):
        if (self.ravelMask is not None) and (self.intensities is not None):
            ravelImageMasked = np.ravel(image).astype(float)/255. * self.ravelMask.astype(float)
            for iBin in range(len(self.bins)):
                iPixels = self.pixels[iBin]
                if (len(iPixels) > 0):
                    self.intensities[iBin] = np.sum(ravelImageMasked[iPixels]) / float(len(iPixels))
                else:
                    self.intensities[iBin] = 0.0
             
            angle_lo_b = min([self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi']])
            angle_hi_b = max([self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi']])
 
            iValid                = np.where((angle_lo_b < self.bins) * (self.bins < angle_hi_b))[0]
            self.binsValid        = self.bins[iValid]
            self.intensitiesValid = self.intensities[iValid]

                        
    # update_edge_stats()
    # Calculate the leading and trailing edge angles,
    # and the amplitude & mean stroke angle.
    #                       
    def update_edge_stats(self):                
        self.angle_leading_b = None
        self.angle_trailing_b = None
        self.angle_amplitude = None
        self.angle_mean = None
            
        if (self.intensitiesValid is not None):
            if self.bFlying:
                if (len(self.intensitiesValid)>1):
                    (self.angle_leading_b, self.angle_trailing_b) = self.get_edges()
                    self.angle_amplitude = np.abs(self.angle_leading_b - self.angle_trailing_b)
                    self.angle_mean = np.mean([self.angle_leading_b, self.angle_trailing_b])
                    
            else: # not flying
                self.angle_leading_b = -180.
                self.angle_trailing_b = -180.
                self.angle_amplitude = 0.
                self.angle_mean = 0.


    def update_wing_contrast(self):
        if (self.intensitiesValid is not None) and (len(self.intensitiesValid)>1):
            self.contrast = np.abs( np.max(self.intensitiesValid) - np.min(self.intensitiesValid) )


    def update_flight_status(self):
        if (self.contrast > self.params['contrast_threshold']):
            self.bFlying = True
        else:
            self.bFlying = False
                

    def publish_wingdata(self):
        if (self.contrast is not None):
            self.pubContrast.publish(self.contrast)
            
        if (self.angle_mean is not None):
            self.pubAngleMean.publish(self.angle_mean)
            
        if (self.angle_amplitude is not None):
            self.pubAngleAmplitude.publish(self.angle_amplitude)
            
        if (self.angle_leading_b is not None):
            self.pubAngleLeading.publish(self.angle_leading_b)
            
        if (self.angle_trailing_b is not None):
            self.pubAngleTrailing.publish(self.angle_trailing_b)
                
                    
    def serve_bins(self, request):
        if (self.bins is not None):
            return float32listResponse(self.bins)
            #return float32listResponse(self.binsValid[:-1])
            
    def serve_histogram(self, request):
        if (self.intensities is not None):
            return float32listResponse(self.intensities)
            #return float32listResponse(np.diff(self.intensitiesValid))
            #return float32listResponse(self.filter_median(np.diff(self.intensitiesValid)))
            
    def serve_edges(self, request):
        if (self.angle_trailing_b is not None):            
            return float32listResponse([self.angle_trailing_b, self.angle_leading_b])
            
            
            
###############################################################################
###############################################################################
class Button:
    def __init__(self, name=None, text=None, rect=(0,0,0,0)):
        self.name = name
        self.rect = rect
        self.state = 'up'
        self.set_text(text)

        self.colorWhite = cv.Scalar(255,255,255,0)
        self.colorBlack = cv.Scalar(0,0,0,0)
        self.colorBtn = cv.Scalar(128,128,128,0)
        self.colorHilight = cv.Scalar(192,192,192,0)
        self.colorLolight = cv.Scalar(64,64,64,0)
        self.colorText = cv.Scalar(255,255,255,0)

        self.ptLT = (self.rect[0],                 self.rect[1])
        self.ptRT = (self.rect[0]+self.rect[2],    self.rect[1])
        self.ptLB = (self.rect[0],                 self.rect[1]+self.rect[3])
        self.ptRB = (self.rect[0]+self.rect[2],    self.rect[1]+self.rect[3])

        self.ptLT0 = (self.rect[0]-1,              self.rect[1]-1)
        self.ptRT0 = (self.rect[0]+self.rect[2]+1, self.rect[1]-1)
        self.ptLB0 = (self.rect[0]-1,              self.rect[1]+self.rect[3]+1)
        self.ptRB0 = (self.rect[0]+self.rect[2]+1, self.rect[1]+self.rect[3]+1)

        self.ptLT1 = (self.rect[0]+1,              self.rect[1]+1)
        self.ptRT1 = (self.rect[0]+self.rect[2]-1, self.rect[1]+1)
        self.ptLB1 = (self.rect[0]+1,              self.rect[1]+self.rect[3]-1)
        self.ptRB1 = (self.rect[0]+self.rect[2]-1, self.rect[1]+self.rect[3]-1)

        self.ptLT2 = (self.rect[0]+2,              self.rect[1]+2)
        self.ptRT2 = (self.rect[0]+self.rect[2]-2, self.rect[1]+2)
        self.ptLB2 = (self.rect[0]+2,              self.rect[1]+self.rect[3]-2)
        self.ptRB2 = (self.rect[0]+self.rect[2]-2, self.rect[1]+self.rect[3]-2)
        


    def hit_test(self, ptMouse):
        if (self.rect[0] <= ptMouse[0] <= self.rect[0]+self.rect[2]) and (self.rect[1] <= ptMouse[1] <= self.rect[1]+self.rect[3]):
            return True
        else:
            return False
        

    def set_text(self, text):
        self.text = text
        (sizeText,rv) = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        self.ptText = (int(self.rect[0]+self.rect[2]/2-sizeText[0]/2), int(self.rect[1]+self.rect[3]/2+sizeText[1]/2))

                
    # draw_button()
    # Draw a 3D shaded button with text.
    # rect is (left, top, width, height), increasing y goes down.
    def draw(self, image):

        if (self.state=='up'):
            colorOuter = self.colorWhite
            colorInner = self.colorBlack
            colorHilight = self.colorHilight
            colorLolight = self.colorLolight
            colorFill = self.colorBtn
            colorText = self.colorText
            ptText0 = (self.ptText[0], self.ptText[1])
        else:
            colorOuter = self.colorWhite
            colorInner = self.colorBlack
            colorHilight = self.colorLolight
            colorLolight = self.colorHilight
            colorFill = self.colorBtn
            colorText = self.colorText
            ptText0 = (self.ptText[0]+2, self.ptText[1]+2)
            
        cv2.rectangle(image, self.ptLT0, self.ptRB0, colorOuter, 1)
        cv2.rectangle(image, self.ptLT, self.ptRB, colorInner, 1)
        cv2.line(image, self.ptRT1, self.ptRB1, colorLolight)
        cv2.line(image, self.ptLB1, self.ptRB1, colorLolight)
        cv2.line(image, self.ptLT1, self.ptRT1, colorHilight)
        cv2.line(image, self.ptLT1, self.ptLB1, colorHilight)
        cv2.rectangle(image, self.ptLT2, self.ptRB2, colorFill, cv.CV_FILLED)

        cv2.putText(image, self.text, ptText0, cv2.FONT_HERSHEY_SIMPLEX, 0.4, colorText)
        
                
    
            
###############################################################################
###############################################################################
class MainWindow:

    class struct:
        pass
    
    def __init__(self):
        # initialize
        rospy.init_node('strokelitude', anonymous=True)

        # initialize display
        self.window_name = 'Strokelitude'
        cv.NamedWindow(self.window_name,1)
        self.cvbridge = CvBridge()
        
        # Get parameters from parameter server
        self.params = rospy.get_param('strokelitude')
        defaults = {'filenameBackground':'~/strokelitude.png',
                    'image_topic':'/camera/image_raw',
                    'use_gui':True,
                    'mirror':True,
                    'nbins':361,
                    'invert':False,
                    'contrast_threshold':0.1,
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
        self.set_dict_with_preserve(self.params, defaults)
        

        # Background image.
        self.filenameBackground = os.path.expanduser(self.params['filenameBackground'])
        self.imgBackground  = cv2.imread(self.filenameBackground, cv.CV_LOAD_IMAGE_GRAYSCALE)
        
        
        # initialize wings and body
        self.wing_r                 = Wing('right', self.params)
        self.wing_l                 = Wing('left', self.params)
        self.bInitWings = False
        self.bConstrainMaskToImage = False
        
        self.ptBody1 = None
        self.ptBody2 = None
        self.nameSelected = None
        self.typeSelected = None
        self.handlepts = {}
        self.update_handle_points()

        # Publishers.
        self.pubLeftMinusRight     = rospy.Publisher('strokelitude/left_minus_right', Float32)
        self.pubLeftPlusRight      = rospy.Publisher('strokelitude/left_plus_right', Float32)
        self.pubFlightStatus       = rospy.Publisher('strokelitude/flight_status', Float32)
        self.pubCommand            = rospy.Publisher('strokelitude/command', String)

        # Subscriptions.        
        self.subImageRaw           = rospy.Subscriber(self.params['image_topic'], Image, self.image_callback)
        self.subCommand            = rospy.Subscriber('strokelitude/command', String, self.command_callback)

        self.w_gap = 30
        self.scaleText = 0.4
        self.fontface = cv2.FONT_HERSHEY_SIMPLEX

        # UI button specs.
        self.buttons = []
        x = 10
        y = 10
        w = 40
        h = 18
        self.buttons.append(Button(name='exit', 
                                   text='exit', 
                                   rect=(x, y, w, h)))
        
        x += w+2
        w = 65
        self.buttons.append(Button(name='save bg', 
                                   text='save bg', 
                                   rect=(x, y, w, h)))
        
        x += w+2
        w = 90
        self.buttons.append(Button(name='invert', 
                                   text='inverted' if (self.params['invert']) else 'not inverted', 
                                   rect=(x, y, w, h)))

        x += w+2
        w = 95
        self.buttons.append(Button(name='mirror', 
                                   text='mirrored' if (self.params['mirror']) else 'not mirrored', 
                                   rect=(x, y, w, h)))


        # user callbacks
        cv.SetMouseCallback(self.window_name, self.onMouse, param=None)
        
        self.reconfigure = dynamic_reconfigure.server.Server(strokelitudeConfig, self.reconfigure_callback)
        
        
    def reconfigure_callback(self, config, level):
        # Save the new params.
        self.set_dict_with_overwrite(self.params, config)
        
        # Remove dynamic_reconfigure keys from the params.
        try:
            self.params.pop('groups')
        except KeyError:
            pass
        
        # Set it into the wings.
        self.wing_l.set_params(self.params)
        self.wing_r.set_params(self.params)
        
        return config


    # command_callback()
    # Execute any commands sent over the command topic.
    #
    def command_callback(self, command):
        self.command = command.data
        
        if (self.command == 'save_background'):
            self.save_background()
            
        
        if ('invert' in self.command):
            if (self.command == 'invert'):
                self.params['invert'] = not self.params['invert']
                
            for button in self.buttons:
                if (button.name=='invert'):
                    button.set_text('inverted' if (self.params['invert']) else 'not inverted')
                    
            
        if ('mirror' in self.command):
            if (self.command == 'mirror'):
                self.params['mirror'] = not self.params['mirror']
            
            for button in self.buttons:
                if (button.name=='mirror'):
                    button.set_text('mirrored' if (self.params['mirror']) else 'not mirrored')
                    
            
        if (self.command == 'exit'):
            rospy.signal_shutdown('User requested exit.')
        
        
        if (self.command == 'help'):
            rospy.logwarn('The strokelitude/command topic accepts the following string commands:')
            rospy.logwarn('  help                 This message.')
            rospy.logwarn('  invert               If we only see one edge on a wing, the second edge to use is switched via this flag.  Toggle the invert state.')
            rospy.logwarn('  mirror               Toggle the wing envelope symmetry.')
            rospy.logwarn('  save_background      Saves the instant camera image to disk for use with background subtraction.')
            rospy.logwarn('  exit                 Exit the program.')
            rospy.logwarn('')
            rospy.logwarn('You can send commands at the shell prompt via:')
            rospy.logwarn('rostopic pub -1 strokelitude/command std_msgs/String commandtext')
            rospy.logwarn('')
            rospy.logwarn('You may also set general parameters via ROS dynamic_reconfigure, for example:')
            rospy.logwarn('rosrun dynamic_reconfigure dynparam set strokelitude mirror true')
            rospy.logwarn('rosrun dynamic_reconfigure dynparam set strokelitude contrast_threshold 0.2')
            rospy.logwarn('')

        self.wing_l.set_params(self.params)
        self.wing_r.set_params(self.params)
        rospy.set_param('strokelitude', self.params)
    
        
    # set_dict(self, dTarget, dSource, bPreserve)
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
    def set_dict(self, dTarget, dSource, bPreserve):
        for k,v in dSource.iteritems():
            bKeyExists = (k in dTarget)
            if (not bKeyExists) and type(v)==type({}):
                dTarget[k] = {}
            if ((not bKeyExists) or not bPreserve) and (type(v)!=type({})):
                dTarget[k] = v
                    
            if type(v)==type({}):
                self.set_dict(dTarget[k], v, bPreserve)
    
    
    def set_dict_with_preserve(self, dTarget, dSource):
        self.set_dict(dTarget, dSource, True)
    
    def set_dict_with_overwrite(self, dTarget, dSource):
        self.set_dict(dTarget, dSource, False)


    def create_masks(self):
        self.wing_r.create_stroke_mask(self.shapeImage)
        self.wing_r.create_angle_mask(self.shapeImage)
        self.wing_r.assign_pixels_to_bins()
        self.wing_l.create_stroke_mask(self.shapeImage)
        self.wing_l.create_angle_mask(self.shapeImage)
        self.wing_l.assign_pixels_to_bins()


    # Draw user-interface elements on the image.
    def draw_buttons(self, image):
        for i in range(len(self.buttons)):
            self.buttons[i].draw(image)


    def draw_handles(self, image):
        # Draw the handle points.
        for handlename,handlept in self.handlepts.iteritems():
            cv2.circle(image, tuple(handlept),  2, cv.Scalar(255,255,255,0), 2)         

    
    def image_callback(self, rosimage):
        # Receive an image:
        try:
            self.imgCamera = np.uint8(cv.GetMat(self.cvbridge.imgmsg_to_cv(rosimage, 'passthrough')))
        except CvBridgeError, e:
            rospy.logwarn ('Exception converting background image from ROS to opencv:  %s' % e)
            self.imgCamera = None
            
        if (self.imgCamera is not None):
            # Background subtraction.
            if (self.imgBackground is not None):
                imgForeground = cv2.absdiff(self.imgCamera, self.imgBackground)
            else:
                imgForeground = self.imgCamera
                
                
            self.shapeImage = self.imgCamera.shape # (height,width)
            
            if (self.params['use_gui']):
                imgOutput = cv2.cvtColor(imgForeground, cv.CV_GRAY2RGB)

            if (not self.bInitWings):
                self.create_masks()
                self.bInitWings = True
                                
            if (self.params['use_gui']):
                x_left   = 10
                y_bottom = imgOutput.shape[0]-10 
                x_right  = imgOutput.shape[1]-10
                    
                # Draw line to indicate the body.
                if (self.ptBody1 is not None) and (self.ptBody2 is not None):
                    cv2.line(imgOutput, self.ptBody1, self.ptBody2, cv.Scalar(255,0,0,0), 2)
                
                # Draw wings
                if self.wing_r.get_hinge() is not None:
                    self.wing_r.draw(imgOutput)
                if self.wing_l.get_hinge() is not None:
                    self.wing_l.draw(imgOutput)
                
                self.draw_buttons(imgOutput)
                self.draw_handles(imgOutput)
            
            
            self.wing_r.update_intensities(imgForeground)
            self.wing_r.update_edge_stats()
            self.wing_r.update_wing_contrast()
            self.wing_r.update_flight_status()
            self.wing_r.publish_wingdata()
            
            self.wing_l.update_intensities(imgForeground)
            self.wing_l.update_edge_stats()
            self.wing_l.update_wing_contrast()
            self.wing_l.update_flight_status()
            self.wing_l.publish_wingdata()
            
            if (self.params['use_gui']):
                x = x_left

                if (self.wing_l.angle_amplitude is not None):
                    s = 'L:% 6.1f' % self.wing_l.angle_amplitude
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, self.wing_l.color_bgra)
                    w_text = 50
                    x += w_text+self.w_gap
                
                if (self.wing_r.angle_amplitude is not None):
                    s = 'R:% 6.1f' % self.wing_r.angle_amplitude
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, self.wing_r.color_bgra)
                    w_text = 50
                    x += w_text+self.w_gap
                
    
                # Output sum of WBA
                if (self.wing_l.angle_amplitude is not None) and (self.wing_r.angle_amplitude is not None):
                    leftplusright = self.wing_l.angle_amplitude + self.wing_r.angle_amplitude
                    self.pubLeftPlusRight.publish(leftplusright)
                    s = 'L+R:% 6.1f' % leftplusright
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, cv.Scalar(255,64,64,0) )
                    w_text = 70
                    x += w_text+self.w_gap

                    
                # Output difference in WBA
                if (self.wing_l.angle_amplitude is not None) and (self.wing_r.angle_amplitude is not None):
                    leftminusright = self.wing_l.angle_amplitude - self.wing_r.angle_amplitude
                    self.pubLeftMinusRight.publish(leftminusright)
                    s = 'L-R:% 6.1f' % leftminusright
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, cv.Scalar(255,64,64,0) )
                    w_text = 70
                    x += w_text+self.w_gap

                    
                # Output flight status
                if (self.wing_l.bFlying and self.wing_r.bFlying):
                    s = 'FLIGHT'
                    self.pubFlightStatus.publish(1)
                else:
                    s = 'no flight'
                    self.pubFlightStatus.publish(0)
                
                cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, cv.Scalar(255,64,255,0) )
                w_text = 70
                x += w_text+self.w_gap
            

            # display image
            if (self.params['use_gui']):
                cv2.imshow(self.window_name, imgOutput)
                cv2.waitKey(1)


    # save_background()
    # Save the current camera image as the background.
    #
    def save_background(self):
        self.imgBackground = self.imgCamera
        rospy.logwarn ('Saving new background image %s' % self.filenameBackground)
        cv2.imwrite(self.filenameBackground, self.imgBackground)
    
    
    # hit_test()
    # Get the nearest handle point or button to the mouse point.
    # ptMouse    = [x,y]
    # Returns the tag, side, and type of item the mouse has hit, using the 
    # convention that the name is of the form "tag_side", e.g. "hinge_left"
    #
    def hit_test(self, ptMouse):
        
        # Check for button press.
        iPressed = None
        for iButton in range(len(self.buttons)):
            if (self.buttons[iButton].hit_test(ptMouse)):
                iPressed = iButton
            
        if (iPressed is not None):
            nameNearest = self.buttons[iPressed].name
            type = 'button'
        else: # Find the nearest handle point.
            names = self.handlepts.keys()
            pts = np.array(self.handlepts.values())
    
            dx = np.subtract.outer(ptMouse[0], pts[:,0])
            dy = np.subtract.outer(ptMouse[1], pts[:,1])
            d = np.hypot(dx, dy)
            nameNearest = names[np.argmin(d)]
            type = 'handle'
    
        (tag,delim,side) = nameNearest.partition('_')
        
        return (tag, side, type)
        
        
    # update_handle_points()
    # Update the dictionary of handle point names and locations.
    # Compute the various handle points.
    #
    def update_handle_points (self):
        # Hinge Points.
        self.handlepts['hinge_left'] = self.wing_l.get_hinge()
        self.handlepts['hinge_right'] = self.wing_r.get_hinge()
        
        # Left High & Low Angles.
        (angle_lo_i, angle_hi_i) = self.wing_l.get_angles_i_from_b(self.wing_l.params['left']['angle_lo'], self.wing_l.params['left']['angle_hi'])
        self.handlepts['hi_left'] = (self.wing_l.get_hinge() + self.wing_l.params['left']['radius_outer'] * np.array([np.cos(np.deg2rad(angle_hi_i)), 
                                                                                                                    np.sin(np.deg2rad(angle_hi_i))])).astype(int)
        self.handlepts['lo_left'] = (self.wing_l.get_hinge() + self.wing_l.params['left']['radius_outer'] * np.array([np.cos(np.deg2rad(angle_lo_i)), 
                                                                                                                    np.sin(np.deg2rad(angle_lo_i))])).astype(int)

        # Left Inner Radius.
        angle = (angle_hi_i - angle_lo_i)/2 + angle_lo_i 
        self.handlepts['inner_left'] = (self.wing_l.get_hinge() + self.wing_l.params['left']['radius_inner'] * np.array([np.cos(np.deg2rad(angle)), 
                                                                                                                       np.sin(np.deg2rad(angle))])).astype(int)

        # Right High & Low Angles.
        (angle_lo_i, angle_hi_i) = self.wing_r.get_angles_i_from_b(self.wing_r.params['right']['angle_lo'], self.wing_r.params['right']['angle_hi'])
        self.handlepts['hi_right'] = (self.wing_r.get_hinge() + self.wing_r.params['right']['radius_outer'] * np.array([np.cos(np.deg2rad(angle_hi_i)), 
                                                                                                                     np.sin(np.deg2rad(angle_hi_i))])).astype(int)
        self.handlepts['lo_right'] = (self.wing_r.get_hinge() + self.wing_r.params['right']['radius_outer'] * np.array([np.cos(np.deg2rad(angle_lo_i)), 
                                                                                                                     np.sin(np.deg2rad(angle_lo_i))])).astype(int)

        # Right Inner Radius.
        angle = (angle_hi_i - angle_lo_i)/2 + angle_lo_i 
        self.handlepts['inner_right'] = (self.wing_r.get_hinge() + self.wing_r.params['right']['radius_inner'] * np.array([np.cos(np.deg2rad(angle)), 
                                                                                                                        np.sin(np.deg2rad(angle))])).astype(int)

        # Compute the body points
        ptR = self.wing_r.get_hinge()
        ptL = self.wing_l.get_hinge()
        ptC = (ptR+ptL)/2
        x = ptR[0] - ptL[0]
        y = ptR[1] - ptL[1]
        r = max(self.params['left']['radius_inner'], self.params['right']['radius_inner'])
        angle = np.arctan2(y,x) + np.pi/2
        self.ptBody1 = tuple((ptC+r*np.array([np.cos(angle), np.sin(angle)])).astype(int))
        self.ptBody2 = tuple((ptC-r*np.array([np.cos(angle), np.sin(angle)])).astype(int))


    
    def clip(self, x, lo, hi):
        return max(min(x,hi),lo)
    

    # Convert tag and side strings to a name string:  tag_side
    def name_from_tagside(self, tag, side):
        if (len(side)>0):
            name = tag+'_'+side
        else:
            name = tag
            
        return name
    
        
    # onMouse()
    # Handle mouse events.
    #
    def onMouse(self, event, x, y, flags, param):
        ptMouse = np.array([x, y])

        # Keep track of which UI element is selected.
        if (event==cv.CV_EVENT_LBUTTONDOWN):
            # Get the name and type nearest the current point.
            (tag, side, type) = self.hit_test(ptMouse)
            self.nameSelected = self.name_from_tagside(tag,side)
            self.tagSelected = tag
            self.sideSelected = side
            self.typeSelected = type
            self.wingSelected = self.wing_l if (self.sideSelected=='left') else self.wing_r
            
            self.wingSlave = self.wing_r if (self.sideSelected=='left') else self.wing_l
            self.sideSlave = 'right' if (self.sideSelected=='left') else 'left'
                    
            self.nameSelectedNow = self.nameSelected
            self.typeSelectedNow = self.typeSelected



        if (self.typeSelected=='button'):
            # Get the name and type nearest the current point.
            (tag, side, type) = self.hit_test(ptMouse)
            self.nameSelectedNow = self.name_from_tagside(tag,side)
            self.tagSelectedNow = tag
            self.sideSelectedNow = side
            self.typeSelectedNow = type
            
            # Set selected button to 'down', others to 'up'.
            for iButton in range(len(self.buttons)):
                if (self.nameSelected == self.nameSelectedNow == self.buttons[iButton].name) and not (event==cv.CV_EVENT_LBUTTONUP):
                    self.buttons[iButton].state = 'down'
                else:
                    self.buttons[iButton].state = 'up'


            if (event==cv.CV_EVENT_LBUTTONUP):
                # If the mouse is on the same button at mouseup, then do the action.
                if (self.typeSelectedNow=='button'):
                    if (self.nameSelected == self.nameSelectedNow == 'save bg'):
                        self.pubCommand.publish('save_background')
    
                    elif (self.nameSelected == self.nameSelectedNow == 'exit'):
                        self.pubCommand.publish('exit')
                        
                    elif (self.nameSelected == self.nameSelectedNow == 'invert'):
                        self.pubCommand.publish('invert')
                        
                    elif (self.nameSelected == self.nameSelectedNow == 'mirror'):
                        self.pubCommand.publish('mirror')

                        
        # end if (self.typeSelected=='button'):
        
                        
        elif (self.typeSelected=='handle'):
            # Adjust the handle points.
            # Hinge point.
            if (self.tagSelected=='hinge'): 
                if (self.bConstrainMaskToImage):
                    self.params[self.sideSelected]['hinge']['x'] = int(self.clip(ptMouse[0], 0+self.params[self.sideSelected]['radius_outer'], 
                                                                                 self.shapeImage[1]-self.params[self.sideSelected]['radius_outer'])) # Keep the mask onscreen left & right.
                else:
                    self.params[self.sideSelected]['hinge']['x'] = int(ptMouse[0])
                    
                self.params[self.sideSelected]['hinge']['y'] = int(ptMouse[1])


            # High angle.
            elif (self.tagSelected=='hi'): 
                pt = ptMouse - self.wingSelected.get_hinge()
                self.params[self.sideSelected]['angle_hi'] = int(self.wingSelected.transform_angle_b_from_i(np.rad2deg(np.arctan2(pt[1], pt[0]))))
                if (self.bConstrainMaskToImage):
                    self.params[self.sideSelected]['radius_outer'] = int(self.clip(np.linalg.norm(self.wingSelected.get_hinge() - ptMouse), 
                                                                        self.wingSelected.params[self.sideSelected]['radius_inner']+2,               # Outer radius > inner radius.
                                                                        min(self.shapeImage[1]-self.params[self.sideSelected]['hinge']['x'],   # Outer radius < right edge
                                                                            0+self.params[self.sideSelected]['hinge']['x'])))                  # Outer radius > left edge
                else:
                    self.params[self.sideSelected]['radius_outer'] = int(max(self.wingSelected.params[self.sideSelected]['radius_inner']+2, 
                                                                             int(np.linalg.norm(self.wingSelected.get_hinge() - ptMouse)))) # Outer radius > inner radius.
                if (self.params['mirror']):
                    self.params[self.sideSlave]['angle_hi']     = self.params[self.sideSelected]['angle_hi']
                    self.params[self.sideSlave]['radius_outer'] = self.params[self.sideSelected]['radius_outer']
                  
                  
            # Low angle.
            elif (self.tagSelected=='lo'): 
                pt = ptMouse - self.wingSelected.get_hinge()
                self.params[self.sideSelected]['angle_lo'] = int(self.wingSelected.transform_angle_b_from_i(np.rad2deg(np.arctan2(pt[1], pt[0]))))
                if (self.bConstrainMaskToImage):
                    self.params[self.sideSelected]['radius_outer'] = int(self.clip(np.linalg.norm(self.wingSelected.get_hinge() - ptMouse), 
                                                                        self.wingSelected.params[self.sideSelected]['radius_inner']+2,               # Outer radius > inner radius.
                                                                        min(self.shapeImage[1]-self.params[self.sideSelected]['hinge']['x'],   # Outer radius < right edge
                                                                            0+self.params[self.sideSelected]['hinge']['x'])))                  # Outer radius > left edge
                else:
                    self.params[self.sideSelected]['radius_outer'] = int(max(self.wingSelected.params[self.sideSelected]['radius_inner']+2, 
                                                                             int(np.linalg.norm(self.wingSelected.get_hinge() - ptMouse))))
                if (self.params['mirror']):
                    self.params[self.sideSlave]['angle_lo']     = self.params[self.sideSelected]['angle_lo']
                    self.params[self.sideSlave]['radius_outer'] = self.params[self.sideSelected]['radius_outer']
                  
                  
            # Inner radius.
            elif (self.tagSelected=='inner'): 
                self.params[self.sideSelected]['radius_inner'] = int(min(int(np.linalg.norm(self.wingSelected.get_hinge() - ptMouse)), 
                                                                         self.wingSelected.params[self.sideSelected]['radius_outer']-2))
                if (self.params['mirror']):
                    self.params[self.sideSlave]['radius_inner'] = self.params[self.sideSelected]['radius_inner']
                
                
            
            # Set the new params.
            self.wing_l.set_params(self.params)
            self.wing_r.set_params(self.params)
            self.update_handle_points()

        
            # Save the results.
            if (event==cv.CV_EVENT_LBUTTONUP):
                self.create_masks()
                rospy.set_param('strokelitude', self.params)
            
        # end if (self.typeSelected=='handle'):
            

        if (event==cv.CV_EVENT_LBUTTONUP):
            self.nameSelected = None
            self.typeSelected = None
            self.nameSelectedNow = None
            self.typeSelectedNow = None


            
                
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
    rospy.logwarn('*    Left click+drag to move any handle points.                         *')
    rospy.logwarn('*                                                                        *')  
    rospy.logwarn('*                                                                        *') 
    rospy.logwarn('**************************************************************************')
    rospy.logwarn('')
    rospy.logwarn('')

    main = MainWindow()
    main.run()
