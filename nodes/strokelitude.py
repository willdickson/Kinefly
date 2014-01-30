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
from std_msgs.msg import Float32, Header, String
from StrokelitudeROS.srv import *
from StrokelitudeROS.msg import float32list as float32list_msg
from StrokelitudeROS.msg import MsgFlystate, MsgWing, MsgBodypart
from StrokelitudeROS.cfg import strokelitudeConfig



# Colors.
bgra_dict = {'red'           : cv.Scalar(0,0,255,0),
             'green'         : cv.Scalar(0,255,0,0), 
             'blue'          : cv.Scalar(255,0,0,0),
             'cyan'          : cv.Scalar(255,255,0,0),
             'magenta'       : cv.Scalar(255,0,255,0),
             'yellow'        : cv.Scalar(0,255,255,0),
             'black'         : cv.Scalar(0,0,0,0),
             'white'         : cv.Scalar(255,255,255,0),
             'dark_gray'     : cv.Scalar(64,64,64,0),
             'gray'          : cv.Scalar(128,128,128,0),
             'light_gray'    : cv.Scalar(192,192,192,0),
             'light_green'   : cv.Scalar(175,255,175,0),
             }


def get_angle_from_points_i(pt1, pt2):
    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]
    return np.arctan2(y,x)


def filter_median(data):
    data2 = copy.copy(data)
    q = 1       # q is the 'radius' of the filter window.  q==1 is a window of 3.  q==2 is a window of 5.
    for i in range(q,len(data)-q):
        data2[i] = np.median(data[i-q:i+q+1]) # Median filter of window.

    # Left-fill the first values.
    data2[0:q] = data2[q]

    # Right-fill the last values.
    data2[len(data2)-q:len(data2)] = data2[-(q+1)]

    return data2
        

def clip(x, lo, hi):
    return max(min(x,hi),lo)
    

        
###############################################################################
###############################################################################
class Button:
    def __init__(self, name=None, text=None, pt=None, rect=None, scaleText=0.4):
        self.name = name
        self.pt = pt
        self.rect = rect
        self.scaleText = scaleText
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

        self.left   = self.ptLT0[0]
        self.top    = self.ptLT0[1]
        self.right  = self.ptRB0[0]
        self.bottom = self.ptRB0[1]
        


    def hit_test(self, ptMouse):
        if (self.rect[0] <= ptMouse[0] <= self.rect[0]+self.rect[2]) and (self.rect[1] <= ptMouse[1] <= self.rect[1]+self.rect[3]):
            return True
        else:
            return False
        

    def set_text(self, text):
        self.text = text
        (self.sizeText,rv) = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, self.scaleText, 1)
        if (self.rect is not None):
            pass
        elif (self.pt is not None):
            self.rect = [0,0,0,0]
            self.rect[0] = self.pt[0]
            self.rect[1] = self.pt[1]
            self.rect[2] = self.sizeText[0] + 6
            self.rect[3] = self.sizeText[1] + 6
        else:
            rospy.logwarn('Error creating Button().')

        self.ptCenter = (int(self.rect[0]+self.rect[2]/2), int(self.rect[1]+self.rect[3]/2))
        self.ptText = (self.ptCenter[0] - int(self.sizeText[0]/2) - 1, 
                       self.ptCenter[1] + int(self.sizeText[1]/2) - 1)

                
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

        cv2.putText(image, self.text, ptText0, cv2.FONT_HERSHEY_SIMPLEX, self.scaleText, colorText)
        
# end class Button                
    
            
###############################################################################
###############################################################################
class Handle:
    def __init__(self, pt=np.array([0,0])):
        self.pt = pt

        self.bgra = bgra_dict['white']
        self.radius = 4


    def hit_test(self, ptMouse):
        d = np.linalg.norm(self.pt - ptMouse)
        if (d < self.radius+2):
            return True
        else:
            return False
        

    # draw()
    # Draw a handle.
    # 
    def draw(self, image):
        cv2.circle(image, tuple(self.pt.astype(int)),  self.radius, self.bgra, cv.CV_FILLED)
        
# end class Handle
                

###############################################################################
###############################################################################
class Fly(object):
    def __init__(self, params={}):
        self.head    = Bodypart(name='head',    params=params, color='cyan') 
        self.abdomen = Bodypart(name='abdomen', params=params, color='magenta') 
        self.wing_r  = Wing(side='right',       params=params, color='red')
        self.wing_l  = Wing(side='left',        params=params, color='green')
        self.bgra_body = bgra_dict['light_gray']
        self.ptBody1 = None
        self.ptBody2 = None
        self.iCount  = 0
        self.stamp   = 0

        self.pubFlystate = rospy.Publisher('strokelitude/flystate', MsgFlystate)


    def create_masks(self, shapeImage):
        self.head.create_mask (shapeImage)
        
        self.abdomen.create_mask (shapeImage)
        
        self.wing_r.create_stroke_mask (shapeImage)
        self.wing_r.create_angle_mask (shapeImage)
        self.wing_r.assign_pixels_to_bins ()
        
        self.wing_l.create_stroke_mask (shapeImage)
        self.wing_l.create_angle_mask (shapeImage)
        self.wing_l.assign_pixels_to_bins ()


    def set_params(self, params):
        self.head.set_params(params)
        self.abdomen.set_params(params)
        self.wing_l.set_params(params)
        self.wing_r.set_params(params)

        self.angleBody = self.get_bodyangle_i()
        self.ptBodyCenter = (np.array([params['head']['x'], params['head']['y']]) + np.array([params['abdomen']['x'], params['abdomen']['y']])) / 2
        r = max(params['left']['radius_outer'], params['right']['radius_outer'])
        self.ptBody1 = tuple((self.ptBodyCenter + r * np.array([np.cos(self.angleBody), np.sin(self.angleBody)])).astype(int))
        self.ptBody2 = tuple((self.ptBodyCenter - r * np.array([np.cos(self.angleBody), np.sin(self.angleBody)])).astype(int))
    
            
    def get_bodyangle_i(self):
        angle_i = get_angle_from_points_i(self.head.ptCenter_i, self.abdomen.ptCenter_i)
         
        return angle_i
        
                
    def update_handle_points(self):
        self.head.update_handle_points()
        self.abdomen.update_handle_points()
        self.wing_l.update_handle_points()
        self.wing_r.update_handle_points()
        

    def update(self, image, header=None):
        if (header is not None):
            self.stamp = header.stamp
        else:
            self.stamp = 0
        
        self.head.update(image)
        self.abdomen.update(image)
        self.wing_l.update(image)
        self.wing_r.update(image)
    
            
    def draw(self, image):
        # Draw line to indicate the body.
        #cv2.line(image, tuple(self.head.ptCenter_i.astype(int)), tuple(self.abdomen.ptCenter_i.astype(int)), self.bgra_body, 1)
        cv2.line(image, self.ptBody1, self.ptBody2, self.bgra_body, 1) # Draw a line longer than just head-to-abdomen.
                
        self.head.draw(image)
        self.abdomen.draw(image)
        self.wing_l.draw(image)
        self.wing_r.draw(image)

        
    
    def publish(self):
        pt = self.head.ptCenter_i - self.head.ptHinge_i + self.head.ptCOM # The head COM point relative to the hinge.
        angleHead = -(np.arctan2(pt[1], pt[0]) - self.angleBody + np.pi/2)
        angleHead = (angleHead + np.pi) % (2*np.pi) - np.pi
        radiusHead = np.linalg.norm(pt)
        
        pt = self.abdomen.ptCenter_i - self.abdomen.ptHinge_i + self.abdomen.ptCOM # The abdomen COM point relative to the abdomen hinge.
        angleAbdomen = -(np.arctan2(pt[1], pt[0]) - self.angleBody + np.pi/2)
        angleAbdomen = (angleAbdomen + np.pi) % (2*np.pi) - np.pi
        radiusAbdomen = np.linalg.norm(pt)
        
        flystate              = MsgFlystate()
        flystate.header       = Header(seq=self.iCount, stamp=self.stamp, frame_id='Fly')
        flystate.left         = MsgWing(mass=self.wing_l.mass, angle1=self.wing_l.angle_leading_b, angle2=self.wing_l.angle_trailing_b)
        flystate.right        = MsgWing(mass=self.wing_r.mass, angle1=self.wing_r.angle_leading_b, angle2=self.wing_r.angle_trailing_b)
        flystate.head         = MsgBodypart(mass   = self.head.mass,    
                                            radius = radiusHead,    
                                            angle  = angleHead)
        flystate.abdomen      = MsgBodypart(mass   = self.abdomen.mass, 
                                            radius = radiusAbdomen, 
                                            angle  = angleAbdomen)
        self.iCount += 1
        
        self.pubFlystate.publish(flystate)
        

# end class Fly

        
###############################################################################
###############################################################################
# Head or Abdomen.
class Bodypart(object):
    def __init__(self, name=None, params={}, color='white'):
        self.name = name

        self.bgra     = bgra_dict[color]
        self.bgra_com = bgra_dict['red']
        self.pixelmax = 255.

        self.handles = {'center':Handle(np.array([0,0])),
                        'radius_minor':Handle(np.array([0,0])),
                        'radius_major':Handle(np.array([0,0]))
                        }

        self.set_params(params)
        self.mass = 0.0
        self.ptCenter_i = np.array([0,0])
        self.ptCOM = np.array([0,0])

        self.pubAngle = rospy.Publisher('strokelitude/'+self.name+'/angle', Float32)


    # set_params()
    # Set the given params dict into this object.
    #
    def set_params(self, params):
        self.params  = params
        self.angleBody = self.get_bodyangle_i()
        self.cosBodyangle = np.cos(self.angleBody)
        self.sinBodyangle = np.sin(self.angleBody)
        self.ptCenter_i = np.array([self.params[self.name]['x'], self.params[self.name]['y']])
        
        # Compute the hinge location, which is on the intersection of the bodypart ellipse and the body axis.
        ptBodyCenter = (np.array([self.params['head']['x'], self.params['head']['y']]) + np.array([self.params['abdomen']['x'], self.params['abdomen']['y']])) / 2
        r = self.params[self.name]['radius_major']
        ptHinge1 = (self.ptCenter_i + r*np.array([np.cos(self.angleBody), np.sin(self.angleBody)]))
        ptHinge2 = (self.ptCenter_i - r*np.array([np.cos(self.angleBody), np.sin(self.angleBody)]))
        r1 = np.linalg.norm(ptHinge1 - ptBodyCenter)
        r2 = np.linalg.norm(ptHinge2 - ptBodyCenter)
        self.ptHinge_i = ptHinge1 if (r1<r2) else ptHinge2 

        # Refresh the handle points.
        self.update_handle_points()
        
        
    def get_bodyangle_i(self):
        angle_i = get_angle_from_points_i(np.array([self.params['head']['x'], self.params['head']['y']]), 
                                          np.array([self.params['abdomen']['x'], self.params['abdomen']['y']]))
         
        return angle_i
        
                
    # create_mask()
    # Create an image mask.
    #
    def create_mask(self, shape):
        # Create the mask.
        self.imgMask = np.zeros(shape, dtype=np.uint8)
        cv2.ellipse(self.imgMask,
                    (int(self.params[self.name]['x']), int(self.params[self.name]['y'])),
                    (int(self.params[self.name]['radius_major']), int(self.params[self.name]['radius_minor'])),
                    np.rad2deg(self.angleBody),
                    0,
                    360,
                    bgra_dict['white'], 
                    cv.CV_FILLED)
        
        self.ravelMask = np.ravel(self.imgMask)
        
        
    # update_handle_points()
    # Update the dictionary of handle point names and locations.
    # Compute the various handle points.
    #
    def update_handle_points (self):
        self.handles['center'].pt = np.array([self.params[self.name]['x'], self.params[self.name]['y']])
        self.handles['radius_major'].pt = np.array([self.params[self.name]['x'], self.params[self.name]['y']]) + (self.params[self.name]['radius_major'] * np.array([self.cosBodyangle,self.sinBodyangle]))
                                            
        self.handles['radius_minor'].pt = np.array([self.params[self.name]['x'], self.params[self.name]['y']]) + (self.params[self.name]['radius_minor'] * np.array([-self.sinBodyangle,self.cosBodyangle]))
        
        
    # update_state_from_image()
    # Update the bodypart state from the masked image.
    #
    def update_state_from_image(self, image):
        imgMasked = cv2.bitwise_and(image, self.imgMask)
        moments = cv2.moments(imgMasked, binaryImage=False)
        
        if (moments['m00'] != 0.0):
            self.mass  = moments['m00'] / self.pixelmax
            self.ptCOM = np.array([moments['m10']/moments['m00'] - self.params[self.name]['x'], 
                                   moments['m01']/moments['m00'] - self.params[self.name]['y']])
        else:
            self.mass = 0.0
            self.ptCOM = np.array([0,0])
            
    
    
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, image):
        self.update_state_from_image(image)
    
    
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
        # Draw the handle points.
        for tagHandle,handle in self.handles.iteritems():
            handle.draw(image)

    
    # draw()
    # Draw the outline.
    #
    def draw(self, image):
        # Draw the outline.
        cv2.ellipse(image,
                    (int(self.params[self.name]['x']), int(self.params[self.name]['y'])),
                    (int(self.params[self.name]['radius_major']), int(self.params[self.name]['radius_minor'])),
                    np.rad2deg(self.angleBody),
                    0,
                    360,
                    self.bgra, 
                    1)

        self.draw_handles(image)

        # Draw the bodypart center of mass.        
        ptCOM_i = (int(self.ptCOM[0]+self.params[self.name]['x']), int(self.ptCOM[1]+self.params[self.name]['y'])) 
        cv2.ellipse(image,
                    ptCOM_i,
                    (2,2),
                    0,
                    0,
                    360,
                    self.bgra_com, 
                    1)
        
        # Draw the bodypart hinge.        
        ptHinge_i = (int(self.ptHinge_i[0]), int(self.ptHinge_i[1])) 
        cv2.ellipse(image,
                    ptHinge_i,
                    (2,2),
                    0,
                    0,
                    360,
                    self.bgra, 
                    1)
        cv2.line(image, ptHinge_i, ptCOM_i, self.bgra_com, 1)
        
# end class Bodypart

    

###############################################################################
###############################################################################
class Wing(object):
    def __init__(self, side='right', params={}, color='white'):
        self.side = side
        
        self.ravelMaskRoi          = None
        self.ravelMaskAnglesRoi_b  = None

        self.bins                  = None
        self.intensities           = None
        self.binsValid             = None
        self.intensitiesValid      = None
        self.imgTest = None
        
        # Bodyframe angles have zero degrees point orthogonal to body axis: 
        # If body axis is north/south, then 0-deg is east for right wing, west for left wing.
        
        # Imageframe angles are oriented to the image.  0-deg is east, +-pi is west, -pi/2 is north, +pi/2 is south.
        
        
        self.angle_trailing_b = None
        self.angle_leading_b  = None
        self.angle_amplitude  = None
        self.angle_mean       = None
        self.mass             = 0.0
        self.bFlying          = False
        
        self.pixelmax         = 255.
        self.bgra             = bgra_dict[color]
        self.thickness_inner  = 1
        self.thickness_outer  = 1
        self.thickness_wing   = 1
        
        self.handles = {'hinge':Handle(np.array([0,0])),
                        'hi':Handle(np.array([0,0])),
                        'lo':Handle(np.array([0,0])),
                        'inner':Handle(np.array([0,0]))
                        }

        self.set_params(params)

        # services, for live histograms
        self.service_histogram = rospy.Service('wing_histogram_'+side, float32list, self.serve_histogram_callback)
        self.service_bins      = rospy.Service('wing_bins_'+side, float32list, self.serve_bins_callback)
        self.service_edges     = rospy.Service('wing_edges_'+side, float32list, self.serve_edges_callback)
        
    
    # get_angles_i_from_b()
    # Return angle1 and angle2 oriented to the image rather than the fly.
    # * corrected for left/right full-circle angle, i.e. east is 0-deg, west is 270-deg.
    # * corrected for wrapping at delta>np.pi.
    #
    def get_angles_i_from_b(self, angle_lo_b, angle_hi_b):
        angle_lo_i = self.transform_angle_i_from_b(angle_lo_b)
        angle_hi_i = self.transform_angle_i_from_b(angle_hi_b)
        
        if (angle_hi_i-angle_lo_i > np.pi):
            angle_hi_i -= (2*np.pi)
        if (angle_lo_i-angle_hi_i > np.pi):
            angle_lo_i -= (2*np.pi)
            
        return (float(angle_lo_i), float(angle_hi_i))
    
    
    # transform_angle_i_from_b()
    # Transform an angle from the fly frame to the camera image frame.
    #
    def transform_angle_i_from_b(self, angle_b):
        if self.side == 'right':
            angle_i  = self.angleBody + angle_b
        else: # left
            angle_i  = self.angleBody - angle_b + np.pi
             
        angle_i = (angle_i+np.pi) % (2*np.pi) - np.pi
        return angle_i
        

    # transform_angle_b_from_i()
    # Transform an angle from the camera image frame to the fly frame.
    #
    def transform_angle_b_from_i(self, angle_i):
        if self.side == 'right':
            angle_b  = angle_i - self.angleBody
        else:  
            angle_b  = self.angleBody - angle_i + np.pi 

        angle_b = (angle_b+np.pi) % (2*np.pi) - np.pi
        return angle_b
        

    # set_params()
    # Set the given params dict into this object.  Any member vars that come from params should be set here.
    #
    def set_params(self, params):
        self.params    = params
        self.ptHinge   = np.array([self.params[self.side]['hinge']['x'], self.params[self.side]['hinge']['y']])
        nbins          = int((2*np.pi)/self.params['resolution_radians']) + 1
        self.bins      = np.linspace(-np.pi, np.pi, nbins)
        self.angleBody = self.get_bodyangle_i()

        (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi'])
        angle_mid_i              = (angle_hi_i - angle_lo_i)/2 + angle_lo_i
         
        self.cos = {}
        self.sin = {}
        
        self.cos['hi']  = float(np.cos(angle_hi_i)) 
        self.sin['hi']  = float(np.sin(angle_hi_i))
        self.cos['mid'] = float(np.cos(angle_mid_i))
        self.sin['mid'] = float(np.sin(angle_mid_i))
        self.cos['lo']  = float(np.cos(angle_lo_i))
        self.sin['lo']  = float(np.sin(angle_lo_i))

        angle_lo_b = min([self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi']])
        angle_hi_b = max([self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi']])
        
        self.iValidBins = np.where((angle_lo_b < self.bins) * (self.bins < angle_hi_b))[0]
        
        self.update_handle_points()
        
        
    def get_bodyangle_i(self):
        angle_i = get_angle_from_points_i(np.array([self.params['head']['x'], self.params['head']['y']]), 
                                          np.array([self.params['abdomen']['x'], self.params['abdomen']['y']]))
         
        return angle_i - np.pi/2 
        
                
    # create_angle_mask()
    # Create an image where each pixel value is the angle from the hinge.                    
    # 
    def create_angle_mask(self, shape):
        # Set up matrices of x and y coordinates.
        x = np.tile(np.array([range(shape[1])])   - self.ptHinge[0], (shape[0], 1))
        y = np.tile(np.array([range(shape[0])]).T - self.ptHinge[1], (1, shape[1]))
        
        # Calc their angles.
        angles_i = np.arctan2(y,x)
        anglesRoi_i = angles_i[self.roiMask[1]:self.roiMask[3], self.roiMask[0]:self.roiMask[2]]
        
        imgAnglesRoi_b  = self.transform_angle_b_from_i(anglesRoi_i)

        self.ravelMaskAnglesRoi_b = np.ravel(imgAnglesRoi_b)
                   

    # create_stroke_mask()
    # Create a mask of valid wingstroke areas.
    #
    def create_stroke_mask(self, shape):
        # Create the wing mask.
        imgMask = np.zeros(shape, dtype=np.uint8)
        (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi'])

        ptCenter = tuple(self.ptHinge.astype(int))
        cv2.ellipse(imgMask,
                    ptCenter,
                    (int(self.params[self.side]['radius_outer']), int(self.params[self.side]['radius_outer'])),
                    0,
                    np.rad2deg(angle_lo_i),
                    np.rad2deg(angle_hi_i),
                    255, 
                    cv.CV_FILLED)
        cv2.ellipse(imgMask,
                    ptCenter,
                    (int(self.params[self.side]['radius_inner']), int(self.params[self.side]['radius_inner'])),
                    0,
                    np.rad2deg(angle_lo_i),
                    np.rad2deg(angle_hi_i),
                    0, 
                    cv.CV_FILLED)
        

        # Find the ROI of the mask.
        xSum = np.sum(imgMask, 0)
        ySum = np.sum(imgMask, 1)
        xMin = np.where(xSum>0)[0][0]
        xMax = np.where(xSum>0)[0][-1]
        yMin = np.where(ySum>0)[0][0]
        yMax = np.where(ySum>0)[0][-1]
        
        self.roiMask = np.array([xMin, yMin, xMax, yMax])
        self.imgMaskRoi = imgMask[yMin:yMax, xMin:xMax]

        
        self.ravelMaskRoi = np.ravel(self.imgMaskRoi)
        
        
    # assign_pixels_to_bins()
    # Create two lists, one containing the pixel indices for each bin, and the other containing the intensities (mean pixel values).
    #
    def assign_pixels_to_bins(self):
        # Create empty bins.
        self.pixelsRoi = [[] for i in range(len(self.bins))]
        self.intensities = np.zeros(len(self.bins))

        # Put each pixel into an appropriate bin.            
        for iPixel, angle in enumerate(self.ravelMaskAnglesRoi_b):
            if self.ravelMaskRoi[iPixel]:
                iBinBest = np.argmin(np.abs(self.bins - angle))
                self.pixelsRoi[iBinBest].append(iPixel)
                
         
    # update_bin_intensities()
    # Update the list of intensities corresponding to the bin angles.
    #            
    def update_bin_intensities(self, image):
        if (self.ravelMaskRoi is not None) and (self.intensities is not None):
            # Only use the ROI.
            imgRoi = image[self.roiMask[1]:self.roiMask[3], self.roiMask[0]:self.roiMask[2]]

            # Apply the mask.
            imgMasked = cv2.bitwise_and(imgRoi, self.imgMaskRoi)            
            ravelImageMasked = np.ravel(imgMasked)
            self.imgTest = imgMasked
            
            # Get the pixel mass.
            self.mass = np.sum(ravelImageMasked) / self.pixelmax
            
            # Compute the stroke intensity function.
            for iBin in self.iValidBins:
                iPixels = self.pixelsRoi[iBin]
                if (len(iPixels) > 0):
                    pixels = ravelImageMasked[iPixels]                          # TODO: This line is the main cause of slow framerate.
                    #pixels = np.zeros(len(iPixels))                            # Compare with this line.
                    self.intensities[iBin] = np.sum(pixels) / len(iPixels) / self.pixelmax
                else:
                    self.intensities[iBin] = 0.0
             
            self.binsValid        = self.bins[self.iValidBins]
            self.intensitiesValid = self.intensities[self.iValidBins]

                        
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
                    (self.angle_leading_b, self.angle_trailing_b) = self.get_edge_angles()
                    self.angle_amplitude = np.abs(self.angle_leading_b - self.angle_trailing_b)
                    self.angle_mean = np.mean([self.angle_leading_b, self.angle_trailing_b])
                    
            else: # not flying
                self.angle_leading_b  = np.pi/2
                self.angle_trailing_b = np.pi/2
                self.angle_amplitude = 0.
                self.angle_mean = 0.


    def update_flight_status(self):
        if (self.mass > self.params['threshold_flight']):
            self.bFlying = True
        else:
            self.bFlying = False
    
    
    # update_handle_points()
    # Update the dictionary of handle point names and locations.
    # Compute the various handle points.
    #
    def update_handle_points (self):
        # Hinge Points.
        self.handles['hinge'].pt = self.ptHinge
        
        
        # High & Low Angles.
        (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi'])
        self.handles['hi'].pt = (self.ptHinge + self.params[self.side]['radius_outer'] * np.array([self.cos['hi'], 
                                                                                                          self.sin['hi']]))
        self.handles['lo'].pt = (self.ptHinge + self.params[self.side]['radius_outer'] * np.array([self.cos['lo'], 
                                                                                                          self.sin['lo']]))

        # Inner Radius.
        self.handles['inner'].pt = (self.ptHinge + self.params[self.side]['radius_inner'] * np.array([self.cos['mid'], 
                                                                                                             self.sin['mid']]))


    
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, image):
        self.update_bin_intensities(image)
        self.update_edge_stats()
        self.update_flight_status()

    
    # get_edge_angles()
    # Get the angles of the two wing edges.
    #                    
    def get_edge_angles(self):
        diff = filter_median(np.diff(self.intensitiesValid))
            
        iPeak = np.argmax(self.intensitiesValid)

        # Major and minor edges, must have opposite signs.
        iMax = np.argmax(diff)
        iMin = np.argmin(diff)
        (iMajor,iMinor) = (iMax,iMin) if (np.abs(diff[iMax]) > np.abs(diff[iMin])) else (iMin,iMax) 

        iEdge1 = iMajor
        
        # The minor edge must be at least 3/4 the strength of the major edge to be used, else use the end of the array.
        #if (3*np.abs(diff[iMajor])/4 < np.abs(diff[iMinor])):
        if (self.params['n_edges']==2):
            iEdge2 = iMinor
        elif (self.params['n_edges']==1):
            if (self.params['flipedge']):
                iEdge2 = 0              # Front edge.
            else:
                iEdge2 = -1             # Back edge.
        else:
            rospy.logwarn('Parameter n_edges must be 1 or 2.')
        
        # Convert the edge index to an angle.
        angle1 = float(self.binsValid[iEdge1])
        angle2 = float(self.binsValid[iEdge2])
        
        return (angle1, angle2)
        

    # hit_object()
    # Get the UI object, if any, that the mouse is on.    
    def hit_object(self, ptMouse):
        tag = None
        
        # Check for handle hits.
        for tagHandle,handle in self.handles.iteritems():
            if (handle.hit_test(ptMouse)):
                tag = tagHandle
                break
                
        return (self.side, tag)
    

    def draw_handles(self, image):
        # Draw the handle points.
        for tagHandle,handle in self.handles.iteritems():
            handle.draw(image)

    
    # draw()
    # Draw the wing envelope and leading and trailing edges, onto the given image.
    #
    def draw(self, image):
        if self.ptHinge is not None:
            (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi'])
            ptHinge = tuple(self.ptHinge.astype(int))

            # Inner circle
            cv2.ellipse(image, 
                        ptHinge,  
                        (int(self.params[self.side]['radius_inner']), int(self.params[self.side]['radius_inner'])),
                        0,
                        np.rad2deg(angle_lo_i),
                        np.rad2deg(angle_hi_i),
                        color=self.bgra,
                        thickness=self.thickness_inner,
                        )
            
            # Outer circle         
            cv2.ellipse(image, 
                        ptHinge,  
                        (int(self.params[self.side]['radius_outer']), int(self.params[self.side]['radius_outer'])),
                        0,
                        np.rad2deg(angle_lo_i),
                        np.rad2deg(angle_hi_i),
                        color=self.bgra,
                        thickness=self.thickness_outer,
                        )
            
            
            # Leading and trailing edges
            if (self.angle_leading_b is not None):
                (angle_leading_i, angle_trailing_i) = self.get_angles_i_from_b(self.angle_leading_b, self.angle_trailing_b)
    
                x = self.ptHinge[0] + self.params[self.side]['radius_outer'] * np.cos(angle_trailing_i)
                y = self.ptHinge[1] + self.params[self.side]['radius_outer'] * np.sin(angle_trailing_i)
                cv2.line(image, ptHinge, (int(x),int(y)), self.bgra, self.thickness_wing)
                
                x = self.ptHinge[0] + self.params[self.side]['radius_outer'] * np.cos(angle_leading_i)
                y = self.ptHinge[1] + self.params[self.side]['radius_outer'] * np.sin(angle_leading_i)
                cv2.line(image, ptHinge, (int(x),int(y)), self.bgra, self.thickness_wing)
                
        self.draw_handles(image)

                        
    def serve_bins_callback(self, request):
        if (self.binsValid is not None):
            #return float32listResponse(self.bins)
            return float32listResponse(self.binsValid[:-1])
            
    def serve_histogram_callback(self, request):
        if (self.intensitiesValid is not None):
            #return float32listResponse(self.intensities)
            #return float32listResponse(np.diff(self.intensitiesValid))
            return float32listResponse(filter_median(np.diff(self.intensitiesValid)))
            
    def serve_edges_callback(self, request):
        if (self.angle_trailing_b is not None):            
            return float32listResponse([self.angle_trailing_b, self.angle_leading_b])
            

# end class Wing

            
            
###############################################################################
###############################################################################
class MainWindow:

    class struct:
        pass
    
    def __init__(self):
        self.bInitialized = False

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
                    'invertcolor':False,
                    'flipedge':False,
                    'symmetric':True,
                    'resolution_radians':0.0174532925,  # 1.0 degree == 0.0174532925 radians.
                    'threshold_flight':0.1,
                    'scale_image':1.0,
                    'n_edges':1,
                    'right':{'hinge':{'x':300,
                                      'y':100},
                             'radius_outer':30,
                             'radius_inner':10,
                             'angle_hi':0.7854, 
                             'angle_lo':-0.7854
                             },

                    'left':{'hinge':{'x':100,
                                     'y':100},
                            'radius_outer':30,
                            'radius_inner':10,
                            'angle_hi':0.7854, 
                            'angle_lo':-0.7854
                            },
                    'head':{'x':300,
                            'y':150,
                            'radius_minor':50,
                            'radius_major':50},
                    'abdomen':{'x':300,
                            'y':250,
                            'radius_minor':60,
                            'radius_major':70},
                    }
        self.set_dict_with_preserve(self.params, defaults)
        
        # Background image.
        self.filenameBackground = os.path.expanduser(self.params['filenameBackground'])
        self.imgBackground  = cv2.imread(self.filenameBackground, cv.CV_LOAD_IMAGE_GRAYSCALE)
        
        self.scale = self.params['scale_image']
        self.params = self.scale_params(self.params, self.scale)
        
        # initialize wings and body
        self.fly = Fly(self.params)
        
        self.nameSelected = None
        self.uiSelected = None
        self.fly.update_handle_points()

        # Publishers.
        self.pubCommand            = rospy.Publisher('strokelitude/command', String)

        # Subscriptions.        
        self.subImageRaw           = rospy.Subscriber(self.params['image_topic'], Image, self.image_callback)
        self.subCommand            = rospy.Subscriber('strokelitude/command', String, self.command_callback)

        self.w_gap = int(30 * self.scale)
        self.scaleText = 0.4 * self.scale
        self.fontface = cv2.FONT_HERSHEY_SIMPLEX

        # UI button specs.
        self.buttons = []
        x = int(10 * self.scale)
        y = int(10 * self.scale)
        w = int(40 * self.scale)
        h = int(18 * self.scale)
        btn = Button(name='exit', 
                   text='exit', 
                   pt=[x,y],
                   scaleText=self.scaleText)
        self.buttons.append(btn)
        
        x += int((w+2) * self.scale)
        w = int(65 * self.scale)
        x = btn.right+1
        btn = Button(name='save bg', 
                   text='saveBG', 
                   pt=[x,y],
                   scaleText=self.scaleText)
        self.buttons.append(btn)
        
        x += int((w+2) * self.scale)
        w = int(95 * self.scale)
        x = btn.right+1
        btn = Button(name='invertcolor', 
                   text='inverted', # Instantiate with the longest possible text. 
                   pt=[x,y],
                   scaleText=self.scaleText)
        btn.set_text('normal' if (not self.params['invertcolor']) else 'inverted') 
        self.buttons.append(btn)

        x += int((w+2) * self.scale)
        w = int(90 * self.scale)
        x = btn.right+1
        btn = Button(name='flipedge', 
                   text='edge2',  # Instantiate with the longest possible text.
                   pt=[x,y],
                   scaleText=self.scaleText)
        btn.set_text('edge1' if (self.params['flipedge']) else 'edge2')
        self.buttons.append(btn)

        x += int((w+2) * self.scale)
        w = int(90 * self.scale)
        x = btn.right+1
        btn = Button(name='flipsymmetry', 
                   text='asymmetric', 
                   pt=[x,y],
                   scaleText=self.scaleText)
        btn.set_text('symmetric' if (self.params['symmetric']) else 'asymmetric')
        self.buttons.append(btn)


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
        self.fly.set_params(self.params)
        
        return config


    # command_callback()
    # Execute any commands sent over the command topic.
    #
    def command_callback(self, command):
        self.command = command.data
        
        if (self.command == 'exit'):
            rospy.signal_shutdown('User requested exit.')
        
        
        if (self.command == 'flipedge'):
            self.params['flipedge'] = not self.params['flipedge']
            
            for button in self.buttons:
                if (button.name=='flipedge'):
                    button.set_text('edge1' if (self.params['flipedge']) else 'edge2')
                    
            
        if (self.command == 'flipsymmetry'):
            self.params['symmetric'] = not self.params['symmetric']
            
            for button in self.buttons:
                if (button.name=='flipsymmetry'):
                    button.set_text('symmetric' if (self.params['symmetric']) else 'asymmetric')
                    
            
        if (self.command == 'invertcolor'):
            self.params['invertcolor'] = not self.params['invertcolor']
            
            for button in self.buttons:
                if (button.name=='invertcolor'):
                    button.set_text('normal' if (not self.params['invertcolor']) else 'inverted')
                    
            
        if (self.command == 'save_background'):
            self.save_background()
            
        
        if (self.command == 'help'):
            rospy.logwarn('The strokelitude/command topic accepts the following string commands:')
            rospy.logwarn('  help                 This message.')
            rospy.logwarn('  flipedge             Toggle which wing edge to use as default.')
            rospy.logwarn('  flipsymmetry         Toggle symmetry when mousing the handles.')
            rospy.logwarn('  invertcolor          Toggle blackonwhite or whiteonblack.')
            rospy.logwarn('  save_background      Save the instant camera image to disk for')
            rospy.logwarn('                       background subtraction.')
            rospy.logwarn('  exit                 Exit the program.')
            rospy.logwarn('')
            rospy.logwarn('You can send the above commands at the shell prompt via:')
            rospy.logwarn('rostopic pub -1 strokelitude/command std_msgs/String commandtext')
            rospy.logwarn('')
            rospy.logwarn('You may also set some parameters via ROS dynamic_reconfigure, all others')
            rospy.logwarn('are settable as launch-time parameters.')
            rospy.logwarn('')

        self.fly.set_params(self.params)
        rospy.set_param('strokelitude', self.params)
    
        
    def scale_params(self, paramsIn, scale):
		paramsOut = copy.deepcopy(paramsIn)
		for wing in ['left', 'right']:
			paramsOut[wing]['hinge']['x'] = int(paramsIn[wing]['hinge']['x']*scale)  
			paramsOut[wing]['hinge']['y'] = int(paramsIn[wing]['hinge']['y']*scale)  
			paramsOut[wing]['radius_outer'] = int(paramsIn[wing]['radius_outer']*scale)  
			paramsOut[wing]['radius_inner'] = int(paramsIn[wing]['radius_inner']*scale)  

		for bodypart in ['head', 'abdomen']:
			paramsOut[bodypart]['x'] = int(paramsIn[bodypart]['x']*scale) 
			paramsOut[bodypart]['y'] = int(paramsIn[bodypart]['y']*scale)  
			paramsOut[bodypart]['radius_minor'] = int(paramsIn[bodypart]['radius_minor']*scale)  
			paramsOut[bodypart]['radius_major'] = int(paramsIn[bodypart]['radius_major']*scale)
			
		return paramsOut  
	
	
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


    # Draw user-interface elements on the image.
    def draw_buttons(self, image):
        for i in range(len(self.buttons)):
            self.buttons[i].draw(image)


    def image_callback(self, rosimage):
        # Receive an image:
        try:
            if (not self.params['invertcolor']):
                img = np.uint8(cv.GetMat(self.cvbridge.imgmsg_to_cv(rosimage, 'passthrough')))
            else:
                img = 255-np.uint8(cv.GetMat(self.cvbridge.imgmsg_to_cv(rosimage, 'passthrough')))
            
        except CvBridgeError, e:
            rospy.logwarn ('Exception converting background image from ROS to opencv:  %s' % e)
            self.imgCamera = None
            
         
        if (self.scale == 1.0):              
        	self.imgCamera = img
        else:  
        	self.imgCamera = cv2.resize(img, (0,0), fx=self.scale, fy=self.scale) 
                
        if (self.imgCamera is not None):
            # Background subtraction.
            if (self.imgBackground is not None):
                try:
                    imgForeground = cv2.absdiff(self.imgCamera, self.imgBackground)
                except:
                    imgForeground = self.imgCamera
                    self.imgBackground = None
                    rospy.logwarn('Please take a fresh background image.  The existing one is the wrong size or has some other problem.')
                    
            else:
                imgForeground = self.imgCamera
                
                
            self.shapeImage = self.imgCamera.shape # (height,width)
            
            if (not self.bInitialized):
                self.fly.create_masks(self.shapeImage)
                self.bInitialized = True
                                
            if (self.params['use_gui']):
                imgOutput = cv2.cvtColor(imgForeground, cv.CV_GRAY2RGB)
                self.fly.draw(imgOutput)
                self.draw_buttons(imgOutput)
            
                x_left   = int(10 * self.scale)
                y_bottom = int(imgOutput.shape[0] - 10 * self.scale)
                x_right  = int(imgOutput.shape[1] - 10 * self.scale)
                x = x_left

                if (self.fly.wing_l.angle_amplitude is not None):
                    #s = 'L:% 7.1f' % np.rad2deg(self.fly.wing_l.angle_amplitude)
                    s = 'L:% 7.4f' % self.fly.wing_l.angle_amplitude
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, self.fly.wing_l.bgra)
                    w_text = int(50 * self.scale)
                    x += w_text+self.w_gap
                
                if (self.fly.wing_r.angle_amplitude is not None):
                    #s = 'R:% 7.1f' % np.rad2deg(self.fly.wing_r.angle_amplitude)
                    s = 'R:% 7.4f' % self.fly.wing_r.angle_amplitude
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, self.fly.wing_r.bgra)
                    w_text = int(50 * self.scale)
                    x += w_text+self.w_gap
                
    
                # Output sum of WBA
                if (self.fly.wing_l.angle_amplitude is not None) and (self.fly.wing_r.angle_amplitude is not None):
                    leftplusright = self.fly.wing_l.angle_amplitude + self.fly.wing_r.angle_amplitude
                    #s = 'L+R:% 7.1f' % np.rad2deg(leftplusright)
                    s = 'L+R:% 7.4f' % leftplusright
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, cv.Scalar(255,64,64,0) )
                    w_text = int(70 * self.scale)
                    x += w_text+self.w_gap

                    
                # Output difference in WBA
                if (self.fly.wing_l.angle_amplitude is not None) and (self.fly.wing_r.angle_amplitude is not None):
                    leftminusright = self.fly.wing_l.angle_amplitude - self.fly.wing_r.angle_amplitude
                    #s = 'L-R:% 7.1f' % np.rad2deg(leftminusright)
                    s = 'L-R:% 7.4f' % leftminusright
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, cv.Scalar(255,64,64,0) )
                    w_text = int(70 * self.scale)
                    x += w_text+self.w_gap

                    
                # Output flight status
                if (self.fly.wing_l.bFlying and self.fly.wing_r.bFlying):
                    s = 'FLIGHT'
                else:
                    s = 'no flight'
                
                cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, cv.Scalar(255,64,255,0) )
                w_text = int(70 * self.scale)
                x += w_text+self.w_gap
            

                # Display a test image.
                #if self.fly.wing_r.imgTest is not None:
                #    cv2.imshow(self.window_name, self.fly.wing_r.imgTest)

                # Display the image.
                cv2.imshow(self.window_name, imgOutput)
                cv2.waitKey(1)


            # Update the fly internals.
            self.fly.update(imgForeground, rosimage.header)

            # Publish the outputs.
            self.fly.publish()
            
            

    # save_background()
    # Save the current camera image as the background.
    #
    def save_background(self):
        self.imgBackground = self.imgCamera
        rospy.logwarn ('Saving new background image %s' % self.filenameBackground)
        cv2.imwrite(self.filenameBackground, self.imgBackground)
    
    
    # hit_object()
    # Get the nearest handle point or button to the mouse point.
    # ptMouse    = [x,y]
    # Returns the bodypart, tag, and ui of item the mouse has hit, using the 
    # convention that the name is of the form "tag_bodypart", e.g. "hinge_left"
    #
    def hit_object(self, ptMouse):
        tagHit  = None
        bodypartHit = None
        uiHit = None
        
        # Check for button press.
        iPressed = None
        for iButton in range(len(self.buttons)):
            if (self.buttons[iButton].hit_test(ptMouse)):
                iPressed = iButton
            
        if (iPressed is not None):
            nameNearest = self.buttons[iPressed].name
            (tagHit,delim,bodypartHit) = nameNearest.partition('_')
            uiHit = 'button'
        else: # Check for handle hit.
            tag  = [None,None,None,None]
            bodypart = [None,None,None,None]
            (bodypart[0], tag[0]) = self.fly.wing_l.hit_object(ptMouse)
            (bodypart[1], tag[1]) = self.fly.wing_r.hit_object(ptMouse)
            (bodypart[2], tag[2]) = self.fly.head.hit_object(ptMouse)
            (bodypart[3], tag[3]) = self.fly.abdomen.hit_object(ptMouse)
            i = next((i for i in range(len(tag)) if tag[i]!=None), None)
            if (i is not None):
                tagHit  = tag[i]
                bodypartHit = bodypart[i]
                uiHit = 'handle'
    
        
        return (bodypartHit, tagHit, uiHit)
        
        
    # Convert tag and bodypart strings to a name string:  tag_bodypart
    def name_from_tagbodypart(self, tag, bodypart):
        if (bodypart is not None) and (len(bodypart)>0):
            name = tag+'_'+bodypart
        else:
            name = tag
            
        return name
    

    # get_projection_onto_bodyaxis()
    # Project the given point onto the body axis.
    #
    def get_projection_onto_bodyaxis(self, ptAnywhere):
        # Project the point onto the body axis.
        ptB = self.fly.head.ptCenter_i - self.fly.abdomen.ptCenter_i
        ptM = ptAnywhere - self.fly.abdomen.ptCenter_i
        ptAxis = np.dot(ptB,ptM) / np.dot(ptB,ptB) * ptB + self.fly.abdomen.ptCenter_i
            
        return ptAxis
        
                
    def get_reflection_across_bodyaxis(self, ptAnywhere):
        ptAxis = self.get_projection_onto_bodyaxis(ptAnywhere)
        ptReflected = ptAnywhere + 2*(ptAxis-ptAnywhere)
        
        return ptReflected

    
    # update_params_from_handle()
    # Recalculate self.params based on a currently selected handle and mouse location.
    #
    def update_params_from_handle(self, bodypartSelected, tagSelected, ptMouse):             
        bodypartSlave = 'right' if (self.bodypartSelected=='left') else 'left'
                    
        # Head or Abdomen points
        if (bodypartSelected=='head') or (bodypartSelected=='abdomen'):
            if (tagSelected=='center'): 

                # Get the hinge points pre-move.
                if (self.params['symmetric']):
                    ptHead = np.array([self.params['head']['x'], self.params['head']['y']])
                    ptAbdomen = np.array([self.params['abdomen']['x'], self.params['abdomen']['y']])
                    ptCenterPre = (ptHead + ptAbdomen) / 2
                    ptBodyPre = ptHead - ptAbdomen
                    angleBodyPre = np.arctan2(ptBodyPre[1], ptBodyPre[0])
                    ptLeft = np.array([self.params['left']['hinge']['x'], self.params['left']['hinge']['y']])
                    ptRight = np.array([self.params['right']['hinge']['x'], self.params['right']['hinge']['y']])
                    ptLC = ptLeft-ptCenterPre
                    ptRC = ptRight-ptCenterPre
                    rL = np.linalg.norm(ptLC)
                    aL = np.arctan2(ptLC[1], ptLC[0]) - angleBodyPre
                    rR = np.linalg.norm(ptRC)
                    aR = np.arctan2(ptRC[1], ptRC[0]) - angleBodyPre
                
                # Move the selected body point.
                pt = ptMouse
                self.params[bodypartSelected]['x'] = float(pt[0])
                self.params[bodypartSelected]['y'] = float(pt[1])
                
                # Now move the hinge points relative to the new body points.
                if (self.params['symmetric']):
                    ptHead = np.array([self.params['head']['x'], self.params['head']['y']])
                    ptAbdomen = np.array([self.params['abdomen']['x'], self.params['abdomen']['y']])
                    ptCenterPost = (ptHead + ptAbdomen) / 2
                    ptBodyPost = ptHead - ptAbdomen
                    angleBodyPost = np.arctan2(ptBodyPost[1], ptBodyPost[0])
                    ptLeft = ptCenterPost + rL * np.array([np.cos(aL+angleBodyPost), np.sin(aL+angleBodyPost)])
                    ptRight = ptCenterPost + rR * np.array([np.cos(aR+angleBodyPost), np.sin(aR+angleBodyPost)])
                    self.params['left']['hinge']['x'] = float(ptLeft[0])
                    self.params['left']['hinge']['y'] = float(ptLeft[1])
                    self.params['right']['hinge']['x'] = float(ptRight[0])
                    self.params['right']['hinge']['y'] = float(ptRight[1])
                    


            if (tagSelected=='radius_major'): 
                self.params[bodypartSelected]['radius_major'] = float(np.linalg.norm(np.array([self.params[bodypartSelected]['x'],self.params[bodypartSelected]['y']]) - ptMouse))
            if (tagSelected=='radius_minor'): 
                self.params[bodypartSelected]['radius_minor'] = float(np.linalg.norm(np.array([self.params[bodypartSelected]['x'],self.params[bodypartSelected]['y']]) - ptMouse))


        # Wing points.
        elif (bodypartSelected=='left') or (bodypartSelected=='right'):
            # Hinge point.
            if (tagSelected=='hinge'): 
                self.params[bodypartSelected]['hinge']['x'] = float(ptMouse[0])
                self.params[bodypartSelected]['hinge']['y'] = float(ptMouse[1])

                if (self.params['symmetric']):
                    ptSlave = self.get_reflection_across_bodyaxis(ptMouse)
                    self.params[bodypartSlave]['hinge']['x'] = float(ptSlave[0])
                    self.params[bodypartSlave]['hinge']['y'] = float(ptSlave[1])


            # High angle.
            elif (tagSelected=='hi'): 
                pt = ptMouse - self.wingSelected.ptHinge
                self.params[bodypartSelected]['angle_hi'] = float(self.wingSelected.transform_angle_b_from_i(np.arctan2(pt[1], pt[0])))
                self.params[bodypartSelected]['radius_outer'] = float(max(self.wingSelected.params[bodypartSelected]['radius_inner']+2, 
                                                                         np.linalg.norm(self.wingSelected.ptHinge - ptMouse))) # Outer radius > inner radius.
                if (self.params['symmetric']):
                    self.params[bodypartSlave]['angle_hi']     = self.params[bodypartSelected]['angle_hi']
                    self.params[bodypartSlave]['radius_outer'] = self.params[bodypartSelected]['radius_outer']
                  
                  
            # Low angle.
            elif (tagSelected=='lo'): 
                pt = ptMouse - self.wingSelected.ptHinge
                self.params[bodypartSelected]['angle_lo'] = float(self.wingSelected.transform_angle_b_from_i(np.arctan2(pt[1], pt[0])))
                self.params[bodypartSelected]['radius_outer'] = float(max(self.wingSelected.params[bodypartSelected]['radius_inner']+2, 
                                                                          np.linalg.norm(self.wingSelected.ptHinge - ptMouse)))
                if (self.params['symmetric']):
                    self.params[bodypartSlave]['angle_lo']     = self.params[bodypartSelected]['angle_lo']
                    self.params[bodypartSlave]['radius_outer'] = self.params[bodypartSelected]['radius_outer']
                  
                  
            # Inner radius.
            elif (tagSelected=='inner'): 
                self.params[bodypartSelected]['radius_inner'] = float(min(np.linalg.norm(self.wingSelected.ptHinge - ptMouse), 
                                                                          self.wingSelected.params[bodypartSelected]['radius_outer']-2))
                if (self.params['symmetric']):
                    self.params[bodypartSlave]['radius_inner'] = self.params[bodypartSelected]['radius_inner']
                
                
    # onMouse()
    # Handle mouse events.
    #
    def onMouse(self, event, x, y, flags, param):
        ptMouse = np.array([x, y])

        # Keep track of which UI element is selected.
        if (event==cv.CV_EVENT_LBUTTONDOWN):
            # Get the name and ui nearest the current point.
            (bodypart, tag, ui) = self.hit_object(ptMouse)
            self.nameSelected = self.name_from_tagbodypart(tag,bodypart)
            self.tagSelected = tag
            self.bodypartSelected = bodypart
            self.uiSelected = ui
            self.wingSelected = self.fly.wing_l if (self.bodypartSelected=='left') else self.fly.wing_r
            
            self.nameSelectedNow = self.nameSelected
            self.uiSelectedNow = self.uiSelected


        if (self.uiSelected=='button'):
            # Get the bodypart and ui tag nearest the mouse point.
            (bodypart, tag, ui) = self.hit_object(ptMouse)
            self.nameSelectedNow = self.name_from_tagbodypart(tag,bodypart)
            self.tagSelectedNow = tag
            self.bodypartSelectedNow = bodypart
            self.uiSelectedNow = ui
            
            # Set selected button to 'down', others to 'up'.
            for iButton in range(len(self.buttons)):
                if (self.nameSelected == self.nameSelectedNow == self.buttons[iButton].name) and not (event==cv.CV_EVENT_LBUTTONUP):
                    self.buttons[iButton].state = 'down'
                else:
                    self.buttons[iButton].state = 'up'


            if (event==cv.CV_EVENT_LBUTTONUP):
                # If the mouse is on the same button at mouseup, then do the action.
                if (self.uiSelectedNow=='button'):
                    if (self.nameSelected == self.nameSelectedNow == 'save bg'):
                        self.pubCommand.publish('save_background')
    
                    elif (self.nameSelected == self.nameSelectedNow == 'exit'):
                        self.pubCommand.publish('exit')
                        
                    elif (self.nameSelected == self.nameSelectedNow == 'flipedge'):
                        self.pubCommand.publish('flipedge')
                        
                    elif (self.nameSelected == self.nameSelectedNow == 'invertcolor'):
                        self.pubCommand.publish('invertcolor')
                        
                    elif (self.nameSelected == self.nameSelectedNow == 'flipsymmetry'):
                        self.pubCommand.publish('flipsymmetry')
                        
        # end if (self.uiSelected=='button'):

                        
        elif (self.uiSelected=='handle'):
            # Set the new params.
            self.update_params_from_handle(self.bodypartSelected, self.tagSelected, ptMouse)
            self.fly.set_params(self.params)
        
            # Save the results.
            if (event==cv.CV_EVENT_LBUTTONUP):
                self.fly.create_masks(self.shapeImage)
                params1 = self.scale_params(self.params, 1/self.scale)
                rospy.set_param('strokelitude', params1)
            
        # end if (self.uiSelected=='handle'):
            

        if (event==cv.CV_EVENT_LBUTTONUP):
            self.nameSelected = None
            self.uiSelected = None
            self.nameSelectedNow = None
            self.uiSelectedNow = None

            
                
    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    main = MainWindow()

    rospy.logwarn('')
    rospy.logwarn('')
    rospy.logwarn('**************************************************************************')
    rospy.logwarn('')  
    rospy.logwarn('     StrokelitudeROS: Camera-based Wingbeat Analyzer Software for ROS')
    rospy.logwarn('         by Floris van Breugel, Steve Safarik, (c) 2014')
    rospy.logwarn('')  
    rospy.logwarn('     Left click+drag to move any handle points.')
    rospy.logwarn('')  
    rospy.logwarn('')  
    main.command_callback(String('help'))
    rospy.logwarn('**************************************************************************')
    rospy.logwarn('')
    rospy.logwarn('')

    main.run()
