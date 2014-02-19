#!/usr/bin/env python
from __future__ import division
import roslib; roslib.load_manifest('StrokelitudeROS')
import rospy
import rosparam

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


# A class to help with debugging.  It just draws an image.
class Test:
    def __init__(self, bEnable, name):
        self.ravel = None
        self.shape = (0,0)
        self.i = 0
        self.bEnable = bEnable
        self.name = name
        if (self.bEnable):
            cv.NamedWindow(self.name,1)
        
    def show(self):
        if (self.bEnable) and (self.ravel is not None):
            cv2.imshow(self.name, np.reshape(self.ravel, self.shape))
        
gbTestEnable = False   # Set to True to enable some debugging output.
        
    

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


def filter_median(data, q=1): # q is the 'radius' of the filter window.  q==1 is a window of 3.  q==2 is a window of 5.
    data2 = copy.copy(data)
    for i in range(q,len(data)-q):
        data2[i] = np.median(data[i-q:i+q+1]) # Median filter of window.

    # Left-fill the first values.
    try:
        data2[0:q] = data2[q]
    
        # Right-fill the last values.
        data2[len(data2)-q:len(data2)] = data2[-(q+1)]
        dataOut = data2
        
    except IndexError:
        dataOut = data
        
    return dataOut
        

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

        self.angleBody_i = self.get_bodyangle_i()
        self.ptBodyCenter = (np.array([params['head']['x'], params['head']['y']]) + np.array([params['abdomen']['x'], params['abdomen']['y']])) / 2
        r = max(params['left']['radius_outer'], params['right']['radius_outer'])
        self.ptBody1 = tuple((self.ptBodyCenter + r * np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)])).astype(int))
        self.ptBody2 = tuple((self.ptBodyCenter - r * np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)])).astype(int))
    
            
    def get_bodyangle_i(self):
        angle_i = get_angle_from_points_i(self.abdomen.ptCenter_i, self.head.ptCenter_i)
        angleBody  = (angle_i + np.pi) % (2.0*np.pi) - np.pi
         
        return angleBody
        
                
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

        if (gbTestEnable):
            for wing in [self.wing_r, self.wing_l]:
                i = int(wing.test.i/1)
                #iBin = wing.iBinsValid[i % (len(wing.iBinsValid))] # Sweep.
                #iBin = abs(int(len(wing.bins) * wing.angleBody_i / np.pi / 2.0)) # Vertical wedge on the right wing.
                iBin = wing.iMajor
#                rospy.logwarn('%s: bodyangle=%s, nBins=%d, iBin=%d, bin=%s' % (wing.side, wing.angleBody_i, len(wing.bins), iBin, wing.bins[iBin]))
                wing.test.shape = wing.imgRoi.shape
                #wing.test.ravel = copy.deepcopy(np.ravel(wing.imgMaskRoi))
                wing.test.ravel = np.ravel(wing.imgRoi)#copy.deepcopy(np.ravel(wing.imgRoi))
                #wing.test.ravel = copy.deepcopy(np.ravel(wing.imgAnglesRoi_b))
                
                # Draw wedge locations on the test images.
                wing.test.ravel[wing.iPixelsRoi[iBin-3]] = 255#255-wing.test.ravel[wing.iPixelsRoi[iBin-1]]
                wing.test.ravel[wing.iPixelsRoi[iBin-2]] = 128#255-wing.test.ravel[wing.iPixelsRoi[iBin-1]]
                wing.test.ravel[wing.iPixelsRoi[iBin-1]] = 0#255-wing.test.ravel[wing.iPixelsRoi[iBin-1]]
                wing.test.ravel[wing.iPixelsRoi[iBin]] = 128#255-wing.test.ravel[wing.iPixelsRoi[iBin]]
                wing.test.ravel[wing.iPixelsRoi[iBin+1]] = 255#255-wing.test.ravel[wing.iPixelsRoi[iBin+1]]
                wing.test.ravel[wing.iPixelsRoi[iBin+2]] = 128#255-wing.test.ravel[wing.iPixelsRoi[iBin+1]]
                wing.test.ravel[wing.iPixelsRoi[iBin+3]] = 0#255-wing.test.ravel[wing.iPixelsRoi[iBin+1]]
                wing.test.i += 1
    
            
    def draw(self, image):
        # Draw line to indicate the body.
        cv2.line(image, self.ptBody1, self.ptBody2, self.bgra_body, 1) # Draw a line longer than just head-to-abdomen.
                
        self.head.draw(image)
        self.abdomen.draw(image)
        self.wing_l.draw(image)
        self.wing_r.draw(image)

        
    
    def publish(self):
        pt = self.head.ptCenter_i - self.head.ptHinge_i + self.head.ptCOM # The head COM point relative to the hinge.
        angleHead = -(np.arctan2(pt[1], pt[0]) - self.angleBody_i)# + np.pi/2.0)
        angleHead = (angleHead + np.pi) % (2.0*np.pi) - np.pi
        radiusHead = np.linalg.norm(pt)
        
        pt = self.abdomen.ptCenter_i - self.abdomen.ptHinge_i + self.abdomen.ptCOM # The abdomen COM point relative to the abdomen hinge.
        angleAbdomen = -(np.arctan2(pt[1], pt[0]) - self.angleBody_i)# + np.pi/2.0)
        angleAbdomen = (angleAbdomen + np.pi) % (2.0*np.pi) - np.pi
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
        self.bgra_com = bgra_dict['blue']
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
        self.params = params
        self.angleBody_i = self.get_bodyangle_i()
        self.cosBodyangle = np.cos(self.angleBody_i)
        self.sinBodyangle = np.sin(self.angleBody_i)
        self.ptCenter_i = np.array([self.params[self.name]['x'], self.params[self.name]['y']])
        
        # Compute the hinge location, which is on the intersection of the bodypart ellipse and the body axis.
        ptBodyCenter = (np.array([self.params['head']['x'], self.params['head']['y']]) + np.array([self.params['abdomen']['x'], self.params['abdomen']['y']])) / 2
        r = self.params[self.name]['radius_major']
        ptHinge1 = (self.ptCenter_i + r*np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)]))
        ptHinge2 = (self.ptCenter_i - r*np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)]))
        r1 = np.linalg.norm(ptHinge1 - ptBodyCenter)
        r2 = np.linalg.norm(ptHinge2 - ptBodyCenter)
        self.ptHinge_i = ptHinge1 if (r1<r2) else ptHinge2 

        # Refresh the handle points.
        self.update_handle_points()
        
        
    def get_bodyangle_i(self):
        angle_i = get_angle_from_points_i(np.array([self.params['abdomen']['x'], self.params['abdomen']['y']]), 
                                          np.array([self.params['head']['x'], self.params['head']['y']]))
        angleBody  = (angle_i         + np.pi) % (2.0*np.pi) - np.pi
         
        return angleBody
        
                
    # create_mask()
    # Create an image mask.
    #
    def create_mask(self, shape):
        # Create the mask.
        self.imgMask = np.zeros(shape, dtype=np.uint8)
        cv2.ellipse(self.imgMask,
                    (int(self.params[self.name]['x']), int(self.params[self.name]['y'])),
                    (int(self.params[self.name]['radius_major']), int(self.params[self.name]['radius_minor'])),
                    np.rad2deg(self.angleBody_i),
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
                    np.rad2deg(self.angleBody_i),
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
    def __init__(self, side='right', params={}, color='black'):
        self.side = side
        
        self.ravelMaskRoi       = None
        self.ravelAnglesRoi_b   = None
        
        self.bins               = None
        self.intensities        = None
        
        # Bodyframe angles have zero degrees point orthogonal to body axis: 
        # If body axis is north/south, then 0-deg is east for right wing, west for left wing.
        
        # Imageframe angles are oriented to the image.  0-deg is east, +-pi is west, -pi/2 is north, +pi/2 is south.
        
        
        self.angle_trailing_b = None
        self.angle_leading_b  = None
        self.angle_amplitude  = None
        self.angle_mean       = None
        self.mass             = 0.0
        self.bFlying          = False
        self.iMajor           = 0
        
        self.pixelmax         = 255.
        self.bgra             = bgra_dict[color]
        self.thickness_inner  = 1
        self.thickness_outer  = 1
        self.thickness_wing   = 1
        
        self.handles = {'hinge':Handle(np.array([0,0])),
                        'angle_hi':Handle(np.array([0,0])),
                        'angle_lo':Handle(np.array([0,0])),
                        'inner':Handle(np.array([0,0]))
                        }

        self.set_params(params)

        # services, for live histograms
        self.service_intensity = rospy.Service('wing_intensity_'+side, float32list, self.serve_intensity_callback)
        self.service_bins      = rospy.Service('wing_bins_'+side, float32list, self.serve_bins_callback)
        self.service_edges     = rospy.Service('wing_edges_'+side, float32list, self.serve_edges_callback)
        
        self.test = Test(gbTestEnable, self.side)

    
    # get_angles_i_from_b()
    # Return angle1 and angle2 oriented to the image rather than the fly.
    # * corrected for left/right full-circle angle, i.e. east is 0-deg, west is 270-deg.
    # * corrected for wrapping at delta>np.pi.
    #
    def get_angles_i_from_b(self, angle_lo_b, angle_hi_b):
        angle_lo_i = self.transform_angle_i_from_b(angle_lo_b)
        angle_hi_i = self.transform_angle_i_from_b(angle_hi_b)

        if (angle_hi_i-angle_lo_i > np.pi):
            angle_hi_i -= (2.0*np.pi)
        if (angle_lo_i-angle_hi_i > np.pi):
            angle_lo_i -= (2.0*np.pi)
            
        return (float(angle_lo_i), float(angle_hi_i))
    
    
    # transform_angle_i_from_b()
    # Transform an angle from the fly body frame to the camera image frame.
    #
    def transform_angle_i_from_b(self, angle_b):
        if self.side == 'right':
            angle_i  =  angle_b + self.angleBody_i + np.pi/2.0
        else: # left
            angle_i  = -angle_b + self.angleBody_i + np.pi/2.0 + np.pi
             
        angle_i = (angle_i+np.pi) % (2.0*np.pi) - np.pi
        return angle_i
        

    # transform_angle_b_from_i()
    # Transform an angle from the camera image frame to the fly frame.
    #
    def transform_angle_b_from_i(self, angle_i):
        if self.side == 'right':
            angle_b  =  angle_i - self.angleBody_i - np.pi/2.0
        else:  
            angle_b  = -angle_i + self.angleBody_i + np.pi/2.0 + np.pi

        angle_b = ((angle_b+np.pi) % (2.0*np.pi)) - np.pi
        return angle_b
         

    # set_params()
    # Set the given params dict into this object.  Any member vars that come from params should be set here.
    #
    def set_params(self, params):
        self.params      = params
        self.ptHinge     = np.array([self.params[self.side]['hinge']['x'], self.params[self.side]['hinge']['y']]).astype(np.float64)
        resolution_min   = 1.1*np.sqrt(2.0)/self.params[self.side]['radius_inner'] # Enforce at least sqrt(2) pixels wide at inner radius, with a little fudge buffer.
        nbins            = int((2.0*np.pi)/max(self.params['resolution_radians'], resolution_min)) + 1
        self.bins        = np.linspace(-np.pi, np.pi, nbins).astype(np.float64)
        self.intensities = np.zeros(len(self.bins), dtype=np.float64)
        self.angleBody_i = self.get_bodyangle_i()

        (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.params[self.side]['angle_lo'], self.params[self.side]['angle_hi'])
        angle_mid_i              = (angle_hi_i - angle_lo_i)/2 + angle_lo_i
         
         
        # Cache the sin & cos values for drawing handles, etc.
        self.cos = {}
        self.sin = {}
        self.cos['angle_hi']  = float(np.cos(angle_hi_i)) 
        self.sin['angle_hi']  = float(np.sin(angle_hi_i))
        self.cos['angle_mid'] = float(np.cos(angle_mid_i))
        self.sin['angle_mid'] = float(np.sin(angle_mid_i))
        self.cos['angle_lo']  = float(np.cos(angle_lo_i))
        self.sin['angle_lo']  = float(np.sin(angle_lo_i))

        angle_lo = (self.params[self.side]['angle_lo'] + np.pi) % (2.0*np.pi) - np.pi
        angle_hi = (self.params[self.side]['angle_hi'] + np.pi) % (2.0*np.pi) - np.pi

        angle_min = min(angle_lo, angle_hi)
        angle_max = max(angle_lo, angle_hi)
        
        self.iBinsValid = list(np.where((angle_min <= self.bins) * (self.bins <= angle_max))[0])
        
        if (len(self.iBinsValid)==0):
            self.iBinsValid = [0] # TODO: make this the proper bin to match the angle. 

        self.update_handle_points()
        
        
    def get_bodyangle_i(self):
        angle_i = get_angle_from_points_i(np.array([self.params['abdomen']['x'], self.params['abdomen']['y']]), 
                                          np.array([self.params['head']['x'], self.params['head']['y']]))
        angleBody  = (angle_i + np.pi) % (2.0*np.pi) - np.pi
        return angleBody 
        
                
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
        cv2.circle(imgMask,
                    ptCenter,
                    int(self.params[self.side]['radius_inner']),
                    0,
                    cv.CV_FILLED)
        

        # Find the ROI of the mask.
        xSum = np.sum(imgMask, 0)
        ySum = np.sum(imgMask, 1)
        xMin = np.where(xSum>0)[0][0] - 2
        xMax = np.where(xSum>0)[0][-1] + 3
        yMin = np.where(ySum>0)[0][0] - 2
        yMax = np.where(ySum>0)[0][-1] + 3
        
        self.roi = np.array([xMin, yMin, xMax, yMax])
        self.imgMaskRoi = imgMask[yMin:yMax, xMin:xMax]

        
        self.ravelMaskRoi = np.ravel(self.imgMaskRoi)
        
        
    
    def update_imgroi(self, image, roi):
        # Only use the ROI that covers the stroke rect.
        self.imgRoi = image[roi[1]:roi[3], roi[0]:roi[2]]


    
    # create_angle_mask()
    # Create an image where each pixel value is the angle from the hinge, in body coordinates.                    
    # 
    def create_angle_mask(self, shape):
        # Set up matrices of x and y coordinates.
        x = np.tile(np.array([range(shape[1])]).astype(np.float64)   - self.ptHinge[0], (shape[0], 1))
        y = np.tile(np.array([range(shape[0])]).astype(np.float64).T - self.ptHinge[1], (1, shape[1]))

        # Calc the angle at each pixel coordinate.
        imgAngles_i = np.arctan2(y,x) # Ranges [-pi,+pi]
        imgAnglesRoi_i = imgAngles_i[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        
        self.imgAnglesRoi_b  = self.transform_angle_b_from_i(imgAnglesRoi_i)
        self.ravelAnglesRoi_b = np.ravel(self.imgAnglesRoi_b)
                   

    # assign_pixels_to_bins()
    # Put every pixel of the ROI into one of the bins.
    #
    def assign_pixels_to_bins(self):
        # Create empty bins.
        iPixelsRoi = [[] for i in range(len(self.bins))]

        # Put each iPixel into an appropriate bin.            
        for iPixel, angle in enumerate(self.ravelAnglesRoi_b):
            if self.ravelMaskRoi[iPixel]:
                iBinBest = np.argmin(np.abs(self.bins - angle))
                iPixelsRoi[iBinBest].append(iPixel)
        
        # Convert to numpy array.
        self.iPixelsRoi = np.array(np.zeros(len(iPixelsRoi), dtype=object))
        for k in range(len(iPixelsRoi)):
            self.iPixelsRoi[k] = np.array(iPixelsRoi[k], dtype=int)
#         for i in self.iPixelsRoi:
#             rospy.logwarn('%s: %s, %s' % (self.side, i, len(i)))
                
         
    # update_intensity_function()
    # Update the list of intensities corresponding to the bin angles.
    #            
    def update_intensity_function(self):
        if (self.ravelMaskRoi is not None) and (self.intensities is not None):
            ravelRoi = np.ravel(self.imgRoi)
            
            # Get the pixel mass.
            self.mass = np.sum(ravelRoi) / self.pixelmax
            
            # Compute the stroke intensity function.
            for iBin in self.iBinsValid:
                iPixels = self.iPixelsRoi[iBin]
                if (len(iPixels) > 0):
                    pixels = ravelRoi[iPixels]                     # TODO: This line is a cause of slow framerate.
                    #pixels = np.zeros(len(iPixels))               # Compare with this line.
                    self.intensities[iBin] = np.mean(pixels)# / self.pixelmax
                else:
                    self.intensities[iBin] = 0.0
                    
            self.intensities = filter_median(self.intensities, q=1)
             

                        
    # update_edge_stats()
    # Calculate the leading and trailing edge angles,
    # and the amplitude & mean stroke angle.
    #                       
    def update_edge_stats(self):                
        self.angle_leading_b = None
        self.angle_trailing_b = None
        self.angle_amplitude = None
        self.angle_mean = None
            
        if (self.intensities is not None):
            if self.bFlying:
                if (len(self.iBinsValid)>1):
                    (self.angle_leading_b, self.angle_trailing_b) = self.get_edge_angles()
                    self.angle_amplitude                          = np.abs(self.angle_leading_b - self.angle_trailing_b)
                    self.angle_mean                               = np.mean([self.angle_leading_b, self.angle_trailing_b])
                    
            else: # not flying
                self.angle_leading_b  = np.pi/2.0
                self.angle_trailing_b = np.pi/2.0
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
        self.handles['angle_hi'].pt = (self.ptHinge + self.params[self.side]['radius_outer'] * np.array([self.cos['angle_hi'], 
                                                                                                   self.sin['angle_hi']]))
        self.handles['angle_lo'].pt = (self.ptHinge + self.params[self.side]['radius_outer'] * np.array([self.cos['angle_lo'], 
                                                                                                   self.sin['angle_lo']]))

        # Inner Radius.
        self.handles['inner'].pt = (self.ptHinge + self.params[self.side]['radius_inner'] * np.array([self.cos['angle_mid'], 
                                                                                                      self.sin['angle_mid']]))

    
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, image):
        self.update_imgroi(image, self.roi)
        self.update_intensity_function()
        self.update_edge_stats()
        self.update_flight_status()

    
    # get_edge_angles()
    # Get the angles of the two wing edges.
    #                    
    def get_edge_angles(self):
        #diff = filter_median(np.diff(self.intensities[self.iBinsValid]), q=2)
        i = self.intensities[self.iBinsValid]
        diff = i[3:] - i[:-3] # 3rd degree difference.
        # BUG: There is a glitch in the intensity function that I have not tracked down, but that occurs
        # near vertical and horizontal in image coordinates.  As a workaround, we do two things:
        # Filter the intensity function, and do a higher degree diff, rather than just a np.diff().
        
        
        # Major and minor edges must have opposite signs.  Major is the steeper of the two intensity gradients.
        jMax = np.argmax(diff)
        jMin = np.argmin(diff)
#        (jMajor,jMinor) = (jMax,jMin) if (np.abs(diff[jMax]) > np.abs(diff[jMin])) else (jMin,jMax)
        if (np.abs(diff[jMax]) > np.abs(diff[jMin])): 
            (jMajor,jMinor) = (jMax,jMin)  
        else:
            (jMajor,jMinor) = (jMin,jMax)
            
        iMajor = self.iBinsValid[jMajor]
        iMinor = self.iBinsValid[jMinor]
        self.iMajor = iMajor
        self.iMinor = iMinor
#             rospy.logwarn('%s **********************************************' % self.side)
#             rospy.logwarn('iBinsValid: %s' % self.iBinsValid)
# #            rospy.logwarn('iPixelsRoi: %s' % self.iPixelsRoi[self.iBinsValid])
# #            rospy.logwarn('iPixelsRoi[%s]: %s' % (iMajor, self.iPixelsRoi[iMajor]))
# #            rospy.logwarn('iPixelsRoi[%s]: %s' % (iMinor, self.iPixelsRoi[iMinor]))
#             rospy.logwarn(self.ravelRoi[self.iPixelsRoi[iMajor]])
#             rospy.logwarn('%s, %s' % (self.intensities[iMajor], np.mean(self.ravelRoi[self.iPixelsRoi[iMajor]])))

#            cv2.imwrite('/home/ssafarik/test.png', self.imgRoi)
#            rospy.logwarn(self.intensities[self.iBinsValid])
#            rospy.logwarn(np.diff(self.intensitiesValid))
#            rospy.logwarn('%s: %s' % (self.side, (iMajor,iMinor)))
              
        # Output histograms of the wedges around the problem.
#         if (self.side=='right'):# and fTest=='b'):
#             ravelRoi = np.ravel(self.imgRoi)
#             edges = np.array(range(30,180,8))
#             rospy.logwarn('%s **********************************************' % self.side)
#             for i in range(-3,4):
#                 kMajor = iMajor+i
#                 if (min(self.iBinsValid)<=kMajor<=max(self.iBinsValid)):
#                     iPixelsMajor = self.iPixelsRoi[kMajor]
#                     pixelsMajor = ravelRoi[iPixelsMajor]
#                     (hist,edges) = np.histogram(pixelsMajor, edges)
#                     rospy.logwarn('% d: n=%3d, mean=%6.2f, min=%6.2f, max=%6.2f, hist=%s' % (i, len(pixelsMajor), np.mean(pixelsMajor), np.min(pixelsMajor), np.max(pixelsMajor), hist))
# #                rospy.logwarn('iMajor=%s, i=%s, %s<=iBinsValid<=%s, iPixelsMajor=%s' % (iMajor, i, min(self.iBinsValid), max(self.iBinsValid), iPixelsMajor))
# #                rospy.logwarn('iMajor=%s, i=%s, nPixels=%d, %s<=iBinsValid<=%s' % (iMajor, i, len(self.iPixelsRoi[kMajor]), min(self.iBinsValid), max(self.iBinsValid)))
# #                rospy.logwarn('% d: n=%3d, iMajor=%d, pixelsMajor=%s' % (i, len(pixelsMajor), iMajor, pixelsMajor))
# #                rospy.logwarn('% d: %6.2f, %6.2f, %s, %s' % (i, np.mean(pixelsMajor), np.std(pixelsMajor), np.sum(pixelsMajor), len(pixelsMajor)))

        # Convert the edge index to an angle.
        if (self.params['n_edges']==2):
            angle1 = self.bins[iMajor]
            angle2 = self.bins[iMinor]
        elif (self.params['n_edges']==1):
            angle1 = self.bins[iMajor]
            angle2 = 0.0
        else:
            rospy.logwarn('Parameter error:  n_edges must be 1 or 2.')

        
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
                
                x0 = self.ptHinge[0] + self.params[self.side]['radius_inner'] * np.cos(angle_leading_i)
                y0 = self.ptHinge[1] + self.params[self.side]['radius_inner'] * np.sin(angle_leading_i)
                x1 = self.ptHinge[0] + self.params[self.side]['radius_outer'] * np.cos(angle_leading_i)
                y1 = self.ptHinge[1] + self.params[self.side]['radius_outer'] * np.sin(angle_leading_i)
                cv2.line(image, (int(x0),int(y0)), (int(x1),int(y1)), self.bgra, self.thickness_wing)
                
                if (self.params['n_edges']==2):
                    x0 = self.ptHinge[0] + self.params[self.side]['radius_inner'] * np.cos(angle_trailing_i)
                    y0 = self.ptHinge[1] + self.params[self.side]['radius_inner'] * np.sin(angle_trailing_i)
                    x1 = self.ptHinge[0] + self.params[self.side]['radius_outer'] * np.cos(angle_trailing_i)
                    y1 = self.ptHinge[1] + self.params[self.side]['radius_outer'] * np.sin(angle_trailing_i)
                    cv2.line(image, (int(x0),int(y0)), (int(x1),int(y1)), self.bgra, self.thickness_wing)
                
        self.draw_handles(image)
        
        self.test.show()

                        
    def serve_bins_callback(self, request):
        # Bins.
        if (self.bins is not None):
            return float32listResponse(self.bins)
        
        # Diff bins.
#         if (self.bins is not None):
#             return float32listResponse(self.bins[self.iBinsValid][:-1])
            
    def serve_intensity_callback(self, request):
        # Intensity function.
        if (self.intensities is not None):
            return float32listResponse(self.intensities)
        
        # Number of pixels in each bin.
#         if (self.bins is not None):
#             n = []
#             for i in range(len(self.bins)):
#                 n.append(len(self.iPixelsRoi[i]))
#                 
#             n2 = np.array(n, dtype=np.float)
#             return float32listResponse(n2)
        
        # Diff
#         if (self.intensities is not None):
#             return float32listResponse(np.diff(self.intensities[self.iBinsValid]))

        # Filtered diff.
#         if (self.intensities is not None):
#             return float32listResponse(filter_median(np.diff(self.intensities[self.iBinsValid])))
            
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
        
        # Load the parameters yaml file.
        self.parameterfile = os.path.expanduser(rospy.get_param('strokelitude_parameterfile', '~/strokelitude.yaml'))
        try:
            self.params = rosparam.load_file(self.parameterfile)[0][0]
        except rosparam.RosParamException, e:
            rospy.logwarn('%s.  Using default values.' % e)
            self.params = {}
            
        defaults = {'filenameBackground':'~/strokelitude.png',
                    'image_topic':'/camera/image_raw',
                    'use_gui':True,                     # You can turn off the GUI to speed the framerate.
                    'invertcolor':False,                # You want a light fly on a dark background.  Only needed if not using a background image.
                    'symmetric':True,                   # Forces the UI to remain symmetric.
                    'resolution_radians':0.0174532925,  # Coarser resolution will speed the framerate. 1 degree == 0.0174532925 radians.
                    'threshold_flight':0.1,
                    'scale_image':1.0,                  # Reducing the image scale will speed the framerate.
                    'n_edges':1,                        # Number of edges per wing to find.  1 or 2.
                    'abdomen':{'x':300,
                            'y':250,
                            'radius_minor':60,
                            'radius_major':70},
                    'head':{'x':300,
                            'y':150,
                            'radius_minor':50,
                            'radius_major':50},
                    'left':{'hinge':{'x':100,
                                     'y':100},
                            'radius_outer':30,
                            'radius_inner':10,
                            'angle_hi':0.7854, 
                            'angle_lo':-0.7854
                            },
                    'right':{'hinge':{'x':300,
                                      'y':100},
                             'radius_outer':30,
                             'radius_inner':10,
                             'angle_hi':0.7854, 
                             'angle_lo':-0.7854
                             },

                    }
        self.set_dict_with_preserve(self.params, defaults)
        rospy.set_param('strokelitude', self.params)
        
        # Background image.
        self.filenameBackground = os.path.expanduser(self.params['filenameBackground'])
        self.imgBackground  = cv2.imread(self.filenameBackground, cv.CV_LOAD_IMAGE_GRAYSCALE)
        
        self.scale = self.params['scale_image']
        
        # initialize wings and body
        self.fly = Fly(self.params)
        
        self.nameSelected = None
        self.uiSelected = None
        self.fly.update_handle_points()
        self.tPrev = rospy.Time.now().to_sec()
        self.hz = 0.0
        self.hzSum = 0.0
        self.iCount = 0
        
        # Publishers.
        self.pubCommand            = rospy.Publisher('strokelitude/command', String)

        # Subscriptions.        
        self.subImageRaw           = rospy.Subscriber(self.params['image_topic'], Image, self.image_callback)
        self.subCommand            = rospy.Subscriber('strokelitude/command', String, self.command_callback)

        self.w_gap = int(20 * self.scale)
        self.scaleText = 0.4 * self.scale
        self.fontface = cv2.FONT_HERSHEY_SIMPLEX

        # UI button specs.
        self.buttons = []
        x = int(10 * self.scale)
        y = int(10 * self.scale)
        btn = Button(name='exit', 
                   text='exit', 
                   pt=[x,y],
                   scaleText=self.scaleText)
        self.buttons.append(btn)
        
        x = btn.right+1
        btn = Button(name='save bg', 
                   text='saveBG', 
                   pt=[x,y],
                   scaleText=self.scaleText)
        self.buttons.append(btn)
        
        x = btn.right+1
        btn = Button(name='invertcolor', 
                   text='inverted', # Instantiate with the longest possible text. 
                   pt=[x,y],
                   scaleText=self.scaleText)
        btn.set_text('normal' if (not self.params['invertcolor']) else 'inverted') 
        self.buttons.append(btn)

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
        self.fly.set_params(self.scale_params(self.params, self.scale))
        rosparam.dump_params(self.parameterfile, 'strokelitude')
        
        return config


    # command_callback()
    # Execute any commands sent over the command topic.
    #
    def command_callback(self, command):
        self.command = command.data
        
        if (self.command == 'exit'):
            rospy.signal_shutdown('User requested exit.')
        
        
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

        self.fly.set_params(self.scale_params(self.params, self.scale))
        self.set_dict_with_preserve(self.params, rospy.get_param('strokelitude'))
        rospy.set_param('strokelitude', self.params)
        rosparam.dump_params(self.parameterfile, 'strokelitude')
        
    def scale_params(self, paramsIn, scale):
		paramsOut = copy.deepcopy(paramsIn)
		for wing in ['left', 'right']:
			paramsOut[wing]['hinge']['x'] = (paramsIn[wing]['hinge']['x']*scale)  
			paramsOut[wing]['hinge']['y'] = (paramsIn[wing]['hinge']['y']*scale)  
			paramsOut[wing]['radius_outer'] = (paramsIn[wing]['radius_outer']*scale)  
			paramsOut[wing]['radius_inner'] = (paramsIn[wing]['radius_inner']*scale)  

		for bodypart in ['head', 'abdomen']:
			paramsOut[bodypart]['x'] = (paramsIn[bodypart]['x']*scale) 
			paramsOut[bodypart]['y'] = (paramsIn[bodypart]['y']*scale)  
			paramsOut[bodypart]['radius_minor'] = (paramsIn[bodypart]['radius_minor']*scale)  
			paramsOut[bodypart]['radius_major'] = (paramsIn[bodypart]['radius_major']*scale)
			
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
                    if (self.params['n_edges']==1):
                        s = 'L:% 7.4f' % (self.fly.wing_l.angle_leading_b)
                        w = 70
                    else:
                        s = 'L:% 7.4f,% 7.4f' % (self.fly.wing_l.angle_leading_b,self.fly.wing_l.angle_trailing_b)
                        w = 120
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, self.fly.wing_l.bgra)
                    w_text = int(w * self.scale)
                    x += w_text+self.w_gap
                
                if (self.fly.wing_r.angle_amplitude is not None):
                    if (self.params['n_edges']==1):
                        s = 'R:% 7.4f' % (self.fly.wing_r.angle_leading_b)
                        w = 70
                    else:
                        s = 'R:% 7.4f,% 7.4f' % (self.fly.wing_r.angle_leading_b,self.fly.wing_r.angle_trailing_b)
                        w = 120
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, self.fly.wing_r.bgra)
                    w_text = int(w * self.scale)
                    x += w_text+self.w_gap
                
    
                # Output sum of WBA
                if (self.fly.wing_l.angle_amplitude is not None) and (self.fly.wing_r.angle_amplitude is not None):
                    leftplusright = self.fly.wing_l.angle_amplitude + self.fly.wing_r.angle_amplitude
                    #s = 'L+R:% 7.1f' % np.rad2deg(leftplusright)
                    s = 'L+R:% 7.4f' % leftplusright
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, bgra_dict['magenta'])
                    w_text = int(90 * self.scale)
                    x += w_text+self.w_gap

                    
                # Output difference in WBA
                if (self.fly.wing_l.angle_amplitude is not None) and (self.fly.wing_r.angle_amplitude is not None):
                    leftminusright = self.fly.wing_l.angle_amplitude - self.fly.wing_r.angle_amplitude
                    #s = 'L-R:% 7.1f' % np.rad2deg(leftminusright)
                    s = 'L-R:% 7.4f' % leftminusright
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, bgra_dict['magenta'])
                    w_text = int(90 * self.scale)
                    x += w_text+self.w_gap

                    
                # Output flight status
                if (self.fly.wing_l.bFlying and self.fly.wing_r.bFlying):
                    s = 'FLIGHT'
                else:
                    s = 'no flight'
                
                cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, bgra_dict['magenta'])
                w_text = int(50 * self.scale)
                x += w_text+self.w_gap
            

                # Output the framerate.
                tNow = rospy.Time.now().to_sec()
                dt = tNow - self.tPrev
                self.tPrev = tNow
                hzNow = 1/dt if dt != 0.0 else 0.0
                self.iCount += 1
                if (self.iCount > 100):                     
                    a= 0.01
                    self.hz = (1-a)*self.hz + a*hzNow 
                else:                                       
                    if (self.iCount>20):             # Get past the transient response.       
                        self.hzSum += hzNow                 
                    else:
                        self.hzSum = hzNow * self.iCount     
                        
                    self.hz = self.hzSum / self.iCount
                    
                cv2.putText(imgOutput, '%5.1f Hz' % self.hz, (x, y_bottom), self.fontface, self.scaleText, cv.Scalar(255,64,255,0) )
                w_text = int(70 * self.scale)
                x += w_text+self.w_gap
            
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
                    
        params = self.scale_params(self.params, self.scale) 
        
        # Head or Abdomen points
        if (bodypartSelected=='head') or (bodypartSelected=='abdomen'):
            if (tagSelected=='center'): 

                # Get the hinge points pre-move.
                if (params['symmetric']):
                    ptHead = np.array([params['head']['x'], params['head']['y']])
                    ptAbdomen = np.array([params['abdomen']['x'], params['abdomen']['y']])
                    ptCenterPre = (ptHead + ptAbdomen) / 2
                    ptBodyPre = ptHead - ptAbdomen
                    angleBodyPre = np.arctan2(ptBodyPre[1], ptBodyPre[0])
                    ptLeft = np.array([params['left']['hinge']['x'], params['left']['hinge']['y']])
                    ptRight = np.array([params['right']['hinge']['x'], params['right']['hinge']['y']])
                    ptLC = ptLeft-ptCenterPre
                    ptRC = ptRight-ptCenterPre
                    rL = np.linalg.norm(ptLC)
                    aL = np.arctan2(ptLC[1], ptLC[0]) - angleBodyPre # angle from center to hinge in body axis coords.
                    rR = np.linalg.norm(ptRC)
                    aR = np.arctan2(ptRC[1], ptRC[0]) - angleBodyPre

                # Move the selected body point.
                pt = ptMouse
                params[bodypartSelected]['x'] = float(pt[0])
                params[bodypartSelected]['y'] = float(pt[1])
                
                # Now move the hinge points relative to the new body points.
                if (params['symmetric']):
                    ptHead = np.array([params['head']['x'], params['head']['y']])
                    ptAbdomen = np.array([params['abdomen']['x'], params['abdomen']['y']])
                    ptCenterPost = (ptHead + ptAbdomen) / 2
                    ptBodyPost = ptHead - ptAbdomen
                    angleBodyPost = np.arctan2(ptBodyPost[1], ptBodyPost[0])
                    ptLeft = ptCenterPost + rL * np.array([np.cos(aL+angleBodyPost), np.sin(aL+angleBodyPost)])
                    ptRight = ptCenterPost + rR * np.array([np.cos(aR+angleBodyPost), np.sin(aR+angleBodyPost)])
                    params['left']['hinge']['x'] = float(ptLeft[0])
                    params['left']['hinge']['y'] = float(ptLeft[1])
                    params['right']['hinge']['x'] = float(ptRight[0])
                    params['right']['hinge']['y'] = float(ptRight[1])
                    


            if (tagSelected=='radius_major'): 
                params[bodypartSelected]['radius_major'] = float(np.linalg.norm(np.array([params[bodypartSelected]['x'],params[bodypartSelected]['y']]) - ptMouse))
            if (tagSelected=='radius_minor'): 
                params[bodypartSelected]['radius_minor'] = float(np.linalg.norm(np.array([params[bodypartSelected]['x'],params[bodypartSelected]['y']]) - ptMouse))


        # Wing points.
        elif (bodypartSelected=='left') or (bodypartSelected=='right'):
            # Hinge point.
            if (tagSelected=='hinge'): 
                params[bodypartSelected]['hinge']['x'] = float(ptMouse[0])
                params[bodypartSelected]['hinge']['y'] = float(ptMouse[1])

                if (params['symmetric']):
                    ptSlave = self.get_reflection_across_bodyaxis(ptMouse)
                    params[bodypartSlave]['hinge']['x'] = float(ptSlave[0])
                    params[bodypartSlave]['hinge']['y'] = float(ptSlave[1])


            # High angle.
            elif (tagSelected=='angle_hi'): 
                pt = ptMouse - self.wingSelected.ptHinge
                params[bodypartSelected]['angle_hi'] = float(self.wingSelected.transform_angle_b_from_i(np.arctan2(pt[1], pt[0])))
                params[bodypartSelected]['radius_outer'] = float(max(self.wingSelected.params[bodypartSelected]['radius_inner']+2, 
                                                                          np.linalg.norm(self.wingSelected.ptHinge - ptMouse))) # Outer radius > inner radius.
                if (params['symmetric']):
                    params[bodypartSlave]['angle_hi']     = params[bodypartSelected]['angle_hi']
                    params[bodypartSlave]['radius_outer'] = params[bodypartSelected]['radius_outer']
                  
                  
            # Low angle.
            elif (tagSelected=='angle_lo'): 
                pt = ptMouse - self.wingSelected.ptHinge
                params[bodypartSelected]['angle_lo'] = float(self.wingSelected.transform_angle_b_from_i(np.arctan2(pt[1], pt[0])))
                params[bodypartSelected]['radius_outer'] = float(max(self.wingSelected.params[bodypartSelected]['radius_inner']+2, 
                                                                          np.linalg.norm(self.wingSelected.ptHinge - ptMouse)))
                if (params['symmetric']):
                    params[bodypartSlave]['angle_lo']     = params[bodypartSelected]['angle_lo']
                    params[bodypartSlave]['radius_outer'] = params[bodypartSelected]['radius_outer']
                  
                  
            # Inner radius.
            elif (tagSelected=='inner'): 
                params[bodypartSelected]['radius_inner'] = float(min(np.linalg.norm(self.wingSelected.ptHinge - ptMouse), 
                                                                          self.wingSelected.params[bodypartSelected]['radius_outer']-2))
                if (params['symmetric']):
                    params[bodypartSlave]['radius_inner'] = params[bodypartSelected]['radius_inner']
                
        self.params = self.scale_params(params, 1/self.scale) 

                
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
                        
                    elif (self.nameSelected == self.nameSelectedNow == 'invertcolor'):
                        self.pubCommand.publish('invertcolor')
                        
                    elif (self.nameSelected == self.nameSelectedNow == 'flipsymmetry'):
                        self.pubCommand.publish('flipsymmetry')
                        
        # end if (self.uiSelected=='button'):

                        
        elif (self.uiSelected=='handle'):
            # Set the new params.
            self.update_params_from_handle(self.bodypartSelected, self.tagSelected, ptMouse)
            self.fly.set_params(self.scale_params(self.params, self.scale))
            
            self.set_dict_with_preserve(self.params, rospy.get_param('strokelitude'))
            rospy.set_param('strokelitude', self.params)
            rosparam.dump_params(self.parameterfile, 'strokelitude')
            
        
            # Save the results.
            if (event==cv.CV_EVENT_LBUTTONUP):
                self.fly.create_masks(self.shapeImage)
                self.set_dict_with_preserve(self.params, rospy.get_param('strokelitude'))
                rospy.set_param('strokelitude', self.params)
            
        # end if (self.uiSelected=='handle'):
            

        if (event==cv.CV_EVENT_LBUTTONUP):
            self.nameSelected = None
            self.uiSelected = None
            self.nameSelectedNow = None
            self.uiSelectedNow = None

            
                
    def run(self):
        rospy.spin()

        self.set_dict_with_preserve(self.params, rospy.get_param('strokelitude'))
        rospy.set_param('strokelitude', self.params)
        rosparam.dump_params(self.parameterfile, 'strokelitude')

        cv2.destroyAllWindows()


if __name__ == '__main__':

    main = MainWindow()

    rospy.logwarn('')
    rospy.logwarn('')
    rospy.logwarn('**************************************************************************')
    rospy.logwarn('')  
    rospy.logwarn('     StrokelitudeROS: Camera-based Wingbeat Analyzer Software for ROS')
    rospy.logwarn('         by Steve Safarik, Floris van Breugel (c) 2014')
    rospy.logwarn('')  
    rospy.logwarn('     Left click+drag to move any handle points.')
    rospy.logwarn('')  
    rospy.logwarn('')  
    main.command_callback(String('help'))
    rospy.logwarn('**************************************************************************')
    rospy.logwarn('')
    rospy.logwarn('')

    main.run()
