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
from StrokelitudeROS.msg import MsgFlystate, MsgWing, MsgBodypart, MsgCommand
from StrokelitudeROS.cfg import strokelitudeConfig


# A class to help with debugging.  It just draws an image.
class ImageWindow:
    def __init__(self, bEnable, name):
        self.image = None
        self.shape = (0,0)
        self.name = name
        self.bEnable = False
        self.set_enable(bEnable)

        
    def set_shape(self, shape):
        self.shape = shape
        
    def set_ravelled(self, ravel):
        self.image = np.reshape(ravel, self.shape)
        
    def set_image(self, image):
        if (image is not None):
            self.image = image.astype(np.uint8)
        
    def show(self):
        if (self.bEnable) and (self.image is not None) and (self.image.size>0):
            cv2.imshow(self.name, self.image)
        
    def set_enable(self, bEnable):
        if (self.bEnable and not bEnable):
            cv2.destroyWindow(self.name)
            
        if (not self.bEnable and bEnable):
            cv2.namedWindow(self.name)
            
        self.bEnable = bEnable
        
        
gbImageWindowEnable = True   # Set to True to enable debugging output.
        
    

# Colors.
bgra_dict = {'black'         : cv.Scalar(0,0,0,0),
             'white'         : cv.Scalar(255,255,255,0),
             'dark_gray'     : cv.Scalar(64,64,64,0),
             'gray'          : cv.Scalar(128,128,128,0),
             'light_gray'    : cv.Scalar(192,192,192,0),
             'red'           : cv.Scalar(0,0,255,0),
             'green'         : cv.Scalar(0,255,0,0), 
             'blue'          : cv.Scalar(255,0,0,0),
             'cyan'          : cv.Scalar(255,255,0,0),
             'magenta'       : cv.Scalar(255,0,255,0),
             'yellow'        : cv.Scalar(0,255,255,0),
             'dark_red'      : cv.Scalar(0,0,128,0),
             'dark_green'    : cv.Scalar(0,128,0,0), 
             'dark_blue'     : cv.Scalar(128,0,0,0),
             'dark_cyan'     : cv.Scalar(128,128,0,0),
             'dark_magenta'  : cv.Scalar(128,0,128,0),
             'dark_yellow'   : cv.Scalar(0,128,128,0),
             'light_red'     : cv.Scalar(175,175,255,0),
             'light_green'   : cv.Scalar(175,255,175,0), 
             'light_blue'    : cv.Scalar(255,175,175,0),
             'light_cyan'    : cv.Scalar(255,255,175,0),
             'light_magenta' : cv.Scalar(255,175,255,0),
             'light_yellow'  : cv.Scalar(175,255,255,0),
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
    
    
class Struct:
    pass

        
###############################################################################
###############################################################################
class Button:
    def __init__(self, name=None, text=None, pt=None, rect=None, scale=1.0, type='pushbutton', state=False):
        self.name = name
        self.pt = pt
        self.rect = rect
        self.scale = scale
        self.type = type            # 'pushbutton' or 'checkbox'
        self.state = state
        self.widthCheckbox = 10*self.scale
        self.set_text(text)
        
        self.colorWhite = cv.Scalar(255,255,255,0)
        self.colorBlack = cv.Scalar(0,0,0,0)
        self.colorFace = cv.Scalar(128,128,128,0)
        self.colorLightFace = cv.Scalar(192,192,192,0)
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
        
        self.ptCheckCenter = (int(self.ptLT2[0] + 2 + self.widthCheckbox/2), self.ptCenter[1])
        self.ptCheckLT     = (int(self.ptCheckCenter[0]-self.widthCheckbox/2), int(self.ptCheckCenter[1]-self.widthCheckbox/2))
        self.ptCheckRT     = (int(self.ptCheckCenter[0]+self.widthCheckbox/2), int(self.ptCheckCenter[1]-self.widthCheckbox/2))
        self.ptCheckLB     = (int(self.ptCheckCenter[0]-self.widthCheckbox/2), int(self.ptCheckCenter[1]+self.widthCheckbox/2))
        self.ptCheckRB     = (int(self.ptCheckCenter[0]+self.widthCheckbox/2), int(self.ptCheckCenter[1]+self.widthCheckbox/2))


    def hit_test(self, ptMouse):
        if (self.rect[0] <= ptMouse[0] <= self.rect[0]+self.rect[2]) and (self.rect[1] <= ptMouse[1] <= self.rect[1]+self.rect[3]):
            return True
        else:
            return False
        

    def set_text(self, text):
        self.text = text
        (sizeText,rv) = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.4*self.scale, 1)
        self.sizeText = (sizeText[0],    sizeText[1])
            

        if (self.rect is not None):
            pass
        elif (self.pt is not None):
            self.rect = [0,0,0,0]
            if (self.type=='pushbutton'):
                self.rect[0] = self.pt[0]
                self.rect[1] = self.pt[1]
                self.rect[2] = self.sizeText[0] + 6
                self.rect[3] = self.sizeText[1] + 6
            elif (self.type=='checkbox'):
                self.rect[0] = self.pt[0]
                self.rect[1] = self.pt[1]
                self.rect[2] = int(self.sizeText[0] + 6 + self.widthCheckbox + 4)
                self.rect[3] = self.sizeText[1] + 6
        else:
            rospy.logwarn('Error creating Button().')

        if (self.type=='pushbutton'):
            self.ptCenter = (int(self.rect[0]+self.rect[2]/2),                       int(self.rect[1]+self.rect[3]/2))
            self.ptText = (self.ptCenter[0] - int(self.sizeText[0]/2) - 1, 
                           self.ptCenter[1] + int(self.sizeText[1]/2) - 1)
        elif (self.type=='checkbox'):
            self.ptCenter = (int(self.rect[0]+self.rect[2]/2+(self.widthCheckbox+4)/2), int(self.rect[1]+self.rect[3]/2))
            self.ptText = (self.ptCenter[0] - int(self.sizeText[0]/2) - 1 + 2, 
                           self.ptCenter[1] + int(self.sizeText[1]/2) - 1)


                
    # draw_button()
    # Draw a 3D shaded button with text.
    # rect is (left, top, width, height), increasing y goes down.
    def draw(self, image):
        if (self.type=='pushbutton'):
            if (not self.state): # 'up'
                colorOuter = self.colorWhite
                colorInner = self.colorBlack
                colorHilight = self.colorHilight
                colorLolight = self.colorLolight
                colorFill = self.colorFace
                colorText = self.colorText
                colorCheck = self.colorBlack
                ptText0 = (self.ptText[0], self.ptText[1])
            else:
                colorOuter = self.colorWhite
                colorInner = self.colorBlack
                colorHilight = self.colorLolight
                colorLolight = self.colorHilight
                colorFill = self.colorFace
                colorText = self.colorText
                colorCheck = self.colorBlack
                ptText0 = (self.ptText[0]+2, self.ptText[1]+2)
        elif (self.type=='checkbox'):
            colorOuter = self.colorWhite
            colorInner = self.colorBlack
            colorHilight = self.colorLightFace
            colorLolight = self.colorLightFace
            colorFill = self.colorLightFace
            colorText = self.colorText
            colorCheck = self.colorBlack
            ptText0 = (self.ptText[0], self.ptText[1])
            
        cv2.rectangle(image, self.ptLT0, self.ptRB0, colorOuter, 1)
        cv2.rectangle(image, self.ptLT, self.ptRB, colorInner, 1)
        cv2.line(image, self.ptRT1, self.ptRB1, colorLolight)
        cv2.line(image, self.ptLB1, self.ptRB1, colorLolight)
        cv2.line(image, self.ptLT1, self.ptRT1, colorHilight)
        cv2.line(image, self.ptLT1, self.ptLB1, colorHilight)
        cv2.rectangle(image, self.ptLT2, self.ptRB2, colorFill, cv.CV_FILLED)
        if (self.type=='checkbox'):
            cv2.rectangle(image, self.ptCheckLT, self.ptCheckRB, colorCheck, 1)
            if (self.state): # 'down'
                cv2.line(image, self.ptCheckLT, self.ptCheckRB, colorCheck)
                cv2.line(image, self.ptCheckRT, self.ptCheckLB, colorCheck)

        cv2.putText(image, self.text, ptText0, cv2.FONT_HERSHEY_SIMPLEX, 0.4*self.scale, colorText)
        
# end class Button                
    
            
###############################################################################
###############################################################################
class Handle:
    def __init__(self, pt=np.array([0,0]), color=bgra_dict['white']):
        self.pt = pt

        self.color = color
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
        cv2.circle(image, tuple(self.pt.astype(int)),  self.radius, self.color, cv.CV_FILLED)
        
# end class Handle
                

###############################################################################
###############################################################################
class Fly(object):
    def __init__(self, params={}):
        self.head    = Bodypart(name='head',    params=params, color='cyan') 
        self.abdomen = Bodypart(name='abdomen', params=params, color='magenta') 
        self.wing_r  = Wing(name='right',       params=params, color='red')
        self.wing_l  = Wing(name='left',        params=params, color='green')
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
        ptBodyCenter_i = (np.array([params['head']['x'], params['head']['y']]) + np.array([params['abdomen']['x'], params['abdomen']['y']])) / 2
        r = max(params['left']['radius_outer'], params['right']['radius_outer'])
        self.ptBody1 = tuple((ptBodyCenter_i + r * np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)])).astype(int))
        self.ptBody2 = tuple((ptBodyCenter_i - r * np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)])).astype(int))
    
            
    def get_bodyangle_i(self):
        angle_i = get_angle_from_points_i(self.abdomen.ptCenter_i, self.head.ptCenter_i)
        angleBody  = (angle_i + np.pi) % (2.0*np.pi) - np.pi
         
        return angleBody
        
                
    def set_background(self, image):
        self.head.set_background(image)
        self.abdomen.set_background(image)
        self.wing_l.set_background(image)
        self.wing_r.set_background(image)

    
    def update_handle_points(self):
        self.head.update_handle_points()
        self.abdomen.update_handle_points()
        self.wing_l.update_handle_points()
        self.wing_r.update_handle_points()
        

    def update(self, header=None, image=None):
        if (header is not None):
            self.stamp = header.stamp
        else:
            self.stamp = rospy.Time.now()
        
        self.head.update(header, image)
        self.abdomen.update(header, image)
        self.wing_l.update(header, image)
        self.wing_r.update(header, image)

            
    def draw(self, image):
        # Draw line to indicate the body.
        cv2.line(image, self.ptBody1, self.ptBody2, self.bgra_body, 1) # Draw a line longer than just head-to-abdomen.
                
        self.head.draw(image)
        self.abdomen.draw(image)
        self.wing_l.draw(image)
        self.wing_r.draw(image)

        
    
    def publish(self):
#         pt = self.head.ptCenter_i - self.head.ptHinge_i + self.head.state.pt # The head COM point relative to the hinge.
#         angleHead = -(np.arctan2(pt[1], pt[0]) - self.angleBody_i)# + np.pi/2.0)
#         angleHead = (angleHead + np.pi) % (2.0*np.pi) - np.pi
#         radiusHead = np.linalg.norm(pt)
#         
#         pt = self.abdomen.ptCenter_i - self.abdomen.ptHinge_i + self.abdomen.state.pt # The abdomen COM point relative to the abdomen hinge.
#         angleAbdomen = -(np.arctan2(pt[1], pt[0]) - self.angleBody_i)# + np.pi/2.0)
#         angleAbdomen = (angleAbdomen + np.pi) % (2.0*np.pi) - np.pi
#         radiusAbdomen = np.linalg.norm(pt)
        
        flystate              = MsgFlystate()
        flystate.header       = Header(seq=self.iCount, stamp=self.stamp, frame_id='Fly')
        flystate.left         = MsgWing(mass=self.wing_l.mass, angle1=self.wing_l.angle_leading_b, angle2=self.wing_l.angle_trailing_b)
        flystate.right        = MsgWing(mass=self.wing_r.mass, angle1=self.wing_r.angle_leading_b, angle2=self.wing_r.angle_trailing_b)
        flystate.head         = MsgBodypart(mass   = self.head.state.mass,    
                                            radius = self.head.state.radius,    
                                            angle  = self.head.state.angle)
        flystate.abdomen      = MsgBodypart(mass   = self.abdomen.state.mass, 
                                            radius = self.abdomen.state.radius, 
                                            angle  = self.abdomen.state.angle)
        self.iCount += 1
        
        self.pubFlystate.publish(flystate)
        

# end class Fly

        
###############################################################################
###############################################################################
# Head or Abdomen.
class Bodypart(object):
    def __init__(self, name=None, params={}, color='white'):
        self.bInitializedMasks = False
        self.bInitialized = False

        self.name = name

        self.bgra     = bgra_dict[color]
        self.bgra_dim = tuple(np.array(bgra_dict[color])/3)
        self.bgra_state = bgra_dict['red']
        self.pixelmax = 255.0
        self._transforms = {}
    
        self.windowPolar = ImageWindow(False, self.name+'Polar')
        self.windowBG    = ImageWindow(False, self.name+'BG')
        self.windowFG    = ImageWindow(False, self.name+'FG')
        self.windowTest  = ImageWindow(True, 'Test')

        self.state = Struct()
        self.stateInitial = Struct()
        self.stateMin = Struct()
        self.stateMax = Struct()

        self.stampPrev = None
        self.dt = rospy.Time(0)

        self.handles = {'center':Handle(np.array([0,0]), self.bgra),
                        #'radius_ortho':Handle(np.array([0,0]), self.bgra),
                        'radius_axial':Handle(np.array([0,0]), self.bgra),
                        'angle_wedge':Handle(np.array([0,0]), self.bgra)
                        }

        self.shape = (np.inf, np.inf)
        self.ptCOM        = np.array([0.0, 0.0])
        self.ptCenter_i   = np.array([0,0])
        
        self.roi1 = None
        self.roi2 = None
        self.roiClipped = np.array([0,0,0,0])
        
        self.imgFullBackground                     = None
        
        # Size 2x region of interest.
        self.imgRoi2_0                             = None # Untouched roi image.
        self.imgRoi2                               = None # Background subtracted.
        self.imgRoi2Windowed                       = None
        self.imgRoi2WindowedPrev                   = None
        self.imgRoi2Masked                         = None
        self.imgRoi2MaskedPolar                    = None
        self.imgRoi2MaskedPolarCropped             = None
        self.imgRoi2MaskedPolarCroppedWindowed     = None
        self.imgRoi2MaskedPolarCroppedWindowedPrev = None
        self.imgRoi2MaskedPolarCroppedWindowedInitial = None

        self.imgRoi2Background                     = None
        
        # Size 1x region of interest.
        self.imgRoi1_0                           = None # Untouched roi image.
        self.imgRoi1                             = None # Background subtracted.
        self.imgRoi1Masked                       = None
        self.imgRoi1Background                   = None
        
        self.create_wfn                          = self.create_wfn_tukey
        self.wfnRoi2                             = None
        self.wfnRoi2MaskedPolarCropped           = None
        
        self.maskRoiEllipse1                     = None
        self.maskRoiEllipse2                     = None
        
        self.set_params(params)
        self.pubAngle = rospy.Publisher('strokelitude/'+self.name+'/angle', Float32)

    
    # set_params()
    # Set the given params dict into this object.
    #
    def set_params(self, params):
        self.params = params
        self.rc_background = self.params['rc_background']
        self.angleBody_i = self.get_bodyangle_i()
        self.ptCenter_i = np.array([self.params[self.name]['x'], self.params[self.name]['y']])
        
        # Compute the body-outward-facing angle, which is the angle from the body center to the bodypart center.
        ptBodyCenter_i = (np.array([self.params['head']['x'], self.params['head']['y']]) + np.array([self.params['abdomen']['x'], self.params['abdomen']['y']])) / 2
        self.angleBodyOutward_i = np.arctan2(self.ptCenter_i[1]-ptBodyCenter_i[1], self.ptCenter_i[0]-ptBodyCenter_i[0])
#         r = self.params[self.name]['radius_axial']
#         ptHinge1 = (self.ptCenter_i + r*np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)]))
#         ptHinge2 = (self.ptCenter_i - r*np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)]))
#         r1 = np.linalg.norm(ptHinge1 - ptBodyCenter_i)
#         r2 = np.linalg.norm(ptHinge2 - ptBodyCenter_i)
#         if (r1<r2):
#             self.ptHinge_i = ptHinge1
#             self.angleBodyOutward_i = self.angleBody_i + np.pi
#         else:
#             self.ptHinge_i = ptHinge2 
#             self.angleBodyOutward_i = self.angleBody_i
        self.ptHinge_i = self.ptCenter_i
        
        
        self.cosAngleBody = np.cos(self.angleBody_i)
        self.sinAngleBody = np.sin(self.angleBody_i)
        self.cosAngleBodyOutward = np.cos(self.angleBodyOutward_i)
        self.sinAngleBodyOutward = np.sin(self.angleBodyOutward_i)
        self.R = np.array([[self.cosAngleBodyOutward, -self.sinAngleBodyOutward], [self.sinAngleBodyOutward, self.cosAngleBodyOutward]])

        self.imgRoi2MaskedPolarCroppedWindowedInitial = None
        self.iInitial = 0
        self.iCount = 0
        self.p = 0.5

        self.stateInitial.mass = 0.0
        self.stateInitial.angle = 0.0
        self.stateInitial.radius = self.params[self.name]['radius_axial']

        self.state.mass = 0.0
        self.state.angle = 0.0
        self.state.radius = 0.0

        self.stateMin.mass = np.inf
        self.stateMin.angle = np.inf
        self.stateMin.radius = np.inf

        self.stateMax.mass = -np.inf
        self.stateMax.angle = -np.inf
        self.stateMax.radius = -np.inf

        self.imgRoi1Background = None
        self.imgRoi2Background = None
        
        angle = self.params[self.name]['angle_wedge'] # Range is on [0,+2pi]
        
#         # Convert from elliptical angle to circular angle.
#         r1 = self.params[self.name]['radius_axial']
#         r2 = self.params[self.name]['radius_ortho']
#         xy_e = np.array([np.cos(angle)/r1, np.sin(angle)/r1])
#         angle = np.arctan2(xy_e[1], xy_e[0])

        # Map the wedge angle onto [0,pi]
        if (angle==0.0):
            angle = np.pi
            
        if (np.pi < angle):
            angle = 2*np.pi - angle 
        
        self.angle_wedge = angle
        

        # Refresh the handle points.
        self.update_handle_points()

        # Turn on/off the extra windows.
        self.windowPolar.set_enable(self.params['extra_windows'] and self.params[self.name]['track'])
        self.windowBG.set_enable(self.params['extra_windows'] and self.params[self.name]['track'] and self.params[self.name]['subtract_bg'])
        self.windowFG.set_enable(self.params['extra_windows'] and self.params[self.name]['track'])
        
        
    def _get_transform_polar_log(self, i_0, j_0, i_n, j_n, nRho, dRho, nTheta, theta_0, theta_1):
        transform = self._transforms.get((i_0, j_0, i_n, j_n, nRho, nTheta, theta_0, theta_1))
    
        if (transform == None):
            i_k     = []
            j_k     = []
            rho_k   = []
            theta_k = []
    
            aspect = float(i_n) / float(j_n)
            dTheta = (theta_1 - theta_0) / nTheta
            for iRho in range(0, nRho):
                rho = np.exp(iRho * dRho)
                
                for iTheta in range(0, nTheta):
                    theta = theta_0 + iTheta * dTheta
    
                    # i,j points on a circle.
                    i_c = rho * np.sin(theta)
                    j_c = rho * np.cos(theta)
                    
                    # Expand the circle onto an ellipse in the larger dimension.
                    if (aspect>=1.0):
                        i = i_0 + int(i_c * aspect)
                        j = j_0 + int(j_c)
                    else:
                        i = i_0 + int(i_c)
                        j = j_0 + int(j_c / aspect)
    
                    if (0 <= i < i_n) and (0 <= j < j_n):
                        i_k.append(i)
                        j_k.append(j)
                        rho_k.append(iRho)
                        theta_k.append(iTheta)
    
            transform = ((np.array(rho_k), np.array(theta_k)), (np.array(i_k), np.array(j_k)))
            self._transforms[i_0, j_0, i_n, j_n, nRho, nTheta, theta_0, theta_1] = transform
    
        return transform
    
    # transform_polar_log()
    # Remap an image into log-polar coordinates, where (i_0,j_0) is the (y,x) origin in the original image.
    # nRho:     Number of radial (vert) pixels in the output image.
    # aRho:     Multiplier of vertical output dimension instead of nRho.
    # nTheta:   Number of angular (horiz) pixels in the output image.
    # aTheta:   Multiplier of horizontal output dimension instead of nTheta.
    # theta_0:  Starting angle.
    # theta_1:  Ending angle.
    # scale:    0.0=Include to the nearest side (all pixels in output image are valid); 1.0=Include to the 
    #           farthest corner (some pixels in output image from outside the input image). 
    #
    # Credit to http://machineawakening.blogspot.com/2012/02
    #
    def transform_polar_log(self, image, i_0, j_0, nRho=None, aRho=1.0, nTheta=None, aTheta=1.0, theta_0=0.0, theta_1=2.0*np.pi, scale=0.0):
        (i_n, j_n) = image.shape[:2]
        
        i_c = max(i_0, i_n - i_0)
        j_c = max(j_0, j_n - j_0)
        d_c = (i_c ** 2 + j_c ** 2) ** 0.5 # Distance to the farthest image corner.
        d_s = min(i_0, i_n-i_0, j_0, j_n-j_0)  # Distance to the nearest image side.
        d = scale*d_c + (1.0-scale)*d_s
        
        if (nRho == None):
            nRho = int(np.ceil(d*aRho))
        
        if (nTheta == None):
            #nTheta = int(np.ceil(j_n * aTheta))
            nTheta = int(aTheta * 2*np.pi*np.sqrt((i_n**2 + j_n**2)/2)) # Approximate circumference of ellipse in the roi.
        
        dRho = np.log(d) / nRho
        
        (pt, ij) = self._get_transform_polar_log(i_0, j_0, 
                                           i_n, j_n, 
                                           nRho, dRho, 
                                           nTheta, theta_0, theta_1)
        imgTransformed = np.zeros((nRho, nTheta) + image.shape[2:], dtype=image.dtype)
        imgTransformed[pt] = image[ij]

        return imgTransformed


    def _get_transform_polar_elliptical(self, i_0, j_0, i_n, j_n, (r1, r2), angleEllipse, nRho, nTheta, theta_0, theta_1, rClip):
        nTheta = max(1,nTheta)
        transform = self._transforms.get((i_0, j_0, i_n, j_n, nRho, nTheta, theta_0, theta_1))
    
        if (transform == None):
            i_k     = []
            j_k     = []
            rho_k   = []
            theta_k = []

#             # Convert from circular angles to elliptical angles.
#             xy_c = np.array([r1*np.cos(theta_0), r2*np.sin(theta_0)])
#             theta_0e = np.arctan2(xy_c[1], xy_c[0])
#             xy_c = np.array([r1*np.cos(theta_1), r2*np.sin(theta_1)])
#             theta_1e = np.arctan2(xy_c[1], xy_c[0])

            R = np.array([[np.cos(-angleEllipse), -np.sin(-angleEllipse)], [np.sin(-angleEllipse), np.cos(-angleEllipse)]])    
            dTheta = (theta_1 - theta_0) / nTheta
            
            # Step through the wedge from theta_0 to theta_1.
            for iTheta in range(0, nTheta):
                theta = theta_0 + iTheta * dTheta

                # Convert from circular angles to elliptical angles.
                xy_c = np.array([r1*np.cos(theta), r2*np.sin(theta)])
                theta_e = np.arctan2(xy_c[1], xy_c[0])

                # Radius of the ellipse at this angle.
                rho_e = np.linalg.norm([2*r1*np.cos(theta), 2*r2*np.sin(theta)])
                dRho = rho_e / nRho
                
                # Step along the radius from the origin to the ellipse.  rClip is a clip factor on range [0,1]
                for iRho in range(0, int(rClip*nRho)):
                    rho = (iRho * dRho)

                    # i,j points on an upright ellipse, 2*r1 horiz, 2*r2 vert.
                    i_e = rho * np.sin(theta_e)
                    j_e = rho * np.cos(theta_e)
                    
                    # Rotate the point.
                    ij = R.dot([i_e, j_e])
                    
                    # Translate it into position.
                    i = int(i_0 + ij[0])
                    j = int(j_0 + ij[1])
                    
    
                    # Put the transform values into the lists.
                    if (0 <= i < i_n) and (0 <= j < j_n):
                        i_k.append(i)
                        j_k.append(j)
                        rho_k.append(iRho)
                        theta_k.append(iTheta)
    
            transform = ((np.array(rho_k), np.array(theta_k)), (np.array(i_k), np.array(j_k)))
            self._transforms[i_0, j_0, i_n, j_n, nRho, nTheta, theta_0, theta_1] = transform
    
        return transform
    
    # transform_polar_elliptical()
    # Remap an image into linear-polar coordinates, where (i_0,j_0) is the (y,x) origin in the original image.
    # nRho:         Number of radial (vert) pixels in the output image.
    # aRho:         Multiplier of vertical output dimension instead of nRho.
    # radii:        Tuple (r1,r2) of ellipse radii.
    # nTheta:       Number of angular (horiz) pixels in the output image.
    # aTheta:       Multiplier of horizontal output dimension instead of nTheta.
    # angleEllipse: Axis of ellipse rotation.
    # theta_0:      Circular angle to one side of ellipse angle.
    # theta_1:      Circular angle to other side of ellipse angle.
    # scale:        0.0=Include to the nearest side (all pixels in output image are valid); 1.0=Include to the 
    #               farthest corner (some pixels in output image from outside the input image). 
    #
    #
    def transform_polar_elliptical(self, image, i_0, j_0, nRho=None, aRho=1.0, radii=None, angleEllipse=0.0, nTheta=None, aTheta=1.0, theta_0=-np.pi, theta_1=np.pi):
        (i_n, j_n) = image.shape[:2]
        if (radii is None):
            r1 = i_n / 2
            r2 = j_n / 2
        else:
            (r1,r2) = radii
        
#         i_c = max(i_0, i_n-i_0)
#         j_c = max(j_0, j_n-j_0)
#         
#         d_c = (i_c ** 2 + j_c ** 2) ** 0.5      # Distance to the farthest image corner.
#         d_s_near = min(i_0, i_n-i_0, j_0, j_n-j_0)  # Distance to the nearest image side.
#         d_s_far = max(i_0, i_n-i_0, j_0, j_n-j_0)  # Distance to the farthest image side.
        
        # Nearest nonzero distance to a side.
        d_sides = [i_0, i_n-i_0, j_0, j_n-j_0]
        d_nonzero = d_sides[np.where(d_sides>0)[0]]
        d_s_0 = d_nonzero.min()
        
        # Distance to ellipse.
        d_e_r1 = 2*r1
        d_e_r2 = 2*r2
        d_e_wedge = np.linalg.norm([2*r1*np.cos(theta_0), 2*r2*np.sin(theta_0)])

        d_e_min = min(d_e_r1, d_e_wedge, d_e_r2)
        d = d_e_min
        
        (rClip0, rClip1, rClip2, rClip3) = (1.0, 1.0, 1.0, 1.0)
        if (self.roiClipped[0]>0):
            rClip0 = 1.0 - (float(self.roiClipped[0])/float(j_0))
        if (self.roiClipped[1]>0):
            rClip1 = 1.0 - (float(self.roiClipped[1])/float(i_0))
        if (self.roiClipped[2]>0):
            rClip2 = 1.0 - (float(self.roiClipped[2])/float(j_n-j_0))
        if (self.roiClipped[3]>0):
            rClip3 = 1.0 - (float(self.roiClipped[3])/float(i_n-i_0))

        rClip = np.min([rClip0, rClip1, rClip2, rClip3])
        
        # Convert from circular angles to elliptical angles.
        xy_c = np.array([r1*np.cos(theta_0), r2*np.sin(theta_0)])
        theta_0e = np.arctan2(xy_c[1], xy_c[0])
        xy_c = np.array([r1*np.cos(theta_1), r2*np.sin(theta_1)])
        theta_1e = np.arctan2(xy_c[1], xy_c[0])
        
        
        if (nRho == None):
            nRho = int(np.ceil(d * aRho))
        
        # Number of theta steps depends on the number of pixels along the elliptical arc, ideally 1 step per pixel.
        if (nTheta == None):
            circumference = 2*np.pi*np.sqrt(((r1)**2 + (r2)**2)) # Approximate circumference of ellipse.
            fraction_wedge = np.abs(theta_1e - theta_0e)/(2*np.pi) # Approximate fraction that is the wedge of interest.
            nTheta = int(aTheta * circumference * fraction_wedge)  
        
        #rospy.logwarn (((i_n,j_n), (i_0, j_0), d, nRho, (r1,r2), angleEllipse, nTheta, (theta_0e, theta_1e)))
        (pt, ij) = self._get_transform_polar_elliptical(i_0, j_0, 
                                           i_n, j_n,
                                           (r1, r2),
                                           angleEllipse, 
                                           nRho, 
                                           nTheta, theta_0, theta_1, rClip)
        imgTransformed = np.zeros((nRho, nTheta) + image.shape[2:], dtype=image.dtype)
        if (len(pt[0])>0):
            imgTransformed[pt] = image[ij]
        else:
            rospy.logwarn('No points transformed.')
        
        return imgTransformed


    # create_wfn_hanning()
    # Create a 2D Hanning window function.
    #
    def create_wfn_hanning(self, shape):
        (height,width) = shape
        hanning = np.ones(shape, dtype=np.float32)
        if (height>1) and (width>1):
            for i in range(width):
                 for j in range(height):
                     x = 2*np.pi*i/(width-1) # x ranges 0 to 2pi across the image width
                     y = 2*np.pi*j/(height-1) # y ranges 0 to 2pi across the image height
                     hanning[j][i] = 0.5*(1-np.cos(x)) * 0.5*(1-np.cos(y))
                 
        return hanning


    # create_wfn_tukey()
    # Create a 2D Tukey window function.
    #
    def create_wfn_tukey(self, shape):
        (height,width) = shape
        alpha = 0.25 # Width of the flat top.  alpha==0 gives rectangular, alpha=1 gives Hann.
        wfn = np.ones(shape, dtype=np.float32)
        if (height>1) and (width>1):
            for i in range(width):
                for j in range(height):
                    y = np.pi*(2*j/(alpha*(height-1))-1)
                    
                    if (0 <= i <= (alpha*(width-1))/2):
                        x = np.pi*(2*i/(alpha*(width-1))-1)
                    elif ((alpha*(width-1))/2 < i <= (width-1)*(1-alpha/2)):
                        x = 0.0
                    elif ((width-1)*(1-alpha/2) < i <= width-1):
                        x = np.pi*(2*i/(alpha*(width-1))-2/alpha+1)
                        
                    if (0 <= j <= (alpha*(height-1))/2):
                        y = np.pi*(2*j/(alpha*(height-1)) - 1)
                    elif ((alpha*(height-1))/2 < j <= (height-1)*(1-alpha/2)):
                        y = 0.0
                    elif ((height-1)*(1-alpha/2) < j <= height-1):
                        y = np.pi*(2*j/(alpha*(height-1)) - 2/alpha + 1)
                    
                    wfnx = 0.5*(1+np.cos(x))
                    wfny = 0.5*(1+np.cos(y))
                    wfn[j][i] = wfnx * wfny
                 
        return wfn


    # collapse_vertical_bands()
    # Remove any vertical bands of pixels from the image where the top pixel value extends all the way to the bottom.
    #
    def collapse_vertical_bands(self, image):
        imageT = image.T
        nx = len(imageT)
        dx = int(nx * 0.02)
        good = np.ones(nx)
        
        for x in range(nx):
            if (np.mean(imageT[x]) > 0.90*imageT[x][0]):
                xs = range(max(0,x-dx),min(x+dx+1,nx)) # The range of x's near the problem.
                good[xs] = 0
        
        iGood = np.where(good>0)[0]
        imageOut = imageT[iGood].T
        
        return (imageOut, iGood)
        
        
    # get_shift_from_phasecorr()
    # Calculate the coordinate shift between the two images.
    #
    def get_shift_from_phasecorr(self, imgA, imgB):
        if (self.bInitialized) and (imgA is not None) and (imgB is not None) and (imgA.shape==imgB.shape):
            # Phase correlation.
            A  = cv2.dft(imgA)
            B  = cv2.dft(imgB)
            AB = cv2.mulSpectrums(A, B, flags=0, conjB=True)
            crosspower = AB / cv2.norm(AB)
            shift = cv2.idft(crosspower)
            shift0  = np.roll(shift,  int(shift.shape[0]/2), 0)
            shift00 = np.roll(shift0, int(shift.shape[1]/2), 1) # Roll the matrix so 0,0 goes to the center of the image.
            
#             if (self.name=='head'):
#                 img = shift00-np.min(shift00)
#                 img *= 1.0/np.max(img) 
#                 img = np.exp(np.exp(np.exp(img)))
#                 img -= np.min(img)
#                 img *= 255.0/np.max(img) 
#                 self.windowTest.set_image(img)
            
            # Get the coordinates of the maximum shift.
            kShift = np.argmax(shift00)
            (iShift,jShift) = np.unravel_index(kShift, shift00.shape)
            #rospy.logwarn((iShift,jShift))

            # Get weighted centroid of a region around the peak, for sub-pixel accuracy.
#             w = 7
#             r = int((w-1)/2)
#             i0 = clip(iShift-r, 0, shift00.shape[0]-1)
#             i1 = clip(iShift+r, 0, shift00.shape[0]-1)+1
#             j0 = clip(jShift-r, 0, shift00.shape[1]-1)
#             j1 = clip(jShift+r, 0, shift00.shape[1]-1)+1
#             peak = shift00[i0:i1].T[j0:j1].T
#             moments = cv2.moments(peak, binaryImage=False)
#                       
#             if (moments['m00'] != 0.0):
#                 iShift = moments['m01']/moments['m00'] + float(i0)
#                 jShift = moments['m10']/moments['m00'] + float(j0)
            
            #rospy.logwarn((iShift,jShift))

            # Accomodate the matrix roll we did above.
            iShift -= int(shift.shape[0]/2)#float(shift.shape[0])/2.0
            jShift -= int(shift.shape[1]/2)#float(shift.shape[1])/2.0

            # Convert unsigned shifts to signed shifts. 
            height = shift00.shape[0]
            width  = shift00.shape[1]
            iShift  = ((iShift+height/2) % height) - height/2 
            jShift  = ((jShift+width/2) % width) - width/2
            
            rv = np.array([iShift, jShift])
            
        else:
            rv = np.array([0.0, 0.0])
            
        return rv
        
        
    def get_bodyangle_i(self):
        angle_i = get_angle_from_points_i(np.array([self.params['abdomen']['x'], self.params['abdomen']['y']]), 
                                          np.array([self.params['head']['x'], self.params['head']['y']]))
        angleBody  = (angle_i         + np.pi) % (2.0*np.pi) - np.pi
         
        return angleBody
        

    # create_mask()
    # Create elliptical wedge masks, and window functions.
    #
    def create_mask(self, shape):
        # Create the 1x sized mask.
        mask1 = np.zeros(shape, dtype=np.uint8)
        cv2.ellipse(mask1,
                    (int(self.params[self.name]['x']), int(self.params[self.name]['y'])),
                    (int(self.params[self.name]['radius_axial']), int(self.params[self.name]['radius_ortho'])),
                    np.rad2deg(self.angleBodyOutward_i),
                    np.rad2deg(self.angle_wedge),
                    np.rad2deg(-self.angle_wedge),
                    bgra_dict['white'], 
                    cv.CV_FILLED)
        
        # Find the ROI of the mask.
        b=0 # Border
        xSum = np.sum(mask1, 0)
        ySum = np.sum(mask1, 1)
        xMin = np.where(xSum>0)[0][0]  - b
        xMax = np.where(xSum>0)[0][-1] + b+1
        yMin = np.where(ySum>0)[0][0]  - b
        yMax = np.where(ySum>0)[0][-1] + b+1
        
        xMin = np.max([0,xMin])
        yMin = np.max([0,yMin])
        xMax = np.min([xMax, shape[1]-1])
        yMax = np.min([yMax, shape[0]-1])
        self.roi1 = np.array([xMin, yMin, xMax, yMax])
        self.maskRoiEllipse1 = mask1[yMin:yMax, xMin:xMax]
        
        # Create the 2x sized mask (for polar images, so the line-of-interest is at the midpoint).
        mask2 = np.zeros(shape, dtype=np.uint8)
        cv2.ellipse(mask2,
                    (int(self.params[self.name]['x']), int(self.params[self.name]['y'])),
                    (2*int(self.params[self.name]['radius_axial']), 2*int(self.params[self.name]['radius_ortho'])),
                    int(np.rad2deg(self.angleBodyOutward_i)),
                    int(np.rad2deg(self.angle_wedge)),
                    int(np.rad2deg(-self.angle_wedge)),
                    bgra_dict['white'], 
                    cv.CV_FILLED)
        
        # Find the ROI of the mask.
        b=0 # Border
        xSum = np.sum(mask2, 0)
        ySum = np.sum(mask2, 1)
        
        # Dilate with a border.
        xMin0 = np.where(xSum>0)[0][0]  - b
        xMax0 = np.where(xSum>0)[0][-1] + b+1
        yMin0 = np.where(ySum>0)[0][0]  - b
        yMax0 = np.where(ySum>0)[0][-1] + b+1
        
        # Clip border to image edges.
        xMin = np.max([0,xMin0])
        yMin = np.max([0,yMin0])
        xMax = np.min([xMax0, shape[1]-1])
        yMax = np.min([yMax0, shape[0]-1])
        
        self.roi2 = np.array([xMin, yMin, xMax, yMax])
        self.maskRoiEllipse2 = mask2[yMin:yMax, xMin:xMax]
        
        self.wfnRoi2 = None
        self.wfnRoi2MaskedPolarCropped = None
        self.bInitializedMasks = True


        # Find where the mask might be clipped.  First, draw an unclipped ellipse.        
        delta = 1
        isClosed = True
        pts = cv2.ellipse2Poly((int(self.params[self.name]['x']), int(self.params[self.name]['y'])),
                    (2*int(self.params[self.name]['radius_axial']), 2*int(self.params[self.name]['radius_ortho'])),
                    int(np.rad2deg(self.angleBodyOutward_i)),
                    int(np.rad2deg(self.angle_wedge)),
                    int(np.rad2deg(-self.angle_wedge)),
                    delta)
        pts = np.append(pts,[[int(self.params[self.name]['x']), int(self.params[self.name]['y'])]],0)

        # These are the unclipped locations.        
        min0 = pts.min(0)
        max0 = pts.max(0)
        xMin0 = min0[0]
        yMin0 = min0[1]
        xMax0 = max0[0]+1
        yMax0 = max0[1]+1
        
        # Compare unclipped with the as-drawn locations.
        xClip0 = xMin-xMin0
        yClip0 = yMin-yMin0
        xClip1 = xMax0-xMax
        yClip1 = yMax0-yMax
        self.roiClipped = np.array([xClip0, yClip0, xClip1, yClip1])
        
        
    # set_background()
    # Set the given image as the background image.
    #                
    def set_background(self, image):
        if (self.params[self.name]['subtract_bg']):
            self.imgFullBackground = image.astype(np.float32)
            self.imgRoi1Background = None
            self.imgRoi2Background = None
        else:
            self.imgFullBackground = None
            self.imgRoi1Background = None
            self.imgRoi2Background = None
        
        
    def update_background(self):
        if (self.imgRoi2Background is None) and (self.imgFullBackground is not None) and (self.roi2 is not None):
            self.imgRoi2Background = copy.deepcopy(self.imgFullBackground[self.roi2[1]:self.roi2[3], self.roi2[0]:self.roi2[2]])
        
        if (self.imgRoi1Background is None) and (self.imgRoi2Background is not None) and (self.roi1 is not None) and (self.roi2 is not None):
            r0 = self.roi1[0]-self.roi2[0]
            r1 = self.roi1[1]-self.roi2[1]
            r2 = self.roi1[2]-self.roi2[2]
            r3 = self.roi1[3]-self.roi2[3]
            self.imgRoi1Background = self.imgRoi2Background[r1:r3, r0:r2]
            self.minRoi1Background = np.min(self.imgRoi1Background)
            
        dt = max(0, self.dt.to_sec())
        alphaBackground = 1.0 - np.exp(-dt / self.rc_background)

        if (self.imgRoi2Background is not None) and (self.imgRoi2_0 is not None):
            if (self.imgRoi2Background.size==self.imgRoi2_0.size):
                cv2.accumulateWeighted(self.imgRoi2_0.astype(np.float32), self.imgRoi2Background, alphaBackground)
            else:
                self.imgRoi2Background = None
                self.imgRoi2_0 = None
                
        self.windowBG.set_image(self.imgRoi2Background)
        
    # update_handle_points()
    # Update the dictionary of handle point names and locations.
    # Compute the various handle points.
    #
    def update_handle_points (self):
        x = self.params[self.name]['x']
        y = self.params[self.name]['y']
        r1 = self.params[self.name]['radius_axial']
        r2 = self.params[self.name]['radius_ortho']
        angle = self.params[self.name]['angle_wedge']
        
        
        self.handles['center'].pt       = np.array([x, y])
        self.handles['radius_axial'].pt = np.array([x, y]) + ((r1+10) * np.array([self.cosAngleBodyOutward,self.sinAngleBodyOutward]))
        #self.handles['radius_ortho'].pt = np.array([x, y]) + (r2 * np.array([-self.sinAngleBodyOutward,self.cosAngleBodyOutward]))
        self.handles['angle_wedge'].pt  = np.array([x, y]) + np.dot(self.R, np.array([r1*np.cos(angle), -r2*np.sin(angle)]))

        self.ptWedge1    = tuple((np.array([x, y]) + np.dot(self.R, np.array([  r1*np.cos(self.angle_wedge),    -(r2)*np.sin(self.angle_wedge)]))).astype(int))
        self.ptWedge2    = tuple((np.array([x, y]) + np.dot(self.R, np.array([  r1*np.cos(-self.angle_wedge),   -(r2)*np.sin(-self.angle_wedge)]))).astype(int))
        self.ptWedge1_2x = tuple((np.array([x, y]) + np.dot(self.R, np.array([2*r1*np.cos(self.angle_wedge),  -(2*r2)*np.sin(self.angle_wedge)]))).astype(int))
        self.ptWedge2_2x = tuple((np.array([x, y]) + np.dot(self.R, np.array([2*r1*np.cos(-self.angle_wedge), -(2*r2)*np.sin(-self.angle_wedge)]))).astype(int))
        
        
    def update_roi(self, image):
        # Save the prior images.
        if (self.imgRoi2Windowed is not None):
            self.imgRoi2WindowedPrev = self.imgRoi2Windowed
            
        self.shape = image.shape
        
        # Extract the ROI images.
        self.imgRoi2_0 = copy.deepcopy(image[self.roi2[1]:self.roi2[3], self.roi2[0]:self.roi2[2]])
        self.imgRoi1_0 = self.imgRoi2_0[(self.roi1[1]-self.roi2[1]):(self.roi1[3]-self.roi2[1]), (self.roi1[0]-self.roi2[0]):(self.roi1[2]-self.roi2[0])]

        # Background Subtraction.
        if (self.params[self.name]['subtract_bg']):
            if (self.imgRoi1Background is not None):
                self.imgRoi1 = cv2.absdiff(self.imgRoi1_0, self.imgRoi1Background.astype(np.uint8))
                 
            if (self.imgRoi2Background is not None):
                self.imgRoi2 = cv2.absdiff(self.imgRoi2_0, self.imgRoi2Background.astype(np.uint8))
        else:
            self.imgRoi1 = self.imgRoi1_0
            self.imgRoi2 = self.imgRoi2_0
            
        # Rerange the images to black & white.
        self.imgRoi2 -= np.min(self.imgRoi2)
        max2 = np.max(self.imgRoi2)
        self.imgRoi2 *= (255.0/float(max2))
        
        self.windowFG.set_image(self.imgRoi2) #(self.maskRoiEllipse2)# 
        
        # Apply the mask.
        if (self.maskRoiEllipse1 is not None) and (self.imgRoi1 is not None):
            self.imgRoi1Masked = cv2.bitwise_and(self.imgRoi1, self.maskRoiEllipse1)

        if (self.maskRoiEllipse2 is not None) and (self.imgRoi2 is not None):
            self.imgRoi2Masked = self.imgRoi2#cv2.bitwise_and(self.imgRoi2, self.maskRoiEllipse2)
            #self.imgRoiMasked  = cv2.multiply(self.imgRoi.astype(np.float32), self.wfnRoi2)

            
    def update_polar(self):
        if (self.imgRoi2 is not None):
            xOrigin = self.params[self.name]['x'] - self.roi2[0] #int(self.imgRoi2.shape[1]/2)
            yOrigin = self.params[self.name]['y'] - self.roi2[1] #int(self.imgRoi2.shape[0]/2)
            xMax = self.imgRoi2.shape[1]-1
            yMax = self.imgRoi2.shape[0]-1
            
            theta_0a = -self.angle_wedge
            theta_1a = self.angle_wedge
            
            #rospy.logwarn(((self.params[self.name]['x'], self.params[self.name]['y']), self.roi2, (xOrigin, yOrigin), theta_0a, theta_1a))
            self.imgRoi2MaskedPolar  = self.transform_polar_elliptical(self.imgRoi2Masked, 
                                                     yOrigin, 
                                                     xOrigin, 
                                                     aRho = 1.0,
                                                     radii=(self.params[self.name]['radius_axial'], self.params[self.name]['radius_ortho']),
                                                     angleEllipse=self.angleBodyOutward_i,
                                                     aTheta = 1.0,
                                                     theta_0 = min(theta_0a,theta_1a), 
                                                     theta_1 = max(theta_0a,theta_1a))
                
            # Find the y value where the black band should be cropped out.
            sumY = np.sum(self.imgRoi2MaskedPolar,1)
            iSumY = np.where(sumY==0)[0]
            if (len(iSumY)>0):
                iMinY = np.min(iSumY)
            else:
                iMinY = self.imgRoi2MaskedPolar.shape[0]

            self.imgRoi2MaskedPolarCropped = self.imgRoi2MaskedPolar[0:iMinY]
    
             
            if (self.bInitializedMasks):
                if (self.imgRoi2MaskedPolarCroppedWindowed is not None):
                    self.imgRoi2MaskedPolarCroppedWindowedPrev = self.imgRoi2MaskedPolarCroppedWindowed
        
                if (self.wfnRoi2 is None) or (self.imgRoi2.shape != self.wfnRoi2.shape):
                    self.wfnRoi2 = self.create_wfn(self.imgRoi2.shape)
                
                if (self.wfnRoi2MaskedPolarCropped is None) or (self.imgRoi2MaskedPolarCropped.shape != self.wfnRoi2MaskedPolarCropped.shape):
                    self.wfnRoi2MaskedPolarCropped = self.create_wfn(self.imgRoi2MaskedPolarCropped.shape)
                
                self.imgRoi2Windowed                   = cv2.multiply(self.imgRoi2.astype(np.float32), self.wfnRoi2)
                self.imgRoi2MaskedPolarCroppedWindowed = cv2.multiply(self.imgRoi2MaskedPolarCropped.astype(np.float32), self.wfnRoi2MaskedPolarCropped)

            
        if (self.imgRoi2MaskedPolarCroppedWindowedInitial is None):
            self.imgRoi2MaskedPolarCroppedWindowedInitial = self.imgRoi2MaskedPolarCroppedWindowed
            self.iInitial = 1
        
        # Show the image.
        #(img,iGood) = self.collapse_vertical_bands(self.imgRoi2MaskedPolarCropped)
        img = self.imgRoi2MaskedPolarCroppedWindowed
        #img = self.imgRoi2MaskedPolarCropped
        #img = self.imgRoi2MaskedPolarCroppedWindowedInitial
        #img = self.imgRoi1Background
        self.windowPolar.set_image(img)
        

    # update_center_of_mass()
    # Update the bodypart center-of-mass.
    #
    def update_center_of_mass(self):
        if (self.imgRoi1Masked is not None):
            moments = cv2.moments(self.imgRoi1Masked, binaryImage=False)
            
            if (moments['m00'] != 0.0):
                self.mass  = moments['m00'] / self.pixelmax
                self.ptCOM = np.array([moments['m10']/moments['m00'] - self.params[self.name]['x'] + self.roi1[0], 
                                       moments['m01']/moments['m00'] - self.params[self.name]['y'] + self.roi1[1]])
            else:
                self.mass = 0.0
                self.ptCOM = np.array([0,0])
            
    
    # update_state_using_moments()
    # Update the bodypart center-of-mass.
    #
    def update_state_using_moments(self):
        self.state.mass  = self.mass
        self.state.pt = self.ptCOM
            
            
    # update_state_using_phasecorr()
    # Update the bodypart translation & rotation.
    #
    def update_state_using_phasecorr_cartpolar(self): # This is the older version using both the cartesian and polar images.
        # Get the (x,y) translation between successive images.
        (iShift, jShift) = self.get_shift_from_phasecorr(self.imgRoi2Windowed, self.imgRoi2WindowedPrev) # as [x,y]
        ptCartesian = np.array([jShift, iShift])
        ptErr = self.state.pt - self.ptCOM
        self.state.pt += ptCartesian #- 0.005*ptErr # Correct for drift error due to integrating the shifts; pull gently toward the COM.

        # Get the rotation & expansion between successive images.
        if (self.imgRoi2MaskedPolarCroppedWindowed is not None):
            (rShift, aShift) = self.get_shift_from_phasecorr(self.imgRoi2MaskedPolarCroppedWindowed, self.imgRoi2MaskedPolarCroppedWindowedPrev) # as [angle,radius]
            dAngle = aShift * np.abs(2*self.angle_wedge) / self.imgRoi2MaskedPolarCroppedWindowed.shape[1]
            dRadius = rShift
            self.state.angle += dAngle

        self.state.mass = 0.0
        
    def update_state_using_phasecorr(self): # New version using just the polar image.
        imgNow = self.imgRoi2MaskedPolarCroppedWindowed
        imgPrior = self.imgRoi2MaskedPolarCroppedWindowedInitial
        statePrior = self.stateInitial
        
#         if (imgNow is not None) and (imgPrior is not None):
#             self.windowTest.set_image(cv2.absdiff(imgNow,imgPrior))
        
        # Get the rotation & expansion between images.
        if (imgNow is not None) and (imgPrior is not None):
            (rShift, aShift) = self.get_shift_from_phasecorr(imgNow, imgPrior)
            dAngle = aShift * np.abs(2*self.angle_wedge) / float(imgNow.shape[1])
            dRadius = rShift
            self.state.angle  = statePrior.angle + dAngle
            self.state.radius = statePrior.radius + dRadius
            
            # Get min,max's
            self.stateMin.angle  = min(self.stateMin.angle, self.state.angle)
            self.stateMax.angle  = max(self.stateMax.angle, self.state.angle)
            self.stateMin.radius = min(self.stateMin.radius, self.state.radius)
            self.stateMax.radius = max(self.stateMax.radius, self.state.radius)
            
            # Decay min,max values toward the current value.
            RC = 100.0
            a = 1.0-np.exp(-self.dt.to_sec()/RC)
            self.stateMin.angle  += a*(self.state.angle - self.stateMin.angle)
            self.stateMax.angle  -= a*(self.stateMax.angle - self.state.angle)
            self.stateMin.radius += a*(self.state.radius - self.stateMin.radius)
            self.stateMax.radius -= a*(self.stateMax.radius - self.state.radius)
            
            # If angle and radius are near their mean values, then take a new initial image, and set the origin.
            meanAngle = (self.stateMax.angle + self.stateMin.angle)/2.0
            meanRadius = (self.stateMax.radius + self.stateMin.radius)/2.0
            pAngle = np.abs(self.state.angle-meanAngle) / np.abs(self.stateMax.angle-meanAngle)
            pRadius = np.abs(self.state.radius-meanRadius) / np.abs(self.stateMax.radius-meanRadius)
            if (pAngle<self.p) and (pRadius<self.p) and (self.iCount>100):
                self.p *= 0.8 
                rospy.logwarn('Set origin, %s: % 0.3f, % 0.3f' % (self.name, self.state.angle, self.state.radius))
                self.iInitial += 1
                a = 1.0 #1.0/self.iInitial
                self.imgRoi2MaskedPolarCroppedWindowedInitial = (1.0-a)*self.imgRoi2MaskedPolarCroppedWindowedInitial + a*self.imgRoi2MaskedPolarCroppedWindowed
                #self.windowTest.set_image(self.imgRoi2MaskedPolarCroppedWindowedInitial)
                self.stateMin.angle += self.state.angle
                self.stateMax.angle += self.state.angle
                self.state.angle = 0.0
                self.stateMin.radius += self.state.radius-self.params[self.name]['radius_axial']
                self.stateMax.radius += self.state.radius-self.params[self.name]['radius_axial']
                self.state.radius = self.params[self.name]['radius_axial']

        #self.windowTest.set_image(np.roll(self.imgRoi2MaskedPolar, -int(aShift), 1))
        self.state.mass = 0.0
        
        
    # update_state_using_features()
    # Update the bodypart translation & rotation.
    #
    def update_state_using_features(self):
        imgNow = self.imgRoi2MaskedPolarCroppedWindowed
        imgPrior = self.imgRoi2MaskedPolarCroppedWindowedPrev
        RT = cv2.estimateRigidTransform(imgNow.astype(np.uint8), imgPrior.astype(np.uint8), fullAffine=False)
            
        # Get the (x,y) translation between successive images.
        (iShift, jShift) = (RT[0,2], RT[1,2])
        ptCartesian = np.array([jShift, iShift])
        ptErr = self.state.pt - self.ptCOM
        self.state.pt += ptCartesian #- 0.005*ptErr # Correct for drift error due to integrating the shifts; pull gently toward the COM.
        self.state.angle += np.arctan2(RT[1,0], RT[0,0])
        
#         # Get the rotation & expansion between successive images.
#         if (self.imgRoi2MaskedPolarCroppedWindowed is not None):
#             (rShift, aShift) = self.get_shift_from_phasecorr(self.imgRoi2MaskedPolarCroppedWindowed, self.imgRoi2MaskedPolarCroppedWindowedPrev) # as [angle,radius]
#             dAngle = aShift * np.abs(2*self.angle_wedge) / self.imgRoi2MaskedPolarCroppedWindowed.shape[1]
#             dRadius = rShift
#             self.state.angle += dAngle

        self.state.mass = 0.0
        
        
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, header, image):
        self.iCount += 1
        
        if (self.stampPrev is not None):
            self.dt = header.stamp - self.stampPrev
        else:
            self.dt = rospy.Time(0)
        self.stampPrev = header.stamp
        
        if (self.params[self.name]['track']):
            if (not self.bInitializedMasks):
                self.create_mask(image.shape)
                
            self.update_roi(image)
            self.update_background()
            self.update_polar()
            self.update_center_of_mass()
            self.update_state_using_phasecorr()
            #self.update_state_using_features()
            
            if (self.bInitializedMasks):
                self.bInitialized = True
    
    
    # hit_object()
    # Get the UI object, if any, that the mouse is on.    
    def hit_object(self, ptMouse):
        tag = None
        
        # Check for handle hits.
        if (self.params[self.name]['track']):
            for tagHandle,handle in self.handles.iteritems():
                if (handle.hit_test(ptMouse)):
                    tag = tagHandle
                    break
        else:
            tagHandle,handle = ('center',self.handles['center'])
            if (handle.hit_test(ptMouse)):
                tag = tagHandle
            
                
        return (self.name, tag)
    

    def draw_handles(self, image):
        # Draw all handle points, or only just the center handle.
        if (self.params[self.name]['track']):
            for tagHandle,handle in self.handles.iteritems():
                handle.draw(image)
        else:
            tagHandle,handle = ('center',self.handles['center'])
            handle.draw(image)

    
    # draw()
    # Draw the outline.
    #
    def draw(self, image):
        self.draw_handles(image)

        if (self.params[self.name]['track']):
            a=1.0 # Amount to amplify the visual display.
            ptCenter0_i = (int(self.params[self.name]['x']), int(self.params[self.name]['y']))
            
            pt = self.R.dot([self.state.radius*np.cos(self.state.angle), 
                             self.state.radius*np.sin(self.state.angle)]) 
            ptState_i = (int(a*pt[0]+self.params[self.name]['x']), 
                         int(a*pt[1]+self.params[self.name]['y'])) 
            
            # Draw the outline.
            cv2.ellipse(image,
                        (int(self.params[self.name]['x']), int(self.params[self.name]['y'])),
                        (int(self.params[self.name]['radius_axial']), int(self.params[self.name]['radius_ortho'])),
                        np.rad2deg(self.angleBodyOutward_i),
                        np.rad2deg(self.angle_wedge),
                        np.rad2deg(-self.angle_wedge),
                        self.bgra, 
                        1)

            cv2.ellipse(image,
                        (int(self.params[self.name]['x']), int(self.params[self.name]['y'])),
                        (2*int(self.params[self.name]['radius_axial']), 2*int(self.params[self.name]['radius_ortho'])),
                        np.rad2deg(self.angleBodyOutward_i),
                        np.rad2deg(self.angle_wedge),
                        np.rad2deg(-self.angle_wedge),
                        self.bgra_dim, 
                        1)
    
            # Draw wedge.        
            cv2.line(image, self.ptWedge1, ptCenter0_i, self.bgra, 1)
            cv2.line(image, self.ptWedge2, ptCenter0_i, self.bgra, 1)
            cv2.line(image, self.ptWedge1_2x, self.ptWedge1, self.bgra_dim, 1)
            cv2.line(image, self.ptWedge2_2x, self.ptWedge2, self.bgra_dim, 1)

    
            # Draw the bodypart center of mass.    
#             ptCOM_i = (int(a*self.ptCOM[0]+self.params[self.name]['x']), int(a*self.ptCOM[1]+self.params[self.name]['y'])) 
#             cv2.ellipse(image,
#                         ptCOM_i,
#                         (2,2),
#                         0,
#                         0,
#                         360,
#                         bgra_dict['blue'], 
#                         1)
            
            # Draw the bodypart state position.
            cv2.ellipse(image,
                        ptState_i,
                        (2,2),
                        0,
                        0,
                        360,
                        self.bgra_state, 
                        1)
            
            # Draw the bodypart state angle.
#             r = 20
#             pt0 = np.array(ptState_i) + (r*np.array([np.cos(self.state.angle), np.sin(self.state.angle)])).astype(int).clip((0,0),(image.shape[1],image.shape[0])) 
#             pt1 = np.array(ptState_i) - (r*np.array([np.cos(self.state.angle), np.sin(self.state.angle)])).astype(int).clip((0,0),(image.shape[1],image.shape[0])) 
#             cv2.line(image,
#                      tuple(pt0),
#                      tuple(pt1),
#                      self.bgra_state, 
#                      1)
            
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
    
            # Draw line from hinge to center.        
            cv2.line(image, ptHinge_i, ptState_i, self.bgra_state, 1)
            
            self.windowPolar.show()
            self.windowBG.show()
            self.windowFG.show()
            self.windowTest.show()
        
# end class Bodypart

    

###############################################################################
###############################################################################
class Wing(object):
    def __init__(self, name='right', params={}, color='black'):
        self.name = name
        
        self.roi                = None
        self.ravelMaskRoi       = None
        self.ravelAnglesRoi_b   = None
        self.imgRoiBackground   = None
        self.imgRoi             = None
        self.imgFullBackground  = None
        
        self.bins               = None
        self.intensities        = None
        self.shape = (np.inf, np.inf)
        
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
        self.stampPrev = None
        self.dt = rospy.Time(0)
        
        self.windowBG = ImageWindow(False, self.name+'BG')
        self.windowFG = ImageWindow(False, self.name+'FG')

        self.handles = {'hinge':Handle(np.array([0,0]), self.bgra),
                        'angle_hi':Handle(np.array([0,0]), self.bgra),
                        'angle_lo':Handle(np.array([0,0]), self.bgra),
                        'inner':Handle(np.array([0,0]), self.bgra)
                        }

        self.set_params(params)

        # services, for live histograms
        self.service_intensity = rospy.Service('wing_intensity_'+name, float32list, self.serve_intensity_callback)
        self.service_bins      = rospy.Service('wing_bins_'+name, float32list, self.serve_bins_callback)
        self.service_edges     = rospy.Service('wing_edges_'+name, float32list, self.serve_edges_callback)
        
    
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
        if self.name == 'right':
            angle_i  =  angle_b + self.angleBody_i + np.pi/2.0
        else: # left
            angle_i  = -angle_b + self.angleBody_i + np.pi/2.0 + np.pi
             
        angle_i = (angle_i+np.pi) % (2.0*np.pi) - np.pi
        return angle_i
        

    # transform_angle_b_from_i()
    # Transform an angle from the camera image frame to the fly frame.
    #
    def transform_angle_b_from_i(self, angle_i):
        if self.name == 'right':
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
        self.rc_background = self.params['rc_background']
        self.ptHinge     = np.array([self.params[self.name]['hinge']['x'], self.params[self.name]['hinge']['y']]).astype(np.float64)
        resolution_min   = 1.1*np.sqrt(2.0)/self.params[self.name]['radius_inner'] # Enforce at least sqrt(2) pixels wide at inner radius, with a little fudge buffer.
        nbins            = int((2.0*np.pi)/max(self.params['resolution_radians'], resolution_min)) + 1
        self.bins        = np.linspace(-np.pi, np.pi, nbins).astype(np.float64)
        self.intensities = np.zeros(len(self.bins), dtype=np.float64)
        self.angleBody_i = self.get_bodyangle_i()

        (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.params[self.name]['angle_lo'], self.params[self.name]['angle_hi'])
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

        angle_lo = (self.params[self.name]['angle_lo'] + np.pi) % (2.0*np.pi) - np.pi
        angle_hi = (self.params[self.name]['angle_hi'] + np.pi) % (2.0*np.pi) - np.pi

        angle_min = min(angle_lo, angle_hi)
        angle_max = max(angle_lo, angle_hi)
        
        self.iBinsValid = list(np.where((angle_min <= self.bins) * (self.bins <= angle_max))[0])
        
        if (len(self.iBinsValid)==0):
            self.iBinsValid = [0] # TODO: make this the proper bin to match the angle. 

        self.update_handle_points()
        self.imgRoiBackground = None

        self.windowBG.set_enable(self.params['extra_windows'] and self.params[self.name]['track'] and self.params[self.name]['subtract_bg'])
        self.windowFG.set_enable(self.params['extra_windows'] and self.params[self.name]['track'])
        
        
    def get_bodyangle_i(self):
        angle_i = get_angle_from_points_i(np.array([self.params['abdomen']['x'], self.params['abdomen']['y']]), 
                                          np.array([self.params['head']['x'], self.params['head']['y']]))
        angleBody  = (angle_i + np.pi) % (2.0*np.pi) - np.pi
        return angleBody 
        
                
    # create_stroke_mask()
    # Create a mask of valid wingstroke areas.
    #
    def create_stroke_mask(self, shape):
        self.shape = shape
        
        # Create the wing mask.
        mask = np.zeros(shape, dtype=np.uint8)
        (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.params[self.name]['angle_lo'], self.params[self.name]['angle_hi'])

        ptCenter = tuple(self.ptHinge.astype(int))
        cv2.ellipse(mask,
                    ptCenter,
                    (int(self.params[self.name]['radius_outer']), int(self.params[self.name]['radius_outer'])),
                    0,
                    np.rad2deg(angle_lo_i),
                    np.rad2deg(angle_hi_i),
                    255, 
                    cv.CV_FILLED)
        cv2.circle(mask,
                    ptCenter,
                    int(self.params[self.name]['radius_inner']),
                    0,
                    cv.CV_FILLED)
        

        # Find the ROI of the mask.
        xSum = np.sum(mask, 0)
        ySum = np.sum(mask, 1)
        xNonzero = np.where(xSum>0)[0]
        yNonzero = np.where(ySum>0)[0]
        if (len(xNonzero) > 0) and (len(yNonzero) > 0):
            xMin = xNonzero[0]  - 2
            xMax = xNonzero[-1] + 3
            yMin = yNonzero[0]  - 2
            yMax = yNonzero[-1] + 3
        
            self.roi = np.array([xMin, yMin, xMax, yMax])
            self.ravelMaskRoi = np.ravel(mask[yMin:yMax, xMin:xMax])
        else:
            self.roi = None
            self.ravelMaskRoi = None
        
    
    # create_angle_mask()
    # Create an image where each pixel value is the angle from the hinge, in body coordinates.                    
    # 
    def create_angle_mask(self, shape):
        if (self.roi is not None):
            # Set up matrices of x and y coordinates.
            x = np.tile(np.array([range(shape[1])]).astype(np.float64)   - self.ptHinge[0], (shape[0], 1))
            y = np.tile(np.array([range(shape[0])]).astype(np.float64).T - self.ptHinge[1], (1, shape[1]))
    
            # Calc the angle at each pixel coordinate.
            imgAngles_i = np.arctan2(y,x) # Ranges [-pi,+pi]
            imgAnglesRoi_i = imgAngles_i[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
            
            imgAnglesRoi_b  = self.transform_angle_b_from_i(imgAnglesRoi_i)
            self.ravelAnglesRoi_b = np.ravel(imgAnglesRoi_b)
        else:
            self.ravelAnglesRoi_b = None
            
                   

    # assign_pixels_to_bins()
    # Put every pixel of the ROI into one of the bins.
    #
    def assign_pixels_to_bins(self):
        # Create empty bins.
        iPixelsRoi = [[] for i in range(len(self.bins))]

        # Put each iPixel into an appropriate bin.
        if (self.ravelAnglesRoi_b is not None):            
            for iPixel, angle in enumerate(self.ravelAnglesRoi_b):
                if self.ravelMaskRoi[iPixel]:
                    iBinBest = np.argmin(np.abs(self.bins - angle))
                    iPixelsRoi[iBinBest].append(iPixel)
        
        # Convert to numpy array.
        self.iPixelsRoi = np.array(np.zeros(len(iPixelsRoi), dtype=object))
        for k in range(len(iPixelsRoi)):
            self.iPixelsRoi[k] = np.array(iPixelsRoi[k], dtype=int)
#         for i in self.iPixelsRoi:
#             rospy.logwarn('%s: %s, %s' % (self.name, i, len(i)))
                
         
    # set_background()
    # Set the given image as the background image.
    #                
    def set_background(self, image):
        self.imgFullBackground = image.astype(np.float32)
        self.imgRoiBackground = None
        
        
    def update_background(self):
        if (self.imgRoiBackground is None) and (self.params[self.name]['subtract_bg']) and (self.roi is not None):
            self.imgRoiBackground = copy.deepcopy(self.imgFullBackground[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]])
        
        dt = max(0, self.dt.to_sec())
        alphaBackground = 1.0 - np.exp(-dt / self.rc_background)

        if (self.imgRoiBackground is not None) and (self.imgRoi is not None):
            if (self.imgRoiBackground.size==self.imgRoi.size):
                cv2.accumulateWeighted(self.imgRoi.astype(np.float32), self.imgRoiBackground, alphaBackground)
            else:
                self.imgRoiBackground = None
                self.imgRoi = None
        
        self.windowBG.set_image(self.imgRoiBackground)

        
    def update_roi(self, image):
        # Only use the ROI that covers the stroke rect.
        if (self.roi is not None):
            self.imgRoi_0 = image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]

        # Background Subtraction.
        if (self.params[self.name]['subtract_bg']):
            if (self.imgRoiBackground is not None):
                self.imgRoi = cv2.absdiff(self.imgRoi_0, self.imgRoiBackground.astype(np.uint8))
                 
        else:
            self.imgRoi = self.imgRoi_0
        
        self.windowFG.set_image(self.imgRoi)


    # update_intensity_function()
    # Update the list of intensities corresponding to the bin angles.
    #            
    def update_intensity_function(self):
        if (self.imgRoi is not None) and (self.ravelMaskRoi is not None) and (self.intensities is not None):
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
        (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.params[self.name]['angle_lo'], self.params[self.name]['angle_hi'])
        self.handles['angle_hi'].pt = (self.ptHinge + self.params[self.name]['radius_outer'] * np.array([self.cos['angle_hi'], 
                                                                                                   self.sin['angle_hi']]))
        self.handles['angle_lo'].pt = (self.ptHinge + self.params[self.name]['radius_outer'] * np.array([self.cos['angle_lo'], 
                                                                                                   self.sin['angle_lo']]))

        # Inner Radius.
        self.handles['inner'].pt = (self.ptHinge + self.params[self.name]['radius_inner'] * np.array([self.cos['angle_mid'], 
                                                                                                      self.sin['angle_mid']]))

    
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, header, image):
        if (self.params[self.name]['track']):
            self.update_roi(image)
            self.update_background()
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
#             rospy.logwarn('%s **********************************************' % self.name)
#             rospy.logwarn('iBinsValid: %s' % self.iBinsValid)
# #            rospy.logwarn('iPixelsRoi: %s' % self.iPixelsRoi[self.iBinsValid])
# #            rospy.logwarn('iPixelsRoi[%s]: %s' % (iMajor, self.iPixelsRoi[iMajor]))
# #            rospy.logwarn('iPixelsRoi[%s]: %s' % (iMinor, self.iPixelsRoi[iMinor]))
#             rospy.logwarn(self.ravelRoi[self.iPixelsRoi[iMajor]])
#             rospy.logwarn('%s, %s' % (self.intensities[iMajor], np.mean(self.ravelRoi[self.iPixelsRoi[iMajor]])))

#            cv2.imwrite('/home/ssafarik/test.png', self.imgRoi)
#            rospy.logwarn(self.intensities[self.iBinsValid])
#            rospy.logwarn(np.diff(self.intensitiesValid))
#            rospy.logwarn('%s: %s' % (self.name, (iMajor,iMinor)))
              
        # Output histograms of the wedges around the problem.
#         if (self.name=='right'):# and fTest=='b'):
#             ravelRoi = np.ravel(self.imgRoi)
#             edges = np.array(range(30,180,8))
#             rospy.logwarn('%s **********************************************' % self.name)
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
        
        if (self.params[self.name]['track']):
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
    # Draw the wing envelope and leading and trailing edges, onto the given image.
    #
    def draw(self, image):
        if (self.params[self.name]['track']) and (self.ptHinge is not None):
            (angle_lo_i, angle_hi_i) = self.get_angles_i_from_b(self.params[self.name]['angle_lo'], self.params[self.name]['angle_hi'])
            ptHinge = tuple(self.ptHinge.astype(int))

            # Inner circle
            cv2.ellipse(image, 
                        ptHinge,  
                        (int(self.params[self.name]['radius_inner']), int(self.params[self.name]['radius_inner'])),
                        0,
                        np.rad2deg(angle_lo_i),
                        np.rad2deg(angle_hi_i),
                        color=self.bgra,
                        thickness=self.thickness_inner,
                        )
            
            # Outer circle         
            cv2.ellipse(image, 
                        ptHinge,  
                        (int(self.params[self.name]['radius_outer']), int(self.params[self.name]['radius_outer'])),
                        0,
                        np.rad2deg(angle_lo_i),
                        np.rad2deg(angle_hi_i),
                        color=self.bgra,
                        thickness=self.thickness_outer,
                        )
            
            
            # Leading and trailing edges
            if (self.angle_leading_b is not None):
                (angle_leading_i, angle_trailing_i) = self.get_angles_i_from_b(self.angle_leading_b, self.angle_trailing_b)
                
                x0 = self.ptHinge[0] + self.params[self.name]['radius_inner'] * np.cos(angle_leading_i)
                y0 = self.ptHinge[1] + self.params[self.name]['radius_inner'] * np.sin(angle_leading_i)
                x1 = self.ptHinge[0] + self.params[self.name]['radius_outer'] * np.cos(angle_leading_i)
                y1 = self.ptHinge[1] + self.params[self.name]['radius_outer'] * np.sin(angle_leading_i)
                cv2.line(image, (int(x0),int(y0)), (int(x1),int(y1)), self.bgra, self.thickness_wing)
                
                if (self.params['n_edges']==2):
                    x0 = self.ptHinge[0] + self.params[self.name]['radius_inner'] * np.cos(angle_trailing_i)
                    y0 = self.ptHinge[1] + self.params[self.name]['radius_inner'] * np.sin(angle_trailing_i)
                    x1 = self.ptHinge[0] + self.params[self.name]['radius_outer'] * np.cos(angle_trailing_i)
                    y1 = self.ptHinge[1] + self.params[self.name]['radius_outer'] * np.sin(angle_trailing_i)
                    cv2.line(image, (int(x0),int(y0)), (int(x1),int(y1)), self.bgra, self.thickness_wing)
                
            self.draw_handles(image)
        
        self.windowBG.show()
        self.windowFG.show()

                        
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
        self.stampPrev = None
        self.dt = rospy.Time(0)
        
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
        except (rosparam.RosParamException, IndexError), e:
            rospy.logwarn('%s.  Using default values.' % e)
            self.params = {}
            
        defaults = {'filenameBackground':'~/strokelitude.png',
                    'image_topic':'/camera/image_raw',
                    'use_gui':True,                     # You can turn off the GUI to speed the framerate.
                    'extra_windows':True,               # Show the helpful extra windows.
                    'invertcolor':False,                # You want a light fly on a dark background.  Only needed if not using a background image.
                    'symmetric':True,                   # Forces the UI to remain symmetric.
                    'resolution_radians':0.0174532925,  # Coarser resolution will speed the framerate. 1 degree == 0.0174532925 radians.
                    'threshold_flight':0.1,
                    'scale_image':1.0,                  # Reducing the image scale will speed the framerate.
                    'n_edges':1,                        # Number of edges per wing to find.  1 or 2.
                    'rc_background':10.0,                # Time constant of the moving average background.
                    'head':   {'track':True,
                               'autozero':True,
                               'subtract_bg':False,        # Use background subtraction?
                               'x':300,
                               'y':150,
                               'radius_ortho':50,
                               'radius_axial':50,
                               'angle_wedge':0},
                    'abdomen':{'track':True,
                               'autozero':True,
                               'subtract_bg':False,     # Use background subtraction?
                               'x':300,
                               'y':250,
                               'radius_ortho':60,
                               'radius_axial':70,
                               'angle_wedge':0},
                    'left':   {'track':True,
                               'autozero':False,
                               'subtract_bg':True,         # Use background subtraction?
                               'hinge':{'x':100,
                                        'y':100},
                               'radius_outer':30,
                               'radius_inner':10,
                               'angle_hi':0.7854, 
                               'angle_lo':-0.7854},
                    'right':  {'track':True,
                               'autozero':False,
                               'subtract_bg':True,        # Use background subtraction?
                               'hinge':{'x':300,
                                        'y':100},
                               'radius_outer':30,
                               'radius_inner':10,
                               'angle_hi':0.7854, 
                               'angle_lo':-0.7854},

                    }
        self.set_dict_with_preserve(self.params, defaults)
        rospy.set_param('strokelitude', self.params)
        
        # initialize wings and body
        self.fly = Fly(self.params)
        
        # Background image.
        self.filenameBackground = os.path.expanduser(self.params['filenameBackground'])
        self.imgFullBackground  = cv2.imread(self.filenameBackground, cv.CV_LOAD_IMAGE_GRAYSCALE)
        if (self.imgFullBackground is not None):
            self.fly.set_background(self.imgFullBackground)
        
        self.scale = self.params['scale_image']
        self.bMousing = False
        
        self.nameSelected = None
        self.uiSelected = None
        self.stateSelected = None
        self.fly.update_handle_points()
        self.tPrev = rospy.Time.now().to_sec()
        self.hz = 0.0
        self.hzSum = 0.0
        self.iCount = 0
        
        # Publishers.
        self.pubCommand            = rospy.Publisher('strokelitude/command', MsgCommand)

        # Subscriptions.        
        self.subImageRaw           = rospy.Subscriber(self.params['image_topic'], Image, self.image_callback)
        self.subCommand            = rospy.Subscriber('strokelitude/command', MsgCommand, self.command_callback)

        self.w_gap = int(10 * self.scale)
        self.scaleText = 0.4 * self.scale
        self.fontface = cv2.FONT_HERSHEY_SIMPLEX

        # UI button specs.
        self.buttons = []
        x = int(1 * self.scale)
        y = int(1 * self.scale)
        btn = Button(pt=[x,y], scale=self.scale, type='pushbutton', name='exit', text='exit')
        self.buttons.append(btn)
        
        x = btn.right+1
        btn = Button(pt=[x,y], scale=self.scale, type='pushbutton', name='save_bg', text='saveBG')
        self.buttons.append(btn)
        
        x = btn.right+1
        btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='subtract_bg', text='subtractBG', state=self.params['right']['subtract_bg'])
        self.buttons.append(btn)

        x = btn.right+1
        btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='invertcolor', text='invert', state=self.params['invertcolor'])
        self.buttons.append(btn)

        x = btn.right+1
        btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='symmetry', text='symmetric', state=self.params['symmetric'])
        self.buttons.append(btn)

        x = btn.right+1
        btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='head', text='head', state=self.params['head']['track'])
        self.buttons.append(btn)

        x = btn.right+1
        btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='abdomen', text='abdomen', state=self.params['abdomen']['track'])
        self.buttons.append(btn)

        x = btn.right+1
        btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='wings', text='wings', state=self.params['right']['track'])
        self.buttons.append(btn)

        x = btn.right+1
        btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='extra_windows', text='windows', state=self.params['extra_windows'])
        self.buttons.append(btn)

        self.yToolbar = btn.bottom + 1


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
    def command_callback(self, msg):
        self.command = msg.command
        
        if (self.command == 'exit'):
            rospy.signal_shutdown('User requested exit.')
        
        
        if (self.command == 'save_background'):
            self.save_background()
            
        
        if (self.command == 'use_gui'):
            self.params['use_gui'] = (msg.arg1 > 0)
            
        
        if (self.command == 'help'):
            rospy.logwarn('The strokelitude/command topic accepts the following string commands:')
            rospy.logwarn('  help                 This message.')
            rospy.logwarn('  save_background      Save the instant camera image to disk for')
            rospy.logwarn('                       background subtraction.')
            rospy.logwarn('  use_gui #            Turn off|on the user windows (#=0|1).')
            rospy.logwarn('  exit                 Exit the program.')
            rospy.logwarn('')
            rospy.logwarn('You can send the above commands at the shell prompt via:')
            rospy.logwarn('rostopic pub -1 strokelitude/command StrokelitudeROS/MsgCommand commandtext N')
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
			paramsOut[bodypart]['radius_ortho'] = (paramsIn[bodypart]['radius_ortho']*scale)  
			paramsOut[bodypart]['radius_axial'] = (paramsIn[bodypart]['radius_axial']*scale)
			
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
        if (self.stampPrev is not None):
            self.dt = rosimage.header.stamp - self.stampPrev
        else:
            self.dt = rospy.Time(0)
        self.stampPrev = rosimage.header.stamp

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
#             if (self.params['subtract_bg']):
#                 if (self.imgFullBackground is not None):
#                     try:
#                         imgForeground = cv2.absdiff(self.imgCamera, self.imgFullBackground)
#                     except:
#                         imgForeground = self.imgCamera
#                         self.imgFullBackground = None
#                         rospy.logwarn('Please take a fresh background image.  The existing one is the wrong size or has some other problem.')
#                         
#                 else:
#                     imgForeground = self.imgCamera
#             else:
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

                # Output the framerate.
                if (not self.bMousing):
                    tNow = rospy.Time.now().to_sec()
                    dt = tNow - self.tPrev
                    self.tPrev = tNow
                    hzNow = 1/dt if (dt != 0.0) else 0.0
                    self.iCount += 1
                    if (self.iCount > 100):                     
                        a= 0.04
                        self.hz = (1-a)*self.hz + a*hzNow 
                    else:                                       
                        if (self.iCount>20):             # Get past the transient response.       
                            self.hzSum += hzNow                 
                        else:
                            self.hzSum = hzNow * self.iCount     
                            
                        self.hz = self.hzSum / self.iCount
                        
                    cv2.putText(imgOutput, '%5.1f Hz' % self.hz, (x, y_bottom), self.fontface, self.scaleText, bgra_dict['dark_yellow'] )
                    
                w_text = int(55 * self.scale)
                x += w_text+self.w_gap
            

                # Output the head state.
                if (self.params['head']['track']):
                    s = 'HEAD:% 7.4f' % (self.fly.head.state.angle)
                    w = 90
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, self.fly.head.bgra)
                    w_text = int(w * self.scale)
                    x += w_text+self.w_gap
                

                # Output the abdomen state.
                if (self.params['abdomen']['track']):
                    s = 'ABDOMEN:% 7.4f' % (self.fly.abdomen.state.angle)
                    w = 115
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, self.fly.abdomen.bgra)
                    w_text = int(w * self.scale)
                    x += w_text+self.w_gap
                

                # Output the wings state.
                if (self.params['right']['track']):
                    if (self.fly.wing_l.angle_amplitude is not None):
                        if (self.params['n_edges']==1):
                            s = 'L:% 7.4f' % (self.fly.wing_l.angle_leading_b)
                            w = 65
                        else:
                            s = 'L:% 7.4f,% 7.4f' % (self.fly.wing_l.angle_leading_b,self.fly.wing_l.angle_trailing_b)
                            w = 120
                        cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, self.fly.wing_l.bgra)
                        w_text = int(w * self.scale)
                        x += w_text+self.w_gap
                    
                    if (self.fly.wing_r.angle_amplitude is not None):
                        if (self.params['n_edges']==1):
                            s = 'R:% 7.4f' % (self.fly.wing_r.angle_leading_b)
                            w = 65
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
                        w_text = int(82 * self.scale)
                        x += w_text+self.w_gap
    
                        
                    # Output difference in WBA
                    if (self.fly.wing_l.angle_amplitude is not None) and (self.fly.wing_r.angle_amplitude is not None):
                        leftminusright = self.fly.wing_l.angle_amplitude - self.fly.wing_r.angle_amplitude
                        #s = 'L-R:% 7.1f' % np.rad2deg(leftminusright)
                        s = 'L-R:% 7.4f' % leftminusright
                        cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, bgra_dict['magenta'])
                        w_text = int(82 * self.scale)
                        x += w_text+self.w_gap
    
                        
                    # Output flight status
                    if (self.fly.wing_l.bFlying and self.fly.wing_r.bFlying):
                        s = 'FLIGHT'
                    else:
                        s = 'no flight'
                    
                    cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, bgra_dict['magenta'])
                    w_text = int(50 * self.scale)
                    x += w_text+self.w_gap
            
                # end if (self.params['right']['track'])


                # Display the image.
                cv2.imshow(self.window_name, imgOutput)
                cv2.waitKey(1)


            if (not self.bMousing):
                # Update the fly internals.
                self.fly.update(rosimage.header, imgForeground)
    
                # Publish the outputs.
                self.fly.publish()
            
            

    # save_background()
    # Save the current camera image as the background.
    #
    def save_background(self):
        self.fly.set_background(self.imgCamera)
        self.imgFullBackground = self.imgCamera
        rospy.logwarn ('Saving new background image %s' % self.filenameBackground)
        cv2.imwrite(self.filenameBackground, self.imgFullBackground)
    
    
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
        iButtonHit = None
        for iButton in range(len(self.buttons)):
            if (self.buttons[iButton].hit_test(ptMouse)):
                iButtonHit = iButton
            
        if (iButtonHit is not None):
            nameNearest = self.buttons[iButtonHit].name
            (tagHit,delim,bodypartHit) = nameNearest.partition('_')
            uiHit = self.buttons[iButtonHit].type
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
    
        
        return (bodypartHit, tagHit, uiHit, iButtonHit)
        
        
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

    
    def get_bodyangle_i(self):
        angle_i = get_angle_from_points_i(np.array([self.params['abdomen']['x'], self.params['abdomen']['y']]), 
                                          np.array([self.params['head']['x'], self.params['head']['y']]))
        angleBody  = (angle_i + np.pi) % (2.0*np.pi) - np.pi
        return angleBody 
        
                
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
                    


            if (tagSelected=='radius_axial'): 
                params[bodypartSelected]['radius_axial'] = max(float(np.linalg.norm(np.array([params[bodypartSelected]['x'],params[bodypartSelected]['y']]) - ptMouse))-10,0)
                params[bodypartSelected]['radius_ortho'] = params[bodypartSelected]['radius_axial'] # Force it to be circular. 
            if (tagSelected=='radius_ortho'): 
                params[bodypartSelected]['radius_ortho'] = float(np.linalg.norm(np.array([params[bodypartSelected]['x'],params[bodypartSelected]['y']]) - ptMouse))
            if (tagSelected=='angle_wedge'):
                ptBodyCenter_i = (np.array([self.params['head']['x'], self.params['head']['y']]) + np.array([self.params['abdomen']['x'], self.params['abdomen']['y']])) / 2
                angleBodyOutward_i = np.arctan2(params[bodypartSelected]['y']-ptBodyCenter_i[1], params[bodypartSelected]['x']-ptBodyCenter_i[0])

                cosAngleBodyOutward = np.cos(angleBodyOutward_i)
                sinAngleBodyOutward = np.sin(angleBodyOutward_i)
                R = np.array([[cosAngleBodyOutward, -sinAngleBodyOutward], [sinAngleBodyOutward, cosAngleBodyOutward]])
                r1 = params[bodypartSelected]['radius_axial']
                r2 = params[bodypartSelected]['radius_ortho']
                y = (ptMouse[1]-params[bodypartSelected]['y'])
                x = (ptMouse[0]-params[bodypartSelected]['x'])
                xy = np.dot(np.linalg.inv(R), np.array([x,y]))
                
                angle = float(-np.arctan2(xy[1]/r2, xy[0]/r1)) % (2*np.pi)
                #angle = float(-np.arctan2(xy[1], xy[0])) % (2*np.pi)
                if (np.abs(angle-np.pi)<0.02): # If within about 1 degree of pi or 0, then snap to those.
                    angle = np.pi
                if (np.abs(angle)<0.02):
                    angle = 0.0
                params[bodypartSelected]['angle_wedge'] = angle
                #rospy.logwarn(params[bodypartSelected]['angle_wedge'])


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
        ptMouse = np.array([x, y]).clip((0,0), (self.shapeImage[1],self.shapeImage[0]))

        # Keep track of which UI element is selected.
        if (event==cv.CV_EVENT_LBUTTONDOWN):
            self.bMousing = True
            
            # Get the name and ui nearest the current point.
            (bodypart, tag, ui, iButtonSelected) = self.hit_object(ptMouse)
            self.nameSelected = self.name_from_tagbodypart(tag,bodypart)
            self.tagSelected = tag
            self.bodypartSelected = bodypart
            self.uiSelected = ui
            self.iButtonSelected = iButtonSelected

            self.wingSelected = self.fly.wing_l if (self.bodypartSelected=='left') else self.fly.wing_r
            if (self.iButtonSelected is not None):
                self.stateSelected = self.buttons[self.iButtonSelected].state
            
            self.nameSelectedNow = self.nameSelected
            self.uiSelectedNow = self.uiSelected


        if (self.uiSelected=='pushbutton') or (self.uiSelected=='checkbox'):
            # Get the bodypart and ui tag nearest the mouse point.
            (bodypart, tag, ui, iButtonSelected) = self.hit_object(ptMouse)
            self.nameSelectedNow     = self.name_from_tagbodypart(tag,bodypart)
            self.tagSelectedNow      = tag
            self.bodypartSelectedNow = bodypart
            self.uiSelectedNow       = ui
            self.iButtonSelectedNow  = iButtonSelected


            # Set selected button to 'down', others to 'up'.
            for iButton in range(len(self.buttons)):
                if (self.buttons[iButton].type=='pushbutton'):
                    if (iButton==self.iButtonSelectedNow==self.iButtonSelected) and not (event==cv.CV_EVENT_LBUTTONUP):
                        self.buttons[iButton].state = True # 'down'
                    else:
                        self.buttons[iButton].state = False # 'up'
                        
            # Set the checkbox.
            if (self.uiSelected=='checkbox'):
                if (self.nameSelected == self.nameSelectedNow):
                    self.buttons[self.iButtonSelected].state = not self.stateSelected # Set it to the other state when we're on the checkbox.
                else:
                    self.buttons[self.iButtonSelected].state = self.stateSelected # Set it to the original state when we're off the checkbox.

            
                        
        # end if (self.uiSelected=='pushbutton'):

                        
        elif (self.uiSelected=='handle'):
            # Set the new params.
            self.update_params_from_handle(self.bodypartSelected, self.tagSelected, ptMouse.clip((0,self.yToolbar), (self.shapeImage[1],self.shapeImage[0])))
            self.fly.set_params(self.scale_params(self.params, self.scale))
        
            # Save the results.
            if (event==cv.CV_EVENT_LBUTTONUP):
                self.set_dict_with_preserve(self.params, rospy.get_param('strokelitude'))
                rospy.set_param('strokelitude', self.params)
                rosparam.dump_params(self.parameterfile, 'strokelitude')
                
                self.fly.create_masks(self.shapeImage)
                self.set_dict_with_preserve(self.params, rospy.get_param('strokelitude'))
                rospy.set_param('strokelitude', self.params)
            
        # end if (self.uiSelected=='handle'):
            

        if (event==cv.CV_EVENT_LBUTTONUP):
            # If the mouse is on the same button at mouseup, then do the action.
            if (self.uiSelected=='pushbutton'):
                if (self.nameSelected == self.nameSelectedNow == 'save_bg'):
                    self.pubCommand.publish(MsgCommand('save_background',0))

                elif (self.nameSelected == self.nameSelectedNow == 'exit'):
                    self.pubCommand.publish(MsgCommand('exit',0))
                    
                    
            elif (self.uiSelected=='checkbox'):
                if (self.nameSelected == self.nameSelectedNow):
                    self.buttons[self.iButtonSelected].state = not self.stateSelected
                else:
                    self.buttons[self.iButtonSelected].state = self.stateSelected


                if (self.nameSelected == self.nameSelectedNow == 'invertcolor'):
                    self.params['invertcolor'] = self.buttons[self.iButtonSelected].state
                    
                elif (self.nameSelected == self.nameSelectedNow == 'symmetry'):
                    self.params['symmetric'] = self.buttons[self.iButtonSelected].state
                    
                elif (self.nameSelected == self.nameSelectedNow == 'subtract_bg'):
                    if (self.imgFullBackground is None):
                        self.buttons[iButtonSelected].state = False
                        rospy.logwarn('No background image.  Cannot use background subtraction.')

                    self.params['left']['subtract_bg'] = self.buttons[iButtonSelected].state
                    self.params['right']['subtract_bg'] = self.buttons[iButtonSelected].state
                    
                elif (self.nameSelected == self.nameSelectedNow == 'head'):
                    self.params['head']['track'] = self.buttons[iButtonSelected].state
                    
                elif (self.nameSelected == self.nameSelectedNow == 'abdomen'):
                    self.params['abdomen']['track'] = self.buttons[iButtonSelected].state

                elif (self.nameSelected == self.nameSelectedNow == 'wings'):
                    self.params['right']['track'] = self.buttons[iButtonSelected].state
                    self.params['left']['track']  = self.buttons[iButtonSelected].state

                elif (self.nameSelected == self.nameSelectedNow == 'extra_windows'):
                    self.params['extra_windows'] = self.buttons[iButtonSelected].state


            self.fly.set_params(self.scale_params(self.params, self.scale))
            self.fly.create_masks(self.shapeImage)

            self.bMousing           = False
            self.nameSelected       = None
            self.nameSelectedNow    = None
            self.uiSelected         = None
            self.uiSelectedNow      = None
            self.iButtonSelected    = None
            self.iButtonSelectedNow = None

            
                
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
    main.command_callback(MsgCommand('help',0))
    rospy.logwarn('**************************************************************************')
    rospy.logwarn('')
    rospy.logwarn('')

    main.run()
