#!/usr/bin/env python
#! coding=latin-1
from __future__ import division
import roslib; roslib.load_manifest('Kinefly')
import rospy
import rosparam

import copy
#import cProfile
import cv
import cv2
import numpy as np
import os
import sys
import threading
import dynamic_reconfigure.server

from setdict import SetDict

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Header, String
from Kinefly.srv import SrvWingdata, SrvWingdataResponse, SrvTipdata, SrvTipdataResponse
from Kinefly.msg import MsgFlystate, MsgState, MsgCommand
from Kinefly.cfg import kineflyConfig

# Define button sides for drawing.
SIDE_TOP = 1
SIDE_BOTTOM = 2
SIDE_LEFT = 4
SIDE_RIGHT = 8
SIDE_ALL = (SIDE_TOP | SIDE_BOTTOM | SIDE_LEFT | SIDE_RIGHT)

# gImageTime = 0.0
gbShowMasks = False

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


# Intersection of two lines, given two points on each line.
def get_intersection(pt1a, pt1b, pt2a, pt2b):
    # Line 1
    x1 = pt1a[0]
    y1 = pt1a[1]
    x2 = pt1b[0]
    y2 = pt1b[1]
    
    # Line 2
    x3 = pt2a[0]
    y3 = pt2a[1]
    x4 = pt2b[0]
    y4 = pt2b[1]
    
    den = (x1*y3 - x3*y1 - x1*y4 - x2*y3 + x3*y2 + x4*y1 + x2*y4 - x4*y2)
    if (den != 0.0):
        x = (x1*x3*y2 - x2*x3*y1 - x1*x4*y2 + x2*x4*y1 - x1*x3*y4 + x1*x4*y3 + x2*x3*y4 - x2*x4*y3) / den
        y = (x1*y2*y3 - x2*y1*y3 - x1*y2*y4 + x2*y1*y4 - x3*y1*y4 + x4*y1*y3 + x3*y2*y4 - x4*y2*y3) / den
    else:
        x = x3   
        y = y3   
    
    return np.array([x,y])
            
            
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
    
    
# clip a point to the image shape.  pt is (x,y), and shape is (yMax+1, xMax+1)
def clip_pt(pt, shape):
    return (clip(pt[0], 0, shape[1]-1), clip(pt[1], 0, shape[0]-1))
    
    
class Struct:
    pass

        

###############################################################################
###############################################################################
# A class to just show an image.
class ImageWindow(object):
    def __init__(self, bEnable, name):
        self.image = None
        self.shape = (100,100)
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
            self.shape = image.shape
        else:
            self.image = None
        
        
    def show(self):
        if (self.bEnable):
            if (self.image is not None) and (self.image.size>0):
                img = self.image
            else:
                img = np.zeros(self.shape)
            
            (h,w) = img.shape
            if (h>0) and (w>0):
                cv2.imshow(self.name, img)
        
        
    def set_enable(self, bEnable):
        if (self.bEnable and not bEnable):
            cv2.destroyWindow(self.name)
            
        if (not self.bEnable and bEnable):
            cv2.namedWindow(self.name)
            
        self.bEnable = bEnable
        
        
    
###############################################################################
###############################################################################
class Button(object):
    def __init__(self, name=None, text=None, pt=None, rect=None, scale=1.0, type='pushbutton', state=False, sides=SIDE_ALL):
        self.name = name
        self.pt = pt
        self.rect = rect
        self.scale = scale
        self.type = type            # 'pushbutton' or 'checkbox'
        self.state = state
        self.sides = sides
        
        self.widthCheckbox = 10*self.scale
        self.set_text(text)
        
        # Set the colors of the button types that don't change based on their state.
        if (self.type=='pushbutton'):
            self.colorFill = bgra_dict['gray']
            self.colorText = bgra_dict['white']
            self.colorCheck = bgra_dict['black']
        if (self.type=='checkbox'):
            self.colorOuter = bgra_dict['white']
            self.colorInner = bgra_dict['black']
            self.colorHilight = bgra_dict['light_gray']
            self.colorLolight = bgra_dict['light_gray']
            self.colorFill = bgra_dict['light_gray']
            self.colorText = bgra_dict['white']
            self.colorCheck = bgra_dict['black']
        elif (self.type=='static'):
            self.colorOuter = bgra_dict['white']
            self.colorInner = bgra_dict['black']
            self.colorHilight = bgra_dict['light_gray']
            self.colorLolight = bgra_dict['light_gray']
            self.colorFill = bgra_dict['light_gray']
            self.colorText = bgra_dict['white']
            self.colorCheck = bgra_dict['black']


    # hit_test()
    # Detect if the mouse is on the button.
    #
    def hit_test(self, ptMouse):
        if (self.rect[0] <= ptMouse[0] <= self.rect[0]+self.rect[2]) and (self.rect[1] <= ptMouse[1] <= self.rect[1]+self.rect[3]):
            return True
        else:
            return False
        
    # set_pos()
    # Set the button to locate at the given upper-left point, or to the given rect.
    #
    def set_pos(self, pt=None, rect=None):
        if (rect is not None):
            self.rect = rect
            
        elif (pt is not None):
            self.pt = pt
            self.rect = [0,0,0,0]
            self.rect[0] = self.pt[0]
            self.rect[1] = self.pt[1]
            self.rect[2] = self.sizeText[0] + 6
            self.rect[3] = self.sizeText[1] + 6
            if (self.type=='checkbox'):
                self.rect[2] += int(self.widthCheckbox + 4)
                
        
            # Position adjustments for missing sides.
            (l,r,t,b) = (0,0,0,0)
            if (self.sides & SIDE_LEFT)==0:
                l = -1
            if (self.sides & SIDE_RIGHT)==0:
                r = 1
            if (self.sides & SIDE_TOP)==0:
                t = -1
            if (self.sides & SIDE_BOTTOM)==0:
                b = 1


            # Set the locations of the various button pieces.        
            
            # The colorOuter lines.
            self.ptLT0 = (self.rect[0]-1,              self.rect[1]-1)
            self.ptRT0 = (self.rect[0]+self.rect[2]+1, self.rect[1]-1)
            self.ptLB0 = (self.rect[0]-1,              self.rect[1]+self.rect[3]+1)
            self.ptRB0 = (self.rect[0]+self.rect[2]+1, self.rect[1]+self.rect[3]+1)
    
            # The colorInner lines.
            self.ptLT1 = (self.rect[0]+l,                 self.rect[1]+t)
            self.ptRT1 = (self.rect[0]+self.rect[2]+r,    self.rect[1]+t)
            self.ptLB1 = (self.rect[0]+l,                 self.rect[1]+self.rect[3]+b)
            self.ptRB1 = (self.rect[0]+self.rect[2]+r,    self.rect[1]+self.rect[3]+b)
    
            # The colorLolight & colorHilight lines.
            self.ptLT2 = (self.rect[0]+1+2*l,              self.rect[1]+1+2*t)
            self.ptRT2 = (self.rect[0]+self.rect[2]-1+2*r, self.rect[1]+1+2*t)
            self.ptLB2 = (self.rect[0]+1+2*l,              self.rect[1]+self.rect[3]-1+2*b)
            self.ptRB2 = (self.rect[0]+self.rect[2]-1+2*r, self.rect[1]+self.rect[3]-1+2*b)
    
            # Fill color.
            self.ptLT3 = (self.rect[0]+2+3*l,              self.rect[1]+2+3*t)
            self.ptRT3 = (self.rect[0]+self.rect[2]-2+3*r, self.rect[1]+2+3*t)
            self.ptLB3 = (self.rect[0]+2+3*l,              self.rect[1]+self.rect[3]-2+3*b)
            self.ptRB3 = (self.rect[0]+self.rect[2]-2+3*r, self.rect[1]+self.rect[3]-2+3*b)

    
            self.left   = self.ptLT0[0]
            self.top    = self.ptLT0[1]
            self.right  = self.ptRB0[0]
            self.bottom = self.ptRB0[1]
            
            if (self.type=='pushbutton') or (self.type=='static'):
                self.ptCenter = (int(self.rect[0]+self.rect[2]/2),                          int(self.rect[1]+self.rect[3]/2))
                self.ptText = (self.ptCenter[0] - int(self.sizeText[0]/2) - 1, 
                               self.ptCenter[1] + int(self.sizeText[1]/2) - 1)
            elif (self.type=='checkbox'):
                self.ptCenter = (int(self.rect[0]+self.rect[2]/2+(self.widthCheckbox+4)/2), int(self.rect[1]+self.rect[3]/2))
                self.ptText = (self.ptCenter[0] - int(self.sizeText[0]/2) - 1 + 2, 
                               self.ptCenter[1] + int(self.sizeText[1]/2) - 1)

            self.ptText0 = (self.ptText[0], self.ptText[1])
                
            self.ptCheckCenter = (int(self.ptLT3[0] + 2 + self.widthCheckbox/2), self.ptCenter[1])
            self.ptCheckLT     = (int(self.ptCheckCenter[0]-self.widthCheckbox/2), int(self.ptCheckCenter[1]-self.widthCheckbox/2))
            self.ptCheckRT     = (int(self.ptCheckCenter[0]+self.widthCheckbox/2), int(self.ptCheckCenter[1]-self.widthCheckbox/2))
            self.ptCheckLB     = (int(self.ptCheckCenter[0]-self.widthCheckbox/2), int(self.ptCheckCenter[1]+self.widthCheckbox/2))
            self.ptCheckRB     = (int(self.ptCheckCenter[0]+self.widthCheckbox/2), int(self.ptCheckCenter[1]+self.widthCheckbox/2))

        else:
            rospy.logwarn('Error setting button size and position.')


    # set_text()
    # Set the button text, and size the button to fit.
    #
    def set_text(self, text):
        self.text = text
        (sizeText,rv) = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.4*self.scale, 1)
        self.sizeText = (sizeText[0],    sizeText[1])
        
        self.set_pos(pt=self.pt, rect=self.rect)

                
    # draw_button()
    # Draw a 3D shaded button with text.
    # rect is (left, top, width, height), increasing y goes down.
    def draw(self, image):
        # Pushbutton colors change depending on state.
        if (self.type=='pushbutton'):
            if (not self.state): # 'up'
                self.colorOuter = bgra_dict['white']
                self.colorInner = bgra_dict['black']
                self.colorHilight = bgra_dict['light_gray']
                self.colorLolight = bgra_dict['dark_gray']
                self.ptText0 = (self.ptText[0], self.ptText[1])
            else:
                self.colorOuter = bgra_dict['white']
                self.colorInner = bgra_dict['black']
                self.colorHilight = bgra_dict['dark_gray']
                self.colorLolight = bgra_dict['light_gray']
                self.ptText0 = (self.ptText[0]+2, self.ptText[1]+2)
            

        # Draw outer, inner, hilights & lolights.
        if (self.sides & SIDE_LEFT):
            cv2.line(image, self.ptLT0, self.ptLB0, self.colorOuter)
            cv2.line(image, self.ptLT1, self.ptLB1,  self.colorInner)
            cv2.line(image, self.ptLT2, self.ptLB2, self.colorHilight)
        if (self.sides & SIDE_TOP):
            cv2.line(image, self.ptLT0, self.ptRT0, self.colorOuter)
            cv2.line(image, self.ptLT1, self.ptRT1,  self.colorInner)
            cv2.line(image, self.ptLT2, self.ptRT2, self.colorHilight)
        if (self.sides & SIDE_RIGHT):
            cv2.line(image, self.ptRT0, self.ptRB0, self.colorOuter)
            cv2.line(image, self.ptRT1, self.ptRB1,  self.colorInner)
            cv2.line(image, self.ptRT2, self.ptRB2, self.colorLolight)
        if (self.sides & SIDE_BOTTOM):
            cv2.line(image, self.ptLB0, self.ptRB0, self.colorOuter)
            cv2.line(image, self.ptLB1, self.ptRB1,  self.colorInner)
            cv2.line(image, self.ptLB2, self.ptRB2, self.colorLolight)

        # Draw the fill.
        cv2.rectangle(image, self.ptLT3, self.ptRB3, self.colorFill, cv.CV_FILLED)
        
        # Draw the checkbox.
        if (self.type=='checkbox'):
            cv2.rectangle(image, self.ptCheckLT, self.ptCheckRB, self.colorCheck, 1)
            if (self.state): # 'down'
                cv2.line(image, self.ptCheckLT, self.ptCheckRB, self.colorCheck)
                cv2.line(image, self.ptCheckRT, self.ptCheckLB, self.colorCheck)

        # Draw the text.
        cv2.putText(image, self.text, self.ptText0, cv2.FONT_HERSHEY_SIMPLEX, 0.4*self.scale, self.colorText)
        
# end class Button                
    
            
###############################################################################
###############################################################################
class Handle(object):
    def __init__(self, pt=np.array([0,0]), color=bgra_dict['white'], name=None):
        self.pt = pt
        self.name = name
        self.scale = 1.0

        self.color = color
        self.radiusDraw = 3
        self.radiusHit = 6


    def hit_test(self, ptMouse):
        d = np.linalg.norm(self.pt - ptMouse)
        if (d < self.radiusHit):
            return True
        else:
            return False
        

    # draw()
    # Draw a handle.
    # 
    def draw(self, image):
        cv2.circle(image, tuple(self.pt.astype(int)),  self.radiusDraw, self.color, cv.CV_FILLED)
        
        #ptText = self.pt+np.array([5,5])
        #cv2.putText(image, self.name, tuple(ptText.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4*self.scale, self.color)
        
# end class Handle
                

###############################################################################
###############################################################################
class PolarTransforms(object):
    def __init__(self):
        self._transforms = {}

        
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
    # nRho:         Number of radial (vert) pixels in the output image.
    # amplifyRho:   Multiplier of vertical output dimension instead of nRho.
    # nTheta:       Number of angular (horiz) pixels in the output image.
    # amplifyTheta: Multiplier of horizontal output dimension instead of nTheta.
    # theta_0:      Starting angle.
    # theta_1:      Ending angle.
    # scale:        0.0=Include to the nearest side (all pixels in output image are valid); 1.0=Include to the 
    #               farthest corner (some pixels in output image from outside the input image). 
    #
    # Credit to http://machineawakening.blogspot.com/2012/02
    #
    def transform_polar_log(self, image, i_0, j_0, nRho=None, amplifyRho=1.0, nTheta=None, amplifyTheta=1.0, theta_0=0.0, theta_1=2.0*np.pi, scale=0.0):
        (i_n, j_n) = image.shape[:2]
        
        i_c = max(i_0, i_n - i_0)
        j_c = max(j_0, j_n - j_0)
        d_c = (i_c ** 2 + j_c ** 2) ** 0.5 # Distance to the farthest image corner.
        d_s = min(i_0, i_n-i_0, j_0, j_n-j_0)  # Distance to the nearest image side.
        d = scale*d_c + (1.0-scale)*d_s
        
        if (nRho == None):
            nRho = int(np.ceil(d*amplifyRho))
        
        if (nTheta == None):
            #nTheta = int(np.ceil(j_n * amplifyTheta))
            nTheta = int(amplifyTheta * 2*np.pi*np.sqrt((i_n**2 + j_n**2)/2)) # Approximate circumference of ellipse in the roi.
        
        dRho = np.log(d) / nRho
        
        (pt, ij) = self._get_transform_polar_log(i_0, j_0, 
                                           i_n, j_n, 
                                           nRho, dRho, 
                                           nTheta, theta_0, theta_1)
        imgTransformed = np.zeros((nRho, nTheta) + image.shape[2:], dtype=image.dtype)
        imgTransformed[pt] = image[ij]

        return imgTransformed


    def _get_transform_polar_elliptical(self, i_0, j_0, i_n, j_n, (raxial, rortho), drStrip, angleEllipse, nRho, nTheta, theta_0, theta_1, rClip):
        nTheta = max(1,nTheta)
        transform = self._transforms.get((i_0, j_0, i_n, j_n, nRho, drStrip, nTheta, theta_0, theta_1, rClip))
    
        if (transform == None):
            i_k     = []
            j_k     = []
            rho_k   = []
            theta_k = []

#             # Convert from circular angles to elliptical angles.
#             xy_e = np.array([raxial*np.cos(theta_0), rortho*np.sin(theta_0)])
#             theta_0e = np.arctan2(xy_e[1], xy_e[0])
#             xy_e = np.array([raxial*np.cos(theta_1), rortho*np.sin(theta_1)])
#             theta_1e = np.arctan2(xy_e[1], xy_e[0])

            # Radii of the inner and outer ellipses.
            raxial_outer = raxial + drStrip
            rortho_outer = raxial + drStrip
            raxial_inner = raxial - drStrip
            rortho_inner = raxial - drStrip
            
            R = np.array([[np.cos(-angleEllipse), -np.sin(-angleEllipse)], 
                          [np.sin(-angleEllipse), np.cos(-angleEllipse)]])    
            dTheta = (theta_1 - theta_0) / nTheta
            
            # Step through the wedge from theta_0 to theta_1.
            for iTheta in range(0, nTheta):
                theta = theta_0 + iTheta * dTheta

                # Convert from circular angles to elliptical angles.
                xy_e = np.array([raxial*np.cos(theta), rortho*np.sin(theta)])
                theta_e = np.arctan2(xy_e[1], xy_e[0])

                # Radii of ellipses at this angle.
                #rho_e = np.linalg.norm([2*raxial*np.cos(theta), 2*rortho*np.sin(theta)])
                rho_e_inner = np.linalg.norm([raxial_inner*np.cos(theta), rortho_inner*np.sin(theta)])
                rho_e_outer = np.linalg.norm([raxial_outer*np.cos(theta), rortho_outer*np.sin(theta)])
                dRho = (rho_e_outer - rho_e_inner) / nRho
                
                # Step along the radius from the inner to the outer ellipse.  rClip is a clip factor on range [0,1]
                for iRho in range(0, int(np.ceil(rClip*nRho))):
                    rho = rho_e_inner + iRho * dRho

                    # i,j points on an upright ellipse.
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
            self._transforms[i_0, j_0, i_n, j_n, nRho, drStrip, nTheta, theta_0, theta_1, rClip] = transform
    
        return transform
    
    # transform_polar_elliptical()
    # Remap an image into linear-polar coordinates, where (i_0,j_0) is the (y,x) origin in the original image.
    # raxial:       Ellipse radius at angleEllipse.
    # rortho:       Ellipse radius orthogonal to angleEllipse.
    # dradiusStrip: Half-width of the mask strip in the axial direction.
    # nRho:         Number of radial (vert) pixels in the output image.
    # amplifyRho:   Multiplier of vertical output dimension instead of nRho.
    # rClip:        Fraction of the the image that should be calculated vertically.  i.e. Limit r when calculating pixels.
    # angleEllipse: Axis of ellipse rotation.
    # theta_0:      Circular angle to one side of ellipse angle.
    # theta_1:      Circular angle to other side of ellipse angle.
    # nTheta:       Number of angular (horiz) pixels in the output image.
    # amplifyTheta: Multiplier of horizontal output dimension instead of nTheta.
    #
    #
    def transform_polar_elliptical(self, image, i_0, j_0, raxial=None, rortho=None, dradiusStrip=None, nRho=None, amplifyRho=1.0, rClip=0.0, angleEllipse=0.0, theta_0=-np.pi, theta_1=np.pi, nTheta=None, amplifyTheta=1.0):
        (i_n, j_n) = image.shape[:2]
        if (raxial is None):
            raxial = i_n / 2
        if (rortho is None):
            rortho = j_n / 2
            
        if (dradiusStrip is None):
            dradiusStrip = raxial-5
        
#         i_c = max(i_0, i_n-i_0)
#         j_c = max(j_0, j_n-j_0)
#         
#         d_c = (i_c ** 2 + j_c ** 2) ** 0.5            # Distance to the farthest image corner.
#         d_s_near = min(i_0, i_n-i_0, j_0, j_n-j_0)    # Distance to the nearest image side.
#         d_s_far = max(i_0, i_n-i_0, j_0, j_n-j_0)     # Distance to the farthest image side.

        
        # Radii of the inner and outer ellipses.
        raxial_outer = raxial + dradiusStrip
        rortho_outer = raxial + dradiusStrip
        raxial_inner = raxial - dradiusStrip
        rortho_inner = raxial - dradiusStrip

            
        # Nearest nonzero distance of point (i_0,j_0) to a side.
        #d_sides = [i_0, i_n-i_0, j_0, j_n-j_0]
        #d_nonzero = d_sides[np.where(d_sides>0)[0]]
        #d_s_0 = d_nonzero.min()
        

        # Distance to nearest point of outer elliptical wedge of point (i_0,j_0).
        d_e_raxial_outer = raxial_outer # Distance to the axial point.
        d_e_rortho_outer = rortho_outer # Distance to the ortho point.
        d_e_wedge_outer = np.linalg.norm([raxial_outer*np.cos(theta_0), rortho_outer*np.sin(theta_0)]) # Distance to the theta_0 wedge point.
        if (np.abs(theta_0) >= np.pi/2.0):
            d_e_min_outer = min(d_e_raxial_outer, d_e_wedge_outer, d_e_rortho_outer) # Nearest of the three.
        else:
            d_e_min_outer = min(d_e_raxial_outer, d_e_wedge_outer) # Nearest of the three.

        # Distance to nearest point of inner elliptical wedge of point (i_0,j_0).
        d_e_raxial_inner = raxial_inner # Distance to the axial point.
        d_e_rortho_inner = rortho_inner # Distance to the ortho point.
        d_e_wedge_inner = np.linalg.norm([raxial_inner*np.cos(theta_0), rortho_inner*np.sin(theta_0)]) # Distance to the theta_0 wedge point.
        if (np.abs(theta_0) >= np.pi/2.0):
            d_e_min_inner = min(d_e_raxial_inner, d_e_wedge_inner, d_e_rortho_inner) # Nearest of the three.
        else:
            d_e_min_inner = min(d_e_raxial_inner, d_e_wedge_inner) # Nearest of the three.

        
        d = d_e_min_outer - d_e_min_inner 
        
        
        # Convert from circular angles to elliptical angles.
        xy_e = np.array([raxial*np.cos(theta_0), rortho*np.sin(theta_0)])
        theta_0e = np.arctan2(xy_e[1], xy_e[0])
        xy_e = np.array([raxial*np.cos(theta_1), rortho*np.sin(theta_1)])
        theta_1e = np.arctan2(xy_e[1], xy_e[0])
        
        
        if (nRho == None):
            nRho = int(np.ceil(d * amplifyRho))
        
        # Number of angular steps depends on the number of pixels along the elliptical arc, ideally 1 step per pixel.
        if (nTheta == None):
            circumference = 2*np.pi*np.sqrt(((raxial)**2 + (rortho)**2)) # Approximate circumference of ellipse.
            fraction_wedge = np.abs(theta_1e - theta_0e)/(2*np.pi) # Approximate fraction that is the wedge of interest.
            nTheta = int(amplifyTheta * circumference * fraction_wedge)  
        
        (pt, ij) = self._get_transform_polar_elliptical(i_0, j_0, 
                                           i_n, j_n,
                                           (raxial, rortho),
                                           dradiusStrip,
                                           angleEllipse, 
                                           nRho, 
                                           nTheta, theta_0, theta_1, rClip)
        imgTransformed = np.zeros((nRho, nTheta) + image.shape[2:], dtype=image.dtype)
        if (len(pt[0])>0):
            imgTransformed[pt] = image[ij]
        else:
            rospy.logwarn('No points transformed.')
        
        return imgTransformed

# End class PolarTransforms


###############################################################################
###############################################################################
class PhaseCorrelation(object):
    # get_shift()
    # Calculate the coordinate shift between two images.
    #
    def get_shift(self, imgA, imgB):
        rv = np.array([0.0, 0.0])
        if (imgA is not None) and (imgB is not None) and (imgA.shape==imgB.shape):
            # Phase correlation.
            A  = cv2.dft(imgA)
            B  = cv2.dft(imgB)
            AB = cv2.mulSpectrums(A, B, flags=0, conjB=True)
            normAB = cv2.norm(AB)
            if (normAB != 0.0):
                crosspower = AB / normAB
                shift = cv2.idft(crosspower)
                shift0  = np.roll(shift,  int(shift.shape[0]/2), 0)
                shift00 = np.roll(shift0, int(shift.shape[1]/2), 1) # Roll the matrix so 0,0 goes to the center of the image.
                
                # Get the coordinates of the maximum shift.
                kShift = np.argmax(shift00)
                (iShift,jShift) = np.unravel_index(kShift, shift00.shape)
    
                # Get weighted centroid of a region around the peak, for sub-pixel accuracy.
                w = 7
                r = int((w-1)/2)
                i0 = clip(iShift-r, 0, shift00.shape[0]-1)
                i1 = clip(iShift+r, 0, shift00.shape[0]-1)+1
                j0 = clip(jShift-r, 0, shift00.shape[1]-1)
                j1 = clip(jShift+r, 0, shift00.shape[1]-1)+1
                peak = shift00[i0:i1].T[j0:j1].T
                moments = cv2.moments(peak, binaryImage=False)
                           
                if (moments['m00'] != 0.0):
                    iShiftSubpixel = moments['m01']/moments['m00'] + float(i0)
                    jShiftSubpixel = moments['m10']/moments['m00'] + float(j0)
                else:
                    iShiftSubpixel = float(shift.shape[0])/2.0
                    jShiftSubpixel = float(shift.shape[1])/2.0
                
                # Accomodate the matrix roll we did above.
                iShiftSubpixel -= float(shift.shape[0])/2.0
                jShiftSubpixel -= float(shift.shape[1])/2.0
    
                # Convert unsigned shifts to signed shifts. 
                height = float(shift00.shape[0])
                width  = float(shift00.shape[1])
                iShiftSubpixel  = ((iShiftSubpixel+height/2.0) % height) - height/2.0
                jShiftSubpixel  = ((jShiftSubpixel+width/2.0) % width) - width/2.0
                
                rv = np.array([iShiftSubpixel, jShiftSubpixel])

            
        return rv
        
        
###############################################################################
###############################################################################
class WindowFunctions(object):
    # create_hanning()
    # Create a 2D Hanning window function.
    #
    def create_hanning(self, shape):
        (height,width) = shape
        wfn = np.ones(shape, dtype=np.float32)
        if (height>1) and (width>1):
            for i in range(width):
                 for j in range(height):
                     x = 2*np.pi*i/(width-1) # x ranges 0 to 2pi across the image width
                     y = 2*np.pi*j/(height-1) # y ranges 0 to 2pi across the image height
                     wfn[j][i] = 0.5*(1-np.cos(x)) * 0.5*(1-np.cos(y))
                 
        return wfn


    # create_tukey()
    # Create a 2D Tukey window function.
    #
    def create_tukey(self, shape):
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

# End class WindowFunctions


    
###############################################################################
###############################################################################
# Contains the base behavior for tracking a bodypart via pixel intensity.
#
class IntensityTrackedBodypart(object):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        self.name        = name
        self.bEqualizeHist = bEqualizeHist

        self.bValidMask = False

        self.bgra        = bgra_dict[color]
        self.bgra_dim    = tuple(0.5*np.array(bgra_dict[color]))
        self.bgra_state  = bgra_dict[color]#bgra_dict['red']
        self.pixelmax    = 255.0

        self.shape     = (np.inf, np.inf)
        self.ptCenter_i = np.array([0,0])
        self.roi       = None
        
        self.maskRoi           = None
        self.sumMask = 1.0
        
        self.dt = np.inf

        self.params = {}
        self.handles = {'center':Handle(np.array([0,0]), self.bgra, name='center'),
                        'radius1':Handle(np.array([0,0]), self.bgra, name='radius1'),
                        'radius2':Handle(np.array([0,0]), self.bgra, name='radius2')
                        }
        
        # Region of interest images.
        self.imgFullBackground                  = None
        self.imgRoiBackground                   = None
        self.imgRoi                             = None # Untouched roi image.
        self.imgRoiFg                           = None # Background subtracted.
        self.imgRoiFgMasked                     = None

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
        self.ptCenter_i = np.array([self.params['gui'][self.name]['center']['x'], self.params['gui'][self.name]['center']['y']])
        
        self.cosAngle = np.cos(self.params['gui'][self.name]['angle'])
        self.sinAngle = np.sin(self.params['gui'][self.name]['angle'])

        # Turn on/off the extra windows.
        self.windowBG.set_enable(self.params['gui']['windows'] and self.params['gui'][self.name]['track'] and self.params['gui'][self.name]['subtract_bg'])
        self.windowFG.set_enable(self.params['gui']['windows'] and self.params['gui'][self.name]['track'])

        # Refresh the handle points.
        self.update_handle_points()



    # create_mask()
    # Create elliptical wedge masks, and window functions.
    #
    def create_mask(self, shape):
        if (not self.bValidMask):
            x     = int(self.params['gui'][self.name]['center']['x'])
            y     = int(self.params['gui'][self.name]['center']['y'])
            r1    = int(self.params['gui'][self.name]['radius1'])
            r2    = int(self.params['gui'][self.name]['radius2'])
            angle = self.params['gui'][self.name]['angle']
            bgra  = bgra_dict['white']
            
            # Create the mask.
            mask = np.zeros(shape, dtype=np.uint8)
            cv2.ellipse(mask, (x, y), (r1, r2), int(np.rad2deg(angle)), 0, 360, bgra, cv.CV_FILLED)
            self.windowMask.set_image(mask)
            
            # Find the ROI of the mask.
            b=0 # Border
            xSum = np.sum(mask, 0)
            ySum = np.sum(mask, 1)
            xMask = np.where(xSum>0)[0]
            yMask = np.where(ySum>0)[0]
            
            if (len(xMask)>0) and (len(yMask)>0): 
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
                
                self.roi = np.array([xMin, yMin, xMax, yMax])
                self.maskRoi = mask[yMin:yMax, xMin:xMax]
                self.sumMask = np.sum(self.maskRoi).astype(np.float32)
        
                self.bValidMask = True
            else:
                rospy.logwarn('%s: Empty mask.' % self.name)
                self.bValidMask = False
        
        
    # set_background()
    # Set the given image as the background image.
    #                
    def set_background(self, image):
        self.imgFullBackground = image.astype(np.float32)
        self.imgRoiBackground = None
        
        
    # invert_background()
    # Invert the color of the background image.
    #                
    def invert_background(self):
        if (self.imgRoiBackground is not None):
            self.imgRoiBackground = 255-self.imgRoiBackground
        
        
    def update_background(self):
        alphaBackground = 1.0 - np.exp(-self.dt / self.rc_background)

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
                
        self.windowBG.set_image(self.imgRoiBackground)
        

    def update_roi(self, image, bInvertColor):
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
                    max2 = np.max(self.imgRoiFg)
                    self.imgRoiFg *= (255.0/float(max2))
                
            # Apply the mask.
            if (self.maskRoi is not None):
                self.imgRoiFgMasked = cv2.bitwise_and(self.imgRoiFg, self.maskRoi)
    
            self.windowFG.set_image(self.imgRoiFgMasked) 
        

            
    # update_handle_points()
    # Update the dictionary of handle point names and locations.
    # Compute the various handle points.
    #
    def update_handle_points (self):
        x = self.params['gui'][self.name]['center']['x']
        y = self.params['gui'][self.name]['center']['y']
        radius1 = self.params['gui'][self.name]['radius1']
        radius2 = self.params['gui'][self.name]['radius2']
        angle = self.params['gui'][self.name]['angle']
        
        
        self.handles['center'].pt  = np.array([x, y])
        self.handles['radius1'].pt = np.array([x, y]) + radius1 * np.array([self.cosAngle, self.sinAngle])
        self.handles['radius2'].pt = np.array([x, y]) + radius2 * np.array([self.sinAngle,-self.cosAngle])

        
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, dt, image, bInvertColor):
        self.iCount += 1
        
        self.dt = dt
        
        if (self.params['gui'][self.name]['track']):
            self.create_mask(image.shape)
            self.update_background()
            self.update_roi(image, bInvertColor)


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
                
        return (self.name, tag)
    

    def draw_handles(self, image):
        # Draw all handle points, or only just the hinge handle.
        if (self.params['gui'][self.name]['track']):
            for tagHandle,handle in self.handles.iteritems():
                handle.draw(image)

    
    # draw()
    # Draw the outline.
    #
    def draw(self, image):
        self.draw_handles(image)

        if (self.params['gui'][self.name]['track']):
            x = int(self.params['gui'][self.name]['center']['x'])
            y = int(self.params['gui'][self.name]['center']['y'])
            radius1 = int(self.params['gui'][self.name]['radius1'])
            radius2 = int(self.params['gui'][self.name]['radius2'])

            # Draw the outer arc.
            cv2.ellipse(image,
                        (x, y),
                        (radius1, radius2),
                        np.rad2deg(self.params['gui'][self.name]['angle']),
                        0,
                        360,
                        self.bgra, 
                        1)
    
            # Show the extra windows.
            self.windowBG.show()
            self.windowFG.show()
            self.windowMask.show()
                
# end class IntensityTrackedBodypart


###############################################################################
###############################################################################
# Contains the common behavior for tracking a bodypart via a polar coordinates transform, e.g. Head, Abdomen, or Wing.
#
class PolarTrackedBodypart(object):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        self.name           = name
        self.bEqualizeHist  = bEqualizeHist

        self.bValidMask = False

        self.color          = color
        self.bgra           = bgra_dict[color]
        self.bgra_dim       = tuple(0.5*np.array(bgra_dict[color]))
        self.bgra_state     = bgra_dict[color]#bgra_dict['red']
        self.pixelmax       = 255.0
                        
        self.shape          = (np.inf, np.inf)
        self.ptHinge_i      = np.array([0,0])
        self.roi            = None
        
        self.create_wfn               = WindowFunctions().create_tukey
        self.wfnRoi                   = None
        self.wfnRoiMaskedPolarCropped = None
        
        self.maskRoi        = None
        self.sumMask        = 1.0
        
        self.dt = np.inf
        self.polartransforms = PolarTransforms()

        self.params = {}
        self.handles = {'hinge':Handle(np.array([0,0]), self.bgra, name='hinge'),
                        'angle_hi':Handle(np.array([0,0]), self.bgra, name='angle_hi'),
                        'angle_lo':Handle(np.array([0,0]), self.bgra, name='angle_lo'),
                        'radius_inner':Handle(np.array([0,0]), self.bgra, name='radius_inner')
                        }
        
        # Region of interest images.
        self.imgFullBackground                  = None
        self.imgRoiBackground                   = None
        self.imgRoi                             = None # Untouched roi image.
        self.imgRoiFg                           = None # Background subtracted.
        self.imgRoiFgMasked                     = None
        self.imgRoiFgMaskedPolar                = None
        self.imgRoiFgMaskedPolarCropped         = None
        self.imgRoiFgMaskedPolarCroppedWindowed = None
        self.imgComparison                      = None

        # Extra windows.
        self.windowBG         = ImageWindow(False, self.name+'BG')
        self.windowFG         = ImageWindow(False, self.name+'FG')
        self.windowPolar      = ImageWindow(False, self.name+'Polar')
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

        self.ptHinge_i = np.array([self.params['gui'][self.name]['hinge']['x'], self.params['gui'][self.name]['hinge']['y']])
        
        # Compute the body-outward-facing angle, which is the angle from the body center to the bodypart hinge.
#         pt1 = [params['head']['hinge']['x'], params['head']['hinge']['y']]
#         pt2 = [params['abdomen']['hinge']['x'], params['abdomen']['hinge']['y']]
#         pt3 = [params['gui']['left']['hinge']['x'], params['gui']['left']['hinge']['y']]
#         pt4 = [params['right']['hinge']['x'], params['right']['hinge']['y']]
#         ptBodyCenter_i = get_intersection(pt1,pt2,pt3,pt4)
#         self.angleOutward_i = float(np.arctan2(self.ptHinge_i[1]-ptBodyCenter_i[1], self.ptHinge_i[0]-ptBodyCenter_i[0]))

        # Compute the body-outward-facing angle, which is the angle to the current point from the relative part's point (e.g. left hinge to right hinge).
        nameRelative = {'head':'abdomen', 
                        'abdomen':'head', 
                        'left':'right', 
                        'right':'left'}
        self.angleOutward_i = float(np.arctan2(self.params['gui'][self.name]['hinge']['y']-self.params['gui'][nameRelative[self.name]]['hinge']['y'], 
                                               self.params['gui'][self.name]['hinge']['x']-self.params['gui'][nameRelative[self.name]]['hinge']['x']))
        self.angleOutward_b = self.angleOutward_i - self.angleBody_i

        
        cosAngleOutward_i = np.cos(self.angleOutward_i)
        sinAngleOutward_i = np.sin(self.angleOutward_i)
        self.R = np.array([[cosAngleOutward_i, -sinAngleOutward_i], 
                           [sinAngleOutward_i, cosAngleOutward_i]])

        # Turn on/off the extra windows.
        self.windowPolar.set_enable(self.params['gui']['windows'] and self.params['gui'][self.name]['track'])
        self.windowBG.set_enable(self.params['gui']['windows'] and self.params['gui'][self.name]['track'] and self.params['gui'][self.name]['subtract_bg'])
        self.windowFG.set_enable(self.params['gui']['windows'] and self.params['gui'][self.name]['track'])

        self.angle_hi_i = self.transform_angle_i_from_b(self.params['gui'][self.name]['angle_hi'])
        self.angle_lo_i = self.transform_angle_i_from_b(self.params['gui'][self.name]['angle_lo'])
        
        # Refresh the handle points.
        self.update_handle_points()


    # transform_angle_b_from_p()
    # Transform an angle from the bodypart frame to the fly body frame.
    #
    def transform_angle_b_from_p(self, angle_p):
        angle_b =  self.sense * angle_p + self.angleOutward_b
        return angle_b


    # transform_angle_p_from_b()
    # Transform an angle from the fly body frame to the bodypart frame.
    #
    def transform_angle_p_from_b(self, angle_b):
        angle_p =  self.sense * (angle_b - self.angleOutward_b)
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
        angle_i = get_angle_from_points_i(np.array([self.params['gui']['abdomen']['hinge']['x'], self.params['gui']['abdomen']['hinge']['y']]), 
                                          np.array([self.params['gui']['head']['hinge']['x'], self.params['gui']['head']['hinge']['y']]))
        #angleBody_i  = (angle_i + np.pi) % (2.0*np.pi) - np.pi
        angleBody_i  = float(angle_i)
        return angleBody_i 
        
                
    # create_mask()
    # Create elliptical wedge masks, and window functions.
    #
    def create_mask(self, shape):
        if (not self.bValidMask):
            # Create the mask (for polar images the line-of-interest is at the midpoint).
            mask = np.zeros(shape, dtype=np.uint8)
            
            # Args for the two ellipse calls.
            x = int(self.params['gui'][self.name]['hinge']['x'])
            y = int(self.params['gui'][self.name]['hinge']['y'])
            r_outer = int(np.ceil(self.params['gui'][self.name]['radius_outer']))
            r_inner = int(np.floor(self.params['gui'][self.name]['radius_inner']))-1
    
            hi = int(np.ceil(np.rad2deg(self.angle_hi_i)))
            lo = int(np.floor(np.rad2deg(self.angle_lo_i)))
            
            # Draw the mask.
            cv2.ellipse(mask, (x, y), (r_outer, r_outer), 0, hi, lo, bgra_dict['white'], cv.CV_FILLED)
            cv2.ellipse(mask, (x, y), (r_inner, r_inner), 0, 0, 360, bgra_dict['black'], cv.CV_FILLED)
            mask = cv2.dilate(mask, np.ones([3,3])) # Make the mask one pixel bigger to account for pixel aliasing.
            self.windowMask.set_image(mask)
            
            # Find the ROI of the mask.
            xSum = np.sum(mask, 0)
            ySum = np.sum(mask, 1)
            xMask = np.where(xSum>0)[0]
            yMask = np.where(ySum>0)[0]
            
            if (len(xMask)>0) and (len(yMask)>0): 
                # Dilate with a border.
                b=0 # Border
                xMin0 = np.where(xSum>0)[0][0]  - b
                xMax0 = np.where(xSum>0)[0][-1] + b+1
                yMin0 = np.where(ySum>0)[0][0]  - b
                yMax0 = np.where(ySum>0)[0][-1] + b+1
                
                # Clip border to image edges.
                xMin = np.max([0,xMin0])
                yMin = np.max([0,yMin0])
                xMax = np.min([xMax0, shape[1]-1])
                yMax = np.min([yMax0, shape[0]-1])
                
                self.roi = np.array([xMin, yMin, xMax, yMax])
                self.maskRoi = mask[yMin:yMax, xMin:xMax]
                self.sumMask = np.sum(self.maskRoi).astype(np.float32)
    
        
                self.i_0 = self.params['gui'][self.name]['hinge']['y'] - self.roi[1]
                self.j_0 = self.params['gui'][self.name]['hinge']['x'] - self.roi[0]
                
                self.wfnRoi = None
                self.wfnRoiMaskedPolarCropped = None
                self.bValidMask = True
        
        
                # Find where the mask might be clipped.  First, draw an unclipped ellipse.        
                delta = 1
                pts =                cv2.ellipse2Poly((x, y), (r_outer, r_outer), 0, hi, lo, delta)
                pts = np.append(pts, cv2.ellipse2Poly((x, y), (r_inner, r_inner), 0, hi, lo, delta), 0)
                #pts = np.append(pts, [list of line1 pixels connecting the two arcs], 0) 
                #pts = np.append(pts, [list of line2 pixels connecting the two arcs], 0) 
        
        
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
            else:
                rospy.logwarn('%s: Empty mask.' % self.name)        
        
    # set_background()
    # Set the given image as the background image.
    #                
    def set_background(self, image):
        self.imgFullBackground = image.astype(np.float32)
        self.imgRoiBackground = None
        
        
    # invert_background()
    # Invert the color of the background image.
    #                
    def invert_background(self):
        if (self.imgRoiBackground is not None):
            self.imgRoiBackground = 255-self.imgRoiBackground
        
        
    def update_background(self):
        alphaBackground = 1.0 - np.exp(-self.dt / self.rc_background)

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
        

            
    def update_polar(self):
        if (self.imgRoiFg is not None):
            # Apply the mask.
            if (self.maskRoi is not None):
                self.imgRoiFgMasked = cv2.bitwise_and(self.imgRoiFg, self.maskRoi) #self.imgRoiFg#
                #self.imgRoiFgMasked  = cv2.multiply(self.imgRoiFg.astype(np.float32), self.wfnRoi)

            xMax = self.imgRoiFg.shape[1]-1
            yMax = self.imgRoiFg.shape[0]-1
            
            theta_0a = self.angle_lo_i - self.angleOutward_i
            theta_1a = self.angle_hi_i - self.angleOutward_i
            
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
                                                     angleEllipse=self.angleOutward_i,
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
                if (self.wfnRoi is None) or (self.imgRoiFg.shape != self.wfnRoi.shape):
                    self.wfnRoi = self.create_wfn(self.imgRoiFg.shape)
                
                if (self.wfnRoiMaskedPolarCropped is None) or (self.imgRoiFgMaskedPolarCropped.shape != self.wfnRoiMaskedPolarCropped.shape):
                    self.wfnRoiMaskedPolarCropped = self.create_wfn(self.imgRoiFgMaskedPolarCropped.shape)
                
                self.imgRoiFgMaskedPolarCroppedWindowed = cv2.multiply(self.imgRoiFgMaskedPolarCropped.astype(np.float32), self.wfnRoiMaskedPolarCropped)

            
        # Show the image.
        img = self.imgRoiFgMaskedPolarCroppedWindowed
        #img = self.imgRoiFgMaskedPolarCropped
        self.windowPolar.set_image(img)
        

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
            self.create_mask(image.shape)
            self.update_background()
            self.update_roi(image, bInvertColor)
            self.update_polar()


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
            self.windowPolar.show()
            self.windowMask.show()
                
# end class PolarTrackedBodypart



###############################################################################
###############################################################################
class WingbeatDetector(object):
    def __init__(self, fw_min, fw_max):
        self.n = 64
        self.buffer = np.zeros([2*self.n, 2]) # Holds intensities & framerates.
        self.set(fw_min, fw_max)
        
                
    def set(self, fw_min, fw_max):
        self.i = 0
        
        # Set the desired passband.
        self.fw_min = fw_min
        self.fw_max = fw_max
        self.fw_center = (fw_min + fw_max) / 2.0

        # Framerate needed to measure the desired band.        
        self.fs_dict = self.fs_dict_from_wingband(fw_min, fw_max)
        if (len(self.fs_dict['fs_range_list'])>0):
            (self.fs_lo, self.fs_hi) = self.fs_dict['fs_range_list'][0]
        else:
            (self.fs_lo, self.fs_hi) = (0.0, 0.0)
        

    def warn(self):    
        rospy.logwarn('Note: The wingbeat detector is set to measure wingbeat frequencies in the ')
        rospy.logwarn('range [%0.1f, %0.1f] Hz.  To make a valid measurement, the camera ' % (self.fw_min, self.fw_max))
        rospy.logwarn('framerate must be in, and stay in, one of the following ranges:')
        rospy.logwarn(self.fs_dict['fs_range_list'])


    # fs_dict_from_wingband()
    # Compute the camera frequency range [fs_lo, fs_hi] required to undersample the given wingbeat frequency band.
    # fw_min:    Lower bound for wingbeat frequency.
    # fw_max:    Upper bound for wingbeat frequency.
    # fs_lo:   Lower bound for sampling frequency.
    # fs_hi:   Upper bound for sampling frequency.
    #
    # Returns a list of all the possible framerate ranges: [[lo,hi],[lo,hi],...]
    #
    def fs_dict_from_wingband(self, fw_min, fw_max):
        fs_dict = {}
        fs_range_list = []
        m_list = []
        
        bw = fw_max - fw_min
        m = 1
        while (True):
            fs_hi =     (2.0 * self.fw_center - bw) / m
            fs_lo = max((2.0 * self.fw_center + bw) / (m+1), 2*bw)
            if (2*bw < fs_hi):
                fs_range_list.append([fs_lo, fs_hi])
                m_list.append(m)
            else:
                break
            m += 1
        
        # Put the list into low-to-high order.
        fs_range_list.reverse()
        m_list.reverse()

        fs_dict['fs_range_list'] = fs_range_list
        fs_dict['m_list'] = m_list
        
        return fs_dict
    
    
    # wingband_from_fs()
    # Compute the frequency band we can measure with the given undersampling framerate.
    # fs:          Sampling frequency, i.e. camera framerate.
    # fw_center:    Desired center of the wingbeat frequencies. 
    # fw_min:        Lower frequency of band that can be measured containing fw_center.
    # fw_max:        Upper frequency of band.
    # bReversed:   If True, then the aliased frequencies will be in reverse order.
    #
    def wingband_from_fs(self, fs_lo, fs_hi, fw_center):
        # TODO: Note that this function does not work right.  It should return the inverse of self.fs_dict_from_wingband().
        bw_lo = fs_lo / 2.0
        
        fs = (fs_lo+fs_hi)/2.0
        
        if (fs != 0.0):
            # Find which multiples of fs/2 to use.
            n = np.round(fw_center / (fs/2))
            if (n*fs/2 < fw_center):
                fw_min = n*fs/2
                fw_max = (n+1)*fs/2
            else:
                fw_min = (n-1)*fs/2
                fw_max = n*fs/2
    
            bReversed = ((n%2)==1)
        else:
            fw_min = 0.0
            fw_max = 1.0
            bReversed = False
            
        return (fw_min, fw_max, bReversed)
        
        
    # get_baseband_range()
    # Get the baseband wingbeat alias frequency range for the given sampling frequency and m.
    def get_baseband_range(self, fs, m):
        
        kMax = int(np.floor(2*self.fw_center / m))
        
        # If m is even, then step down from fw in multiples of fs, keeping the last valid range above zero.
        if (m%2==0):
            for k in range(kMax):
                fbb_min_tmp = self.fw_min - k * fs   
                fbb_max_tmp = self.fw_max - k * fs
                if (fbb_min_tmp >= 0):
                    fbb_min = fbb_min_tmp   
                    fbb_max = fbb_max_tmp
                else:
                    break
                   
        else: # if m is odd, then step up from -fw in multiples of fs, keeping the first valid range above zero.
            for k in range(kMax):
                fbb_min = -self.fw_min + k * fs   
                fbb_max = -self.fw_max + k * fs
                if (fbb_max >= 0):
                    break
            
        return (fbb_min, fbb_max)
        
        
    # get_baseband_range_from_framerates()
    # Check if buffered framerates have stayed within an allowable 
    # range to make a valid measurement of the wingbeat passband.
    # If so, then compute the baseband frequency range
    #
    # Returns (bValid, [fbb_min, fbb_max])
    #
    def get_baseband_range_from_framerates(self, framerates):
        bValid = False

        fs_lo = np.min(framerates)
        fs_hi = np.max(framerates)
        #(fw_min, fw_max, bReversed) = self.wingband_from_fs(fs_lo, fs_hi, self.fw_center)
        for iRange in range(len(self.fs_dict['fs_range_list'])):
            (fs_min, fs_max) = self.fs_dict['fs_range_list'][iRange]
            if (fs_min < fs_lo < fs_hi < fs_max):
                bValid = True
                break

        m = self.fs_dict['m_list'][iRange]

        if (bValid):
            fs = np.mean(framerates)
            (fbb_min, fbb_max) = self.get_baseband_range(fs, m)
        else:
            fbb_min = 0.0
            fbb_max = np.Inf

        
        return (bValid, np.array([fbb_min, fbb_max]))
        
        
    # freq_from_intensity()
    # Get the wingbeat frequency by undersampling the image intensity, and then using 
    # an alias to get the image of the spectrum in a frequency band (typically 180-220hz).
    #
    # intensity:     The current pixel intensity.
    # fs:            The current framerate.
    #
    def freq_from_intensity(self, intensity, fs=0):            
        # Update the sample buffer.
        self.buffer[self.i]        = [intensity, fs]
        self.buffer[self.i+self.n] = [intensity, fs]

        # The buffered framerates.        
        framerates = self.buffer[(self.i+1):(self.i+1+self.n),1]
        
        # The baseband alias range.
        (bValid, fbb_range) = self.get_baseband_range_from_framerates(framerates)
        if (fbb_range[0] < fbb_range[1]):
            fbb_min = fbb_range[0]
            fbb_max = fbb_range[1]
            bReverse = False
        else:
            fbb_min = fbb_range[1]
            fbb_max = fbb_range[0]
            bReverse = True
                    
        # Get the wingbeat frequency.
        if (bValid):
            intensities = self.buffer[(self.i+1):(self.i+1+self.n),0]
            
#             # Multiplying by an alternating sequence has the effect of frequency-reversal in freq domain.
#             if (bReverse):
#                 a = np.ones(len(intensities))
#                 a[1:len(a):2] = -1
#                 intensities *= a

            # Get the dominant frequency, and shift it from baseband to wingband.
            fft = np.fft.rfft(intensities)
            fft[0] = 0                      # Ignore the DC component.
            i_max = np.argmax(np.abs(fft))  # Index of the dominant frequency.
            f_width = fbb_max - fbb_min     # Width of the passband.
            f_offset = np.abs(np.fft.fftfreq(self.n)[i_max]) * fs - fbb_min # * 2.0 * f_width    # Offset into the passband of the dominant freq.
            if (bReverse):
                freq = self.fw_max - f_offset
            else:
                freq = self.fw_min + f_offset

            #rospy.logwarn((fs, [fbb_min, fbb_max], bReverse, i_max, f_width, np.fft.fftfreq(self.n)[i_max], f_offset, freq))
        
        else:
            freq = 0.0
        
        # Go the the next sample slot.
        self.i += 1
        self.i %= self.n
        

        return freq
    
# End class WingbeatDetector            
    
    

###############################################################################
###############################################################################
class Fly(object):
    def __init__(self, params={}):
        self.nodename = rospy.get_name().rstrip('/')

        # Create the head tracker.        
        if (params['head']['tracker']=='area'):
            self.head    = AreaTracker(name='head',      params=params, color='cyan',    bEqualizeHist=True)
        elif (params['head']['tracker']=='edge'):
            self.head    = EdgeTracker(name='head',      params=params, color='cyan',    bEqualizeHist=False)
        elif (params['head']['tracker']=='tip'):
            self.head    = TipTracker(name='head',       params=params, color='cyan',    bEqualizeHist=False)
        elif (params['head']['tracker']=='intensity'):
            self.head    = IntensityTracker(name='head', params=params, color='cyan',    bEqualizeHist=False)
        else:
            rospy.logwarn('Head tracker parameter must be one of [''area'', ''edge'', ''tip'', ''intensity'']')
             
        # Create the abdomen tracker.        
        if (params['abdomen']['tracker']=='area'):
            self.abdomen    = AreaTracker(name='abdomen',      params=params, color='magenta',    bEqualizeHist=True)
        elif (params['abdomen']['tracker']=='edge'):
            self.abdomen    = EdgeTracker(name='abdomen',      params=params, color='magenta',    bEqualizeHist=False)
        elif (params['abdomen']['tracker']=='tip'):
            self.abdomen    = TipTracker(name='abdomen',       params=params, color='magenta',    bEqualizeHist=False)
        elif (params['abdomen']['tracker']=='intensity'):
            self.abdomen    = IntensityTracker(name='abdomen', params=params, color='magenta',    bEqualizeHist=False)
        else:
            rospy.logwarn('Abdomen tracker parameter must be one of [''area'', ''edge'', ''tip'', ''intensity'']')
             
        # Create the right wing tracker.        
        if (params['right']['tracker']=='area'):
            self.right    = AreaTracker(name='right',      params=params, color='red',    bEqualizeHist=True)
        elif (params['right']['tracker']=='edge'):
            self.right    = EdgeTracker(name='right',      params=params, color='red',    bEqualizeHist=False)
        elif (params['right']['tracker']=='tip'):
            self.right    = TipTracker(name='right',       params=params, color='red',    bEqualizeHist=False)
        elif (params['right']['tracker']=='intensity'):
            self.right    = IntensityTracker(name='right', params=params, color='red',    bEqualizeHist=False)
        else:
            rospy.logwarn('Right wing tracker parameter must be one of [''area'', ''edge'', ''tip'', ''intensity'']')
             
        # Create the left wing tracker.        
        if (params['left']['tracker']=='area'):
            self.left    = AreaTracker(name='left',      params=params, color='green',    bEqualizeHist=True)
        elif (params['left']['tracker']=='edge'):
            self.left    = EdgeTracker(name='left',      params=params, color='green',    bEqualizeHist=False)
        elif (params['left']['tracker']=='tip'):
            self.left    = TipTracker(name='left',       params=params, color='green',    bEqualizeHist=False)
        elif (params['left']['tracker']=='intensity'):
            self.left    = IntensityTracker(name='left', params=params, color='green',    bEqualizeHist=False)
        else:
            rospy.logwarn('Left wing tracker parameter must be one of [''area'', ''edge'', ''tip'', ''intensity'']')
             
        # Create the aux tracker.        
        self.aux    = IntensityTracker(name='aux',    params=params, color='yellow',    bEqualizeHist=False)
             

        self.windowInvertColorArea      = ImageWindow(False, 'InvertColorArea')
        
        self.bgra_body = bgra_dict['light_gray']
        self.ptBodyIndicator1 = None
        self.ptBodyIndicator2 = None
        self.bInvertColor = False
        self.iCount  = 0
        self.stampPrev = None
        self.stampPrevAlt = None
        self.stamp = rospy.Time(0)
        

        self.pubFlystate = rospy.Publisher(self.nodename+'/flystate', MsgFlystate)
 

    def set_params(self, params):
        self.params = params
        
        self.head.set_params(params)
        self.abdomen.set_params(params)
        self.left.set_params(params)
        self.right.set_params(params)
        self.aux.set_params(params)

        pt1 = [params['gui']['head']['hinge']['x'], params['gui']['head']['hinge']['y']]
        pt2 = [params['gui']['abdomen']['hinge']['x'], params['gui']['abdomen']['hinge']['y']]
        pt3 = [params['gui']['left']['hinge']['x'], params['gui']['left']['hinge']['y']]
        pt4 = [params['gui']['right']['hinge']['x'], params['gui']['right']['hinge']['y']]
        self.ptBodyCenter_i = get_intersection(pt1,pt2,pt3,pt4)

        r = max(params['gui']['left']['radius_outer'], params['gui']['right']['radius_outer'])
        self.angleBody_i = self.get_bodyangle_i()
        self.ptBodyIndicator1 = tuple((self.ptBodyCenter_i + r * np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)])).astype(int))
        self.ptBodyIndicator2 = tuple((self.ptBodyCenter_i - r * np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)])).astype(int))
        
        # Radius of an area approximately where the thorax would be.
        self.rInvertColorArea = np.linalg.norm(np.array([params['gui']['head']['hinge']['x'], params['gui']['head']['hinge']['y']]) - np.array([params['gui']['abdomen']['hinge']['x'], params['gui']['abdomen']['hinge']['y']]))/2.0
        self.bInvertColorValid = False
        
    
    def create_masks(self, shapeImage):
        if (self.params['gui']['head']['track']):
            self.head.create_mask (shapeImage)
            
        if (self.params['gui']['abdomen']['track']):
            self.abdomen.create_mask (shapeImage)
            
        if (self.params['gui']['right']['track']):
            self.right.create_mask (shapeImage)
            
        if (self.params['gui']['left']['track']):
            self.left.create_mask (shapeImage)
            
        if (self.params['gui']['aux']['track']):
            self.aux.create_mask (shapeImage)


    def get_bodyangle_i(self):
        angle_i = get_angle_from_points_i(self.abdomen.ptHinge_i, self.head.ptHinge_i)
        #angleBody_i  = (angle_i + np.pi) % (2.0*np.pi) - np.pi
        angleBody_i  = angle_i
        
        return angleBody_i
        

    # Calculate what we think the bInvertColor flag should be to make white-on-black.        
    def get_invertcolor(self, image):
        # Get a roi around the body center.
        xMin = max(0,self.ptBodyCenter_i[0]-int(0.75*self.rInvertColorArea))
        yMin = max(0,self.ptBodyCenter_i[1]-int(0.75*self.rInvertColorArea))
        xMax = min(self.ptBodyCenter_i[0]+int(0.75*self.rInvertColorArea), image.shape[1]-1)
        yMax = min(self.ptBodyCenter_i[1]+int(0.75*self.rInvertColorArea), image.shape[0]-1)
        imgInvertColorArea = image[yMin:yMax, xMin:xMax]
        self.windowInvertColorArea.set_image(imgInvertColorArea)

        # Midpoint between darkest & lightest colors.
        threshold = np.mean(image) 
        #rospy.logwarn((np.min(image), np.median(image), np.mean(image), np.max(image), np.mean(imgInvertColorArea)))
        
        # If the roi is too dark, then set bInvertColor.
        if (np.mean(imgInvertColorArea) <= threshold):
            bInvertColor = True
        else:
            bInvertColor = False

        return bInvertColor
        
        
                
    def set_background(self, image):
        self.head.set_background(image)
        self.abdomen.set_background(image)
        self.left.set_background(image)
        self.right.set_background(image)
        self.aux.set_background(image)

    
    def update_handle_points(self):
        self.head.update_handle_points()
        self.abdomen.update_handle_points()
        self.left.update_handle_points()
        self.right.update_handle_points()
        self.aux.update_handle_points()
        

    def update(self, header=None, image=None):
        if (image is not None):
            self.header = header
            
            if (not self.bInvertColorValid):
                self.bInvertColor = self.get_invertcolor(image)
                self.bInvertColorValid = True

            
            # Get the dt.  Keep track of both the camera timestamp, and the now() timestamp,
            # and use the now() timestamp only if the camera timestamp isn't changing.
            self.stamp = self.header.stamp
            stampAlt = rospy.Time.now()
            if (self.stampPrev is not None):
                dt = max(0.0, (self.header.stamp - self.stampPrev).to_sec())
                
                # If the camera is not giving good timestamps, then use our own clock.
                if (dt == 0.0):
                    dt = max(0.0, (stampAlt - self.stampPrevAlt).to_sec())
                    self.stamp = stampAlt

                # If time wrapped, then just assume a value.
                if (dt == 0.0):
                    dt = 1.0
                    
            else:
                dt = np.inf
            self.stampPrev = self.header.stamp
            self.stampPrevAlt = stampAlt

            self.head.update(dt, image, self.bInvertColor)
            self.abdomen.update(dt, image, self.bInvertColor)
            self.left.update(dt, image, self.bInvertColor)
            self.right.update(dt, image, self.bInvertColor)
            self.aux.update(dt, image, self.bInvertColor)
            
            
    def draw(self, image):
        # Draw line to indicate the body axis.
        cv2.line(image, self.ptBodyIndicator1, self.ptBodyIndicator2, self.bgra_body, 1) # Draw a line longer than just head-to-abdomen.
                
        self.head.draw(image)
        self.abdomen.draw(image)
        self.left.draw(image)
        self.right.draw(image)
        self.aux.draw(image)

        self.windowInvertColorArea.show()
        
    
    def publish(self):
        flystate              = MsgFlystate()
        flystate.header       = Header(seq=self.iCount, stamp=self.stamp, frame_id='Fly')

        if (self.params['gui']['head']['track']):
            flystate.head     = self.head.state
        else:
            flystate.head     = MsgState()

        if (self.params['gui']['abdomen']['track']):
            flystate.abdomen  = self.abdomen.state
        else:
            flystate.abdomen  = MsgState()

        if (self.params['gui']['left']['track']):
            flystate.left     = self.left.state
        else:
            flystate.left     = MsgState()
            
        if (self.params['gui']['right']['track']):
            flystate.right    = self.right.state
        else:
            flystate.right    = MsgState()
            
        if (self.params['gui']['aux']['track']):
            flystate.aux      = self.aux.state
        else:
            flystate.aux      = MsgState()


        self.iCount += 1
        
        self.pubFlystate.publish(flystate)
        


# end class Fly

        
###############################################################################
###############################################################################
# The 'aux' area to track intensity.
class IntensityTracker(IntensityTrackedBodypart):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        IntensityTrackedBodypart.__init__(self, name, params, color, bEqualizeHist)
        
        self.state = MsgState()
        self.state.intensity = 0.0
        self.state.freq = 0.0
        
        self.wingbeat = WingbeatDetector(0, 1000)

        self.set_params(params)

        
    
    # set_params()
    # Set the given params dict into this object.
    #
    def set_params(self, params):
        IntensityTrackedBodypart.set_params(self, params)

        self.imgRoiBackground = None
        self.iCount = 0
        self.state.intensity = 0.0

        self.wingbeat.set(self.params['wingbeat_min'], 
                          self.params['wingbeat_max'])

        
    # update_state()
    #
    def update_state(self):
        #f = 175         # Simulate this freq.
        #t = gImageTime 
        #self.state.intensity = np.cos(2*np.pi*f*t)
        self.state.intensity = np.sum(self.imgRoiFgMasked).astype(np.float32) / self.sumMask
        self.state.freq = self.wingbeat.freq_from_intensity(self.state.intensity, 1.0/self.dt)
    
        
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, dt, image, bInvertColor):
        IntensityTrackedBodypart.update(self, dt, image, bInvertColor)

        if (self.params['gui'][self.name]['track']):
            self.update_state()
    
            


    # draw()
    # Draw the outline.
    #
    def draw(self, image):
        IntensityTrackedBodypart.draw(self, image)

# end class IntensityTracker

    

###############################################################################
###############################################################################
# AreaTracker()
# Track a bodypart using the image correlation technique.  i.e. track the
# motion based on the movement of the pixels in the area of the image.
# Usually used for Head or Abdomen.
#
class AreaTracker(PolarTrackedBodypart):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        PolarTrackedBodypart.__init__(self, name, params, color, bEqualizeHist)
        
        self.phasecorr         = PhaseCorrelation()
        self.state             = MsgState() # In the bodypart frame.
        self.stateOrigin_p     = MsgState() # In the bodypart frame.
        self.stateLo_p         = MsgState() # In the bodypart frame.
        self.stateHi_p         = MsgState() # In the bodypart frame.
        
        self.windowStabilized = ImageWindow(False, self.name+'Stable')
        self.windowTest       = ImageWindow(True, self.name+'Test')
        self.set_params(params)

    
    # set_params()
    # Set the given params dict into this object.
    #
    def set_params(self, params):
        PolarTrackedBodypart.set_params(self, params)
        
        self.imgRoiBackground = None
        self.imgComparison = None
        self.windowTest.set_image(self.imgComparison)
        
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
        self.windowTest.set_enable(self.params['gui']['windows'] and self.params['gui'][self.name]['track'])


        
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
                # If angle is closer than the mean of (hi,lo), then take a new initial image, and shift the (hi,lo)
                angleRef_p = (self.stateHi_p.angles[0] + self.stateLo_p.angles[0])/2.0

                if (angleRef_p < self.state.angles[0] < self.stateOrigin_p.angles[0]) or (self.stateOrigin_p.angles[0] < self.state.angles[0] < angleRef_p):
                    self.imgComparison = imgNow
                    self.windowTest.set_image(self.imgComparison)

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
            self.state.intensity = float(np.sum(self.imgRoiFgMasked) / self.sumMask)
        else:
            self.state.intensity = 0.0            
        
        
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, dt, image, bInvertColor):
        PolarTrackedBodypart.update(self, dt, image, bInvertColor)

        if (self.imgComparison is None) and (self.iCount>1):
            self.imgComparison = self.imgRoiFgMaskedPolarCroppedWindowed
            self.windowTest.set_image(self.imgComparison)
        
        if (self.params['gui'][self.name]['track']):
            self.update_state()
            
    
    # draw()
    # Draw the outline.
    #
    def draw(self, image):
        PolarTrackedBodypart.draw(self, image)

        if (self.params['gui'][self.name]['track']):
            # Draw the bodypart state position.
            angle = self.state.angles[0] * self.sense
            pt = self.R.dot([self.state.radii[0]*np.cos(angle), 
                             self.state.radii[0]*np.sin(angle)]) 
            ptState_i = clip_pt((int(pt[0]+self.params['gui'][self.name]['hinge']['x']), 
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
                ptStateLo_i = clip_pt((int(pt[0]+self.params['gui'][self.name]['hinge']['x']), 
                                       int(pt[1]+self.params['gui'][self.name]['hinge']['y'])), image.shape) 
                
                
                angle = self.stateHi_p.angles[0] * self.sense
                pt = self.R.dot([self.state.radii[0]*np.cos(angle), 
                                 self.state.radii[0]*np.sin(angle)]) 
                ptStateHi_i = clip_pt((int(pt[0]+self.params['gui'][self.name]['hinge']['x']), 
                                       int(pt[1]+self.params['gui'][self.name]['hinge']['y'])), image.shape) 
                
                # Set the pixels.
                image[ptStateLo_i[1]][ptStateLo_i[0]] = np.array([255,255,255]) - image[ptStateLo_i[1]][ptStateLo_i[0]]
                image[ptStateHi_i[1]][ptStateHi_i[0]] = np.array([255,255,255]) - image[ptStateHi_i[1]][ptStateHi_i[0]]
                
            except ValueError:
                pass

            
            # Draw line from hinge to state point.        
            ptHinge_i = clip_pt((int(self.ptHinge_i[0]), int(self.ptHinge_i[1])), image.shape) 
            cv2.line(image, ptHinge_i, ptState_i, self.bgra_state, 1)
            
            self.windowStabilized.show()
            self.windowTest.show()
        
# end class AreaTracker

    

###############################################################################
###############################################################################
# Find the two largest gradients in the horizontal intensity profile of an image.
#
class EdgeDetector(object):
    def __init__(self, threshold=0.0, n_edges_max=1000, sense=1):
        self.intensities = []
        self.diff = []
        self.diffF = []
        self.set_threshold(threshold, n_edges_max, sense)


    def set_threshold(self, threshold, n_edges_max, sense):
        self.threshold = threshold
        self.n_edges_max = n_edges_max
        self.sense = sense


    # find()
    # Get the horizontal pixel position of all the vertical edge pairs that exceed a magnitude threshold.
    #
    def find(self, image):
        axis = 0
        intensitiesRaw = np.sum(image, axis).astype(np.float32)
        intensitiesRaw /= (255.0*image.shape[axis]) # Put into range [0,1]
        self.intensities = intensitiesRaw#filter_median(intensitiesRaw, q=1)
        
        # Compute the intensity gradient. 
        n = 5
        diffRaw = self.intensities[n:] - self.intensities[:-n]
        diffF = diffRaw#filter_median(diffRaw, q=1)
        #diffF -= np.mean(diffF)
        self.diff = np.append(diffF, np.zeros(n))
        
        # Make copies for positive-going and negative-going edges.
        diffP = copy.copy( self.sense*diffF)
        diffN = copy.copy(-self.sense*diffF)

        # Threshold the positive and negative diffs.
        iZero = np.where(diffP < self.threshold)[0] # 4*np.std(diffP))[0] #
        diffP[iZero] = 0.0
        
        iZero = np.where(diffN < self.threshold)[0] # 4*np.std(diffN))[0] #
        diffN[iZero] = 0.0

        # Find positive-going edges, and negative-going edges, alternately P & N.
        edgesP = [] # Positive-going edges.
        edgesN = [] # Negative-going edges.
        absP = []
        absN = []
        nCount = self.n_edges_max + (self.n_edges_max % 2) # Round up to the next multiple of 2 so that we look at both P and N diffs.
        q = 0 # Alternate between P & N:  0=P, 1=N
        diff_list = [diffP, diffN]
        edges_list = [edgesP, edgesN] 
        abs_list = [absP, absN] 
        iCount = 0
        
        # While there are edges to find, put them in lists in order of decending strength.
        while ((0.0 < np.max(diffP)) or (0.0 < np.max(diffN))) and (iCount < nCount):

            # If there's an edge in this diff.
            if (0.0 < np.max(diff_list[q])):
                # Append the strongest edge to the list of edges.
                iMax = np.argmax(diff_list[q])
                edges_list[q].append(iMax)
                abs_list[q].append(diff_list[q][iMax])
                iCount += 1
                
                # Zero all the values associated with this edge.
                for i in range(iMax-1, -1, -1):
                    if (0.0 < diff_list[q][i]):
                        diff_list[q][i] = 0.0
                    else:
                        break
                for i in range(iMax, len(diff_list[q])):
                    if (0.0 < diff_list[q][i]):
                        diff_list[q][i] = 0.0
                    else:
                        break

            q = 1-q # Go to the other list.


        #(edgesMajor, edgesMinor) = self.SortEdgesPairwise(edgesP, absP, edgesN, absN)
        (edgesMajor, edgesMinor) = self.SortEdgesOneEdge(edgesP, absP, edgesN, absN)

        # Combine & sort the two lists by decreasing absolute gradient.
        edges = copy.copy(edgesMajor)
        edges.extend(edgesMinor)
        if (len(edges)>0):
            nedges = np.array(edges)
            gradients = self.diff[nedges]
            s = np.argsort(np.abs(gradients))
            edges_sorted = nedges[s[::-1]]
            gradients_sorted = gradients[s[::-1]]
        else:
            edges_sorted = []
            gradients_sorted = []
    
        return (edges_sorted, gradients_sorted)
        
           
    
    # SortEdgesOneEdge()
    # Make sure that if there's just one edge, that it's in the major list.
    #
    def SortEdgesOneEdge(self, edgesP, absP, edgesN, absN):
        # If we have too many edges, then remove the weakest one.
        lP = len(edgesP)
        lN = len(edgesN)
        if (self.n_edges_max < lP+lN):
            if (0<lP) and (0<lN):
                if (absP[-1] < absN[-1]):
                    edgesP.pop()
                    absP.pop()
                else:
                    edgesN.pop()
                    absN.pop()
            elif (0<lP):
                edgesP.pop()
                absP.pop()
            elif (0<lN):
                edgesN.pop()
                absN.pop()


        # Sort the edges.            
        if (len(edgesP)==0) and (len(edgesN)>0):
            edgesMajor = edgesN
            edgesMinor = edgesP
        else:#if (len(edgesN)==0) and (len(edgesP)>0):
            edgesMajor = edgesP
            edgesMinor = edgesN
        #else:
        #    edgesMajor = edgesP
        #    edgesMinor = edgesN

            
        return (edgesMajor, edgesMinor)
            
        
    # SortEdgesPairwise()
    # For each pair of (p,n) edges, the stronger edge of the pair is the major one.  
    #
    def SortEdgesPairwise(self, edgesP, absP, edgesN, absN):
        edgesMajor = []
        edgesMinor = []
        edges_list = [edgesMajor, edgesMinor]
        abs_list = [absP, absN]
        iCount = 0 

        m = max(len(edgesP), len(edgesN))
        for i in range(m):
            (absP1,edgeP1) = (absP[i],edgesP[i]) if (i<len(edgesP)) else (0.0, 0)
            (absN1,edgeN1) = (absN[i],edgesN[i]) if (i<len(edgesN)) else (0.0, 0)
            
            if (absP1 < absN1) and (iCount < self.n_edges_max):
                edgesMajor.append(edgeN1)
                iCount += 1
                if (0.0 < absP1) and (iCount < self.n_edges_max):
                    edgesMinor.append(edgeP1)
                    iCount += 1
                    
            elif (iCount < self.n_edges_max):
                edgesMajor.append(edgeP1)
                iCount += 1
                if (0.0 < absN1) and (iCount < self.n_edges_max):
                    edgesMinor.append(edgeN1)
                    iCount += 1

        return (edgesMajor, edgesMinor)


    def SortEdgesMax(self, edgesP, absP, edgesN, absN):
        # The P or N list with the 'stronger' edge is considered to be the "major" one.
        if (np.max(absN) < np.max(absP)):
            edgesMajor = edgesP 
            edgesMinor = edgesN
        else:  
            edgesMajor = edgesN 
            edgesMinor = edgesP 
            
        return (edgesMajor, edgesMinor)
    
# End class EdgeDetector
    
    
###############################################################################
###############################################################################
# EdgeTracker()
# Track radial bodypart edges using the gradient of the image intensity.  
# i.e. Compute how the intensity changes with angle, and find the locations
# of the greatest change.  Usually used for Wings.
#
class EdgeTracker(PolarTrackedBodypart):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        PolarTrackedBodypart.__init__(self, name, params, color, bEqualizeHist)
        
        self.name       = name
        self.detector   = EdgeDetector()
        self.state      = MsgState()
        self.windowTest = ImageWindow(False, self.name+'Edge')
        self.set_params(params)

        # Services, for live intensities plots via live_wing_histograms.py
        self.service_wingdata    = rospy.Service('wingdata_'+name, SrvWingdata, self.serve_wingdata_callback)
    
    
    # set_params()
    # Set the given params dict into this object.
    #
    def set_params(self, params):
        PolarTrackedBodypart.set_params(self, params)
        
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

        self.detector.set_threshold(params[self.name]['threshold'], params['n_edges_max'], self.sense)
    
        
    # update_state()
    #
    def update_state(self):
        imgNow = self.imgRoiFgMaskedPolarCropped
        self.windowTest.set_image(imgNow)
        
        # Get the rotation & expansion between images.
        if (imgNow is not None):
            # Pixel position and strength of the edges.
            (edges, gradients) = self.detector.find(imgNow)

            anglePerPixel = (self.params['gui'][self.name]['angle_hi']-self.params['gui'][self.name]['angle_lo']) / float(imgNow.shape[1])

            # Convert pixel to angle units, and put angle into the wing frame.
            self.state.angles = []
            self.state.gradients = []
            for i in range(len(edges)):
                edge = edges[i]
                gradient = gradients[i]
                angle_b = self.params['gui'][self.name]['angle_lo'] + edge * anglePerPixel
                angle_p = (self.transform_angle_p_from_b(angle_b) + np.pi) % (2*np.pi) - np.pi
                self.state.angles.append(angle_p)
                self.state.gradients.append(gradient)
                

            self.state.intensity = np.mean(imgNow)/255.0

        
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, dt, image, bInvertColor):
        PolarTrackedBodypart.update(self, dt, image, bInvertColor)

        if (self.params['gui'][self.name]['track']):
            self.update_state()
            
            
    
    
    # draw()
    # Draw the outline.
    #
    def draw(self, image):
        PolarTrackedBodypart.draw(self, image)
        
        if (self.params['gui'][self.name]['track']):
            # Draw the major and minor edges alternately, until the max number has been reached.
            bgra = self.bgra
            for i in range(len(self.state.angles)):
                angle_b = self.transform_angle_b_from_p(self.state.angles[i])
                angle_i = self.transform_angle_i_from_b(angle_b)

                x0 = self.ptHinge_i[0] + self.params['gui'][self.name]['radius_inner'] * np.cos(angle_i)
                y0 = self.ptHinge_i[1] + self.params['gui'][self.name]['radius_inner'] * np.sin(angle_i)
                x1 = self.ptHinge_i[0] + self.params['gui'][self.name]['radius_outer'] * np.cos(angle_i)
                y1 = self.ptHinge_i[1] + self.params['gui'][self.name]['radius_outer'] * np.sin(angle_i)
                cv2.line(image, (int(x0),int(y0)), (int(x1),int(y1)), bgra, 1)
                bgra = tuple(0.5*np.array(bgra))
                
            self.windowTest.show()
        
        
    def serve_wingdata_callback(self, request):
        abscissa = np.linspace(self.params['gui'][self.name]['angle_lo'], self.params['gui'][self.name]['angle_hi'], len(self.detector.intensities))
        
            
        angles = []
        gradients = []
        for i in range(len(self.state.angles)):
            angle = self.state.angles[i]
            gradient = self.state.gradients[i]
            angle_b = ((self.transform_angle_b_from_p(angle) + np.pi) % (2*np.pi)) - np.pi
            angles.append(angle_b)
            gradients.append(gradient)

        return SrvWingdataResponse(abscissa, self.detector.intensities, self.detector.diff, angles, gradients)
        
        
# end class EdgeTracker

    
###############################################################################
###############################################################################
# Find the lowest image point where image intensity exceeds a threshold.
#
class TipDetector(object):
    def __init__(self, threshold=0.0, sense=1):
        self.intensities = []
        self.diff = []
        self.diffF = []
        self.set_threshold(threshold, sense)
        self.imgThreshold = None


    def set_threshold(self, threshold, sense):
        self.threshold = threshold
        self.sense = sense


    # find()
    # Get the pixel position in the given image of the bottommost thresholded value.
    #
    def find(self, image):
        #(threshold, img) = cv2.threshold(image, int(self.threshold*255), 255, cv2.THRESH_TOZERO) # BINARY)#
        (threshold, img) = (0, image)
        
        axis = 1
        self.intensities = np.sum(img, axis).astype(np.float32)
        self.intensities /= (255.0 * img.shape[axis]) # Put into range [0,1]

        # Compute the intensity gradient. 
        n = 5
        diffRaw = self.intensities[n:] - self.intensities[:-n]
        diffF = np.abs(diffRaw)
        self.diff = np.append(diffF, np.zeros(n))
        
        
        
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
# Finds the point at the greatest distance from the hinge.  Usually used for Wings.
#
class TipTracker(PolarTrackedBodypart):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        PolarTrackedBodypart.__init__(self, name, params, color, bEqualizeHist)
        
        self.name     = name
        self.detector = TipDetector()
        self.state    = MsgState()
        self.set_params(params)
        self.windowTest = ImageWindow(False, self.name+'Tip')
        self.iAngle = 0
        self.iRadius = 0

        # Services, for live intensities plots via live_wing_histograms.py
        self.service_tipdata    = rospy.Service('tipdata_'+self.name, SrvTipdata, self.serve_tipdata_callback)

    
    # set_params()
    # Set the given params dict into this object.
    #
    def set_params(self, params):
        PolarTrackedBodypart.set_params(self, params)
        
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

        self.detector.set_threshold(params[self.name]['threshold'], self.sense)
    
        
    # update_state()
    #
    def update_state(self):
        imgNow = self.imgRoiFgMaskedPolarCropped
        
        # Get the rotation & expansion between images.
        if (imgNow is not None):
            # Pixel position and strength of the edges.
            (self.iAngle, self.iRadius) = self.detector.find(imgNow)

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

        
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, dt, image, bInvertColor):
        PolarTrackedBodypart.update(self, dt, image, bInvertColor)

        if (self.params['gui'][self.name]['track']):
            self.update_state()
            self.windowTest.set_image(self.detector.imgThreshold)
            
            
    
    
    # draw()
    # Draw the state.
    #
    def draw(self, image):
        PolarTrackedBodypart.draw(self, image)

        if (self.params['gui'][self.name]['track']) and (len(self.state.angles)>0):
            # Draw the bodypart state position.
            angle = self.state.angles[0] * self.sense
            pt = self.R.dot([self.state.radii[0]*np.cos(angle), 
                             self.state.radii[0]*np.sin(angle)]) 
            ptState_i = clip_pt((int(pt[0]+self.params['gui'][self.name]['hinge']['x']), 
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
            ptHinge_i = clip_pt((int(self.ptHinge_i[0]), int(self.ptHinge_i[1])), image.shape) 
            cv2.line(image, ptHinge_i, ptState_i, self.bgra_state, 1)
            self.windowTest.show()
        
        
    def serve_tipdata_callback(self, request):
        intensities = self.detector.intensities
        diffs = self.detector.diff
        abscissa = range(len(intensities))
        if (self.iRadius is not None):
            detectionH = self.iRadius
        else:
            detectionH = 0
            
        detectionV = self.params[self.name]['threshold']
        
            
        return SrvTipdataResponse(self.color, abscissa, intensities, diffs, detectionH, detectionV)
        
        
# end class TipTracker

    
###############################################################################
###############################################################################
class MainWindow:
    def __init__(self):
        self.stampPrev = None
        self.stampPrevAlt = None
        self.stampRosimagePrev = None
        self.lockParams = threading.Lock()
        self.lockBuffer = threading.Lock()
        
        # initialize
        rospy.init_node('kinefly')
        self.nodename = rospy.get_name().rstrip('/')
        
        # initialize display
        self.window_name = self.nodename.strip('/')
        cv2.namedWindow(self.window_name,1)
        self.cvbridge = CvBridge()
        
        self.yamlfile = os.path.expanduser(rospy.get_param(self.nodename+'/yamlfile', '~/%s.yaml' % self.nodename.strip('/')))
        with self.lockParams:
            # Load the parameters from server.
            try:
                self.params = rospy.get_param(self.nodename, {})
            except (rosparam.RosParamException, IndexError), e:
                rospy.logwarn('%s.  Using default values.' % e)
                self.params = {}

            # Load the GUI parameters yaml file.
            try:
                self.params['gui'] = rosparam.load_file(self.yamlfile)[0][0]
            except (rosparam.RosParamException, IndexError), e:
                rospy.logwarn('%s.  Using default values.' % e)
            
        defaults = {'filenameBackground':'~/%s.png' % self.nodename.strip('/'),
                    'image_topic':'/camera/image_raw',
                    'use_gui':True,                     # You can turn off the GUI to speed the framerate.
                    'scale_image':1.0,                  # Reducing the image scale will speed the framerate.
                    'n_edges_max':1,                    # Max number of edges per wing to find, subject to threshold.
                    'rc_background':1000.0,             # Time constant of the moving average background.
                    'wingbeat_min':180,                 # Bounds for wingbeat frequency measurement.
                    'wingbeat_max':220,
                    'head':   {'tracker':'area',
                               'autozero':True,         # Automatically figure out where is the center of motion.
                               'threshold':0.0},        
                    'abdomen':{'tracker':'area',
                               'autozero':True,
                               'threshold':0.0},
                    'left':   {'tracker':'edge',
                               'autozero':True,
                               'threshold':0.0},
                    'right':  {'tracker':'edge',
                               'autozero':True,
                               'threshold':0.0},
                    'aux':    {'tracker':'intensity'},
                    'gui': {'windows':True,                     # Show the helpful extra windows.
                            'symmetric':True,                   # Forces the UI to remain symmetric.
                            'head':   {'track':True,            # To track, or not to track.
                                       'subtract_bg':False,     # Use background subtraction?
                                       'stabilize':False,       # Image stabilization of the bodypart.
                                       'hinge':{'x':300,        # Hinge position in image coordinates.
                                                'y':150},
                                       'radius_outer':80,       # Outer radius in pixel units.
                                       'radius_inner':50,       # Inner radius in pixel units.
                                       'angle_hi':0.7854,       # Angle limit in radians.
                                       'angle_lo':-0.7854},     # Angle limit in radians.
                            'abdomen':{'track':True,
                                       'subtract_bg':False,
                                       'stabilize':False,
                                       'hinge':{'x':300,
                                                'y':250},
                                       'radius_outer':120,
                                       'radius_inner':100,
                                       'angle_hi':3.927, 
                                       'angle_lo':2.3562},
                            'left':   {'track':True,
                                       'subtract_bg':False,
                                       'stabilize':False,
                                       'hinge':{'x':250,
                                                'y':200},
                                       'radius_outer':80,
                                       'radius_inner':50,
                                       'angle_hi':-0.7854, 
                                       'angle_lo':-2.3562},
                            'right':  {'track':True,
                                       'subtract_bg':False,
                                       'stabilize':False,
                                       'hinge':{'x':350,
                                                'y':200},
                                       'radius_outer':80,
                                       'radius_inner':50,
                                       'angle_hi':2.3562, 
                                       'angle_lo':0.7854},
                            'aux':    {'track':True,
                                       'subtract_bg':False,
                                       'center':{'x':350,
                                                 'y':150},
                                       'radius1':30,
                                       'radius2':20,
                                       'angle':0.0},
                            }
                    }

        SetDict().set_dict_with_preserve(self.params, defaults)
        self.params = self.legalizeParams(self.params)
        rospy.set_param(self.nodename+'/gui', self.params['gui'])
        
        self.scale = self.params['scale_image']
        self.bMousing = False
        self.bQuitting = False
        
        # Create the fly.
        self.fly = Fly(self.params)
        
        # Background image.
        self.filenameBackground = os.path.expanduser(self.params['filenameBackground'])
        imgDisk  = cv2.imread(self.filenameBackground, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        if (imgDisk is not None):
            if (self.scale == 1.0):              
                imgFullBackground = imgDisk
            else:  
                imgFullBackground = cv2.resize(imgDisk, (0,0), fx=self.scale, fy=self.scale)
             
            self.fly.set_background(imgFullBackground)
            self.bHaveBackground = True
        else:
            self.bHaveBackground = False
        
        
        self.nameSelected   = None
        self.uiSelected     = None
        self.stateSelected  = None
        self.fly.update_handle_points()
        self.stampDiff      = rospy.Duration(0)
        self.stampMax       = rospy.Duration(0)
        self.dtCamera       = np.inf
        self.hzActual       = 0.0
        self.hzActualSum    = 0.0
        self.hzAvailable    = 0.0
        self.hzAvailableSum = 0.0
        self.iCount         = 0
        self.iDroppedFrame  = 0

        nQueue              = 2
        self.bufferImages   = [None]*nQueue # Circular buffer for incoming images.
        self.iImgLoading    = 0  # Index of the next slot to load.
        self.iImgWorking    = 0  # Index of the slot to process, i.e. the oldest image in the buffer.
        self.imgUnscaled    = None
        self.imgScaled      = None
        
        self.h_gap          = int(5 * self.scale)
        self.w_gap          = int(10 * self.scale)
        self.scaleText      = 0.4 * self.scale
        self.fontface       = cv2.FONT_HERSHEY_SIMPLEX
        self.buttons        = None
        self.yToolbar       = 0
        
        # Publishers.
        self.pubCommand     = rospy.Publisher(self.nodename+'/command', MsgCommand)

        # Subscriptions.        
        sizeImage = 128+1024*1024 # Size of header + data.
        self.subImage       = rospy.Subscriber(self.params['image_topic'], Image,      self.image_callback,   queue_size=2, buff_size=2*sizeImage, tcp_nodelay=True)
        self.subCommand     = rospy.Subscriber(self.nodename+'/command',   MsgCommand, self.command_callback, queue_size=1000)

        # user callbacks
        cv2.setMouseCallback(self.window_name, self.onMouse, param=None)
        
        self.reconfigure = dynamic_reconfigure.server.Server(kineflyConfig, self.reconfigure_callback)
        

    # Check the given button to see if it extends outside the image, and if so then reposition it to the next line.
    def wrap_button(self, btn, shape):
        if (btn.right >= shape[1]):
            btn.set_pos(pt=[1, btn.bottom+1])

        
    # Create the button bar, with overflow onto more than one line if needed to fit on the image.        
    def create_buttons(self, shape):
        if (self.buttons is None):
            # UI button specs.
            self.buttons = []
            x = 1
            y = 1
            btn = Button(pt=[x,y], scale=self.scale, type='pushbutton', name='exit', text='exit')
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='pushbutton', name='save_bg', text='saveBG')
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='static', name='track', text='track:', sides=SIDE_LEFT|SIDE_TOP|SIDE_BOTTOM)
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='head', text='H', state=self.params['gui']['head']['track'], sides=SIDE_TOP|SIDE_BOTTOM)
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='abdomen', text='A', state=self.params['gui']['abdomen']['track'], sides=SIDE_TOP|SIDE_BOTTOM)
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='left', text='L', state=self.params['gui']['left']['track'], sides=SIDE_TOP|SIDE_BOTTOM)
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='right', text='R', state=self.params['gui']['right']['track'], sides=SIDE_TOP|SIDE_BOTTOM)
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='aux', text='X', state=self.params['gui']['aux']['track'], sides=SIDE_TOP|SIDE_BOTTOM|SIDE_RIGHT)
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='static', name='subtract', text='subtract:', sides=SIDE_LEFT|SIDE_TOP|SIDE_BOTTOM)
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='subtract_head', text='H', state=self.params['gui']['head']['subtract_bg'], sides=SIDE_TOP|SIDE_BOTTOM)
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='subtract_abdomen', text='A', state=self.params['gui']['abdomen']['subtract_bg'], sides=SIDE_TOP|SIDE_BOTTOM)
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='subtract_lr', text='LR', state=self.params['gui']['right']['subtract_bg'], sides=SIDE_TOP|SIDE_BOTTOM)
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='subtract_aux', text='X', state=self.params['gui']['aux']['subtract_bg'], sides=SIDE_TOP|SIDE_BOTTOM|SIDE_RIGHT)
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='stabilize', text='stabilize', state=self.params['gui']['head']['stabilize'])
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='symmetry', text='symmetric', state=self.params['gui']['symmetric'])
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            x = btn.right+1
            y = btn.top+1
            btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='windows', text='windows', state=self.params['gui']['windows'])
            self.wrap_button(btn, shape)
            self.buttons.append(btn)
            
            self.yToolbar = btn.bottom + 1


    # legalizeParams()
    # Make sure that all the parameters contain legal values.
    #
    def legalizeParams(self, params):
        paramsOut = copy.copy(params)
        for partname in ['head','abdomen','left','right']:
            if (partname in paramsOut):
                if ('hinge' in paramsOut['gui'][partname]):
                    if ('x' in paramsOut['gui'][partname]['hinge']):
                        paramsOut['gui'][partname]['hinge']['x'] = max(0, paramsOut['gui'][partname]['hinge']['x'])
        
                    if ('y' in paramsOut['gui'][partname]['hinge']):
                        paramsOut['gui'][partname]['hinge']['y'] = max(0, paramsOut['gui'][partname]['hinge']['y'])
    
                if ('radius_inner' in paramsOut['gui'][partname]):
                    paramsOut['gui'][partname]['radius_inner'] = max(5*paramsOut['scale_image'], paramsOut['gui'][partname]['radius_inner'])
    
                if ('radius_outer' in paramsOut['gui'][partname]) and ('radius_inner' in paramsOut['gui'][partname]):
                    paramsOut['gui'][partname]['radius_outer'] = max(paramsOut['gui'][partname]['radius_outer'], paramsOut['gui'][partname]['radius_inner']+5*paramsOut['scale_image'])

        return paramsOut
        
        
    def reconfigure_callback(self, config, level):
        # Save the new params.
        SetDict().set_dict_with_overwrite(self.params, config)
        
        # Remove dynamic_reconfigure keys from the params.
        try:
            self.params.pop('groups')
        except KeyError:
            pass
        
        # Set it into the wings.
        self.fly.set_params(self.scale_params(self.params, self.scale))
        with self.lockParams:
            rosparam.dump_params(self.yamlfile, self.nodename+'/gui')
        
        return config


    # command_callback()
    # Execute any commands sent over the command topic.
    #
    def command_callback(self, msg):
        self.command = msg.command
        
        if (self.command == 'exit'):
            self.bQuitting = True
            
            # Save the params.
            SetDict().set_dict_with_preserve(self.params, rospy.get_param(self.nodename, {}))
            rospy.set_param(self.nodename+'/gui', self.params['gui'])
            with self.lockParams:
                rosparam.dump_params(self.yamlfile, self.nodename+'/gui')

            rospy.signal_shutdown('User requested exit.')
        
        
        if (self.command == 'save_background'):
            self.save_background()
            
        
        if (self.command == 'gui_on'):
            self.params['use_gui'] = True
            
        
        if (self.command == 'gui_off'):
            self.params['use_gui'] = False
            
        
        if (self.command == 'help'):
            rospy.logwarn('The %s/command topic accepts the following string commands:' % self.nodename)
            rospy.logwarn('  help                 This message.')
            rospy.logwarn('  save_background      Saves the instant image to disk for use as the')
            rospy.logwarn('                       background.')
            rospy.logwarn('  gui_on               Turn on the graphical user interface.')
            rospy.logwarn('  gui_off              Turn off the graphical user interface.')
            rospy.logwarn('  exit                 Exit the program.')
            rospy.logwarn('')
            rospy.logwarn('You can send the above commands at the shell prompt via:')
            rospy.logwarn('rostopic pub -1 %s/command Kinefly/MsgCommand commandtext' % self.nodename)
            rospy.logwarn('')

        
        
    def scale_params(self, paramsIn, scale):
        paramsScaled = copy.deepcopy(paramsIn)

        for partname in ['head', 'abdomen', 'left', 'right']:
            paramsScaled['gui'][partname]['hinge']['x'] = (paramsIn['gui'][partname]['hinge']['x']*scale)  
            paramsScaled['gui'][partname]['hinge']['y'] = (paramsIn['gui'][partname]['hinge']['y']*scale)  
            paramsScaled['gui'][partname]['radius_outer'] = (paramsIn['gui'][partname]['radius_outer']*scale)  
            paramsScaled['gui'][partname]['radius_inner'] = (paramsIn['gui'][partname]['radius_inner']*scale)  
            
        for partname in ['aux']:
            paramsScaled['gui'][partname]['center']['x'] = (paramsIn['gui'][partname]['center']['x']*scale)  
            paramsScaled['gui'][partname]['center']['y'] = (paramsIn['gui'][partname]['center']['y']*scale)  
            paramsScaled['gui'][partname]['radius1'] = (paramsIn['gui'][partname]['radius1']*scale)  
            paramsScaled['gui'][partname]['radius2'] = (paramsIn['gui'][partname]['radius2']*scale)  
            
        return paramsScaled  
    
    
    # Draw user-interface elements on the image.
    def draw_buttons(self, image):
        if (self.buttons is not None):
            for i in range(len(self.buttons)):
                self.buttons[i].draw(image)


    def image_callback(self, rosimg):
#         global gImageTime
#         gImageTime = self.header.stamp.to_sec()

        # Receive the image:
        if (not self.bQuitting):
            with self.lockBuffer:
                if (self.bufferImages[self.iImgLoading] is None):   # There's an empty slot in the buffer.
                    iImgLoadingNext = (self.iImgLoading+1) % len(self.bufferImages)
                    iImgWorkingNext = self.iImgWorking
                    self.iDroppedFrame = 0
                else:                                               # The buffer is full; we'll overwrite the oldest entry.
                    iImgLoadingNext = (self.iImgLoading+1) % len(self.bufferImages)
                    iImgWorkingNext = (self.iImgWorking+1) % len(self.bufferImages)
                    self.iDroppedFrame += 1
    
#                if ('kinefly3' in self.nodename) and (self.stampPrev is not None):
#                    rospy.logwarn('Filled %d: %0.6f, %0.6f' % (self.iImgLoading, self.stampPrev.to_sec(), rosimg.header.stamp.to_sec()))
                self.bufferImages[self.iImgLoading] = rosimg
                self.iImgLoading = iImgLoadingNext
                self.iImgWorking = iImgWorkingNext

#            # Warn if the camera is delaying.
#            if (self.stampRosimagePrev is not None):
#                if (rosimg.header.stamp.to_sec() - self.stampRosimagePrev.to_sec() > 1):
#                    rospy.logwarn('%s: %d, %d %0.6f' % (self.nodename, rosimg.header.seq, (rosimg.header.stamp.secs*1e9)+rosimg.header.stamp.nsecs, rosimg.header.stamp.to_sec() - self.stampRosimagePrev.to_sec()))
#
#            self.stampRosimagePrev = rosimg.header.stamp

            

                
    def process_image_fake(self):
        with self.lockBuffer:
            # Mark this buffer entry as available for loading.
            self.bufferImages[self.iImgWorking] = None

            # Go to the next image.
            self.iImgWorking = (self.iImgWorking+1) % len(self.bufferImages)

                
                
    def process_image(self):
        stamp0 = rospy.Time.now()
        rosimg = None
        
#        if ('kinefly3' in self.nodename) and (self.stampPrev is not None):
#            rospy.logwarn('Checking %d: %0.6f' % (self.iImgWorking, self.stampPrev.to_sec()))

        with self.lockBuffer:
            if (self.bufferImages[self.iImgWorking] is not None):
                rosimg = self.bufferImages[self.iImgWorking]
                
                # Mark this buffer entry as available for loading.
                self.bufferImages[self.iImgWorking] = None
    
                # Go to the next image.
                self.iImgWorking = (self.iImgWorking+1) % len(self.bufferImages)
        


        if (rosimg is not None):            
            if (self.stampPrev is not None):
                self.dtCamera = max(0, (rosimg.header.stamp - self.stampPrev).to_sec())
                
                # If the camera is not giving good timestamps, then use our own clock.
                if (self.dtCamera == 0.0):
                    self.dtCamera = max(0, (stamp0 - self.stampPrevAlt).to_sec())
                    
                # If time wrapped, then just assume a value.
                if (self.dtCamera == 0.0):
                    self.dtCamera = 1.0
                    
#                if ('kinefly3' in self.nodename):
#                    rospy.logwarn('********************')
#                    s1 = 'prev=%0.6f, img=%0.6f: img-prev=%0.6f, dt=%0.6f' % (self.stampPrev.to_sec(), rosimg.header.stamp.to_sec(), rosimg.header.stamp.to_sec()-self.stampPrev.to_sec(), self.dtCamera)
#                    if (self.iDroppedFrame>0):
#                        s1 += '    Dropped Frames: %d' % self.iDroppedFrame
#                    rospy.logwarn(s1)
#
#                    s2 = 'prev0=%0.6f, 0=%0.6f: 0-prev0=%0.6f' % (self.stampPrevAlt.to_sec(), stamp0.to_sec(), stamp0.to_sec()-self.stampPrevAlt.to_sec())
#                    if (self.iDroppedFrame>0):
#                        s2 += '    Dropped Frames: %d' % self.iDroppedFrame
#                    rospy.logwarn(s2)

            else:
                self.dtCamera = np.inf

                
            self.stampPrev = rosimg.header.stamp
            self.stampPrevAlt = stamp0

            try:
                img = np.uint8(cv.GetMat(self.cvbridge.imgmsg_to_cv(rosimg, 'passthrough')))
                
            except CvBridgeError, e:
                rospy.logwarn ('Exception converting background image from ROS to opencv:  %s' % e)
                img = np.zeros((320,240))

            # Scale the image.
            self.imgUnscaled = img
            if (self.scale == 1.0):              
                self.imgScaled = self.imgUnscaled
            else:  
                self.imgScaled = cv2.resize(img, (0,0), fx=self.scale, fy=self.scale) 

            
            self.shapeImage = self.imgScaled.shape # (height,width)
            
            # Create the button bar if needed.    
            self.create_buttons(self.imgScaled.shape)
        
            if (not self.bMousing):
                # Update the fly internals.
                self.fly.update(rosimg.header, self.imgScaled)
    
                # Publish the outputs.
                self.fly.publish()
                
            
            if (self.params['use_gui']):
        
                imgOutput = cv2.cvtColor(self.imgScaled, cv2.COLOR_GRAY2RGB)
                self.fly.draw(imgOutput)
                self.draw_buttons(imgOutput)
            
                x_left   = int(10 * self.scale)
                y_bottom = int(imgOutput.shape[0] - 10 * self.scale)
                x_right  = int(imgOutput.shape[1] - 10 * self.scale)
                x = x_left
                y = y_bottom
                h = 10

                w = 55
                if (not self.bMousing):

                    # Output the framerate.
                    hzNow = 1/self.dtCamera
                    self.iCount += 1
                    if (self.iCount > 100):                     
                        a= 0.01 # Filter the framerate.
                        self.hzActual = (1-a)*self.hzActual + a*hzNow 
                    else:                                       
                        if (self.iCount>20):             # Get past the transient response.       
                            self.hzActualSum += hzNow                 
                        else:
                            self.hzActualSum = hzNow * self.iCount     
                            
                        self.hzActual = self.hzActualSum / self.iCount
                        
                    dtAvailable = self.stampDiff.to_sec()
                    if (dtAvailable != 0.0):
                        hzAvailableNow = 1/dtAvailable
                        if (self.iCount > 100):                     
                            a= 0.04 # Filter the framerate.
                            self.hzAvailable = (1-a)*self.hzAvailable + a*hzAvailableNow 
                        else:                                       
                            if (self.iCount>20):             # Get past the transient response.       
                                self.hzAvailableSum += hzAvailableNow                 
                            else:
                                self.hzAvailableSum = hzAvailableNow * self.iCount     
                                
                            self.hzAvailable = self.hzAvailableSum / self.iCount
                        
                    
                    
                    s = '%5.1fHz (%5.1fHz avail)' % (self.hzActual, self.hzAvailable)
                        
                    if (self.iDroppedFrame>0):
                        s += '    Dropped Frames: %d' % self.iDroppedFrame

                    cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, bgra_dict['dark_red'] )
                    
                    h_text = int(h * self.scale)
                    y -= h_text+self.h_gap
                
    
                    # Output the aux state.
                    if (self.params['gui']['aux']['track']):
                        s = 'AUX: (%0.3f)' % (self.fly.aux.state.intensity)
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.aux.bgra)
                        h_text = int(h * self.scale)
                        y -= h_text+self.h_gap
                    
                        s = 'WB Freq: '
                        if (self.fly.aux.state.freq != 0.0):
                            s += '%0.0fhz' % (self.fly.aux.state.freq)
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.aux.bgra)
                        h_text = int(h * self.scale)
                        y -= h_text+self.h_gap
                    
    
                    # Output the wings state.
                    if (self.params['gui']['left']['track']) and (self.params['gui']['right']['track']):
                        # L+R
                        s = 'L+R:'
                        if (len(self.fly.left.state.angles)>0) and (len(self.fly.right.state.angles)>0):
                            leftplusright = self.fly.left.state.angles[0] + self.fly.right.state.angles[0]
                            s += '% 7.4f' % leftplusright
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, bgra_dict['blue'])
                        h_text = int(h * self.scale)
                        y -= h_text+self.h_gap
        
                            
                        # L-R
                        s = 'L-R:'
                        if (len(self.fly.left.state.angles)>0) and (len(self.fly.right.state.angles)>0):
                            leftminusright = self.fly.left.state.angles[0] - self.fly.right.state.angles[0]
                            s += '% 7.4f' % leftminusright
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, bgra_dict['blue'])
                        h_text = int(h * self.scale)
                        y -= h_text+self.h_gap
        
                    if (self.params['gui']['right']['track']):
                        # Right
                        s = 'R:'
                        if (len(self.fly.right.state.angles)>0):
                            s += '% 7.4f' % self.fly.right.state.angles[0]
                            #for i in range(1,len(self.fly.right.state.angles)):
                            #    s += ', % 7.4f' % self.fly.right.state.angles[i]
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.right.bgra)
                        h_text = int(h * self.scale)
                        y -= h_text+self.h_gap
            
                    if (self.params['gui']['left']['track']):
                        # Left
                        s = 'L:'
                        if (len(self.fly.left.state.angles)>0):
                            s += '% 7.4f' % self.fly.left.state.angles[0]
                            #for i in range(1,len(self.fly.left.state.angles)):
                            #    s += ', % 7.4f' % self.fly.left.state.angles[i]
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.left.bgra)
                        h_text = int(h * self.scale)
                        y -= h_text+self.h_gap
                        
                            
    
    
                    # Output the abdomen state.
                    if (self.params['gui']['abdomen']['track']):
                        s = 'ABDOMEN:% 7.4f' % (self.fly.abdomen.state.angles[0])
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.abdomen.bgra)
                        h_text = int(h * self.scale)
                        y -= h_text+self.h_gap
                    
    
                    # Output the head state.
                    if (self.params['gui']['head']['track']):
                        s = 'HEAD:% 7.4f' % (self.fly.head.state.angles[0])
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.head.bgra)
                        h_text = int(h * self.scale)
                        y -= h_text+self.h_gap
                

                # Display the image.
                cv2.imshow(self.window_name, imgOutput)
                cv2.waitKey(1)

            # Compute processing time.
            stamp1 = rospy.Time.now()
            self.stampDiff = (stamp1 - stamp0)
            self.stampMax = max(self.stampMax, self.stampDiff)
            if (self.iCount % 100)==0:
                self.stampMax = rospy.Duration(0)
        else:
            if (self.hzAvailable != 0.0):
                rospy.sleep(1/self.hzAvailable) # Pretend we spent time processing.
        
#            if ('kinefly3' in self.nodename):
#                rospy.logwarn('Buffer is empty: %0.6f' % stamp0.to_sec())
#
#                if (self.stampPrev is not None):
#                    s1 = 'prev=%0.6f: dt=%0.6f' % (self.stampPrev.to_sec(), self.dtCamera)
#                    if (self.iDroppedFrame>0):
#                        s1 += '    Dropped Frames: %d' % self.iDroppedFrame
#                    rospy.logwarn(s1)
#
#                    s2 = 'prev0=%0.6f, now=%0.6f: 0-prev=%0.6f' % (self.stampPrevAlt.to_sec(), stamp0.to_sec(), stamp0.to_sec()-self.stampPrevAlt.to_sec())
#                    if (self.iDroppedFrame>0):
#                        s2 += '    Dropped Frames: %d' % self.iDroppedFrame
#                    rospy.logwarn(s2)
            
            
                
    # save_background()
    # Save the current camera image as the background.
    #
    def save_background(self):
        rospy.logwarn ('Saving new background image %s' % self.filenameBackground)
        cv2.imwrite(self.filenameBackground, self.imgUnscaled)
        self.fly.set_background(self.imgScaled)
        self.bHaveBackground = True
    
    
    # hit_object()
    # Get the nearest handle point or button to the mouse point.
    # ptMouse    = [x,y]
    # Returns the partname, tag, and ui of item the mouse has hit, using the 
    # convention that the name is of the form "tag_partname", e.g. "hinge_left"
    #
    def hit_object(self, ptMouse):
        tagHit  = None
        partnameHit = None
        uiHit = None
        
        # Check for button press.
        iButtonHit = None
        for iButton in range(len(self.buttons)):
            if (self.buttons[iButton].hit_test(ptMouse)):
                iButtonHit = iButton
            
        if (iButtonHit is not None):
            nameNearest = self.buttons[iButtonHit].name
            (tagHit,delim,partnameHit) = nameNearest.partition('_')
            uiHit = self.buttons[iButtonHit].type
        else: # Check for handle hit.
            tag  = [None,None,None,None,None]
            partname = [None,None,None,None,None]
            (partname[0], tag[0]) = self.fly.left.hit_object(ptMouse)
            (partname[1], tag[1]) = self.fly.right.hit_object(ptMouse)
            (partname[2], tag[2]) = self.fly.head.hit_object(ptMouse)
            (partname[3], tag[3]) = self.fly.abdomen.hit_object(ptMouse)
            (partname[4], tag[4]) = self.fly.aux.hit_object(ptMouse)
            i = next((i for i in range(len(tag)) if tag[i]!=None), None)
            if (i is not None):
                tagHit  = tag[i]
                partnameHit = partname[i]
                uiHit = 'handle'
    
        
        return (uiHit, tagHit, partnameHit, iButtonHit)
        
        
    # Convert tag and partname strings to a name string:  tag_partname
    def name_from_tagpartname(self, tag, partname):
        if (partname is not None) and (len(partname)>0):
            name = tag+'_'+partname
        else:
            name = tag
            
        return name
    

    # get_projection_onto_bodyaxis()
    # Project the given point onto the body axis.
    #
    def get_projection_onto_bodyaxis(self, ptAnywhere):
        # Project the point onto the body axis.
        ptB = self.fly.head.ptHinge_i - self.fly.abdomen.ptHinge_i
        ptM = ptAnywhere - self.fly.abdomen.ptHinge_i
        ptAxis = np.dot(ptB,ptM) / np.dot(ptB,ptB) * ptB + self.fly.abdomen.ptHinge_i
            
        return ptAxis
        
                
    def get_reflection_across_bodyaxis(self, ptAnywhere):
        ptAxis = self.get_projection_onto_bodyaxis(ptAnywhere)
        ptReflected = ptAnywhere + 2*(ptAxis-ptAnywhere)
        
        return ptReflected

    
    def bodypart_from_partname(self, partname):
        if (partname=='left'):
            bodypart = self.fly.left
        elif (partname=='right'):
            bodypart = self.fly.right
        elif (partname=='head'):
            bodypart = self.fly.head
        elif (partname=='abdomen'):
            bodypart = self.fly.abdomen
        elif (partname=='aux'):
            bodypart = self.fly.aux
        else:
            bodypart = None
            
        return bodypart
    
                
    # update_params_from_mouse()
    # Recalculate self.params based on a currently selected handle and mouse location.
    #
    def update_params_from_mouse(self, tagSelected, partnameSelected, ptMouse):
        partnameSlave = {'head':'abdomen', 'abdomen':'head', 'right':'left', 'left':'right', 'aux':None}[partnameSelected]
        tagThis = tagSelected
        tagOther = 'angle_lo' if (tagSelected=='angle_hi') else 'angle_hi'
        tagSlave = tagOther
        bodypartSelected = self.bodypart_from_partname(partnameSelected)
        bodypartSlave    = self.bodypart_from_partname(partnameSlave)

        # Scale the parameters in order to work on them.
        paramsScaled = self.scale_params(self.params, self.scale) 
        
        # Hinge.
        if (tagSelected=='hinge'): 
            if (partnameSelected=='head') or (partnameSelected=='abdomen'):

                # Get the hinge points pre-move.
                if (paramsScaled['gui']['symmetric']):
                    ptHead = np.array([paramsScaled['gui']['head']['hinge']['x'], paramsScaled['gui']['head']['hinge']['y']])
                    ptAbdomen = np.array([paramsScaled['gui']['abdomen']['hinge']['x'], paramsScaled['gui']['abdomen']['hinge']['y']])
                    ptCenterPre = (ptHead + ptAbdomen) / 2
                    ptBodyPre = ptHead - ptAbdomen
                    angleBodyPre = np.arctan2(ptBodyPre[1], ptBodyPre[0])
                    ptLeft = np.array([paramsScaled['gui']['left']['hinge']['x'], paramsScaled['gui']['left']['hinge']['y']])
                    ptRight = np.array([paramsScaled['gui']['right']['hinge']['x'], paramsScaled['gui']['right']['hinge']['y']])
                    ptLC = ptLeft-ptCenterPre
                    ptRC = ptRight-ptCenterPre
                    rL = np.linalg.norm(ptLC)
                    aL = np.arctan2(ptLC[1], ptLC[0]) - angleBodyPre # angle from body center to hinge in body axis coords.
                    rR = np.linalg.norm(ptRC)
                    aR = np.arctan2(ptRC[1], ptRC[0]) - angleBodyPre

                # Move the selected hinge point.
                pt = ptMouse
                paramsScaled['gui'][partnameSelected]['hinge']['x'] = float(pt[0])
                paramsScaled['gui'][partnameSelected]['hinge']['y'] = float(pt[1])
                
                # Invalidate the masks.
                self.fly.head.bValidMask    = False
                self.fly.abdomen.bValidMask = False
                self.fly.left.bValidMask    = False
                self.fly.right.bValidMask   = False
                
                # Now move the hinge points relative to the new body axis.
                if (paramsScaled['gui']['symmetric']):
                    ptHead = np.array([paramsScaled['gui']['head']['hinge']['x'], paramsScaled['gui']['head']['hinge']['y']])
                    ptAbdomen = np.array([paramsScaled['gui']['abdomen']['hinge']['x'], paramsScaled['gui']['abdomen']['hinge']['y']])
                    ptCenterPost = (ptHead + ptAbdomen) / 2
                    ptBodyPost = ptHead - ptAbdomen
                    angleBodyPost = np.arctan2(ptBodyPost[1], ptBodyPost[0])
                    ptLeft = ptCenterPost + rL * np.array([np.cos(aL+angleBodyPost), np.sin(aL+angleBodyPost)])
                    ptRight = ptCenterPost + rR * np.array([np.cos(aR+angleBodyPost), np.sin(aR+angleBodyPost)])
                    paramsScaled['gui']['left']['hinge']['x'] = float(ptLeft[0])
                    paramsScaled['gui']['left']['hinge']['y'] = float(ptLeft[1])
                    paramsScaled['gui']['right']['hinge']['x'] = float(ptRight[0])
                    paramsScaled['gui']['right']['hinge']['y'] = float(ptRight[1])

                    
            elif (partnameSelected=='left') or (partnameSelected=='right'):
                paramsScaled['gui'][partnameSelected]['hinge']['x'] = float(ptMouse[0])
                paramsScaled['gui'][partnameSelected]['hinge']['y'] = float(ptMouse[1])
                bodypartSelected.bValidMask = False

                if (paramsScaled['gui']['symmetric']):
                    ptSlave = self.get_reflection_across_bodyaxis(ptMouse)
                    paramsScaled['gui'][partnameSlave]['hinge']['x'] = float(ptSlave[0])
                    paramsScaled['gui'][partnameSlave]['hinge']['y'] = float(ptSlave[1])
                    bodypartSlave.bValidMask = False



        elif (tagSelected in ['angle_hi','angle_lo']):
            pt = ptMouse - bodypartSelected.ptHinge_i
            if (tagSelected=='angle_lo'): 
                angle_lo_b = float(bodypartSelected.transform_angle_b_from_i(np.arctan2(pt[1], pt[0])))

                if (partnameSelected in ['head','abdomen']) and (paramsScaled['gui']['symmetric']):
                    angle_hi_b = -angle_lo_b
                else:
                    angle_hi_b = paramsScaled['gui'][partnameSelected][tagOther]

            elif (tagSelected=='angle_hi'):
                angle_hi_b = float(bodypartSelected.transform_angle_b_from_i(np.arctan2(pt[1], pt[0])))
                
                if (partnameSelected in ['head','abdomen']) and (paramsScaled['gui']['symmetric']):
                    angle_lo_b = -angle_hi_b
                else:
                    angle_lo_b = paramsScaled['gui'][partnameSelected][tagOther]
            
            paramsScaled['gui'][partnameSelected]['radius_outer'] = float(max(bodypartSelected.params['gui'][partnameSelected]['radius_inner']+2*self.scale, 
                                                                      np.linalg.norm(bodypartSelected.ptHinge_i - ptMouse)))
                
            # Make angles relative to bodypart origin.
            angle_lo_b -= bodypartSelected.angleOutward_b
            angle_lo_b = (angle_lo_b+np.pi) % (2.0*np.pi) - np.pi
            angle_hi_b -= bodypartSelected.angleOutward_b
            angle_hi_b = (angle_hi_b+np.pi) % (2.0*np.pi) - np.pi
            
            # Switch to the other handle.
            if (not (angle_lo_b < angle_hi_b)):
                self.tagSelected = tagOther
                
            # Set the order of the two angles
            paramsScaled['gui'][partnameSelected]['angle_lo'] = min(angle_lo_b, angle_hi_b)
            paramsScaled['gui'][partnameSelected]['angle_hi'] = max(angle_lo_b, angle_hi_b)
            
            # Make angles relative to fly origin. 
            paramsScaled['gui'][partnameSelected]['angle_lo'] += bodypartSelected.angleOutward_b
            paramsScaled['gui'][partnameSelected]['angle_hi'] += bodypartSelected.angleOutward_b
            paramsScaled['gui'][partnameSelected]['angle_lo'] = (paramsScaled['gui'][partnameSelected]['angle_lo']+np.pi) % (2.0*np.pi) - np.pi
            paramsScaled['gui'][partnameSelected]['angle_hi'] = (paramsScaled['gui'][partnameSelected]['angle_hi']+np.pi) % (2.0*np.pi) - np.pi
            
            
            if (paramsScaled['gui'][partnameSelected]['angle_hi'] < paramsScaled['gui'][partnameSelected]['angle_lo']):
                paramsScaled['gui'][partnameSelected]['angle_hi'] += 2*np.pi
            
            bodypartSelected.bValidMask = False

            if (partnameSelected in ['left','right']):
                if (paramsScaled['gui']['symmetric']):
                    paramsScaled['gui'][partnameSlave][tagSlave]     = -paramsScaled['gui'][partnameSelected][tagSelected]
                    paramsScaled['gui'][partnameSlave][tagSelected]  = -paramsScaled['gui'][partnameSelected][tagSlave]
                    paramsScaled['gui'][partnameSlave]['radius_outer'] = paramsScaled['gui'][partnameSelected]['radius_outer']
                    bodypartSlave.bValidMask = False
                    
#                 if (paramsScaled['gui'][partnameSlave]['angle_hi'] < 0 < paramsScaled['gui'][partnameSlave]['angle_lo']):
#                     paramsScaled['gui'][partnameSlave]['angle_hi'] += 2*np.pi
 
              
        # Inner radius.
        elif (tagSelected=='radius_inner'): 
            paramsScaled['gui'][partnameSelected]['radius_inner'] = float(min(np.linalg.norm(bodypartSelected.ptHinge_i - ptMouse), 
                                                                                      bodypartSelected.params['gui'][partnameSelected]['radius_outer']-2*self.scale))
            
            bodypartSelected.bValidMask = False
            
            if (partnameSelected in ['left','right']) and (paramsScaled['gui']['symmetric']):
                paramsScaled['gui'][partnameSlave]['radius_inner'] = paramsScaled['gui'][partnameSelected]['radius_inner']
                bodypartSlave.bValidMask = False
                
        # Center.
        elif (tagSelected=='center'): 
            if (partnameSelected=='aux'):

                # Move the center point.
                pt = ptMouse
                paramsScaled['gui'][partnameSelected]['center']['x'] = float(pt[0])
                paramsScaled['gui'][partnameSelected]['center']['y'] = float(pt[1])
                bodypartSelected.bValidMask = False
                
        # Radius.
        elif (tagSelected=='radius1'): 
            pt = ptMouse - bodypartSelected.ptCenter_i
            paramsScaled['gui'][partnameSelected]['radius1'] = float(np.linalg.norm(pt))
            paramsScaled['gui'][partnameSelected]['angle'] = float(np.arctan2(pt[1], pt[0]))
            bodypartSelected.bValidMask = False        
        elif (tagSelected=='radius2'): 
            pt = bodypartSelected.ptCenter_i - ptMouse
            paramsScaled['gui'][partnameSelected]['radius2'] = float(np.linalg.norm(pt))
            paramsScaled['gui'][partnameSelected]['angle'] = float(np.arctan2(pt[1], pt[0])-np.pi/2.0)
            bodypartSelected.bValidMask = False        
                

        # Unscale the parameters since we're finished adjusting them.
        self.params = self.scale_params(paramsScaled, 1/self.scale) 


    # onMouse()
    # Handle mouse events.
    #
    def onMouse(self, event, x, y, flags, param):
        ptMouse = np.array([x, y]).clip((0,0), (self.shapeImage[1],self.shapeImage[0]))

        # Keep track of which UI element is selected.
        if (event==cv2.EVENT_LBUTTONDOWN):
            self.bMousing = True
            
            # Get the name and ui nearest the current point.
            (ui, tag, partname, iButtonSelected) = self.hit_object(ptMouse)
            self.tagSelected = tag                                          # hinge, angle_hi, angle_lo, center, radius1, radius2, radius_inner.
            self.partnameSelected = partname                                # left, right, head, abdomen.
            self.uiSelected = ui                                            # pushbutton, checkbox, handle.
            self.iButtonSelected = iButtonSelected                          # Index of the button.
            self.nameSelected = self.name_from_tagpartname(tag,partname)    # tag_partname.
            #rospy.logwarn((ui, tag, partname, iButtonSelected))

            if (self.iButtonSelected is not None):
                self.stateSelected = self.buttons[self.iButtonSelected].state
            
            self.nameSelectedNow = self.nameSelected
            self.uiSelectedNow = self.uiSelected
            

        if (self.uiSelected=='pushbutton') or (self.uiSelected=='checkbox'):
            # Get the partname and ui tag nearest the mouse point.
            (ui, tag, partname, iButtonSelected) = self.hit_object(ptMouse)
            self.nameSelectedNow     = self.name_from_tagpartname(tag,partname)
            self.tagSelectedNow      = tag
            self.partnameSelectedNow = partname
            self.uiSelectedNow       = ui
            self.iButtonSelectedNow  = iButtonSelected


            # Set selected button to 'down', others to 'up'.
            for iButton in range(len(self.buttons)):
                if (self.buttons[iButton].type=='pushbutton'):
                    if (iButton==self.iButtonSelectedNow==self.iButtonSelected) and not (event==cv2.EVENT_LBUTTONUP):
                        self.buttons[iButton].state = True # 'down'
                    else:
                        self.buttons[iButton].state = False # 'up'
                        
            # Set the checkbox.
            if (self.uiSelected=='checkbox'):
                if (self.nameSelected == self.nameSelectedNow):
                    self.buttons[self.iButtonSelected].state = not self.stateSelected # Set it to the other state when we're on the checkbox.
                else:
                    self.buttons[self.iButtonSelected].state = self.stateSelected # Set it to the original state when we're off the checkbox.

        # end if ('pushbutton' or 'checkbox'):

                        
        elif (self.uiSelected=='handle'):
            # Set the new params.
            self.update_params_from_mouse(self.tagSelected, self.partnameSelected, ptMouse.clip((0,self.yToolbar), (self.shapeImage[1],self.shapeImage[0])))
            self.fly.set_params(self.scale_params(self.params, self.scale))
        
        # end if ('handle'):
            

        if (event==cv2.EVENT_LBUTTONUP):
            # If the mouse is on the same button at mouseup, then do the action.
            if (self.uiSelected=='pushbutton'):
                if (self.nameSelected == self.nameSelectedNow == 'save_bg'):
                    self.pubCommand.publish(MsgCommand('save_background'))

                elif (self.nameSelected == self.nameSelectedNow == 'exit'):
                    self.pubCommand.publish(MsgCommand('exit'))
                    
                    
            elif (self.uiSelected=='checkbox'):
                if (self.nameSelected == self.nameSelectedNow):
                    self.buttons[self.iButtonSelected].state = not self.stateSelected
                else:
                    self.buttons[self.iButtonSelected].state = self.stateSelected


                if (self.nameSelected == self.nameSelectedNow == 'symmetry'):
                    self.params['gui']['symmetric'] = self.buttons[self.iButtonSelected].state
                    
                elif (self.nameSelected == self.nameSelectedNow == 'subtract_lr'):
                    if (not self.bHaveBackground):
                        self.buttons[iButtonSelected].state = False
                        rospy.logwarn('No background image.  Cannot use background subtraction for left/right.')

                    self.params['gui']['left']['subtract_bg']  = self.buttons[iButtonSelected].state
                    self.params['gui']['right']['subtract_bg'] = self.buttons[iButtonSelected].state
                    
                elif (self.nameSelected == self.nameSelectedNow == 'subtract_aux'):
                    if (not self.bHaveBackground):
                        self.buttons[iButtonSelected].state = False
                        rospy.logwarn('No background image.  Cannot use background subtraction for aux.')

                    self.params['gui']['aux']['subtract_bg']  = self.buttons[iButtonSelected].state
                    
                elif (self.nameSelected == self.nameSelectedNow == 'subtract_head'):
                    if (not self.bHaveBackground):
                        self.buttons[iButtonSelected].state = False
                        rospy.logwarn('No background image.  Cannot use background subtraction for head.')

                    self.params['gui']['head']['subtract_bg']  = self.buttons[iButtonSelected].state
                    
                elif (self.nameSelected == self.nameSelectedNow == 'subtract_abdomen'):
                    if (not self.bHaveBackground):
                        self.buttons[iButtonSelected].state = False
                        rospy.logwarn('No background image.  Cannot use background subtraction for abdomen.')

                    self.params['gui']['abdomen']['subtract_bg']  = self.buttons[iButtonSelected].state
                    
                elif (self.nameSelected == self.nameSelectedNow == 'head'):
                    self.params['gui']['head']['track'] = self.buttons[iButtonSelected].state
                    
                elif (self.nameSelected == self.nameSelectedNow == 'abdomen'):
                    self.params['gui']['abdomen']['track'] = self.buttons[iButtonSelected].state

                elif (self.nameSelected == self.nameSelectedNow == 'left'):
                    self.params['gui']['left']['track']  = self.buttons[iButtonSelected].state

                elif (self.nameSelected == self.nameSelectedNow == 'right'):
                    self.params['gui']['right']['track'] = self.buttons[iButtonSelected].state

                elif (self.nameSelected == self.nameSelectedNow == 'aux'):
                    self.params['gui']['aux']['track'] = self.buttons[iButtonSelected].state
                    if (self.params['gui']['aux']['track']):
                        self.fly.aux.wingbeat.warn()
                    

                elif (self.nameSelected == self.nameSelectedNow == 'stabilize'):
                    self.params['gui']['head']['stabilize']    = self.buttons[iButtonSelected].state
                    self.params['gui']['abdomen']['stabilize'] = self.buttons[iButtonSelected].state
                    self.params['gui']['left']['stabilize']    = self.buttons[iButtonSelected].state
                    self.params['gui']['right']['stabilize']   = self.buttons[iButtonSelected].state

                elif (self.nameSelected == self.nameSelectedNow == 'windows'):
                    self.params['gui']['windows'] = self.buttons[iButtonSelected].state


            if (self.uiSelected in ['handle','checkbox']):
                self.fly.set_params(self.scale_params(self.params, self.scale))
                self.fly.create_masks(self.shapeImage)
    
                # Save the results.
                SetDict().set_dict_with_preserve(self.params, rospy.get_param(self.nodename, {}))

                rospy.set_param(self.nodename+'/gui', self.params['gui'])
                with self.lockParams:
                    rosparam.dump_params(self.yamlfile, self.nodename+'/gui')

            # Dump the params to the screen, for debugging.
#             for k,v in self.params.iteritems():
#                 rospy.logwarn((k,type(v)))
#                 if (type(v)==type({})):
#                     for k2,v2 in v.iteritems():
#                         rospy.logwarn('     %s, %s' % (k2,type(v2)))
                

            self.bMousing           = False
            self.nameSelected       = None
            self.nameSelectedNow    = None
            self.uiSelected         = None
            self.uiSelectedNow      = None
            self.iButtonSelected    = None
            self.iButtonSelectedNow = None

            
                
    def run(self):
        if (self.params['gui']['aux']['track']):
            self.fly.aux.wingbeat.warn()
        
        while (not rospy.is_shutdown()):
            self.process_image()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main = MainWindow()

    rospy.logwarn('')
    rospy.logwarn('**************************************************************************')
    rospy.logwarn('     Kinefly: Camera-based Tethered Insect Kinematics Analyzer for ROS')
    rospy.logwarn('         by Steve Safarik, Floris van Breugel (c) 2014')
    rospy.logwarn('**************************************************************************')
    rospy.logwarn('')


    main.run()

    #cProfile.run('main.run()', '/home/ssafarik/profile.pstats')
    # Note to do profiling:
    # $ sudo apt-get install graphviz
    # $ git clone https://code.google.com/p/jrfonseca.gprof2dot/ gprof2dot
    # $ mkdir ~/bin
    # $ ln -s "$PWD"/gprof2dot/gprof2dot.py ~/bin
    # $ cd /home/ssafarik
    # $ ~/bin/gprof2dot.py -f pstats profile.pstats | dot -Tsvg -o callgraph.svg
    
