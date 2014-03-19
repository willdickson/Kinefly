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
import threading
import dynamic_reconfigure.server

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Header, String
from StrokelitudeROS.srv import *
from StrokelitudeROS.msg import float32list as float32list_msg
from StrokelitudeROS.msg import MsgFlystate, MsgWing, MsgBodypart, MsgCommand
from StrokelitudeROS.cfg import strokelitudeConfig

gOffsetHandle = 10 # How far from the point-of-interest we place some of the handles.



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
    
    x = (x1*x3*y2 - x2*x3*y1 - x1*x4*y2 + x2*x4*y1 - x1*x3*y4 + x1*x4*y3 + x2*x3*y4 - x2*x4*y3)/(x1*y3 - x3*y1 - x1*y4 - x2*y3 + x3*y2 + x4*y1 + x2*y4 - x4*y2)
    y = (x1*y2*y3 - x2*y1*y3 - x1*y2*y4 + x2*y1*y4 - x3*y1*y4 + x4*y1*y3 + x3*y2*y4 - x4*y2*y3)/(x1*y3 - x3*y1 - x1*y4 - x2*y3 + x3*y2 + x4*y1 + x2*y4 - x4*y2)        
    
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
class Handle(object):
    def __init__(self, pt=np.array([0,0]), color=bgra_dict['white'], name=None):
        self.pt = pt
        self.name = name
        self.scale = 1.0

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
        
        #ptText = self.pt+np.array([5,5])
        #cv2.putText(image, self.name, tuple(ptText.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4*self.scale, self.color)
        
# end class Handle
                

###############################################################################
###############################################################################
class SetDict(object):
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



###############################################################################
###############################################################################
class PhaseCorrelation(object):
    # get_shift()
    # Calculate the coordinate shift between the two images.
    #
    def get_shift(self, imgA, imgB):
        if (imgA is not None) and (imgB is not None) and (imgA.shape==imgB.shape):
            # Phase correlation.
            A  = cv2.dft(imgA)
            B  = cv2.dft(imgB)
            AB = cv2.mulSpectrums(A, B, flags=0, conjB=True)
            crosspower = AB / cv2.norm(AB)
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
            
        else:
            rv = np.array([0.0, 0.0])
            
        return rv
        
        
###############################################################################
###############################################################################
# Find the two largest gradients in the horizontal intensity profile of an image.
#
class EdgeDetector(object):
    # get_edges()
    # Get the angles of the two wing edges.
    #
    def get_edges(self, image):
        angle1 = 0.0
        angle2 = 0.0
        
        intensities = np.sum(image, 0).astype(np.int64)
        intensitiesF = filter_median(intensities, q=1)
        
        #diff = np.diff(intensitiesF) 
        diff = intensitiesF[3:] - intensitiesF[:-3]

        #rospy.logwarn((image.dtype, intensities.dtype, intensitiesF.dtype, diff.dtype))
        #rospy.logwarn(intensitiesF)
        #rospy.logwarn(diff)
        
        iMax = np.argmax(diff)
        iMin = np.argmin(diff)
        if (np.abs(diff[iMax]) > np.abs(diff[iMin])): 
            (iMajor,iMinor) = (iMax,iMin)  
        else:
            (iMajor,iMinor) = (iMin,iMax)
        
        pixelmean = np.mean(image)
        return (iMajor, iMinor, pixelmean/255.0)
    
    
    
###############################################################################
###############################################################################
class WindowFunctions(object):
    # create_wfn_hanning()
    # Create a 2D Hanning window function.
    #
    def create_wfn_hanning(self, shape):
        (height,width) = shape
        wfn = np.ones(shape, dtype=np.float32)
        if (height>1) and (width>1):
            for i in range(width):
                 for j in range(height):
                     x = 2*np.pi*i/(width-1) # x ranges 0 to 2pi across the image width
                     y = 2*np.pi*j/(height-1) # y ranges 0 to 2pi across the image height
                     wfn[j][i] = 0.5*(1-np.cos(x)) * 0.5*(1-np.cos(y))
                 
        return wfn


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


    
###############################################################################
###############################################################################
class Fly(object):
    def __init__(self, params={}):
        self.head    = BodySegment(name='head',    params=params, color='cyan', bEqualizeHist=True) 
        self.abdomen = BodySegment(name='abdomen', params=params, color='magenta', bEqualizeHist=True) 
        self.right   = Wing(name='right',       params=params, color='red', bEqualizeHist=False)
        self.left    = Wing(name='left',        params=params, color='green', bEqualizeHist=False)
        self.bgra_body = bgra_dict['light_gray']
        self.ptBody1 = None
        self.ptBody2 = None
        self.bInvertcolor = False
        self.iCount  = 0
        self.stamp   = 0

        self.pubFlystate = rospy.Publisher('strokelitude/flystate', MsgFlystate)


    def set_params(self, params):
        self.params = params
        
        self.head.set_params(params)
        self.abdomen.set_params(params)
        self.left.set_params(params)
        self.right.set_params(params)

        self.angleBody_i = self.get_bodyangle_i()
        pt1 = [params['head']['hinge']['x'], params['head']['hinge']['y']]
        pt2 = [params['abdomen']['hinge']['x'], params['abdomen']['hinge']['y']]
        pt3 = [params['left']['hinge']['x'], params['left']['hinge']['y']]
        pt4 = [params['right']['hinge']['x'], params['right']['hinge']['y']]
        self.ptBodyCenter_i = get_intersection(pt1,pt2,pt3,pt4)
        r = max(params['left']['radius_outer'], params['right']['radius_outer'])
        self.ptBody1 = tuple((self.ptBodyCenter_i + r * np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)])).astype(int))
        self.ptBody2 = tuple((self.ptBodyCenter_i - r * np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)])).astype(int))
        
        # Radius of an area approximately where the thorax would be.
        self.rThorax = np.linalg.norm(np.array([params['head']['hinge']['x'], params['head']['hinge']['y']]) - np.array([params['abdomen']['hinge']['x'], params['abdomen']['hinge']['y']]))/2.0
        self.bInvertColorValid = False
        
    
    def create_masks(self, shapeImage):
        self.head.create_mask (shapeImage)
        self.abdomen.create_mask (shapeImage)
        self.right.create_mask (shapeImage)
        self.left.create_mask (shapeImage)


    def get_bodyangle_i(self):
        angle_i = get_angle_from_points_i(self.abdomen.ptHinge_i, self.head.ptHinge_i)
        #angleBody_i  = (angle_i + np.pi) % (2.0*np.pi) - np.pi
        angleBody_i  = angle_i
        
        return angleBody_i
        

    # Calculate what we think the bInvertcolor flag should be to make white-on-black.        
    def calc_invertcolor(self, image):
        # Get a roi around the body center.
        xMin = max(0,self.ptBodyCenter_i[0]-int(0.75*self.rThorax))
        yMin = max(0,self.ptBodyCenter_i[1]-int(0.75*self.rThorax))
        xMax = min(self.ptBodyCenter_i[0]+int(0.75*self.rThorax), image.shape[1]-1)
        yMax = min(self.ptBodyCenter_i[1]+int(0.75*self.rThorax), image.shape[0]-1)
        imgThorax = image[yMin:yMax, xMin:xMax]

        # Midpoint between darkest & lightest colors.
        threshold = np.mean(image) 
        #rospy.logwarn((np.min(image), np.median(image), np.mean(image), np.max(image), np.mean(imgThorax)))
        
        # If the roi is too dark, then set bInvertcolor.
        if (np.mean(imgThorax) <= threshold):
            self.bInvertcolor = True
        else:
            self.bInvertcolor = False
            
        self.bInvertColorValid = True
        
        
                
    def set_background(self, image):
        self.head.set_background(image)
        self.abdomen.set_background(image)
        self.left.set_background(image)
        self.right.set_background(image)

    
    def update_handle_points(self):
        self.head.update_handle_points()
        self.abdomen.update_handle_points()
        self.left.update_handle_points()
        self.right.update_handle_points()
        

    def update(self, header=None, image=None):
        if (image is not None):
            if (not self.bInvertColorValid):
                self.calc_invertcolor(image)
            
            if (self.bInvertcolor):
                image = 255-image
    
            if (header is not None):
                self.stamp = header.stamp
            else:
                self.stamp = rospy.Time.now()
            
            self.head.update(header, image)
            self.abdomen.update(header, image)
            self.left.update(header, image)
            self.right.update(header, image)
            

            
    def draw(self, image):
        # Draw line to indicate the body axis.
        cv2.line(image, self.ptBody1, self.ptBody2, self.bgra_body, 1) # Draw a line longer than just head-to-abdomen.
                
        self.head.draw(image)
        self.abdomen.draw(image)
        self.left.draw(image)
        self.right.draw(image)

        
    
    def publish(self):
        flystate              = MsgFlystate()
        flystate.header       = Header(seq=self.iCount, stamp=self.stamp, frame_id='Fly')
        if (self.params['left']['track']) and (self.left.state.angle1 is not None) and (self.left.state.angle2 is not None):
            flystate.left     = MsgWing(intensity=self.left.state.intensity, 
                                        angle1=self.left.state.angle1, 
                                        angle2=self.left.state.angle2)
        else:
            flystate.left     = MsgWing(intensity=0.0, angle1=0.0, angle2=0.0)
            
        if (self.params['right']['track']) and (self.right.state.angle1 is not None) and (self.right.state.angle2 is not None):
            flystate.right    = MsgWing(intensity=self.right.state.intensity, 
                                        angle1=self.right.state.angle1, 
                                        angle2=self.right.state.angle2)
        else:
            flystate.right    = MsgWing(intensity=0.0, angle1=0.0, angle2=0.0)
            
        if (self.params['head']['track']):
            flystate.head     = MsgBodypart(intensity=self.head.state.intensity,    
                                            radius=self.head.state.radius,    
                                            angle=self.head.state.angle)
        else:
            flystate.head     = MsgBodypart(intensity=0.0, angle=0.0, radius=0.0)

        if (self.params['abdomen']['track']):
            flystate.abdomen  = MsgBodypart(intensity=self.abdomen.state.intensity, 
                                            radius=self.abdomen.state.radius, 
                                            angle=self.abdomen.state.angle)
        else:
            flystate.abdomen  = MsgBodypart(intensity=0.0, angle=0.0, radius=0.0)


        self.iCount += 1
        
        self.pubFlystate.publish(flystate)
        

# end class Fly

        
###############################################################################
###############################################################################
# Contains the common behavior for tracking a bodypart via a polar coordinates transform, e.g. Head, Abdomen, or Wing.
#
class PolarTrackedBodypart(object):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        self.name        = name
        self.bEqualizeHist = bEqualizeHist

        self.bInitializedMasks = False

        self.bgra        = bgra_dict[color]
        self.bgra_dim    = tuple(np.array(bgra_dict[color])/2)
        self.bgra_state  = bgra_dict['red']
        self.pixelmax    = 255.0

        self.shape     = (np.inf, np.inf)
        self.ptHinge_i = np.array([0,0])
        self.roi       = None
        
        self.create_wfn               = WindowFunctions().create_wfn_tukey
        self.wfnRoi                   = None
        self.wfnRoiMaskedPolarCropped = None
        
#         self.maskRoiEllipse1        = None
        self.maskRoiEllipse           = None
        
        self.stampPrev = None
        self.dt = rospy.Time(0)
        self.polartransforms   = PolarTransforms()

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

        # Auxiliary windows.
        self.windowBG         = ImageWindow(False, self.name+'BG')
        self.windowFG         = ImageWindow(False, self.name+'FG')
        self.windowPolar      = ImageWindow(False, self.name+'Polar')
        self.windowMask       = ImageWindow(False, self.name+'Mask')
#         self.windowStabilized = ImageWindow(False, self.name+'Stable')


    
    # set_params()
    # Set the given params dict into this object, and cache a few values.
    #
    def set_params(self, params):
        self.params = params

        self.rc_background = self.params['rc_background']
        self.angleBody_i = self.get_bodyangle_i()
        self.ptHinge_i = np.array([self.params[self.name]['hinge']['x'], self.params[self.name]['hinge']['y']])
        
        # Compute the body-outward-facing angle, which is the angle from the body center to the bodypart hinge.
        pt1 = [params['head']['hinge']['x'], params['head']['hinge']['y']]
        pt2 = [params['abdomen']['hinge']['x'], params['abdomen']['hinge']['y']]
        pt3 = [params['left']['hinge']['x'], params['left']['hinge']['y']]
        pt4 = [params['right']['hinge']['x'], params['right']['hinge']['y']]
        ptBodyCenter_i = get_intersection(pt1,pt2,pt3,pt4)
        self.angleOutward_i = float(np.arctan2(self.ptHinge_i[1]-ptBodyCenter_i[1], self.ptHinge_i[0]-ptBodyCenter_i[0]))
        
        self.cosAngleBody = np.cos(self.angleBody_i)
        self.sinAngleBody = np.sin(self.angleBody_i)
        self.cosAngleOutward = np.cos(self.angleOutward_i)
        self.sinAngleOutward = np.sin(self.angleOutward_i)
        self.R = np.array([[self.cosAngleOutward, -self.sinAngleOutward], [self.sinAngleOutward, self.cosAngleOutward]])

        # Turn on/off the extra windows.
        self.windowPolar.set_enable(self.params['extra_windows'] and self.params[self.name]['track'])
        self.windowBG.set_enable(self.params['extra_windows'] and self.params[self.name]['track'] and self.params[self.name]['subtract_bg'])
        self.windowFG.set_enable(self.params['extra_windows'] and self.params[self.name]['track'])
#         self.windowStabilized.set_enable(self.params['extra_windows'] and self.params[self.name]['track'])

        self.angle_hi_i = self.transform_angle_i_from_b(self.params[self.name]['angle_hi'])
        self.angle_lo_i = self.transform_angle_i_from_b(self.params[self.name]['angle_lo'])
        
        # Refresh the handle points.
        self.update_handle_points()



    # transform_angle_i_from_b()
    # Transform an angle from the fly body frame to the camera image frame.
    #
    def transform_angle_i_from_b(self, angle_b):
#         if self.name == 'right':
#             angle_i  =  angle_b + self.angleBody_i + np.pi/2.0
#         else: # left
#             angle_i  = -angle_b + self.angleBody_i + np.pi/2.0 + np.pi
        angle_i = angle_b + self.angleBody_i 
             
#         angle_i = (angle_i+np.pi) % (2.0*np.pi) - np.pi
        return angle_i
        

    # transform_angle_b_from_i()
    # Transform an angle from the camera image frame to the fly frame: longitudinal axis head is 0, CW positive.
    #
    def transform_angle_b_from_i(self, angle_i):
#         if (self.name in ['head','abdomen','right']:
#             angle_b  =  angle_i - self.angleOutward_i
#         else:  
#             angle_b  = -angle_i + self.angleBody_i + np.pi/2.0 + np.pi
        angle_b = angle_i - self.angleBody_i
        angle_b = ((angle_b+np.pi) % (2.0*np.pi)) - np.pi

        return angle_b
         

    def get_bodyangle_i(self):
        angle_i = get_angle_from_points_i(np.array([self.params['abdomen']['hinge']['x'], self.params['abdomen']['hinge']['y']]), 
                                          np.array([self.params['head']['hinge']['x'], self.params['head']['hinge']['y']]))
        #angleBody_i  = (angle_i + np.pi) % (2.0*np.pi) - np.pi
        angleBody_i  = float(angle_i)
        return angleBody_i 
        
                
    # create_mask()
    # Create elliptical wedge masks, and window functions.
    #
    def create_mask(self, shape):
        # Create the mask (for polar images the line-of-interest is at the midpoint).
        mask = np.zeros(shape, dtype=np.uint8)
        
        # Args for the two ellipse calls.
        x = int(self.params[self.name]['hinge']['x'])
        y = int(self.params[self.name]['hinge']['y'])
        r_outer = int(self.params[self.name]['radius_outer'])
        r_inner = int(self.params[self.name]['radius_inner'])
        bgra = bgra_dict['white']
        
        # Draw the mask.
        cv2.ellipse(mask, (x, y), (r_outer, r_outer), 0, int(np.rad2deg(self.angle_hi_i)), int(np.rad2deg(self.angle_lo_i)), bgra, cv.CV_FILLED)
        cv2.ellipse(mask, (x, y), (r_inner, r_inner), 0, 0, 360, bgra_dict['black'], cv.CV_FILLED)
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
            self.maskRoiEllipse = mask[yMin:yMax, xMin:xMax]
    
            self.i_0 = self.params[self.name]['hinge']['y'] - self.roi[1]
            self.j_0 = self.params[self.name]['hinge']['x'] - self.roi[0]
            
            self.wfnRoi = None
            self.wfnRoiMaskedPolarCropped = None
            self.bInitializedMasks = True
    
    
            # Find where the mask might be clipped.  First, draw an unclipped ellipse.        
            delta = 1
            pts =                cv2.ellipse2Poly((x, y), (r_outer, r_outer), 0, int(np.rad2deg(self.angle_hi_i)), int(np.rad2deg(self.angle_lo_i)), delta)
            pts = np.append(pts, cv2.ellipse2Poly((x, y), (r_inner, r_inner), 0, int(np.rad2deg(self.angle_hi_i)), int(np.rad2deg(self.angle_lo_i)), delta), 0)
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
            if (roiClipped[0]>0):
                rClip0 = 1.0 - (float(roiClipped[0])/float(self.j_0))
            if (roiClipped[1]>0):
                rClip1 = 1.0 - (float(roiClipped[1])/float(self.i_0))
            if (roiClipped[2]>0):
                rClip2 = 1.0 - (float(roiClipped[2])/float(j_n-self.j_0))
            if (roiClipped[3]>0):
                rClip3 = 1.0 - (float(roiClipped[3])/float(i_n-self.i_0))
    
            self.rClip = np.min([rClip0, rClip1, rClip2, rClip3])
        else:
            rospy.logwarn('Empty mask.')
        
        
    # set_background()
    # Set the given image as the background image.
    #                
    def set_background(self, image):
        self.imgFullBackground = image.astype(np.float32)
        self.imgRoiBackground = None
        
        
    def update_background(self):
        dt = max(0, self.dt.to_sec())
        alphaBackground = 1.0 - np.exp(-dt / self.rc_background)

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
        

    def update_roi(self, image):
        self.shape = image.shape
        
        # Extract the ROI images.
        self.imgRoi = copy.deepcopy(image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]])

        # Background Subtraction.
        if (self.params[self.name]['subtract_bg']):
            if (self.imgRoiBackground is not None):
                self.imgRoiFg = cv2.absdiff(self.imgRoi, self.imgRoiBackground.astype(np.uint8))
            else:
                rospy.logwarn('You need to save a background image to use SubtractBG.')
        else:
            self.imgRoiFg = self.imgRoi
            
        # Equalize the brightness/contrast.
        if (self.bEqualizeHist):
            if (self.imgRoiFg is not None):
                self.imgRoiFg -= np.min(self.imgRoiFg)
                max2 = np.max(self.imgRoiFg)
                self.imgRoiFg *= (255.0/float(max2))
            
        self.windowFG.set_image(self.imgRoiFg) 
        

            
    def update_polar(self):
        if (self.imgRoiFg is not None):
            # Apply the mask.
            if (self.maskRoiEllipse is not None):
                self.imgRoiFgMasked = self.imgRoiFg#cv2.bitwise_and(self.imgRoiFg, self.maskRoiEllipse)
                #self.imgRoiFgMasked  = cv2.multiply(self.imgRoiFg.astype(np.float32), self.wfnRoi)

            xMax = self.imgRoiFg.shape[1]-1
            yMax = self.imgRoiFg.shape[0]-1
            
            theta_0a = self.angle_lo_i - self.angleOutward_i
            theta_1a = self.angle_hi_i - self.angleOutward_i
            
            radius_mid = (self.params[self.name]['radius_outer']+self.params[self.name]['radius_inner'])/2.0
            dr = (self.params[self.name]['radius_outer']-self.params[self.name]['radius_inner'])/2.0
                
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
                
            # Find the y value where the black band should be cropped out.
            sumY = np.sum(self.imgRoiFgMaskedPolar,1)
            iSumY = np.where(sumY==0)[0]
            if (len(iSumY)>0):
                iMinY = np.min(iSumY)
            else:
                iMinY = self.imgRoiFgMaskedPolar.shape[0]

            self.imgRoiFgMaskedPolarCropped = self.imgRoiFgMaskedPolar[0:iMinY]
    
             
            if (self.bInitializedMasks):
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
        x = self.params[self.name]['hinge']['x']
        y = self.params[self.name]['hinge']['y']
        radius_outer = self.params[self.name]['radius_outer']
        radius_inner = self.params[self.name]['radius_inner']
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
                
            self.update_background()
            self.update_roi(image)
            self.update_polar()


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
            tagHandle,handle = ('hinge',self.handles['hinge'])
            if (handle.hit_test(ptMouse)):
                tag = tagHandle
            
                
        return (self.name, tag)
    

    def draw_handles(self, image):
        # Draw all handle points, or only just the hinge handle.
        if (self.params[self.name]['track']):
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

        if (self.params[self.name]['track']):
            x = int(self.params[self.name]['hinge']['x'])
            y = int(self.params[self.name]['hinge']['y'])
            radius_outer = int(self.params[self.name]['radius_outer'])
            radius_inner = int(self.params[self.name]['radius_inner'])
            radius_mid = int((self.params[self.name]['radius_outer']+self.params[self.name]['radius_inner'])/2.0)

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
                        self.bgra, 
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

    
            # Show the aux windows.
            self.windowBG.show()
            self.windowFG.show()
            self.windowPolar.show()
            self.windowMask.show()
#             self.windowStabilized.show()
                
# end class PolarTrackedBodypart


###############################################################################
###############################################################################
# Head or Abdomen.
class BodySegment(PolarTrackedBodypart):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        PolarTrackedBodypart.__init__(self, name, params, color, bEqualizeHist)
        
        self.phasecorr         = PhaseCorrelation()
        self.state             = Struct()
        self.stateOrigin       = Struct()
        self.stateOriginOffset = Struct()
        self.stateLo           = Struct()
        self.stateHi           = Struct()
        
        self.windowTest = ImageWindow(False, self.name+'Test')
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

        self.stateOrigin.intensity = 0.0
        self.stateOrigin.angle = 0.0
        self.stateOrigin.radius = (self.params[self.name]['radius_outer']+self.params[self.name]['radius_inner'])/2.0

        self.state.intensity = 0.0
        self.state.angle = 0.0
        self.state.radius = 0.0

        self.stateLo.intensity = np.inf
        self.stateLo.angle = np.inf
        self.stateLo.radius = np.inf

        self.stateHi.intensity = -np.inf
        self.stateHi.angle = -np.inf
        self.stateHi.radius = -np.inf
        

        
    # update_state()
    # Update the bodypart translation & rotation.
    #
    def update_state(self):
        imgNow = self.imgRoiFgMaskedPolarCroppedWindowed
        
        # Get the rotation & expansion between images.
        if (imgNow is not None) and (self.imgComparison is not None):
            (rShift, aShift) = self.phasecorr.get_shift(imgNow, self.imgComparison)
            dAngleOffset = aShift * (self.params[self.name]['angle_hi']-self.params[self.name]['angle_lo']) / float(imgNow.shape[1])
            dRadiusOffset = rShift
            self.state.angle  = self.stateOrigin.angle  + dAngleOffset
            self.state.radius = self.stateOrigin.radius + dRadiusOffset
            
            # Get min,max's
            self.stateLo.angle  = min(self.stateLo.angle, self.state.angle)
            self.stateHi.angle  = max(self.stateHi.angle, self.state.angle)
            self.stateLo.radius = min(self.stateLo.radius, self.state.radius)
            self.stateHi.radius = max(self.stateHi.radius, self.state.radius)
            
            # Control the (angle,radius) offset to be at the midpoint of loangle, hiangle
            # Whenever an image appears that is closer to the midpoint of loangle, hiangle, then
            # take that image as the new comparison image.  Thus driving the comparison image 
            # toward the midpoint image over time.
            if (self.iCount>100):
                # If angle and radius are near their mean values, then take a new initial image, and set the origin.
                refAngle = np.abs((self.stateHi.angle + self.stateLo.angle)/2.0)
                
                if (-refAngle < self.state.angle < refAngle):
                    self.imgComparison = imgNow
                    self.windowTest.set_image(self.imgComparison)

                    # Converge the origin to zero.
                    self.stateLo.angle -= self.state.angle
                    self.stateHi.angle -= self.state.angle
                    self.state.angle = 0.0
                    

            # Stabilized image.
#             size = (self.imgRoiFgMaskedPolar.shape[1],
#                     self.imgRoiFgMaskedPolar.shape[0])
#             center = (self.imgRoiFgMaskedPolar.shape[1]/2.0+aShift, 
#                       self.imgRoiFgMaskedPolar.shape[0]/2.0+rShift)
#             self.imgStabilized = cv2.getRectSubPix(self.imgRoiFgMaskedPolar, size, center)
#             
#             self.windowStabilized.set_image(self.imgStabilized)
            
        self.state.intensity = float(np.mean(self.imgRoiFgMaskedPolar)/255.0)
        
        
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, header, image):
        PolarTrackedBodypart.update(self, header, image)

        if (self.imgComparison is None) and (self.iCount>1):
            self.imgComparison = self.imgRoiFgMaskedPolarCroppedWindowed
            self.windowTest.set_image(self.imgComparison)
        
        if (self.params[self.name]['track']):
            self.update_state()
            
    
    
    # draw()
    # Draw the outline.
    #
    def draw(self, image):
        PolarTrackedBodypart.draw(self, image)

        if (self.params[self.name]['track']):
            # Draw the bodypart state position.
            pt = self.R.dot([self.state.radius*np.cos(self.state.angle), 
                             self.state.radius*np.sin(self.state.angle)]) 
            ptState_i = clip_pt((int(pt[0]+self.params[self.name]['hinge']['x']), 
                                 int(pt[1]+self.params[self.name]['hinge']['y'])), image.shape) 
            
            cv2.ellipse(image,
                        ptState_i,
                        (2,2),
                        0,
                        0,
                        360,
                        self.bgra_state, 
                        1)
            
            # Draw the min/max state position.
            try:
                pt = self.R.dot([self.state.radius*np.cos(self.stateLo.angle), 
                                 self.state.radius*np.sin(self.stateLo.angle)]) 
                ptStateLo_i = clip_pt((int(pt[0]+self.params[self.name]['hinge']['x']), 
                                       int(pt[1]+self.params[self.name]['hinge']['y'])), image.shape) 
                
                
                pt = self.R.dot([self.state.radius*np.cos(self.stateHi.angle), 
                                 self.state.radius*np.sin(self.stateHi.angle)]) 
                ptStateHi_i = clip_pt((int(pt[0]+self.params[self.name]['hinge']['x']), 
                                       int(pt[1]+self.params[self.name]['hinge']['y'])), image.shape) 
                
                image[ptStateLo_i[1]][ptStateLo_i[0]] = 0
                image[ptStateHi_i[1]][ptStateHi_i[0]] = 0
            except ValueError:
                pass

            
            # Draw line from hinge to state point.        
            ptHinge_i = clip_pt((int(self.ptHinge_i[0]), int(self.ptHinge_i[1])), image.shape) 
            cv2.line(image, ptHinge_i, ptState_i, self.bgra_state, 1)
            
            self.windowTest.show()
        
# end class BodySegment

    

###############################################################################
###############################################################################
# Track a Wing.
class Wing(PolarTrackedBodypart):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        PolarTrackedBodypart.__init__(self, name, params, color, bEqualizeHist)
        
        self.name = name
        self.edgedetector      = EdgeDetector()
        self.state             = Struct()
        self.bFlying = True
        
        self.windowTest = ImageWindow(False, self.name+'Edge')
        self.set_params(params)

    
    # set_params()
    # Set the given params dict into this object.
    #
    def set_params(self, params):
        PolarTrackedBodypart.set_params(self, params)
        
        self.imgRoiBackground = None
        
        self.iCount = 0

        self.state.intensity = 0.0
        self.state.angle1 = 0.0
        self.state.angle2 = 0.0

        

        
    # update_state()
    # Compute wing angles.
    #
    def update_state(self):
        imgNow = self.imgRoiFgMaskedPolarCropped
        self.windowTest.set_image(imgNow)
        
        # Get the rotation & expansion between images.
        if (imgNow is not None):
            (iEdge1, iEdge2, intensity) = self.edgedetector.get_edges(imgNow)
            angle1 = self.params[self.name]['angle_lo'] + iEdge1 * (self.params[self.name]['angle_hi']-self.params[self.name]['angle_lo']) / float(imgNow.shape[1])
            angle2 = self.params[self.name]['angle_lo'] + iEdge2 * (self.params[self.name]['angle_hi']-self.params[self.name]['angle_lo']) / float(imgNow.shape[1])
            
            # Put angle into the bodypart frame.
            self.state.angle1 = angle1 - (self.angleOutward_i-self.angleBody_i)
            self.state.angle2 = angle2 - (self.angleOutward_i-self.angleBody_i)
            
            if (self.name=='left'):
                self.state.angle1 *= -1
                self.state.angle2 *= -1
                
            self.state.intensity = intensity

        
    def update_flight_status(self):
        if (self.state.intensity > self.params['threshold_intensity_flight']):
            self.bFlying = True
        else:
            self.bFlying = False
    
    
    # update()
    # Update all the internals given a foreground camera image.
    #
    def update(self, header, image):
        PolarTrackedBodypart.update(self, header, image)

        if (self.params[self.name]['track']):
            self.update_state()
            self.update_flight_status()
            
            
    
    
    # draw()
    # Draw the outline.
    #
    def draw(self, image):
        PolarTrackedBodypart.draw(self, image)

        # Leading and trailing edges
        if (self.state.angle1 is not None):
            a = 1
            if (self.name=='left'):
                a = -1

            angle1 =  a*self.state.angle1 + (self.angleOutward_i-self.angleBody_i)
            angle2 =  a*self.state.angle2 + (self.angleOutward_i-self.angleBody_i)
                
            
            angle1_i = self.transform_angle_i_from_b(angle1)
            angle2_i = self.transform_angle_i_from_b(angle2)
            
            x0 = self.ptHinge_i[0] + self.params[self.name]['radius_inner'] * np.cos(angle1_i)
            y0 = self.ptHinge_i[1] + self.params[self.name]['radius_inner'] * np.sin(angle1_i)
            x1 = self.ptHinge_i[0] + self.params[self.name]['radius_outer'] * np.cos(angle1_i)
            y1 = self.ptHinge_i[1] + self.params[self.name]['radius_outer'] * np.sin(angle1_i)
            cv2.line(image, (int(x0),int(y0)), (int(x1),int(y1)), self.bgra, 1)
            
            if (self.params['n_edges']==2):
                x0 = self.ptHinge_i[0] + self.params[self.name]['radius_inner'] * np.cos(angle2_i)
                y0 = self.ptHinge_i[1] + self.params[self.name]['radius_inner'] * np.sin(angle2_i)
                x1 = self.ptHinge_i[0] + self.params[self.name]['radius_outer'] * np.cos(angle2_i)
                y1 = self.ptHinge_i[1] + self.params[self.name]['radius_outer'] * np.sin(angle2_i)
                cv2.line(image, (int(x0),int(y0)), (int(x1),int(y1)), self.bgra, 1)


        if (self.params[self.name]['track']):
            self.windowTest.show()
        
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
        self.lock = threading.Lock()
        
        # initialize
        rospy.init_node('strokelitude', anonymous=True)
        
        # initialize display
        self.window_name = 'Strokelitude'
        cv2.namedWindow(self.window_name,1)
        self.cvbridge = CvBridge()
        
        # Load the parameters yaml file.
        self.parameterfile = os.path.expanduser(rospy.get_param('strokelitude/parameterfile', '~/strokelitude.yaml'))
        with self.lock:
            try:
                self.params = rosparam.load_file(self.parameterfile)[0][0]
            except (rosparam.RosParamException, IndexError), e:
                rospy.logwarn('%s.  Using default values.' % e)
                self.params = {}
            
        defaults = {'filenameBackground':'~/strokelitude.png',
                    'image_topic':'/camera/image_raw',
                    'use_gui':True,                     # You can turn off the GUI to speed the framerate.
                    'extra_windows':True,               # Show the helpful extra windows.
                    'symmetric':True,                   # Forces the UI to remain symmetric.
                    'resolution_radians':0.0174532925,  # Coarser resolution will speed the framerate. 1 degree == 0.0174532925 radians.
                    'threshold_intensity_flight':0.1,
                    'scale_image':1.0,                  # Reducing the image scale will speed the framerate.
                    'n_edges':1,                        # Number of edges per wing to find.  1 or 2.
                    'rc_background':10.0,                # Time constant of the moving average background.
                    'head':   {'track':True,
                               'autozero':True,
                               'subtract_bg':False,        # Use background subtraction?
                               'hinge':{'x':300,
                                        'y':150},
                               'radius_outer':100,
                               'radius_inner':10,
                               'angle_hi':0.7854, 
                               'angle_lo':-0.7854},
                    'abdomen':{'track':True,
                               'autozero':True,
                               'subtract_bg':False,     # Use background subtraction?
                               'hinge':{'x':300,
                                        'y':250},
                               'radius_outer':120,
                               'radius_inner':10,
                               'angle_hi':0.7854, 
                               'angle_lo':-0.7854},
                    'left':   {'track':True,
                               'autozero':False,
                               'subtract_bg':True,         # Use background subtraction?
                               'hinge':{'x':100,
                                        'y':100},
                               'radius_outer':80,
                               'radius_inner':10,
                               'angle_hi':0.7854, 
                               'angle_lo':-0.7854},
                    'right':  {'track':True,
                               'autozero':False,
                               'subtract_bg':True,        # Use background subtraction?
                               'hinge':{'x':300,
                                        'y':100},
                               'radius_outer':80,
                               'radius_inner':10,
                               'angle_hi':0.7854, 
                               'angle_lo':-0.7854},

                    }
        SetDict().set_dict_with_preserve(self.params, defaults)
        SetDict().set_dict_with_preserve(self.params, rospy.get_param('strokelitude'))
        self.params = self.legalizeParams(self.params)
        rospy.set_param('strokelitude', self.params)
        
        # initialize wings and body
        self.fly = Fly(self.params)
        
        # Background image.
        self.filenameBackground = os.path.expanduser(self.params['filenameBackground'])
        self.imgFullBackground  = cv2.imread(self.filenameBackground, cv2.CV_LOAD_IMAGE_GRAYSCALE)
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
        self.subImageRaw           = rospy.Subscriber(self.params['image_topic'], Image, self.image_callback, queue_size=2)
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
        btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='head', text='head', state=self.params['head']['track'])
        self.buttons.append(btn)

        x = btn.right+1
        btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='abdomen', text='abdomen', state=self.params['abdomen']['track'])
        self.buttons.append(btn)

        x = btn.right+1
        btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='wings', text='wings', state=self.params['right']['track'])
        self.buttons.append(btn)

        x = btn.right+1
        btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='subtract_bg', text='subtractBG', state=self.params['right']['subtract_bg'])
        self.buttons.append(btn)

        x = btn.right+1
        btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='symmetry', text='symmetric', state=self.params['symmetric'])
        self.buttons.append(btn)

        x = btn.right+1
        btn = Button(pt=[x,y], scale=self.scale, type='checkbox', name='extra_windows', text='windows', state=self.params['extra_windows'])
        self.buttons.append(btn)

        self.yToolbar = btn.bottom + 1


        # user callbacks
        cv2.setMouseCallback(self.window_name, self.onMouse, param=None)
        
        self.reconfigure = dynamic_reconfigure.server.Server(strokelitudeConfig, self.reconfigure_callback)
        
        
    # legalizeParams()
    # Make sure that all the parameters contain legal values.
    #
    def legalizeParams(self, params):
        paramsOut = copy.copy(params)
        for partname in ['head','abdomen','left','right']:
            if (partname in paramsOut):
                if ('hinge' in paramsOut[partname]):
                    if ('x' in paramsOut[partname]['hinge']):
                        paramsOut[partname]['hinge']['x'] = max(0, paramsOut[partname]['hinge']['x'])
        
                    if ('y' in paramsOut[partname]['hinge']):
                        paramsOut[partname]['hinge']['y'] = max(0, paramsOut[partname]['hinge']['y'])
    
                if ('radius_inner' in paramsOut[partname]):
                    paramsOut[partname]['radius_inner'] = max(5, paramsOut[partname]['radius_inner'])
    
                if ('radius_outer' in paramsOut[partname]) and ('radius_inner' in paramsOut[partname]):
                    paramsOut[partname]['radius_outer'] = max(paramsOut[partname]['radius_outer'], paramsOut[partname]['radius_inner']+5)

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
        with self.lock:
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

        
        
    def scale_params(self, paramsIn, scale):
        paramsScaled = copy.deepcopy(paramsIn)

        for partname in ['head', 'abdomen', 'left', 'right']:
            paramsScaled[partname]['hinge']['x'] = (paramsIn[partname]['hinge']['x']*scale)  
            paramsScaled[partname]['hinge']['y'] = (paramsIn[partname]['hinge']['y']*scale)  
            paramsScaled[partname]['radius_outer'] = (paramsIn[partname]['radius_outer']*scale)  
            paramsScaled[partname]['radius_inner'] = (paramsIn[partname]['radius_inner']*scale)  
			
        return paramsScaled  
	
	
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
            img = np.uint8(cv.GetMat(self.cvbridge.imgmsg_to_cv(rosimage, 'passthrough')))
            
        except CvBridgeError, e:
            rospy.logwarn ('Exception converting background image from ROS to opencv:  %s' % e)
            self.imgCamera = None
            
         
        if (self.scale == 1.0):              
        	self.imgCamera = img
        else:  
        	self.imgCamera = cv2.resize(img, (0,0), fx=self.scale, fy=self.scale) 
                
                
        if (self.imgCamera is not None):
            self.shapeImage = self.imgCamera.shape # (height,width)
            
            if (not self.bInitialized):
                self.fly.create_masks(self.shapeImage)
                self.bInitialized = True
                                
            if (self.params['use_gui']):
        
                imgOutput = cv2.cvtColor(self.imgCamera, cv2.COLOR_GRAY2RGB)
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
                    if (self.fly.left.state.angle1 is not None):
                        if (self.params['n_edges']==1):
                            s = 'L:% 7.4f' % (self.fly.left.state.angle1)
                            w = 65
                        else:
                            s = 'L:% 7.4f,% 7.4f' % (self.fly.left.state.angle1, self.fly.left.state.angle2)
                            w = 120
                        cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, self.fly.left.bgra)
                        w_text = int(w * self.scale)
                        x += w_text+self.w_gap
                    
                    if (self.fly.right.state.angle1 is not None):
                        if (self.params['n_edges']==1):
                            s = 'R:% 7.4f' % (self.fly.right.state.angle1)
                            w = 65
                        else:
                            s = 'R:% 7.4f,% 7.4f' % (self.fly.right.state.angle1, self.fly.right.state.angle2)
                            w = 120
                        cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, self.fly.right.bgra)
                        w_text = int(w * self.scale)
                        x += w_text+self.w_gap
                    
        
                    # Output sum of WBA
                    if (self.fly.left.state.angle1 is not None) and (self.fly.right.state.angle1 is not None):
                        leftplusright = self.fly.left.state.angle1 + self.fly.right.state.angle1
                        #s = 'L+R:% 7.1f' % -np.rad2deg(leftplusright)
                        s = 'L+R:% 7.4f' % -leftplusright
                        cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, bgra_dict['magenta'])
                        w_text = int(82 * self.scale)
                        x += w_text+self.w_gap
    
                        
                    # Output difference in WBA
                    if (self.fly.left.state.angle1 is not None) and (self.fly.right.state.angle1 is not None):
                        leftminusright = self.fly.left.state.angle1 - self.fly.right.state.angle1
                        #s = 'L-R:% 7.1f' % -np.rad2deg(leftminusright)
                        s = 'L-R:% 7.4f' % -leftminusright
                        cv2.putText(imgOutput, s, (x, y_bottom), self.fontface, self.scaleText, bgra_dict['magenta'])
                        w_text = int(82 * self.scale)
                        x += w_text+self.w_gap
    
                        
                    # Output flight status
                    if (self.fly.left.bFlying and self.fly.right.bFlying):
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
                self.fly.update(rosimage.header, self.imgCamera)
    
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
            tag  = [None,None,None,None]
            partname = [None,None,None,None]
            (partname[0], tag[0]) = self.fly.left.hit_object(ptMouse)
            (partname[1], tag[1]) = self.fly.right.hit_object(ptMouse)
            (partname[2], tag[2]) = self.fly.head.hit_object(ptMouse)
            (partname[3], tag[3]) = self.fly.abdomen.hit_object(ptMouse)
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

    
    def BodypartFromPartname(self, partname):
        if (partname=='left'):
            bodypart = self.fly.left
        elif (partname=='right'):
            bodypart = self.fly.right
        elif (partname=='head'):
            bodypart = self.fly.head
        elif (partname=='abdomen'):
            bodypart = self.fly.abdomen
        else:
            bodypart = None
            
        return bodypart
    
                
    # update_params_from_mouse()
    # Recalculate self.params based on a currently selected handle and mouse location.
    #
    def update_params_from_mouse(self, tagSelected, partnameSelected, ptMouse):             
        partnameSlave = 'right' if (self.partnameSelected=='left') else 'left'
        tagThis = tagSelected
        tagOther = 'angle_lo' if (tagSelected=='angle_hi') else 'angle_hi'
        tagSlave = tagOther
        bodypartSelected = self.BodypartFromPartname(partnameSelected)
        bodypartSlave    = self.BodypartFromPartname(partnameSlave)

        paramsScaled = self.scale_params(self.params, self.scale) 
        
        # Hinge.
        if (tagSelected=='hinge'): 
            if (partnameSelected=='head') or (partnameSelected=='abdomen'):

                # Get the hinge points pre-move.
                if (paramsScaled['symmetric']):
                    ptHead = np.array([paramsScaled['head']['hinge']['x'], paramsScaled['head']['hinge']['y']])
                    ptAbdomen = np.array([paramsScaled['abdomen']['hinge']['x'], paramsScaled['abdomen']['hinge']['y']])
                    ptCenterPre = (ptHead + ptAbdomen) / 2
                    ptBodyPre = ptHead - ptAbdomen
                    angleBodyPre = np.arctan2(ptBodyPre[1], ptBodyPre[0])
                    ptLeft = np.array([paramsScaled['left']['hinge']['x'], paramsScaled['left']['hinge']['y']])
                    ptRight = np.array([paramsScaled['right']['hinge']['x'], paramsScaled['right']['hinge']['y']])
                    ptLC = ptLeft-ptCenterPre
                    ptRC = ptRight-ptCenterPre
                    rL = np.linalg.norm(ptLC)
                    aL = np.arctan2(ptLC[1], ptLC[0]) - angleBodyPre # angle from body center to hinge in body axis coords.
                    rR = np.linalg.norm(ptRC)
                    aR = np.arctan2(ptRC[1], ptRC[0]) - angleBodyPre

                # Move the selected hinge point.
                pt = ptMouse
                paramsScaled[partnameSelected]['hinge']['x'] = float(pt[0])
                paramsScaled[partnameSelected]['hinge']['y'] = float(pt[1])
                
                # Now move the hinge points relative to the new body axis.
                if (paramsScaled['symmetric']):
                    ptHead = np.array([paramsScaled['head']['hinge']['x'], paramsScaled['head']['hinge']['y']])
                    ptAbdomen = np.array([paramsScaled['abdomen']['hinge']['x'], paramsScaled['abdomen']['hinge']['y']])
                    ptCenterPost = (ptHead + ptAbdomen) / 2
                    ptBodyPost = ptHead - ptAbdomen
                    angleBodyPost = np.arctan2(ptBodyPost[1], ptBodyPost[0])
                    ptLeft = ptCenterPost + rL * np.array([np.cos(aL+angleBodyPost), np.sin(aL+angleBodyPost)])
                    ptRight = ptCenterPost + rR * np.array([np.cos(aR+angleBodyPost), np.sin(aR+angleBodyPost)])
                    paramsScaled['left']['hinge']['x'] = float(ptLeft[0])
                    paramsScaled['left']['hinge']['y'] = float(ptLeft[1])
                    paramsScaled['right']['hinge']['x'] = float(ptRight[0])
                    paramsScaled['right']['hinge']['y'] = float(ptRight[1])

                    
            elif (partnameSelected=='left') or (partnameSelected=='right'):
                paramsScaled[partnameSelected]['hinge']['x'] = float(ptMouse[0])
                paramsScaled[partnameSelected]['hinge']['y'] = float(ptMouse[1])

                if (paramsScaled['symmetric']):
                    ptSlave = self.get_reflection_across_bodyaxis(ptMouse)
                    paramsScaled[partnameSlave]['hinge']['x'] = float(ptSlave[0])
                    paramsScaled[partnameSlave]['hinge']['y'] = float(ptSlave[1])



        # High angle.
        elif (tagSelected in ['angle_hi','angle_lo']): 
            pt = ptMouse - bodypartSelected.ptHinge_i
            if (tagSelected=='angle_lo'): 
                angle_lo_b = float(bodypartSelected.transform_angle_b_from_i(np.arctan2(pt[1], pt[0])))
                if (partnameSelected in ['head','abdomen']):
                    angle_hi_b = -angle_lo_b
                else:
                    angle_hi_b = paramsScaled[partnameSelected][tagOther]
    #            angle_hi_b = paramsScaled[partnameSelected][tagOther]
            elif (tagSelected=='angle_hi'):
                angle_hi_b = float(bodypartSelected.transform_angle_b_from_i(np.arctan2(pt[1], pt[0])))
                
                if (partnameSelected in ['head','abdomen']):
                    angle_lo_b = -angle_hi_b
                else:
                    angle_lo_b = paramsScaled[partnameSelected][tagOther]
    #            angle_lo_b = paramsScaled[partnameSelected][tagOther]
            
            paramsScaled[partnameSelected]['radius_outer'] = float(max(bodypartSelected.params[partnameSelected]['radius_inner']+2, 
                                                                      np.linalg.norm(bodypartSelected.ptHinge_i - ptMouse)))
                
            # Make angles relative to bodypart origin. 
            angle_lo_b -= (bodypartSelected.angleOutward_i - bodypartSelected.angleBody_i)
            angle_hi_b -= (bodypartSelected.angleOutward_i - bodypartSelected.angleBody_i)
            angle_lo_b = (angle_lo_b+np.pi) % (2.0*np.pi) - np.pi
            angle_hi_b = (angle_hi_b+np.pi) % (2.0*np.pi) - np.pi
            
            # Switch to the other handle.
            if (not (angle_lo_b < angle_hi_b)):
                self.tagSelected = tagOther
                
            # Set the order of the two angles
            paramsScaled[partnameSelected]['angle_lo'] = min(angle_lo_b, angle_hi_b)
            paramsScaled[partnameSelected]['angle_hi'] = max(angle_lo_b, angle_hi_b)
            
            # Make angles relative to fly origin. 
            paramsScaled[partnameSelected]['angle_lo'] += (bodypartSelected.angleOutward_i - bodypartSelected.angleBody_i)
            paramsScaled[partnameSelected]['angle_hi'] += (bodypartSelected.angleOutward_i - bodypartSelected.angleBody_i)
            paramsScaled[partnameSelected]['angle_lo'] = (paramsScaled[partnameSelected]['angle_lo']+np.pi) % (2.0*np.pi) - np.pi
            paramsScaled[partnameSelected]['angle_hi'] = (paramsScaled[partnameSelected]['angle_hi']+np.pi) % (2.0*np.pi) - np.pi
            
            
            if (paramsScaled[partnameSelected]['angle_hi'] < paramsScaled[partnameSelected]['angle_lo']):
                paramsScaled[partnameSelected]['angle_hi'] += 2*np.pi
            
            if (partnameSelected in ['left','right']):
                if (paramsScaled['symmetric']):
                    paramsScaled[partnameSlave][tagSlave]     = -paramsScaled[partnameSelected][tagSelected]
                    paramsScaled[partnameSlave][tagSelected]  = -paramsScaled[partnameSelected][tagSlave]
                    paramsScaled[partnameSlave]['radius_outer'] = paramsScaled[partnameSelected]['radius_outer']
                    
#                 if (paramsScaled[partnameSlave]['angle_hi'] < 0 < paramsScaled[partnameSlave]['angle_lo']):
#                     paramsScaled[partnameSlave]['angle_hi'] += 2*np.pi
 
              
        # Inner radius.
        elif (tagSelected=='radius_inner'): 
            paramsScaled[partnameSelected]['radius_inner'] = float(min(np.linalg.norm(bodypartSelected.ptHinge_i - ptMouse), 
                                                                                      bodypartSelected.params[partnameSelected]['radius_outer']-2))
            if (partnameSelected in ['left','right']) and (paramsScaled['symmetric']):
                paramsScaled[partnameSlave]['radius_inner'] = paramsScaled[partnameSelected]['radius_inner']
                
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
            self.nameSelected = self.name_from_tagpartname(tag,partname)
            self.tagSelected = tag
            self.partnameSelected = partname
            self.uiSelected = ui
            self.iButtonSelected = iButtonSelected


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
                    self.pubCommand.publish(MsgCommand('save_background',0))

                elif (self.nameSelected == self.nameSelectedNow == 'exit'):
                    self.pubCommand.publish(MsgCommand('exit',0))
                    
                    
            elif (self.uiSelected=='checkbox'):
                if (self.nameSelected == self.nameSelectedNow):
                    self.buttons[self.iButtonSelected].state = not self.stateSelected
                else:
                    self.buttons[self.iButtonSelected].state = self.stateSelected


                if (self.nameSelected == self.nameSelectedNow == 'symmetry'):
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
    
                # Save the results.
                SetDict().set_dict_with_preserve(self.params, rospy.get_param('strokelitude'))

                rospy.set_param('strokelitude', self.params)
                with self.lock:
                    rosparam.dump_params(self.parameterfile, 'strokelitude')


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
        rospy.spin()

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
