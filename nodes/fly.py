#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from std_msgs.msg import Float32, Header, String
from imagewindow import ImageWindow
from tracker_axis import AxisTracker
from tracker_area import AreaTracker
from tracker_edge import EdgeTrackerByIntensityProfile, EdgeTrackerByHoughTransform
from tracker_tip import TipTracker
from tracker_intensity import IntensityTracker
import ui
from Kinefly.msg import MsgFlystate, MsgState
import imageprocessing


###############################################################################
###############################################################################
class Fly(object):
    def __init__(self, params={}):
        self.nodename = rospy.get_name().rstrip('/')
        EdgeTracker = EdgeTrackerByIntensityProfile
        #EdgeTracker = EdgeTrackerByHoughTransform


        # Create the body axis tracker.        
        self.axis        = AxisTracker(name='axis',      params=params, color='yellow')

        # Create the head tracker.        
        if (params['head']['tracker']=='area'):
            self.head    = AreaTracker(name='head',      params=params, color='cyan',    bEqualizeHist=False)
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
            self.abdomen    = AreaTracker(name='abdomen',      params=params, color='magenta',    bEqualizeHist=False)
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
            self.right    = AreaTracker(name='right',      params=params, color='red',    bEqualizeHist=False)
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
            self.left    = AreaTracker(name='left',      params=params, color='green',    bEqualizeHist=False)
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
        
        self.bgra_body = ui.bgra_dict['light_gray']
        self.ptBodyIndicator1 = None
        self.ptBodyIndicator2 = None
        self.bInvertColor = False
        self.bInvertColorAuto = True
        self.iCount  = 0
        self.stampPrev = None
        self.stampPrevAlt = None
        self.stamp = rospy.Time(0)
        

        self.pubFlystate = rospy.Publisher(self.nodename+'/flystate', MsgFlystate, queue_size=100)
 

    def set_params(self, params):
        self.params = params
        
        self.axis.set_params(params)
        self.head.set_params(params)
        self.abdomen.set_params(params)
        self.left.set_params(params)
        self.right.set_params(params)
        self.aux.set_params(params)

        pt1 = [params['gui']['head']['hinge']['x'], params['gui']['head']['hinge']['y']]
        pt2 = [params['gui']['abdomen']['hinge']['x'], params['gui']['abdomen']['hinge']['y']]
        pt3 = [params['gui']['left']['hinge']['x'], params['gui']['left']['hinge']['y']]
        pt4 = [params['gui']['right']['hinge']['x'], params['gui']['right']['hinge']['y']]
        self.ptBodyCenter_i = imageprocessing.get_intersection(pt1,pt2,pt3,pt4)

        r = max(params['gui']['left']['radius_outer'], params['gui']['right']['radius_outer'])
        self.angleBody_i = self.get_bodyangle_i()
        self.ptBodyIndicator1 = tuple((self.ptBodyCenter_i + r * np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)])).astype(int))
        self.ptBodyIndicator2 = tuple((self.ptBodyCenter_i - r * np.array([np.cos(self.angleBody_i), np.sin(self.angleBody_i)])).astype(int))
        
        # Radius of an area approximately where the thorax would be.
        self.rInvertColorArea = np.linalg.norm(np.array([params['gui']['head']['hinge']['x'], params['gui']['head']['hinge']['y']]) - np.array([params['gui']['abdomen']['hinge']['x'], params['gui']['abdomen']['hinge']['y']]))/2.0
        self.bInvertColorValid = False
        
    
    def create_masks(self, shapeImage):
        if (self.params['gui']['axis']['track']):
            if (not self.axis.bValidMask):
                self.axis.create_mask (shapeImage)
                self.axis.bValidMask = True
                
        if (self.params['gui']['head']['track']):
            if (not self.head.bValidMask):
                self.head.create_mask (shapeImage)
                self.head.bValidMask = True
                
        if (self.params['gui']['abdomen']['track']):
            if (not self.abdomen.bValidMask):
                self.abdomen.create_mask (shapeImage)
                self.abdomen.bValidMask = True
            
        if (self.params['gui']['right']['track']):
            if (not self.right.bValidMask):
                self.right.create_mask (shapeImage)
                self.right.bValidMask = True
            
        if (self.params['gui']['left']['track']):
            if (not self.left.bValidMask):
                self.left.create_mask (shapeImage)
                self.left.bValidMask = True
            
        if (self.params['gui']['aux']['track']):
            if (not self.aux.bValidMask):
                self.aux.create_mask (shapeImage)
                self.aux.bValidMask = True


    def get_bodyangle_i(self):
        angle_i = imageprocessing.get_angle_from_points_i(self.abdomen.ptHinge_i, self.head.ptHinge_i)
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
        self.axis.set_background(image)

    
    def update_handle_points(self):
        self.head.update_handle_points()
        self.abdomen.update_handle_points()
        self.left.update_handle_points()
        self.right.update_handle_points()
        self.aux.update_handle_points()
        self.axis.update_handle_points()
        

    def update(self, header=None, image=None):
        if (image is not None):
            self.header = header
            
            if (not self.bInvertColorValid) and (self.bInvertColorAuto):
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
            self.axis.update(dt, image, self.bInvertColor)
            
            
    def draw(self, image):
        # Draw line to indicate the body axis.
        cv2.line(image, self.ptBodyIndicator1, self.ptBodyIndicator2, self.bgra_body, 1) # Draw a line longer than just head-to-abdomen.
                
        self.axis.draw(image)
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

        

