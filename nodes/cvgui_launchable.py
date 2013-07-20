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


def get_angle(origin, pt):
    x = pt[0] - origin[0]
    y = pt[1] - origin[1]
    return np.arctan2(y,x)*180/np.pi+180

# put angle between 0 and 360
def unwrap_angle(angle):
    if angle < 0:
        while angle < 0:
            angle = angle + 360
        return angle
    elif angle >= 360:
        while angle >= 360:
            angle = angle - 360
        return angle
    else:
        return angle

# remap +/- 180 to 0, and +/- 0 to +/- 180
def flip_angle(angle):
    if angle > 0:
        return angle -180
    else:
        return angle +180

##########################################################################
class Wing(object):
    def __init__(self, side='right', config=None):
        self.side = side
        if side == 'right':
            self.color ='green'
            self.sign  = 1
        elif side == 'left':
            self.color ='red'
            self.sign  = -1
        self.nbins              = 50
        self.bin_resolution     = 360/float(self.nbins)
        self.bins               = np.linspace(-180, 180, self.nbins)
        self.mask               = None
        self.histogram          = None
        self.angle_per_pixel    = None
        
        self.extract_parameters(config)
        
        self.trailing_edge_angle= None
        self.leading_edge_angle = None
        self.amp                = None
        self.msa                = None
        self.flying             = False
        
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
        srv_name                = 'wing_histogram_' + side
        self.service_histogram  = rospy.Service(srv_name, float32list, self.serve_histogram)
        
        srv_name                = 'wing_bins_' + side
        self.service_bins       = rospy.Service(srv_name, float32list, self.serve_bins)
        
        srv_name                = 'wing_edges_' + side
        self.service_edges       = rospy.Service(srv_name, float32list, self.serve_edges)
        
        # publishers
        pub_name                = 'strokelitude/wba/' + self.side + '/sum'
        self.wing_sum           = rospy.Publisher(pub_name, Float32)
        
        pub_name                = 'strokelitude/wba/' + self.side + '/contrast'
        self.wing_contrast      = rospy.Publisher(pub_name, Float32)
        
        pub_name                = 'strokelitude/wba/' + self.side + '/amplitude'
        self.wing_amp           = rospy.Publisher(pub_name, Float32)
        
        pub_name                = 'strokelitude/wba/' + self.side + '/mean_stroke_angle'
        self.wing_msa           = rospy.Publisher(pub_name, Float32)
        
        pub_name                = 'strokelitude/wba/' + self.side + '/leading_edge'
        self.wing_leading       = rospy.Publisher(pub_name, Float32)
        
        pub_name                = 'strokelitude/wba/' + self.side + '/trailing_edge'
        self.wing_trailing      = rospy.Publisher(pub_name, Float32)
        
    # extract parameters from parameter server, given "strokelitude" parameter
    def extract_parameters(self, config):
        if config is not None:
            self.origin         = (config['wing'][self.side]['origin']['x'], config['wing'][self.side]['origin']['y'])
            self.angle1         = config['wing']['angle1']
            self.angle2         = config['wing']['angle2']
            self.radius_outer   = config['wing']['radius_outer']
            self.radius_inner   = config['wing']['radius_inner']
            self.bodyangle      = config['bodyangle']
            self.threshold      = config['wing'][self.side]['threshold']
            self.flight_threshold=config['wing'][self.side]['flight_threshold']
            
            if config['wing']['nbins'] != self.nbins:
                self.nbins          = config['wing']['nbins']
                self.bin_resolution = 360/float(self.nbins)
                self.bins           = np.linspace(-180, 180, self.nbins)
                if self.angle_per_pixel is not None:
                    self.calculate_bin_indices()
                
    def draw(self, cvimage ):
    
        if self.side == 'right':
            angle1 = unwrap_angle(self.angle1 + self.bodyangle-180) 
            angle2 = unwrap_angle(self.angle2 + self.bodyangle-180) 
            
        if self.side == 'left':
            angle1 = unwrap_angle(self.angle1*self.sign + self.bodyangle)
            angle2 = unwrap_angle(self.angle2*self.sign + self.bodyangle)
            
        if np.abs(angle1-angle2) > 180:
            smaller_angle = np.argmin([angle1, angle2])
            if smaller_angle == 0:
                angle1 += 360
            else:
                angle2 += 360
                
        # inner circle
        cv.Ellipse(     cvimage, 
                        self.origin,  
                        ( self.radius_inner, self.radius_inner ),
                        180,
                        angle1,
                        angle2,
                        color=self.numeric_color,
                        thickness=self.thickness_inner,
                        )
        
        # outer circle         
        cv.Ellipse(     cvimage, 
                        self.origin,  
                        ( self.radius_outer, self.radius_outer ),
                        180,
                        angle1,
                        angle2,
                        color=self.numeric_color,
                        thickness=self.thickness_outer,
                        )
                        
        # wing leading and trailing edges
        if self.trailing_edge_angle is not None:
            angle = self.trailing_edge_angle*self.sign+self.bodyangle
            if self.side == 'left':
                angle = flip_angle(angle)
            p1 = int(self.origin[0] + np.cos( (angle)*np.pi/180. )*self.radius_outer)
            p2 = int(self.origin[1] + np.sin( (angle)*np.pi/180. )*self.radius_outer)
            cv.Line(cvimage, self.origin, (p1,p2), self.numeric_color, 2)
        if self.trailing_edge_angle is not None:
            angle = self.leading_edge_angle*self.sign+self.bodyangle
            if self.side == 'left':
                angle = flip_angle(angle)
            p1 = int(self.origin[0] + np.cos( (angle)*np.pi/180. )*self.radius_outer)
            p2 = int(self.origin[1] + np.sin( (angle)*np.pi/180. )*self.radius_outer)
            cv.Line(cvimage, self.origin, (p1,p2), self.numeric_color, 2)
                        
    def calc_mask(self, npimage ):
        
        self.radius_per_pixel = np.zeros([npimage.shape[0], npimage.shape[1]])
        self.angle_per_pixel = np.zeros([npimage.shape[0], npimage.shape[1]])
        for i in range(npimage.shape[0]):
            for j in range(npimage.shape[1]):
                radius = np.sqrt( (i-self.origin[1])**2 + (j-self.origin[0])**2 )
                self.radius_per_pixel[i,j] = radius
                
                angle = get_angle(self.origin, (j,i))
                angle_rel_to_body = unwrap_angle(angle-self.bodyangle)-180
                if self.side == 'left':
                    angle_rel_to_body = flip_angle(angle_rel_to_body)
                self.angle_per_pixel[i,j] = angle_rel_to_body*self.sign
                    
        self.mask = (self.radius_per_pixel<self.radius_outer)*( self.radius_per_pixel > self.radius_inner )
        
        # ravel arrays
        self.angle_per_pixel_array = copy.copy(self.angle_per_pixel) # for debugging
        self.radius_per_pixel = np.ravel(self.radius_per_pixel)
        self.angle_per_pixel = np.ravel(self.angle_per_pixel)
        self.mask = np.ravel(self.mask)
        
        self.calculate_bin_indices()
        
    def calculate_bin_indices(self):
        print 'Calculating wing mask: ', self.side
        print 'This could take a minute...'
        print
        self.bin_indices = {}
        self.histogram = {}
        for b in self.bins:
            self.bin_indices.setdefault(b, [])
            self.histogram.setdefault(b, None)
        for k, angle in enumerate(self.angle_per_pixel):
            if self.mask[k]:
                bind = np.argmin(np.abs(self.bins-angle))
                best_bin = self.bins[bind]
                self.bin_indices[best_bin].append(k)
    '''   
    def publish_wing_sum(self):
        if self.histogram is not None:
            s = np.sum(self.histogram_values[self.angle_indices_ok])
            if s > self.flight_threshold:
                self.flying = True
            else:
                self.flying = False
            self.wing_sum.publish(s)
    '''
         
    def publish_wing_contrast(self):
        if self.histogram is not None:
            max_contrast = np.abs( np.max(self.histogram_values) - np.min(self.histogram_values) ) 
            if max_contrast > self.flight_threshold:
                self.flying = True
            else:
                self.flying = False
            self.wing_contrast.publish(max_contrast)
                
    def publish_wing_edges(self):
        if self.histogram is not None:
            if self.flying:
                bins    = self.histogram_bins[self.angle_indices_ok]
                values  = self.histogram_values[self.angle_indices_ok]
                
                # auto rescale
                values -= np.min(values)
                values /= np.max(values)
                
                wing_indices = np.where(values>self.threshold)[0]
                
                leading_edge_index = wing_indices[0]
                trailing_edge_index = wing_indices[-1]
        
                leading_edge_angle = int(bins[leading_edge_index])
                trailing_edge_angle = int(bins[trailing_edge_index])
        
                amp = np.abs(leading_edge_angle - trailing_edge_angle)
                msa = np.mean([leading_edge_angle, trailing_edge_angle])
            
            else: # not flying
                leading_edge_angle = -180
                trailing_edge_angle = -180
                amp = 0
                msa = 0
                
            self.wing_msa.publish(msa)
            self.wing_amp.publish(amp)
            self.wing_leading.publish(leading_edge_angle)
            self.wing_trailing.publish(trailing_edge_angle)
            
            self.amp                 = amp
            self.msa                 = msa
            self.leading_edge_angle  = leading_edge_angle
            self.trailing_edge_angle = trailing_edge_angle
                    
    def calc_histogram(self, npimage):
        if self.mask is not None:
            npimage_raveled = np.ravel(npimage)/255.*self.mask
            for b, indices in self.bin_indices.items():
                if len(indices) > 0:
                    self.histogram[b] = np.sum(npimage_raveled[indices]) / float(len(indices))
                else:
                    self.histogram[b] = 0
            
            # rearranging histogram
            bins                    = np.array(self.histogram.keys())
            sorting_indices         = np.argsort(bins)
            self.histogram_bins     = bins[sorting_indices]
            self.histogram_values   = np.array(self.histogram.values())[sorting_indices]
            
            lo_angle = np.min([self.angle1, self.angle2])
            hi_angle = np.max([self.angle1, self.angle2])
            
            self.angle_indices_ok   = np.where( (self.histogram_bins<hi_angle)*(self.histogram_bins>lo_angle) )[0]
            
            # wing beat analyzer
            self.publish_wing_contrast()
            self.publish_wing_edges()
                        
    def serve_histogram(self, request):
        if self.histogram is not None:
            return float32listResponse(self.histogram.values())
            
    def serve_bins(self, request):
        if self.histogram is not None:
            return float32listResponse(self.histogram.keys())
            
    def serve_edges(self, request):
        if self.trailing_edge_angle is not None:
            return float32listResponse([self.trailing_edge_angle, self.leading_edge_angle])
            
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
        self.drawing_body_is_active = False
        self.drawing_wings_is_active= False
        self.angletext              = 'none' # for debugging

        # initialize
        node_name                   = 'cvgui'
        rospy.init_node(node_name, anonymous=True)

        # publishers
        pub_name                    = 'strokelitude/wba/' + 'LeftMinusRight'
        self.wing_leftminusright    = rospy.Publisher(pub_name, Float32)
        
        pub_name                    = 'strokelitude/wba/' + 'LeftPlusRight'
        self.wing_leftplusright    = rospy.Publisher(pub_name, Float32)
        
        pub_name                    = 'strokelitude/wba/' + 'flight_status'
        self.flight_status          = rospy.Publisher(pub_name, Float32)
        
        # subscribe to images        
        self.image_sub              = rospy.Subscriber('/camera/image_raw',Image,self.image_callback)

        # user callbacks
        cv.SetMouseCallback(self.display_name, self.mouse, param=None)
        
    def get_configuration(self, config=None):
        if config is None:
            config = rospy.get_param('strokelitude')
        self.bodyangle = config['bodyangle']
        self.wing_r.extract_parameters(config)
        self.wing_l.extract_parameters(config)
        
    def image_callback(self,image):
        # Receive an image:
        try:
            cv_image = np.uint8(cv.GetMat(self.cvbridge.imgmsg_to_cv(image, 'passthrough')))
        except CvBridgeError, e:
            rospy.logwarn ('Exception converting background image from ROS to opencv:  %s' % e)
            cv_image = None
            
        # get correct image types
        self.npimage = cv_image
        self.npimagecolor = cv2.cvtColor(cv_image, cv.CV_GRAY2RGB)
        cvimage = cv.fromarray(self.npimagecolor)
            
        left_pixel   = 10
        bottom_pixel = self.npimagecolor.shape[0]-10 
        right_pixel  = self.npimagecolor.shape[1]-10
            
        # draw body
        if self.wing_r.origin is not None and self.wing_l.origin is not None:
            cv.Line(cvimage, self.wing_r.origin, self.wing_l.origin, cv.Scalar(255,0,0,0), 2)
            #body_center_x = int( (self.wing_r.origin[0] + self.wing_l.origin[0])/2. ) 
            #body_center_y = int( (self.wing_r.origin[1] + self.wing_l.origin[1])/2. ) 
        
        # draw wings
        if self.wing_r.origin is not None:
            self.wing_r.draw(cvimage)
        if self.wing_l.origin is not None:
            self.wing_l.draw(cvimage)
        
        if not self.drawing_wings_is_active:
            # calculate wing beat analyzer stats
            self.wing_r.calc_histogram(self.npimage)
            self.wing_l.calc_histogram(self.npimage)
            
            # publish difference in WBA
            if self.wing_l.amp is not None and self.wing_r.amp is not None:
                LeftMinusRight = self.wing_l.amp - self.wing_r.amp
                self.wing_leftminusright.publish(LeftMinusRight)
                s = 'L-R WBA: ' + str(LeftMinusRight)
                cv2.putText(self.npimagecolor, s, (right_pixel-100, bottom_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cv.Scalar(255,50,50,0) )
                
            # publish sum of WBA
            if self.wing_l.amp is not None and self.wing_r.amp is not None:
                LeftPlusRight = self.wing_l.amp + self.wing_r.amp
                self.wing_leftplusright.publish(LeftPlusRight)
                s = 'L+R WBA: ' + str(LeftPlusRight)
                cv2.putText(self.npimagecolor, s, (right_pixel-200, bottom_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cv.Scalar(255,50,50,0) )
                
            # publish flight status
            if self.wing_l.flying and self.wing_r.flying:
                s = 'FLIGHT'
                self.flight_status.publish(1)
            else:
                s = 'no flight'
                self.flight_status.publish(0)
            cv2.putText(self.npimagecolor, s, (right_pixel-300, bottom_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cv.Scalar(255,50,255,0) )
            
        if self.wing_r.amp is not None:
            s = 'WBA R: ' + str(self.wing_r.amp)
            cv2.putText(self.npimagecolor, s, (left_pixel,bottom_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.wing_r.numeric_color)
        if self.wing_l.amp is not None:
            s = 'WBA L: ' + str(self.wing_l.amp)
            cv2.putText(self.npimagecolor, s, (left_pixel+100, bottom_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.wing_l.numeric_color)
        
        
        # display image
        cv2.imshow("Display", self.npimagecolor)
        cv2.waitKey(3)


    def mouse(self, event, x, y, flags, param):
        pt = (x,y)
        
        # FETCH PARAMETERS : left button double click
        if event == cv.CV_EVENT_LBUTTONDBLCLK:
            print 'Fetching parameters from parameter server'
            print 'This may take a few minutes'
            print
            self.get_configuration()
            self.wing_r.calc_mask(self.npimage)
            self.wing_l.calc_mask(self.npimage)
            
        # BODY : MBUTTON
        if event == cv.CV_EVENT_MBUTTONDOWN and not self.drawing_body_is_active:
            self.drawing_body_is_active = True
            if not flags==cv.CV_EVENT_FLAG_SHIFTKEY:
                self.wing_r.origin      = pt
            else:
                self.wing_l.origin      = pt
        elif event == cv.CV_EVENT_MBUTTONDOWN and self.drawing_body_is_active:
            self.drawing_body_is_active = False
            if not flags==cv.CV_EVENT_FLAG_SHIFTKEY:
                self.wing_l.origin      = pt
            else:
                self.wing_r.origin      = pt
            self.bodyangle              = int(get_angle(self.wing_r.origin, self.wing_l.origin))
            self.wing_r.bodyangle       = self.bodyangle
            self.wing_l.bodyangle       = self.bodyangle
            
            rospy.set_param('strokelitude/wing/right/origin/x', self.wing_r.origin[0])
            rospy.set_param('strokelitude/wing/right/origin/y', self.wing_r.origin[1])
            rospy.set_param('strokelitude/wing/left/origin/x', self.wing_l.origin[0])
            rospy.set_param('strokelitude/wing/left/origin/y', self.wing_l.origin[1])
            rospy.set_param('strokelitude/bodyangle', self.bodyangle)
            
        if self.drawing_body_is_active:
            if flags==cv.CV_EVENT_FLAG_SHIFTKEY or flags==cv.CV_EVENT_FLAG_SHIFTKEY+cv.CV_EVENT_FLAG_CTRLKEY:
                self.wing_r.origin      = pt
            else:
                self.wing_l.origin      = pt
            self.bodyangle              = int(get_angle(self.wing_r.origin, self.wing_l.origin))
            self.wing_r.bodyangle       = self.bodyangle
            self.wing_l.bodyangle       = self.bodyangle
            
        # WINGS : RBUTTON         
        if self.wing_r.origin is not None and self.wing_l.origin is not None:
            if flags==cv.CV_EVENT_FLAG_SHIFTKEY or flags==cv.CV_EVENT_FLAG_SHIFTKEY+cv.CV_EVENT_FLAG_CTRLKEY:
                wing = self.wing_l
            else:
                wing = self.wing_r
            radius                      = int(np.sqrt( (pt[1] - wing.origin[1])**2 + (pt[0] - wing.origin[0])**2 ))
            angle                       = get_angle(wing.origin, pt)-180
            angle_rel_to_body           = unwrap_angle(int(angle - self.bodyangle))-180
            if flags==cv.CV_EVENT_FLAG_SHIFTKEY or flags==cv.CV_EVENT_FLAG_SHIFTKEY+cv.CV_EVENT_FLAG_CTRLKEY:
                angle_rel_to_body           = angle_rel_to_body*-1
            else:
                angle_rel_to_body           = flip_angle(angle_rel_to_body)
                
            if self.wing_l.angle_per_pixel is not None:
                self.angletext          = str(wing.angle_per_pixel_array[y,x])
            
        if event == cv.CV_EVENT_RBUTTONDOWN and not self.drawing_wings_is_active:
            self.drawing_wings_is_active= True
            self.wing_r.angle1          = angle_rel_to_body
            self.wing_l.angle1          = angle_rel_to_body
            
            if flags==cv.CV_EVENT_FLAG_CTRLKEY or flags==cv.CV_EVENT_FLAG_SHIFTKEY+cv.CV_EVENT_FLAG_CTRLKEY:
                self.wing_r.radius_inner    = radius
                self.wing_l.radius_inner    = radius
            else:          
                self.wing_l.radius_outer    = radius
                self.wing_r.radius_outer    = radius
            
            
        elif event == cv.CV_EVENT_RBUTTONDOWN and self.drawing_wings_is_active:
            self.drawing_wings_is_active= False
            self.wing_r.angle2          = angle_rel_to_body
            self.wing_l.angle2          = angle_rel_to_body
            
            rospy.set_param('strokelitude/wing/angle1', self.wing_r.angle1)
            rospy.set_param('strokelitude/wing/angle2', self.wing_r.angle2)
            rospy.set_param('strokelitude/wing/radius_outer', self.wing_r.radius_outer)
            rospy.set_param('strokelitude/wing/radius_inner', self.wing_r.radius_inner)
            
        if self.drawing_wings_is_active:
            self.wing_r.angle2          = angle_rel_to_body
            self.wing_l.angle2          = angle_rel_to_body
            

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print "Shutting down"
        cv2.DestroyAllWindows()

if __name__ == '__main__':

    #parser = OptionParser()
    #parser.add_option("--color", action="store_true", dest="color", default=True)
    #(options, args) = parser.parse_args()


    print
    print
    print "**************************************************************************"
    print "*                                                                        *"  
    print "*    StrokelitudeROS: Camera based wingbeat analyzer software for ROS    *"
    print "*        by Floris van Breugel (c) 2013                                  *"
    print "*                                                                        *"
    print "*    Controls:                                                           *"
    print "*                                                                        *"  
    print "*    Middle Mouse (single click): select wing hinge                      *"
    print "*        Default   : Right wing (green)                                  *"
    print "*        Shift key : Left wing (red)                                     *"
    print "*    Right Mouse (single click) : draw radius mask                       *"
    print "*        Default   : Right wing (green), Outer radius (thick)            *"
    print "*        Shift key : Left wing (red)                                     *"
    print "*        Ctrl  key : Inner radius (thin)                                 *"
    print "*    Left Mouse (double click)  : apply parameters, calculate WBA mask   *"
    print "*                                                                        *" 
    print "**************************************************************************"
    print
    print

    im = ImageDisplay()
    im.run()
