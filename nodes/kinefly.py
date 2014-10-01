#!/usr/bin/env python
#! coding=latin-1
from __future__ import division
import roslib; roslib.load_manifest('Kinefly')
import rospy
import rosparam

import copy
import cProfile
import cv
import cv2
import numpy as np
import os
import threading
import dynamic_reconfigure.server

from setdict import SetDict
import ui
from fly import Fly

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Header, String

from Kinefly.srv import SrvTrackerdata, SrvTrackerdataResponse
from Kinefly.msg import MsgFlystate, MsgState
from Kinefly.cfg import kineflyConfig
import imageprocessing

# gImageTime = 0.0
gbShowMasks = False

class Struct:
    pass

        

###############################################################################
###############################################################################
class MainWindow:
    def __init__(self):
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
                    'n_queue_images':2,
                    'use_gui':True,                     # You can turn off the GUI to speed the framerate.
                    'scale_image':1.0,                  # Reducing the image scale will speed the framerate.
                    'n_edges_max':1,                    # Max number of edges per wing to detect, subject to threshold.
                    'rc_background':1000.0,             # Time constant of the moving average background.
                    'wingbeat_min':180,                 # Bounds for wingbeat frequency measurement.
                    'wingbeat_max':220,
                    'head':   {'tracker':'area',
                               'autozero':True,         # Automatically figure out where is the center of motion.
                               'threshold':0.0,
                               'saturation_correction':False},        
                    'abdomen':{'tracker':'area',
                               'autozero':True,
                               'threshold':0.0,
                               'saturation_correction':False},
                    'left':   {'tracker':'edge',
                               'autozero':True,
                               'threshold':0.0,
                               'saturation_correction':False},
                    'right':  {'tracker':'edge',
                               'autozero':True,
                               'threshold':0.0,
                               'saturation_correction':False},
                    'aux':    {'tracker':'intensity',
                               'saturation_correction':False},
                    'gui': {'windows':True,                     # Show the helpful extra windows.
                            'symmetric':True,                   # Forces the UI to remain symmetric.
                            'axis':   {'track':False,            # To track, or not to track.
                                       'pt1':{'x':310,
                                              'y':150},
                                       'pt2':{'x':310,
                                              'y':250}},
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
        self.stampCameraPrev= rospy.Time(0)
        self.stampCameraDiff= rospy.Duration(0)
        self.stampROSPrev   = rospy.Time(0)
        self.stampROSDiff   = rospy.Duration(0)
        self.stampMax       = rospy.Duration(0)
        self.dtCamera       = np.inf
        self.hzCameraF      = 0.0
        self.hzCameraSum    = 0.0
        self.hzROSF         = 0.0
        self.hzROSSum       = 0.0
        self.iCountCamera   = 0
        self.iCountROS      = 0
        self.iDroppedFrame  = 0
        
        self.nQueuePrev     = 0     # Length of the image queue.
        self.dnQueueF       = 0.0   # Rate of change of the image queue length.
        
        self.bufferImages   = [None]*self.params['n_queue_images'] # Circular buffer for incoming images.
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
        self.shapeToolbar   = (0,0)
        
        # Publishers.
        self.pubCommand     = rospy.Publisher(self.nodename+'/command', String)
        self.pubImage       = rospy.Publisher(self.nodename+'/image_output', Image)

        # Subscriptions.        
        sizeImage = 128+1024*1024 # Size of header + data.
        self.subImage       = rospy.Subscriber(self.params['image_topic'], Image,      self.image_callback,   queue_size=2, buff_size=2*sizeImage, tcp_nodelay=True)
        self.subCommand     = rospy.Subscriber(self.nodename+'/command',   String, self.command_callback, queue_size=1000)

        # user callbacks
        cv2.setMouseCallback(self.window_name, self.onMouse, param=None)
        
        self.reconfigure = dynamic_reconfigure.server.Server(kineflyConfig, self.reconfigure_callback)
        
        # Preload a "Waiting..." image.
        h=int(480)
        w=int(640)
        imgInitial = np.zeros((h,w), dtype=np.uint8)
        for i in range(20):
            color = cv.Scalar(int(255*np.random.random()), int(255*np.random.random()), int(255*np.random.random()), 0)
            cv2.putText(imgInitial, 'Waiting for Camera...', (int(w*0.8*np.random.random()),int(h*np.random.random())), self.fontface, 2*self.scaleText, color)
        rosimg = self.cvbridge.cv_to_imgmsg(cv.fromarray(imgInitial), 'passthrough')
        self.image_callback(rosimg)
        self.bValidImage = False
        

    # Check the given button to see if it extends outside the image, and if so then reposition it to the next line.
    def wrap_button(self, btn, shape):
        if (btn.right >= shape[1]):
            btn.set_pos(pt=[1, btn.bottom+1])

        
    # Create the button bar, with overflow onto more than one line if needed to fit on the image.        
    def create_buttons(self, shape):
        self.shapeToolbar = shape
        
        # UI button specs.
        self.buttons = []
        x = 1
        y = 1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='pushbutton', name='exit', text='exit')
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='pushbutton', name='save_bg', text='saveBG')
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='static', name='track', text='track:', sides=ui.SIDE_LEFT|ui.SIDE_TOP|ui.SIDE_BOTTOM)
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='checkbox', name='head', text='H', state=self.params['gui']['head']['track'], sides=ui.SIDE_TOP|ui.SIDE_BOTTOM)
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='checkbox', name='abdomen', text='A', state=self.params['gui']['abdomen']['track'], sides=ui.SIDE_TOP|ui.SIDE_BOTTOM)
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='checkbox', name='left', text='L', state=self.params['gui']['left']['track'], sides=ui.SIDE_TOP|ui.SIDE_BOTTOM)
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='checkbox', name='right', text='R', state=self.params['gui']['right']['track'], sides=ui.SIDE_TOP|ui.SIDE_BOTTOM)
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='checkbox', name='aux', text='X', state=self.params['gui']['aux']['track'], sides=ui.SIDE_TOP|ui.SIDE_BOTTOM|ui.SIDE_RIGHT)
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='static', name='subtract', text='subtract:', sides=ui.SIDE_LEFT|ui.SIDE_TOP|ui.SIDE_BOTTOM)
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='checkbox', name='subtract_head', text='H', state=self.params['gui']['head']['subtract_bg'], sides=ui.SIDE_TOP|ui.SIDE_BOTTOM)
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='checkbox', name='subtract_abdomen', text='A', state=self.params['gui']['abdomen']['subtract_bg'], sides=ui.SIDE_TOP|ui.SIDE_BOTTOM)
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='checkbox', name='subtract_lr', text='LR', state=self.params['gui']['right']['subtract_bg'], sides=ui.SIDE_TOP|ui.SIDE_BOTTOM)
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='checkbox', name='subtract_aux', text='X', state=self.params['gui']['aux']['subtract_bg'], sides=ui.SIDE_TOP|ui.SIDE_BOTTOM|ui.SIDE_RIGHT)
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        # A button to allow the user to override the automatic invertcolor detector.  A better autodetect algorithm might eliminate the need for this.
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='checkbox', name='invertcolor', text='invertcolor', state=self.fly.bInvertColor)
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        self.ibtnInvertColor = len(self.buttons)-1
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='checkbox', name='stabilize', text='stabilize', state=self.params['gui']['head']['stabilize'])
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='checkbox', name='symmetry', text='symmetric', state=self.params['gui']['symmetric'])
        self.wrap_button(btn, shape)
        self.buttons.append(btn)
        
        x = btn.right+1
        y = btn.top+1
        btn = ui.Button(pt=[x,y], scale=self.scale, type='checkbox', name='windows', text='windows', state=self.params['gui']['windows'])
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
        self.command = msg.data
        
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
            rospy.logwarn('rostopic pub -1 %s/command std_msgs/String commandtext' % self.nodename)
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
                # Check for dropped frame.
                if (self.bufferImages[self.iImgLoading] is None):   # There's an empty slot in the buffer.
                    iImgLoadingNext = (self.iImgLoading+1) % len(self.bufferImages)
                    iImgWorkingNext = self.iImgWorking
                    self.iDroppedFrame = 0
                else:                                               # The buffer is full; we'll overwrite the oldest entry.
                    iImgLoadingNext = (self.iImgLoading+1) % len(self.bufferImages)
                    iImgWorkingNext = (self.iImgWorking+1) % len(self.bufferImages)
                    self.iDroppedFrame += 1
    
                # Put the image into the queue.
                self.bufferImages[self.iImgLoading] = rosimg
                self.iImgLoading = iImgLoadingNext
                self.iImgWorking = iImgWorkingNext
            
#            # Warn if the camera is delaying.
#            if (self.stampROSimagePrev is not None):
#                if (self.stampCamera.to_sec() - self.stampROSimagePrev.to_sec() > 1):
#                    rospy.logwarn('%s: %d, %d %0.6f' % (self.nodename, rosimg.header.seq, (self.stampCamera.secs*1e9)+self.stampCamera.nsecs, self.stampCamera.to_sec() - self.stampROSimagePrev.to_sec()))
#
#            self.stampROSimagePrev = self.stampCamera

            self.bValidImage = True

            

                
    def process_image_fake(self):
        with self.lockBuffer:
            # Mark this buffer entry as available for loading.
            self.bufferImages[self.iImgWorking] = None

            # Go to the next image.
            self.iImgWorking = (self.iImgWorking+1) % len(self.bufferImages)

                
                
    def process_image(self):
        rosimg = None
        
        with self.lockBuffer:
            # The image queue length.
            nQueue = (self.iImgLoading - self.iImgWorking) %  len(self.bufferImages)
            if (nQueue==0) and (self.bufferImages[self.iImgLoading] is not None):
                nQueue += len(self.bufferImages)
                        
            # Rate of change of the queue length.
            if (nQueue == len(self.bufferImages)):
                dnQueue = 1.0
            elif (nQueue <= 1):
                dnQueue = -1.0
            else:
                dnQueue = nQueue - self.nQueuePrev
    
            # Bring the bar back to red, if it's green and we dropped a frame.
            if (self.iDroppedFrame>0) and (self.dnQueueF<0.0):
                self.dnQueueF = 0.1
            else:  
                a = 0.001
                self.dnQueueF = (1-a)*self.dnQueueF + a*dnQueue
            
            self.aQueue = float(nQueue)/float(len(self.bufferImages))
            self.nQueuePrev = nQueue
                    
                    
            # Pull the image from the queue.
            if (self.bufferImages[self.iImgWorking] is not None):
                rosimg = self.bufferImages[self.iImgWorking]
                
                # Mark this buffer entry as available for loading.
                self.bufferImages[self.iImgWorking] = None
    
                # Go to the next image.
                self.iImgWorking = (self.iImgWorking+1) % len(self.bufferImages)
                

            if False:#('kinefly2' in self.nodename):
                rospy.logwarn('nQueue %3d, dnQueue %3d, dnQueueF %0.3f' % (nQueue, dnQueue, self.dnQueueF))


        # Compute processing times.
        self.stampROS        = rospy.Time.now()
        self.stampROSDiff    = (self.stampROS - self.stampROSPrev)
        self.stampROSPrev    = self.stampROS
        self.dtROS           = max(0, self.stampROSDiff.to_sec())

        # If time wrapped, then just assume a value.
        if (self.dtROS == 0.0):
            self.dtROS = 1.0

        # Compute system freq.
        hzROS = 1/self.dtROS
        self.iCountROS += 1
        if (self.iCountROS > 100):                     
            a= 0.04 # Filter the framerate.
            self.hzROSF = (1-a)*self.hzROSF + a*hzROS 
        else:                                       
            if (self.iCountROS>20):             # Get past the transient response.       
                self.hzROSSum += hzROS                 
            else:
                self.hzROSSum = hzROS * self.iCountROS     
                
            self.hzROSF = self.hzROSSum / self.iCountROS
                        
            
        if (rosimg is not None):            
            # Compute processing times.
            self.stampCamera     = rosimg.header.stamp
            self.stampCameraDiff = (self.stampCamera - self.stampCameraPrev)
            self.stampCameraPrev = self.stampCamera
            self.dtCamera        = max(0, self.stampCameraDiff.to_sec())

            # If the camera is not giving good timestamps, then use our own clock.
            if (self.dtCamera == 0.0):
                self.dtCamera = self.dtROS
                
            # If time wrapped, then just assume a value.
            if (self.dtCamera == 0.0):
                self.dtCamera = 1.0
                    
            # Compute processing freq.
            hzCamera = 1/self.dtCamera
            self.iCountCamera += 1
            if (self.iCountCamera > 100):                     
                a= 0.01 # Filter the framerate.
                self.hzCameraF = (1-a)*self.hzCameraF + a*hzCamera 
            else:                                       
                if (self.iCountCamera>20):             # Get past the transient response.       
                    self.hzCameraSum += hzCamera                 
                else:
                    self.hzCameraSum = hzCamera * self.iCountCamera     
                    
                self.hzCameraF = self.hzCameraSum / self.iCountCamera
                        

            # Convert the image.
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
            if (self.buttons is None) or (self.imgScaled.shape != self.shapeToolbar):
                self.create_buttons(self.imgScaled.shape)
        
            if (not self.bMousing) and (self.bValidImage):
                # Update the fly internals.
                self.fly.update(rosimg.header, self.imgScaled)
    
                # Publish the outputs.
                self.start()
                self.fly.publish()
                self.stop()
                
            
            if (self.params['use_gui']):
        
                imgOutput = cv2.cvtColor(self.imgScaled, cv2.COLOR_GRAY2RGB)
                self.fly.draw(imgOutput)
            
                x_left   = int(10 * self.scale)
                y_bottom = int(imgOutput.shape[0] - 10 * self.scale)
                x_right  = int(imgOutput.shape[1] - 10 * self.scale)
                x = x_left
                y = y_bottom
                h = 10

                w = 55
                if (not self.bMousing):

                    # Output the framerate.
                    s = '%5.1fHz [queue:          ]' % (self.hzCameraF)
                    if (self.iDroppedFrame>0):
                        s += '    Dropped Frames: %d' % self.iDroppedFrame
                    cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, ui.bgra_dict['dark_red'] )
                    h_text = int(h * self.scale)
                    
                    # Draw the gauge bar for the image queue length.
                    nBar = int(50*self.scale) # pixels max.
                    xBar = int(x+110*self.scale)
                    yBar = y-int(h_text/2)+1
                    ptBar0 = (xBar,                       yBar)
                    ptBar1 = (xBar+int(self.aQueue*nBar), yBar)
                    if (self.dnQueueF>0):
                        bgra = ui.bgra_dict['dark_red']
                    else:
                        bgra = ui.bgra_dict['dark_green']
                        
                    cv2.line(imgOutput, ptBar0, ptBar1, bgra, max(1,int(2*self.scale)))
                
    
                    # Output the aux state.
                    if (self.params['gui']['aux']['track']):
                        y -= h_text+self.h_gap
                        s = 'AUX: (%0.3f)' % (self.fly.aux.state.intensity)
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.aux.bgra)
                        h_text = int(h * self.scale)
                    
                        y -= h_text+self.h_gap
                        s = 'WB Freq: '
                        if (self.fly.aux.state.freq != 0.0):
                            s += '%0.0fhz' % (self.fly.aux.state.freq)
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.aux.bgra)
                        h_text = int(h * self.scale)
                    
    
                    # Output the wings state.
                    if (self.params['gui']['left']['track']) and (self.params['gui']['right']['track']):
                        # L+R
                        y -= h_text+self.h_gap
                        s = 'L+R:'
                        if (len(self.fly.left.state.angles)>0) and (len(self.fly.right.state.angles)>0):
                            leftplusright = self.fly.left.state.angles[0] + self.fly.right.state.angles[0]
                            s += '% 7.4f' % leftplusright
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, ui.bgra_dict['blue'])
                        h_text = int(h * self.scale)
        
                            
                        # L-R
                        y -= h_text+self.h_gap
                        s = 'L-R:'
                        if (len(self.fly.left.state.angles)>0) and (len(self.fly.right.state.angles)>0):
                            leftminusright = self.fly.left.state.angles[0] - self.fly.right.state.angles[0]
                            s += '% 7.4f' % leftminusright
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, ui.bgra_dict['blue'])
                        h_text = int(h * self.scale)
        
                    if (self.params['gui']['right']['track']):
                        # Right
                        y -= h_text+self.h_gap
                        s = 'R:'
                        if (len(self.fly.right.state.angles)>0):
                            s += '% 7.4f' % self.fly.right.state.angles[0]
                            #for i in range(1,len(self.fly.right.state.angles)):
                            #    s += ', % 7.4f' % self.fly.right.state.angles[i]
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.right.bgra)
                        h_text = int(h * self.scale)
            
                    if (self.params['gui']['left']['track']):
                        # Left
                        y -= h_text+self.h_gap
                        s = 'L:'
                        if (len(self.fly.left.state.angles)>0):
                            s += '% 7.4f' % self.fly.left.state.angles[0]
                            #for i in range(1,len(self.fly.left.state.angles)):
                            #    s += ', % 7.4f' % self.fly.left.state.angles[i]
                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.left.bgra)
                        h_text = int(h * self.scale)
                        
                            
    
    
                    # Output the abdomen state.
                    if (self.params['gui']['abdomen']['track']):
                        y -= h_text+self.h_gap
                        s = 'ABDOMEN:'
                        if (len(self.fly.abdomen.state.angles)>0):
                            s += '% 7.4f' % (self.fly.abdomen.state.angles[0])

                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.abdomen.bgra)
                        h_text = int(h * self.scale)
                    
    
                    # Output the head state.
                    if (self.params['gui']['head']['track']):
                        y -= h_text+self.h_gap
                        s = 'HEAD:'
                        if (len(self.fly.head.state.angles)>0):
                            s+= '% 7.4f' % (self.fly.head.state.angles[0])

                        cv2.putText(imgOutput, s, (x, y), self.fontface, self.scaleText, self.fly.head.bgra)
                        h_text = int(h * self.scale)

                self.pubImage.publish(imgOutput)
                self.buttons[self.ibtnInvertColor].state = self.fly.bInvertColor # Set the button state to reflect the fly's bInvertColor flag. 
                self.draw_buttons(imgOutput)

                # Display the image.
                cv2.imshow(self.window_name, imgOutput)

#         else:
#             if (self.hzROSF != 0.0):
#                 rospy.sleep(1/self.hzROSF) # Pretend we spent time processing.
        
        cv2.waitKey(1)

    # End process_image()
            
                
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
            tag      = [None,None,None,None,None,None]
            partname = [None,None,None,None,None,None]
            (partname[0], tag[0]) = self.fly.left.hit_object(ptMouse)
            (partname[1], tag[1]) = self.fly.right.hit_object(ptMouse)
            (partname[2], tag[2]) = self.fly.head.hit_object(ptMouse)
            (partname[3], tag[3]) = self.fly.abdomen.hit_object(ptMouse)
            (partname[4], tag[4]) = self.fly.aux.hit_object(ptMouse)
            (partname[5], tag[5]) = self.fly.axis.hit_object(ptMouse)
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
        elif (partname=='axis'):
            bodypart = self.fly.axis
        else:
            bodypart = None
            
        return bodypart
    
                
    # update_params_from_mouse()
    # Recalculate self.params based on a currently selected handle and mouse location.
    #
    def update_params_from_mouse(self, tagSelected, partnameSelected, ptMouse):
        partnameSlave = {'axis':None, 'head':'abdomen', 'abdomen':'head', 'right':'left', 'left':'right', 'aux':None}[partnameSelected]
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
                self.fly.axis.bValidMask    = False
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
                    ptSlave = imageprocessing.get_reflection_across_axis(ptMouse, (self.fly.abdomen.ptHinge_i, self.fly.head.ptHinge_i))
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
            angle_lo_b -= bodypartSelected.angleBodypart_b
            angle_lo_b = (angle_lo_b+np.pi) % (2.0*np.pi) - np.pi
            angle_hi_b -= bodypartSelected.angleBodypart_b
            angle_hi_b = (angle_hi_b+np.pi) % (2.0*np.pi) - np.pi
            
            # Switch to the other handle.
            if (not (angle_lo_b < angle_hi_b)):
                self.tagSelected = tagOther
                
            # Set the order of the two angles
            paramsScaled['gui'][partnameSelected]['angle_lo'] = min(angle_lo_b, angle_hi_b)
            paramsScaled['gui'][partnameSelected]['angle_hi'] = max(angle_lo_b, angle_hi_b)
            
            # Make angles relative to fly origin. 
            paramsScaled['gui'][partnameSelected]['angle_lo'] += bodypartSelected.angleBodypart_b
            paramsScaled['gui'][partnameSelected]['angle_hi'] += bodypartSelected.angleBodypart_b
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
                    paramsScaled['gui'][partnameSlave]['radius_inner'] = paramsScaled['gui'][partnameSelected]['radius_inner']
                    bodypartSlave.bValidMask = False
                    
#                 if (paramsScaled['gui'][partnameSlave]['angle_hi'] < 0 < paramsScaled['gui'][partnameSlave]['angle_lo']):
#                     paramsScaled['gui'][partnameSlave]['angle_hi'] += 2*np.pi
 
              
        # Inner radius.
        elif (tagSelected=='radius_inner'): 
            paramsScaled['gui'][partnameSelected]['radius_inner'] = float(min(np.linalg.norm(bodypartSelected.ptHinge_i - ptMouse), 
                                                                              bodypartSelected.params['gui'][partnameSelected]['radius_outer']-2*self.scale))
            
            bodypartSelected.bValidMask = False
            
            if (partnameSelected in ['left','right']) and (paramsScaled['gui']['symmetric']):
                paramsScaled['gui'][partnameSlave]['radius_outer'] = paramsScaled['gui'][partnameSelected]['radius_outer']
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
                
        # Axis Points.
        elif (tagSelected in ['pt1','pt2']): 
            if (partnameSelected=='axis'):
                # Move the point.
                pt = ptMouse
                paramsScaled['gui'][partnameSelected][tagSelected]['x'] = float(pt[0])
                paramsScaled['gui'][partnameSelected][tagSelected]['y'] = float(pt[1])
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
        
    # End update_params_from_mouse() 


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
                    self.pubCommand.publish('save_background')

                elif (self.nameSelected == self.nameSelectedNow == 'exit'):
                    self.pubCommand.publish('exit')
                    
                    
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
                    
                # A button to allow the user to override the automatic invertcolor detector.  A better autodetect algorithm might eliminate the need for this.
                elif (self.nameSelected == self.nameSelectedNow == 'invertcolor'):
                    self.fly.bInvertColor      = self.buttons[iButtonSelected].state
                    self.fly.bInvertColorValid = True
                    self.fly.bInvertColorAuto  = False # If the user clicked this button, then switch to manual mode.

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
        
    # End onMouse()

            
    def start(self):
        self.t0 = rospy.Time.now().to_sec()
                
    def stop(self):
        self.t1 = rospy.Time.now().to_sec()
                
    def run(self):
        if (self.params['gui']['aux']['track']):
            self.fly.aux.wingbeat.warn()
        
        tMin = np.Inf
        tMax = -np.Inf
        i = 0
        
        bFailed = False
        while (not rospy.is_shutdown()):
            self.process_image()

#             # Report results from use of self.start() & self.stop()
#             i += 1
#             if (i>1000):
#                 tMin = np.min([tMin, self.t1-self.t0])
#                 tMax = np.max([tMax, self.t1-self.t0])
#                 rospy.logwarn('tMin=%0.6f, t=%0.6f, tMax=%0.6f' % (tMin, self.t1-self.t0, tMax))

                
#             # Profile the cause of dropped frames.  Profile up to the point we get a bunch of drops, and keep that result.
#             if ('kinefly2' in rospy.get_name().rstrip('/')):
#                 if not bFailed:
#                     cProfile.runctx('self.process_image()', globals(), locals(), filename='/home/ssafarik/profile.pstats')
#                 else:
#                     self.process_image()
#                 if (self.iDroppedFrame>20) and (self.iCountCamera>100):
#                     bFailed = True
#                     rospy.logwarn('self.iDroppedFrame: %d' % self.iDroppedFrame)
#             else:
#                 self.process_image()

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
#     if ('kinefly2' in rospy.get_name().rstrip('/')):
#         cProfile.run('main.run()', '/home/ssafarik/profile.pstats')
#     else:
#         main.run()
    # Note to do profiling:
    # $ sudo apt-get install graphviz
    # $ git clone https://code.google.com/p/jrfonseca.gprof2dot/ gprof2dot
    # $ mkdir ~/bin
    # $ ln -s "$PWD"/gprof2dot/gprof2dot.py ~/bin
    # $ cd /home/ssafarik
    # $ ~/bin/gprof2dot.py -f pstats profile.pstats | dot -Tsvg -o callgraph.svg
    
