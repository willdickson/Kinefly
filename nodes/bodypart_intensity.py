#!/usr/bin/env python
import rospy
import copy
import cv2
import numpy as np
from imagewindow import ImageWindow
import ui


class Struct:
    pass

gbShowMasks = False

###############################################################################
###############################################################################
# Contains the base behavior for tracking a bodypart via pixel intensity.
#
class IntensityTrackedBodypart(object):
    def __init__(self, name=None, params={}, color='white', bEqualizeHist=False):
        self.name        = name
        self.bEqualizeHist = bEqualizeHist

        self.bValidMask = False

        self.bgra        = ui.bgra_dict[color]
        self.bgra_dim    = tuple(0.5*np.array(ui.bgra_dict[color]))
        self.bgra_state  = ui.bgra_dict[color]#ui.bgra_dict['red']
        self.pixelmax    = 255.0

        self.shape     = (np.inf, np.inf)
        self.ptCenter_i = np.array([0,0])
        self.roi       = None
        
        self.mask = Struct()
        self.mask.img = None
        self.mask.sum = 1.0
        
        self.dt = np.inf

        self.params = {}
        self.handles = {'center':ui.Handle(np.array([0,0]), self.bgra, name='center'),
                        'radius1':ui.Handle(np.array([0,0]), self.bgra, name='radius1'),
                        'radius2':ui.Handle(np.array([0,0]), self.bgra, name='radius2')
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
        x     = int(self.params['gui'][self.name]['center']['x'])
        y     = int(self.params['gui'][self.name]['center']['y'])
        r1    = int(self.params['gui'][self.name]['radius1'])
        r2    = int(self.params['gui'][self.name]['radius2'])
        angle = self.params['gui'][self.name]['angle']
        bgra  = ui.bgra_dict['white']
        
        # Create the mask.
        img = np.zeros(shape, dtype=np.uint8)
        cv_filled = -1
        cv2.ellipse(img, (x, y), (r1, r2), int(np.rad2deg(angle)), 0, 360, bgra, cv_filled)
        self.windowMask.set_image(img)
        
        # Find the ROI of the mask.
        xSum = np.sum(img, 0)
        ySum = np.sum(img, 1)
        xMask = np.where(xSum>0)[0]
        yMask = np.where(ySum>0)[0]
        
        if (len(xMask)>0) and (len(yMask)>0): 
            # Dilate with a border.
            xMin = np.where(xSum>0)[0][0]  
            xMax = np.where(xSum>0)[0][-1] + 1
            yMin = np.where(ySum>0)[0][0]  
            yMax = np.where(ySum>0)[0][-1] + 1
            
            # Clip border to image edges.
            self.mask.xMin = np.max([0,xMin])
            self.mask.yMin = np.max([0,yMin])
            self.mask.xMax = np.min([xMax, shape[1]-1])
            self.mask.yMax = np.min([yMax, shape[0]-1])
            
            self.roi = np.array([self.mask.xMin, self.mask.yMin, self.mask.xMax, self.mask.yMax])
            self.mask.img = img[self.mask.yMin:self.mask.yMax, self.mask.xMin:self.mask.xMax]
            self.mask.sum = np.sum(self.mask.img).astype(np.float32)
    
            self.bValidMask = True
        else:
            self.mask.xMin = None
            self.mask.yMin = None
            self.mask.xMax = None
            self.mask.yMax = None
            
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
            if (self.mask.img is not None):
                self.imgRoiFgMasked = cv2.bitwise_and(self.imgRoiFg, self.mask.img)
    
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
            if (not self.bValidMask):
                self.create_mask(image.shape)
                self.bValidMask = True
                
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



