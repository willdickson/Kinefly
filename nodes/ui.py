#!/usr/bin/env python
import rospy
import cv2
import numpy as np

# Define button sides for drawing.
SIDE_TOP = 1
SIDE_BOTTOM = 2
SIDE_LEFT = 4
SIDE_RIGHT = 8
SIDE_ALL = (SIDE_TOP | SIDE_BOTTOM | SIDE_LEFT | SIDE_RIGHT)

# Colors.
bgra_dict = {'black'         : (  0.0,   0.0,   0.0, 0.0),
             'white'         : (255.0, 255.0, 255.0, 0.0),
             'dark_gray'     : ( 64.0,  64.0,  64.0, 0.0),
             'gray'          : (128.0, 128.0, 128.0, 0.0),
             'light_gray'    : (192.0, 192.0, 192.0, 0.0),
             'red'           : (  0.0,   0.0, 255.0, 0.0),
             'green'         : (  0.0, 255.0,   0.0, 0.0), 
             'blue'          : (255.0,   0.0,   0.0, 0.0),
             'cyan'          : (255.0, 255.0,   0.0, 0.0),
             'magenta'       : (255.0,   0.0, 255.0, 0.0),
             'yellow'        : (  0.0, 255.0, 255.0, 0.0),
             'dark_red'      : (  0.0,   0.0, 128.0, 0.0),
             'dark_green'    : (  0.0, 128.0,   0.0, 0.0), 
             'dark_blue'     : (128.0,   0.0,   0.0, 0.0),
             'dark_cyan'     : (128.0, 128.0,   0.0, 0.0),
             'dark_magenta'  : (128.0,   0.0, 128.0, 0.0),
             'dark_yellow'   : (  0.0, 128.0, 128.0, 0.0),
             'light_red'     : (175.0, 175.0, 255.0, 0.0),
             'light_green'   : (175.0, 255.0, 175.0, 0.0), 
             'light_blue'    : (255.0, 175.0, 175.0, 0.0),
             'light_cyan'    : (255.0, 255.0, 175.0, 0.0),
             'light_magenta' : (255.0, 175.0, 255.0, 0.0),
             'light_yellow'  : (175.0, 255.0, 255.0, 0.0),
             }



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
        cv_filled = -1
        cv2.rectangle(image, self.ptLT3, self.ptRB3, self.colorFill, cv_filled)
        
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
        cv_filled = -1
        cv2.circle(image, tuple(self.pt.astype(int)),  self.radiusDraw, self.color, cv_filled)
        
        #ptText = self.pt+np.array([5,5])
        #cv2.putText(image, self.name, tuple(ptText.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4*self.scale, self.color)
        
# end class Handle
                


