#!/usr/bin/env python
from __future__ import division
import roslib; roslib.load_manifest('StrokelitudeROS')
import rospy
import rosparam

import copy
import numpy as np

from std_msgs.msg import String

from StrokelitudeROS.msg import MsgFlystate
import Phidgets
import Phidgets.Devices.Analog


###############################################################################
###############################################################################
class Strokelitude2PhidgetsAnalog:

    def __init__(self):
        self.bInitialized = False
        self.iCount = 0
        
        # initialize
        rospy.init_node('strokelitude2phidgetsanalog', anonymous=True)
        
        # Load the parameters.
        self.params = rospy.get_param('strokelitude/phidgetsanalog', {})
        self.defaults = {'v0enable':True, 'v1enable':True, 'v2enable':True, 'v3enable':True, 
#                          'v00': 0.0, 'v0l1':1.0, 'v0l2':0.0, 'v0r1':0.0, 'v0r2':0.0, 'v0ha':0.0, 'v0hr':0.0, 'v0aa':0.0, 'v0ar':0.0, # L
#                          'v10': 0.0, 'v1l1':0.0, 'v1l2':0.0, 'v1r1':1.0, 'v1r2':0.0, 'v1ha':0.0, 'v1hr':0.0, 'v1aa':0.0, 'v1ar':0.0, # R
#                          'v20': 0.0, 'v2l1':1.0, 'v2l2':0.0, 'v2r1':1.0, 'v2r2':0.0, 'v2ha':0.0, 'v2hr':0.0, 'v2aa':0.0, 'v2ar':0.0, # L+R
#                          'v30': 0.0, 'v3l1':1.0, 'v3l2':0.0, 'v3r1':-1.0, 'v3r2':0.0, 'v3ha':0.0, 'v3hr':0.0, 'v3aa':0.0, 'v3ar':0.0, # L-R
#
                         'v00': 0.0, 'v0l1':-1.0, 'v0l2':0.0, 'v0r1':0.0, 'v0r2':0.0, 'v0ha':0.0, 'v0hr':0.0, 'v0aa':0.0, 'v0ar':0.0, # L
                         'v10': 0.0, 'v1l1':-1.0, 'v1l2':0.0, 'v1r1':0.0, 'v1r2':0.0, 'v1ha':0.0, 'v1hr':0.0, 'v1aa':0.0, 'v1ar':0.0, # R
                         'v20': 0.0, 'v2l1':1.0, 'v2l2':0.0, 'v2r1':0.0, 'v2r2':0.0, 'v2ha':0.0, 'v2hr':0.0, 'v2aa':0.0, 'v2ar':0.0, # L+R
                         'v30': 0.0, 'v3l1':1.0, 'v3l2':0.0, 'v3r1':0.0, 'v3r2':0.0, 'v3ha':0.0, 'v3hr':0.0, 'v3aa':0.0, 'v3ar':0.0, # L-R
                         'autorange':False
                         }
        self.set_dict_with_preserve(self.params, self.defaults)
        rospy.set_param('strokelitude/phidgetsanalog', self.params)
        

        self.fMin = np.array([ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf])
        self.fMax = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        
        self.update_coefficients()
        
        
        # Subscriptions.        
        self.subFlystate = rospy.Subscriber('strokelitude/flystate', MsgFlystate, self.flystate_callback)
        self.subCommand  = rospy.Subscriber('strokelitude2phidgetsanalog/command', String, self.command_callback)
        rospy.sleep(1) # Allow time to connect publishers & subscribers.

        # Connect to the Phidget.
        self.analog = Phidgets.Devices.Analog.Analog()
        self.analog.openPhidget()
        while (True):
            rospy.logwarn('Waiting for PhidgetsAnalog device...')
            try:
                self.analog.waitForAttach(1000)
            except Phidgets.PhidgetException.PhidgetException:
                pass
            
            if (self.analog.isAttached()):
                break
            
        rospy.logwarn('Attached to: %s, ID=%s' % (self.analog.getDeviceName(), self.analog.getDeviceID()))
        
        for i in range(4):
            self.analog.setEnabled(i, self.enable[i])
        
        self.bInitialized = True
        

    def update_coefficients(self):
        if (not self.params['autorange']):
            self.update_coefficients_from_params()
        else:
            self.update_coefficients_from_autorange()
            

        self.enable = [self.params['v0enable'], self.params['v1enable'], self.params['v2enable'], self.params['v3enable']]  

    
    # update_coefficients_from_params()
    #
    # Make a coefficients matrix out of the params dict values.
    # There are four voltage channels (v0,v1,v2,v3), and each channel has 
    # coefficients to make a user-specified voltage from wing, head, and abdomen angles.
    # 
    def update_coefficients_from_params(self):
        self.a = np.array([[self.params['v00'], self.params['v0l1'], self.params['v0l2'], self.params['v0r1'], self.params['v0r2'], self.params['v0ha'], self.params['v0hr'], self.params['v0aa'], self.params['v0ar']],
                           [self.params['v10'], self.params['v1l1'], self.params['v1l2'], self.params['v1r1'], self.params['v1r2'], self.params['v1ha'], self.params['v1hr'], self.params['v1aa'], self.params['v1ar']],
                           [self.params['v20'], self.params['v2l1'], self.params['v2l2'], self.params['v2r1'], self.params['v2r2'], self.params['v2ha'], self.params['v2hr'], self.params['v2aa'], self.params['v2ar']],
                           [self.params['v30'], self.params['v3l1'], self.params['v3l2'], self.params['v3r1'], self.params['v3r2'], self.params['v3ha'], self.params['v3hr'], self.params['v3aa'], self.params['v3ar']]
                          ],
                          dtype=np.float32
                          )
            
            
    # update_coefficients_from_autorange()
    #
    # Make a coefficients matrix to automatically set voltage ranges.
    # Set the voltage outputs 0,1,2,3 to L, R, L-R, L+R, respectively, such that they each span the voltage range [-10,+10].
    # 
    def update_coefficients_from_autorange(self):
        lmax = self.fMax[1]
        lmin = self.fMin[1]
        lmean = (lmax-lmin)/2

        rmax = self.fMax[3]
        rmin = self.fMin[3]
        rmean = (rmax-rmin)/2

        # From Matlab:
        # q0=solve(10==a0+a1*lmax, -10==a0+a1*lmin, a0,a1)
        # q1=solve(10==a0+a3*rmax, -10==a0+a3*rmin, a0,a3)
        # q2=solve(10==a0+a1*lmax+a3*rmin, -10==a0+a1*lmin+a3*rmax, 0==a0+a1*lmin+a3*rmin, a0,a1,a3)
        # q3=solve(10==a0+a1*lmax+a3*rmax, -10==a0+a1*lmin+a3*rmin, 0==a0+a1*lmax+a3*rmin, a0,a1,a3)

        a00 = -(10*(lmax+lmin))/(lmax-lmin)
        a01 = 20/(lmax-lmin)
        
        a10 = -(10*(rmax+rmin))/(rmax-rmin)
        a13 = 20/(rmax-rmin)
        
        a20 = (10*(lmax*rmin - lmin*rmax))/((lmax - lmin)*(rmax - rmin))
        a21 = 10/(lmax - lmin)
        a23 = -10/(rmax - rmin)
        
        a30 = -(10*(lmax*rmax - lmin*rmin))/((lmax - lmin)*(rmax - rmin))
        a31 = 10/(lmax - lmin)
        a33 = 10/(rmax - rmin)
                
        #                     1   L1   L2   R1 R2 HA HR AA AR
        self.a = np.array([[a00, a01,   0,   0, 0, 0, 0, 0, 0],
                           [a10,   0,   0, a13, 0, 0, 0, 0, 0],
                           [a20, a21,   0, a23, 0, 0, 0, 0, 0],
                           [a30, a31,   0, a33, 0, 0, 0, 0, 0]],
                          )
        
        
    def flystate_callback(self, flystate):
        self.iCount += 1
        if (self.analog.isAttached()):
            voltages = self.voltages_from_flystate(flystate)
            for i in range(4):
                if (self.enable[i]):
                    self.analog.setVoltage(i, voltages[i])
    
    
    # get_voltages()
    #
    def voltages_from_flystate(self, flystate):
        f = np.array([1.0,
                      flystate.left.angle1,
                      flystate.left.angle2,
                      flystate.right.angle1,
                      flystate.right.angle2,
                      flystate.head.angle,
                      flystate.head.radius,
                      flystate.abdomen.angle,
                      flystate.abdomen.radius
                      ], dtype=np.float32)
        
        if (self.iCount>10):
            self.fMin = np.min([self.fMin, f], 0)
            self.fMax = np.max([f, self.fMax], 0)

            # Decay toward the mean.
            fMean = (self.fMin + self.fMax) * 0.5
            d = (self.fMax - fMean) * 0.001
            self.fMax -= d
            self.fMin += d
            
        #rospy.logwarn((self.fMin,self.fMax))
        if (self.params['autorange']):
            self.update_coefficients_from_autorange()
            
        voltages = np.dot(self.a, f)
        
        # L1,L2,R1,R2,HA,AA are all in radians.
        # v00,v0l1,v0l2,v0r1,v0r2,v0ha,v0hr,v0aa,v0ar are coefficients to convert to voltage.
#         voltages[0] = self.v00 + self.v0l1*L1 + self.v0l2*L2 + \
#                                  self.v0r1*R1 + self.v0r2*R2 + \
#                                  self.v0ha*HA + self.v0hr*HR + \
#                                  self.v0aa*AA + self.v0ar*AR # Angle + Radius
                       
        return voltages.clip(-10.0, 10.0)

    
        
    # command_callback()
    # Execute any commands sent over the command topic.
    #
    def command_callback(self, command):
        self.command = command.data
        
        if (self.command == 'exit'):
            rospy.signal_shutdown('User requested exit.')


        if (self.command == 'help'):
            rospy.logwarn('The strokelitude2phidgetsanalog/command topic accepts the following string commands:')
            rospy.logwarn('  help                 This message.')
            rospy.logwarn('  exit                 Exit the program.')
            rospy.logwarn('')
            rospy.logwarn('You can send the above commands at the shell prompt via:')
            rospy.logwarn('rostopic pub -1 strokelitude2phidgetsanalog/command std_msgs/String commandtext')
            rospy.logwarn('')
            rospy.logwarn('Parameters are settable as launch-time parameters.')
            rospy.logwarn('')

    
        
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


    def run(self):
        rospy.spin()

        if (self.analog.isAttached()):
            for i in range(4):
                self.analog.setVoltage(i, 0.0)


if __name__ == '__main__':

    s2pa = Strokelitude2PhidgetsAnalog()
    s2pa.run()

