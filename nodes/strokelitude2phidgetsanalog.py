#!/usr/bin/env python
from __future__ import division
import roslib; roslib.load_manifest('StrokelitudeROS')
import rospy
import rosparam

import copy
import numpy as np

from std_msgs.msg import Float32, Header, String

from StrokelitudeROS.msg import MsgFlystate, MsgWing, MsgBodypart
import Phidgets.Devices.Analog


###############################################################################
###############################################################################
class Strokelitude2Voltage:

    def __init__(self):
        self.bInitialized = False

        # initialize
        rospy.init_node('strokelitude2voltage', anonymous=True)
        
        # Load the parameters.
        self.params = rospy.get_param('strokelitude/voltage', {})
        self.defaults = {'x0': 0.0,
                         'xl1':1.0,
                         'xl2':0.0,
                         'xr1':-1.0,
                         'xr2':0.0,
                         'xha':0.0,
                         'xhr':0.0,
                         'xaa':0.0,
                         'xar':0.0,
                         'y0': 0.0,
                         'yl1':0.0,
                         'yl2':0.0,
                         'yr1':0.0,
                         'yr2':0.0,
                         'yha':0.0,
                         'yhr':0.0,
                         'yaa':0.0,
                         'yar':0.0,
                        }
        self.set_dict_with_preserve(self.params, self.defaults)
        self.update_coefficients_from_params()
        #rospy.set_param('strokelitude',self.params)
        
        # Subscriptions.        
        self.subFlystate = rospy.Subscriber('strokelitude/flystate', MsgFlystate, self.flystate_callback)
        self.subCommand  = rospy.Subscriber('strokelitude2voltage/command', String, self.command_callback)
        rospy.sleep(1) # Time to connect publishers & subscribers.

        self.analog = Phidgets.Devices.Analog.Analog()
        self.analog.openPhidget()
        while (True):
            rospy.logwarn('Waiting for PhidgetAnalog device...')
            self.analog.waitForAttach(1000)
            if (self.analog.isAttached()):
                break
        rospy.logwarn('Attached to PhidgetAnalog device...')
        for i in range(4):
            self.analog.setEnabled(i, True)
        
        self.bInitialized = True
        

    def update_coefficients_from_params(self):
        self.x0  = self.params['x0']
        self.xl1 = self.params['xl1']
        self.xl2 = self.params['xl2']
        self.xr1 = self.params['xr1']
        self.xr2 = self.params['xr2']
        self.xha = self.params['xha']
        self.xhr = self.params['xhr']
        self.xaa = self.params['xaa']
        self.xar = self.params['xar']
        
        
    def flystate_callback(self, flystate):
        voltages = self.get_voltages(flystate)
        for i in range(4):
            self.analog.setVoltage(i, voltages[i])
    
    
    # get_voltages()
    #
    def get_voltages(self, flystate):
        voltages = [0.0, 0.0, 0.0, 0.0]
        
        L1 = flystate.left.angle1
        L2 = flystate.left.angle2
        R1 = flystate.right.angle1
        R2 = flystate.right.angle2
        HA = flystate.head.angle
        HR = flystate.head.radius
        AA = flystate.abdomen.angle
        AR = flystate.abdomen.radius
        
        # L1,L2,R1,R2,HA,HR,AA,AR are all in radians.
        # x0,xl1,xl2,xr1,xr2,xha,xhr,xaa,xar are coefficients to convert to frames.
        voltages[0] = self.x0 + self.xl1*L1 + self.xl2*L2 + \
                                self.xr1*R1 + self.xr2*R2 + \
                                self.xha*HA + self.xhr*HR + \
                                self.xaa*AA + self.xar*AR # Angle + Radius
                       
        return voltages

    
        
    # command_callback()
    # Execute any commands sent over the command topic.
    #
    def command_callback(self, command):
        self.command = command.data
        
        if (self.command == 'exit'):
            rospy.signal_shutdown('User requested exit.')


        if (self.command == 'help'):
            rospy.logwarn('The strokelitude2voltage/command topic accepts the following string commands:')
            rospy.logwarn('  help                 This message.')
            rospy.logwarn('  exit                 Exit the program.')
            rospy.logwarn('')
            rospy.logwarn('You can send the above commands at the shell prompt via:')
            rospy.logwarn('rostopic pub -1 strokelitude2voltage/command std_msgs/String commandtext')
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


if __name__ == '__main__':

    s2l = Strokelitude2Voltage()
    s2l.run()

