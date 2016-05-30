#!/usr/bin/env python
from __future__ import division
import rospy
import rosparam
import copy
import numpy as np
from std_msgs.msg import String
from Kinefly.msg import MsgAnalogIn
import Phidgets
import Phidgets.Devices.Analog
from setdict import SetDict


###############################################################################
###############################################################################
# Msg2PhidgetsAnalog()
#
# Subscribe to a MsgAnalogIn message topic, and output the values to
# a PhidgetsAnalog device.
#
class Msg2PhidgetsAnalog:

    def __init__(self):
        self.bInitialized = False
        self.bAttached = False
        
        # initialize
        self.name = 'Msg2PhidgetsAnalog'
        rospy.init_node(self.name, anonymous=True)
        self.nodename = rospy.get_name()
        self.namespace = rospy.get_namespace()
        
        # Load the parameters.
        self.params = rospy.get_param('%s' % self.nodename.rstrip('/'), {})
        self.defaults = {'v0enable':True, 'v1enable':True, 'v2enable':True, 'v3enable':True,
                         'serial':0,         # The serial number of the Phidget.  0==any.
                         'topic':'ai',
                         'scale':1.0
                         }
        SetDict().set_dict_with_preserve(self.params, self.defaults)
        rospy.set_param('%s' % self.nodename.rstrip('/'), self.params)
        
        # Enable the voltage output channels.
        self.enable = [self.params['v0enable'], self.params['v1enable'], self.params['v2enable'], self.params['v3enable']]  

        # Connect to the Phidget.
        self.analog = Phidgets.Devices.Analog.Analog()
        if (self.params['serial']==0):
            self.analog.openPhidget()
        else:
            self.analog.openPhidget(self.params['serial'])
            
        self.analog.setOnAttachHandler(self.attach_callback)
        self.analog.setOnDetachHandler(self.detach_callback)

        # Subscriptions.        
        self.subAI = rospy.Subscriber(self.params['topic'], MsgAnalogIn, self.ai_callback)
        self.subCommand  = rospy.Subscriber('%s/command' % self.nodename.rstrip('/'), String, self.command_callback, queue_size=1000)
        #rospy.sleep(1) # Allow time to connect publishers & subscribers.

        self.bInitialized = True
        
        
    def attach_callback(self, phidget):
        for i in range(4):
            self.analog.setEnabled(i, self.enable[i])

        self.phidgetserial = self.analog.getSerialNum()
        self.phidgetname = self.analog.getDeviceName()
        rospy.sleep(1) # Wait so that other nodes can display their banner first.
        rospy.logwarn('%s - %s Attached: serial=%s' % (self.namespace, self.phidgetname, self.phidgetserial))
        self.bAttached = True
        

    def detach_callback(self, phidget):
        rospy.logwarn ('%s - %s Detached: serial=%s.' % (self.namespace, self.phidgetname, self.phidgetserial))
        self.bAttached = False


    def ai_callback(self, msg):
        if (self.bAttached):
            iMax = min(len(msg.voltages),4)
            for i in range(iMax):
                if (self.enable[i]):
                    try:
                        self.analog.setVoltage(i, self.params['scale']*msg.voltages[i])
                    except Phidgets.PhidgetException.PhidgetException:
                        pass
        else:
            try:
                self.analog.waitForAttach(10) # 10ms
            except Phidgets.PhidgetException.PhidgetException:
                pass
            
            if (self.analog.isAttached()):
                self.bAttached = True
        
        
    # command_callback()
    # Execute any commands sent over the command topic.
    #
    def command_callback(self, command):
        self.command = command.data
        
        if (self.command == 'exit'):
            rospy.signal_shutdown('User requested exit.')


        if (self.command == 'help'):
            rospy.logwarn('')
            rospy.logwarn('Subscribe to a MsgAnalogIn message topic (default: ai), and output the voltages to')
            rospy.logwarn('a PhidgetsAnalog device.')
            rospy.logwarn('')
            rospy.logwarn('The %s/command topic accepts the following string commands:' % self.nodename.rstrip('/'))
            rospy.logwarn('  help                 This message.')
            rospy.logwarn('  exit                 Exit the program.')
            rospy.logwarn('')
            rospy.logwarn('You can send the above commands at the shell prompt via:')
            rospy.logwarn('rostopic pub -1 %s/command std_msgs/String commandtext' % self.nodename.rstrip('/'))
            rospy.logwarn('')
            rospy.logwarn('Parameters are settable as launch-time parameters.')
            rospy.logwarn('')

    
        
    def run(self):
        rospy.spin()

        if (self.analog.isAttached()):
            for i in range(4):
                self.analog.setVoltage(i, 0.0)
        
        self.analog.closePhidget()


if __name__ == '__main__':

    main = Msg2PhidgetsAnalog()
    main.run()

