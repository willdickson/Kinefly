#!/usr/bin/env python
from __future__ import division
import rospy
import rosparam
import copy
import threading
import numpy as np
from std_msgs.msg import String
from Kinefly.msg import MsgAnalogIn

# WBD
# ----------------------------------------
#import Phidgets
#import Phidgets.Devices.Analog
# ----------------------------------------
import Phidget22.Devices.VoltageOutput
import Phidget22.PhidgetException 
import Phidget22.Phidget
# ----------------------------------------

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
        self.lock = threading.Lock()
        
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

        # WBD
        # ----------------------------------------------------------------------
        # Connect to the Phidget.
        #self.analog = Phidgets.Devices.Analog.Analog()
        #if (self.params['serial']==0):
        #    self.analog.openPhidget()
        #else:
        #    self.analog.openPhidget(self.params['serial'])
        #    
        #self.analog.setOnAttachHandler(self.attach_callback)
        #self.analog.setOnDetachHandler(self.detach_callback)
        # -----------------------------------------------------------------------
        with self.lock:
        self.aout_chan_list = []
            for i in range(4):
                aout_chan = Phidget22.Devices.VoltageOutput.VoltageOutput()
                aout_chan.setOnAttaself.analogHandler(self.attach_callback)
                aout_chan.setOnDetaself.analogHandler(self.detach_callback))
                aout_chan.setChannel(i)
                aout_chan.openWaitForAttachment(5000)
                self.aout_chan_list.append(aout_chan)
        # -----------------------------------------------------------------------

        # Subscriptions.        
        self.subAI = rospy.Subscriber(self.params['topic'], MsgAnalogIn, self.ai_callback)
        self.subCommand  = rospy.Subscriber('%s/command' % self.nodename.rstrip('/'), String, self.command_callback, queue_size=1000)
        #rospy.sleep(1) # Allow time to connect publishers & subscribers.

        self.bInitialized = True
        
        
    def attach_callback(self, phidget):
        # WBD
        # ---------------------------------------------------------------------------
        #for i in range(4):
        #    self.analog.setEnabled(i, self.enable[i])

        #self.phidgetserial = self.analog.getSerialNum()
        #self.phidgetname = self.analog.getDeviceName()
        # ----------------------------------------------------------------------------
        with self.lock:
            for aout_chan in self.aout_chan_list:
                aout_chan.setEnabled(True)

            # Get serial number and name - use channel 0 as representaive, assume all chans are on the same device
            aout_chan_rep = self.aout_chan_list[0]
            self.phidgetserial = aout_chan_rep.getDeviceSerialNumber()
            self.phidgetname = aout_chan_rep.getDeviceName()
            self.bAttached = True
        # ----------------------------------------------------------------------------
        rospy.sleep(1) # Wait so that other nodes can display their banner first.
        rospy.logwarn('%s - %s Attached: serial=%s' % (self.namespace, self.phidgetname, self.phidgetserial))
        

    def detach_callback(self, phidget):
        rospy.logwarn ('%s - %s Detached: serial=%s.' % (self.namespace, self.phidgetname, self.phidgetserial))
        self.bAttached = False


    def ai_callback(self, msg):
        with self.lock:
            if (self.bAttached):
                iMax = min(len(msg.voltages),4)
                for i in range(iMax):
                    if (self.enable[i]):
                        # WBD
                        # ----------------------------------------------------------------------------
                        #try:
                        #    self.analog.setVoltage(i, self.params['scale']*msg.voltages[i])
                        #except Phidgets.PhidgetException.PhidgetException:
                        #    pass
                        # ----------------------------------------------------------------------------
                        try:
                            aout_chan = self.aout_chan_list[i]
                            aout_chan.setVoltage(self.params['scale']*msg.voltages[i])
                        except Phidget22.PhidgetException.PhidgetException:
                            pass
                        # ----------------------------------------------------------------------------
            else:
                # WBD
                # ------------------------------------------------------------------------------------
                #try:
                #    self.analog.waitForAttach(10) # 10ms
                #except Phidgets.PhidgetException.PhidgetException:
                #    pass
                #if (self.analog.isAttached()):
                #    self.bAttached = True
                # ------------------------------------------------------------------------------------
                is_attached = True 
                for chan in self.aout_chan_list:
                    try:
                        aout_chan.openWaitForAttachment(10)
                    except Phidget22.PhidgetException.PhidgetException:
                        pass
                    is_attached = is_attached and chan.getAttached()
                self.bAttached = is_attached
                # ------------------------------------------------------------------------------------
            
        
        
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

        # WBD
        # ----------------------------------------------------------------------
        #if (self.analog.isAttached()):
        #    for i in range(4):
        #        self.analog.setVoltage(i, 0.0)
        #
        #self.analog.closePhidget()
        # -----------------------------------------------------------------------
        with self.lock:
            for aout_chan in self.aout_chan_list:
                if aout_chan.getAttached():
                    aout_chan.setVoltage(0.0)
                aout_chan.close()
        # -----------------------------------------------------------------------


if __name__ == '__main__':

    main = Msg2PhidgetsAnalog()
    main.run()

